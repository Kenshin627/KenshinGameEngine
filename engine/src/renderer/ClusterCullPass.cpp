#include "kenshin/renderer/ClusterCullPass.h"

#include <cfloat>
#include <cstring>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/manager/ShaderManager.h"

KENSHIN_NS_BEGIN

namespace renderer {

namespace {

bool compileShader(manager::ShaderManager& shaders, const char* path,
                   std::vector<u32>& spirv) {
  std::string log;
  if (!shaders.compileFile(path, spirv, log, "main")) {
    KENSHIN_ERROR("[cluster-cull] {} compile failed: {}", path, log);
    return false;
  }
  return true;
}

bool dispatch(backend::IRenderBackend& backend,
              backend::IRenderBackend::Job& job, u32 groupCountX,
              std::vector<u8> outputs[]) {
  job.groupCountX = groupCountX;
  return backend.runComputeOneshot(job, outputs);
}

}  // namespace

bool expandWorkItemsOnGpu(backend::IRenderBackend& backend,
                          manager::ShaderManager& shaders,
                          std::span<const u32> visible,
                          std::span<const CullGpuInstance> instances,
                          std::span<const GpuMeshRangePair> meshRanges,
                          std::vector<glm::uvec2>& workItems) {
  workItems.clear();
  if (visible.empty()) return true;

  // Total = sum of visible instances' cluster counts.
  u32 total = 0;
  for (const u32 idx : visible)
    total += meshRanges[instances[idx].meshId].clusterEnd -
             meshRanges[instances[idx].meshId].clusterBegin;
  workItems.resize(total);

  std::vector<u32> spirv;
  if (!compileShader(shaders, "expand_workitems.slang", spirv)) return false;

  backend::IRenderBackend::Job job;
  job.spirv = spirv.data();
  job.spirvBytes = spirv.size() * sizeof(u32);
  job.buffers[0] = {workItems.data(), total * sizeof(glm::uvec2), true};
  const u32 counters[2] = {0, static_cast<u32>(visible.size())};
  job.buffers[1] = {counters, sizeof(counters), true};
  job.buffers[2] = {visible.data(), visible.size() * sizeof(u32), false};
  job.buffers[3] = {instances.data(),
                    instances.size() * sizeof(CullGpuInstance), false};
  job.buffers[4] = {meshRanges.data(),
                    meshRanges.size() * sizeof(GpuMeshRangePair), false};

  std::vector<u8> outputs[backend::IRenderBackend::Job::kMaxBuffers];
  const u32 groups = (static_cast<u32>(visible.size()) + 63u) / 64u;
  if (!dispatch(backend, job, groups, outputs)) {
    KENSHIN_ERROR("[cluster-cull] expand dispatch failed");
    return false;
  }
  // Work items were written in place (readBack buffers[0]); copy back.
  std::memcpy(workItems.data(), outputs[0].data(), total * sizeof(glm::uvec2));
  return true;
}

bool cullClustersOnGpu(backend::IRenderBackend& backend,
                       manager::ShaderManager& shaders, const glm::vec3& camPos,
                       f32 pixelThreshold, f32 worldPerPixel,
                       const std::vector<u8>& clusterData,
                       const std::vector<u8>& groupData,
                       std::span<const glm::uvec2> workItems,
                       std::span<const CullGpuInstance> instances,
                       std::vector<glm::uvec2>& visiblePairs, u32 stats[4],
                       const OcclusionInputs& occlusion) {
  visiblePairs.clear();
  stats[0] = stats[1] = stats[2] = 0;
  if (workItems.empty()) return true;

  visiblePairs.resize(workItems.size());

  std::vector<u32> spirv;
  if (!compileShader(shaders, "cull_clusters.slang", spirv)) return false;

  // M3.2: params grew to float4[8] - [2..5] prev viewProj, [6] the
  // occlusion flags, [7] prev camPos (zeroed when disabled).
  const glm::vec4 params0(camPos, pixelThreshold);
  const glm::vec4 params1(worldPerPixel, static_cast<f32>(workItems.size()),
                          0.0f, 0.0f);
  glm::vec4 params[8] = {params0, params1};
  if (occlusion.enabled) {
    for (int c = 0; c < 4; ++c) params[2 + c] = occlusion.prevViewProj[c];
    params[6] =
        glm::vec4(1.0f, static_cast<f32>(occlusion.screenW),
                  static_cast<f32>(occlusion.screenH), occlusion.tanHalfFov);
    params[7] = glm::vec4(occlusion.prevCamPos, 0.0f);
  }
  // [3] = work item count (the shader's input guard), [4] = occlusion
  // culled counter (out), rest zeroed outs.
  const u32 countersInit[5] = {0, 0, 0, static_cast<u32>(workItems.size()), 0};

  backend::IRenderBackend::Job job;
  job.spirv = spirv.data();
  job.spirvBytes = spirv.size() * sizeof(u32);
  job.buffers[0] = {params, sizeof(params), false};
  job.buffers[1] = {clusterData.data(), clusterData.size(), false};
  job.buffers[2] = {groupData.data(), groupData.size(), false};
  job.buffers[3] = {workItems.data(), workItems.size() * sizeof(glm::uvec2),
                    false};
  job.buffers[4] = {instances.data(),
                    instances.size() * sizeof(CullGpuInstance), false};
  job.buffers[5] = {visiblePairs.data(),
                    visiblePairs.size() * sizeof(glm::uvec2), true};
  job.buffers[6] = {countersInit, sizeof(countersInit), true};
  // binding 7 is ALWAYS bound (the shader declares it; a missing
  // binding would fall back to a reflection-mismatched layout) - a
  // zeroed dummy when disabled. All-zero texels read min=0, and with
  // the flag off the branch never executes anyway.
  static const u32 kDummyHzb[4] = {0, 0, 0, 0};
  job.buffers[7] =
      occlusion.enabled
          ? backend::IRenderBackend::Job::Buffer{occlusion.hzbTexels,
                                                 occlusion.hzbCount *
                                                     sizeof(u32),
                                                 false}
          : backend::IRenderBackend::Job::Buffer{kDummyHzb, sizeof(kDummyHzb),
                                                 false};
  // binding 8 (cluster AABBs) same discipline as 7.
  static const glm::vec4 kDummyAabb[2] = {};
  job.buffers[8] =
      occlusion.enabled
          ? backend::IRenderBackend::Job::Buffer{occlusion.clusterAabbs,
                                                 clusterData.size() / 48 * 2 *
                                                     sizeof(glm::vec4),
                                                 false}
          : backend::IRenderBackend::Job::Buffer{kDummyAabb, sizeof(kDummyAabb),
                                                 false};
  if (occlusion.enabled &&
      (occlusion.hzbTexels == nullptr || occlusion.hzbCount == 0 ||
       occlusion.clusterAabbs == nullptr))
    return false;

  std::vector<u8> outputs[backend::IRenderBackend::Job::kMaxBuffers];
  const u32 groups = (static_cast<u32>(workItems.size()) + 63u) / 64u;
  if (!dispatch(backend, job, groups, outputs)) {
    KENSHIN_ERROR("[cluster-cull] cull dispatch failed");
    return false;
  }

  const u32* counters = reinterpret_cast<const u32*>(outputs[6].data());
  stats[0] = counters[0];
  stats[1] = counters[1];
  stats[2] = counters[2];
  stats[3] = counters[4];
  visiblePairs.resize(stats[0]);
  // Compact: pairs were written to scattered slots; copy the valid ones.
  std::memcpy(visiblePairs.data(), outputs[5].data(),
              stats[0] * sizeof(glm::uvec2));
  return true;
}

std::vector<glm::vec4> computeClusterAabbs(
    std::span<const glm::vec4> positions4, const std::vector<u8>& clusterData) {
  std::vector<glm::vec4> aabbs(clusterData.size() / 48 * 2);
  for (usize c = 0; c < clusterData.size() / 48; ++c) {
    const u8* e = clusterData.data() + c * 48;
    u32 vertexOffset, vertexCount;
    std::memcpy(&vertexOffset, e + 24, 4);
    std::memcpy(&vertexCount, e + 28, 4);
    glm::vec3 mn(FLT_MAX), mx(-FLT_MAX);
    for (u32 v = 0; v < vertexCount; ++v) {
      const glm::vec3 p(positions4[vertexOffset + v]);
      mn = glm::min(mn, p);
      mx = glm::max(mx, p);
    }
    aabbs[c * 2 + 0] = glm::vec4(mn, 0.0f);
    aabbs[c * 2 + 1] = glm::vec4(mx, 0.0f);
  }
  return aabbs;
}

}  // namespace renderer

KENSHIN_NS_END
