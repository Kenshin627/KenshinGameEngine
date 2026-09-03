#include "kenshin/renderer/ClusterCullPass.h"

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
                       std::vector<glm::uvec2>& visiblePairs, u32 stats[3]) {
  visiblePairs.clear();
  stats[0] = stats[1] = stats[2] = 0;
  if (workItems.empty()) return true;

  visiblePairs.resize(workItems.size());

  std::vector<u32> spirv;
  if (!compileShader(shaders, "cull_clusters.slang", spirv)) return false;

  const glm::vec4 params0(camPos, pixelThreshold);
  const glm::vec4 params1(worldPerPixel, static_cast<f32>(workItems.size()),
                          0.0f, 0.0f);
  const glm::vec4 params[2] = {params0, params1};
  // [3] = work item count (the shader's input guard), rest zeroed outs.
  const u32 zeroStats[4] = {0, 0, 0, static_cast<u32>(workItems.size())};

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
  job.buffers[6] = {zeroStats, sizeof(zeroStats), true};

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
  visiblePairs.resize(stats[0]);
  // Compact: pairs were written to scattered slots; copy the valid ones.
  std::memcpy(visiblePairs.data(), outputs[5].data(),
              stats[0] * sizeof(glm::uvec2));
  return true;
}

}  // namespace renderer

KENSHIN_NS_END
