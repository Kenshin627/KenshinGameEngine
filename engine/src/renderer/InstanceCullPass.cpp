#include "kenshin/renderer/InstanceCullPass.h"

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/manager/ShaderManager.h"

#include <cstring>

KENSHIN_NS_BEGIN

namespace renderer {

void extractFrustumPlanes(const glm::mat4& viewProj, glm::vec4 planes[6]) {
  // Row combinations of the clip matrix (Gribb-Hartmann). glm stores
  // column-major: row i = (m[0][i], m[1][i], m[2][i], m[3][i]).
  const auto row = [&](int i) {
    return glm::vec4(viewProj[0][i], viewProj[1][i], viewProj[2][i],
                     viewProj[3][i]);
  };
  const glm::vec4 r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
  // Vulkan ZO NDC: -w <= x <= w, -w <= y <= w, 0 <= z <= w.
  const glm::vec4 raw[6] = {
      r3 + r0,  // left:   x + w >= 0
      r3 - r0,  // right:  w - x >= 0
      r3 + r1,  // bottom: y + w >= 0
      r3 - r1,  // top:    w - y >= 0
      r2,       // near:   z >= 0 (ZO: z_clip = z row alone)
      r3 - r2,  // far:    w - z >= 0
  };
  for (int p = 0; p < 6; ++p) {
    const f32 len = glm::length(glm::vec3(raw[p]));
    planes[p] = len > 0.0f ? raw[p] / len : raw[p];
  }
}

bool cullInstancesOnGpu(backend::IRenderBackend& backend,
                        manager::ShaderManager& shaders,
                        std::span<const CullGpuInstance> instances,
                        const glm::vec4 planes[6],
                        std::vector<u32>& visibleOut) {
  visibleOut.clear();
  if (instances.empty()) return true;

  std::vector<u32> spirv;
  std::string log;
  if (!shaders.compileFile("cull_instances.slang", spirv, log, "main")) {
    KENSHIN_ERROR("[cull] shader compile failed: {}", log);
    return false;
  }

  // buffers[i] -> binding i (Job convention).
  backend::IRenderBackend::Job job;
  job.spirv = spirv.data();
  job.spirvBytes = spirv.size() * sizeof(u32);
  job.groupCountX =
      (static_cast<u32>(instances.size()) + 63u) / 64u;

  const u32 counters[2] = {0, static_cast<u32>(instances.size())};
  job.buffers[0] = {planes, 6 * sizeof(glm::vec4), false};
  job.buffers[1] = {instances.data(),
                    instances.size() * sizeof(CullGpuInstance), false};
  job.buffers[2] = {nullptr,
                    instances.size() * sizeof(u32), true};   // visible list
  job.buffers[3] = {counters, sizeof(counters), true};       // [0]=count out, [1]=N

  std::vector<u8> outputs[backend::IRenderBackend::Job::kMaxBuffers];
  if (!backend.runComputeOneshot(job, outputs)) {
    KENSHIN_ERROR("[cull] dispatch failed");
    return false;
  }

  // Counter first; only that many list entries are valid.
  const u32 count = *reinterpret_cast<const u32*>(outputs[3].data());
  const u32* list = reinterpret_cast<const u32*>(outputs[2].data());
  visibleOut.assign(list, list + count);
  return true;
}

}  // namespace renderer

KENSHIN_NS_END
