#include "kenshin/renderer/SoftRasterPass.h"

#include <cstring>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/manager/ShaderManager.h"

KENSHIN_NS_BEGIN

namespace renderer {

std::vector<u8> packRasterParams(const SoftRasterFrame& frame, u32 pairCount) {
  // float4[12]; glm columns copied raw (the shader combines them
  // explicitly - glm column-major, Slang mul reads rows).
  std::vector<u8> data(12 * sizeof(glm::vec4));
  std::memcpy(data.data() + 0 * sizeof(glm::vec4), &frame.view,
              sizeof(glm::mat4));
  std::memcpy(data.data() + 4 * sizeof(glm::vec4), &frame.proj,
              sizeof(glm::mat4));
  const glm::vec4 tail(static_cast<f32>(frame.width),
                       static_cast<f32>(frame.height),
                       static_cast<f32>(pairCount), frame.nearZ);
  std::memcpy(data.data() + 8 * sizeof(glm::vec4), &tail, sizeof(glm::vec4));
  // [9].x yFlip; [9].w the pair-list capacity for the in-guard (the
  // one-shot uploads exactly `pairCount` - nothing may drop).
  const glm::vec4 flags(frame.yFlip ? 1.0f : 0.0f, 0.0f, 0.0f,
                        static_cast<f32>(pairCount));
  std::memcpy(data.data() + 9 * sizeof(glm::vec4), &flags, sizeof(glm::vec4));
  return data;
}

bool rasterizeOnGpu(backend::IRenderBackend& backend,
                    manager::ShaderManager& shaders,
                    const SoftRasterFrame& frame,
                    std::span<const glm::vec4> positions,
                    std::span<const u32> indices,
                    const std::vector<u8>& clusterData,
                    std::span<const CullGpuInstance> instances,
                    std::span<const glm::uvec2> visiblePairs,
                    std::vector<u64>& visOut, u32 stats[4]) {
  visOut.assign(static_cast<usize>(frame.width) * frame.height, 0);
  stats[0] = stats[1] = stats[2] = stats[3] = 0;
  if (visiblePairs.empty()) return true;

  std::vector<u32> spirv;
  std::string log;
  if (!shaders.compileFile("rasterize_sw.slang", spirv, log, "main")) {
    KENSHIN_ERROR("[swraster] shader compile failed: {}", log);
    return false;
  }

  const std::vector<u8> params =
      packRasterParams(frame, static_cast<u32>(visiblePairs.size()));
  // [0] = pair count (the shader's input guard), rest zeroed outs.
  const u32 zeroStats[4] = {static_cast<u32>(visiblePairs.size()), 0, 0, 0};

  backend::IRenderBackend::Job job;
  job.spirv = spirv.data();
  job.spirvBytes = spirv.size() * sizeof(u32);
  job.buffers[0] = {params.data(), params.size(), false};
  job.buffers[1] = {positions.data(), positions.size() * sizeof(glm::vec4),
                    false};
  job.buffers[2] = {indices.data(), indices.size() * sizeof(u32), false};
  job.buffers[3] = {clusterData.data(), clusterData.size(), false};
  job.buffers[4] = {instances.data(),
                    instances.size() * sizeof(CullGpuInstance), false};
  job.buffers[5] = {visiblePairs.data(),
                    visiblePairs.size() * sizeof(glm::uvec2), false};
  // Zero-init via initData: an all-zero u64 VisBuffer (0 = nothing won)
  // and zeroed counters.
  job.buffers[6] = {visOut.data(), visOut.size() * sizeof(u64), true};
  job.buffers[7] = {zeroStats, sizeof(zeroStats), true};

  std::vector<u8> outputs[backend::IRenderBackend::Job::kMaxBuffers];
  job.groupCountX = static_cast<u32>(visiblePairs.size());
  if (!backend.runComputeOneshot(job, outputs)) {
    KENSHIN_ERROR("[swraster] dispatch failed");
    return false;
  }

  std::memcpy(visOut.data(), outputs[6].data(), visOut.size() * sizeof(u64));
  const u32* counters = reinterpret_cast<const u32*>(outputs[7].data());
  stats[0] = counters[0];
  stats[1] = counters[1];
  stats[2] = counters[2];
  stats[3] = counters[3];
  return true;
}

}  // namespace renderer

KENSHIN_NS_END
