#include "kenshin/renderer/HzbPass.h"

#include <cstring>
#include <glm/glm.hpp>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/manager/ShaderManager.h"

KENSHIN_NS_BEGIN

namespace renderer {

bool buildHzbOneshot(backend::IRenderBackend& backend,
                     manager::ShaderManager& shaders,
                     const std::vector<u64>& vis, u32 width, u32 height,
                     std::vector<u32>& out) {
  out.clear();
  if (vis.size() != usize(width) * height) return false;
  const HzbLayout layout = hzbLayout(width, height);
  out.assign(layout.totalTexels, 0xFFFFFFFFu);

  std::vector<u32> spirv;
  std::string log;
  if (!shaders.compileFile("build_hzb.slang", spirv, log, "main")) {
    KENSHIN_ERROR("[hzb] shader compile failed: {}", log);
    return false;
  }

  // Single atomic-climb dispatch: the pyramid is 0xFF-initialized (the
  // kernel's contract for the ancestor min-reduction).
  const glm::vec4 p(static_cast<f32>(width), static_cast<f32>(height), 0.0f,
                    0.0f);
  backend::IRenderBackend::Job job;
  job.spirv = spirv.data();
  job.spirvBytes = spirv.size() * sizeof(u32);
  job.groupCountX = (width * height + 63) / 64;
  job.buffers[0] = {&p, sizeof(p), false};
  job.buffers[1] = {vis.data(), vis.size() * sizeof(u64), false};
  job.buffers[2] = {out.data(), out.size() * sizeof(u32), true};
  std::vector<u8> outputs[backend::IRenderBackend::Job::kMaxBuffers];
  if (!backend.runComputeOneshot(job, outputs)) {
    KENSHIN_ERROR("[hzb] dispatch failed");
    return false;
  }
  std::memcpy(out.data(), outputs[2].data(), outputs[2].size());
  return true;
}

}  // namespace renderer

KENSHIN_NS_END
