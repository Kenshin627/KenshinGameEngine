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
  out.resize(layout.totalTexels, 0);

  std::vector<u32> spirv;
  std::string log;
  if (!shaders.compileFile("build_hzb.slang", spirv, log, "main")) {
    KENSHIN_ERROR("[hzb] shader compile failed: {}", log);
    return false;
  }

  // Each level is its own one-shot (buffers are uploaded per job, so
  // the chain runs through the host: level 0 extracts the depth words
  // from the vis, every later level reduces the previous readback).
  std::vector<u8> srcWords;
  for (u32 level = 0; level < layout.levelCount; ++level) {
    const bool extract = level == 0;
    // Source bytes: the u64 vis (mode 0) or the previous level's
    // u32 texels (mode 1).
    if (extract) {
      srcWords.resize(vis.size() * sizeof(u64));
      std::memcpy(srcWords.data(), vis.data(), srcWords.size());
    }
    const u32 srcW = extract ? width : layout.levelW[level - 1];
    const u32 srcH = extract ? height : layout.levelH[level - 1];
    // The one-shot chain's src buffer holds ONLY the previous level
    // (standalone), so the shader's src offset is always 0 here - the
    // pyramid-relative offset applies to the in-frame variant, whose
    // The one-shot jobs use STANDALONE per-level buffers (src = the
    // previous level's readback, dst = this level's output), so both
    // shader offsets are 0 - the pyramid-relative offsets belong to the
    // in-frame variant, whose src/dst are the whole pyramid buffer.
    // (A nonzero dstOffset writes out of the standalone buffer - the
    // differential caught it as a level-1+ garbage pattern.)

    const glm::vec4 p0(static_cast<f32>(srcW), static_cast<f32>(srcH), 0.0f,
                       extract ? 0.0f : 1.0f);
    const glm::vec4 p1(0.0f, static_cast<f32>(layout.levelW[level]),
                       static_cast<f32>(layout.levelH[level]), 0.0f);
    const f32 params[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};

    backend::IRenderBackend::Job job;
    job.spirv = spirv.data();
    job.spirvBytes = spirv.size() * sizeof(u32);
    job.groupCountX = (layout.levelW[level] * layout.levelH[level] + 63) / 64;
    job.buffers[0] = {params, sizeof(params), false};
    job.buffers[1] = {srcWords.data(), srcWords.size(), false};
    job.buffers[2] = {
        nullptr,
        usize(layout.levelW[level]) * layout.levelH[level] * sizeof(u32), true};
    std::vector<u8> outputs[backend::IRenderBackend::Job::kMaxBuffers];
    if (!backend.runComputeOneshot(job, outputs)) {
      KENSHIN_ERROR("[hzb] level {} dispatch failed", level);
      return false;
    }
    std::memcpy(out.data() + layout.levelOffset[level], outputs[2].data(),
                outputs[2].size());
    // The next level's source is this level's texels.
    srcWords = std::move(outputs[2]);
  }
  return true;
}

}  // namespace renderer

KENSHIN_NS_END
