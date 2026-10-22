#include "kenshin/renderer/BloomPass.h"

#include <string>
#include <vector>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/manager/ShaderManager.h"

KENSHIN_NS_BEGIN

namespace renderer {

bool BloomPass::init(backend::IRenderBackend& backend,
                     manager::ShaderManager& shaders) {
  std::vector<u32> vs, down, up;
  std::string log;
  if (!shaders.compileFile("bloom.slang", vs, log, "vsMain") ||
      !shaders.compileFile("bloom.slang", down, log, "fsDown") ||
      !shaders.compileFile("bloom.slang", up, log, "fsUp")) {
    KENSHIN_ERROR("[bloom] shader compile failed: {}", log);
    return false;
  }
  backend::IRenderBackend::FullscreenPipelineInfo info;
  info.vsSpirv = vs.data();
  info.vsBytes = vs.size() * sizeof(u32);
  for (u32 k = 0; k < kLevels; ++k) {
    info.fsSpirv = down.data();
    info.fsBytes = down.size() * sizeof(u32);
    m_down[k] = backend.createBloomPipeline(info, false);
    if (m_down[k] == backend::IRenderBackend::kInvalidPipelineHandle) {
      KENSHIN_ERROR("[bloom] downsample pipeline {} failed", k);
      return false;
    }
    if (k + 1 < kLevels) {
      info.fsSpirv = up.data();
      info.fsBytes = up.size() * sizeof(u32);
      m_up[k] = backend.createBloomPipeline(info, true);
      if (m_up[k] == backend::IRenderBackend::kInvalidPipelineHandle) {
        KENSHIN_ERROR("[bloom] upsample pipeline {} failed", k);
        return false;
      }
    }
  }
  m_ready = true;
  return true;
}

bool BloomPass::draw(backend::IRenderBackend& backend, bool fromTaa, u32 width,
                     u32 height) {
  if (!m_ready) return false;
  return backend.drawBloom(m_down, m_up, kLevels, fromTaa, width, height);
}

}  // namespace renderer

KENSHIN_NS_END
