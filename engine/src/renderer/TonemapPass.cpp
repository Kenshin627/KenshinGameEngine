#include "kenshin/renderer/TonemapPass.h"

#include <vector>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/manager/ShaderManager.h"

KENSHIN_NS_BEGIN

namespace renderer {

bool TonemapPass::init(backend::IRenderBackend& backend,
                       manager::ShaderManager& shaders) {
  std::vector<u32> vs, fs;
  std::string log;
  if (!shaders.compileFile("tonemap.slang", vs, log, "vsMain") ||
      !shaders.compileFile("tonemap.slang", fs, log, "fsMain")) {
    KENSHIN_ERROR("[tonemap] shader compile failed: {}", log);
    return false;
  }
  backend::IRenderBackend::FullscreenPipelineInfo info;
  info.vsSpirv = vs.data();
  info.vsBytes = vs.size() * sizeof(u32);
  info.fsSpirv = fs.data();
  info.fsBytes = fs.size() * sizeof(u32);
  m_pipeline = backend.createTonemapPipeline(info);
  if (m_pipeline == backend::IRenderBackend::kInvalidPipelineHandle) {
    KENSHIN_ERROR("[tonemap] pipeline creation failed");
    return false;
  }
  return true;
}

void TonemapPass::draw(backend::IRenderBackend& backend, Operator op,
                       float exposure, const glm::vec4& clearColor, u32 width,
                       u32 height) {
  if (m_pipeline == ~0u || width == 0 || height == 0) return;
  // TonemapPush: tonemap.slang's TonemapPush, packed identically.
  struct TonemapPushC {
    glm::vec4 clearColorExp;
    glm::vec4 modeViewport;
  } push;
  push.clearColorExp =
      glm::vec4(clearColor.r, clearColor.g, clearColor.b, exposure);
  push.modeViewport = glm::vec4(static_cast<f32>(static_cast<int>(op)),
                                1.0f / static_cast<f32>(width),
                                1.0f / static_cast<f32>(height), 0.0f);
  backend.drawTonemap(m_pipeline, &push, sizeof(push), width, height);
}

void TonemapPass::onShutdown(backend::IRenderBackend&) {
  m_pipeline = ~0u;  // the pipeline dies with PipelineStore; nothing owned
}

}  // namespace renderer

KENSHIN_NS_END
