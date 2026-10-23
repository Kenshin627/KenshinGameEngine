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
  // The exposure-state SSBO slot must always hold a valid buffer.
  const float zeros[8] = {};
  backend::IRenderBackend::BufferDesc desc;
  desc.size = sizeof(zeros);
  desc.usage = backend::IRenderBackend::BufferStorage;
  desc.initialData = zeros;
  desc.debugName = "tonemap-exposure-dummy";
  m_dummyState = backend.createBuffer(desc);
  if (m_dummyState == backend::IRenderBackend::kInvalidBufferHandle) {
    KENSHIN_ERROR("[tonemap] dummy exposure buffer creation failed");
    return false;
  }
  return true;
}

void TonemapPass::draw(backend::IRenderBackend& backend, Operator op,
                       float exposure, const glm::vec4& clearColor, u32 width,
                       u32 height, bool fromTaa, float sharpen,
                       float bloomStrength, float bloomInvLevels,
                       u32 exposureState) {
  if (m_pipeline == ~0u || width == 0 || height == 0) return;
  // TonemapPush: tonemap.slang's TonemapPush, packed identically.
  struct TonemapPushC {
    glm::vec4 clearColorExp;
    glm::vec4 modeViewport;
    glm::vec4 bloomParams;
  } push;
  push.clearColorExp =
      glm::vec4(clearColor.r, clearColor.g, clearColor.b, exposure);
  push.modeViewport = glm::vec4(static_cast<f32>(static_cast<int>(op)),
                                1.0f / static_cast<f32>(width),
                                1.0f / static_cast<f32>(height), sharpen);
  const bool autoExposure = exposureState != ~0u;
  push.bloomParams = glm::vec4(bloomStrength, bloomInvLevels,
                               autoExposure ? 1.0f : 0.0f, 0.0f);
  backend.drawTonemap(m_pipeline, &push, sizeof(push), width, height,
                      fromTaa, autoExposure ? exposureState : m_dummyState);
}

void TonemapPass::onShutdown(backend::IRenderBackend& backend) {
  if (m_dummyState != ~0u) backend.destroyBuffer(m_dummyState);
  m_dummyState = ~0u;
  m_pipeline = ~0u;  // the pipeline dies with PipelineStore
}

}  // namespace renderer

KENSHIN_NS_END
