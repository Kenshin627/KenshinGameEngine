#include "kenshin/renderer/Renderer.h"

#include <cstring>

#include "kenshin/Engine.h"
#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/core/Window.h"
#include "kenshin/manager/ShaderManager.h"
#include "kenshin/renderer/BaselineForwardPipeline.h"

KENSHIN_NS_BEGIN

namespace renderer {

bool Renderer::onInit(Engine& engine) {
  m_engine = &engine;
  core::Window& window = engine.get<core::Window>();

  // Backend creation stays engine-internal; it is registered so other
  // engine subsystems (DebugOverlay) can resolve it — apps never do.
  m_backend = &engine.registerInstance<backend::IRenderBackend>(
      backend::createVulkanBackend());
  backend::BackendInitParams params;
  params.nativeWindowHandle = window.nativeHandle();
  params.enableValidation = m_config.validation;
  if (!m_backend->initialize(params)) {
    KENSHIN_ERROR("[renderer] backend init failed");
    return false;
  }
  if (!m_backend->caps().enabledShaderBufferInt64Atomics) {
    // HARD GATE: the 64-bit atomic VisBuffer depends on this feature.
    KENSHIN_ERROR(
        "[renderer] GATE FAILED: shaderBufferInt64Atomics not "
        "supported — stop and decide the 32-bit packed fallback "
        "(see staged-plan risk table)");
    return false;
  }
  KENSHIN_INFO("[vulkan] gate shaderBufferInt64Atomics: OK");

  m_pipeline = std::make_unique<BaselineForwardPipeline>();
  return m_pipeline->init(m_backend, engine.get<manager::ShaderManager>());
}

void Renderer::onShutdown() {
  if (m_pipeline && m_backend) m_pipeline->shutdown(m_backend);
  m_pipeline.reset();
  m_backend = nullptr;
}

bool Renderer::beginFrame(u32 width, u32 height, const f32 clearColor[4]) {
  m_width = width;
  m_height = height;
  backend::FrameBeginInfo info;
  info.width = width;
  info.height = height;
  std::memcpy(info.clearColor, clearColor, sizeof(info.clearColor));
  return m_backend->beginFrame(info);
}

void Renderer::submit(const FrameView& view, std::span<const DrawItem> items) {
  m_frameView = view;
  m_stream.assign(items.begin(), items.end());
}

void Renderer::endFrame() {
  if (m_pipeline) {
    const FrameContext ctx{m_backend, &m_frameView, m_stream, m_width,
                           m_height};
    m_pipeline->execute(ctx);
  }
  m_backend->endFrame();
}

void* Renderer::viewportTexture() { return m_backend->viewportImGuiTexture(); }

void Renderer::requestFrameCapture() { m_backend->requestFullFrameCapture(); }

bool Renderer::readbackViewport(std::vector<u8>& rgbaOut, u32& width,
                                u32& height) {
  return m_backend->readbackFrame(rgbaOut, width, height);
}

bool Renderer::readbackFrame(std::vector<u8>& rgbaOut, u32& width,
                             u32& height) {
  return m_backend->readbackFullFrame(rgbaOut, width, height);
}

}  // namespace renderer

KENSHIN_NS_END
