#include "kenshin/renderer/LightingPass.h"

#include <cstring>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/manager/ShaderManager.h"

KENSHIN_NS_BEGIN

namespace renderer {

bool LightingPass::init(backend::IRenderBackend& backend,
                        manager::ShaderManager& shaders) {
  std::vector<u32> vs, fs;
  std::string log;
  if (!shaders.compileFile("lighting.slang", vs, log, "vsMain") ||
      !shaders.compileFile("lighting.slang", fs, log, "fsMain")) {
    KENSHIN_ERROR("[lighting] shader compile failed: {}", log);
    return false;
  }
  backend::IRenderBackend::FullscreenPipelineInfo info;
  info.vsSpirv = vs.data();
  info.vsBytes = vs.size() * sizeof(u32);
  info.fsSpirv = fs.data();
  info.fsBytes = fs.size() * sizeof(u32);
  m_pipeline = backend.createLightingPipeline(info);
  if (m_pipeline == backend::IRenderBackend::kInvalidPipelineHandle) {
    KENSHIN_ERROR("[lighting] pipeline creation failed");
    return false;
  }
  backend::IRenderBackend::BufferDesc desc;
  desc.size = kMaxLights * sizeof(GpuLightC);
  desc.usage = backend::IRenderBackend::BufferStorage |
               backend::IRenderBackend::BufferHostVisible;
  desc.debugName = "light-table";
  m_lightBuffer = backend.createBuffer(desc);
  return m_lightBuffer != backend::IRenderBackend::kInvalidBufferHandle;
}

void LightingPass::draw(backend::IRenderBackend& backend,
                        const glm::mat4& invViewProj, const glm::vec3& camPos,
                        std::span<const GpuLightC> lights,
                        const glm::vec4& clearColor, float ambient,
                        float exposure, u32 width, u32 height) {
  if (m_pipeline == ~0u || width == 0 || height == 0) return;
  const u32 count =
      static_cast<u32>(std::min<usize>(lights.size(), kMaxLights));
  // LightingPush: lighting.slang's LightPush, packed identically.
  struct LightingPushC {
    glm::mat4 invViewProj;
    glm::vec4 camPosCount;
    glm::vec4 clearColor;
    glm::vec4 viewportAmb;
  } push;
  push.invViewProj = invViewProj;
  push.camPosCount = glm::vec4(camPos, static_cast<f32>(count));
  push.clearColor = clearColor;
  push.viewportAmb =
      glm::vec4(1.0f / static_cast<f32>(width), 1.0f / static_cast<f32>(height),
                ambient, exposure);
  if (count > 0) {
    auto* table = static_cast<GpuLightC*>(backend.mappedPtr(m_lightBuffer));
    std::memcpy(table, lights.data(), count * sizeof(GpuLightC));
  }
  backend.drawLighting(m_pipeline, &m_lightBuffer, 1, &push, sizeof(push),
                       width, height);
}

void LightingPass::onShutdown(backend::IRenderBackend& backend) {
  if (m_lightBuffer != ~0u) backend.destroyBuffer(m_lightBuffer);
  m_lightBuffer = ~0u;
  m_pipeline = ~0u;  // the pipeline dies with PipelineStore; nothing owned
}

}  // namespace renderer

KENSHIN_NS_END
