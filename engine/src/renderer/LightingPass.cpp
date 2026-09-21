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
  // The 1-entry default material table (v2 path binds this - the
  // shader always reads materials[id], the binding must be valid).
  desc.size = 32;
  desc.initialData = nullptr;
  desc.debugName = "lighting-material-dummy";
  m_materialDummy = backend.createBuffer(desc);
  if (m_materialDummy != backend::IRenderBackend::kInvalidBufferHandle) {
    auto* mat = static_cast<f32*>(backend.mappedPtr(m_materialDummy));
    mat[0] = 0.75f;
    mat[1] = 0.75f;
    mat[2] = 0.78f;
    mat[3] = -1.0f;
    mat[4] = 0.5f;
    mat[5] = 0.0f;
    mat[6] = 0.0f;
    mat[7] = 0.0f;
  }
  return m_lightBuffer != backend::IRenderBackend::kInvalidBufferHandle &&
         m_materialDummy != backend::IRenderBackend::kInvalidBufferHandle;
}

void LightingPass::draw(backend::IRenderBackend& backend,
                        const glm::mat4& invViewProj, const glm::vec3& camPos,
                        std::span<const GpuLightC> lights, u32 materialTable,
                        u32 materialCount, const glm::vec4& clearColor,
                        float ambient, float exposure, u32 width, u32 height) {
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
  // clearColor.a carries the material count (lighting.slang's clamp
  // bound; the clear color's own alpha is meaningless). The dummy
  // fallback has exactly ONE entry - clamping to the scene's count
  // would read past it.
  const bool usingDummy =
      materialTable == backend::IRenderBackend::kInvalidBufferHandle;
  const u32 effectiveCount = usingDummy ? 1u : std::max(materialCount, 1u);
  push.clearColor = glm::vec4(clearColor.r, clearColor.g, clearColor.b,
                              static_cast<f32>(effectiveCount));
  push.viewportAmb =
      glm::vec4(1.0f / static_cast<f32>(width), 1.0f / static_cast<f32>(height),
                ambient, exposure);
  if (count > 0) {
    auto* table = static_cast<GpuLightC*>(backend.mappedPtr(m_lightBuffer));
    std::memcpy(table, lights.data(), count * sizeof(GpuLightC));
  }
  const u32 buffers[2] = {m_lightBuffer,
                          usingDummy ? m_materialDummy : materialTable};
  backend.drawLighting(m_pipeline, buffers, 2, &push, sizeof(push), width,
                       height);
}

void LightingPass::onShutdown(backend::IRenderBackend& backend) {
  if (m_lightBuffer != ~0u) backend.destroyBuffer(m_lightBuffer);
  if (m_materialDummy != ~0u) backend.destroyBuffer(m_materialDummy);
  m_lightBuffer = m_materialDummy = ~0u;
  m_pipeline = ~0u;  // the pipeline dies with PipelineStore; nothing owned
}

}  // namespace renderer

KENSHIN_NS_END
