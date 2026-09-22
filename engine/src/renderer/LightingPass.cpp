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
  // M8.3: the environment table (binding 4): [0] sky rgb + pad,
  // [1] ground rgb + pad - the hemisphere ambient's two poles.
  desc.size = 2 * sizeof(glm::vec4);
  desc.debugName = "env-table";
  m_envBuffer = backend.createBuffer(desc);
  // M9.1: the shadow parameter block (binding 6).
  desc.size = 5 * sizeof(glm::vec4);
  desc.debugName = "shadow-params";
  m_shadowParams = backend.createBuffer(desc);
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
         m_materialDummy != backend::IRenderBackend::kInvalidBufferHandle &&
         m_envBuffer != backend::IRenderBackend::kInvalidBufferHandle;
  m_shadowParams != backend::IRenderBackend::kInvalidBufferHandle;
}

void LightingPass::draw(backend::IRenderBackend& backend,
                        const glm::mat4& invViewProj, const glm::vec3& camPos,
                        std::span<const GpuLightC> lights, u32 materialTable,
                        u32 materialCount, const glm::vec4& clearColor,
                        float ambient, const glm::vec3& skyColor,
                        const glm::vec3& groundColor,
                        const glm::mat4& lightViewProj, float shadowBias,
                        float shadowTexelWorld, bool shadowsEnabled, u32 width,
                        u32 height) {
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
  push.viewportAmb = glm::vec4(1.0f / static_cast<f32>(width),
                               1.0f / static_cast<f32>(height), ambient, 0.0f);
  if (count > 0) {
    auto* table = static_cast<GpuLightC*>(backend.mappedPtr(m_lightBuffer));
    std::memcpy(table, lights.data(), count * sizeof(GpuLightC));
  }
  auto* env = static_cast<glm::vec4*>(backend.mappedPtr(m_envBuffer));
  env[0] = glm::vec4(skyColor, 0.0f);
  env[1] = glm::vec4(groundColor, 0.0f);
  auto* sp = static_cast<glm::vec4*>(backend.mappedPtr(m_shadowParams));
  sp[0] = lightViewProj[0];
  sp[1] = lightViewProj[1];
  sp[2] = lightViewProj[2];
  sp[3] = lightViewProj[3];
  sp[4] = glm::vec4(2048.0f, shadowBias, shadowTexelWorld,
                    shadowsEnabled ? 1.0f : 0.0f);
  const u32 buffers[4] = {m_lightBuffer,
                          usingDummy ? m_materialDummy : materialTable,
                          m_envBuffer, m_shadowParams};
  backend.drawLighting(m_pipeline, buffers, 4, &push, sizeof(push), width,
                       height);
}

void LightingPass::onShutdown(backend::IRenderBackend& backend) {
  if (m_lightBuffer != ~0u) backend.destroyBuffer(m_lightBuffer);
  if (m_materialDummy != ~0u) backend.destroyBuffer(m_materialDummy);
  if (m_envBuffer != ~0u) backend.destroyBuffer(m_envBuffer);
  if (m_shadowParams != ~0u) backend.destroyBuffer(m_shadowParams);
  m_lightBuffer = m_materialDummy = ~0u;
  m_pipeline = ~0u;  // the pipeline dies with PipelineStore; nothing owned
}

}  // namespace renderer

KENSHIN_NS_END
