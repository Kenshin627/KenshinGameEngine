#include "kenshin/renderer/MaterialDecodePass.h"

#include <cstring>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/manager/ShaderManager.h"
#include "kenshin/renderer/VirtualGeometryScene.h"

KENSHIN_NS_BEGIN

namespace renderer {

namespace {}  // namespace

bool MaterialDecodePass::init(backend::IRenderBackend& backend,
                              manager::ShaderManager& shaders) {
  std::vector<u32> vs, fs;
  std::string log;
  if (!shaders.compileFile("decode_material.slang", vs, log, "vsMain") ||
      !shaders.compileFile("decode_material.slang", fs, log, "fsMain")) {
    KENSHIN_ERROR("[decode] shader compile failed: {}", log);
    return false;
  }
  backend::IRenderBackend::FullscreenPipelineInfo info;
  info.vsSpirv = vs.data();
  info.vsBytes = vs.size() * sizeof(u32);
  info.fsSpirv = fs.data();
  info.fsBytes = fs.size() * sizeof(u32);
  m_pipeline = backend.createFullscreenPipeline(info);
  if (m_pipeline == backend::IRenderBackend::kInvalidPipelineHandle) {
    KENSHIN_ERROR("[decode] fullscreen pipeline creation failed");
    return false;
  }
  // M8.1: the GBuffer (fsGBuffer) MRT variant - same bindings/push, the
  // deferred pair's producer half.
  std::vector<u32> fsG;
  if (!shaders.compileFile("decode_material.slang", fsG, log, "fsGBuffer")) {
    KENSHIN_ERROR("[decode] fsGBuffer compile failed: {}", log);
    return false;
  }
  info.fsSpirv = fsG.data();
  info.fsBytes = fsG.size() * sizeof(u32);
  m_gbufferPipeline = backend.createGBufferPipeline(info);
  if (m_gbufferPipeline == backend::IRenderBackend::kInvalidPipelineHandle) {
    KENSHIN_ERROR("[decode] GBuffer pipeline creation failed");
    return false;
  }
  // M9.1: the shadow depth variant (fsShadowDepth -> the R32F target).
  std::vector<u32> fsS;
  if (!shaders.compileFile("decode_material.slang", fsS, log,
                           "fsShadowDepth")) {
    KENSHIN_ERROR("[decode] fsShadowDepth compile failed: {}", log);
    return false;
  }
  info.fsSpirv = fsS.data();
  info.fsBytes = fsS.size() * sizeof(u32);
  // One pipeline INSTANCE per clipmap level: the levels' decodes run
  // in the SAME frame sharing this pass's per-slot descriptor set - a
  // second update would invalidate the first decode's still-recording
  // command buffer (the M3.3 same-frame double-update violation).
  for (int lv = 0; lv < 3; ++lv) {
    m_shadowPipeline[lv] = backend.createShadowDecodePipeline(info);
    if (m_shadowPipeline[lv] ==
        backend::IRenderBackend::kInvalidPipelineHandle) {
      KENSHIN_ERROR("[decode] shadow pipeline creation failed");
      return false;
    }
  }
  // M6.3/M6.4: the mode-switch buffer (binding 11: [0].x paged flag,
  // [0].y/.z pool element totals - the clamp bounds; [1].x pair
  // capacity) + v2 stand-ins for the paged table bindings (8..10) and
  // the pair tables (12/13, params-gated reads only).
  backend::IRenderBackend::BufferDesc desc;
  desc.size = 32;
  desc.usage = backend::IRenderBackend::BufferStorage |
               backend::IRenderBackend::BufferHostVisible;
  desc.debugName = "decode-params";
  m_decodeParams = backend.createBuffer(desc);
  if (m_decodeParams == backend::IRenderBackend::kInvalidBufferHandle)
    return false;
  desc.usage = backend::IRenderBackend::BufferStorage;
  desc.initialData = nullptr;
  desc.size = 4;
  m_dummyU32[0] = backend.createBuffer(desc);
  m_dummyU32[1] = backend.createBuffer(desc);
  m_dummyPairs[0] = backend.createBuffer(desc);
  m_dummyPairs[1] = backend.createBuffer(desc);
  desc.size = 48;  // v5 page header (12 u32)
  m_dummyPageHeader = backend.createBuffer(desc);
  desc.size = 32;  // v7 material record (8 f32) - the v2 stand-in
  m_dummyMaterial = backend.createBuffer(desc);
  return m_dummyU32[0] != backend::IRenderBackend::kInvalidBufferHandle &&
         m_dummyU32[1] != backend::IRenderBackend::kInvalidBufferHandle &&
         m_dummyPairs[0] != backend::IRenderBackend::kInvalidBufferHandle &&
         m_dummyPairs[1] != backend::IRenderBackend::kInvalidBufferHandle &&
         m_dummyPageHeader != backend::IRenderBackend::kInvalidBufferHandle &&
         m_dummyMaterial != backend::IRenderBackend::kInvalidBufferHandle;
}

/// Shared recorder: pool bindings 3..6 = `pool` (the v2 scene pools or
/// the slot pool), paged tables 8..10, the mode buffer 11, the pair
/// tables 12/13 (the payload indirection).
static void recordDecode(
    backend::IRenderBackend& backend, u32 pipeline, u32 decodeParams, int mode,
    const glm::mat4& viewProj, u32 visHandle, u32 clusterBuffer,
    u32 instanceBuffer, u32 groupBuffer, u32 poolPos, u32 poolNrm, u32 poolUv,
    u32 poolIdx, u32 clusterPage, u32 pageTable, u32 pageHeader, bool paged,
    u32 slotU32Total, u32 clusterMeta, u32 pairs, u32 hwPairs, u32 postPairs,
    u32 pairCapacity, const DecodeParams& params, u32 clusterCount,
    u32 groupCount, u32 instanceCount, u32 maxLodLevel, u32 width, u32 height,
    u32 materialTable, u32 materialCount, u32 albedoTexture, u32 shadowLevel,
    bool shadowEnabled) {
  // mode: 0 = final color, 1 = GBuffer MRT, 2 = shadow depth.
  // The push block (decode_material.slang's DecodePush, packed
  // identically - glm raw).
  struct DecodePushC {
    glm::mat4 viewProj;
    glm::vec4 camFlags;
    glm::vec4 lightDir;
    glm::vec4 kdScale;
    glm::vec4 viewport;
  } push;
  push.viewProj = viewProj;
  push.camFlags =
      glm::vec4(params.camPos, static_cast<f32>(static_cast<int>(params.view)));
  // w carries the instance count the shader clamps payload indices with.
  push.lightDir = glm::vec4(params.lightDir, static_cast<f32>(instanceCount));
  // w carries the max LOD level (mode 1 ramp denominator).
  push.kdScale = glm::vec4(params.kd, static_cast<f32>(maxLodLevel));
  // zw carry the table counts the shader clamps payload indices with.
  push.viewport =
      glm::vec4(static_cast<f32>(width), static_cast<f32>(height),
                static_cast<f32>(clusterCount), static_cast<f32>(groupCount));
  // The mode flags + every clamp bound ride binding 11 (the push block
  // is full at 128B): [0].x paged, [0].y/.z pool totals, [1].x the
  // pair-list capacity, [1].y the material count (M8.2).
  auto* dp = static_cast<glm::vec4*>(backend.mappedPtr(decodeParams));
  dp[0] = glm::vec4(paged ? 1.0f : 0.0f, static_cast<f32>(slotU32Total),
                    static_cast<f32>(slotU32Total), 0.0f);
  dp[1] = glm::vec4(static_cast<f32>(pairCapacity),
                    static_cast<f32>(materialCount), 0.0f, 0.0f);
  const u32 handles[17] = {
      visHandle,   clusterBuffer, instanceBuffer, poolPos,     poolNrm,
      poolUv,      poolIdx,       groupBuffer,    clusterPage, pageTable,
      pageHeader,  decodeParams,  pairs,          hwPairs,     postPairs,
      clusterMeta, materialTable};
  switch (mode) {
    case 1:
      backend.drawGBuffer(pipeline, handles, 17, albedoTexture, &push,
                          sizeof(push), width, height);
      break;
    case 2:
      backend.drawShadowDecode(pipeline, handles, 17, &push, sizeof(push),
                               width, height, shadowLevel, shadowEnabled);
      break;
    default:
      backend.drawFullscreen(pipeline, handles, 16, &push, sizeof(push));
      break;
  }
}

void MaterialDecodePass::draw(backend::IRenderBackend& backend,
                              const glm::mat4& viewProj,
                              const VirtualGeometryScene& scene,
                              const DecodeParams& params, u32 width, u32 height,
                              u32 visHandle, u32 pairs, u32 hwPairs,
                              u32 postPairs, u32 pairCapacity) {
  if (m_pipeline == ~0u || width == 0 || height == 0) return;
  if (visHandle == backend::IRenderBackend::kInvalidBufferHandle) return;
  // The decode and its VisBuffer are same-frame now; the size guard
  // remains as a resize-transition backstop.
  if (backend.bufferSize(visHandle) != usize(width) * height * sizeof(u64))
    return;
  recordDecode(backend, m_pipeline, m_decodeParams, 0, viewProj, visHandle,
               scene.clusterBuffer(), scene.instanceBuffer(),
               scene.groupBuffer(), scene.positionBuffer(),
               scene.normalBuffer(), scene.uvBuffer(), scene.indexBuffer(),
               m_dummyU32[0], m_dummyU32[1], m_dummyPageHeader, false, 0,
               m_dummyU32[0], pairs, hwPairs, postPairs, pairCapacity, params,
               scene.clusterCount(), scene.groupCount(), scene.instanceCount(),
               scene.maxLodLevel(), width, height, m_dummyMaterial, 0,
               backend::IRenderBackend::kInvalidTextureHandle, 0, false);
}

void MaterialDecodePass::drawGBuffer(backend::IRenderBackend& backend,
                                     const glm::mat4& viewProj,
                                     const VirtualGeometryScene& scene,
                                     const DecodeParams& params, u32 width,
                                     u32 height, u32 visHandle, u32 pairs,
                                     u32 hwPairs, u32 postPairs,
                                     u32 pairCapacity, u32 albedoTexture) {
  if (m_gbufferPipeline == ~0u || width == 0 || height == 0) return;
  if (visHandle == backend::IRenderBackend::kInvalidBufferHandle) return;
  if (backend.bufferSize(visHandle) != usize(width) * height * sizeof(u64))
    return;
  recordDecode(backend, m_gbufferPipeline, m_decodeParams, true, viewProj,
               visHandle, scene.clusterBuffer(), scene.instanceBuffer(),
               scene.groupBuffer(), scene.positionBuffer(),
               scene.normalBuffer(), scene.uvBuffer(), scene.indexBuffer(),
               m_dummyU32[0], m_dummyU32[1], m_dummyPageHeader, false, 0,
               m_dummyU32[0], pairs, hwPairs, postPairs, pairCapacity, params,
               scene.clusterCount(), scene.groupCount(), scene.instanceCount(),
               scene.maxLodLevel(), width, height, m_dummyMaterial, 0,
               albedoTexture, 0, false);
}

void MaterialDecodePass::drawGBufferPaged(
    backend::IRenderBackend& backend, const glm::mat4& viewProj,
    const VirtualGeometryScene& scene, const DecodeParams& params, u32 width,
    u32 height, u32 visHandle, const DecodePagedInputs& paged, u32 pairs,
    u32 hwPairs, u32 postPairs, u32 pairCapacity, u32 albedoTexture) {
  if (m_gbufferPipeline == ~0u || width == 0 || height == 0) return;
  if (visHandle == backend::IRenderBackend::kInvalidBufferHandle) return;
  if (backend.bufferSize(visHandle) != usize(width) * height * sizeof(u64))
    return;
  recordDecode(backend, m_gbufferPipeline, m_decodeParams, true, viewProj,
               visHandle, scene.clusterBuffer(), scene.instanceBuffer(),
               scene.groupBuffer(), paged.slotPool, paged.slotPool,
               paged.slotPool, paged.slotPool, paged.clusterPage,
               paged.pageTable, paged.pageHeader, true, paged.slotU32Total,
               paged.clusterMeta, pairs, hwPairs, postPairs, pairCapacity,
               params, scene.clusterCount(), scene.groupCount(),
               scene.instanceCount(), scene.maxLodLevel(), width, height,
               scene.materialTableBuffer(), scene.materialCount(),
               albedoTexture, 0, false);
}

void MaterialDecodePass::drawPaged(
    backend::IRenderBackend& backend, const glm::mat4& viewProj,
    const VirtualGeometryScene& scene, const DecodeParams& params, u32 width,
    u32 height, u32 visHandle, const DecodePagedInputs& paged, u32 pairs,
    u32 hwPairs, u32 postPairs, u32 pairCapacity) {
  if (m_pipeline == ~0u || width == 0 || height == 0) return;
  if (visHandle == backend::IRenderBackend::kInvalidBufferHandle) return;
  if (backend.bufferSize(visHandle) != usize(width) * height * sizeof(u64))
    return;
  recordDecode(backend, m_pipeline, m_decodeParams, 0, viewProj, visHandle,
               scene.clusterBuffer(), scene.instanceBuffer(),
               scene.groupBuffer(), paged.slotPool, paged.slotPool,
               paged.slotPool, paged.slotPool, paged.clusterPage,
               paged.pageTable, paged.pageHeader, true, paged.slotU32Total,
               paged.clusterMeta, pairs, hwPairs, postPairs, pairCapacity,
               params, scene.clusterCount(), scene.groupCount(),
               scene.instanceCount(), scene.maxLodLevel(), width, height,
               scene.materialTableBuffer(), scene.materialCount(),
               backend::IRenderBackend::kInvalidTextureHandle, 0, false);
}

void MaterialDecodePass::drawShadowPaged(
    backend::IRenderBackend& backend, const glm::mat4& viewProj,
    const VirtualGeometryScene& scene, const DecodeParams& params, u32 width,
    u32 height, u32 visHandle, const DecodePagedInputs& paged, u32 pairs,
    u32 hwPairs, u32 postPairs, u32 pairCapacity, u32 shadowLevel,
    bool shadowEnabled) {
  if (shadowLevel >= 3 || m_shadowPipeline[shadowLevel] == ~0u || width == 0 ||
      height == 0)
    return;
  // The DISABLED path (clear-to-one) has no VisBuffer to guard on.
  if (shadowEnabled) {
    if (visHandle == backend::IRenderBackend::kInvalidBufferHandle) return;
    if (backend.bufferSize(visHandle) != usize(width) * height * sizeof(u64))
      return;
  }
  recordDecode(backend, m_shadowPipeline[shadowLevel], m_decodeParams, 2,
               viewProj, visHandle, scene.clusterBuffer(),
               scene.instanceBuffer(), scene.groupBuffer(), paged.slotPool,
               paged.slotPool, paged.slotPool, paged.slotPool,
               paged.clusterPage, paged.pageTable, paged.pageHeader, true,
               paged.slotU32Total, paged.clusterMeta, pairs, hwPairs, postPairs,
               pairCapacity, params, scene.clusterCount(), scene.groupCount(),
               scene.instanceCount(), scene.maxLodLevel(), width, height,
               scene.materialTableBuffer(), scene.materialCount(),
               backend::IRenderBackend::kInvalidTextureHandle, shadowLevel,
               shadowEnabled);
}

void MaterialDecodePass::drawShadow(
    backend::IRenderBackend& backend, const glm::mat4& viewProj,
    const VirtualGeometryScene& scene, const DecodeParams& params, u32 width,
    u32 height, u32 visHandle, u32 pairs, u32 hwPairs, u32 postPairs,
    u32 pairCapacity, u32 shadowLevel, bool shadowEnabled) {
  if (shadowLevel >= 3 || m_shadowPipeline[shadowLevel] == ~0u || width == 0 ||
      height == 0)
    return;
  if (shadowEnabled) {
    if (visHandle == backend::IRenderBackend::kInvalidBufferHandle) return;
    if (backend.bufferSize(visHandle) != usize(width) * height * sizeof(u64))
      return;
  }
  recordDecode(
      backend, m_shadowPipeline[shadowLevel], m_decodeParams, 2, viewProj,
      visHandle, scene.clusterBuffer(), scene.instanceBuffer(),
      scene.groupBuffer(), scene.positionBuffer(), scene.normalBuffer(),
      scene.uvBuffer(), scene.indexBuffer(), m_dummyU32[0], m_dummyU32[1],
      m_dummyPageHeader, false, 0, m_dummyU32[0], pairs, hwPairs, postPairs,
      pairCapacity, params, scene.clusterCount(), scene.groupCount(),
      scene.instanceCount(), scene.maxLodLevel(), width, height,
      m_dummyMaterial, 0, backend::IRenderBackend::kInvalidTextureHandle,
      shadowLevel, shadowEnabled);
}

void MaterialDecodePass::onShutdown(backend::IRenderBackend& backend) {
  if (m_decodeParams != ~0u) backend.destroyBuffer(m_decodeParams);
  for (const u32 h : m_dummyU32)
    if (h != ~0u) backend.destroyBuffer(h);
  for (const u32 h : m_dummyPairs)
    if (h != ~0u) backend.destroyBuffer(h);
  if (m_dummyPageHeader != ~0u) backend.destroyBuffer(m_dummyPageHeader);
  if (m_dummyMaterial != ~0u) backend.destroyBuffer(m_dummyMaterial);
  m_decodeParams = m_dummyPageHeader = m_dummyMaterial = ~0u;
  m_dummyU32[0] = m_dummyU32[1] = ~0u;
  m_dummyPairs[0] = m_dummyPairs[1] = ~0u;
  m_pipeline = m_gbufferPipeline = ~0u;  // pipelines die with PipelineStore
  for (int lv = 0; lv < 3; ++lv) m_shadowPipeline[lv] = ~0u;
}

}  // namespace renderer

KENSHIN_NS_END
