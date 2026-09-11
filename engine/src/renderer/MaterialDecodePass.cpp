#include "kenshin/renderer/MaterialDecodePass.h"

#include <cstring>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/manager/ShaderManager.h"
#include "kenshin/renderer/VirtualGeometryScene.h"

KENSHIN_NS_BEGIN

namespace renderer {

namespace {
}  // namespace

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
  // M6.3: the paged-mode switch (binding 11) + v2 stand-ins for the
  // paged table bindings (8..10, params-gated reads).
  backend::IRenderBackend::BufferDesc desc;
  desc.size = 16;
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
  desc.size = 24;
  m_dummyPageHeader = backend.createBuffer(desc);
  return m_dummyU32[0] != backend::IRenderBackend::kInvalidBufferHandle &&
         m_dummyU32[1] != backend::IRenderBackend::kInvalidBufferHandle &&
         m_dummyPageHeader != backend::IRenderBackend::kInvalidBufferHandle;
}

/// Shared recorder: pool bindings 3..6 = `pool` (the v2 scene pools or
/// the slot pool), paged tables 8..10, the flag buffer 11.
static void recordDecode(
    backend::IRenderBackend& backend, u32 pipeline, u32 decodeParams,
    const glm::mat4& viewProj, u32 visHandle, u32 clusterBuffer,
    u32 instanceBuffer, u32 groupBuffer, u32 poolPos, u32 poolNrm,
    u32 poolUv, u32 poolIdx, u32 clusterPage, u32 pageTable,
    u32 pageHeader, bool paged, u32 slotVertexTotal, u32 slotIndexTotal,
    const DecodeParams& params, u32 clusterCount,
    u32 groupCount, u32 instanceCount, u32 maxLodLevel, u32 width,
    u32 height) {
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
  // The paged flag + the address-clamp totals ride binding 11 (the
  // push block is full at 128B).
  *static_cast<glm::vec4*>(backend.mappedPtr(decodeParams)) = glm::vec4(
      paged ? 1.0f : 0.0f, static_cast<f32>(slotVertexTotal),
      static_cast<f32>(slotIndexTotal), 0.0f);
  const u32 handles[12] = {visHandle,  clusterBuffer, instanceBuffer,
                           poolPos,    poolNrm,       poolUv,
                           poolIdx,    groupBuffer,   clusterPage,
                           pageTable,  pageHeader,    decodeParams};
  backend.drawFullscreen(pipeline, handles, 12, &push, sizeof(push));
}

void MaterialDecodePass::draw(backend::IRenderBackend& backend,
                              const glm::mat4& viewProj,
                              const VirtualGeometryScene& scene,
                              const DecodeParams& params, u32 width, u32 height,
                              u32 visHandle) {
  if (m_pipeline == ~0u || width == 0 || height == 0) return;
  if (visHandle == backend::IRenderBackend::kInvalidBufferHandle) return;
  // The decode and its VisBuffer are same-frame now; the size guard
  // remains as a resize-transition backstop.
  if (backend.bufferSize(visHandle) != usize(width) * height * sizeof(u64))
    return;
  recordDecode(backend, m_pipeline, m_decodeParams, viewProj, visHandle,
               scene.clusterBuffer(), scene.instanceBuffer(),
               scene.groupBuffer(), scene.positionBuffer(),
               scene.normalBuffer(), scene.uvBuffer(), scene.indexBuffer(),
               m_dummyU32[0], m_dummyU32[1], m_dummyPageHeader, false, 0, 0,
               params, scene.clusterCount(), scene.groupCount(),
               scene.instanceCount(), scene.maxLodLevel(), width, height);
}

void MaterialDecodePass::drawPaged(backend::IRenderBackend& backend,
                                   const glm::mat4& viewProj,
                                   const VirtualGeometryScene& scene,
                                   const DecodeParams& params, u32 width,
                                   u32 height, u32 visHandle,
                                   const DecodePagedInputs& paged) {
  if (m_pipeline == ~0u || width == 0 || height == 0) return;
  if (visHandle == backend::IRenderBackend::kInvalidBufferHandle) return;
  if (backend.bufferSize(visHandle) != usize(width) * height * sizeof(u64))
    return;
  recordDecode(backend, m_pipeline, m_decodeParams, viewProj, visHandle,
               scene.clusterBuffer(), scene.instanceBuffer(),
               scene.groupBuffer(), paged.slotPool, paged.slotPool,
               paged.slotPool, paged.slotPool, paged.clusterPage,
               paged.pageTable, paged.pageHeader, true,
               paged.slotVertexTotal, paged.slotIndexTotal, params,
               scene.clusterCount(), scene.groupCount(), scene.instanceCount(),
               scene.maxLodLevel(), width, height);
}

void MaterialDecodePass::onShutdown(backend::IRenderBackend& backend) {
  if (m_decodeParams != ~0u) backend.destroyBuffer(m_decodeParams);
  for (const u32 h : m_dummyU32)
    if (h != ~0u) backend.destroyBuffer(h);
  if (m_dummyPageHeader != ~0u) backend.destroyBuffer(m_dummyPageHeader);
  m_decodeParams = m_dummyPageHeader = ~0u;
  m_dummyU32[0] = m_dummyU32[1] = ~0u;
  m_pipeline = ~0u;  // the pipeline dies with PipelineStore; nothing owned
}

}  // namespace renderer

KENSHIN_NS_END
