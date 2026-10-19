#include "kenshin/renderer/ImposterPass.h"

#include <cstring>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/manager/ShaderManager.h"

KENSHIN_NS_BEGIN

namespace renderer {

namespace {
constexpr u32 kMaxQuads = 64;                // tiny instances per frame
constexpr u32 kMaxPairs = 256;               // the decode clamps pairIdx to 255
constexpr u32 kQuadsWords = kMaxQuads * 16;  // 4 float4 per quad
constexpr u32 kParamsBytes = 8 * 16;         // float4[8]
}  // namespace

bool ImposterPass::init(backend::IRenderBackend& backend,
                        manager::ShaderManager& shaders) {
  std::vector<u32> cs;
  std::string log;
  if (!shaders.compileFile("imposter_inject.slang", cs, log, "main")) {
    KENSHIN_ERROR("[imposter] shader compile failed: {}", log);
    return false;
  }
  backend::IRenderBackend::ComputePipelineInfo info;
  info.csSpirv = cs.data();
  info.csBytes = cs.size() * sizeof(u32);
  m_pipeline = backend.createComputePipeline(info);
  if (m_pipeline == backend::IRenderBackend::kInvalidPipelineHandle) {
    KENSHIN_ERROR("[imposter] pipeline creation failed");
    return false;
  }
  const auto host = [&](usize bytes, const char* name) {
    backend::IRenderBackend::BufferDesc desc;
    desc.size = bytes;
    desc.usage = backend::IRenderBackend::BufferStorage |
                 backend::IRenderBackend::BufferHostVisible;
    desc.initialData = nullptr;
    desc.debugName = name;
    return backend.createBuffer(desc);
  };
  m_params = host(kParamsBytes, "imposter-params");
  m_quads = host(kQuadsWords * 4, "imposter-quads");
  m_pairs = host(kMaxPairs * 8, "imposter-pairs");
  m_atlas = host(1, "imposter-atlas");  // replaced by uploadAtlas
  return m_params != backend::IRenderBackend::kInvalidBufferHandle &&
         m_quads != backend::IRenderBackend::kInvalidBufferHandle &&
         m_pairs != backend::IRenderBackend::kInvalidBufferHandle &&
         m_atlas != backend::IRenderBackend::kInvalidBufferHandle;
}

u32 ImposterPass::uploadAtlas(backend::IRenderBackend& backend,
                              const u32* depth, const u32* tri, u32 words) {
  // Atlases append into one growing SSBO; the return is the WORD base
  // the quads carry. (Re)creation copies the old contents.
  if (m_atlasWords + words > m_atlasCapacity) {
    // Grown as HOST-VISIBLE: the appends write through the mapped
    // pointer (the backend has no updateBuffer - the mapped-write is
    // the codebase's CPU->GPU currency).
    const u32 newCap = std::max(m_atlasCapacity * 2u, m_atlasWords + words);
    std::vector<u32> fresh(usize(newCap) * 2, 0);
    if (m_atlasWords > 0) {
      std::vector<u8> old(usize(m_atlasWords) * 8);
      if (!backend.readbackBuffer(m_atlas, 0, old.size(), old)) return kNoAtlas;
      std::memcpy(fresh.data(), old.data(), old.size());
    }
    backend::IRenderBackend::BufferDesc desc;
    desc.size = usize(newCap) * 8;
    desc.usage = backend::IRenderBackend::BufferStorage |
                 backend::IRenderBackend::BufferHostVisible;
    desc.initialData = fresh.data();
    desc.debugName = "imposter-atlas";
    const u32 handle = backend.createBuffer(desc);
    if (handle == backend::IRenderBackend::kInvalidBufferHandle)
      return kNoAtlas;
    if (m_atlas != backend::IRenderBackend::kInvalidBufferHandle)
      backend.destroyBuffer(m_atlas);
    m_atlas = handle;
    m_atlasCapacity = newCap;
  }
  // The shader reads uint2 per texel: interleave the planes.
  u32* dst =
      static_cast<u32*>(backend.mappedPtr(m_atlas)) + usize(m_atlasWords) * 2;
  for (u32 i = 0; i < words; ++i) {
    dst[usize(i) * 2] = depth[i];
    dst[usize(i) * 2 + 1] = tri[i];
  }
  const u32 base = m_atlasWords;
  m_atlasWords += words;
  return base;
}

void ImposterPass::setPairs(backend::IRenderBackend& backend,
                            const glm::uvec2* pairs, u32 count) {
  if (m_pairs == backend::IRenderBackend::kInvalidBufferHandle ||
      count > kMaxPairs)
    return;
  std::memcpy(backend.mappedPtr(m_pairs), pairs, usize(count) * 8);
}

void ImposterPass::draw(backend::IRenderBackend& backend, u32 visBuffer,
                        const glm::mat4& viewProj, const glm::vec3& camPos,
                        u32 width, u32 height, const ImposterQuad* quads,
                        u32 quadCount) {
  if (m_pipeline == ~0u || quadCount == 0 ||
      visBuffer == backend::IRenderBackend::kInvalidBufferHandle)
    return;
  quadCount = std::min(quadCount, kMaxQuads);

  auto* p = static_cast<glm::vec4*>(backend.mappedPtr(m_params));
  for (int c = 0; c < 4; ++c) p[c] = viewProj[c];
  p[4] = glm::vec4(camPos, static_cast<f32>(width));
  p[5] =
      glm::vec4(1.0f / static_cast<f32>(width), 1.0f / static_cast<f32>(height),
                static_cast<f32>(height), static_cast<f32>(quadCount));

  auto* q = static_cast<glm::vec4*>(backend.mappedPtr(m_quads));
  std::memcpy(q, quads, sizeof(ImposterQuad) * quadCount);

  // Slot snapshots: the params/quads carry JITTERED viewProj/rects
  // (different bytes every frame) in a shared host block - binding it
  // directly lets this frame's CPU rewrite tear the previous frame's
  // still-running dispatch.
  const u32 handles[4] = {visBuffer, backend.snapshotHostBuffer(m_params),
                          backend.snapshotHostBuffer(m_quads), m_atlas};
  const u32 groups =
      (quadCount * 64 + 63) / 64;  // one 64-thread group per quad row span
  backend.dispatchCompute(m_pipeline, handles, 4, quadCount, 1, 1);
  // Compute write -> the decode's fragment read.
  backend.bufferBarrier(visBuffer, false);
}

void ImposterPass::onShutdown(backend::IRenderBackend& backend) {
  for (const u32 h : {m_params, m_quads, m_pairs, m_atlas})
    if (h != backend::IRenderBackend::kInvalidBufferHandle)
      backend.destroyBuffer(h);
  m_params = m_quads = m_pairs = m_atlas = ~0u;
  m_pipeline = ~0u;
}

}  // namespace renderer

KENSHIN_NS_END
