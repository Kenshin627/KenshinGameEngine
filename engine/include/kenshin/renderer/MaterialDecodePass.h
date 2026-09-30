#pragma once

// ── MaterialDecodePass (S4.5) ────────────────────────────────────
// The VisBuffer consumer: a fullscreen-triangle pass whose fragment
// shader decodes each pixel (cluster/triangle/barycentrics from the
// S4.4 VisBuffer) and shades it (Karis p.99). Owns the fullscreen
// pipeline and the per-frame vis buffer upload; the static pools come
// from VirtualGeometryScene's GPU buffers.

#include <glm/glm.hpp>
#include <vector>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace backend {
class IRenderBackend;
}
namespace manager {
class ShaderManager;
}
namespace renderer {
class VirtualGeometryScene;
}

namespace renderer {

/// View mode selector (S4.6 debug views; camFlags.w carries it).
enum class DecodeView : int {
  Final = 0,      // Blinn-Phong, baseline-parity shading
  LodLevel = 1,   // cluster LOD level ramp (near fine / far coarse)
  ErrorHeat = 2,  // log-scaled cluster error heatmap
  VisColor = 3,   // per-triangle hash color (VisBuffer structure)
  Checker = 4,    // UV checkerboard (analytic-derivative debug)
  Clusters = 5,   // per-cluster hash color (cut boundaries)
  RoutePath = 6,  // SW / HW / post-revived rasterizer routing (listSel)
};

struct DecodeParams {
  glm::vec3 camPos{0.0f};
  glm::vec3 lightDir{0.35f, 0.8f, 0.45f};  // TO light, world
  glm::vec3 kd{0.75f, 0.75f, 0.78f};
  DecodeView view = DecodeView::Final;
};

/// M6.3 paged decode inputs: the slot pool (bindings 3..6) + the paged
/// tables (bindings 8..10). The pass owns the tiny decodeParams buffer
/// (binding 11: .x = the paged flag, .y/.z = the slot pool's float4/u32
/// element totals - the address-clamp bounds, the driver-fault dodge).
struct DecodePagedInputs {
  u32 slotPool = 0xFFFFFFFFu;
  u32 clusterPage = 0xFFFFFFFFu;
  u32 pageTable = 0xFFFFFFFFu;
  u32 pageHeader = 0xFFFFFFFFu;
  u32 slotU32Total = 0;           // M7.1 v5: one u32 capacity for the pool
  u32 clusterMeta = 0xFFFFFFFFu;  // v5 per-cluster meta ordinals
};

class MaterialDecodePass {
 public:
  /// Compile decode_material.slang and create the fullscreen pipeline.
  /// The scene's pool/instance buffers must already be uploaded
  /// (flushPools + setInstances).
  bool init(backend::IRenderBackend& backend, manager::ShaderManager& shaders);

  /// Record the fullscreen decode draw in the CURRENT frame scope. The
  /// viewProj must be the camera's OWN matrix (Y-flipped) - the same
  /// one the chain rasterized with, so the VisBuffer rows arrive in
  /// framebuffer order. `visHandle` is the SAME frame's VisBuffer (the
  /// in-frame chain's output; its pixel count must equal w*h).
  /// `pairs`/`hwPairs` are the SAME frame's pair lists (the payload
  /// indirection tables) with their capacity.
  void draw(backend::IRenderBackend& backend, const glm::mat4& viewProj,
            const VirtualGeometryScene& scene, const DecodeParams& params,
            u32 width, u32 height, u32 visHandle, u32 pairs, u32 hwPairs,
            u32 postPairs, u32 pairCapacity);

  /// M6.3 paged variant: explicit handles instead of the scene's v2
  /// pools (the slot pool fills bindings 3..6; cluster/instance/group
  /// tables still come from the scene).
  void drawPaged(backend::IRenderBackend& backend, const glm::mat4& viewProj,
                 const VirtualGeometryScene& scene, const DecodeParams& params,
                 u32 width, u32 height, u32 visHandle,
                 const DecodePagedInputs& paged, u32 pairs, u32 hwPairs,
                 u32 postPairs, u32 pairCapacity);

  /// M8.1 GBuffer variants: the SAME decode chain writing the per-slot
  /// GBuffer targets (fsGBuffer) instead of the final color - the
  /// deferred pair's producer half. Follow with LightingPass::draw.
  /// M8.2: `albedoTexture` is the TextureManager's texture-array handle
  /// (kInvalidTextureHandle binds the backend's 1x1 dummy); the
  /// material table + count come from the scene (paged) or a zero
  /// dummy (v2 - the shader's paged gate keeps it unread).
  void drawGBuffer(backend::IRenderBackend& backend, const glm::mat4& viewProj,
                   const VirtualGeometryScene& scene,
                   const DecodeParams& params, u32 width, u32 height,
                   u32 visHandle, u32 pairs, u32 hwPairs, u32 postPairs,
                   u32 pairCapacity, u32 albedoTexture,
                   u32 imposterPairs = 0);
  void drawGBufferPaged(backend::IRenderBackend& backend,
                        const glm::mat4& viewProj,
                        const VirtualGeometryScene& scene,
                        const DecodeParams& params, u32 width, u32 height,
                        u32 visHandle, const DecodePagedInputs& paged,
                        u32 pairs, u32 hwPairs, u32 postPairs, u32 pairCapacity,
                        u32 albedoTexture, u32 imposterPairs = 0);
  /// M9.1: decode the SHADOW chain's VisBuffer (the light's view) into
  /// light-space depth (fsShadowDepth -> the backend's R32F shadow
  /// target). `viewProj` is the LIGHT's own (Y-flipped) ortho matrix;
  /// `shadowEnabled` false clears the target to "far" so lighting's
  /// sampling stays valid.
  void drawShadowPaged(backend::IRenderBackend& backend,
                       const glm::mat4& viewProj,
                       const VirtualGeometryScene& scene,
                       const DecodeParams& params, u32 width, u32 height,
                       u32 visHandle, const DecodePagedInputs& paged, u32 pairs,
                       u32 hwPairs, u32 postPairs, u32 pairCapacity, u32 level,
                       bool shadowEnabled, const u32 scissorRect[4] = nullptr);
  void drawShadow(backend::IRenderBackend& backend, const glm::mat4& viewProj,
                  const VirtualGeometryScene& scene, const DecodeParams& params,
                  u32 width, u32 height, u32 visHandle, u32 pairs, u32 hwPairs,
                  u32 postPairs, u32 pairCapacity, u32 level,
                  bool shadowEnabled, const u32 scissorRect[4] = nullptr);

  void onShutdown(backend::IRenderBackend& backend);

 private:
  u32 m_pipeline = ~0u;
  u32 m_gbufferPipeline = ~0u;                // M8.1: the fsGBuffer MRT variant
  u32 m_shadowPipeline[3] = {~0u, ~0u, ~0u};  // per-level (M3.3
      // same-frame set-update discipline)
  u32 m_decodeParams = ~0u;  // 32B host-visible: [0] mode flags, [1].x pair cap
  u32 m_dummyMaterial = ~0u;         // v2 material-table stand-in (32B)
  u32 m_dummyU32[2] = {~0u, ~0u};    // v2 stand-ins (bindings 8/9)
  u32 m_dummyPageHeader = ~0u;       // v2 stand-in (binding 10)
  u32 m_dummyPairs[2] = {~0u, ~0u};  // decode-only tests' stand-ins (12/13)
};

}  // namespace renderer

KENSHIN_NS_END
