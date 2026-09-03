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
};

struct DecodeParams {
  glm::vec3 camPos{0.0f};
  glm::vec3 lightDir{0.35f, 0.8f, 0.45f};  // TO light, world
  glm::vec3 kd{0.75f, 0.75f, 0.78f};
  DecodeView view = DecodeView::Final;
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
  void draw(backend::IRenderBackend& backend, const glm::mat4& viewProj,
            const VirtualGeometryScene& scene, const DecodeParams& params,
            u32 width, u32 height, u32 visHandle);

  void onShutdown(backend::IRenderBackend& backend);

 private:
  u32 m_pipeline = ~0u;
};

}  // namespace renderer

KENSHIN_NS_END
