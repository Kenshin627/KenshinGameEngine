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

struct DecodeParams {
  glm::vec3 camPos{0.0f};
  glm::vec3 lightDir{0.35f, 0.8f, 0.45f};  // TO light, world
  glm::vec3 kd{0.75f, 0.75f, 0.78f};
  bool checkerboard = false;  // UV debug view
};

class MaterialDecodePass {
 public:
  /// Compile decode_material.slang and create the fullscreen pipeline.
  /// The scene's pool/instance buffers must already be uploaded
  /// (flushPools + setInstances).
  bool init(backend::IRenderBackend& backend, manager::ShaderManager& shaders);

  /// Replace the VisBuffer contents (from the S4.4 one-shot chain).
  void updateVis(backend::IRenderBackend& backend, std::vector<u64> vis);

  /// Record the fullscreen decode draw in the CURRENT frame scope. The
  /// viewProj must be the camera's OWN matrix (Y-flipped) - the same
  /// one the chain rasterized with, so the VisBuffer rows arrive in
  /// framebuffer order.
  void draw(backend::IRenderBackend& backend, const glm::mat4& viewProj,
            const VirtualGeometryScene& scene, const DecodeParams& params,
            u32 width, u32 height);

  void onShutdown(backend::IRenderBackend& backend);

 private:
  u32 m_pipeline = ~0u;
  u32 m_visBuffer = ~0u;
  usize m_visCount = 0;
};

}  // namespace renderer

KENSHIN_NS_END
