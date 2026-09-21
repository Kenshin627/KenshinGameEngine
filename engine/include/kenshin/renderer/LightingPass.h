#pragma once

// ── LightingPass (M8.1) — deferred lighting over the GBuffer ─────
// The second half of the deferred pair: MaterialDecodePass (its
// fsGBuffer entry) writes the per-slot GBuffer targets; this pass
// samples them, accumulates minimal GGX over the light table and
// writes the final color into the frame's color target (the offscreen
// the ImGui viewport and the screenshots read - the compositional red
// line of this codebase).
//
// The backend binds this frame slot's GBuffer views to the shader's
// reflected Sampler2D bindings automatically (createLightingPipeline /
// drawLighting) - the app only supplies the light SSBO.

#include <glm/glm.hpp>
#include <span>
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

/// One light: directional (toLight + color*intensity) or point
/// (position + color*intensity + range).
struct GpuLightC {
  glm::vec4 a{};  // xyz dir-or-pos, w type (0 dir, 1 point)
  glm::vec4 b{};  // rgb color*intensity, w range
};

class LightingPass {
 public:
  bool init(backend::IRenderBackend& backend, manager::ShaderManager& shaders);
  void onShutdown(backend::IRenderBackend& backend);

  /// Record the fullscreen lighting draw: `lights` uploads into the
  /// host-visible table this frame; `invViewProj` is the inverse of the
  /// camera's own (Y-flipped) view-proj - the exact matrix pair the
  /// decode pass used, so positions roundtrip.
  void draw(backend::IRenderBackend& backend, const glm::mat4& invViewProj,
            const glm::vec3& camPos, std::span<const GpuLightC> lights,
            const glm::vec4& clearColor, float ambient, float exposure,
            u32 width, u32 height);

 private:
  u32 m_pipeline = ~0u;
  u32 m_lightBuffer = ~0u;  // host-visible, 32 lights max
  static constexpr u32 kMaxLights = 32;
};

}  // namespace renderer

KENSHIN_NS_END
