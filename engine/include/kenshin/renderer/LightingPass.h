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

#include <array>
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
  /// decode pass used, so positions roundtrip. M8.2: `materialTable`
  /// (the scene's 8-f32 records) feeds the per-pixel
  /// roughness/metallic the decode pass indexed by GBuffer B's alpha;
  /// `materialCount` clamps it (0 = a 1-entry dummy keeps the binding
  /// validly written and shades with the defaults).
  /// M8.3: skyColor/groundColor drive the hemisphere ambient (the env
  /// table, binding 4); output is LINEAR HDR into the backend's
  /// RGBA16F slot target - follow with TonemapPass::draw.
  /// M9.1 Stage B: the clipmap rig - `levels[L].viewProj` are the
  /// per-level light ortho matrices (extent0 * 2^L), bias/texel per
  /// level; extent0 + origin drive the sampling side's log2 level
  /// selection. levelCount 0..3 (0 = disabled, bindings stay valid).
  struct ShadowLevel {
    glm::mat4 viewProj{1.0f};
    float bias = 0.0015f;
    float texelWorld = 1.0f;
  };
  /// Diffuse IBL: the environment probe's irradiance as 9 L2 SH
  /// coefficients (rgb in xyz), ALREADY cosine-convolved and divided by
  /// pi - the shader's diffuse ambient is albedo * sum(c_k * Y_k(n)).
  /// The same probe feeds the specular mips, so diffuse and specular
  /// ambient stay consistent (UE's sky light: SH irradiance + the
  /// prefiltered cubemap from ONE capture, one intensity).
  using IrradianceSH = std::array<glm::vec4, 9>;
  /// Project an equirectangular Z-up RGBA f32 radiance map (u =
  /// atan2(y,x)/2pi + 0.5, v = acos(z)/pi - lighting.slang's sampling
  /// convention) onto IrradianceSH.
  static IrradianceSH projectIrradianceSH(const float* rgba, u32 width,
                                          u32 height);

  /// `envSH` null = no probe: the hemisphere sky/ground * ambient
  /// fallback. `envIntensity` scales BOTH IBL terms (diffuse SH and the
  /// specular probe).
  void draw(backend::IRenderBackend& backend, const glm::mat4& invViewProj,
            const glm::vec3& camPos, std::span<const GpuLightC> lights,
            u32 materialTable, u32 materialCount, const glm::vec4& clearColor,
            float ambient, const glm::vec3& skyColor,
            const glm::vec3& groundColor, const ShadowLevel* levels,
            u32 levelCount, float extent0, const glm::vec3& origin, u32 width,
            u32 height, u32 envTexture, float envMaxLod,
            const IrradianceSH* envSH = nullptr, float envIntensity = 1.0f);

 private:
  u32 m_pipeline = ~0u;
  u32 m_lightBuffer = ~0u;    // host-visible, 32 lights max
  u32 m_materialDummy = ~0u;  // 1-entry default table (v2/unpaged path)
  u32 m_envBuffer = ~0u;      // host-visible, 11 float4s: [0] sky rgb +
                              // SH flag, [1] ground rgb + env intensity,
                              // [2..10] irradiance SH
  u32 m_shadowParams = ~0u;   // M9.1 Stage B: 20 float4s - header,
                              // origin, 3 x (pad, params, lvp x4)
  static constexpr u32 kMaxLights = 32;
};

}  // namespace renderer

KENSHIN_NS_END
