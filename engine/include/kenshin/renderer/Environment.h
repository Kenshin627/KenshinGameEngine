#pragma once

// ── Environment — the IBL probe (HDRI or procedural studio) ─────────
// One equirectangular RGBA16F texture (Z-up: u = atan2(y,x)/2pi + 0.5,
// v = acos(z)/pi - lighting.slang's convention) whose GPU mip chain is
// the prefiltered radiance ladder, plus its SH irradiance for diffuse.
// Captured HDRIs get their sun split out (the Filament / UE sky+sun
// split): texels far above the sky are summed into a directional light
// and clamped in the probe, so the key light and its shadows agree
// with the reflections instead of lighting the scene a second time.

#include <filesystem>

#include <glm/glm.hpp>

#include "kenshin/Common.h"
#include "kenshin/renderer/LightingPass.h"

KENSHIN_NS_BEGIN

namespace backend {
class IRenderBackend;
}

namespace renderer {

struct EnvironmentProbe {
  u32 texture = 0xFFFFFFFFu;  // IRenderBackend texture-array handle
  f32 maxLod = 1.0f;          // highest prefiltered mip
  LightingPass::IrradianceSH sh{};
  bool shValid = false;
  /// A captured HDRI (true) or the procedural studio probe (false).
  bool captured = false;
  /// Sun split out of a captured HDRI: direction TO the sun, normalized
  /// color and irradiance peak (directional light = color * intensity).
  bool hasSun = false;
  glm::vec3 sunDir{0.0f, 0.0f, 1.0f};
  glm::vec3 sunColor{1.0f};
  f32 sunIntensity = 0.0f;

  bool valid() const { return texture != 0xFFFFFFFFu; }
};

/// Load a Radiance .hdr (sun extraction included). Returns false - and
/// leaves `out` untouched - when the file is missing or undecodable.
bool loadHdrEnvironment(backend::IRenderBackend& backend,
                        const std::filesystem::path& path,
                        EnvironmentProbe& out);

/// The procedural HDR studio probe: dim sky/ground gradient, a cool fill
/// and a bright softbox centered on the key light (TO light, Z-up).
/// KENSHIN_ENV_SCALE scales the ambient part (calibration knob).
EnvironmentProbe createStudioProbe(backend::IRenderBackend& backend);

}  // namespace renderer

KENSHIN_NS_END
