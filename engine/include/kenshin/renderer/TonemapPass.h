#pragma once

// ── TonemapPass (M8.3) — the HDR chain's present half ────────────
// lighting.slang writes LINEAR HDR into the backend's per-slot
// RGBA16F target; this pass samples it, applies exposure + the
// selected operator (extended Reinhard / ACES fitted) and writes the
// final LDR into the frame's color target (slot.offscreen - the imgui
// viewport and the screenshots only ever read THAT, the compositional
// red line). Background (exact zero in the HDR target) passes the
// frame's clear color through untonemapped, keeping the regression
// net's "expected color" invariant byte-exact.

#include <glm/glm.hpp>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace backend {
class IRenderBackend;
}
namespace manager {
class ShaderManager;
}

namespace renderer {

class TonemapPass {
 public:
  bool init(backend::IRenderBackend& backend, manager::ShaderManager& shaders);
  void onShutdown(backend::IRenderBackend& backend);

  /// Operator selection (the panel combo).
  enum class Operator : int {
    Reinhard = 0,
    Aces = 1,
  };

  /// Record the fullscreen tonemap draw. Follows LightingPass::draw
  /// every deferred frame; reads this slot's HDR target implicitly.
  /// bloomStrength > 0 composites the BloomPass chain drawn this frame
  /// (bloomInvLevels = 1 / BloomPass::kLevels); 0 = no bloom.
  /// exposureState (ExposurePass::stateBuffer, written this frame)
  /// replaces `exposure` with the adapted scale; ~0u = manual.
  void draw(backend::IRenderBackend& backend, Operator op, float exposure,
            const glm::vec4& clearColor, u32 width, u32 height,
            bool fromTaa = false, float sharpen = 0.0f,
            float bloomStrength = 0.0f, float bloomInvLevels = 0.0f,
            u32 exposureState = ~0u);

 private:
  u32 m_pipeline = ~0u;
  u32 m_dummyState = ~0u;  // bound when auto exposure is off
};

}  // namespace renderer

KENSHIN_NS_END
