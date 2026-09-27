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
  void draw(backend::IRenderBackend& backend, Operator op, float exposure,
            const glm::vec4& clearColor, u32 width, u32 height,
            bool fromTaa = false);

 private:
  u32 m_pipeline = ~0u;
};

}  // namespace renderer

KENSHIN_NS_END
