#pragma once

// ── ExposurePass — histogram auto exposure (exposure.slang) ─────────
// UE's AEM_Histogram eye adaptation: a 64-bin log2-luminance histogram
// of the resolved HDR, an outlier-trimmed mean (low/high percent), the
// target exposure that lands that mean on 18% grey * 2^bias, and UE's
// linear-then-exponential temporal adaptation. Runs after the TAA
// resolve; TonemapPass reads stateBuffer() in the same frame. The state
// buffer is host-visible, so the panel can show the live exposure.

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

/// UE 5.8 defaults (FCameraExposureSettings, non-extended range).
struct ExposureSettings {
  float biasEv = 1.0f;          // AutoExposureBias (f-stops; UE5
                                // r.DefaultFeature.AutoExposure.Bias)
  float lowPercent = 10.0f;     // AutoExposureLowPercent
  float highPercent = 90.0f;    // AutoExposureHighPercent
  float minBrightness = 0.03f;  // AutoExposureMinBrightness (white point)
  float maxBrightness = 8.0f;   // AutoExposureMaxBrightness
  float speedUp = 3.0f;         // f-stops per second
  float speedDown = 1.0f;
  float histogramLogMin = -8.0f;
  float histogramLogMax = 4.0f;
};

/// What the meter sees behind the geometry: the IBL environment in each
/// background pixel's view direction (UE meters the sky too).
struct ExposureView {
  glm::mat4 invViewProj{1.0f};  // unjittered
  u32 envTexture = ~0u;         // ~0u = background pixels ignored
  float envMip = 4.0f;          // a blurred level (metering, not detail)
  float envIntensity = 1.0f;
};

class ExposurePass {
 public:
  bool init(backend::IRenderBackend& backend, manager::ShaderManager& shaders);
  void onShutdown(backend::IRenderBackend& backend);

  /// Record histogram + adaptation over this frame's resolved HDR.
  /// `forceTarget` jumps straight to the target (headless screenshots -
  /// deterministic regardless of frame timing - and camera cuts).
  bool draw(backend::IRenderBackend& backend, const ExposureSettings& s,
            const ExposureView& view, bool fromTaa, u32 width, u32 height,
            float dt, bool forceTarget);

  bool ready() const { return m_ready; }
  u32 stateBuffer() const { return m_state; }
  /// Last written state (scale, targetScale, avgLum, valid) - read from
  /// the host-visible buffer, one or two frames behind the GPU.
  bool readState(backend::IRenderBackend& backend, float out[4]) const;

 private:
  u32 m_histogramPipeline = ~0u;
  u32 m_adaptPipeline = ~0u;
  u32 m_histogram = ~0u;
  u32 m_state = ~0u;
  bool m_ready = false;
};

}  // namespace renderer

KENSHIN_NS_END
