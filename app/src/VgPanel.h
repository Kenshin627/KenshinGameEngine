#pragma once

// The demo's "Virtual Geometry" panel: the UI model (VgDebugState), its
// ImGui drawing, the mapping to the engine's SceneRenderSettings (UI +
// CLI A/B switches -> resolved knobs) and the stats readback.

#include <vector>

#include <glm/glm.hpp>

#include "AppArgs.h"
#include "KgeoPreview.h"
#include "kenshin/renderer/SceneRenderer.h"

namespace kenshin::renderer {
class StreamingManager;
}

namespace demo {

/// S4.6 debug state: view mode + the chain's last stats and histories
/// (the panel reads it; updated per frame from the chain stats).
struct VgDebugState {
  int viewMode = 0;  // DecodeView as int (0..6)
  kenshin::u32 visibleClusters = 0, cutTris = 0, pxWrites = 0;
  kenshin::u32 bfCulls = 0, nearClips = 0;  // M9.1 Stage B ground debug
  kenshin::u32 occCulled = 0;
  kenshin::u32 revived = 0;
  kenshin::u32 hwClusters = 0;
  kenshin::u32 workItems = 0, tooCoarse = 0, srcTooCoarse = 0;
  kenshin::u32 pendingSkips = 0;  // M6.3 non-resident skips
  kenshin::u32 missPages = 0;     // M6.3 touched-absent pages
  bool occlusion = true;
  bool hierCull = true;  // M5.2: BVH8 traversal (off = linear A/B)
  // HW-path pixel writes in the writes/coverage stats (costly: one
  // atomic per HW fragment) - off = SW pixels only.
  bool hwPxStats = false;
  // M8.1 lighting rig (panel-adjustable; the light TABLE stays
  // deterministic - static positions, no time animation, keeping the
  // render regression byte-comparable).
  glm::vec3 lightDir{0.35f, 0.8f, 0.45f};
  // Neutral key light: a warm tint (the old 1.0/0.97/0.92) biases
  // r-g by ~8% - enough to flip every low-saturation olive/green
  // texture patch to warm tan after lighting.
  glm::vec3 lightColor{1.0f, 1.0f, 1.0f};
  // Recalibrated for textured assets: the old 4.0 x 2.2 (~9x gain)
  // auto-normalized every albedo into the filmic shoulder (a narrow
  // bright band) - hue structure died, dark camo textures rendered
  // as flat warm tan. New gain keeps mid-grays mid and dark textures
  // dark enough for their color to read.
  kenshin::f32 lightIntensity = 2.4f;
  kenshin::f32 pointIntensity = 1.2f;
  kenshin::f32 ambient = 0.15f;  // hemisphere fallback only (no probe)
  // The IBL probe's intensity on BOTH ambient terms (SH diffuse +
  // prefiltered specular).
  kenshin::f32 envIntensity = 0.7f;
  kenshin::f32 exposure = 1.5f;
  // Histogram auto exposure (ExposurePass, UE AEM_Histogram); off =
  // the manual `exposure` above. autoScale: the GPU's last adapted
  // scale (panel readout; also the TAA's HDR weight).
  bool autoExposure = true;
  kenshin::renderer::ExposureSettings exposureSettings;
  float autoScale = 0.0f, autoAvgLum = 0.0f;
  // The IBL environment behind the model (tonemap.slang); skyBlur =
  // the prefiltered mip it samples (0 = sharp, the Sketchfab-style
  // background blur above that).
  bool sky = true;
  kenshin::f32 skyBlur = 0.0f;
  // HDR bloom (BloomPass): the composite weight s of
  // scene * (1 - s) + bloom * s, and the emissive multiplier (1 = the
  // asset's authored emissiveFactor). emissivePassThrough: opaque
  // emission behind translucent layers is not dimmed by them (the
  // Sketchfab look - the Vulture's eyes behind a 91%-opaque visor; the
  // physical alternative needed a global x16 that overdrove every other
  // asset's emissive maps).
  bool bloom = true;
  kenshin::f32 bloomStrength = 0.08f;
  kenshin::f32 emissiveStrength = 1.0f;
  bool emissivePassThrough = true;
  // M8.3: the hemisphere ambient's two poles + tonemap operator.
  // Neutral-gray hemisphere by default: colored poles (the old blue
  // sky / warm ground) swing r-g by normal direction more than a dark
  // texture's own albedo signal - the camo washed out. Luminance-only
  // ambient keeps the shape cue; colored ambients stay a panel knob.
  glm::vec3 skyColor{0.6f, 0.6f, 0.6f};
  glm::vec3 groundColor{0.3f, 0.3f, 0.3f};
  int tonemapMode = 2;         // 0 Reinhard, 1 ACES(Narkowicz), 2 Filmic(UE)
  bool shadows = true;         // M9.1: the directional light's shadow map
  int shadowLevels = 3;        // Stage B clipmap level count (1..3)
  bool taa = true;             // M9 temporal AA (jitter + history resolve)
  bool imposterDebug = false;  // M9.2: replace extra-mesh instances
  kenshin::f32 taaSharpen = 0.3f;      // post-resolve unsharp (0 = off)
  kenshin::f32 pixelThreshold = 1.0f;  // the CHAIN's LOD threshold
  kenshin::f32 hwThresholdPx = 1.0f;   // SW/HW routing edge threshold
                                       // (moved off KgeoPreview at the
                                       // M4.4 CPU-cut retirement)
  kenshin::f32 coveragePct = 0.0f;
  std::vector<float> clusterHist, triHist, coverHist;
  std::vector<float> occHist, reviveHist;  // M3.4: occlusion curves
  std::vector<float> hwHist;               // M4.4: HW-routing curve
  void pushHistory() {
    clusterHist.push_back(static_cast<float>(visibleClusters));
    triHist.push_back(static_cast<float>(cutTris));
    coverHist.push_back(coveragePct);
    occHist.push_back(static_cast<float>(occCulled));
    reviveHist.push_back(static_cast<float>(revived));
    hwHist.push_back(static_cast<float>(hwClusters));
    visitedHist.push_back(static_cast<float>(visitedNodes));
    prunedHist.push_back(static_cast<float>(prunedNodes));
    pendingHist.push_back(static_cast<float>(pendingSkips));
    missHist.push_back(static_cast<float>(missPages));
    if (clusterHist.size() > 240) {
      clusterHist.erase(clusterHist.begin());
      triHist.erase(triHist.begin());
      coverHist.erase(coverHist.begin());
      occHist.erase(occHist.begin());
      reviveHist.erase(reviveHist.begin());
      hwHist.erase(hwHist.begin());
      visitedHist.erase(visitedHist.begin());
      prunedHist.erase(prunedHist.begin());
      pendingHist.erase(pendingHist.begin());
      missHist.erase(missHist.begin());
    }
  }
  unsigned visitedNodes = 0;
  unsigned prunedNodes = 0;
  std::vector<float> visitedHist;
  std::vector<float> prunedHist;
  std::vector<float> pendingHist;  // M6.3: non-resident skip curve
  std::vector<float> missHist;     // M6.3: touched-absent page curve
};

/// Draw the panel (ImGui frame scope).
void drawVgPanel(VgDebugState& vgDebug, KgeoPreview& kgeoPreview,
                 const kenshin::renderer::StreamingManager* streamer);

/// UI state + the command line's A/B switches -> the frame's settings.
kenshin::renderer::SceneRenderSettings resolveSettings(
    const VgDebugState& vgDebug, const Args& args);

/// Copy the renderer's last stats into the panel model (+ histories).
void absorbStats(VgDebugState& vgDebug,
                 const kenshin::renderer::SceneRenderStats& stats);

}  // namespace demo
