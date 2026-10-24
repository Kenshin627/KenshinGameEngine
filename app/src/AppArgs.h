#pragma once

// Demo command line: every flag the KENSHINDemo binary accepts (the
// research A/B switches included) and the demo's fixed constants.

#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "kenshin/Common.h"

namespace demo {

struct Args {
  int frames = -1;  // -1 = run until the user closes
  int width = 1600;
  int height = 900;
  bool validation = true;     // dev default ON; --no-validation to disable
  std::string kgeo;           // .kgeo virtual-geometry preview (S3.4)
  float lodThreshold = 1.0f;  // initial LOD pixel-error threshold
  std::string mesh = "all";   // all (full scene) | cube | sphere |
                              // torus | cylinder | cone | knot | <path.obj>
  std::string screenshot;
  std::string screenshotUi;     // full window incl. ImGui
  bool virtualPath = false;     // accepted no-op since M3.4: kgeo defaults
                                // to the virtual chain
  bool baseline = false;        // accepted no-op since M4.4: the CPU-cut
                                // render path retired (--virtual twin)
  bool checker = false;         // UV checkerboard debug view (decode path)
  bool noOcclusion = false;     // A/B: disable the occlusion cull chain
  bool noHw = false;            // accepted no-op since the 2026-09-19
                                // threshold sweep: pure SW IS the default
                                // (mode-1 SW beat every routing config on
                                // all five scene shapes)
  bool routeHw = false;         // A/B: route clusters to HW (hwMode 1)
  bool forceHw = false;         // A/B: all clusters HW (hwMode 2)
  bool noAsync = false;         // M4.4 A/B: chain onto the graphics queue
  bool cullLinear = false;      // M5.2 A/B: force the linear expand path
  bool cullPersistent = false;  // M5.3 A/B: the persistent driver
  bool rasterOpt = false;       // accepted alias: mode 1 IS the default since
                                // the 2026-09-19 A/B (+82% pure-SW, bit-exact)
  bool rasterBase = false;      // A/B: force the baseline walk (mode 0)
  bool rasterProbe = false;     // coverage histogram probe ([rasthist])
  bool hwPxStats = false;       // --hw-px-stats: count HW pixel writes
  bool noTaa = false;           // M9 A/B: disable temporal AA
  bool forceTaa = false;        // --taa: force ON over the parked default
  bool imposterDebug = false;   // --imposter-debug: imposters replace the
                                // extra-mesh instances (acceptance mode)
  bool imposterOff = false;     // --no-imposter: kill the routing (A/B)
  bool noShadows = false;       // M9.1 A/B: disable the shadow map
  std::string ground;           // M9.1: a ground-plane .kgeo whose
  std::vector<std::string> meshes;  // --mesh <kgeo>: extra assets,
                                    // instanced left-to-right
                                    // instances receive the shadows
  bool multi = false;               // S4.6 acceptance: near/mid/far instances
  bool dense = false;               // M3.4 acceptance: front-back occluders
  int grid = 0;                     // --grid N: N-instance occlusion carpet
                                    // (1..32; the VisBuffer payload's 5b)
  bool wall = false;                // --wall: big occluder + hidden fleet
  // --cam: yaw,pitch,dist[,target xyz] in degrees/meters; nullopt = the
  // controller's interactive defaults.
  bool hasCam = false;
  // --frame yaw,pitch[,scale]: auto-frame the loaded content's bounds
  // from this angle at scale x its radius (defaults 45, 6, 2.8). Yaw 180
  // looks at a glTF model's front (+Z there, -Y here).
  float frameYaw = 45.0f, framePitch = 6.0f, frameScale = 2.8f;
  float camYaw = 0, camPitch = 0, camDist = 0;
  float camTarget[3] = {0, 0, 0};
  int viewMode = -1;            // --view; -1 = leave the panel state at Final
  float exposure = -1.0f;       // --exposure: seeds the panel value
  float envIntensity = -1.0f;   // --env-intensity: seeds the panel value
  bool noBloom = false;         // --no-bloom
  bool noSky = false;           // --no-sky: clear color behind the model
  bool noEmissivePass = false;  // --no-emissive-pass-through
  float skyBlur = -1.0f;        // --sky-blur <mip>: seeds the panel value
  bool noAutoExposure = false;  // --no-auto-exposure (--exposure implies it)
  float exposureBias = -1e9f;   // --exposure-bias <EV>: seeds the panel
  std::string envPath;          // --env: an explicit Radiance HDRI
  float taaSharpen = -1.0f;     // --taa-sharpen: seeds the panel value
  float bloomStrength = -1.0f;  // --bloom-strength: seeds the panel value
  float emissiveStrength = -1.0f;  // --emissive-strength: ditto
  float hwThresholdPx = -1.0f;  // --hw-threshold: the SW/HW routing edge
                                // threshold (UE's r.Nanite.MinPixelsPerEdgeHW
                                // twin; <0 = keep the engine default)
  // M6.3: page-streaming budget in MB. 0 = unlimited (everything
  // streams in and stays resident - the v2-equivalent fast path);
  // N = the slot pool + residency cap for .kgeo v3 assets.
  int streamBudgetMb = 0;
  // 1.3d: a project's scene through the asset database. --scene is a
  // project-relative path (a .kscene, a .kprefab or a model); the
  // startup scene of project.kproject when omitted.
  std::string project;
  std::string scene;
  // Seconds between hot-reload polls (0 = every frame: scripted runs
  // reload a fixed number of frames after a change); <0 = no watching.
  float watchInterval = 0.25f;
  // --touch <frame> <src> <dst>: at that frame, rewrite <dst> with the
  // bytes of <src> (the hot-reload test's scripted edit; a fresh write,
  // so the file's mtime changes like an editor's save).
  int touchFrame = -1;
  std::string touchSrc, touchDst;
};

Args parseArgs(int argc, char** argv);

/// Locate engine/shaders: relative to the exe (build/<preset>/app/<cfg>),
/// falling back to the working directory.
std::string findShaderRoot();

/// Deterministic clear color: constant dark slate (exact k/255 values
/// keep the float32 -> UNORM round trip byte-exact across runs; the
/// last-frame log above is what screenshot verification asserts).
inline constexpr kenshin::f32 kClearColor[4] = {14.0f / 255.0f, 16.0f / 255.0f,
                                         22.0f / 255.0f, 1.0f};

/// Studio light from upper-left-front (TO light, world space) — scene
/// state, passed to the renderer as pure data each frame.
inline constexpr kenshin::f32 kStudioLight[3] = {0.35f, 0.8f, 0.45f};

// M8.1: the static point-light accents (deterministic - the render
// regression depends on frame-to-frame stability). Bracketing the
// --multi rig's mid instance (x=7, the regression camera's subject);
// the far instance (x=59) stays directional-only by design.
inline constexpr glm::vec3 kPointLights[2] = {{3.0f, 2.0f, 2.5f}, {6.5f, 1.5f, 2.5f}};
inline constexpr glm::vec3 kPointColors[2] = {{0.9f, 0.35f, 0.2f},
                                       {0.25f, 0.5f, 0.95f}};
inline constexpr kenshin::f32 kPointRange = 8.0f;

}  // namespace demo
