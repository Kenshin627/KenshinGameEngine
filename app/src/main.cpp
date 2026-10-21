// KENSHINDemo entry: parse args, build the demo scene, boot the engine
// (subsystem registry), drive the frame loop. The app talks to the
// Renderer / MeshPool / DebugOverlay services only — the render backend
// stays engine-internal.
// Usage:
//   KENSHINDemo                                  windowed loop (ESC quits)
//   KENSHINDemo --frames N                       auto-exit after N frames
//   KENSHINDemo --no-validation                 disable the validation layer (dev default: ON)
//   KENSHINDemo --size WxH                       client size (default 1600x900)
//   KENSHINDemo --screenshot path.bmp            capture the last rendered frame
//   KENSHINDemo --kgeo path.kgeo               virtual geometry chain +
//                                               decode (the default path
//                                               since M3.4; the VisBuffer
//                                               renders in-frame)
//   KENSHINDemo --kgeo path.kgeo --baseline    no-op (the CPU-cut
//                                               preview retired at M4.4;
//                                               parses for old scripts)
//   KENSHINDemo --kgeo path.kgeo --checker     UV checkerboard view
//                                               (--virtual still parses,
//                                               a no-op for old scripts)
//   KENSHINDemo --cam yaw,pitch,dist[,tx,ty,tz]    fixed camera pose in
//                                                degrees/meters (S4.6
//                                                acceptance screenshots)
//   KENSHINDemo --view N                         decode view mode 0..5
//                                                (0 final, 1 LOD, 2 error,
//                                                 3 triangles, 4 checker,
//                                                 5 clusters)
//   KENSHINDemo --kgeo X --virtual --multi       S4.6 acceptance scene:
//                                                one mesh at near/mid/far
//                                                instances along +x

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "DemoScene.h"
#include "KgeoPreview.h"
#include "stb/stb_image.h"  // env.hdr decode (impl: engine's stb_image_impl)

#include "kenshin/DebugOverlay.h"
#include "kenshin/Engine.h"
#include "kenshin/Version.h"
#include "kenshin/core/FrameStats.h"
#include "kenshin/core/Input.h"
#include "kenshin/core/Log.h"
#include "kenshin/core/Timer.h"
#include "kenshin/core/Window.h"
#include "kenshin/geometry/PagePacker.h"
#include "kenshin/manager/MeshPool.h"
#include "kenshin/manager/ShaderManager.h"
#include "kenshin/manager/TextureManager.h"
#include "kenshin/renderer/ClusterCullPass.h"
#include "kenshin/renderer/ImposterPass.h"
#include "kenshin/renderer/LightingPass.h"
#include "kenshin/renderer/MaterialDecodePass.h"
#include "kenshin/renderer/Renderer.h"
#include "kenshin/renderer/SoftRasterPass.h"
#include "kenshin/renderer/StreamingManager.h"
#include "kenshin/renderer/TaaPass.h"
#include "kenshin/renderer/TonemapPass.h"
#include "kenshin/renderer/VirtualGeometryScene.h"
#include "kenshin/renderer/VisBufferChain.h"
#include "kenshin/scene/Camera.h"
#include "kenshin/scene/CameraController.h"
#include "kenshin/util/BmpWriter.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")

#pragma warning(push, 0)
#include <imgui.h>
#pragma warning(pop)

namespace {
// Crash triage aid: log a symbolized stack on any unhandled SEH
// exception (this is what caught the Slang module-cache use-after-free).
LONG WINAPI crashLogFilter(EXCEPTION_POINTERS* ep) {
  HANDLE proc = GetCurrentProcess();
  SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
  SymInitialize(proc, nullptr, TRUE);
  void* frames[64]{};
  const USHORT n = CaptureStackBackTrace(0, 64, frames, nullptr);
  fprintf(stderr, "=== UNHANDLED EXCEPTION code=%08lx addr=%p\n",
          ep->ExceptionRecord->ExceptionCode,
          ep->ExceptionRecord->ExceptionAddress);
  for (USHORT i = 0; i < n; ++i) {
    char buf[sizeof(SYMBOL_INFO) + 256]{};
    auto* sym = reinterpret_cast<SYMBOL_INFO*>(buf);
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen = 255;
    DWORD64 disp = 0;
    if (SymFromAddr(proc, reinterpret_cast<DWORD64>(frames[i]), &disp, sym))
      fprintf(stderr, "  #%02u %s+%llu\n", i, sym->Name,
              static_cast<unsigned long long>(disp));
    else
      fprintf(stderr, "  #%02u %p\n", i, frames[i]);
  }
  return EXCEPTION_CONTINUE_SEARCH;
}
}  // namespace

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace {
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
  float camYaw = 0, camPitch = 0, camDist = 0;
  float camTarget[3] = {0, 0, 0};
  int viewMode = -1;            // --view; -1 = leave the panel state at Final
  float hwThresholdPx = -1.0f;  // --hw-threshold: the SW/HW routing edge
                                // threshold (UE's r.Nanite.MinPixelsPerEdgeHW
                                // twin; <0 = keep the engine default)
  // M6.3: page-streaming budget in MB. 0 = unlimited (everything
  // streams in and stays resident - the v2-equivalent fast path);
  // N = the slot pool + residency cap for .kgeo v3 assets.
  int streamBudgetMb = 0;
};

Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--no-validation") == 0)
      args.validation = false;
    else if (std::strcmp(argv[i], "--stream-budget") == 0 && i + 1 < argc)
      args.streamBudgetMb = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--no-hw") == 0)
      args.noHw = true;
    else if (std::strcmp(argv[i], "--route-hw") == 0)
      args.routeHw = true;
    else if (std::strcmp(argv[i], "--force-hw") == 0)
      args.forceHw = true;
    else if (std::strcmp(argv[i], "--hw-threshold") == 0 && i + 1 < argc)
      args.hwThresholdPx = static_cast<float>(std::atof(argv[++i]));
    else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc)
      args.frames = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc)
      sscanf_s(argv[++i], "%dx%d", &args.width, &args.height);
    else if (std::strcmp(argv[i], "--kgeo") == 0 && i + 1 < argc)
      args.kgeo = argv[++i];
    else if (std::strcmp(argv[i], "--lod-threshold") == 0 && i + 1 < argc)
      args.lodThreshold = static_cast<float>(std::atof(argv[++i]));
    else if (std::strcmp(argv[i], "--screenshot") == 0 && i + 1 < argc)
      args.screenshot = argv[++i];
    else if (std::strcmp(argv[i], "--screenshot-ui") == 0 && i + 1 < argc)
      args.screenshotUi = argv[++i];
    else if (std::strcmp(argv[i], "--virtual") == 0)
      args.virtualPath = true;
    else if (std::strcmp(argv[i], "--baseline") == 0)
      args.baseline = true;  // no-op: CPU-cut path retired at M4.4
    else if (std::strcmp(argv[i], "--checker") == 0)
      args.checker = true;
    else if (std::strcmp(argv[i], "--no-occ") == 0)
      args.noOcclusion = true;
    else if (std::strcmp(argv[i], "--no-async") == 0)
      args.noAsync = true;
    else if (std::strcmp(argv[i], "--cull-linear") == 0)
      args.cullLinear = true;
    else if (std::strcmp(argv[i], "--cull-persistent") == 0)
      args.cullPersistent = true;
    else if (std::strcmp(argv[i], "--raster-opt") == 0)
      args.rasterOpt = true;
    else if (std::strcmp(argv[i], "--raster-base") == 0)
      args.rasterBase = true;
    else if (std::strcmp(argv[i], "--raster-probe") == 0)
      args.rasterProbe = true;
    else if (std::strcmp(argv[i], "--no-shadows") == 0)
      args.noShadows = true;
    else if (std::strcmp(argv[i], "--no-taa") == 0)
      args.noTaa = true;
    else if (std::strcmp(argv[i], "--taa") == 0)
      args.forceTaa = true;
    else if (std::strcmp(argv[i], "--imposter-debug") == 0)
      args.imposterDebug = true;
    else if (std::strcmp(argv[i], "--no-imposter") == 0)
      args.imposterOff = true;
    else if (std::strcmp(argv[i], "--ground") == 0 && i + 1 < argc)
      args.ground = argv[++i];
    else if (std::strcmp(argv[i], "--mesh") == 0 && i + 1 < argc)
      args.meshes.push_back(argv[++i]);
    else if (std::strcmp(argv[i], "--view") == 0 && i + 1 < argc)
      args.viewMode = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--multi") == 0)
      args.multi = true;
    else if (std::strcmp(argv[i], "--dense") == 0)
      args.dense = true;
    else if (std::strcmp(argv[i], "--grid") == 0 && i + 1 < argc)
      args.grid = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--wall") == 0)
      args.wall = true;
    else if (std::strcmp(argv[i], "--cam") == 0 && i + 1 < argc) {
      const int n = sscanf_s(argv[++i], "%f,%f,%f,%f,%f,%f", &args.camYaw,
                             &args.camPitch, &args.camDist, &args.camTarget[0],
                             &args.camTarget[1], &args.camTarget[2]);
      args.hasCam = n >= 3;  // target triple optional
    }
  }
  if (args.width <= 0 || args.height <= 0) {
    args.width = 1600;
    args.height = 900;
  }
  return args;
}

/// Deterministic clear color: constant dark slate (exact k/255 values
/// keep the float32 -> UNORM round trip byte-exact across runs; the
/// last-frame log above is what screenshot verification asserts).
constexpr kenshin::f32 kClearColor[4] = {14.0f / 255.0f, 16.0f / 255.0f,
                                         22.0f / 255.0f, 1.0f};

/// Studio light from upper-left-front (TO light, world space) — scene
/// state, passed to the renderer as pure data each frame.
constexpr kenshin::f32 kStudioLight[3] = {0.35f, 0.8f, 0.45f};

/// M9.1 Stage C: which 512px tiles of a 2048^2 clipmap level the main
/// view's frustum touches. UE marks pages from the receiver mask
/// (GBuffer depth); this CPU v1 projects the frustum corners into the
/// level's light clip, clamps to the box, takes the convex hull, and
/// tests the 4x4 tile centers against it (inflated by half a tile so
/// partial overlaps count) - conservative, zero plumbing. The GPU
/// precise marking drops in at the same call site later.
kenshin::u32 shadowTileMask(const glm::mat4& lightVP,
                            const glm::mat4& invViewProj) {
  glm::vec2 pts[8];
  for (int i = 0; i < 8; ++i) {
    const glm::vec4 ndc((i & 1) ? 1.0f : -1.0f, (i & 2) ? 1.0f : -1.0f,
                        (i & 4) ? 1.0f : 0.0f, 1.0f);
    const glm::vec4 clip = lightVP * (invViewProj * ndc);
    pts[i] = glm::clamp(glm::vec2(clip.x, clip.y) * 0.5f + 0.5f,
                        glm::vec2(0.0f), glm::vec2(1.0f));
  }
  // Convex hull (monotone chain, CCW).
  std::sort(pts, pts + 8, [](const glm::vec2& a, const glm::vec2& b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
  });
  const auto cross = [](const glm::vec2& o, const glm::vec2& a,
                        const glm::vec2& b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  };
  glm::vec2 hull[8];
  int n = 0;
  for (int i = 0; i < 8; ++i) {
    while (n >= 2 && cross(hull[n - 2], hull[n - 1], pts[i]) <= 0.0f) --n;
    hull[n++] = pts[i];
  }
  const int lower = n + 1;
  for (int i = 6; i >= 0; --i) {
    while (n >= lower && cross(hull[n - 2], hull[n - 1], pts[i]) <= 0.0f) --n;
    hull[n++] = pts[i];
  }
  if (n > 0) --n;  // the last point repeats the first
  kenshin::u32 mask = 0;
  if (n < 3) {
    // Degenerate projection: fall back to the point bbox.
    glm::vec2 mn(1.0f), mx(0.0f);
    for (const auto& p : pts) {
      mn = glm::min(mn, p);
      mx = glm::max(mx, p);
    }
    if (mn.x > mx.x) return 0;
    for (kenshin::u32 ty = kenshin::u32(mn.y * 4.0f);
         ty <= kenshin::u32(mx.y * 4.0f) && ty < 4; ++ty)
      for (kenshin::u32 tx = kenshin::u32(mn.x * 4.0f);
           tx <= kenshin::u32(mx.x * 4.0f) && tx < 4; ++tx)
        mask |= 1u << (ty * 4u + tx);
    return mask;
  }
  // Tile centers vs the hull; the -0.18 margin (half a tile diagonal
  // in cross units) admits centers whose TILE still overlaps the hull.
  for (kenshin::u32 ty = 0; ty < 4; ++ty)
    for (kenshin::u32 tx = 0; tx < 4; ++tx) {
      const glm::vec2 c((tx + 0.5f) * 0.25f, (ty + 0.5f) * 0.25f);
      bool inside = true;
      for (int e = 0; e < n && inside; ++e)
        inside = cross(hull[e], hull[(e + 1) % n], c) >= -0.18f;
      if (inside) mask |= 1u << (ty * 4u + tx);
    }
  return mask;
}

/// M9 TAA jitter: Halton(2,3) - the standard low-discrepancy 8-phase
/// subpixel sequence (UE's r.PostProcessing... TAA jitter twin).
glm::vec2 halton23(int i) {
  auto radical = [](int a, int base) {
    float r = 0.0f, f = 1.0f;
    while (a > 0) {
      f /= float(base);
      r += f * float(a % base);
      a /= base;
    }
    return r;
  };
  return {radical(i + 1, 2), radical(i + 1, 3)};
}

// M8.1: the static point-light accents (deterministic - the render
// regression depends on frame-to-frame stability). Bracketing the
// --multi rig's mid instance (x=7, the regression camera's subject);
// the far instance (x=59) stays directional-only by design.
constexpr glm::vec3 kPointLights[2] = {{3.0f, 2.0f, 2.5f}, {6.5f, 1.5f, 2.5f}};
constexpr glm::vec3 kPointColors[2] = {{0.9f, 0.35f, 0.2f},
                                       {0.25f, 0.5f, 0.95f}};
constexpr kenshin::f32 kPointRange = 8.0f;

/// Locate engine/shaders: relative to the exe (build/<preset>/app/<cfg>),
/// falling back to the working directory.
std::string findShaderRoot() {
  wchar_t exePath[MAX_PATH]{};
  GetModuleFileNameW(nullptr, exePath, MAX_PATH);
  auto dir = std::filesystem::path(exePath).parent_path();
  const std::filesystem::path candidates[] = {
      dir / ".." / ".." / ".." / ".." / "engine" / "shaders",
      std::filesystem::path("engine/shaders"),
  };
  for (const auto& candidate : candidates) {
    std::error_code ec;
    if (std::filesystem::exists(candidate / "test_atomic.slang", ec))
      return std::filesystem::weakly_canonical(candidate, ec).string();
  }
  return "engine/shaders";
}

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
  kenshin::f32 ambient = 0.15f;
  kenshin::f32 exposure = 1.5f;
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

}  // namespace

int main(int argc, char** argv) {
  SetUnhandledExceptionFilter(crashLogFilter);
  const Args args = parseArgs(argc, argv);

  kenshin::log::init();
  KENSHIN_INFO("{}", kenshin::getBanner());

  // Engine facade: global subsystem registry (type-indexed get<T>()).
  kenshin::Engine& engine = kenshin::Engine::get();

  // Per-scene state stays local (cameras are view state, not services).
  kenshin::scene::Camera camera;
  kenshin::scene::CameraController controller;

  auto& window = engine.registerSubsystem<kenshin::core::Window>();
  if (!window.create({"KENSHIN Demo", static_cast<kenshin::u32>(args.width),
                      static_cast<kenshin::u32>(args.height)})) {
    KENSHIN_ERROR("window create failed");
    kenshin::log::shutdown();
    return 1;
  }
  window.onResize = [](kenshin::u32 w, kenshin::u32 h) {
    KENSHIN_INFO("resize {}x{}", w, h);
  };
  window.onClose = []() { KENSHIN_INFO("closing (user)"); };

  // Shader service must know the shader root before the renderer's
  // pipeline compiles baseline.slang.
  auto& shaders = engine.registerSubsystem<kenshin::manager::ShaderManager>();
  shaders.init(findShaderRoot());

  // Renderer subsystem: creates + registers the backend (engine-internal),
  // enforces the int64-atomics gate, compiles and stages the baseline
  // pipeline. MeshPool: GPU mesh lifetime. DebugOverlay: debug UI
  // (resolves backend + window from the registry when it registers).
  auto& renderer = engine.registerSubsystem<kenshin::renderer::Renderer>(
      kenshin::renderer::Renderer::Config{args.validation});
  auto& meshPool = engine.registerSubsystem<kenshin::manager::MeshPool>();
  // M8.2: albedo textures (LIFO shutdown destroys images before the
  // backend tears down).
  auto& textureManager =
      engine.registerSubsystem<kenshin::manager::TextureManager>();

  auto& overlay = engine.registerSubsystem<kenshin::DebugOverlay>();
  // First-run docking: if no imgui.ini sits next to the EXE (the path
  // ImGui actually reads - a CWD-relative check sees stray files and
  // skips the build), the overlay builds a default layout; a saved
  // arrangement always wins.
  {
    char exePath[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, exePath, MAX_PATH);
    const std::filesystem::path ini =
        std::filesystem::path(exePath).parent_path() / "imgui.ini";
    if (!std::filesystem::exists(ini)) overlay.markFirstRun();
  }

  // ---- demo scene (S2.2): one instance of every shape on the X axis;
  // --mesh <name> picks one, --kgeo switches to the LOD preview instead.
  auto scene = args.kgeo.empty() ? demo::buildScene(args.mesh)
                                 : std::vector<demo::SceneShape>{};
  demo::uploadScene(scene, meshPool);

  // ---- S4.5 virtual path state (one-shot chain + decode) ---------------
  // Only meaningful with --kgeo + --virtual; declared here so the fill
  // inside the kgeo block below reaches the frame loop.
  std::vector<kenshin::renderer::CullGpuInstance> chainInstances;
  std::vector<kenshin::renderer::GpuMeshRangePair> chainMeshRanges;
  kenshin::renderer::MaterialDecodePass decodePass;
  kenshin::renderer::LightingPass lightingPass;  // M8.1 deferred pair
  bool lightingReady = false;
  kenshin::renderer::TonemapPass tonemapPass;    // M8.3 HDR chain tail
  kenshin::renderer::TaaPass taaPass;            // M9 temporal resolve
  kenshin::renderer::ImposterPass imposterPass;  // M9.2 tiny instances
  bool imposterReady = false;
  std::vector<kenshin::u32> meshAtlasBase;    // per meshId -> atlas word
  std::vector<kenshin::u32> meshRootCluster;  // per meshId -> cluster
  // Staged at setup (the previews die with their scope): per mesh the
  // atlas planes + the single-cluster cut, empty = no imposter.
  std::vector<std::vector<kenshin::u32>> stagedDepth, stagedTri;
  std::vector<glm::vec4> realBounds;  // M9.2: instance bounds pre-parking
  bool taaReady = false;
  glm::mat4 taaPrevViewProj{1.0f};  // UNJITTERED previous frame VP
  bool taaPrevValid = false;
  int taaWarmup = 0;  // frames since (re)validation
  bool tonemapReady = false;
  kenshin::renderer::DecodeParams decodeParams;
  // M6.3: page-streaming driver (v3 assets; null otherwise).
  std::unique_ptr<kenshin::renderer::StreamingManager> streamer;
  // M8.2: the albedo texture-array handle (kInvalidTextureHandle =
  // no textures loaded; the backend binds its 1x1 dummy).
  kenshin::u32 albedoTexture =
      kenshin::backend::IRenderBackend::kInvalidTextureHandle;
  VgDebugState vgDebug;
  if (args.imposterDebug) vgDebug.imposterDebug = true;
  if (args.viewMode >= 0 && args.viewMode <= 6)
    vgDebug.viewMode = args.viewMode;
  else if (args.checker)
    vgDebug.viewMode = 4;  // checker = DecodeView::Checker

  // ---- .kgeo data source (S3.4; CPU-cut render half retired M4.4) ---
  demo::KgeoPreview kgeoPreview;
  bool kgeoMode = false;
  if (!args.kgeo.empty()) {
    std::string kgeoLog;
    if (demo::loadKgeo(args.kgeo, kgeoPreview, kgeoLog)) {
      kgeoMode = true;
      vgDebug.pixelThreshold = args.lodThreshold;
      if (args.hwThresholdPx > 0.0f) vgDebug.hwThresholdPx = args.hwThresholdPx;

      // S4.1: the dag also goes to the GPU-resident virtual geometry
      // pools (cluster/group tables + shared vertex/index pools), CRC
      // round-trip verified at upload. The S4.2+ culling passes consume
      // these; rendering still runs the CPU-cut path until S4.5.
      auto& vgScene =
          engine.registerSubsystem<kenshin::renderer::VirtualGeometryScene>();
      // Multi-material assets carry their own base-color factors in
      // the material table - the studio kd (0.75) would wash them.
      // Single-material assets keep the studio kd bit-identical (their
      // material factor is 1, so the product is unchanged).
      if (kgeoPreview.dag.materials.size() > 1)
        decodeParams.kd = glm::vec3(1.0f);
      // M6.3: v3 assets (tables-only dag) take the paged branch - the
      // vertex pools are never uploaded, the pages stream via the
      // StreamingManager. slotCap = the budget's structural form.
      const bool pagedAsset = kgeoPreview.dag.paged;
      // M8.2: resolve the dag's material albedo references (bare file
      // names against the .kgeo's directory - the mtllib rule) into
      // texture-array LAYERS before the scene bakes them into its GPU
      // table. A failed load keeps texLayer = -1 (untextured kd).
      if (pagedAsset && !kgeoPreview.dag.materials.empty()) {
        const auto kgeoDir = std::filesystem::path(args.kgeo).parent_path();
        for (auto& mat : kgeoPreview.dag.materials) {
          for (int slot = 0; slot < 2; ++slot) {
            std::string& path = slot == 0 ? mat.texPath : mat.mrPath;
            kenshin::f32& layerF = slot == 0 ? mat.texLayer : mat.mrTexLayer;
            if (path.empty()) continue;
            const auto resolved =
                path.find(':') == std::string::npos &&
                        !std::filesystem::exists(kgeoDir / path)
                    ? std::filesystem::path(path)
                    : (kgeoDir / path);
            const kenshin::u32 layer = textureManager.load(resolved.string());
            if (layer != kenshin::manager::TextureManager::kInvalidLayer)
              layerF = static_cast<kenshin::f32>(layer);
          }
        }
        albedoTexture = textureManager.textureArray();
      }
      const kenshin::u32 vgMeshId = pagedAsset
                                        ? vgScene.addMeshPaged(kgeoPreview.dag)
                                        : vgScene.addMesh(kgeoPreview.dag);
      // v5 gate: pre-v5 paged assets are rejected (their empty v5 tables
      // would silently drop every dispatch - black viewport, no warning).
      if (vgMeshId == 0xFFFFFFFFu) {
        KENSHIN_ERROR(
            "[main] asset rejected: rebuild {} with the current "
            "kgeo_builder",
            args.kgeo);
        return 1;
      }
      // M9.1: an optional ground-plane asset (a second paged mesh;
      // its clusters stamp material id 0 which indexes the FIRST
      // mesh's material - the plane wears the knot's checker here,
      // fine for the shadow demo; the rebasing is the documented v1
      // multi-mesh limitation).
      kenshin::u32 groundMeshId = 0xFFFFFFFFu;
      bool groundPaged = false;
      if (!args.ground.empty()) {
        demo::KgeoPreview groundPreview;
        std::string groundLog;
        if (demo::loadKgeo(args.ground, groundPreview, groundLog)) {
          if (groundPreview.dag.paged) {
            groundMeshId = vgScene.addMeshPaged(groundPreview.dag);
            groundPaged = true;
          } else
            groundMeshId = vgScene.addMesh(groundPreview.dag);
          if (groundMeshId == 0xFFFFFFFFu) {
            KENSHIN_ERROR("[main] ground asset rejected: rebuild {}",
                          args.ground);
            return 1;
          }
          KENSHIN_INFO("[main] ground mesh loaded: {} clusters",
                       groundPreview.dag.clusters.size());
        } else {
          KENSHIN_ERROR("[main] ground load failed: {}", groundLog);
          return 1;
        }
      }
      // --mesh: extra assets (the real-content gallery). Materials
      // resolve like the primary's; the scene's per-mesh material base
      // handles the table rebasing (M8.2 v2).
      std::vector<kenshin::u32> extraMeshIds;
      std::vector<std::string> extraTexPaths;
      std::vector<demo::KgeoPreview> extraPreviews;
      for (const auto& path : args.meshes) {
        demo::KgeoPreview mp;
        std::string mlog;
        if (!demo::loadKgeo(path, mp, mlog)) {
          KENSHIN_ERROR("[main] --mesh load failed: {} ({})", path, mlog);
          return 1;
        }
        if (!mp.dag.paged) {
          KENSHIN_ERROR("[main] --mesh {} is not a paged asset; rebuild it",
                        path);
          return 1;
        }
        if (!mp.dag.materials.empty()) {
          const auto dir = std::filesystem::path(path).parent_path();
          for (auto& mat : mp.dag.materials) {
            if (mat.texPath.empty()) continue;
            const auto resolved =
                mat.texPath.find(':') == std::string::npos &&
                        !std::filesystem::exists(dir / mat.texPath)
                    ? std::filesystem::path(mat.texPath)
                    : (dir / mat.texPath);
            const kenshin::u32 layer = textureManager.load(resolved.string());
            if (layer != kenshin::manager::TextureManager::kInvalidLayer)
              mat.texLayer = static_cast<kenshin::f32>(layer);
          }
          albedoTexture = textureManager.textureArray();
        }
        const kenshin::u32 id = vgScene.addMeshPaged(mp.dag);
        if (id == 0xFFFFFFFFu) {
          KENSHIN_ERROR(
              "[main] --mesh {} rejected (rebuild with the "
              "current kgeo_builder)",
              path);
          return 1;
        }
        extraMeshIds.push_back(id);
        extraTexPaths.push_back(path);
        extraPreviews.push_back(std::move(mp));
        KENSHIN_INFO("[main] extra mesh {}: {} ({} clusters)",
                     extraMeshIds.size() - 1, path, mp.dag.clusters.size());
      }
      stagedDepth.assign(vgScene.meshes().size(), {});
      stagedTri.assign(vgScene.meshes().size(), {});
      {
        const auto stage = [&](kenshin::u32 id, const demo::KgeoPreview& pv) {
          if (pv.dag.imposterDepth.empty()) return;
          if (pv.dag.imposterCluster == 0xFFFFFFFFu) return;
          stagedDepth[id] = pv.dag.imposterDepth;
          stagedTri[id] = pv.dag.imposterTri;
          meshRootCluster[id] = pv.dag.imposterCluster;
        };
        stagedDepth.resize(1 + extraMeshIds.size());
        stagedTri.resize(1 + extraMeshIds.size());
        meshRootCluster.resize(1 + extraMeshIds.size(), 0xFFFFFFFFu);
        stage(0, kgeoPreview);
        for (kenshin::u32 e = 0; e < extraMeshIds.size(); ++e)
          stage(extraMeshIds[e], extraPreviews[e]);
      }
      const kenshin::u32 slotCap =
          pagedAsset && args.streamBudgetMb > 0
              ? kenshin::u32((kenshin::u64(args.streamBudgetMb) * 1024 * 1024) /
                             kenshin::geometry::kPageSlotSize)
              : 0xFFFFFFFFu;
      vgScene.flushPools(slotCap);
      vgScene.setInstances(
          {kenshin::renderer::VgInstance{glm::mat4(1.0f), vgMeshId}});
      // M6.3: the streaming driver (v3 only). Declared above the kgeo
      // block so the frame loop and shutdown see it; its shutdown runs
      // explicitly before the engine tears the scene/backend down.
      if (pagedAsset) {
        streamer = std::make_unique<kenshin::renderer::StreamingManager>();
        if (!streamer->init(
                engine.get<kenshin::backend::IRenderBackend>(), vgScene,
                args.streamBudgetMb > 0
                    ? kenshin::usize(args.streamBudgetMb) * 1024 * 1024
                    : 0)) {
          KENSHIN_ERROR(
              "[stream] manager init failed - falling back is "
              "impossible for a v3 asset, exiting");
          return 1;
        }
        streamer->addAsset(vgMeshId, args.kgeo);
        // The ground mesh's pages stream from ITS file too - without
        // the asset entry the streamer marks its pages wanted but can
        // never read them (clusters skip-resident forever, the M9.1
        // ground-invisible bug).
        if (groundMeshId != 0xFFFFFFFFu && groundPaged)
          streamer->addAsset(groundMeshId, args.ground);
        for (kenshin::u32 e = 0; e < extraMeshIds.size(); ++e)
          streamer->addAsset(extraMeshIds[e], extraTexPaths[e]);
        // First frame coarse: the leading (coarsest) pages resident
        // before the loop starts - no black flash while the async
        // thread spins up.
        if (!streamer->warmup())
          KENSHIN_WARN(
              "[stream] warmup incomplete - early frames may be "
              "sparse");
      }
      // M4.4 CPU-cut retirement: kgeo IS the virtual chain; the
      // chain instances are set up unconditionally (--baseline is a
      // logged no-op for old scripts).
      {
        const kenshin::renderer::VgMeshRange& mesh = vgScene.meshes()[vgMeshId];
        // meshId + mesh bounds must come from the INSTANCE'S mesh - the
        // ground uses groundMeshId; a hardcoded vgMeshId renders a
        // second copy of the first mesh instead (the M9.1 ground bug).
        const auto makeInstFor = [&](kenshin::u32 id, const glm::mat4& model,
                                     kenshin::u32 casts = 1u) {
          // casts: 0 receiver-only, 1 caster, 2 imposterable (M9.2)
          const kenshin::renderer::VgMeshRange& m = vgScene.meshes()[id];
          kenshin::renderer::CullGpuInstance inst;
          inst.model = model;
          inst.meshId = id;
          inst.pad[1] = casts;  // 72: castShadows (receiver-only = 0)
          // WORLD bounds sphere (the cull input): center transformed,
          // radius scaled by the max column length - the same derivation
          // VirtualGeometryScene::setInstances applies. Without this the
          // bounds stay (0,0,0,-1) and the frustum test degenerates to
          // "does the view contain the world origin" - half the camera
          // orientations wrongly culled everything (the S4.6 "clipped at
          // certain angles" report).
          const kenshin::f32 scale =
              std::max({glm::length(model[0]), glm::length(model[1]),
                        glm::length(model[2])});
          inst.bounds = glm::vec4(
              glm::vec3(model * glm::vec4(m.boundsSphere.x, m.boundsSphere.y,
                                          m.boundsSphere.z, 1.0f)),
              m.boundsSphere.w * scale);
          return inst;
        };
        const auto makeInst = [&](const glm::mat4& model) {
          return makeInstFor(vgMeshId, model);
        };
        if (args.wall) {
          // Big-occluder scene (the UE5-faithful occlusion demo): one
          // 6x-scaled sphere at the origin as a wall, 24 unit spheres
          // in a 6-wide x 4-deep block BEHIND it (positive x; the
          // yaw-270 demo camera looks along +x from -x). The wall
          // covers a large screen fraction at range - the condition
          // under which the conservative HZB band still culls (the
          // D4 edge conservatism is official Nanite behavior; the
          // win comes from big occluders, not pyramid tricks).
          std::vector<kenshin::renderer::VgInstance> vgInsts;
          {
            const glm::mat4 m =
                glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, 0)) *
                glm::scale(glm::mat4(1.0f), glm::vec3(6.0f));
            chainInstances.push_back(makeInst(m));
            vgInsts.push_back({m, vgMeshId});
          }
          for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 6; ++col) {
              const glm::vec3 o(3.0f + 2.25f * row, 0.0f, (col - 2.5f) * 2.25f);
              const glm::mat4 m = glm::translate(glm::mat4(1.0f), o);
              chainInstances.push_back(makeInst(m));
              vgInsts.push_back({m, vgMeshId});
            }
          vgScene.setInstances(vgInsts);
        } else if (args.grid > 0) {
          // Occlusion stress carpet: a near-square XZ grid of the mesh
          // - from any low-angle camera, front rows hide back rows and
          // the cull has real work to save. 256 max (the pair-list
          // payload's 24b index and the frame's pair capacity bound;
          // the VisBuffer payload itself is instance-count-free since
          // the wide-instance pair indirection).
          const int n = std::clamp(args.grid, 1, 256);
          // DEPTH-major: 8-deep lines along +x (the view axis of the
          // demo camera), the rest across z. From head-on each line is
          // a dense occluder stack - the configuration the cull pays
          // in (M3.4: scattered carpets lose to background pollution).
          const int depth = std::max(1, std::min(8, (n + 3) / 4));
          const kenshin::f32 step = 2.25f;  // dense-like: neighbors'
                                            // silhouettes overlap
          std::vector<kenshin::renderer::VgInstance> vgInsts;
          for (int i = 0; i < n; ++i) {
            const glm::vec3 o((i % depth) * step, 0.0f, (i / depth) * step);
            const glm::mat4 m = glm::translate(glm::mat4(1.0f), o);
            chainInstances.push_back(makeInst(m));
            vgInsts.push_back({m, vgMeshId});
          }
          vgScene.setInstances(vgInsts);
        } else if (args.multi) {
          // S4.6 acceptance scene: the same mesh at near/mid/far along
          // +x - one threshold, visibly different LOD levels per
          // instance. --dense (M3.4) pulls them close instead: a
          // front-back occluder stack for the occlusion acceptance.
          const std::vector<glm::vec3> offsets =
              args.dense ? std::vector<glm::vec3>{{0.0f, 0.0f, 0.0f},
                                                  {2.2f, 0.0f, 0.0f},
                                                  {4.4f, 0.0f, 0.0f}}
                         : std::vector<glm::vec3>{{0.0f, 0.0f, 0.0f},
                                                  {7.0f, 0.0f, 0.0f},
                                                  {59.0f, 0.0f, 0.0f}};
          std::vector<kenshin::renderer::VgInstance> vgInsts;
          for (const glm::vec3& o : offsets) {
            const glm::mat4 m = glm::translate(glm::mat4(1.0f), o);
            chainInstances.push_back(makeInst(m));
            vgInsts.push_back({m, vgMeshId});
          }
          if (groundMeshId != 0xFFFFFFFFu) {
            // Centered between the default camera target (origin) and
            // the mid instance - the 40m span covers everything the
            // default and acceptance views look at. Z-up drop like the
            // default branch.
            const glm::mat4 gm =
                glm::translate(glm::mat4(1.0f), glm::vec3(5.0f, 0.0f, -1.5f));
            chainInstances.push_back(makeInstFor(groundMeshId, gm));
            vgInsts.push_back({gm, groundMeshId});
          }
          vgScene.setInstances(vgInsts);
        } else {
          realBounds.clear();
          chainInstances = {makeInst(glm::mat4(1.0f))};
          realBounds.push_back(chainInstances.back().bounds);
          std::vector<kenshin::renderer::VgInstance> vgInsts = {
              kenshin::renderer::VgInstance{glm::mat4(1.0f), vgMeshId}};
          if (groundMeshId != 0xFFFFFFFFu) {
            // Z-up: ground_z.kgeo is a z=0 plane; drop it below the
            // knot (radius ~1.5) so the knot sits on it.
            const glm::mat4 gm =
                glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -1.5f));
            chainInstances.push_back(makeInstFor(groundMeshId, gm, 0u));
            vgInsts.push_back({gm, groundMeshId});
          }
          // --mesh gallery: spaced along -x (the default camera's view
          // axis from its +x+y orbit), sitting on the same ground.
          for (kenshin::u32 e = 0; e < extraMeshIds.size(); ++e) {
            const glm::vec3 o(-2.5f - 3.0f * float(e), 0.0f, 0.0f);
            const glm::mat4 m = glm::translate(glm::mat4(1.0f), o);
            const bool hasAtlas = e < extraPreviews.size() &&
                                  !extraPreviews[e].dag.imposterDepth.empty();
            auto inst = makeInstFor(extraMeshIds[e], m, hasAtlas ? 2u : 1u);
            realBounds.push_back(inst.bounds);
            chainInstances.push_back(std::move(inst));
            vgInsts.push_back({m, extraMeshIds[e]});
          }
          vgScene.setInstances(vgInsts);
        }
        for (kenshin::u32 i = 0; i < chainInstances.size(); ++i)
          chainMeshRanges.push_back({mesh.clusterBegin, mesh.clusterEnd});
      }
      overlay.setExtraPanel([&kgeoPreview, &vgDebug, &streamer] {
        demo::drawLodPanel(kgeoPreview);
        {
          ImGui::Begin("Virtual Geometry");
          const char* names[] = {"final",     "LOD level", "error heat",
                                 "triangles", "checker",   "clusters",
                                 "route path"};
          ImGui::SetNextItemWidth(150.0f);
          ImGui::Combo("view", &vgDebug.viewMode, names, 7);
          ImGui::Separator();
          ImGui::Text(
              "clusters %u  tris %u  writes %u  cover %.1f%%\n"
              "occ-culled %u  revived %u  hw %u",
              vgDebug.visibleClusters, vgDebug.cutTris, vgDebug.pxWrites,
              vgDebug.coveragePct, vgDebug.occCulled, vgDebug.revived,
              vgDebug.hwClusters);
          ImGui::Checkbox("occlusion", &vgDebug.occlusion);
          ImGui::Checkbox("BVH8 cull", &vgDebug.hierCull);
          ImGui::SliderFloat("pixel error", &vgDebug.pixelThreshold, 0.05f,
                             32.0f, "%.2f px", ImGuiSliderFlags_Logarithmic);
          ImGui::SliderFloat("HW route >=", &vgDebug.hwThresholdPx, 1.0f,
                             128.0f, "%.0f px", ImGuiSliderFlags_Logarithmic);
          if (vgDebug.viewMode == 0) {
            // M8.1 lighting rig (only meaningful in the deferred final
            // view). Static table - deterministic for the regression.
            ImGui::Separator();
            ImGui::Text("lighting (deferred)");
            ImGui::SliderFloat3("dir light", &vgDebug.lightDir.x, -1.0f, 1.0f,
                                "%.2f");
            ImGui::SliderFloat("intensity", &vgDebug.lightIntensity, 0.0f,
                               10.0f, "%.2f");
            ImGui::SliderFloat("points", &vgDebug.pointIntensity, 0.0f, 30.0f,
                               "%.2f");
            ImGui::Checkbox("shadows", &vgDebug.shadows);
            ImGui::SameLine();
            ImGui::Checkbox("TAA", &vgDebug.taa);
            ImGui::Checkbox("imposter", &vgDebug.imposterDebug);
            if (vgDebug.taa)
              ImGui::SliderFloat("sharpen", &vgDebug.taaSharpen, 0.0f, 1.0f);
            ImGui::SliderInt("shadow levels", &vgDebug.shadowLevels, 1, 3);
            ImGui::Combo("tonemap", &vgDebug.tonemapMode, "Reinhard\0ACES\0");
            ImGui::ColorEdit3("sky", &vgDebug.skyColor.x);
            ImGui::ColorEdit3("ground", &vgDebug.groundColor.x);
            ImGui::SliderFloat("ambient", &vgDebug.ambient, 0.0f, 0.5f, "%.3f");
            ImGui::SliderFloat("exposure", &vgDebug.exposure, 0.1f, 4.0f,
                               "%.2f");
          }
          ImGui::PlotLines("##clusters", vgDebug.clusterHist.data(),
                           int(vgDebug.clusterHist.size()), 0, nullptr, 0.0f,
                           1200.0f, ImVec2(240, 40));
          ImGui::PlotLines("##tris", vgDebug.triHist.data(),
                           int(vgDebug.triHist.size()), 0, nullptr, 0.0f,
                           200000.0f, ImVec2(240, 40));
          ImGui::PlotLines("##coverage", vgDebug.coverHist.data(),
                           int(vgDebug.coverHist.size()), 0, nullptr, 0.0f,
                           100.0f, ImVec2(240, 40));
          ImGui::PlotLines("##occ", vgDebug.occHist.data(),
                           int(vgDebug.occHist.size()), 0, nullptr, 0.0f,
                           1500.0f, ImVec2(240, 40));
          ImGui::PlotLines("##revive", vgDebug.reviveHist.data(),
                           int(vgDebug.reviveHist.size()), 0, nullptr, 0.0f,
                           1500.0f, ImVec2(240, 40));
          ImGui::PlotLines("##hw", vgDebug.hwHist.data(),
                           int(vgDebug.hwHist.size()), 0, nullptr, 0.0f,
                           1500.0f, ImVec2(240, 40));
          if (ImGui::CollapsingHeader("Traversal (M5)")) {
            ImGui::Text("visited %u  pruned %u", vgDebug.visitedNodes,
                        vgDebug.prunedNodes);
            ImGui::PlotLines("##visited", vgDebug.visitedHist.data(),
                             int(vgDebug.visitedHist.size()), 0, nullptr, 0.0f,
                             20000.0f, ImVec2(240, 40));
            ImGui::PlotLines("##pruned", vgDebug.prunedHist.data(),
                             int(vgDebug.prunedHist.size()), 0, nullptr, 0.0f,
                             20000.0f, ImVec2(240, 40));
          }
          if (streamer && ImGui::CollapsingHeader("Streaming (M6)")) {
            const double mb = 1024.0 * 1024.0;
            ImGui::Text(
                "resident %.1f MB (%u pages)  pending skips %u\n"
                "missed pages %u  io queue %u",
                double(streamer->residentBytes()) / mb,
                streamer->residentPages(), vgDebug.pendingSkips,
                vgDebug.missPages, streamer->pendingRequests());
            ImGui::PlotLines("##pending", vgDebug.pendingHist.data(),
                             int(vgDebug.pendingHist.size()), 0, nullptr, 0.0f,
                             2000.0f, ImVec2(240, 40));
            ImGui::PlotLines("##miss", vgDebug.missHist.data(),
                             int(vgDebug.missHist.size()), 0, nullptr, 0.0f,
                             500.0f, ImVec2(240, 40));
          }
          ImGui::End();
        }
      });
      KENSHIN_INFO("[kgeo] loaded: {} clusters / {} groups / {} levels",
                   kgeoPreview.dag.clusters.size(),
                   kgeoPreview.dag.groups.size(), [&] {
                     kenshin::i32 m = 0;
                     for (auto& g : kgeoPreview.dag.groups)
                       m = std::max(m, g.lodLevel);
                     return m + 1;
                   }());
      {
        const auto* hn =
            kgeoPreview.dag.hierarchyRoot <
                    kgeoPreview.dag.hierarchyNodes.size()
                ? &kgeoPreview.dag.hierarchyNodes[kgeoPreview.dag.hierarchyRoot]
                : nullptr;
        if (hn)
          KENSHIN_INFO(
              "[kgeo] hierarchy {} nodes, root {} aabb "
              "min({:.2f},{:.2f},{:.2f}) max({:.2f},{:.2f},{:.2f}) err {:.4f}",
              kgeoPreview.dag.hierarchyNodes.size(),
              kgeoPreview.dag.hierarchyRoot, hn->aabbMin[0], hn->aabbMin[1],
              hn->aabbMin[2], hn->aabbMax[0], hn->aabbMax[1], hn->aabbMax[2],
              hn->maxSimplifyError);
      }
    } else
      KENSHIN_WARN("[kgeo] load failed: {} (continuing with --mesh)", kgeoLog);
  }

  // ---- virtual path plumbing (S4.5) --------------------------------------
  // The S4.x chain/decode APIs are engine-side renderer code that speaks
  // the backend interface; resolved from the registry the same way the
  // scene's own onInit does.
  const bool virtualPath = kgeoMode;  // the only kgeo path since the
                                      // M4.4 CPU-cut retirement
  kenshin::renderer::VirtualGeometryScene* vgScene = nullptr;
  kenshin::backend::IRenderBackend* vgBackend = nullptr;
  kenshin::u32 envTexture =
      kenshin::backend::IRenderBackend::kInvalidTextureHandle;
  float envMaxLod = 1.0f;
  if (virtualPath) {
    vgScene = &engine.get<kenshin::renderer::VirtualGeometryScene>();
    vgBackend = &engine.get<kenshin::backend::IRenderBackend>();

    // IBL environment: an env.hdr next to the loaded asset wins (a
    // real Radiance HDRI - rich structured reflections); the
    // procedural studio probe is the fallback.
  {
    const std::filesystem::path envPath =
        std::filesystem::path(args.kgeo).parent_path() / "env.hdr";
    if (std::filesystem::exists(envPath)) {
      int ew = 0, eh = 0, ec = 0;
      float* data = stbi_loadf(envPath.string().c_str(), &ew, &eh, &ec, 4);
      if (data && ew > 0 && eh > 0) {
        std::vector<kenshin::u16> half(kenshin::usize(ew) * eh * 4);
        for (kenshin::usize i = 0; i < half.size(); ++i) {
          const kenshin::u32 b =
              *reinterpret_cast<const kenshin::u32*>(&data[i]);
          const kenshin::u32 sign = (b >> 16) & 0x8000u;
          const kenshin::u32 ex = (b >> 23) & 0xFFu;
          const kenshin::u32 man = b & 0x7FFFFFu;
          kenshin::u16 hf;
          if (ex >= 255)
            hf = kenshin::u16(sign | 0x7C00u | (man ? 0x200u : 0));
          else if (ex < 98)
            hf = 0;
          else {
            const kenshin::u32 e2 = ex - 127 + 15;
            hf = e2 >= 31 ? kenshin::u16(sign | 0x7C00u)
                          : kenshin::u16(sign | (e2 << 10) | (man >> 13));
          }
          half[i] = hf;
        }
        stbi_image_free(data);
        if (vgBackend) {
          kenshin::backend::IRenderBackend::TextureArrayDesc ed;
          ed.width = kenshin::u32(ew);
          ed.height = kenshin::u32(eh);
          ed.layers = 1;
          ed.hdr = true;
          ed.rgba = reinterpret_cast<const kenshin::u8*>(half.data());
          ed.bytes = half.size() * sizeof(kenshin::u16);
          ed.debugName = "ibl-env-hdr";
          envTexture = vgBackend->createTextureArray(ed);
          envMaxLod = std::floor(std::log2(float(ew))) + 1.0f - 1.0f;
          KENSHIN_INFO("[ibl] loaded {} ({}x{}, {} mips) -> handle {}",
                       envPath.string(), ew, eh, int(envMaxLod) + 1,
                       envTexture);
        }
      } else if (data == nullptr) {
        KENSHIN_WARN("[ibl] {} failed to decode (stbi: {})", envPath.string(),
                     stbi_failure_reason() ? stbi_failure_reason() : "?");
      }
    }
  }
  // IBL: a procedural HDR studio environment, equirectangular Z-up
    // (u = atan2(y,x)/2pi + 0.5, v = acos(z)/pi - the lighting shader's
    // sampling convention): a dim sky/ground gradient plus a bright
    // softbox centered on the key-light direction (reflections stay
    // consistent with the direct light + shadows). The GPU mip chain is
    // the prefiltered radiance ladder; roughness -> LOD at sample time.
    // env.hdr in effect? skip the procedural probe.
    if (envTexture !=
        kenshin::backend::IRenderBackend::kInvalidTextureHandle) {
      KENSHIN_INFO("[ibl] env.hdr in effect (probe skipped)");
    } else {
      constexpr int kW = 512, kH = 256;
      std::vector<float> probe(kenshin::usize(kW) * kH * 4);
      const glm::vec3 kd = glm::normalize(glm::vec3(0.35f, 0.8f, 0.45f));
      for (int y = 0; y < kH; ++y) {
        const float v = (y + 0.5f) / kH;
        const float theta = v * glm::pi<float>();
        const float z = std::cos(theta), lat = std::sin(theta);
        for (int x = 0; x < kW; ++x) {
          const float u = (x + 0.5f) / kW;
          const float a = (u - 0.5f) * 2.0f * glm::pi<float>();
          const glm::vec3 dir(std::cos(a) * lat, std::sin(a) * lat, z);
          const float t = z * 0.5f + 0.5f;
          // A real studio reads BRIGHT to a dark-f0 metal: broad
          // ceiling/key radiance, not a pinpoint on near-black sky
          // (f0 0.05-0.15 x 0.05 = black surface with specks).
          // True HDR magnitudes: a dark-f0 metal (linear ~0.04 after
          // the sRGB decode) only reads as metal against studio-HDR
          // radiance - ceiling ~3, softbox ~8. The ground's diffuse
          // never sees the env (hemisphere constants), so this cannot
          // blow it.
          // 6.5x: the dark camo's linear f0 (~0.04 after the sRGB
          // decode) needs studio-HDI radiance for the metal to read.
          const float es = getenv("KENSHIN_ENV_SCALE")
                               ? atof(getenv("KENSHIN_ENV_SCALE"))
                               : 6.5f;
          glm::vec3 c =
              glm::mix(glm::vec3(0.35f, 0.34f, 0.32f) * es,
                       glm::vec3(1.6f, 1.65f, 1.75f) * es, t * t);
          const float dKey = std::max(0.0f, glm::dot(dir, kd));
          c += glm::vec3(1.0f, 0.97f, 0.92f) * std::exp((dKey - 1.0f) * 12.0f) *
               5.0f;
          const float dFill = std::max(0.0f, glm::dot(dir, -kd));
          c += glm::vec3(0.55f, 0.65f, 0.85f) *
               std::exp((dFill - 1.0f) * 20.0f) * 1.2f;
          float* o = probe.data() + (kenshin::usize(y) * kW + x) * 4;
          o[0] = c.r;
          o[1] = c.g;
          o[2] = c.b;
          o[3] = 1.0f;
        }
      }
      // RGBA16F upload: convert the f32 rows to half bits (the HDR
      // path's 8-byte texels are f16 pairs - feeding f32 bits reads
      // as denormal garbage, a black environment).
      std::vector<kenshin::u16> half;
      half.reserve(probe.size());
      for (const float v : probe) {
        const kenshin::u32 b = *reinterpret_cast<const kenshin::u32*>(&v);
        const kenshin::u32 sign = (b >> 16) & 0x8000u;
        const kenshin::u32 ex = (b >> 23) & 0xFFu;
        const kenshin::u32 man = b & 0x7FFFFFu;
        kenshin::u16 hf;
        if (ex >= 255)
          hf = kenshin::u16(sign | 0x7C00u | (man ? 0x200u : 0));
        else if (ex < 98)  // underflow to zero/denormal (env range:
                           // values 0..8, plain zero is fine)
          hf = 0;
        else {
          const kenshin::u32 e2 = ex - 127 + 15;
          if (e2 >= 31)
            hf = kenshin::u16(sign | 0x7C00u);
          else
            hf = kenshin::u16(sign | (e2 << 10) | (man >> 13));
        }
        half.push_back(hf);
      }
      kenshin::backend::IRenderBackend::TextureArrayDesc ed;
      ed.width = kW;
      ed.height = kH;
      ed.layers = 1;
      ed.hdr = true;
      ed.rgba = reinterpret_cast<const kenshin::u8*>(half.data());
      ed.bytes = half.size() * sizeof(kenshin::u16);
      ed.debugName = "ibl-env";
      envTexture = vgBackend->createTextureArray(ed);
      envMaxLod = std::floor(std::log2(float(kW))) + 1.0f - 1.0f;
      KENSHIN_INFO("[ibl] studio probe {}x{} ({} mips) -> handle {}", kW, kH,
                   int(envMaxLod) + 1, envTexture);
    }

    if (args.noAsync) vgBackend->setAsyncChainEnabled(false);  // M4.4 A/B
  }

  // --cam: fixed initial pose for the S4.6 acceptance screenshots
  // (deterministic; the loop's per-frame updateMatrices re-applies the
  // live viewport aspect over this).
  if (args.hasCam)
    controller.setup(
        args.camYaw, args.camPitch, args.camDist,
        glm::vec3(args.camTarget[0], args.camTarget[1], args.camTarget[2]),
        static_cast<kenshin::f32>(args.width) /
            static_cast<kenshin::f32>(args.height));
  else if (vgScene && !vgScene->meshes().empty() &&
           vgScene->meshes()[0].boundsSphere.w > 0.0f) {
    // Auto-frame the default camera to the loaded asset (no --cam):
    // target the bounds-sphere center, distance ~2.4 radii so the
    // object fills a good third of the frame at a 45-degree vantage
    // (the fixed old default framed a 2-unit knot; any other asset
    // rendered as a tiny blob in a sea of ground).
    const glm::vec4& bs = vgScene->meshes()[0].boundsSphere;
    controller.setup(45.0f, 6.0f, bs.w * 2.8f, glm::vec3(bs),
                     static_cast<kenshin::f32>(args.width) /
                         static_cast<kenshin::f32>(args.height));
  }

  kenshin::core::FrameStats<240> stats;
  kenshin::Timer frameTimer;
  bool cameraChecked = false;
  std::vector<kenshin::renderer::DrawItem> items;

  // The in-frame chain (B3): every frame records cull->expand->cull->
  // raster into the current command buffer; the decode below consumes
  // the SAME frame's VisBuffer (camera lag and inter-frame stalls gone).
  kenshin::renderer::VisBufferChain chain;
  // M9.1: the shadow chain (a SECOND instance - own VisBuffer/pairs;
  // the same-frame second chain records on the graphics queue via the
  // M4.3 fallback).
  kenshin::renderer::VisBufferChain shadowChains[3];  // M9.1 Stage B
  bool shadowChainReady = false;
  // M9.1 Stage C pre-work: the shadow CONTENT cache key (UE's
  // FClipmapCacheKey semantics - VirtualShadowMapCacheManager.h: a
  // directional light's clipmap invalidates on light-direction/level
  // changes, plus primitive-visibility transitions). Our analogue of
  // the primitive trigger is streaming residency (a newly resident
  // page can close holes in the cached maps); the LOD threshold rides
  // along because our cut is threshold-driven (UE's page LOD is not).
  // While the key is unchanged the clipmap chains + decodes are
  // skipped entirely - the persistent R32F maps carry the content.
  struct ShadowCacheKey {
    glm::vec3 L{0.0f, 0.0f, -2.0f};  // impossible direction = dirty
    kenshin::f32 threshold = -1.0f;
    kenshin::u32 levels = 0, residentPages = 0xFFFFFFFFu;
    kenshin::u32 cullMode = 0xFFFFFFFFu, hwMode = 0xFFFFFFFFu;
    bool valid = false;
  } shadowKey;
  // Stage C: per-level dirty TILE masks (4x4 tiles of 512px over the
  // 2048^2 map). A key change dirties everything; each frame re-renders
  // only dirty tiles the main view NEEDS (the frustum-hull marking
  // below) and clears those bits - the scissored raster + LOAD-op
  // decode keep the rest of the map cached. UE's page semantics at
  // tile granularity (a whole-map "page" per 512px).
  kenshin::u32 shadowDirtyTiles[3] = {0u, 0u, 0u};
  kenshin::u32 shadowChainStats[3][16] = {};
  bool chainReady = false;
  bool decodeReady = false;  // pipeline is lazy: needs the swapchain format
  // Coverage histogram state (only with --raster-probe; the snapshot
  // follows the mapped-read contract - read before execute, logged at
  // the 60-frame cadence; values trail by two frames like [vgstats]).
  kenshin::u32 probeHist[8] = {};
  bool probeValid = false;
  const auto writeScreenshot =
      [&renderer](const char* what, const std::string& path, bool fullFrame) {
        std::vector<kenshin::u8> rgba;
        kenshin::u32 w = 0, h = 0;
        if (const auto parent = std::filesystem::path(path).parent_path();
            !parent.empty())
          std::filesystem::create_directories(parent);
        const bool read = fullFrame ? renderer.readbackFrame(rgba, w, h)
                                    : renderer.readbackViewport(rgba, w, h);
        if (read && kenshin::util::writeBmp24(path, rgba, w, h))
          KENSHIN_INFO("{} written: {} ({}x{})", what, path, w, h);
        else
          KENSHIN_ERROR("{} FAILED", what);
      };

  int frame = 0;
  while (window.pumpEvents()) {
    auto& input = kenshin::core::Input::get();
    if (input.keyPressed(VK_ESCAPE)) {
      KENSHIN_INFO("ESC pressed -> close");
      window.requestClose();
    }
    input.nextFrame();

    const kenshin::f32 dt =
        static_cast<kenshin::f32>(frameTimer.elapsedSeconds());
    frameTimer.reset();
    engine.beginFrame(dt);

    const bool isLastFrame = args.frames >= 0 && frame + 1 >= args.frames;

    if (isLastFrame) {
      const int r = static_cast<int>(kClearColor[0] * 255.0f + 0.5f);
      const int g = static_cast<int>(kClearColor[1] * 255.0f + 0.5f);
      const int b = static_cast<int>(kClearColor[2] * 255.0f + 0.5f);
      KENSHIN_INFO("frame {} expected color = ({},{},{})", frame, r, g, b);
    }

    const glm::vec2 vp = overlay.viewportSize();
    if (!renderer.beginFrame(static_cast<kenshin::u32>(vp.x),
                             static_cast<kenshin::u32>(vp.y), kClearColor)) {
      KENSHIN_ERROR("beginFrame failed");
      break;
    }

    overlay.draw(renderer.viewportTexture(), stats, camera, controller);

    controller.handleInput(camera, overlay.controllerInput(dt));
    controller.updateMatrices(camera, vp.x / glm::max(vp.y, 1.0f));

    if (!cameraChecked &&
        glm::abs(glm::dot(camera.forward(), camera.up())) < 1e-4f) {
      KENSHIN_INFO("[camera] view basis orthonormal: OK");
      cameraChecked = true;
    }

    // Assemble this frame's draw stream (pure data; the renderer layer
    // consumes it between beginFrame and endFrame).
    const bool drawable = vp.x > 1.0f && vp.y > 1.0f;
    items.clear();
    if (!kgeoMode && drawable && !scene.empty()) {
      for (const auto& shape : scene)
        items.push_back({shape.handle,
                         glm::translate(glm::mat4(1.0f), shape.offset),
                         shape.kd});
    }

    const kenshin::renderer::FrameView view{
        camera.viewProjMatrix(), camera.position(),
        glm::vec3(kStudioLight[0], kStudioLight[1], kStudioLight[2])};
    renderer.submit(view, items);

    if (virtualPath && drawable) {
      // Lazy init (pools ready; vis buffers start 1x1, first execute
      // resizes).
      if (!chainReady) {
        std::vector<kenshin::u32> meshHierRoots;
        for (const auto& mesh : vgScene->meshes())
          meshHierRoots.push_back(mesh.hierarchyRoot);
        // M6.3: paged assets feed the slot pool + tables; the v2
        // pools stay empty (the chain binds the pool at 1/2 itself).
        kenshin::renderer::ChainPagedInputs pagedInputs;
        kenshin::renderer::ChainPagedInputs* pagedPtr = nullptr;
        if (vgScene->paged()) {
          pagedInputs.slotPool = vgScene->pageSlotPool();
          pagedInputs.clusterPage = vgScene->clusterPageBuffer();
          pagedInputs.pageTable = vgScene->pageTableBuffer();
          pagedInputs.pageHeader = vgScene->pageHeaderBuffer();
          pagedInputs.pageCount = vgScene->pageCount();
          pagedInputs.slotU32Total =
              vgScene->pageSlotCount() * (kenshin::geometry::kPageSlotSize / 4);
          pagedInputs.clusterMeta = vgScene->clusterMetaBuffer();
          pagedPtr = &pagedInputs;
        }
        chainReady = chain.init(
            *vgBackend, shaders, vgScene->positions4(), vgScene->indices(),
            vgScene->clusterData(), vgScene->groupData(),
            vgScene->hierarchyData(), vgScene->groupRanges(), meshHierRoots,
            chainInstances, chainMeshRanges,
            // Pair-list capacity: min(the theoretical worst case,
            // 4M pairs) - the cull writers saturate and the consumers
            // clamp at this cap (the wide-instance scaling form).
            kenshin::u32(std::min<kenshin::u64>(
                kenshin::u64(chainInstances.size()) * vgScene->clusterCount(),
                kenshin::u64(4) << 20)),
            vgScene->clusterAabbs(), pagedPtr, 1, 1);
        // M9.1 Stage B: three clipmap levels (extent doubles per
        // level; each chain the same 2048 buffer - LOD rides
        // worldPerPixel, which the frame loop scales per level).
        // Once-only guard (the `x = f() && x` form latches false
        // forever - the M8.3 review's brace-less-if lesson).
        if (!shadowChainReady) {
          shadowChainReady = true;
          // Receiver-only instances (castShadows == 0, the ground) get a
          // ZERO transform in the shadow chains: degenerate bounds
          // frustum-cull them in EVERY cull mode, the index space stays
          // aligned with the scene's decode buffer, and no pairs exist
          // for the decode to touch. A ground plane in its own map
          // self-receives at minified sampling (the moire band artifact
          // no bias clears) and occludes nothing above it in a
          // directional light anyway.
          std::vector<kenshin::renderer::CullGpuInstance> shadowInsts =
              chainInstances;
          for (auto& si2 : shadowInsts)
            if (si2.pad[1] == 0u) {
              // Park the bounds sphere FAR outside every frustum: the
              // zero-transform alone leaves the instance at the origin
              // (inside the light box) and its CLUSTER bounds still
              // pass the cluster-level tests.
              si2.model = glm::mat4(0.0f);
              si2.bounds = glm::vec4(1e9f, 1e9f, 1e9f, 0.0f);
            }
          KENSHIN_INFO("[dbg] shadowInsts: {} entries, b1=({},{},{},{})",
                       shadowInsts.size(), shadowInsts[1].bounds.x,
                       shadowInsts[1].bounds.y, shadowInsts[1].bounds.z,
                       shadowInsts[1].bounds.w);
          for (int lv = 0; lv < 3; ++lv)
            shadowChainReady =
                shadowChains[lv].init(
                    *vgBackend, shaders, vgScene->positions4(),
                    vgScene->indices(), vgScene->clusterData(),
                    vgScene->groupData(), vgScene->hierarchyData(),
                    vgScene->groupRanges(), meshHierRoots, shadowInsts,
                    chainMeshRanges, chain.maxPairs(), vgScene->clusterAabbs(),
                    pagedPtr, 2048, 2048) &&
                shadowChainReady;
        }
        // The timing pool is per-slot shared with the main chain - the
        // shadow chains must not touch it (see setTimingEnabled).
        for (int lv = 0; lv < 3; ++lv) shadowChains[lv].setTimingEnabled(false);
        if (chainReady)
          KENSHIN_INFO("[vgchain] in-frame chain ready{}",
                       pagedPtr ? " (paged)" : "");
      }
      if (!decodeReady) decodeReady = decodePass.init(*vgBackend, shaders);
      if (!lightingReady)
        lightingReady = lightingPass.init(*vgBackend, shaders);
      // Own once-only guard (a bare second statement under the
      // brace-less if above would re-create the pipeline stack every
      // frame - the review's catch).
      if (!tonemapReady) tonemapReady = tonemapPass.init(*vgBackend, shaders);
      if (!taaReady) taaReady = taaPass.init(*vgBackend, shaders);
      if (chainReady && decodeReady) {
        // Same camera, same frame: the chain's inputs and the decode's
        // matrices cannot disagree anymore.
        kenshin::renderer::ChainInputs ci;
        ci.view = camera.viewMatrix();
        // M9 TAA: subpixel NDC jitter (8-phase Halton(2,3)) rides the
        // projection's [2][0]/[2][1] (clip.xy += jitter * clip.w); the
        // resolve unprojects with the UNJITTERED pair so the jitter
        // never leaks into reprojection.
        const bool taaOn = (vgDebug.taa || args.forceTaa) && !args.noTaa;
        glm::mat4 projJ = camera.projMatrix();
        glm::vec2 taaJitterUV(0.0f);
        if (taaOn) {
          // NOTE: ci.width is assigned AFTER this block (from vp) - the
          // divisor MUST be vp.x/vp.y. Dividing by the still-zero
          // ci.width produced an inf jitter that garbage'd the whole
          // projection (the chain culled to zero clusters - THE TAA
          // black-screen root cause; the pass/descriptor stack was
          // never broken).
          const glm::vec2 h = halton23(int(frame) & 7);
          const glm::vec2 jn((h.x * 2.0f - 1.0f) / vp.x,
                             (h.y * 2.0f - 1.0f) / vp.y);
          projJ[2][0] += jn.x;
          projJ[2][1] += jn.y;
          taaJitterUV = jn * 0.5f;  // NDC -> UV for the resolve
        }
        const glm::mat4 viewProjJ = projJ * camera.viewMatrix();
        ci.proj = projJ;
        ci.nearZ = camera.nearPlane();
        ci.camPos = camera.position();
        ci.pixelThreshold = vgDebug.pixelThreshold;
        ci.worldPerPixel = 2.0f *
                           std::tan(glm::radians(controller.fovYDeg) * 0.5f) /
                           static_cast<kenshin::f32>(vp.y);
        ci.width = static_cast<kenshin::u32>(vp.x);
        ci.height = static_cast<kenshin::u32>(vp.y);
        kenshin::renderer::extractFrustumPlanes(viewProjJ, ci.planes);
        ci.imposterThresholdPx =
            (args.imposterDebug || vgDebug.imposterDebug) ? 4096.0f : 0.0f;
        ci.occlusionEnabled = vgDebug.occlusion && !args.noOcclusion;
        // Pure SW is the default since the 2026-09-19 threshold sweep:
        // mode-1 SW beat every routing configuration on all five scene
        // shapes (micro-heavy carpet 147 vs 102 fps, coarse-LOD 144 vs
        // 129, close-up zoom 357 vs 326, occlusion-on 155 vs 95) - the
        // LOD bounds triangle screen sizes well below the point where
        // fixed-function raster pays, and the near-plane guard keeps the
        // genuinely huge triangles in SW anyway. --route-hw / --force-hw
        // keep the HW path reachable for A/B; --no-hw is the default's
        // no-op alias.
        ci.hwMode = args.forceHw ? 2u : (args.routeHw ? 1u : 0u);
        (void)args.noHw;
        ci.hwThresholdPx = vgDebug.hwThresholdPx;  // CLI seeds, panel drives
        ci.cullMode = !vgDebug.hierCull || args.cullLinear
                          ? 0u
                          : (args.cullPersistent ? 2u : 1u);
        // Mode 1 (the exact row-span clamp + stat aggregation) is the
        // measured default since the 2026-09-19 A/B: bit-identical
        // output (6-config differential + dual-mode CPU-mirror pins),
        // +82% fps pure-SW / stabilized routed SW segment. --raster-base
        // keeps the baseline walk reachable for same-binary A/B.
        ci.rasterMode = args.rasterBase ? 0u : 1u;
        ci.rasterProbe = args.rasterProbe ? 1u : 0u;
        // Panel stats: NON-BLOCKING mapped snapshot, taken BEFORE this
        // frame's execute. At this point the CURRENT slot's counters
        // still hold the frame-before-last's final values, and
        // beginFrame's pacing wait has PROVEN that frame complete -
        // reading after execute (the old site) raced the GPU's own
        // writes to the not-yet-finished previous submission (a
        // "submitted" frame is not an "executed" frame: single-queue
        // submissions may overlap), which showed up as torn counters
        // - intermittent zero clusters, wildly varying write counts.
        probeValid =
            args.rasterProbe && chain.readRasterProbe(*vgBackend, probeHist);
        {
          kenshin::u32 cs[16];
          if (chainReady && chain.readStatsMapped(*vgBackend, cs)) {
            vgDebug.visibleClusters = cs[2];
            vgDebug.cutTris = cs[11];
            vgDebug.pxWrites = cs[5] + cs[10];  // main + post raster
            vgDebug.occCulled = cs[8];
            vgDebug.revived = cs[9];
            vgDebug.hwClusters = cs[12];
            vgDebug.bfCulls = cs[6];
            vgDebug.nearClips = cs[7];
            vgDebug.workItems = cs[1];
            vgDebug.tooCoarse = cs[3];
            vgDebug.srcTooCoarse = cs[4];
            vgDebug.visitedNodes = cs[13];
            vgDebug.prunedNodes = cs[14];
            vgDebug.pendingSkips = cs[15];
            const kenshin::f32 cover =
                100.0f * static_cast<kenshin::f32>(cs[5] + cs[10]) /
                (ci.width * ci.height);
            vgDebug.coveragePct = cover;
            vgDebug.pushHistory();
          }
        }

        chain.execute(*vgBackend, ci);

        // ---- M9.2: the imposter injection --------------------------------
        if (!imposterReady) {
          imposterReady = imposterPass.init(*vgBackend, shaders);
          meshAtlasBase.assign(vgScene->meshes().size(),
                               kenshin::renderer::ImposterPass::kNoAtlas);
          // meshRootCluster came from the setup-time staging - the
          // in-frame assign wiped it (the "cluster 4294967295" log).
          for (kenshin::u32 m = 0; m < stagedDepth.size(); ++m) {
            if (stagedDepth[m].empty()) continue;
            const kenshin::u32 base = imposterPass.uploadAtlas(
                *vgBackend, stagedDepth[m].data(), stagedTri[m].data(),
                kenshin::u32(stagedDepth[m].size()));
            meshAtlasBase[m] = base;
            KENSHIN_INFO("[imposter] mesh {}: atlas @{} (cluster {})", m, base,
                         meshRootCluster[m]);
          }
        }
        if (imposterReady) {
          kenshin::renderer::ImposterQuad quads[64];
          glm::uvec2 pairs[256];
          kenshin::u32 quadCount = 0, pairCount = 0;
          for (kenshin::u32 ii = 0;
               ii < chainInstances.size() && quadCount < 64 && pairCount < 255;
               ++ii) {
            const auto& inst = chainInstances[ii];
            if (inst.meshId >= meshAtlasBase.size()) continue;
            if (meshAtlasBase[inst.meshId] ==
                kenshin::renderer::ImposterPass::kNoAtlas)
              continue;
            // M9.2: mirror the cull's tiny-skip EXACTLY.
            const float thresholdPx = ci.imposterThresholdPx;
            if (thresholdPx <= 0.0f || inst.pad[1] != 2u) continue;
            if (ii >= realBounds.size()) continue;
            const glm::vec3 c(realBounds[ii]);
            const float r = realBounds[ii].w;
            const float d =
                glm::max(glm::distance(c, camera.position()), 1e-3f);
            if (r / d / ci.worldPerPixel >= thresholdPx) continue;
            const glm::vec4 clip = viewProjJ * glm::vec4(c, 1.0f);
            if (clip.w <= 0.0f) continue;
            const glm::vec2 ndc = glm::vec2(clip.x, clip.y) / clip.w;
            const float cx = (ndc.x * 0.5f + 0.5f) * float(ci.width);
            const float cy = (ndc.y * 0.5f + 0.5f) * float(ci.height);
            const int R = std::max(6, int(r / d / ci.worldPerPixel));
            if (R > int(ci.width)) continue;
            auto& q = quads[quadCount];
            q.rect =
                glm::ivec4(int(cx) - R, int(cy) - R, int(cx) + R, int(cy) + R);
            q.centerAtlas = glm::vec4(c, float(meshAtlasBase[inst.meshId]));
            q.halfPair = glm::vec4(r, r, r, float(pairCount));
            q.pad = glm::vec4(0.0f);
            ++quadCount;
            pairs[pairCount++] = glm::uvec2(ii, meshRootCluster[inst.meshId]);
          }
          if (quadCount > 0) {
            imposterPass.setPairs(*vgBackend, pairs, pairCount);
            imposterPass.draw(*vgBackend, chain.visBufferCurrent(*vgBackend),
                              camera.viewProjMatrix(), camera.position(),
                              ci.width, ci.height, quads, quadCount);
          }
        }

        // M9.1 Stage B: the clipmap rig - three levels, extent
        // 12/24/48m, each a light-view chain + decode. worldPerPixel
        // scales with extent (UE's texel-size LOD calibration);
        // sampling picks the level by log2 distance with coarse
        // fallback.
        const bool shadowsOn = vgDebug.shadows && !args.noShadows &&
                               shadowChainReady && decodeReady;
        const bool shadowPhase = vgDebug.viewMode == 0 && lightingReady &&
                                 shadowChainReady && decodeReady;
        kenshin::renderer::DecodePagedInputs spd{};
        if (vgScene->paged()) {
          spd.slotPool = vgScene->pageSlotPool();
          spd.clusterPage = vgScene->clusterPageBuffer();
          spd.pageTable = vgScene->pageTableBuffer();
          spd.pageHeader = vgScene->pageHeaderBuffer();
          spd.slotU32Total =
              vgScene->pageSlotCount() * (kenshin::geometry::kPageSlotSize / 4);
          spd.clusterMeta = vgScene->clusterMetaBuffer();
        }
        kenshin::renderer::LightingPass::ShadowLevel shadowLevels[3];
        kenshin::u32 shadowLevelCount = 0;
        glm::vec3 shadowOrigin(0.0f);
        if (shadowsOn) {
          const glm::vec3 L = glm::normalize(vgDebug.lightDir);
          const glm::vec3 center(args.hasCam ? args.camTarget[0] : 0.0f,
                                 args.hasCam ? args.camTarget[1] : 0.0f,
                                 args.hasCam ? args.camTarget[2] : 0.0f);
          constexpr float kExtent0 = 12.0f;
          constexpr float kDepth = 40.0f;
          // Z-up world: the light view's up rides the Z axis (a
          // horizontal up sits near-parallel to L for low lights).
          const glm::vec3 up =
              std::abs(L.z) > 0.95f ? glm::vec3(1, 0, 0) : glm::vec3(0, 0, 1);
          shadowLevelCount = kenshin::u32(vgDebug.shadowLevels);
          shadowOrigin = center;
          // UE FClipmapCacheKey: light direction + level count, plus
          // our primitive-visibility analogues (streaming residency)
          // and the LOD threshold. A key change dirties every TILE of
          // every level; each frame then renders only dirty tiles the
          // main view needs and clears those bits (Stage C).
          const kenshin::u32 residentNow =
              streamer ? streamer->residentPages() : 0xFFFFFFFFu;
          const bool keyChanged =
              !shadowKey.valid || glm::distance(L, shadowKey.L) > 1e-4f ||
              vgDebug.pixelThreshold != shadowKey.threshold ||
              shadowLevelCount != shadowKey.levels ||
              residentNow != shadowKey.residentPages ||
              ci.cullMode != shadowKey.cullMode ||
              ci.hwMode != shadowKey.hwMode;
          if (keyChanged) {
            shadowDirtyTiles[0] = shadowDirtyTiles[1] = shadowDirtyTiles[2] =
                0xFFFFu;
            shadowKey.L = L;
            shadowKey.threshold = vgDebug.pixelThreshold;
            shadowKey.levels = shadowLevelCount;
            shadowKey.residentPages = residentNow;
            shadowKey.cullMode = ci.cullMode;
            shadowKey.hwMode = ci.hwMode;
            shadowKey.valid = true;
          }
          const glm::mat4 invMainVP = glm::inverse(camera.viewProjMatrix());
          for (kenshin::u32 lv = 0; lv < shadowLevelCount; ++lv) {
            const float extent = kExtent0 * float(1 << lv);
            // The light eye rides the LIGHT'S side of the scene (+L):
            // from the far side (-L) the single-sided ground is
            // backface-culled out of the shadow map AND every occluder
            // above it is farther from the eye than the ground itself -
            // the ground can never test as shadowed (the M9.1 missing
            // ground shadow).
            const glm::mat4 lightView =
                glm::lookAtRH(center + L * (kDepth * 0.5f), center, up);
            glm::mat4 lightProj =
                glm::orthoRH_ZO(-extent, extent, -extent, extent, 0.1f, kDepth);
            lightProj[1][1] *= -1.0f;  // the engine's Y-flip convention
            shadowLevels[lv].viewProj = lightProj * lightView;
            shadowLevels[lv].texelWorld = (2.0f * extent) / 2048.0f;

            // Stage C: dirty tiles the main view needs. The bounding
            // rect of the worked tiles scissors BOTH the chain raster
            // and the decode (LOAD-op) - the rest of the map keeps its
            // cached content.
            const kenshin::u32 work =
                shadowDirtyTiles[lv] &
                shadowTileMask(shadowLevels[lv].viewProj, invMainVP);
            if (work == 0u) continue;
            kenshin::u32 tx0 = 4, ty0 = 4, tx1 = 0, ty1 = 0;
            for (kenshin::u32 ty = 0; ty < 4; ++ty)
              for (kenshin::u32 tx = 0; tx < 4; ++tx)
                if (work & (1u << (ty * 4u + tx))) {
                  tx0 = std::min(tx0, tx);
                  ty0 = std::min(ty0, ty);
                  tx1 = std::max(tx1, tx);
                  ty1 = std::max(ty1, ty);
                }
            constexpr kenshin::u32 kTilePx = 2048 / 4;
            const kenshin::u32 rect[4] = {tx0 * kTilePx, ty0 * kTilePx,
                                          tx1 * kTilePx + kTilePx - 1,
                                          ty1 * kTilePx + kTilePx - 1};

            kenshin::renderer::ChainInputs si{};
            si.view = lightView;
            si.proj = lightProj;
            si.nearZ = 0.1f;
            // The EYE, not the look-at center: the routing guard and the
            // LOD distances measure from camPos - with the center, every
            // cluster near the target has its bounds sphere containing
            // camPos (dist clamps to 0.1) and the HW near-guard rejects
            // it (the shadow hw=0 case).
            si.camPos = center + L * (kDepth * 0.5f);
            si.scissorRect[0] = rect[0];
            si.scissorRect[1] = rect[1];
            si.scissorRect[2] = rect[2];
            si.scissorRect[3] = rect[3];
            si.pixelThreshold = vgDebug.pixelThreshold * 2.0f;  // D11.5 bias
            si.worldPerPixel = (2.0f * extent) / 2048.0f;
            si.width = 2048;
            si.height = 2048;
            kenshin::renderer::extractFrustumPlanes(shadowLevels[lv].viewProj,
                                                    si.planes);
            si.occlusionEnabled = false;  // HZB is main-view; shadows skip
            // UE-faithful routing: UE's DepthOnly/shadow raster runs the
            // dual path with routing (NaniteCullRaster: HardwareThen-
            // Software for shadow maps), so giant occluder triangles go
            // to HW instead of SW-scanning the whole 2048^2 map. The
            // main chain keeps its all-SW default; --no-hw / --force-hw
            // keep their A/B meaning here.
            si.hwMode = args.noHw ? 0u : (args.forceHw ? 2u : 1u);
            si.cullNonCasters = 1u;  // belt: the host also zero-scales
                                     // non-casters in the shadow chains
            si.rasterMode = 1;
            shadowChains[lv].execute(*vgBackend, si);

            kenshin::renderer::DecodeParams sp{};
            sp.camPos = center;
            sp.lightDir = vgDebug.lightDir;
            sp.view = kenshin::renderer::DecodeView::Final;
            if (vgScene->paged()) {
              decodePass.drawShadowPaged(
                  *vgBackend, shadowLevels[lv].viewProj, *vgScene, sp, 2048,
                  2048, shadowChains[lv].visBufferCurrent(*vgBackend), spd,
                  shadowChains[lv].pairsCurrent(*vgBackend),
                  shadowChains[lv].hwPairsCurrent(*vgBackend),
                  shadowChains[lv].postPairsCurrent(*vgBackend),
                  shadowChains[lv].maxPairs(), lv, true, rect);
            } else {
              decodePass.drawShadow(
                  *vgBackend, shadowLevels[lv].viewProj, *vgScene, sp, 2048,
                  2048, shadowChains[lv].visBufferCurrent(*vgBackend),
                  shadowChains[lv].pairsCurrent(*vgBackend),
                  shadowChains[lv].hwPairsCurrent(*vgBackend),
                  shadowChains[lv].postPairsCurrent(*vgBackend),
                  shadowChains[lv].maxPairs(), lv, true, rect);
            }
            shadowDirtyTiles[lv] &= ~work;
          }
        } else if (shadowPhase) {
          // Disabled this frame: still touch a target so lighting's
          // sampling reads a fully-lit map (clear-to-one path). The
          // clear destroys the cached content - re-render on toggle-on.
          shadowKey.valid = false;
          decodePass.drawShadow(*vgBackend, glm::mat4(1.0f), *vgScene, {}, 16,
                                16, 0, 0, 0, 0, 0, 0, false);
        }

        decodeParams.camPos = camera.position();
        decodeParams.view =
            static_cast<kenshin::renderer::DecodeView>(vgDebug.viewMode);
        // M8.1: view mode 0 (Final) runs the deferred pair - the decode
        // writes the GBuffer, the lighting pass shades it (multi-light
        // GGX) into the frame color target. Debug modes (1..6) keep the
        // legacy direct decode (no GBuffer involvement).
        const bool finalView = vgDebug.viewMode == 0;
        if (vgScene->paged()) {
          kenshin::renderer::DecodePagedInputs pd;
          pd.slotPool = vgScene->pageSlotPool();
          pd.clusterPage = vgScene->clusterPageBuffer();
          pd.pageTable = vgScene->pageTableBuffer();
          pd.pageHeader = vgScene->pageHeaderBuffer();
          pd.slotU32Total =
              vgScene->pageSlotCount() * (kenshin::geometry::kPageSlotSize / 4);
          pd.clusterMeta = vgScene->clusterMetaBuffer();
          if (finalView) {
            decodePass.drawGBufferPaged(
                *vgBackend, viewProjJ, *vgScene, decodeParams, ci.width,
                ci.height, chain.visBufferCurrent(*vgBackend), pd,
                chain.pairsCurrent(*vgBackend),
                chain.hwPairsCurrent(*vgBackend),
                chain.postPairsCurrent(*vgBackend), chain.maxPairs(),
                albedoTexture, imposterReady ? imposterPass.pairsBuffer() : 0);
          } else {
            decodePass.drawPaged(
                *vgBackend, viewProjJ, *vgScene, decodeParams, ci.width,
                ci.height, chain.visBufferCurrent(*vgBackend), pd,
                chain.pairsCurrent(*vgBackend),
                chain.hwPairsCurrent(*vgBackend),
                chain.postPairsCurrent(*vgBackend), chain.maxPairs());
          }
        } else {
          if (finalView) {
            decodePass.drawGBuffer(*vgBackend, viewProjJ, *vgScene,
                                   decodeParams, ci.width, ci.height,
                                   chain.visBufferCurrent(*vgBackend),
                                   chain.pairsCurrent(*vgBackend),
                                   chain.hwPairsCurrent(*vgBackend),
                                   chain.postPairsCurrent(*vgBackend),
                                   chain.maxPairs(), albedoTexture);
          } else {
            decodePass.draw(
                *vgBackend, viewProjJ, *vgScene, decodeParams, ci.width,
                ci.height, chain.visBufferCurrent(*vgBackend),
                chain.pairsCurrent(*vgBackend),
                chain.hwPairsCurrent(*vgBackend),
                chain.postPairsCurrent(*vgBackend), chain.maxPairs());
          }
        }
        if (finalView && lightingReady) {
          // The deterministic demo light rig: one directional (the
          // studio light, panel-adjustable) + two static point accents.
          std::vector<kenshin::renderer::GpuLightC> lights;
          kenshin::renderer::GpuLightC dir;
          dir.a = glm::vec4(glm::normalize(vgDebug.lightDir), 0.0f);
          dir.b = glm::vec4(vgDebug.lightColor * vgDebug.lightIntensity, 0.0f);
          lights.push_back(dir);
          for (int p = 0; p < 2; ++p) {
            kenshin::renderer::GpuLightC pt;
            pt.a = glm::vec4(kPointLights[p], 1.0f);
            pt.b = glm::vec4(kPointColors[p] * vgDebug.pointIntensity,
                             kPointRange);
            lights.push_back(pt);
          }
          lightingPass.draw(
              *vgBackend, glm::inverse(viewProjJ), camera.position(), lights,
              vgScene->paged()
                  ? vgScene->materialTableBuffer()
                  : kenshin::backend::IRenderBackend::kInvalidBufferHandle,
              vgScene->paged() ? vgScene->materialCount() : 0u,
              glm::vec4(kClearColor[0], kClearColor[1], kClearColor[2], 1.0f),
              vgDebug.ambient, vgDebug.skyColor, vgDebug.groundColor,
              shadowLevels,
              shadowsOn && vgDebug.viewMode == 0 ? shadowLevelCount : 0u, 12.0f,
              shadowOrigin, ci.width, ci.height, envTexture, envMaxLod);
          // M9 TAA resolve (skip the first frames after (re)validation:
          // the history is garbage until the ping-pong fills).
          bool resolvedByTaa = false;
          if (taaReady && taaOn) {
            kenshin::renderer::TaaFrame tf;
            tf.invViewProj = glm::inverse(camera.viewProjMatrix());
            tf.prevViewProj = taaPrevViewProj;
            tf.camPos = camera.position();
            tf.blend = taaPrevValid && taaWarmup > 1 ? 0.95f : 0.0f;
            tf.jitter = taaJitterUV;
            taaPass.draw(*vgBackend, tf, ci.width, ci.height);
            resolvedByTaa = true;
          }
          taaPrevViewProj = camera.viewProjMatrix();
          taaPrevValid = true;
          ++taaWarmup;
          if (tonemapReady)
            tonemapPass.draw(
                *vgBackend,
                static_cast<kenshin::renderer::TonemapPass::Operator>(
                    vgDebug.tonemapMode),
                vgDebug.exposure,
                glm::vec4(kClearColor[0], kClearColor[1], kClearColor[2], 1.0f),
                ci.width, ci.height, resolvedByTaa,
                resolvedByTaa ? vgDebug.taaSharpen : 0.0f);
        }
      }
    }

    // Full-frame capture must be flagged before endFrame records it.
    const bool exiting = isLastFrame || !window.isOpen();
    if (!args.screenshotUi.empty() && exiting) renderer.requestFrameCapture();

    renderer.endFrame();

    // M6.3: the streaming pump rides between endFrame and the next
    // execute (readTouchedPages' pacing contract): consume the touched
    // bitmap, drive LRU + misses, drain finished IO into uploads.
    if (streamer && chainReady) {
      streamer->onFrame(chain);
      vgDebug.missPages = streamer->missedPages();
    }
    engine.endFrame();

    stats.push(dt * 1000.0f);
    if (frame > 0 && frame % 60 == 0) {
      KENSHIN_INFO("[pacing] avg={:.3f}ms std={:.3f}ms max={:.3f}ms fps={:.1f}",
                   stats.averageMs(), stats.stddevMs(), stats.maxMs(),
                   stats.fps());
      if (chainReady)
        KENSHIN_INFO(
            "[vgstats] work {} clusters {} tris {} writes {} occ {} "
            "revived {} hw {} tooCoarse {} srcTooCoarse {} visited {} "
            "pruned {} pend {} miss {} gpuMs {}",
            vgDebug.workItems, vgDebug.visibleClusters, vgDebug.cutTris,
            vgDebug.pxWrites, vgDebug.occCulled, vgDebug.revived,
            vgDebug.hwClusters, vgDebug.tooCoarse, vgDebug.srcTooCoarse,
            vgDebug.visitedNodes, vgDebug.prunedNodes, vgDebug.pendingSkips,
            vgDebug.missPages, vgBackend->gpuTimings().frameMs);
      // Chain segment GPU times (execution only - scheduling/signal
      // latency sits outside timestamps; the M5.3 lesson keeps
      // wall-clock the cross-variant primary).
      if (chainReady) {
        const auto& gt = vgBackend->gpuTimings();
        KENSHIN_INFO(
            "[chainms] cull {:.3f} sw {:.3f} hzb1 {:.3f} post {:.3f} "
            "hzb2 {:.3f} tail {:.3f}{}",
            gt.chainMs[0], gt.chainMs[1], gt.chainMs[2], gt.chainMs[3],
            gt.chainMs[4], gt.chainMs[5],
            gt.chainValid ? "" : " (unavailable)");
      }
      // Coverage histogram (only with --raster-probe): triangles by
      // covered-pixel bucket 0 / 1 / 2-4 / 5-16 / 17-64 / >64, then
      // the pixel sum and the attempted-triangle count (main + post).
      // Snapshot taken before this frame's execute (the mapped-read
      // contract); values trail by two frames like [vgstats].
      if (probeValid)
        KENSHIN_INFO(
            "[rasthist] b0 {} b1 {} b2_4 {} b5_16 {} b17_64 {} b64p {} "
            "pix {} tris {} bfcull {} nearclip {}",
            probeHist[0], probeHist[1], probeHist[2], probeHist[3],
            probeHist[4], probeHist[5], probeHist[6], probeHist[7],
            vgDebug.bfCulls, vgDebug.nearClips);
      // Shadow clipmap content stats (Stage C pre-work): per level the
      // cut triangles + pixel writes of the LAST render; while cached
      // the values are stale by design - that is the cache working.
      if (chainReady && shadowChainReady) {
        for (kenshin::u32 lv = 0; lv < 3; ++lv)
          if (!shadowChains[lv].readStatsMapped(*vgBackend,
                                                shadowChainStats[lv]))
            break;
        KENSHIN_INFO(
            "[shadowstats] key {} dirty {:02x}/{:02x}/{:02x} | lv0 tris {} "
            "px {} hw {} pend {} | lv1 tris {} px {} pend {} | lv2 tris {} "
            "px {} pend {}",
            shadowKey.valid ? 1 : 0, shadowDirtyTiles[0] & 0xFFFFu,
            shadowDirtyTiles[1] & 0xFFFFu, shadowDirtyTiles[2] & 0xFFFFu,
            shadowChainStats[0][11],
            shadowChainStats[0][5] + shadowChainStats[0][10],
            shadowChainStats[0][12], shadowChainStats[0][15],
            shadowChainStats[1][11],
            shadowChainStats[1][5] + shadowChainStats[1][10],
            shadowChainStats[1][15], shadowChainStats[2][11],
            shadowChainStats[2][5] + shadowChainStats[2][10],
            shadowChainStats[2][15]);
      }
    }

    if (exiting) {
      if (!args.screenshot.empty())
        writeScreenshot("screenshot", args.screenshot, false);
      if (!args.screenshotUi.empty())
        writeScreenshot("ui screenshot", args.screenshotUi, true);
    }

    ++frame;
    if (isLastFrame) {
      KENSHIN_INFO("reached --frames {} -> close", args.frames);
      window.requestClose();
    }
  }

  KENSHIN_INFO("main loop exited after {} frames", frame);
  if (streamer) streamer->shutdown();
  if (virtualPath) {
    decodePass.onShutdown(*vgBackend);
    if (lightingReady) lightingPass.onShutdown(*vgBackend);
    if (tonemapReady) tonemapPass.onShutdown(*vgBackend);
    if (taaReady) taaPass.onShutdown(*vgBackend);
    if (imposterReady) imposterPass.onShutdown(*vgBackend);
    if (chainReady) chain.onShutdown(*vgBackend);
  }
  // Reverse registration order: overlay context, mesh pool (all GPU
  // meshes), renderer pipeline, shader manager, backend, window.
  engine.shutdown();
  kenshin::log::shutdown();
  return 0;
}
