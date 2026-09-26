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
#include "kenshin/renderer/LightingPass.h"
#include "kenshin/renderer/MaterialDecodePass.h"
#include "kenshin/renderer/Renderer.h"
#include "kenshin/renderer/SoftRasterPass.h"
#include "kenshin/renderer/StreamingManager.h"
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
  bool noShadows = false;       // M9.1 A/B: disable the shadow map
  std::string ground;           // M9.1: a ground-plane .kgeo whose
                                // instances receive the shadows
  bool multi = false;           // S4.6 acceptance: near/mid/far instances
  bool dense = false;           // M3.4 acceptance: front-back occluders
  int grid = 0;                 // --grid N: N-instance occlusion carpet
                                // (1..32; the VisBuffer payload's 5b)
  bool wall = false;            // --wall: big occluder + hidden fleet
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
    else if (std::strcmp(argv[i], "--mesh") == 0 && i + 1 < argc)
      args.mesh = argv[++i];
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
    else if (std::strcmp(argv[i], "--ground") == 0 && i + 1 < argc)
      args.ground = argv[++i];
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
  glm::vec3 lightColor{1.0f, 0.97f, 0.92f};
  kenshin::f32 lightIntensity = 4.0f;
  kenshin::f32 pointIntensity = 24.0f;
  kenshin::f32 ambient = 0.15f;
  kenshin::f32 exposure = 2.2f;
  // M8.3: the hemisphere ambient's two poles + tonemap operator.
  glm::vec3 skyColor{0.65f, 0.75f, 0.95f};
  glm::vec3 groundColor{0.28f, 0.25f, 0.22f};
  int tonemapMode = 2;   // 0 Reinhard, 1 ACES(Narkowicz), 2 Filmic(UE)
  bool shadows = true;   // M9.1: the directional light's shadow map
  int shadowLevels = 3;  // Stage B clipmap level count (1..3)
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
  kenshin::renderer::TonemapPass tonemapPass;  // M8.3 HDR chain tail
  bool tonemapReady = false;
  kenshin::renderer::DecodeParams decodeParams;
  // M6.3: page-streaming driver (v3 assets; null otherwise).
  std::unique_ptr<kenshin::renderer::StreamingManager> streamer;
  // M8.2: the albedo texture-array handle (kInvalidTextureHandle =
  // no textures loaded; the backend binds its 1x1 dummy).
  kenshin::u32 albedoTexture =
      kenshin::backend::IRenderBackend::kInvalidTextureHandle;
  VgDebugState vgDebug;
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
          if (mat.texPath.empty()) continue;
          const auto resolved =
              mat.texPath.find(':') == std::string::npos &&
                      !std::filesystem::exists(kgeoDir / mat.texPath)
                  ? std::filesystem::path(mat.texPath)
                  : (kgeoDir / mat.texPath);
          const kenshin::u32 layer = textureManager.load(resolved.string());
          if (layer != kenshin::manager::TextureManager::kInvalidLayer)
            mat.texLayer = static_cast<kenshin::f32>(layer);
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
        const auto makeInstFor = [&](kenshin::u32 id, const glm::mat4& model) {
          const kenshin::renderer::VgMeshRange& m =
              vgScene.meshes()[id];
          kenshin::renderer::CullGpuInstance inst;
          inst.model = model;
          inst.meshId = id;
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
          inst.bounds =
              glm::vec4(glm::vec3(model * glm::vec4(m.boundsSphere.x,
                                                    m.boundsSphere.y,
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
          chainInstances = {makeInst(glm::mat4(1.0f))};
          std::vector<kenshin::renderer::VgInstance> vgInsts = {
              kenshin::renderer::VgInstance{glm::mat4(1.0f), vgMeshId}};
          if (groundMeshId != 0xFFFFFFFFu) {
            // Z-up: ground_z.kgeo is a z=0 plane; drop it below the
            // knot (radius ~1.5) so the knot sits on it.
            const glm::mat4 gm =
                glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -1.5f));
            chainInstances.push_back(makeInstFor(groundMeshId, gm));
            vgInsts.push_back({gm, groundMeshId});
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
  if (virtualPath) {
    vgScene = &engine.get<kenshin::renderer::VirtualGeometryScene>();
    vgBackend = &engine.get<kenshin::backend::IRenderBackend>();
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
          for (int lv = 0; lv < 3; ++lv)
            shadowChainReady =
                shadowChains[lv].init(
                    *vgBackend, shaders, vgScene->positions4(),
                    vgScene->indices(), vgScene->clusterData(),
                    vgScene->groupData(), vgScene->hierarchyData(),
                    vgScene->groupRanges(), meshHierRoots, chainInstances,
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
      if (chainReady && decodeReady) {
        // Same camera, same frame: the chain's inputs and the decode's
        // matrices cannot disagree anymore.
        kenshin::renderer::ChainInputs ci;
        ci.view = camera.viewMatrix();
        ci.proj = camera.projMatrix();
        ci.nearZ = camera.nearPlane();
        ci.camPos = camera.position();
        ci.pixelThreshold = vgDebug.pixelThreshold;
        ci.worldPerPixel = 2.0f *
                           std::tan(glm::radians(controller.fovYDeg) * 0.5f) /
                           static_cast<kenshin::f32>(vp.y);
        ci.width = static_cast<kenshin::u32>(vp.x);
        ci.height = static_cast<kenshin::u32>(vp.y);
        kenshin::renderer::extractFrustumPlanes(camera.viewProjMatrix(),
                                                ci.planes);
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
          // and the LOD threshold. Unchanged key = the cached R32F
          // maps stay valid; skip the chains + decodes wholesale.
          const kenshin::u32 residentNow =
              streamer ? streamer->residentPages() : 0xFFFFFFFFu;
          const bool shadowDirty =
              !shadowKey.valid ||
              glm::distance(L, shadowKey.L) > 1e-4f ||
              vgDebug.pixelThreshold != shadowKey.threshold ||
              shadowLevelCount != shadowKey.levels ||
              residentNow != shadowKey.residentPages ||
              ci.cullMode != shadowKey.cullMode || ci.hwMode != shadowKey.hwMode;
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

            if (!shadowDirty) continue;  // cached maps carry the content

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
            si.cullMode = ci.cullMode;
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
                  shadowChains[lv].maxPairs(), lv, true);
            } else {
              decodePass.drawShadow(
                  *vgBackend, shadowLevels[lv].viewProj, *vgScene, sp, 2048,
                  2048, shadowChains[lv].visBufferCurrent(*vgBackend),
                  shadowChains[lv].pairsCurrent(*vgBackend),
                  shadowChains[lv].hwPairsCurrent(*vgBackend),
                  shadowChains[lv].postPairsCurrent(*vgBackend),
                  shadowChains[lv].maxPairs(), lv, true);
            }
          }
          if (shadowDirty) {
            shadowKey.L = L;
            shadowKey.threshold = vgDebug.pixelThreshold;
            shadowKey.levels = shadowLevelCount;
            shadowKey.residentPages = residentNow;
            shadowKey.cullMode = ci.cullMode;
            shadowKey.hwMode = ci.hwMode;
            shadowKey.valid = true;
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
                *vgBackend, camera.viewProjMatrix(), *vgScene, decodeParams,
                ci.width, ci.height, chain.visBufferCurrent(*vgBackend), pd,
                chain.pairsCurrent(*vgBackend),
                chain.hwPairsCurrent(*vgBackend),
                chain.postPairsCurrent(*vgBackend), chain.maxPairs(),
                albedoTexture);
          } else {
            decodePass.drawPaged(
                *vgBackend, camera.viewProjMatrix(), *vgScene, decodeParams,
                ci.width, ci.height, chain.visBufferCurrent(*vgBackend), pd,
                chain.pairsCurrent(*vgBackend),
                chain.hwPairsCurrent(*vgBackend),
                chain.postPairsCurrent(*vgBackend), chain.maxPairs());
          }
        } else {
          if (finalView) {
            decodePass.drawGBuffer(*vgBackend, camera.viewProjMatrix(),
                                   *vgScene, decodeParams, ci.width, ci.height,
                                   chain.visBufferCurrent(*vgBackend),
                                   chain.pairsCurrent(*vgBackend),
                                   chain.hwPairsCurrent(*vgBackend),
                                   chain.postPairsCurrent(*vgBackend),
                                   chain.maxPairs(), albedoTexture);
          } else {
            decodePass.draw(
                *vgBackend, camera.viewProjMatrix(), *vgScene, decodeParams,
                ci.width, ci.height, chain.visBufferCurrent(*vgBackend),
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
              *vgBackend, glm::inverse(camera.viewProjMatrix()),
              camera.position(), lights,
              vgScene->paged()
                  ? vgScene->materialTableBuffer()
                  : kenshin::backend::IRenderBackend::kInvalidBufferHandle,
              vgScene->paged() ? vgScene->materialCount() : 0u,
              glm::vec4(kClearColor[0], kClearColor[1], kClearColor[2], 1.0f),
              vgDebug.ambient, vgDebug.skyColor, vgDebug.groundColor,
              shadowLevels,
              shadowsOn && vgDebug.viewMode == 0 ? shadowLevelCount : 0u, 12.0f,
              shadowOrigin, ci.width, ci.height);
          if (tonemapReady)
            tonemapPass.draw(
                *vgBackend,
                static_cast<kenshin::renderer::TonemapPass::Operator>(
                    vgDebug.tonemapMode),
                vgDebug.exposure,
                glm::vec4(kClearColor[0], kClearColor[1], kClearColor[2], 1.0f),
                ci.width, ci.height);
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
            "[shadowstats] cached {} lv0 tris {} px {} hw {} pend {} | "
            "lv1 tris {} px {} pend {} | lv2 tris {} px {} pend {}",
            shadowKey.valid ? 1 : 0, shadowChainStats[0][11],
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
    if (chainReady) chain.onShutdown(*vgBackend);
  }
  // Reverse registration order: overlay context, mesh pool (all GPU
  // meshes), renderer pipeline, shader manager, backend, window.
  engine.shutdown();
  kenshin::log::shutdown();
  return 0;
}
