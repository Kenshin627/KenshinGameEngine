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
//   KENSHINDemo --kgeo path.kgeo --baseline    the CPU-cut LOD preview
//                                               (pre-M3.4 default; A/B
//                                               reference until the cut
//                                               path retires)
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
#include "kenshin/manager/MeshPool.h"
#include "kenshin/manager/ShaderManager.h"
#include "kenshin/renderer/ClusterCullPass.h"
#include "kenshin/renderer/MaterialDecodePass.h"
#include "kenshin/renderer/Renderer.h"
#include "kenshin/renderer/SoftRasterPass.h"
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
  fflush(stderr);
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
  std::string screenshotUi;  // full window incl. ImGui
  bool virtualPath = false;  // accepted no-op since M3.4: kgeo defaults
                             // to the virtual chain (--baseline opts out)
  bool baseline = false;     // CPU-cut LOD preview (the pre-M3.4 default)
  bool checker = false;      // UV checkerboard debug view (decode path)
  bool noOcclusion = false;  // A/B: disable the occlusion cull chain
  bool noAsync = false;      // M4.4 A/B: chain onto the graphics queue
  bool multi = false;        // S4.6 acceptance: near/mid/far instances
  bool dense = false;        // M3.4 acceptance: front-back occluders
  int grid = 0;              // --grid N: N-instance occlusion carpet
                             // (1..32; the VisBuffer payload's 5b)
  bool wall = false;         // --wall: big occluder + hidden fleet
  // --cam: yaw,pitch,dist[,target xyz] in degrees/meters; nullopt = the
  // controller's interactive defaults.
  bool hasCam = false;
  float camYaw = 0, camPitch = 0, camDist = 0;
  float camTarget[3] = {0, 0, 0};
  int viewMode = -1;  // --view; -1 = leave the panel state at Final
};

Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--no-validation") == 0)
      args.validation = false;
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
      args.baseline = true;
    else if (std::strcmp(argv[i], "--checker") == 0)
      args.checker = true;
    else if (std::strcmp(argv[i], "--no-occ") == 0)
      args.noOcclusion = true;
    else if (std::strcmp(argv[i], "--no-async") == 0)
      args.noAsync = true;
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
  int viewMode = 0;  // DecodeView as int (0..4)
  kenshin::u32 visibleClusters = 0, cutTris = 0, pxWrites = 0;
  kenshin::u32 occCulled = 0;
  kenshin::u32 revived = 0;
  kenshin::u32 hwClusters = 0;
  kenshin::u32 workItems = 0, tooCoarse = 0, srcTooCoarse = 0;
  bool occlusion = true;
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
    if (clusterHist.size() > 240) {
      clusterHist.erase(clusterHist.begin());
      triHist.erase(triHist.begin());
      coverHist.erase(coverHist.begin());
      occHist.erase(occHist.begin());
      reviveHist.erase(reviveHist.begin());
      hwHist.erase(hwHist.begin());
    }
  }
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
  auto& overlay = engine.registerSubsystem<kenshin::DebugOverlay>();

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
  kenshin::renderer::DecodeParams decodeParams;
  VgDebugState vgDebug;
  if (args.viewMode >= 0 && args.viewMode <= 5)
    vgDebug.viewMode = args.viewMode;
  else if (args.checker)
    vgDebug.viewMode = 4;  // checker = DecodeView::Checker

  // ---- .kgeo LOD preview (S3.4) ---------------------------------------
  demo::KgeoPreview kgeoPreview;
  bool kgeoMode = false;
  if (!args.kgeo.empty()) {
    std::string kgeoLog;
    if (demo::loadKgeo(args.kgeo, kgeoPreview, kgeoLog)) {
      kgeoMode = true;
      kgeoPreview.pixelThreshold = args.lodThreshold;

      // S4.1: the dag also goes to the GPU-resident virtual geometry
      // pools (cluster/group tables + shared vertex/index pools), CRC
      // round-trip verified at upload. The S4.2+ culling passes consume
      // these; rendering still runs the CPU-cut path until S4.5.
      auto& vgScene =
          engine.registerSubsystem<kenshin::renderer::VirtualGeometryScene>();
      const kenshin::u32 vgMeshId = vgScene.addMesh(kgeoPreview.dag);
      vgScene.flushPools();
      vgScene.setInstances(
          {kenshin::renderer::VgInstance{glm::mat4(1.0f), vgMeshId}});
      // M3.4 retirement step 1: kgeo defaults to the virtual chain;
      // --baseline opts into the CPU-cut LOD preview (the A/B
      // reference until the cut path is deleted).
      if (!args.baseline) {
        const kenshin::renderer::VgMeshRange& mesh = vgScene.meshes()[vgMeshId];
        const auto makeInst = [&](const glm::mat4& model) {
          kenshin::renderer::CullGpuInstance inst;
          inst.model = model;
          inst.meshId = vgMeshId;
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
              glm::vec4(glm::vec3(model * glm::vec4(mesh.boundsSphere.x,
                                                    mesh.boundsSphere.y,
                                                    mesh.boundsSphere.z, 1.0f)),
                        mesh.boundsSphere.w * scale);
          return inst;
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
          // the cull has real work to save. 32 max (VisBuffer payload
          // packs 5 instance bits).
          const int n = std::clamp(args.grid, 1, 32);
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
          vgScene.setInstances(vgInsts);
        } else {
          chainInstances = {makeInst(glm::mat4(1.0f))};
          vgScene.setInstances(
              {kenshin::renderer::VgInstance{glm::mat4(1.0f), vgMeshId}});
        }
        for (kenshin::u32 i = 0; i < chainInstances.size(); ++i)
          chainMeshRanges.push_back({mesh.clusterBegin, mesh.clusterEnd});
      }
      overlay.setExtraPanel([&kgeoPreview, &vgDebug, &args] {
        demo::drawLodPanel(kgeoPreview);
        if (!args.baseline) {
          ImGui::Begin("Virtual Geometry");
          const char* names[] = {"final",     "LOD level", "error heat",
                                 "triangles", "checker",   "clusters"};
          for (int i = 0; i < 6; ++i)
            ImGui::RadioButton(names[i], &vgDebug.viewMode, i);
          ImGui::Separator();
          ImGui::Text(
              "clusters %u  tris %u  writes %u  cover %.1f%%\n"
              "occ-culled %u  revived %u  hw %u",
              vgDebug.visibleClusters, vgDebug.cutTris, vgDebug.pxWrites,
              vgDebug.coveragePct, vgDebug.occCulled, vgDebug.revived,
              vgDebug.hwClusters);
          ImGui::Checkbox("occlusion", &vgDebug.occlusion);
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
  const bool virtualPath = kgeoMode && !args.baseline;
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
  bool chainReady = false;
  bool decodeReady = false;  // pipeline is lazy: needs the swapchain format
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

    if (kgeoMode)
      demo::updateCut(kgeoPreview, meshPool, camera, controller,
                      static_cast<kenshin::u32>(vp.y));

    // Assemble this frame's draw stream (pure data; the renderer layer
    // consumes it between beginFrame and endFrame).
    const bool drawable = vp.x > 1.0f && vp.y > 1.0f;
    items.clear();
    if (!kgeoMode && drawable && !scene.empty()) {
      for (const auto& shape : scene)
        items.push_back({shape.handle,
                         glm::translate(glm::mat4(1.0f), shape.offset),
                         shape.kd});
    } else if (kgeoMode && !virtualPath && drawable &&
               kgeoPreview.meshHandle != kenshin::assets::kInvalidMeshHandle) {
      items.push_back({kgeoPreview.meshHandle, glm::mat4(1.0f),
                       glm::vec3(0.75f, 0.75f, 0.78f)});
    }

    const kenshin::renderer::FrameView view{
        camera.viewProjMatrix(), camera.position(),
        glm::vec3(kStudioLight[0], kStudioLight[1], kStudioLight[2])};
    renderer.submit(view, items);

    if (virtualPath && drawable) {
      // Lazy init (pools ready; vis buffers start 1x1, first execute
      // resizes).
      if (!chainReady) {
        chainReady = chain.init(
            *vgBackend, shaders, vgScene->positions4(), vgScene->indices(),
            vgScene->clusterData(), vgScene->groupData(), chainInstances,
            chainMeshRanges,
            kenshin::u32(chainInstances.size()) * vgScene->clusterCount(), 1,
            1);
        if (chainReady) KENSHIN_INFO("[vgchain] in-frame chain ready");
      }
      if (!decodeReady) decodeReady = decodePass.init(*vgBackend, shaders);
      if (chainReady && decodeReady) {
        // Same camera, same frame: the chain's inputs and the decode's
        // matrices cannot disagree anymore.
        kenshin::renderer::ChainInputs ci;
        ci.view = camera.viewMatrix();
        ci.proj = camera.projMatrix();
        ci.nearZ = camera.nearPlane();
        ci.camPos = camera.position();
        ci.pixelThreshold = kgeoPreview.pixelThreshold;
        ci.worldPerPixel = 2.0f *
                           std::tan(glm::radians(controller.fovYDeg) * 0.5f) /
                           static_cast<kenshin::f32>(vp.y);
        ci.width = static_cast<kenshin::u32>(vp.x);
        ci.height = static_cast<kenshin::u32>(vp.y);
        kenshin::renderer::extractFrustumPlanes(camera.viewProjMatrix(),
                                                ci.planes);
        ci.occlusionEnabled = vgDebug.occlusion && !args.noOcclusion;
        // Panel stats: NON-BLOCKING mapped snapshot, taken BEFORE this
        // frame's execute. At this point the CURRENT slot's counters
        // still hold the frame-before-last's final values, and
        // beginFrame's pacing wait has PROVEN that frame complete -
        // reading after execute (the old site) raced the GPU's own
        // writes to the not-yet-finished previous submission (a
        // "submitted" frame is not an "executed" frame: single-queue
        // submissions may overlap), which showed up as torn counters
        // - intermittent zero clusters, wildly varying write counts.
        {
          kenshin::u32 cs[13];
          if (chainReady && chain.readStatsMapped(*vgBackend, cs)) {
            vgDebug.visibleClusters = cs[2];
            vgDebug.cutTris = cs[11];
            vgDebug.pxWrites = cs[5] + cs[10];  // main + post raster
            vgDebug.occCulled = cs[8];
            vgDebug.revived = cs[9];
            vgDebug.hwClusters = cs[12];
            vgDebug.workItems = cs[1];
            vgDebug.tooCoarse = cs[3];
            vgDebug.srcTooCoarse = cs[4];
            const kenshin::f32 cover =
                100.0f * static_cast<kenshin::f32>(cs[5] + cs[10]) /
                (ci.width * ci.height);
            vgDebug.coveragePct = cover;
            vgDebug.pushHistory();
          }
        }

        chain.execute(*vgBackend, ci);

        decodeParams.camPos = camera.position();
        decodeParams.view =
            static_cast<kenshin::renderer::DecodeView>(vgDebug.viewMode);
        decodePass.draw(*vgBackend, camera.viewProjMatrix(), *vgScene,
                        decodeParams, ci.width, ci.height,
                        chain.visBufferCurrent(*vgBackend));
      }
    }

    // Full-frame capture must be flagged before endFrame records it.
    const bool exiting = isLastFrame || !window.isOpen();
    if (!args.screenshotUi.empty() && exiting) renderer.requestFrameCapture();

    renderer.endFrame();
    engine.endFrame();

    stats.push(dt * 1000.0f);
    if (frame > 0 && frame % 60 == 0) {
      KENSHIN_INFO("[pacing] avg={:.3f}ms std={:.3f}ms max={:.3f}ms fps={:.1f}",
                   stats.averageMs(), stats.stddevMs(), stats.maxMs(),
                   stats.fps());
      if (chainReady)
        KENSHIN_INFO(
            "[vgstats] work {} clusters {} tris {} writes {} occ {} "
            "revived {} hw {} tooCoarse {} srcTooCoarse {} gpuMs {}",
            vgDebug.workItems, vgDebug.visibleClusters, vgDebug.cutTris,
            vgDebug.pxWrites, vgDebug.occCulled, vgDebug.revived,
            vgDebug.hwClusters, vgDebug.tooCoarse, vgDebug.srcTooCoarse,
            vgBackend->gpuTimings().frameMs);
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
  if (virtualPath) {
    decodePass.onShutdown(*vgBackend);
    if (chainReady) chain.onShutdown(*vgBackend);
  }
  // Reverse registration order: overlay context, mesh pool (all GPU
  // meshes), renderer pipeline, shader manager, backend, window.
  engine.shutdown();
  kenshin::log::shutdown();
  return 0;
}
