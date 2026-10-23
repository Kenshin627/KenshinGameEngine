// KENSHINDemo entry: parse args, build the demo scene, boot the engine
// (subsystem registry), drive the frame loop. The frame itself is the
// engine's renderer::SceneRenderer; this file only wires the demo's
// inputs (camera, light rig, panel, command line) into it.
// Usage:
//   KENSHINDemo                                  windowed loop (ESC quits)
//   KENSHINDemo --frames N                       auto-exit after N frames
//   KENSHINDemo --no-validation                  disable the validation layer
//   KENSHINDemo --size WxH                       client size (default 1600x900)
//   KENSHINDemo --screenshot path.bmp            capture the last rendered frame
//   KENSHINDemo --kgeo path.kgeo                 virtual geometry scene
//   KENSHINDemo --cam yaw,pitch,dist[,tx,ty,tz]  fixed camera pose
//   KENSHINDemo --view N                         decode view 0..6 (0 final)
// The full flag list lives in AppArgs.h.

#include <glm/glm.hpp>

#include <chrono>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "AppArgs.h"
#include "DemoScene.h"
#include "KgeoScene.h"
#include "VgPanel.h"
#include "kenshin/DebugOverlay.h"
#include "kenshin/Engine.h"
#include "kenshin/Version.h"
#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/FrameStats.h"
#include "kenshin/core/Input.h"
#include "kenshin/core/Log.h"
#include "kenshin/core/Timer.h"
#include "kenshin/core/Window.h"
#include "kenshin/manager/MeshPool.h"
#include "kenshin/manager/ShaderManager.h"
#include "kenshin/manager/TextureManager.h"
#include "kenshin/platform/CrashHandler.h"
#include "kenshin/platform/Paths.h"
#include "kenshin/renderer/Environment.h"
#include "kenshin/renderer/Renderer.h"
#include "kenshin/renderer/SceneRenderer.h"
#include "kenshin/renderer/VirtualGeometryScene.h"
#include "kenshin/scene/Camera.h"
#include "kenshin/scene/CameraController.h"
#include "kenshin/util/BmpWriter.h"

namespace kr = kenshin::renderer;

namespace {

/// IBL environment, first hit wins: --env <path>, an env.hdr next to the
/// loaded asset, an env.hdr in the working directory; the procedural
/// studio probe is the fallback.
kr::EnvironmentProbe loadEnvironment(const demo::Args& args,
                                     kenshin::backend::IRenderBackend& backend) {
  std::filesystem::path envPath;
  if (!args.envPath.empty()) {
    envPath = args.envPath;
    if (!std::filesystem::exists(envPath))
      KENSHIN_WARN("[ibl] --env {} not found (studio probe)", envPath.string());
  } else {
    envPath = std::filesystem::path(args.kgeo).parent_path() / "env.hdr";
    if (!std::filesystem::exists(envPath)) envPath = "env.hdr";
  }
  kr::EnvironmentProbe probe;
  if (kr::loadHdrEnvironment(backend, envPath, probe)) {
    KENSHIN_INFO("[ibl] env.hdr in effect (probe skipped)");
    return probe;
  }
  return kr::createStudioProbe(backend);
}

/// The demo light rig: the panel's directional key + two static point
/// accents (deterministic - the render regression depends on it).
std::vector<kr::GpuLightC> buildLights(const demo::VgDebugState& vg) {
  std::vector<kr::GpuLightC> lights;
  kr::GpuLightC dir;
  dir.a = glm::vec4(glm::normalize(vg.lightDir), 0.0f);
  dir.b = glm::vec4(vg.lightColor * vg.lightIntensity, 0.0f);
  lights.push_back(dir);
  for (int p = 0; p < 2; ++p) {
    kr::GpuLightC pt;
    pt.a = glm::vec4(demo::kPointLights[p], 1.0f);
    pt.b = glm::vec4(demo::kPointColors[p] * vg.pointIntensity,
                     demo::kPointRange);
    lights.push_back(pt);
  }
  return lights;
}

/// Seed the panel model from the command line.
void seedPanelFromArgs(const demo::Args& args, demo::VgDebugState& vg) {
  if (args.imposterDebug) vg.imposterDebug = true;
  if (args.viewMode >= 0 && args.viewMode <= 6)
    vg.viewMode = args.viewMode;
  else if (args.checker)
    vg.viewMode = 4;  // checker = DecodeView::Checker
  vg.pixelThreshold = args.lodThreshold;
  if (args.hwThresholdPx > 0.0f) vg.hwThresholdPx = args.hwThresholdPx;
  // An explicit --exposure pins the manual pipeline (existing command
  // lines keep reproducing their images).
  if (args.exposure > 0.0f) {
    vg.exposure = args.exposure;
    vg.autoExposure = false;
  }
  if (args.noAutoExposure) vg.autoExposure = false;
  if (args.exposureBias > -1e8f) vg.exposureSettings.biasEv = args.exposureBias;
  if (args.envIntensity >= 0.0f) vg.envIntensity = args.envIntensity;
  if (args.noBloom) vg.bloom = false;
  if (args.noSky) vg.sky = false;
  if (args.noEmissivePass) vg.emissivePassThrough = false;
  if (args.skyBlur >= 0.0f) vg.skyBlur = args.skyBlur;
  if (args.taaSharpen >= 0.0f) vg.taaSharpen = args.taaSharpen;
  if (args.bloomStrength >= 0.0f) vg.bloomStrength = args.bloomStrength;
  if (args.emissiveStrength >= 0.0f) vg.emissiveStrength = args.emissiveStrength;
}

}  // namespace

int main(int argc, char** argv) {
  kenshin::platform::installCrashHandler();
  const demo::Args args = demo::parseArgs(argc, argv);

  kenshin::log::init();
  KENSHIN_INFO("{}", kenshin::getBanner());

  // Engine facade: global subsystem registry (type-indexed get<T>()).
  kenshin::Engine& engine = kenshin::Engine::get();
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

  // Shader service first: the renderer's pipeline compiles on init.
  auto& shaders = engine.registerSubsystem<kenshin::manager::ShaderManager>();
  shaders.init(demo::findShaderRoot());
  // Renderer (creates + registers the backend, int64-atomics gate),
  // MeshPool (GPU mesh lifetime), TextureManager (material textures;
  // LIFO shutdown destroys them before the backend), DebugOverlay (UI).
  auto& renderer = engine.registerSubsystem<kr::Renderer>(
      kr::Renderer::Config{args.validation});
  auto& meshPool = engine.registerSubsystem<kenshin::manager::MeshPool>();
  auto& textureManager =
      engine.registerSubsystem<kenshin::manager::TextureManager>();
  auto& overlay = engine.registerSubsystem<kenshin::DebugOverlay>();
  // First-run docking: no imgui.ini next to the EXE -> default layout.
  if (!std::filesystem::exists(kenshin::platform::executableDir() / "imgui.ini"))
    overlay.markFirstRun();

  // ---- baseline demo scene (no --kgeo): one instance of every shape.
  auto scene = args.kgeo.empty() ? demo::buildScene(args.mesh)
                                 : std::vector<demo::SceneShape>{};
  demo::uploadScene(scene, meshPool);

  // ---- virtual-geometry scene (--kgeo)
  demo::VgDebugState vgDebug;
  seedPanelFromArgs(args, vgDebug);
  demo::KgeoScene kgeo;
  bool kgeoMode = false;
  if (!args.kgeo.empty()) {
    switch (demo::buildKgeoScene(args, engine, textureManager, kgeo)) {
      case demo::KgeoSceneResult::Fatal:
        return 1;
      case demo::KgeoSceneResult::Ok:
        kgeoMode = true;
        overlay.setExtraPanel([&] {
          demo::drawVgPanel(vgDebug, kgeo.preview, kgeo.streamer.get());
        });
        break;
      case demo::KgeoSceneResult::LoadFailed:
        break;
    }
  }
  kr::SceneRenderer sceneRenderer;
  kr::VirtualGeometryScene* vgScene = nullptr;
  if (kgeoMode) {
    auto& backend = engine.get<kenshin::backend::IRenderBackend>();
    vgScene = kgeo.desc.scene;
    const kr::EnvironmentProbe probe = loadEnvironment(args, backend);
    if (probe.hasSun) {
      // The HDRI's sun takes over the key light (its shadows then agree
      // with the reflections). A captured HDRI is in its own radiometric
      // scale: 1.0 = as shot; an explicit --env-intensity still wins.
      vgDebug.lightDir = probe.sunDir;
      vgDebug.lightColor = probe.sunColor;
      vgDebug.lightIntensity = probe.sunIntensity;
      if (args.envIntensity < 0.0f) vgDebug.envIntensity = 1.0f;
    }
    if (!sceneRenderer.setup(backend, shaders, std::move(kgeo.desc))) return 1;
    sceneRenderer.setEnvironment(probe);
    if (args.noAsync) backend.setAsyncChainEnabled(false);  // A/B
  }

  // --cam: a fixed initial pose (deterministic screenshots); otherwise
  // auto-frame the loaded asset (bounds center, ~2.8 radii).
  const kenshin::f32 aspect =
      static_cast<kenshin::f32>(args.width) / static_cast<kenshin::f32>(args.height);
  if (args.hasCam)
    controller.setup(args.camYaw, args.camPitch, args.camDist,
                     glm::vec3(args.camTarget[0], args.camTarget[1],
                               args.camTarget[2]),
                     aspect);
  else if (vgScene && !vgScene->meshes().empty() &&
           vgScene->meshes()[0].boundsSphere.w > 0.0f) {
    const glm::vec4& bs = vgScene->meshes()[0].boundsSphere;
    controller.setup(45.0f, 6.0f, bs.w * 2.8f, glm::vec3(bs), aspect);
  }

  kenshin::core::FrameStats<240> stats;
  kenshin::Timer frameTimer;
  bool cameraChecked = false;
  std::vector<kr::DrawItem> items;
  const auto writeScreenshot = [&renderer](const char* what,
                                           const std::string& path,
                                           bool fullFrame) {
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
    // Minimized / zero client area: nothing can be presented - idle.
    if (window.hasNoDrawableArea() && args.frames < 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(16));
      frameTimer.reset();
      continue;
    }
    auto& input = kenshin::core::Input::get();
    if (input.keyPressed(kenshin::core::Key::Escape)) {
      KENSHIN_INFO("ESC pressed -> close");
      window.requestClose();
    }
    input.nextFrame();

    const kenshin::f32 dt = static_cast<kenshin::f32>(frameTimer.elapsedSeconds());
    frameTimer.reset();
    engine.beginFrame(dt);

    const bool isLastFrame = args.frames >= 0 && frame + 1 >= args.frames;
    if (isLastFrame) {
      const int r = static_cast<int>(demo::kClearColor[0] * 255.0f + 0.5f);
      const int g = static_cast<int>(demo::kClearColor[1] * 255.0f + 0.5f);
      const int b = static_cast<int>(demo::kClearColor[2] * 255.0f + 0.5f);
      KENSHIN_INFO("frame {} expected color = ({},{},{})", frame, r, g, b);
    }

    const glm::vec2 vp = overlay.viewportSize();
    const kr::FrameStatus frameStatus = renderer.beginFrame(
        static_cast<kenshin::u32>(vp.x), static_cast<kenshin::u32>(vp.y),
        demo::kClearColor);
    if (frameStatus == kr::FrameStatus::Error) {
      KENSHIN_ERROR("beginFrame failed");
      break;
    }
    if (frameStatus == kr::FrameStatus::Skipped) {
      // Nothing to present (minimized between our check and the acquire):
      // no recording, no endFrame - try again next frame.
      std::this_thread::sleep_for(std::chrono::milliseconds(16));
      frameTimer.reset();
      continue;
    }

    overlay.draw(renderer.viewportTexture(), stats, camera, controller);
    controller.handleInput(camera, overlay.controllerInput(dt));
    controller.updateMatrices(camera, vp.x / glm::max(vp.y, 1.0f));
    if (!cameraChecked &&
        glm::abs(glm::dot(camera.forward(), camera.up())) < 1e-4f) {
      KENSHIN_INFO("[camera] view basis orthonormal: OK");
      cameraChecked = true;
    }

    // The baseline forward path's draw stream (pure data).
    const bool drawable = vp.x > 1.0f && vp.y > 1.0f;
    items.clear();
    if (!kgeoMode && drawable && !scene.empty())
      for (const auto& shape : scene)
        items.push_back({shape.handle,
                         glm::translate(glm::mat4(1.0f), shape.offset),
                         shape.kd});
    renderer.submit(
        kr::FrameView{camera.viewProjMatrix(), camera.position(),
                      glm::vec3(demo::kStudioLight[0], demo::kStudioLight[1],
                                demo::kStudioLight[2])},
        items);

    if (kgeoMode && drawable) {
      const std::vector<kr::GpuLightC> lights = buildLights(vgDebug);
      const kr::SceneRenderSettings settings =
          demo::resolveSettings(vgDebug, args);
      kr::SceneRenderer::FrameInputs fi;
      fi.camera = &camera;
      fi.fovYDeg = controller.fovYDeg;
      fi.width = static_cast<kenshin::u32>(vp.x);
      fi.height = static_cast<kenshin::u32>(vp.y);
      fi.frameIndex = static_cast<kenshin::u32>(frame);
      fi.dt = dt;
      fi.lights = lights;
      sceneRenderer.render(fi, settings);
      const kr::SceneRenderStats& st = sceneRenderer.stats();
      demo::absorbStats(vgDebug, st);
      if (isLastFrame && settings.autoExposure && st.exposureValid)
        KENSHIN_INFO("[exposure] scale {:.4f} target {:.4f} avg lum {:.5f}",
                     st.autoScale, st.autoTargetScale, st.autoAvgLum);
    }

    // Full-frame capture must be flagged before endFrame records it.
    const bool exiting = isLastFrame || !window.isOpen();
    if (!args.screenshotUi.empty() && exiting) renderer.requestFrameCapture();
    renderer.endFrame();
    if (kgeoMode) {
      sceneRenderer.afterEndFrame();
      vgDebug.missPages = sceneRenderer.stats().missPages;
    }
    engine.endFrame();

    stats.push(dt * 1000.0f);
    if (frame > 0 && frame % 60 == 0) {
      KENSHIN_INFO("[pacing] avg={:.3f}ms std={:.3f}ms max={:.3f}ms fps={:.1f}",
                   stats.averageMs(), stats.stddevMs(), stats.maxMs(),
                   stats.fps());
      if (kgeoMode) sceneRenderer.logStats();
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
  if (kgeo.streamer) kgeo.streamer->shutdown();
  if (kgeoMode) sceneRenderer.shutdown();
  // Reverse registration order: overlay context, mesh pool (all GPU
  // meshes), renderer pipeline, shader manager, backend, window.
  engine.shutdown();
  kenshin::log::shutdown();
  return 0;
}
