#include "kenshin/DebugOverlay.h"

#include "kenshin/Engine.h"
#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Window.h"
#include "kenshin/scene/Camera.h"

#include "kenshin/platform/Paths.h"

#pragma warning(push, 0)
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_internal.h>  // DockBuilder (first-run layout)
#pragma warning(pop)

#include <SDL3/SDL_events.h>

#include <filesystem>

KENSHIN_NS_BEGIN

bool DebugOverlay::onInit(Engine& engine) {
  return init(&engine.get<backend::IRenderBackend>(),
              engine.get<core::Window>());
}

bool DebugOverlay::init(backend::IRenderBackend* backend,
                        core::Window& window) {
  m_backend = backend;
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  // Persist the layout next to the exe (default is the CWD, which is
  // whatever directory the app was launched from).
  {
    static std::string iniPath =
        (platform::executableDir() / "imgui.ini").string();
    io.IniFilename = iniPath.c_str();
  }
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  ImGui::StyleColorsDark();
  // Seed the viewport size with the window's until the first panel
  // layout lands (the app sizes the render target from it).
  m_viewportSize = glm::vec2(static_cast<f32>(window.width()),
                             static_cast<f32>(window.height()));
  // Platform half (SDL3): input/display events in, cursor/IME out. The
  // window still sees every event (onRawEvent only observes).
  if (!ImGui_ImplSDL3_InitForVulkan(
          static_cast<SDL_Window*>(window.nativeHandle()))) {
    ImGui::DestroyContext();
    return false;
  }
  m_platformInited = true;
  window.onRawEvent = [](const void* e) {
    ImGui_ImplSDL3_ProcessEvent(static_cast<const SDL_Event*>(e));
  };
  if (!backend->initImGuiBackend(window.width(), window.height(),
                                 [] { ImGui_ImplSDL3_NewFrame(); })) {
    ImGui_ImplSDL3_Shutdown();
    m_platformInited = false;
    ImGui::DestroyContext();
    return false;
  }
  return true;
}

void DebugOverlay::shutdown() {
  if (m_platformInited) {
    ImGui_ImplSDL3_Shutdown();
    m_platformInited = false;
  }
  ImGui::DestroyContext();
}

void DebugOverlay::draw(void* viewportTexture, core::FrameStats<240>& stats,
                        scene::Camera& camera,
                        scene::CameraController& controller) {
  // Full-window dockspace host (the ImGui demo's manual host-window
  // pattern, not DockSpaceOverViewport): every panel - viewport
  // included - can be dragged out, docked, split; layout persists.
  // The ID is derived from a fixed seed so the first-run DockBuilder
  // pass below can address the node BEFORE the DockSpace submits.
  const ImGuiViewport* vp = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(vp->WorkPos);
  ImGui::SetNextWindowSize(vp->WorkSize);
  ImGui::SetNextWindowViewport(vp->ID);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::Begin("KENSHIN Dock Host", nullptr,
               ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
                   ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                   ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoBringToFrontOnFocus |
                   ImGuiWindowFlags_NoNavFocus);
  ImGui::PopStyleVar(3);
  const ImGuiID dockId = ImGui::GetID("KENSHINDockSpace");
  // First-run default layout (the app checked for imgui.ini BEFORE
  // ImGui loaded it - a saved arrangement always wins): without this a
  // fresh checkout floats every panel and the viewport lands in a
  // small corner window that reads as "nothing renders". Built BEFORE
  // the DockSpace submits (the demo's ordering).
  if (m_firstRun && !m_dockLayoutBuilt) {
    m_dockLayoutBuilt = true;
    ImGui::DockBuilderRemoveNode(dockId);
    ImGui::DockBuilderAddNode(dockId, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockId, vp->WorkSize);
    ImGuiID left = 0, main = 0;
    ImGui::DockBuilderSplitNode(dockId, ImGuiDir_Left, 0.26f, &left, &main);
    ImGui::DockBuilderDockWindow("Frame Stats", left);
    ImGui::DockBuilderDockWindow("Camera", left);
    ImGui::DockBuilderDockWindow("Virtual Geometry", left);
    ImGui::DockBuilderDockWindow("Viewport", main);
    ImGui::DockBuilderFinish(dockId);
  }
  ImGui::DockSpace(dockId, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);
  ImGui::End();

  drawViewportPanel(viewportTexture);
  drawStatsPanel(stats);
  drawCameraPanel(camera, controller);
  if (m_extraPanel) m_extraPanel();
}

void DebugOverlay::drawViewportPanel(void* viewportTexture) {
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  // Size seed (ignored when docked - the dock node owns size then).
  // Without it a first-ever frame renders zero content and the window
  // auto-fits collapsed to a 32x13 title bar; the first-run layout
  // then docks that collapsed window and the 3D view "disappears".
  ImGui::SetNextWindowSize(ImVec2(1152.0f, 830.0f), ImGuiCond_FirstUseEver);
  ImGui::Begin("Viewport");
  const ImVec2 avail = ImGui::GetContentRegionAvail();
  if (avail.x >= 1.0f && avail.y >= 1.0f)
    m_viewportSize = glm::vec2(avail.x, avail.y);
  if (viewportTexture && avail.x >= 1.0f && avail.y >= 1.0f)
    ImGui::Image(ImTextureRef(viewportTexture), avail);
  m_viewportHovered = ImGui::IsItemHovered();
  ImGui::End();
  ImGui::PopStyleVar();
}

void DebugOverlay::drawStatsPanel(core::FrameStats<240>& stats) {
  ImGui::SetNextWindowPos(ImVec2(740.0f, 60.0f), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(340.0f, 220.0f), ImGuiCond_FirstUseEver);
  ImGui::Begin("Frame Stats");
  ImGui::Text("FPS %.1f", stats.fps());
  ImGui::Text("avg  %.3f ms", stats.averageMs());
  ImGui::Text("std  %.3f ms", stats.stddevMs());
  ImGui::Text("max  %.3f ms", stats.maxMs());
  ImGui::Text("samples %zu", stats.count());
  if (m_backend) {
    const auto gpu = m_backend->gpuTimings();
    ImGui::Text("gpu  %.3f ms (opaque %.3f)", gpu.frameMs, gpu.opaqueMs);
  }
  ImGui::Separator();
  static float intervals[240] = {};
  kenshin::usize count = 0;
  stats.copyTo(intervals, count);
  if (count > 1)
    ImGui::PlotLines("##intervals", intervals, static_cast<int>(count), 0,
                     nullptr, 0.0f, 50.0f, ImVec2(0, 60));
  ImGui::End();
}

void DebugOverlay::drawCameraPanel(scene::Camera& camera,
                                   scene::CameraController& controller) {
  ImGui::SetNextWindowPos(ImVec2(60.0f, 580.0f), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(380.0f, 300.0f), ImGuiCond_FirstUseEver);
  ImGui::Begin("Camera");
  // Sliders write through setup(): Camera state is private and the
  // controller is its only entry point.
  float yaw = controller.yawDeg();
  float pitch = controller.pitchDeg();
  float dist = controller.distance();
  glm::vec3 target = controller.target();
  if (ImGui::SliderFloat("yaw", &yaw, -180.0f, 180.0f, "%.1f deg"))
    controller.setup(yaw, pitch, dist, target, 1.0f);
  if (ImGui::SliderFloat("pitch", &pitch, -85.0f, 85.0f, "%.1f deg"))
    controller.setup(yaw, pitch, dist, target, 1.0f);
  if (ImGui::SliderFloat("distance", &dist, 0.1f, 200.0f, "%.2f m"))
    controller.setup(yaw, pitch, dist, target, 1.0f);
  if (ImGui::DragFloat3("target", &target.x, 0.05f))
    controller.setup(yaw, pitch, dist, target, 1.0f);
  ImGui::SliderFloat("fov", &controller.fovYDeg, 20.0f, 120.0f, "%.0f deg");
  const glm::vec3 pos = camera.position();
  ImGui::Text("pos (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
  ImGui::Text("(LMB drag rotates, wheel zooms, WASD/QE pans)");
  ImGui::End();
}

scene::ControllerInput DebugOverlay::controllerInput(f32 dt) const {
  ImGuiIO& io = ImGui::GetIO();
  scene::ControllerInput in;
  in.deltaTime = dt > 0.0f ? dt : 1.0f / 120.0f;
  in.rotateActive =
      m_viewportHovered && ImGui::IsMouseDragging(ImGuiMouseButton_Left);
  if (m_viewportHovered) {
    in.mouseDeltaX = io.MouseDelta.x;
    in.mouseDeltaY = io.MouseDelta.y;
    in.wheel = io.MouseWheel;
  }
  in.moveF = ImGui::IsKeyDown(ImGuiKey_W);
  in.moveB = ImGui::IsKeyDown(ImGuiKey_S);
  in.moveL = ImGui::IsKeyDown(ImGuiKey_A);
  in.moveR = ImGui::IsKeyDown(ImGuiKey_D);
  in.moveUp = ImGui::IsKeyDown(ImGuiKey_E);
  in.moveDown = ImGui::IsKeyDown(ImGuiKey_Q);
  in.fast = io.KeyShift;
  return in;
}

KENSHIN_NS_END
