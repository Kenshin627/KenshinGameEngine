#include "kenshin/DebugOverlay.h"

#include "kenshin/Engine.h"
#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Window.h"
#include "kenshin/scene/Camera.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#pragma warning(push, 0)
#include <imgui.h>
#include <imgui_impl_win32.h>  // after Windows.h: backend expects HWND declared
#pragma warning(pop)

// imgui_impl_win32.h intentionally wraps the WndProcHandler declaration in
// '#if 0' (to avoid dragging <windows.h> into the header). Per its comment,
// forward declare it here — Windows.h is already included above.
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd,
                                                             UINT msg,
                                                             WPARAM wParam,
                                                             LPARAM lParam);

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
    wchar_t exePath[MAX_PATH]{};
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    static std::string iniPath =
        (std::filesystem::path(exePath).parent_path() / "imgui.ini").string();
    io.IniFilename = iniPath.c_str();
  }
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  ImGui::StyleColorsDark();
  // Seed the viewport size with the window's until the first panel
  // layout lands (the app sizes the render target from it).
  m_viewportSize = glm::vec2(static_cast<f32>(window.width()),
                             static_cast<f32>(window.height()));
  window.onRawMessage = [](void* hwnd, unsigned msg, kenshin::usize wparam,
                           kenshin::isize lparam) {
    return ImGui_ImplWin32_WndProcHandler(static_cast<HWND>(hwnd), msg,
                                          static_cast<WPARAM>(wparam),
                                          static_cast<LPARAM>(lparam)) != 0;
  };
  if (!backend->initImGuiBackend(window.nativeHandle(), window.width(),
                                 window.height())) {
    ImGui::DestroyContext();
    return false;
  }
  return true;
}

void DebugOverlay::shutdown() { ImGui::DestroyContext(); }

void DebugOverlay::draw(void* viewportTexture, core::FrameStats<240>& stats,
                        scene::Camera& camera,
                        scene::CameraController& controller) {
  // Full-window dockspace host: every panel (viewport included) can be
  // dragged out, docked, split; layout persists across sessions.
  ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(),
                               ImGuiDockNodeFlags_PassthruCentralNode);

  drawViewportPanel(viewportTexture);
  drawStatsPanel(stats);
  drawCameraPanel(camera, controller);
  if (m_extraPanel) m_extraPanel();
}

void DebugOverlay::drawViewportPanel(void* viewportTexture) {
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::SetNextWindowPos(ImVec2(60.0f, 60.0f), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(640.0f, 480.0f), ImGuiCond_FirstUseEver);
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
