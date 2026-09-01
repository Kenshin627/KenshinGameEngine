#pragma once

// ── DebugOverlay (root-level debug UI host) ──────────────────────
// Owns the ImGui context for a windowed app: docking workspace, the
// viewport / frame-stats / camera panels, and camera input routing.
// App-specific panels (e.g. the LOD preview) plug in via setExtraPanel.
// The Vulkan-side ImGui pass lives in the backend (initImGuiBackend +
// NewFrame in beginFrame); this class is the context + panels layer.

#include <functional>
#include <glm/glm.hpp>

#include "kenshin/Common.h"
#include "kenshin/core/FrameStats.h"
#include "kenshin/scene/CameraController.h"

KENSHIN_NS_BEGIN

class Engine;
namespace core {
class Window;
}
namespace backend {
class IRenderBackend;
}

class DebugOverlay {
 public:
  /// Subsystem hook: resolve backend + window from the engine registry
  /// and init — registering the overlay initializes it.
  bool onInit(Engine& engine);

  /// ImGui context + ini file next to the exe + docking config + Win32
  /// wndproc hookup + backend ImGui init. Call once, after the backend
  /// and the window exist. Returns false if the backend init fails
  /// (context is torn down again).
  bool init(backend::IRenderBackend* backend, core::Window& window);
  void shutdown();

  /// Draw this frame's panels. Call between beginFrame and endFrame,
  /// and before controllerInput() (hover state must be fresh).
  void draw(void* viewportTexture, core::FrameStats<240>& stats,
            scene::Camera& camera, scene::CameraController& controller);

  /// Extra panel(s) drawn last each frame; capture what they need.
  void setExtraPanel(std::function<void()> panel) {
    m_extraPanel = std::move(panel);
  }

  /// Camera input assembled from ImGui, gated by viewport hover.
  scene::ControllerInput controllerInput(f32 dt) const;

  const glm::vec2& viewportSize() const { return m_viewportSize; }
  bool viewportHovered() const { return m_viewportHovered; }

 private:
  void drawViewportPanel(void* viewportTexture);
  void drawStatsPanel(core::FrameStats<240>& stats);
  void drawCameraPanel(scene::Camera& camera,
                       scene::CameraController& controller);

  std::function<void()> m_extraPanel;
  glm::vec2 m_viewportSize{1280.0f, 720.0f};
  bool m_viewportHovered = false;
};

KENSHIN_NS_END
