#pragma once

// ── Window — the platform window (SDL3) ─────────────────────────────
// One OS window + its event pump. Feeds core::Input with physical keys
// and the mouse, reports pixel-size changes (DPI-correct: width/height
// are framebuffer PIXELS), and hands the Vulkan backend exactly two
// things through Vulkan-free signatures: the instance extensions the
// window system needs and a surface for a created instance. Nothing
// above this class sees SDL or OS types.

#include <functional>
#include <string>
#include <vector>

#include "kenshin/Common.h"
#include "kenshin/core/Keys.h"

KENSHIN_NS_BEGIN

namespace core {

struct WindowDesc {
  std::string title = "KENSHIN";
  u32 width = 1600;   // logical size; the pixel size follows display scale
  u32 height = 900;
  bool resizable = true;
  bool hidden = false;  // never shown (tests, headless editor/agent runs)
};

class Window {
 public:
  std::function<void(u32 w, u32 h)> onResize;  // pixel size, never 0x0
  std::function<void()> onClose;
  std::function<void(Key key, bool down)> onKey;  // repeats filtered
  std::function<void(MouseButton button, bool down, int x, int y)>
      onMouseButton;
  std::function<void(int x, int y)> onMouseMove;
  std::function<void(float delta)> onWheel;  // positive = scroll up
  /// Every native event (an SDL_Event*) before the window handles it -
  /// the UI layer's tap (ImGui). Observing only: the window and Input
  /// still see the event.
  std::function<void(const void* nativeEvent)> onRawEvent;

  Window() = default;
  ~Window();

  Window(const Window&) = delete;
  Window& operator=(const Window&) = delete;

  bool create(const WindowDesc& desc);
  void destroy();
  bool isOpen() const { return m_window != nullptr; }
  /// Minimized or zero-area: nothing to present (skip the frame).
  bool hasNoDrawableArea() const;

  /// Drain pending events. Returns false once the window is closing.
  bool pumpEvents();

  void requestClose();

  u32 width() const { return m_width; }    // pixels
  u32 height() const { return m_height; }  // pixels

  /// The SDL_Window* (platform/UI glue only - e.g. the ImGui backend).
  void* nativeHandle() const { return m_window; }

  /// Vulkan instance extensions this window system needs to present
  /// (VK_KHR_surface + the platform's surface extension).
  std::vector<std::string> vulkanInstanceExtensions() const;
  /// Create a VkSurfaceKHR for `vkInstance` (a VkInstance). The caller
  /// owns the surface (vkDestroySurfaceKHR). Returns 0 on failure.
  u64 createVulkanSurface(void* vkInstance) const;

 private:
  void handleClose();
  void refreshPixelSize();

  void* m_window = nullptr;  // SDL_Window*
  u32 m_width = 0;
  u32 m_height = 0;
  bool m_quitPosted = false;
  bool m_videoInited = false;
};

}  // namespace core

KENSHIN_NS_END
