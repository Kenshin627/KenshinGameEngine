#include "kenshin/core/Window.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <cstring>

#include "kenshin/core/Input.h"
#include "kenshin/core/Log.h"

KENSHIN_NS_BEGIN

namespace core {

namespace {

SDL_Window* sdl(void* w) { return static_cast<SDL_Window*>(w); }

MouseButton mapButton(Uint8 sdlButton) {
  switch (sdlButton) {
    case SDL_BUTTON_LEFT:
      return MouseButton::Left;
    case SDL_BUTTON_MIDDLE:
      return MouseButton::Middle;
    case SDL_BUTTON_RIGHT:
      return MouseButton::Right;
    case SDL_BUTTON_X1:
      return MouseButton::X1;
    default:
      return MouseButton::X2;
  }
}

}  // namespace

Window::~Window() { destroy(); }

bool Window::create(const WindowDesc& desc) {
  if (!SDL_InitSubSystem(SDL_INIT_VIDEO)) {
    KENSHIN_ERROR("SDL video init failed: {}", SDL_GetError());
    return false;
  }
  m_videoInited = true;
  SDL_WindowFlags flags = SDL_WINDOW_VULKAN | SDL_WINDOW_HIGH_PIXEL_DENSITY;
  if (desc.resizable) flags |= SDL_WINDOW_RESIZABLE;
  if (desc.hidden) flags |= SDL_WINDOW_HIDDEN;
  m_window = SDL_CreateWindow(desc.title.c_str(), static_cast<int>(desc.width),
                              static_cast<int>(desc.height), flags);
  if (m_window == nullptr) {
    KENSHIN_ERROR("SDL_CreateWindow failed: {}", SDL_GetError());
    SDL_QuitSubSystem(SDL_INIT_VIDEO);
    m_videoInited = false;
    return false;
  }
  m_quitPosted = false;
  refreshPixelSize();
  KENSHIN_INFO("Window created {}x{}", m_width, m_height);
  return true;
}

void Window::destroy() {
  if (m_window != nullptr) {
    SDL_DestroyWindow(sdl(m_window));
    m_window = nullptr;
    m_width = m_height = 0;
  }
  if (m_videoInited) {
    SDL_QuitSubSystem(SDL_INIT_VIDEO);
    m_videoInited = false;
  }
}

void Window::refreshPixelSize() {
  int w = 0, h = 0;
  if (m_window != nullptr) SDL_GetWindowSizeInPixels(sdl(m_window), &w, &h);
  m_width = static_cast<u32>(w > 0 ? w : 0);
  m_height = static_cast<u32>(h > 0 ? h : 0);
}

bool Window::pumpEvents() {
  if (m_window == nullptr) return false;
  const SDL_WindowID id = SDL_GetWindowID(sdl(m_window));
  SDL_Event e;
  while (SDL_PollEvent(&e)) {
    if (onRawEvent) onRawEvent(&e);
    Input& input = Input::get();
    switch (e.type) {
      case SDL_EVENT_QUIT:
        handleClose();
        break;
      case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
        if (e.window.windowID == id) handleClose();
        break;
      case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
        if (e.window.windowID == id) {
          const u32 w = static_cast<u32>(e.window.data1);
          const u32 h = static_cast<u32>(e.window.data2);
          if (w != m_width || h != m_height) {
            m_width = w;
            m_height = h;
            if (onResize && w != 0 && h != 0) onResize(w, h);
          }
        }
        break;
      case SDL_EVENT_WINDOW_FOCUS_LOST:
        // Keys released while another window has focus never report
        // their key-up here.
        if (e.window.windowID == id) input.releaseAll();
        break;
      case SDL_EVENT_KEY_DOWN:
      case SDL_EVENT_KEY_UP: {
        if (e.key.repeat) break;
        // SDL scancodes ARE USB HID usage IDs - core::Key's values.
        const Key key = static_cast<Key>(e.key.scancode);
        const bool down = e.type == SDL_EVENT_KEY_DOWN;
        input.onKey(key, down);
        if (onKey) onKey(key, down);
        break;
      }
      case SDL_EVENT_MOUSE_MOTION: {
        const int x = static_cast<int>(e.motion.x);
        const int y = static_cast<int>(e.motion.y);
        input.onMouseMove(x, y);
        if (onMouseMove) onMouseMove(x, y);
        break;
      }
      case SDL_EVENT_MOUSE_BUTTON_DOWN:
      case SDL_EVENT_MOUSE_BUTTON_UP: {
        const MouseButton b = mapButton(e.button.button);
        const bool down = e.type == SDL_EVENT_MOUSE_BUTTON_DOWN;
        input.onMouseButton(b, down);
        if (onMouseButton)
          onMouseButton(b, down, static_cast<int>(e.button.x),
                        static_cast<int>(e.button.y));
        break;
      }
      case SDL_EVENT_MOUSE_WHEEL: {
        float delta = e.wheel.y;
        if (e.wheel.direction == SDL_MOUSEWHEEL_FLIPPED) delta = -delta;
        input.onWheel(delta);
        if (onWheel) onWheel(delta);
        break;
      }
      default:
        break;
    }
  }
  return m_window != nullptr && !m_quitPosted;
}

bool Window::hasNoDrawableArea() const {
  if (m_window == nullptr) return true;
  if (SDL_GetWindowFlags(sdl(m_window)) & SDL_WINDOW_MINIMIZED) return true;
  int w = 0, h = 0;
  SDL_GetWindowSizeInPixels(sdl(m_window), &w, &h);
  return w <= 0 || h <= 0;
}

void Window::requestClose() { handleClose(); }

void Window::handleClose() {
  if (m_quitPosted) return;
  m_quitPosted = true;
  if (onClose) onClose();
}

std::vector<std::string> Window::vulkanInstanceExtensions() const {
  std::vector<std::string> out;
  Uint32 count = 0;
  const char* const* names = SDL_Vulkan_GetInstanceExtensions(&count);
  if (names == nullptr) {
    KENSHIN_ERROR("SDL_Vulkan_GetInstanceExtensions failed: {}",
                  SDL_GetError());
    return out;
  }
  for (Uint32 i = 0; i < count; ++i) out.emplace_back(names[i]);
  return out;
}

u64 Window::createVulkanSurface(void* vkInstance) const {
  if (m_window == nullptr || vkInstance == nullptr) return 0;
  VkSurfaceKHR surface = 0;  // SDL_vulkan.h declares no VK_NULL_HANDLE

  if (!SDL_Vulkan_CreateSurface(sdl(m_window),
                                static_cast<VkInstance>(vkInstance), nullptr,
                                &surface)) {
    KENSHIN_ERROR("SDL_Vulkan_CreateSurface failed: {}", SDL_GetError());
    return 0;
  }
  // Non-dispatchable handle: a pointer on 64-bit builds, a uint64_t on
  // 32-bit ones - copy the bits either way.
  u64 handle = 0;
  std::memcpy(&handle, &surface, sizeof(surface));
  return handle;
}

}  // namespace core

KENSHIN_NS_END
