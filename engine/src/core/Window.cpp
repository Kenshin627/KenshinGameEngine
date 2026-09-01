#include "kenshin/core/Window.h"

#include "kenshin/core/Input.h"
#include "kenshin/core/Log.h"

#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <Windowsx.h>

#include <cstdlib>

KENSHIN_NS_BEGIN

namespace core {

namespace {
constexpr const wchar_t* kClassName = L"KENSHINWindow";
bool s_classRegistered = false;

std::wstring toWide(const std::string& utf8) {
  if (utf8.empty()) return std::wstring();
  const int len = MultiByteToWideChar(
      CP_UTF8, 0, utf8.c_str(), static_cast<int>(utf8.size()), nullptr, 0);
  std::wstring wide(static_cast<size_t>(len), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), static_cast<int>(utf8.size()),
                      wide.data(), len);
  return wide;
}
}  // namespace

// ── Lifetime ────────────────────────────────────────────────

Window::~Window() { destroy(); }

bool Window::create(const WindowDesc& desc) {
  const HINSTANCE hInstance = GetModuleHandleW(nullptr);

  if (!s_classRegistered) {
    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = [](HWND hwnd, UINT msg, WPARAM wparam,
                        LPARAM lparam) -> LRESULT {
      if (msg == WM_NCCREATE) {
        const auto* cs = reinterpret_cast<CREATESTRUCTW*>(lparam);
        SetWindowLongPtrW(hwnd, GWLP_USERDATA,
                          reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
        return DefWindowProcW(hwnd, msg, wparam, lparam);
      }
      auto* self =
          reinterpret_cast<Window*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
      if (self) return self->handleMessage(hwnd, msg, wparam, lparam);
      return DefWindowProcW(hwnd, msg, wparam, lparam);
    };
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
    wc.lpszClassName = kClassName;
    if (!RegisterClassExW(&wc)) {
      KENSHIN_ERROR("RegisterClassExW failed: {}", GetLastError());
      return false;
    }
    s_classRegistered = true;
  }

  // Desired client size -> outer window size
  RECT rect{0, 0, static_cast<LONG>(desc.width),
            static_cast<LONG>(desc.height)};
  const DWORD style = WS_OVERLAPPEDWINDOW &
                      ~(desc.resizable ? 0 : (WS_THICKFRAME | WS_MAXIMIZEBOX));
  AdjustWindowRect(&rect, style, FALSE);

  m_hwnd = CreateWindowExW(0, kClassName, toWide(desc.title).c_str(), style,
                           CW_USEDEFAULT, CW_USEDEFAULT, rect.right - rect.left,
                           rect.bottom - rect.top, nullptr, nullptr, hInstance,
                           this);
  if (!m_hwnd) {
    KENSHIN_ERROR("CreateWindowExW failed: {}", GetLastError());
    return false;
  }

  m_width = desc.width;
  m_height = desc.height;
  ShowWindow(static_cast<HWND>(m_hwnd), SW_SHOW);
  UpdateWindow(static_cast<HWND>(m_hwnd));

  KENSHIN_INFO("Window created {}x{}", m_width, m_height);
  return true;
}

void Window::destroy() {
  if (m_hwnd) {
    DestroyWindow(static_cast<HWND>(m_hwnd));
    m_hwnd = nullptr;
    m_width = m_height = 0;
  }
}

bool Window::pumpEvents() {
  MSG msg{};
  while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
    if (msg.message == WM_QUIT) {
      m_quitPosted = true;
      break;
    }
    TranslateMessage(&msg);
    DispatchMessageW(&msg);
  }
  return m_hwnd != nullptr && !m_quitPosted;
}

void Window::requestClose() { handleClose(); }

WindowHandle Window::nativeInstance() const {
  return GetModuleHandleW(nullptr);
}

// ── Message handling ────────────────────────────────────────

i64 Window::handleMessage(void* hwndRaw, unsigned msg, usize wparamRaw,
                          isize lparamRaw) {
  const HWND hwnd = static_cast<HWND>(hwndRaw);
  const WPARAM wparam = static_cast<WPARAM>(wparamRaw);
  const LPARAM lparam = static_cast<LPARAM>(lparamRaw);

  if (onRawMessage && onRawMessage(hwndRaw, msg, wparamRaw, lparamRaw))
    return 0;

  switch (msg) {
    case WM_SIZE: {
      const u32 w = LOWORD(lparam), h = HIWORD(lparam);
      if (w != m_width || h != m_height) {
        m_width = w;
        m_height = h;
        if (onResize && w != 0 && h != 0) onResize(w, h);
      }
      return 0;
    }
    case WM_KEYDOWN:
    case WM_SYSKEYDOWN: {
      const bool repeat = (lparam & (1 << 30)) != 0;
      if (!repeat) {
        Input::get().onKey(static_cast<int>(wparam), true);
        if (onKey) onKey(static_cast<int>(wparam), true);
      }
      return 0;
    }
    case WM_KEYUP:
    case WM_SYSKEYUP: {
      Input::get().onKey(static_cast<int>(wparam), false);
      if (onKey) onKey(static_cast<int>(wparam), false);
      return 0;
    }
    case WM_LBUTTONDOWN:
    case WM_LBUTTONUP:
    case WM_MBUTTONDOWN:
    case WM_MBUTTONUP:
    case WM_RBUTTONDOWN:
    case WM_RBUTTONUP: {
      const int button = (msg == WM_LBUTTONDOWN || msg == WM_LBUTTONUP)   ? 0
                         : (msg == WM_MBUTTONDOWN || msg == WM_MBUTTONUP) ? 1
                                                                          : 2;
      const bool down = (msg == WM_LBUTTONDOWN || msg == WM_MBUTTONDOWN ||
                         msg == WM_RBUTTONDOWN);
      const int x = GET_X_LPARAM(lparam), y = GET_Y_LPARAM(lparam);
      Input::get().onMouseButton(button, down);
      if (onMouseButton) onMouseButton(button, down, x, y);
      return 0;
    }
    case WM_MOUSEMOVE: {
      const int x = GET_X_LPARAM(lparam), y = GET_Y_LPARAM(lparam);
      Input::get().onMouseMove(x, y);
      if (onMouseMove) onMouseMove(x, y);
      return 0;
    }
    case WM_MOUSEWHEEL: {
      const float delta = GET_WHEEL_DELTA_WPARAM(wparam) / 120.0f;
      Input::get().onWheel(delta);
      if (onWheel) onWheel(delta);
      return 0;
    }
    case WM_CLOSE:
      handleClose();
      return 0;
    case WM_DESTROY:
      PostQuitMessage(0);
      return 0;
    default:
      return DefWindowProcW(hwnd, static_cast<UINT>(msg),
                            static_cast<WPARAM>(wparam),
                            static_cast<LPARAM>(lparam));
  }
}

void Window::handleClose() {
  if (onClose) onClose();
  if (m_hwnd) DestroyWindow(static_cast<HWND>(m_hwnd));
  m_hwnd = nullptr;
  KENSHIN_INFO("Window closed");
}

}  // namespace core

KENSHIN_NS_END
