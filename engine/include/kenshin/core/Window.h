#pragma once

// ── Win32 window (S1.1) ─────────────────────────────────────────
// Hand-written window layer: events surface through callbacks
// (resize/keys/mouse/wheel/close); the native handle is exposed for
// Vulkan surface creation (S1.3).

#include "kenshin/Common.h"

#include <functional>
#include <string>

KENSHIN_NS_BEGIN

namespace core
{

    /// Native window handle (HWND on Windows).
    using WindowHandle = void*;

    struct WindowDesc
    {
        std::string title = "KENSHIN";
        u32 width  = 1600;
        u32 height = 900;
        bool resizable = true;
    };

    class Window
    {
    public:
        // Event callbacks (attached by app/engine). Resize fires with w/h == 0
        // when minimized; ignore that case.
        std::function<void(u32 w, u32 h)> onResize;
        std::function<void()> onClose;
        std::function<void(int vk, bool down)> onKey;              // vk = Win32 virtual key code
        std::function<void(int button, bool down, int x, int y)> onMouseButton; // button: 0=L 1=M 2=R
        std::function<void(int x, int y)> onMouseMove;
        std::function<void(float delta)> onWheel;                  // positive = scroll up

        /// Raw Win32 message hook (used by the ImGui Win32 backend).
        /// Receives de-typed (hwnd, msg, wparam, lparam); return true when
        /// the message is consumed and default handling should be skipped.
        std::function<bool(void* hwnd, unsigned msg, usize wparam, isize lparam)> onRawMessage;

        Window() = default;
        ~Window();

        Window(const Window&) = delete;
        Window& operator=(const Window&) = delete;

        bool create(const WindowDesc& desc);
        void destroy();
        bool isOpen() const { return m_hwnd != nullptr; }

        /// Pump messages (non-blocking). Returns false when the app should exit.
        bool pumpEvents();

        /// Request close from outside (e.g. ESC).
        void requestClose();

        /// Client-area size in pixels.
        u32 width() const { return m_width; }
        u32 height() const { return m_height; }

        WindowHandle nativeHandle() const { return m_hwnd; }
        WindowHandle nativeInstance() const;

    private:
        void handleResize(u32 w, u32 h);
        void handleClose();
        /// Win32 message handler (signature de-Win32'd to keep this header clean;
        /// cast back to native types in the .cpp).
        i64 handleMessage(void* hwnd, unsigned msg, usize wparam, isize lparam);

        void* m_hwnd = nullptr;
        u32 m_width  = 0;
        u32 m_height = 0;
        bool m_quitPosted = false;
    };

} // namespace core

KENSHIN_NS_END
