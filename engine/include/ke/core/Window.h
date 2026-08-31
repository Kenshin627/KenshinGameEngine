#pragma once

// ── Win32 窗口（S1.1）───────────────────────────────────────────
// 手写窗口层：事件经回调上抛（resize/keys/mouse/wheel/close），
// 原生句柄供 Vulkan surface 创建（S1.3）使用。

#include "ke/Common.h"

#include <functional>
#include <string>

KE_NS_BEGIN

namespace core
{

    /// 原生窗口句柄（Windows 上即 HWND）。
    using WindowHandle = void*;

    struct WindowDesc
    {
        std::string title = "KEngine";
        u32 width  = 1600;
        u32 height = 900;
        bool resizable = true;
    };

    class Window
    {
    public:
        // 事件回调（App/Engine 挂接；resize 在最小化时 w/h 为 0，自行忽略）
        std::function<void(u32 w, u32 h)> onResize;
        std::function<void()> onClose;
        std::function<void(int vk, bool down)> onKey;              // vk = Win32 虚拟键码
        std::function<void(int button, bool down, int x, int y)> onMouseButton; // button: 0=L 1=M 2=R
        std::function<void(int x, int y)> onMouseMove;
        std::function<void(float delta)> onWheel;                  // 正 = 向上滚

        Window() = default;
        ~Window();

        Window(const Window&) = delete;
        Window& operator=(const Window&) = delete;

        bool create(const WindowDesc& desc);
        void destroy();
        bool isOpen() const { return m_hwnd != nullptr; }

        /// 泵消息（非阻塞）。返回 false 表示应退出（收到关闭）。
        bool pumpEvents();

        /// 外部请求关闭（如 ESC）。
        void requestClose();

        /// 客户区尺寸（像素）。
        u32 width() const { return m_width; }
        u32 height() const { return m_height; }

        WindowHandle nativeHandle() const { return m_hwnd; }
        WindowHandle nativeInstance() const;

    private:
        void handleResize(u32 w, u32 h);
        void handleClose();
        /// Win32 消息处理（类型去 Windows 化以避免头文件污染；.cpp 内转回）。
        i64 handleMessage(void* hwnd, unsigned msg, usize wparam, isize lparam);

        void* m_hwnd = nullptr;
        u32 m_width  = 0;
        u32 m_height = 0;
        bool m_quitPosted = false;
    };

} // namespace core

KE_NS_END
