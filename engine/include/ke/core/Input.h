#pragma once

// ── 输入状态（S1.1）─────────────────────────────────────────────
// Window 原始事件 → Input 状态（按住/按下/松开 + 鼠标增量 + 滚轮）。
// 每帧末调用 nextFrame() 清除逐帧状态。

#include "ke/Common.h"

KE_NS_BEGIN

namespace core
{

    class Input
    {
    public:
        static Input& get();

        /// 查询（本帧有效）
        bool keyDown(int vk) const;      // 持续按住
        bool keyPressed(int vk) const;   // 本帧刚按下
        bool keyReleased(int vk) const;  // 本帧刚松开

        int mouseX() const { return m_mouseX; }
        int mouseY() const { return m_mouseY; }
        int mouseDeltaX() const { return m_deltaX; }
        int mouseDeltaY() const { return m_deltaY; }

        bool mouseDown(int button) const;      // button: 0=L 1=M 2=R
        bool mousePressed(int button) const;

        float wheelDelta() const { return m_wheel; } // 本帧滚轮格数（正=向上）

        /// 帧末调用：清逐帧状态（按下/松开/增量/滚轮）。
        void nextFrame();

        // ---- Window 内部调用 ----
        void onKey(int vk, bool down);
        void onMouseMove(int x, int y);
        void onMouseButton(int button, bool down);
        void onWheel(float delta);

    private:
        Input() = default;

        bool m_keyDown[256]     = {};
        bool m_keyPressed[256]  = {};
        bool m_keyReleased[256] = {};

        bool m_mouseDown[3]     = {};
        bool m_mousePressed[3]  = {};

        int m_mouseX = 0, m_mouseY = 0;
        int m_deltaX = 0, m_deltaY = 0;
        int m_lastX = -1, m_lastY = -1;

        float m_wheel = 0.0f;
    };

} // namespace core

KE_NS_END
