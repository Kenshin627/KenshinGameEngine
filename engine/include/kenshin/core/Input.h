#pragma once

// ── Input state (S1.1) ──────────────────────────────────────────
// Raw Window events -> Input state (down/pressed/released + mouse deltas + wheel).
// Call nextFrame() at the end of each frame to clear per-frame state.

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace core
{

    class Input
    {
    public:
        static Input& get();

        // Queries (valid for the current frame)
        bool keyDown(int vk) const;      // held down
        bool keyPressed(int vk) const;   // went down this frame
        bool keyReleased(int vk) const;  // went up this frame

        int mouseX() const { return m_mouseX; }
        int mouseY() const { return m_mouseY; }
        int mouseDeltaX() const { return m_deltaX; }
        int mouseDeltaY() const { return m_deltaY; }

        bool mouseDown(int button) const;      // button: 0=L 1=M 2=R
        bool mousePressed(int button) const;

        float wheelDelta() const { return m_wheel; } // wheel notches this frame (positive = up)

        /// End of frame: clear per-frame state (pressed/released/deltas/wheel).
        void nextFrame();

        // ---- called by Window ----
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

KENSHIN_NS_END
