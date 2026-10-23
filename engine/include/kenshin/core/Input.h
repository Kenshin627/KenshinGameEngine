#pragma once

// ── Input state (S1.1) ──────────────────────────────────────────
// Window events -> per-frame input state: keys and mouse buttons as
// down / pressed (went down this frame) / released (went up this
// frame), mouse position + delta, wheel. Keys are physical (core::Key,
// HID usage IDs). The Window feeds the global instance; call
// nextFrame() once per frame AFTER the frame consumed the state.
// Instances are plain values so tests can drive their own.

#include "kenshin/Common.h"
#include "kenshin/core/Keys.h"

KENSHIN_NS_BEGIN

namespace core {

class Input {
 public:
  /// The process-wide instance the Window feeds.
  static Input& get();

  Input() = default;

  // Queries (valid for the current frame)
  bool keyDown(Key key) const;      // held down
  bool keyPressed(Key key) const;   // went down this frame
  bool keyReleased(Key key) const;  // went up this frame

  int mouseX() const { return m_mouseX; }
  int mouseY() const { return m_mouseY; }
  int mouseDeltaX() const { return m_deltaX; }
  int mouseDeltaY() const { return m_deltaY; }

  bool mouseDown(MouseButton button) const;
  bool mousePressed(MouseButton button) const;
  bool mouseReleased(MouseButton button) const;

  /// Wheel notches this frame (positive = up).
  float wheelDelta() const { return m_wheel; }

  /// End of frame: clear per-frame state (pressed/released/deltas/wheel).
  void nextFrame();
  /// Release everything that is held (focus loss: a key released while
  /// another window has focus never reports its key-up).
  void releaseAll();

  // ---- fed by the Window ----
  void onKey(Key key, bool down);
  void onMouseMove(int x, int y);
  void onMouseButton(MouseButton button, bool down);
  void onWheel(float delta);

 private:
  bool m_keyDown[kKeyCount] = {};
  bool m_keyPressed[kKeyCount] = {};
  bool m_keyReleased[kKeyCount] = {};

  bool m_mouseDown[kMouseButtonCount] = {};
  bool m_mousePressed[kMouseButtonCount] = {};
  bool m_mouseReleased[kMouseButtonCount] = {};

  int m_mouseX = 0, m_mouseY = 0;
  int m_deltaX = 0, m_deltaY = 0;
  int m_lastX = -1, m_lastY = -1;

  float m_wheel = 0.0f;
};

}  // namespace core

KENSHIN_NS_END
