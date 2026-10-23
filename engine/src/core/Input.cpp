#include "kenshin/core/Input.h"

KENSHIN_NS_BEGIN

namespace core {

namespace {

bool validKey(Key key) { return static_cast<u32>(key) < kKeyCount; }
bool validButton(MouseButton b) {
  return static_cast<u32>(b) < kMouseButtonCount;
}

}  // namespace

Input& Input::get() {
  static Input s_instance;
  return s_instance;
}

bool Input::keyDown(Key key) const {
  return validKey(key) && m_keyDown[static_cast<u32>(key)];
}

bool Input::keyPressed(Key key) const {
  return validKey(key) && m_keyPressed[static_cast<u32>(key)];
}

bool Input::keyReleased(Key key) const {
  return validKey(key) && m_keyReleased[static_cast<u32>(key)];
}

bool Input::mouseDown(MouseButton button) const {
  return validButton(button) && m_mouseDown[static_cast<u32>(button)];
}

bool Input::mousePressed(MouseButton button) const {
  return validButton(button) && m_mousePressed[static_cast<u32>(button)];
}

bool Input::mouseReleased(MouseButton button) const {
  return validButton(button) && m_mouseReleased[static_cast<u32>(button)];
}

void Input::nextFrame() {
  for (u32 i = 0; i < kKeyCount; ++i) {
    m_keyPressed[i] = false;
    m_keyReleased[i] = false;
  }
  for (u32 i = 0; i < kMouseButtonCount; ++i) {
    m_mousePressed[i] = false;
    m_mouseReleased[i] = false;
  }
  m_deltaX = m_deltaY = 0;
  m_wheel = 0.0f;
}

void Input::releaseAll() {
  for (u32 i = 0; i < kKeyCount; ++i)
    if (m_keyDown[i]) {
      m_keyDown[i] = false;
      m_keyReleased[i] = true;
    }
  for (u32 i = 0; i < kMouseButtonCount; ++i)
    if (m_mouseDown[i]) {
      m_mouseDown[i] = false;
      m_mouseReleased[i] = true;
    }
}

void Input::onKey(Key key, bool down) {
  if (!validKey(key)) return;
  const u32 k = static_cast<u32>(key);
  if (down) {
    if (!m_keyDown[k]) m_keyPressed[k] = true;
    m_keyDown[k] = true;
  } else {
    if (m_keyDown[k]) m_keyReleased[k] = true;
    m_keyDown[k] = false;
  }
}

void Input::onMouseMove(int x, int y) {
  if (m_lastX >= 0) {
    m_deltaX += x - m_lastX;
    m_deltaY += y - m_lastY;
  }
  m_lastX = m_mouseX = x;
  m_lastY = m_mouseY = y;
}

void Input::onMouseButton(MouseButton button, bool down) {
  if (!validButton(button)) return;
  const u32 b = static_cast<u32>(button);
  if (down && !m_mouseDown[b]) m_mousePressed[b] = true;
  if (!down && m_mouseDown[b]) m_mouseReleased[b] = true;
  m_mouseDown[b] = down;
}

void Input::onWheel(float delta) { m_wheel += delta; }

}  // namespace core

KENSHIN_NS_END
