#include "kenshin/core/Input.h"

KENSHIN_NS_BEGIN

namespace core {

Input& Input::get() {
  static Input s_instance;
  return s_instance;
}

bool Input::keyDown(int vk) const {
  return vk >= 0 && vk < 256 && m_keyDown[vk];
}

bool Input::keyPressed(int vk) const {
  return vk >= 0 && vk < 256 && m_keyPressed[vk];
}

bool Input::keyReleased(int vk) const {
  return vk >= 0 && vk < 256 && m_keyReleased[vk];
}

bool Input::mouseDown(int button) const {
  return button >= 0 && button < 3 && m_mouseDown[button];
}

bool Input::mousePressed(int button) const {
  return button >= 0 && button < 3 && m_mousePressed[button];
}

void Input::nextFrame() {
  for (int i = 0; i < 256; ++i) {
    m_keyPressed[i] = false;
    m_keyReleased[i] = false;
  }
  for (int i = 0; i < 3; ++i) m_mousePressed[i] = false;
  m_deltaX = m_deltaY = 0;
  m_wheel = 0.0f;
}

void Input::onKey(int vk, bool down) {
  if (vk < 0 || vk >= 256) return;
  if (down) {
    if (!m_keyDown[vk]) m_keyPressed[vk] = true;
    m_keyDown[vk] = true;
  } else {
    m_keyDown[vk] = false;
    m_keyReleased[vk] = true;
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

void Input::onMouseButton(int button, bool down) {
  if (button < 0 || button >= 3) return;
  if (down && !m_mouseDown[button]) m_mousePressed[button] = true;
  m_mouseDown[button] = down;
}

void Input::onWheel(float delta) { m_wheel += delta; }

}  // namespace core

KENSHIN_NS_END
