#include "kenshin/scene/CameraController.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <glm/gtc/constants.hpp>

KENSHIN_NS_BEGIN

namespace scene {

namespace {
f32 radians(f32 deg) { return deg * glm::pi<f32>() / 180.0f; }
f32 degrees(f32 rad) { return rad * 180.0f / glm::pi<f32>(); }
}  // namespace

void CameraController::setup(f32 yawDeg, f32 pitchDeg, f32 distance,
                             const glm::vec3& target, f32 aspect) {
  m_yawDeg = yawDeg;
  m_pitchDeg = glm::clamp(pitchDeg, -85.0f, 85.0f);
  m_distance = distance;
  m_target = target;
  (void)aspect;
}

void CameraController::handleInput(Camera& camera,
                                   const ControllerInput& input) {
  if (input.rotateActive) {
    m_yawDeg -= input.mouseDeltaX * rotateSpeedDegPerPx;
    m_pitchDeg += input.mouseDeltaY * rotateSpeedDegPerPx;
    m_pitchDeg = glm::clamp(m_pitchDeg, -85.0f, 85.0f);
  }

  if (input.wheel != 0.0f) {
    // Geometric zoom: each notch scales the distance by a factor,
    // symmetric and scale-free.
    const f32 factor = std::pow(0.90f, input.wheel);
    m_distance = glm::clamp(m_distance * factor, minDistance, maxDistance);
  }

  // WASD/QE pan the target in the camera's screen plane (Z-up basis).
  if (input.moveF || input.moveB || input.moveL || input.moveR ||
      input.moveUp || input.moveDown) {
    // Basis from the camera's current orientation (Z-up).
    const glm::vec3 fwd = camera.forward();
    const glm::vec3 worldUp(0.0f, 0.0f, 1.0f);
    glm::vec3 right = glm::cross(fwd, worldUp);
    const f32 rl = glm::length(right);
    right = rl > 1e-5f ? right / rl
                       : glm::vec3(1.0f, 0.0f, 0.0f);  // degenerate at poles

    const f32 speed =
        moveSpeed * (input.fast ? fastMultiplier : 1.0f) * input.deltaTime;
    glm::vec3 move(0.0f);
    if (input.moveF) move += fwd;
    if (input.moveB) move -= fwd;
    if (input.moveR) move += right;
    if (input.moveL) move -= right;
    if (input.moveUp) move += worldUp;
    if (input.moveDown) move -= worldUp;
    const f32 ml = glm::length(move);
    if (ml > 1e-6f) m_target += move / ml * speed;
  }
}

void CameraController::updateMatrices(Camera& camera, f32 aspect) {
  camera.setPerspective(fovYDeg, aspect, nearPlane, farPlane);
  camera.setTarget(m_target);
  camera.setZoomLimits(minDistance, maxDistance);
  camera.setSpherical(radians(m_yawDeg), radians(m_pitchDeg), m_distance);
  camera.computeMatrices();
}

}  // namespace scene

KENSHIN_NS_END
