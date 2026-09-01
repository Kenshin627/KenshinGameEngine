#include "kenshin/scene/Camera.h"

#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

KENSHIN_NS_BEGIN

namespace scene {
Camera::Camera() {
  m_view = glm::identity<glm::mat4>();
  m_proj = glm::identity<glm::mat4>();
  m_viewProj = glm::identity<glm::mat4>();
}

// ---------------------------------------------------------------------------
// Internal state (called by CameraController only)
// ---------------------------------------------------------------------------

void Camera::setPerspective(float fovY_deg, float aspect, float near,
                            float far) {
  m_fovY = fovY_deg;
  m_aspect = aspect;
  m_nearPlane = near;
  m_farPlane = far;
  m_projDirty = true;
}

void Camera::setTarget(const glm::vec3& target) {
  m_target = target;
  m_viewDirty = true;
}

void Camera::setDistance(float d) {
  m_distance = glm::clamp(d, m_minDist, m_maxDist);
  m_viewDirty = true;
}

void Camera::setSpherical(float theta, float phi, float distance) {
  m_theta = theta;
  m_phi = glm::clamp(phi, kMinPhi, kMaxPhi);
  m_distance = glm::clamp(distance, m_minDist, m_maxDist);
  m_viewDirty = true;
}

void Camera::rotate(float dTheta, float dPhi) {
  m_theta += dTheta;
  m_phi += dPhi;
  m_phi = glm::clamp(m_phi, kMinPhi, kMaxPhi);
  m_viewDirty = true;
}

void Camera::zoom(float delta) {
  // Geometric (multiplicative) zoom: each notch scales distance by a fixed
  // percentage, so the step is proportional to the current distance — close
  // inspection stays fine-grained, wide overviews move in big strides, all
  // independent of scene scale. delta>0 pulls in (factor <1), delta<0 pushes
  // out; symmetric (a notch in then out returns exactly), never crosses 0.
  m_distance *= std::exp(-delta);
  m_distance = glm::clamp(m_distance, m_minDist, m_maxDist);
  m_viewDirty = true;
}

void Camera::setZoomLimits(float minD, float maxD) {
  m_minDist = minD;
  m_maxDist = maxD;
  // Re-clamp current distance so it stays inside the new range.
  float clamped = glm::clamp(m_distance, m_minDist, m_maxDist);
  if (clamped != m_distance) {
    m_distance = clamped;
    m_viewDirty = true;
  }
}

void Camera::pan(float dx, float dy) {
  glm::vec3 fwd = glm::normalize((m_target - position()));
  glm::vec3 right =
      glm::normalize(glm::cross(fwd, glm::vec3(0.0f, 0.0f, 1.0f)));  // Z-up
  glm::vec3 up = glm::normalize(glm::cross(right, fwd));

  float speed = m_distance * 0.002f;
  m_target = m_target + right * (-dx * speed) + up * (dy * speed);
  m_viewDirty = true;
}

// ---------------------------------------------------------------------------
// Matrices
// ---------------------------------------------------------------------------

glm::vec3 Camera::position() const {
  // Z-up spherical: phi lifts the camera along +Z.
  float cosPhi = std::cos(m_phi);
  return m_target + glm::vec3{
                        m_distance * cosPhi * std::sin(m_theta),
                        m_distance * cosPhi * std::cos(m_theta),
                        m_distance * std::sin(m_phi),
                    };
}

void Camera::computeMatrices() {
  if (m_projDirty) {
    // m_fovY is degrees, glm takes radians. RH_ZO pairs with the RH
    // lookAt view for Vulkan's [0,1] depth; the Y negation flips
    // winding once so frontFace=CCW survives Vulkan's NDC->viewport
    // Y mapping (see the pipeline's viewportState comment).
    m_proj = glm::perspectiveRH_ZO(glm::radians(m_fovY), m_aspect, m_nearPlane,
                                   m_farPlane);
    m_proj[1][1] *= -1.0f;  // Vulkan clip-space Y points down
    m_projDirty = false;
  }
  if (m_viewDirty) {
    m_view = glm::lookAt(position(), m_target, {0.0f, 0.0f, 1.0f});  // Z-up
    m_viewDirty = false;
  }
  m_viewProj = m_proj * m_view;
}

// ---------------------------------------------------------------------------
// Picking
// ---------------------------------------------------------------------------

Ray Camera::screenToRay(float ndcX, float ndcY) const {
  glm::vec3 eye = position();
  glm::vec3 fwd = glm::normalize((m_target - eye));
  glm::vec3 right =
      glm::normalize(glm::cross(fwd, {0.0f, 0.0f, 1.0f}));  // Z-up
  glm::vec3 up = glm::cross(right, fwd);

  float tanHalfFov = std::tan(m_fovY * kPi / 180.0f * 0.5f);
  glm::vec3 dir =
      (fwd + right * (ndcX * m_aspect * tanHalfFov) + up * (ndcY * tanHalfFov));

  return {eye, dir};
}

}  // namespace scene

KENSHIN_NS_END
