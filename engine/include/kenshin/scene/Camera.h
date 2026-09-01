#pragma once

// ---------------------------------------------------------------------------
// Camera — internal math-only orbit camera
//
// CameraController is the ONLY public entry point.  Application code should
// not touch Camera directly.
// ---------------------------------------------------------------------------

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "kenshin/Common.h"
#include "kenshin/math/ray.h"

KENSHIN_NS_BEGIN

namespace scene {
class CameraController;  //fwd

class Camera {
 public:
  friend class CameraController;

  Camera();

  // --- call after any change ----------------------------------------------
  void computeMatrices();

  // --- matrices (read-only) -----------------------------------------------
  const glm::mat4& viewMatrix() const { return m_view; }
  const glm::mat4& projMatrix() const { return m_proj; }
  const glm::mat4& viewProjMatrix() const { return m_viewProj; }

  const float* viewMatrixData() const { return glm::value_ptr(m_view); }
  const float* projMatrixData() const { return glm::value_ptr(m_proj); }
  const float* viewProjMatrixData() const { return glm::value_ptr(m_viewProj); }

  // --- derived ------------------------------------------------------------
  glm::vec3 position() const;
  const glm::vec3& target() const { return m_target; }
  glm::vec3 forward() const {
    glm::vec3 res = m_target - position();
    return glm::normalize(res);
  }
  glm::vec3 up() const {
    glm::vec3 f = forward();
    glm::vec3 r = glm::cross(f, glm::vec3(0.0f, 0.0f, 1.0f));  // Z-up
    r = glm::normalize(r);
    glm::vec3 u = glm::cross(r, f);
    u = glm::normalize(u);
    return u;
  }
  float nearPlane() const { return m_nearPlane; }
  float farPlane() const { return m_farPlane; }

  Ray screenToRay(float ndcX, float ndcY) const;

 private:
  // --- internal state manipulation (only CameraController calls these) ----
  void setPerspective(float fovY_deg, float aspect, float near, float far);
  void setTarget(const glm::vec3& target);
  void setDistance(float d);
  void setSpherical(float theta, float phi, float distance);
  void rotate(float dTheta, float dPhi);
  void zoom(float delta);
  void pan(float dx, float dy);
  void setZoomLimits(float minD, float maxD);  // set + re-clamp m_distance

  // --- spherical coords ---------------------------------------------------
  glm::vec3 m_target{0.0f, 0.0f, 0.0f};
  float m_distance = 5.0f;
  float m_theta = 0.0f;
  float m_phi = 0.6f;
  float m_minDist =
      0.1f;  // zoom lower bound (scene-adaptive, set by controller)
  float m_maxDist =
      100.0f;  // zoom upper bound (scene-adaptive, set by controller)

  // --- projection ---------------------------------------------------------
  float m_fovY = 45.0f;
  float m_aspect = 1.0f;
  float m_nearPlane = 0.1f;
  float m_farPlane = 100.0f;

  // --- matrices -----------------------------------------------------------
  glm::mat4 m_view;
  glm::mat4 m_proj;
  glm::mat4 m_viewProj;
  bool m_viewDirty = true;
  bool m_projDirty = true;

  static constexpr float kPi = 3.141592653589793f;
  static constexpr float kMinPhi = -kPi * 0.49f;
  static constexpr float kMaxPhi = kPi * 0.49f;
};
}  // namespace scene

KENSHIN_NS_END
