#include "kenshin/ecs/Components.h"

#include <cmath>

KENSHIN_NS_BEGIN
namespace ecs {

glm::mat4 Transform::matrix() const {
  const glm::mat3 r = glm::mat3_cast(rotation);
  glm::mat4 m(1.0f);
  m[0] = glm::vec4(r[0] * scale.x, 0.0f);
  m[1] = glm::vec4(r[1] * scale.y, 0.0f);
  m[2] = glm::vec4(r[2] * scale.z, 0.0f);
  m[3] = glm::vec4(position, 1.0f);
  return m;
}

bool decomposeTransform(const glm::mat4& m, Transform& out) {
  glm::vec3 axis[3] = {glm::vec3(m[0]), glm::vec3(m[1]), glm::vec3(m[2])};
  glm::vec3 scale(glm::length(axis[0]), glm::length(axis[1]),
                  glm::length(axis[2]));
  constexpr f32 kMinScale = 1e-8f;
  if (!(scale.x > kMinScale && scale.y > kMinScale && scale.z > kMinScale))
    return false;  // also rejects NaN
  for (int i = 0; i < 3; ++i) axis[i] /= scale[i];
  // Unit columns: |det| is 1 for TRS, smaller under shear, ~0 when the
  // columns are dependent (no basis to recover a rotation from).
  const f32 det = glm::dot(glm::cross(axis[0], axis[1]), axis[2]);
  if (!(std::abs(det) > 1e-6f)) return false;
  // A mirror cannot be a rotation: fold it into the X scale.
  if (det < 0.0f) {
    scale.x = -scale.x;
    axis[0] = -axis[0];
  }
  // Re-orthonormalize (Gram-Schmidt, X kept exact) so sheared inputs
  // still yield a unit quaternion of the closest-by-X rotation.
  axis[1] = glm::normalize(axis[1] - glm::dot(axis[1], axis[0]) * axis[0]);
  axis[2] = glm::cross(axis[0], axis[1]);

  out.position = glm::vec3(m[3]);
  out.rotation = glm::normalize(glm::quat_cast(glm::mat3(axis[0], axis[1], axis[2])));
  out.scale = scale;
  return true;
}

}  // namespace ecs
KENSHIN_NS_END
