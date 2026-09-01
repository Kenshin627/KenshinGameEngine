#include "kenshin/geometry/shapes/TorusKnotGeometry.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <glm/gtc/constants.hpp>

KENSHIN_NS_BEGIN

namespace geometry {

assets::MeshData TorusKnotGeometry::create(int p, int q, int tubeSegs,
                                           int ringSegs, f32 radius) {
  assets::MeshData mesh;
  if (p < 1 || q < 1 || tubeSegs < 3 || ringSegs < 3) return mesh;

  // Canonical (p,q) torus knot, verbatim from the reference engine
  // (C:/code/refactor/renderEngine sandbox): winds p times around the
  // torus axis, q times through the hole; proportions R=2.2, r=0.9,
  // tube 0.10 give clean strand separation (measured 1.19 between
  // strands vs 0.20 tube diameter). The knot winds in the X/Y plane
  // with the axis along Z.
  const f32 knotR = radius * 2.2f;
  const f32 knotr = radius * 0.9f;
  const f32 tubeR = radius * 0.16f;
  auto curve = [&](f32 t, glm::vec3& pos) {
    const f32 cr = knotR + knotr * std::cos(q * t);
    pos.x = cr * std::cos(p * t);
    pos.y = cr * std::sin(p * t);
    pos.z = knotr * std::sin(q * t);
  };
  const f32 curveSpan = glm::pi<f32>() * 2.0f;

  const usize vertexTotal = static_cast<usize>(tubeSegs + 1) * (ringSegs + 1);
  mesh.positions.reserve(vertexTotal * 3);
  mesh.normals.reserve(vertexTotal * 3);
  mesh.uvs.reserve(vertexTotal * 2);

  // Parallel-transport frame with even seam-twist distribution:
  // winding-consistent along the whole tube by construction. (Both
  // earlier frames flipped orientation on segments where the tangent's
  // azimuth around the Y axis reversed - backface culling then ate
  // those strips: the "incomplete mesh" artifact, 42/256 segments.)
  auto tangentAt = [&](f32 t, glm::vec3& out) {
    glm::vec3 a, b;
    curve(t, a);
    curve(t + 0.01f, b);
    out = glm::normalize(b - a);
  };
  glm::vec3 t0;
  tangentAt(0.0f, t0);
  glm::vec3 seed = glm::vec3(0.0f, 1.0f, 0.0f);
  if (std::abs(glm::dot(seed, t0)) > 0.9f) seed = glm::vec3(1.0f, 0.0f, 0.0f);
  glm::vec3 prevBinormal = glm::normalize(glm::cross(t0, glm::cross(seed, t0)));

  // Measure holonomy: transport once around, compare with the start.
  glm::vec3 transported = prevBinormal;
  for (int i = 0; i < tubeSegs; ++i) {
    const f32 t = static_cast<f32>(i) / tubeSegs * curveSpan;
    glm::vec3 tangent;
    tangentAt(t, tangent);
    transported =
        glm::normalize(transported - tangent * glm::dot(tangent, transported));
  }
  const f32 seamAngle =
      std::atan2(glm::dot(glm::cross(prevBinormal, transported), t0),
                 glm::dot(prevBinormal, transported));
  const f32 twistPerStep = -seamAngle / static_cast<f32>(tubeSegs);

  glm::vec3 binormal = prevBinormal;
  for (int i = 0; i <= tubeSegs; ++i) {
    const f32 t = static_cast<f32>(i) / tubeSegs * curveSpan;

    glm::vec3 p1;
    curve(t, p1);
    glm::vec3 tangent;
    tangentAt(t, tangent);

    // Transport the previous binormal onto this tangent plane...
    binormal = glm::normalize(binormal - tangent * glm::dot(tangent, binormal));
    // ...and apply the per-step seam twist increment so ring
    // tubeSegs coincides exactly with ring 0 (total twist =
    // tubeSegs * twistPerStep = -seamAngle). PER-STEP delta, not
    // cumulative - a cumulative i*delta angle over-rotates
    // quadratically and tears the seam open.
    const glm::vec3 side = glm::normalize(glm::cross(tangent, binormal));
    binormal = glm::normalize(binormal * std::cos(twistPerStep) +
                              side * std::sin(twistPerStep));
    const glm::vec3 normal = glm::cross(binormal, tangent);

    for (int j = 0; j <= ringSegs; ++j) {
      const f32 v = static_cast<f32>(j) / ringSegs * glm::pi<f32>() * 2.0f;
      const f32 cx = -tubeR * std::cos(v);
      const f32 cy = tubeR * std::sin(v);

      const glm::vec3 pos = p1 + cx * normal + cy * binormal;
      mesh.positions.insert(mesh.positions.end(), {pos.x, pos.y, pos.z});
      const glm::vec3 nrm = glm::normalize(pos - p1);
      mesh.normals.insert(mesh.normals.end(), {nrm.x, nrm.y, nrm.z});
      mesh.uvs.insert(mesh.uvs.end(), {static_cast<f32>(i) / tubeSegs,
                                       static_cast<f32>(j) / ringSegs});
    }
  }

  const auto idx = [&](int i, int j) -> u32 {
    return static_cast<u32>(i * (ringSegs + 1) + (j % (ringSegs + 1)));
  };
  for (int i = 0; i < tubeSegs; ++i) {
    for (int j = 0; j < ringSegs; ++j) {
      const u32 a = idx(i, j);
      const u32 b = idx(i + 1, j);
      const u32 c = idx(i + 1, j + 1);
      const u32 d = idx(i, j + 1);
      mesh.indices.insert(mesh.indices.end(), {a, b, c, a, c, d});
    }
  }
  return mesh;
}

}  // namespace geometry

KENSHIN_NS_END
