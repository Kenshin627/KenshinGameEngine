// Shape geometry tests: known counts + validity + normals unit-length.

#include <kenshin/geometry/shapes/BoxGeometry.h>
#include <kenshin/geometry/shapes/CylinderGeometry.h>
#include <kenshin/geometry/shapes/PlaneGeometry.h>
#include <kenshin/geometry/shapes/SphereGeometry.h>
#include <kenshin/geometry/shapes/TorusGeometry.h>
#include <kenshin/geometry/shapes/TorusKnotGeometry.h>

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp>

namespace {
void checkValid(const kenshin::assets::MeshData& m) {
  REQUIRE(m.valid());
  REQUIRE(m.normals.size() == m.positions.size());
  for (kenshin::usize i = 0; i + 2 < m.normals.size(); i += 3) {
    const float l = std::sqrt(m.normals[i] * m.normals[i] +
                              m.normals[i + 1] * m.normals[i + 1] +
                              m.normals[i + 2] * m.normals[i + 2]);
    REQUIRE(std::abs(l - 1.0f) < 1e-4f);
  }
  const kenshin::u32 vc = m.vertexCount();
  for (kenshin::u32 index : m.indices) REQUIRE(index < vc);
}

// Front faces are CCW-outward: each triangle's geometric normal
// cross(b-a, c-a) must agree with the stored vertex normals. Catches
// mixed/inverted face tables (the box shipped with 3 of 6 faces
// inverted; sphere/cylinder/plane shipped fully inverted).
void checkWinding(const kenshin::assets::MeshData& m) {
  REQUIRE(m.indices.size() % 3 == 0);
  const auto pos = [&](kenshin::u32 i) {
    return glm::vec3(m.positions[i * 3], m.positions[i * 3 + 1],
                     m.positions[i * 3 + 2]);
  };
  const auto nrm = [&](kenshin::u32 i) {
    return glm::vec3(m.normals[i * 3], m.normals[i * 3 + 1],
                     m.normals[i * 3 + 2]);
  };
  for (kenshin::usize t = 0; t + 2 < m.indices.size(); t += 3) {
    const glm::vec3 a = pos(m.indices[t]);
    const glm::vec3 b = pos(m.indices[t + 1]);
    const glm::vec3 c = pos(m.indices[t + 2]);
    const glm::vec3 g = glm::cross(b - a, c - a);
    if (glm::length(g) < 1e-9f)
      continue;  // slivers collapsing at the cone apex
    const glm::vec3 n =
        nrm(m.indices[t]) + nrm(m.indices[t + 1]) + nrm(m.indices[t + 2]);
    REQUIRE(glm::dot(g, n) > 0.0f);
  }
}
}  // namespace

TEST_CASE("shape winding CCW-outward everywhere", "[shapes]") {
  checkWinding(kenshin::geometry::BoxGeometry::create(1.0f, 1.0f, 1.0f));
  checkWinding(kenshin::geometry::SphereGeometry::create(16, 8, 1.0f));
  checkWinding(kenshin::geometry::TorusGeometry::create(1.0f, 0.4f, 8, 16));
  checkWinding(
      kenshin::geometry::CylinderGeometry::create(1.0f, 1.0f, 2.0f, 16));
  checkWinding(kenshin::geometry::ConeGeometry::create(1.0f, 2.0f, 16));
  checkWinding(kenshin::geometry::PlaneGeometry::create(2.0f, 2.0f, 2, 2));
  checkWinding(kenshin::geometry::TorusKnotGeometry::create(2, 3, 64, 8, 1.0f));
}

TEST_CASE("box known counts and bounds", "[shapes]") {
  const auto m = kenshin::geometry::BoxGeometry::create(2.0f, 1.0f, 3.0f);
  REQUIRE(m.vertexCount() == 24);
  REQUIRE(m.triangleCount() == 12);
  checkValid(m);
  glm::vec3 mn, mx;
  m.computeBounds(mn, mx);
  using glm::epsilonEqual;
  REQUIRE(glm::all(epsilonEqual(mn, glm::vec3(-1, -0.5f, -1.5f), 1e-6f)));
  REQUIRE(glm::all(epsilonEqual(mx, glm::vec3(1, 0.5f, 1.5f), 1e-6f)));
}

TEST_CASE("sphere exact counts", "[shapes]") {
  const auto m = kenshin::geometry::SphereGeometry::create(64, 32, 1.0f);
  REQUIRE(m.triangleCount() ==
          kenshin::geometry::SphereGeometry::triangleCount(64, 32));
  checkValid(m);
}

TEST_CASE("torus exact counts", "[shapes]") {
  const auto m = kenshin::geometry::TorusGeometry::create(1.0f, 0.4f, 16, 48);
  REQUIRE(m.triangleCount() == 16u * 48u * 2u);
  checkValid(m);
}

TEST_CASE("cylinder counts, cone degenerate top", "[shapes]") {
  const auto cyl =
      kenshin::geometry::CylinderGeometry::create(1.0f, 1.0f, 2.0f, 16);
  // side 16*2 + bottom 16 + top 16 = 64
  REQUIRE(cyl.triangleCount() == 64u);
  checkValid(cyl);

  const auto cone = kenshin::geometry::ConeGeometry::create(1.0f, 2.0f, 16);
  // side 32 + bottom 16 (no top cap when radiusTop == 0)
  REQUIRE(cone.triangleCount() == 48u);
  checkValid(cone);
}

TEST_CASE("plane exact counts", "[shapes]") {
  const auto m = kenshin::geometry::PlaneGeometry::create(4.0f, 4.0f, 8, 8);
  REQUIRE(m.triangleCount() == 128u);
  REQUIRE(m.vertexCount() == 81u);
  checkValid(m);
}

TEST_CASE("torus knot counts and seam", "[shapes]") {
  const auto m =
      kenshin::geometry::TorusKnotGeometry::create(2, 3, 128, 32, 1.0f);
  REQUIRE(m.triangleCount() == 128u * 32u * 2u);
  checkValid(m);
  // seam: ring tubeSegs vertex j must coincide with ring 0 vertex j
  const kenshin::u32 ring = 33;  // ringSegs + 1
  const kenshin::u32 last = 128 * ring;
  for (int j = 0; j <= 32; j += 8) {
    const kenshin::u32 a = last + static_cast<kenshin::u32>(j);
    const kenshin::u32 b = static_cast<kenshin::u32>(j);
    const float dx = m.positions[a * 3] - m.positions[b * 3];
    const float dy = m.positions[a * 3 + 1] - m.positions[b * 3 + 1];
    const float dz = m.positions[a * 3 + 2] - m.positions[b * 3 + 2];
    REQUIRE(std::sqrt(dx * dx + dy * dy + dz * dz) < 0.02f);
  }
}
