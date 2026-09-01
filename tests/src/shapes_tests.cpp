// Shape geometry tests: known counts + validity + normals unit-length.

#include <catch2/catch_test_macros.hpp>

#include <kenshin/geometry/shapes/BoxGeometry.h>
#include <kenshin/geometry/shapes/CylinderGeometry.h>
#include <kenshin/geometry/shapes/PlaneGeometry.h>
#include <kenshin/geometry/shapes/SphereGeometry.h>
#include <kenshin/geometry/shapes/TorusGeometry.h>
#include <kenshin/geometry/shapes/TorusKnotGeometry.h>

#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp>

namespace
{
    void checkValid(const kenshin::assets::MeshData& m)
    {
        REQUIRE(m.valid());
        REQUIRE(m.normals.size() == m.positions.size());
        for (kenshin::usize i = 0; i + 2 < m.normals.size(); i += 3)
        {
            const float l = std::sqrt(m.normals[i] * m.normals[i] +
                                      m.normals[i + 1] * m.normals[i + 1] +
                                      m.normals[i + 2] * m.normals[i + 2]);
            REQUIRE(std::abs(l - 1.0f) < 1e-4f);
        }
        const kenshin::u32 vc = m.vertexCount();
        for (kenshin::u32 index : m.indices)
            REQUIRE(index < vc);
    }
} // namespace

TEST_CASE("box known counts and bounds", "[shapes]")
{
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

TEST_CASE("sphere exact counts", "[shapes]")
{
    const auto m = kenshin::geometry::SphereGeometry::create(64, 32, 1.0f);
    REQUIRE(m.triangleCount() == kenshin::geometry::SphereGeometry::triangleCount(64, 32));
    checkValid(m);
}

TEST_CASE("torus exact counts", "[shapes]")
{
    const auto m = kenshin::geometry::TorusGeometry::create(1.0f, 0.4f, 16, 48);
    REQUIRE(m.triangleCount() == 16u * 48u * 2u);
    checkValid(m);
}

TEST_CASE("cylinder counts, cone degenerate top", "[shapes]")
{
    const auto cyl = kenshin::geometry::CylinderGeometry::create(1.0f, 1.0f, 2.0f, 16);
    // side 16*2 + bottom 16 + top 16 = 64
    REQUIRE(cyl.triangleCount() == 64u);
    checkValid(cyl);

    const auto cone = kenshin::geometry::ConeGeometry::create(1.0f, 2.0f, 16);
    // side 32 + bottom 16 (no top cap when radiusTop == 0)
    REQUIRE(cone.triangleCount() == 48u);
    checkValid(cone);
}

TEST_CASE("plane exact counts", "[shapes]")
{
    const auto m = kenshin::geometry::PlaneGeometry::create(4.0f, 4.0f, 8, 8);
    REQUIRE(m.triangleCount() == 128u);
    REQUIRE(m.vertexCount() == 81u);
    checkValid(m);
}

TEST_CASE("torus knot counts and seam", "[shapes]")
{
    const auto m = kenshin::geometry::TorusKnotGeometry::create(2, 3, 128, 32, 1.0f);
    REQUIRE(m.triangleCount() == 128u * 32u * 2u);
    checkValid(m);
    // seam: ring tubeSegs vertex j must coincide with ring 0 vertex j
    const kenshin::u32 ring = 33; // ringSegs + 1
    const kenshin::u32 last = 128 * ring;
    for (int j = 0; j <= 32; j += 8)
    {
        const kenshin::u32 a = last + static_cast<kenshin::u32>(j);
        const kenshin::u32 b = static_cast<kenshin::u32>(j);
        const float dx = m.positions[a * 3] - m.positions[b * 3];
        const float dy = m.positions[a * 3 + 1] - m.positions[b * 3 + 1];
        const float dz = m.positions[a * 3 + 2] - m.positions[b * 3 + 2];
        REQUIRE(std::sqrt(dx * dx + dy * dy + dz * dz) < 0.02f);
    }
}
