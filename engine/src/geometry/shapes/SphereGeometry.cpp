#include "kenshin/geometry/shapes/SphereGeometry.h"

#include <glm/gtc/constants.hpp>

#define _USE_MATH_DEFINES
#include <cmath>

KENSHIN_NS_BEGIN

namespace geometry
{

    assets::MeshData SphereGeometry::create(int segsU, int segsV, f32 radius)
    {
        assets::MeshData mesh;
        if (segsU < 3 || segsV < 2)
            return mesh;
        const int rings = segsV - 1;
        auto push = [&](f32 x, f32 y, f32 z, f32 u, f32 v) {
            mesh.positions.insert(mesh.positions.end(), {x, y, z});
            mesh.normals.insert(mesh.normals.end(),
                                {x / radius, y / radius, z / radius});
            mesh.uvs.insert(mesh.uvs.end(), {u, v});
        };

        push(0, radius, 0, 0, 1); // north pole
        for (int r = 1; r <= rings; ++r)
        {
            const f32 phi = glm::pi<f32>() * static_cast<f32>(r) / segsV;
            const f32 y = radius * std::cos(phi);
            const f32 s = radius * std::sin(phi);
            for (int u = 0; u < segsU; ++u)
            {
                const f32 theta = 2.0f * glm::pi<f32>() * static_cast<f32>(u) / segsU;
                push(s * std::sin(theta), y, s * std::cos(theta),
                     static_cast<f32>(u) / segsU,
                     1.0f - static_cast<f32>(r) / segsV);
            }
        }
        push(0, -radius, 0, 0, 0); // south pole

        const auto ring = [segsU](int r, int u) -> u32 {
            return 1u + static_cast<u32>(r - 1) * segsU + static_cast<u32>(u % segsU);
        };
        for (int u = 0; u < segsU; ++u)
            mesh.indices.insert(mesh.indices.end(), {0u, ring(1, u + 1), ring(1, u)});
        for (int r = 1; r < rings; ++r)
            for (int u = 0; u < segsU; ++u)
            {
                const u32 a = ring(r, u), b = ring(r, u + 1);
                const u32 c = ring(r + 1, u + 1), d = ring(r + 1, u);
                mesh.indices.insert(mesh.indices.end(), {a, b, c, a, c, d});
            }
        const u32 south = mesh.vertexCount() - 1;
        for (int u = 0; u < segsU; ++u)
            mesh.indices.insert(mesh.indices.end(),
                                {south, ring(rings, u), ring(rings, u + 1)});
        return mesh;
    }

} // namespace geometry

KENSHIN_NS_END
