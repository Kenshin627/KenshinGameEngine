#include "kenshin/geometry/shapes/CylinderGeometry.h"

#define _USE_MATH_DEFINES
#include <cmath>

#include <glm/gtc/constants.hpp>

KENSHIN_NS_BEGIN

namespace geometry
{

    assets::MeshData CylinderGeometry::create(f32 radiusTop, f32 radiusBottom,
                                             f32 height, int radialSegs,
                                             int heightSegs, bool openEnded)
    {
        assets::MeshData mesh;
        if (radialSegs < 3 || heightSegs < 1)
            return mesh;
        const f32 hy = height * 0.5f;

        // Side wall.
        const f32 slope = (radiusBottom - radiusTop) / height;
        for (int h = 0; h <= heightSegs; ++h)
        {
            const f32 t = static_cast<f32>(h) / heightSegs;
            const f32 y = height * t - hy;
            const f32 r = radiusBottom + (radiusTop - radiusBottom) * t;
            f32 ny = slope;
            const f32 nl = std::sqrt(1.0f + ny * ny);
            for (int i = 0; i <= radialSegs; ++i)
            {
                const f32 theta =
                    2.0f * glm::pi<f32>() * static_cast<f32>(i) / radialSegs;
                const f32 c = std::cos(theta), s = std::sin(theta);
                mesh.positions.insert(mesh.positions.end(), {r * c, y, r * s});
                mesh.normals.insert(mesh.normals.end(), {c / nl, ny / nl, s / nl});
                mesh.uvs.insert(mesh.uvs.end(),
                                {static_cast<f32>(i) / radialSegs, t});
            }
        }
        const auto side = [&](int ring, int i) -> u32 {
            return static_cast<u32>(ring * (radialSegs + 1) +
                                   (i % (radialSegs + 1)));
        };
        for (int h = 0; h < heightSegs; ++h)
            for (int i = 0; i < radialSegs; ++i)
            {
                const u32 a = side(h, i), b = side(h, i + 1);
                const u32 c = side(h + 1, i + 1), d = side(h + 1, i);
                mesh.indices.insert(mesh.indices.end(), {a, c, b, a, d, c});
            }

        if (openEnded)
            return mesh;

        // Bottom cap fan, facing -Y.
        const auto addCap = [&](f32 y, f32 r, f32 ny, bool ccw) {
            const u32 base = mesh.vertexCount();
            mesh.positions.insert(mesh.positions.end(), {0, y, 0});
            mesh.normals.insert(mesh.normals.end(), {0, ny, 0});
            mesh.uvs.insert(mesh.uvs.end(), {0.5f, 0.5f});
            for (int i = 0; i <= radialSegs; ++i)
            {
                const f32 theta =
                    2.0f * glm::pi<f32>() * static_cast<f32>(i) / radialSegs;
                const f32 c = std::cos(theta), s = std::sin(theta);
                mesh.positions.insert(mesh.positions.end(), {r * c, y, r * s});
                mesh.normals.insert(mesh.normals.end(), {0, ny, 0});
                mesh.uvs.insert(mesh.uvs.end(), {(c + 1) * 0.5f, (s + 1) * 0.5f});
            }
            for (int i = 0; i < radialSegs; ++i)
            {
                if (ccw)
                    mesh.indices.insert(mesh.indices.end(),
                                        {base, base + 1 + i, base + 1 + i + 1});
                else
                    mesh.indices.insert(mesh.indices.end(),
                                        {base, base + 1 + i + 1, base + 1 + i});
            }
        };
        addCap(-hy, radiusBottom, -1.0f, true);
        if (radiusTop > 0.0f)
            addCap(hy, radiusTop, 1.0f, false);
        return mesh;
    }

} // namespace geometry

KENSHIN_NS_END
