#include "kenshin/geometry/shapes/TorusGeometry.h"

#define _USE_MATH_DEFINES
#include <cmath>

#include <glm/gtc/constants.hpp>

KENSHIN_NS_BEGIN

namespace geometry
{

    assets::MeshData TorusGeometry::create(f32 radius, f32 tube, int radialSegs,
                                           int tubularSegs)
    {
        assets::MeshData mesh;
        if (radialSegs < 3 || tubularSegs < 3)
            return mesh;

        for (int i = 0; i <= tubularSegs; ++i)
        {
            const f32 u = static_cast<f32>(i) / tubularSegs * glm::pi<f32>() * 2.0f;
            const f32 cu = std::cos(u), su = std::sin(u);
            for (int j = 0; j <= radialSegs; ++j)
            {
                const f32 v =
                    static_cast<f32>(j) / radialSegs * glm::pi<f32>() * 2.0f;
                const f32 cv = std::cos(v), sv = std::sin(v);
                const f32 ring = radius + tube * cv;
                mesh.positions.insert(mesh.positions.end(),
                                      {ring * cu, tube * sv, ring * su});
                mesh.normals.insert(mesh.normals.end(), {cv * cu, sv, cv * su});
                mesh.uvs.insert(mesh.uvs.end(),
                                {static_cast<f32>(i) / tubularSegs,
                                 static_cast<f32>(j) / radialSegs});
            }
        }
        const auto idx = [&](int i, int j) -> u32 {
            return static_cast<u32>(i * (radialSegs + 1) + (j % (radialSegs + 1)));
        };
        for (int i = 0; i < tubularSegs; ++i)
            for (int j = 0; j < radialSegs; ++j)
            {
                const u32 a = idx(i, j), b = idx(i, j + 1);
                const u32 c = idx(i + 1, j + 1), d = idx(i + 1, j);
                mesh.indices.insert(mesh.indices.end(), {a, b, c, a, c, d});
            }
        return mesh;
    }

} // namespace geometry

KENSHIN_NS_END
