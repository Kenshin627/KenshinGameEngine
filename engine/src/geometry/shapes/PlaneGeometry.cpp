#include "kenshin/geometry/shapes/PlaneGeometry.h"

KENSHIN_NS_BEGIN

namespace geometry
{

    assets::MeshData PlaneGeometry::create(f32 width, f32 height, int widthSegs,
                                          int heightSegs)
    {
        assets::MeshData mesh;
        if (widthSegs < 1 || heightSegs < 1)
            return mesh;
        const f32 hw = width * 0.5f, hh = height * 0.5f;
        for (int j = 0; j <= heightSegs; ++j)
            for (int i = 0; i <= widthSegs; ++i)
            {
                const f32 u = static_cast<f32>(i) / widthSegs;
                const f32 v = static_cast<f32>(j) / heightSegs;
                mesh.positions.insert(mesh.positions.end(),
                                      {(u * 2 - 1) * hw, 0, (v * 2 - 1) * hh});
                mesh.normals.insert(mesh.normals.end(), {0, 1, 0});
                mesh.uvs.insert(mesh.uvs.end(), {u, v});
            }
        const auto idx = [&](int i, int j) -> u32 {
            return static_cast<u32>(j * (widthSegs + 1) + i);
        };
        for (int j = 0; j < heightSegs; ++j)
            for (int i = 0; i < widthSegs; ++i)
            {
                const u32 a = idx(i, j), b = idx(i + 1, j);
                const u32 c = idx(i + 1, j + 1), d = idx(i, j + 1);
                mesh.indices.insert(mesh.indices.end(), {a, b, c, a, c, d});
            }
        return mesh;
    }

} // namespace geometry

KENSHIN_NS_END
