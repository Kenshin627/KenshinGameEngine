#pragma once

// ── BoxGeometry (three.js style) ────────────────────────────────
// Axis-aligned box with hard edges (per-face normals). 24 vertices, 12
// triangles. Origin at the center.

#include "kenshin/assets/MeshData.h"

KENSHIN_NS_BEGIN

namespace geometry
{

    class BoxGeometry
    {
    public:
        static assets::MeshData create(f32 width, f32 height, f32 depth);

        static constexpr u32 kVertexCount = 24;
        static constexpr u32 kTriangleCount = 12;
    };

} // namespace geometry

KENSHIN_NS_END
