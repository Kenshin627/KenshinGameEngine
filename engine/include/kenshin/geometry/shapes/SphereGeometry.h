#pragma once

// ── SphereGeometry (three.js style) ──────────────────────────────
// UV sphere, poles at +/-Y. Analytic normals, UVs from spherical coords.
// Conventions per staged-plan 0.5: RH, Y-up, CCW front faces, meters.

#include "kenshin/assets/MeshData.h"

KENSHIN_NS_BEGIN

namespace geometry
{

    class SphereGeometry
    {
    public:
        /// @param segsU   longitude segments (>= 3)
        /// @param segsV   latitude segments (>= 2)
        /// @param radius  sphere radius
        static assets::MeshData create(int segsU, int segsV, f32 radius);

        static constexpr u32 triangleCount(int segsU, int segsV)
        {
            return static_cast<u32>(segsU * (2 * (segsV - 1)));
        }
    };

} // namespace geometry

KENSHIN_NS_END
