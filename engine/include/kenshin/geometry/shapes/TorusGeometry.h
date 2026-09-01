#pragma once

// ── TorusGeometry (three.js style) ──────────────────────────────
// Ring of major radius `radius` with a tube of minor radius `tube`,
// lying in the X/Z plane around Y. Analytic normals.

#include "kenshin/assets/MeshData.h"

KENSHIN_NS_BEGIN

namespace geometry
{

    class TorusGeometry
    {
    public:
        /// @param radialSegs   segments around the tube cross-section
        /// @param tubularSegs  segments around the main ring
        static assets::MeshData create(f32 radius, f32 tube, int radialSegs,
                                      int tubularSegs);
    };

} // namespace geometry

KENSHIN_NS_END
