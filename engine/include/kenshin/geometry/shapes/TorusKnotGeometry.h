#pragma once

// ── TorusKnotGeometry (three.js style) ──────────────────────────
// (p,q) torus knot — the reference-geometry workhorse for virtual-geometry
// testing. Curve proportions from the reference engine (R=2.2, r=0.9,
// tube=0.16, all scaled by `radius`); tube built with a parallel-transport
// frame + evenly distributed seam twist (winding-consistent by
// construction, seam gap < 0.005). Knot lies in the X/Y plane, axis Z.

#include "kenshin/assets/MeshData.h"

KENSHIN_NS_BEGIN

namespace geometry
{

    class TorusKnotGeometry
    {
    public:
        /// @param tubeSegs  segments along the knot curve
        /// @param ringSegs  segments around the tube cross-section
        static assets::MeshData create(int p, int q, int tubeSegs, int ringSegs,
                                      f32 radius = 1.0f);
    };

} // namespace geometry

KENSHIN_NS_END
