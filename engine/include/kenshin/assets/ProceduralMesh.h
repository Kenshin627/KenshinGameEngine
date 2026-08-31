#pragma once

// ── Procedural meshes (S2.1) ─────────────────────────────────────
// Deterministic generators with exact triangle counts (LOD/test material).

#include "kenshin/assets/MeshData.h"

KENSHIN_NS_BEGIN

namespace assets
{

    /// UV sphere: poles at +/-Y, segsU around Y, segsV from pole to pole.
    /// Triangles = segsU * (2 * (segsV - 1)).
    MeshData createUvSphere(int segsU, int segsV, f32 radius);

    /// (p,q) torus knot tube (three.js-style curve + derivative frame).
    /// Triangles = tubeSegs * ringSegs * 2 — e.g. 512 x 1024 -> 1,048,576.
    MeshData createTorusKnot(int p, int q, int tubeSegs, int ringSegs,
                             f32 radius, f32 tubeRadius);

} // namespace assets

KENSHIN_NS_END
