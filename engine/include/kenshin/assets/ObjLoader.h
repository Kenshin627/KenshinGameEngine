#pragma once

// ── OBJ loader (S2.1) ────────────────────────────────────────────
// Minimal hand-written parser: v/vn/vt/f, fan triangulation of polygons,
// negative indices, missing attributes get defaults, single MTL material
// (Ka/Kd/map_Kd). Vertex dedup on the (v,vt,vn) index triple.

#include "kenshin/assets/MeshData.h"

#include <string>

KENSHIN_NS_BEGIN

namespace assets
{

    /// Load an OBJ file into `out`. On failure returns false and fills `log`.
    bool loadObj(const std::string& path, MeshData& out, std::string& log);

} // namespace assets

KENSHIN_NS_END
