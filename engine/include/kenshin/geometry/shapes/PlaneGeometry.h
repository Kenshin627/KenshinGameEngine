#pragma once

// ── PlaneGeometry (three.js style) ──────────────────────────────
// width x height grid in the X/Z plane facing +Y, centered at origin.

#include "kenshin/assets/MeshData.h"

KENSHIN_NS_BEGIN

namespace geometry {

class PlaneGeometry {
 public:
  static assets::MeshData create(f32 width, f32 height, int widthSegs,
                                 int heightSegs);
};

}  // namespace geometry

KENSHIN_NS_END
