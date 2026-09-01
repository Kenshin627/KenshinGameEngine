#pragma once

// ── CylinderGeometry (three.js style) ────────────────────────────
// Cylinder / cone / truncated cone along Y, origin at center. Slope-
// corrected side normals; caps have their own vertices facing +/-Y.
// radiusTop = 0 gives a cone (use ConeGeometry for clarity).

#include "kenshin/assets/MeshData.h"

KENSHIN_NS_BEGIN

namespace geometry {

class CylinderGeometry {
 public:
  /// @param openEnded  omit the caps
  static assets::MeshData create(f32 radiusTop, f32 radiusBottom, f32 height,
                                 int radialSegs, int heightSegs = 1,
                                 bool openEnded = false);

  static u32 triangleCount(int radialSegs, int heightSegs, bool openEnded) {
    u32 tris = static_cast<u32>(radialSegs * heightSegs * 2);
    if (!openEnded) {
      tris += static_cast<u32>(radialSegs);  // bottom fan
      tris += static_cast<u32>(radialSegs) *
              0u;  // top added dynamically (cone has no top)
    }
    return tris;
  }
};

class ConeGeometry {
 public:
  static assets::MeshData create(f32 radius, f32 height, int radialSegs,
                                 bool openEnded = false) {
    return CylinderGeometry::create(0.0f, radius, height, radialSegs, 1,
                                    openEnded);
  }
};

}  // namespace geometry

KENSHIN_NS_END
