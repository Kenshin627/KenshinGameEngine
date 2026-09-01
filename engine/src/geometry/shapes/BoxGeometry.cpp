#include "kenshin/geometry/shapes/BoxGeometry.h"

KENSHIN_NS_BEGIN

namespace geometry {

assets::MeshData BoxGeometry::create(f32 width, f32 height, f32 depth) {
  assets::MeshData mesh;
  const f32 he[3] = {width * 0.5f, height * 0.5f, depth * 0.5f};

  // Face r = normal; face s = +u direction; face t = +v direction.
  // Center of a half-extent axis: halfExtent along that axis.
  struct Face {
    f32 r[3], s[3], t[3];
  };
  // Per face: triangle winding comes out CCW-outward only when
  // cross(s, t) == r (the quad corners sweep (s,t) CCW).
  const Face faces[6] = {
      {{0, 0, -1}, {-1, 0, 0}, {0, 1, 0}},  // -Z
      {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}},    // +Z
      {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},   // -Y
      {{1, 0, 0}, {0, 0, -1}, {0, 1, 0}},   // +X
      {{0, 1, 0}, {1, 0, 0}, {0, 0, -1}},   // +Y
      {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}},   // -X
  };

  for (const Face& f : faces) {
    const auto center = [&](const f32* d, f32(&out)[3]) {
      out[0] = out[1] = out[2] = 0.0f;
      for (int axis = 0; axis < 3; ++axis) {
        const int a = d[axis] != 0.0f ? axis : -1;
        if (a >= 0) out[a] = d[a] * he[a];
      }
    };
    f32 cr[3], cs[3], ct[3];
    center(f.r, cr);
    center(f.s, cs);
    center(f.t, ct);
    const f32 corners[4][3] = {
        {cr[0] - cs[0] - ct[0], cr[1] - cs[1] - ct[1], cr[2] - cs[2] - ct[2]},
        {cr[0] + cs[0] - ct[0], cr[1] + cs[1] - ct[1], cr[2] + cs[2] - ct[2]},
        {cr[0] + cs[0] + ct[0], cr[1] + cs[1] + ct[1], cr[2] + cs[2] + ct[2]},
        {cr[0] - cs[0] + ct[0], cr[1] - cs[1] + ct[1], cr[2] - cs[2] + ct[2]},
    };
    const f32 uvs[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    const u32 base = mesh.vertexCount();
    for (int i = 0; i < 4; ++i) {
      mesh.positions.insert(mesh.positions.end(),
                            {corners[i][0], corners[i][1], corners[i][2]});
      mesh.normals.insert(mesh.normals.end(), {f.r[0], f.r[1], f.r[2]});
      mesh.uvs.insert(mesh.uvs.end(), {uvs[i][0], uvs[i][1]});
    }
    mesh.indices.insert(mesh.indices.end(),
                        {base, base + 1, base + 2, base, base + 2, base + 3});
  }
  return mesh;
}

}  // namespace geometry

KENSHIN_NS_END
