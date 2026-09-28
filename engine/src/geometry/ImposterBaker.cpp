#include "kenshin/geometry/ImposterBaker.h"

#include <algorithm>
#include <cstring>

#include <glm/glm.hpp>

KENSHIN_NS_BEGIN

namespace geometry {

glm::vec3 octDecode(const glm::vec2& uv) {
  // uv in [-1,1]^2 (the atlas maps texel centers there); the standard
  // octahedral inverse: fold the L1-excess from |x|+|y| > 1 back over
  // the diagonals, then lift onto the hemisphere with z = 1-|x|-|y|.
  glm::vec2 p = uv;
  float z = 1.0f - std::abs(p.x) - std::abs(p.y);
  if (z < 0.0f) {
    const float ox = p.x;
    p.x = (1.0f - std::abs(p.y)) * (ox < 0.0f ? -1.0f : 1.0f);
    p.y = (1.0f - std::abs(ox)) * (p.y < 0.0f ? -1.0f : 1.0f);
  }
  return glm::normalize(glm::vec3(p.x, p.y, z));
}

namespace {

/// One triangle over one 12x12 ortho tile: perspective-free coverage
/// with barycentric z (the SW raster's affine depth; an ortho view
/// makes affine == exact). Backface triangles skip (the imposter bakes
/// the FRONT of each direction; winding follows the engine's
/// +Z-facing convention under the direction's view basis).
struct TileHit {
  u32 tri = 0xFFFFFFFFu;
  float depth = 1.0f;
};

void rasterTri(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c,
               u32 triIdx, const glm::vec3& dir, const glm::vec3& boundsMin,
               const glm::vec3& boundsMax, TileHit* tile) {
  // View basis: forward = -dir (looking at the mesh along -dir), right
  // and up orthonormal in the plane; Z-up world rides the Z axis when
  // the direction is near-vertical (the M9.1 shadow-view rule).
  const glm::vec3 fwd = -dir;
  const glm::vec3 upRef =
      std::abs(fwd.z) > 0.95f ? glm::vec3(1, 0, 0) : glm::vec3(0, 0, 1);
  const glm::vec3 right = glm::normalize(glm::cross(upRef, fwd));
  const glm::vec3 up = glm::cross(fwd, right);

  // Project into [-1,1] tile space via the AABB extents along the
  // basis (a cube fit; the ortho box spans the AABB).
  const glm::vec3 he = (boundsMax - boundsMin) * 0.5f;
  const glm::vec3 ctr = (boundsMax + boundsMin) * 0.5f;
  const auto toTile = [&](const glm::vec3& p) {
    const glm::vec3 d = p - ctr;
    return glm::vec2(glm::dot(d, right) / he.x,
                     glm::dot(d, up) / he.y);
  };
  const auto depthOf = [&](const glm::vec3& p) {
    // Forward maps the box to [0,1] (near = 0).
    return (glm::dot(p - ctr, fwd) / he.z + 1.0f) * 0.5f;
  };

  const glm::vec2 ta = toTile(a), tb = toTile(b), tc = toTile(c);
  // Backface: the engine's winding renders when the projected
  // cross points toward the viewer (negative Y in screen space is
  // up; the cross sign matches the SW raster's cullSign convention).
  const float area2 = (tb.x - ta.x) * (tc.y - ta.y) -
                      (tb.y - ta.y) * (tc.x - ta.x);
  if (area2 >= 0.0f) return;  // backfacing under this direction

  // Bounding rect in texel space (12x12, pixel centers at +0.5).
  constexpr u32 N = ImposterAtlas::kTexels;
  int x0 = int(std::floor(std::min({ta.x, tb.x, tc.x}) * 0.5f * N + N * 0.5f));
  int x1 = int(std::ceil(std::max({ta.x, tb.x, tc.x}) * 0.5f * N + N * 0.5f));
  int y0 = int(std::floor(std::min({ta.y, tb.y, tc.y}) * 0.5f * N + N * 0.5f));
  int y1 = int(std::ceil(std::max({ta.y, tb.y, tc.y}) * 0.5f * N + N * 0.5f));
  x0 = std::max(x0, 0);
  y0 = std::max(y0, 0);
  x1 = std::min(x1, int(N) - 1);
  y1 = std::min(y1, int(N) - 1);
  if (x0 > x1 || y0 > y1) return;

  const float da = depthOf(a), db = depthOf(b), dc = depthOf(c);
  const float invArea = 1.0f / area2;
  for (int y = y0; y <= y1; ++y) {
    for (int x = x0; x <= x1; ++x) {
      const glm::vec2 px((x + 0.5f) / N * 2.0f - 1.0f,
                         (y + 0.5f) / N * 2.0f - 1.0f);
      const float w0 = ((tb.x - px.x) * (tc.y - px.y) -
                        (tb.y - px.y) * (tc.x - px.x)) * invArea;
      const float w1 = ((tc.x - px.x) * (ta.y - px.y) -
                        (tc.y - px.y) * (ta.x - px.x)) * invArea;
      const float w2 = 1.0f - w0 - w1;
      if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) continue;
      const float z = w0 * da + w1 * db + w2 * dc;
      TileHit& h = tile[y * N + x];
      if (h.tri == 0xFFFFFFFFu || z < h.depth) {
        h.tri = triIdx;
        h.depth = z;
      }
    }
  }
}

}  // namespace

bool bakeImposterAtlas(const ClusterDag& dag,
                       const std::vector<u32>& clusterIndices,
                       ImposterAtlas& out, std::string& log) {
  if (clusterIndices.empty() || dag.clusters.empty()) {
    log = "imposter bake: empty cut";
    return false;
  }

  // Gather the cut's triangles (cluster-local indices into the DAG
  // pools; the runtime pair will point at the SAME clusters).
  struct Tri {
    glm::vec3 a, b, c;
    u32 index;  // 7-bit payload slot = ordinal within the cut
  };
  std::vector<Tri> tris;
  glm::vec3 mn(1e30f), mx(-1e30f);
  for (u32 ci = 0; ci < clusterIndices.size(); ++ci) {
    const DagCluster& cl = dag.clusters[clusterIndices[ci]];
    for (u32 t = 0; t + 2 < cl.indexCount; t += 3) {
      const auto corner = [&](u32 k) {
        const u32 vi = dag.indices[usize(cl.indexOffset + t + k)];
        const glm::vec3 p(
            dag.positions[usize(cl.vertexOffset + vi) * 3 + 0],
            dag.positions[usize(cl.vertexOffset + vi) * 3 + 1],
            dag.positions[usize(cl.vertexOffset + vi) * 3 + 2]);
        mn = glm::min(mn, p);
        mx = glm::max(mx, p);
        return p;
      };
      if (tris.size() >= 128) {
        log = "imposter bake: cut exceeds the 7-bit triangle payload";
        return false;
      }
      tris.push_back({corner(0), corner(1), corner(2),
                      u32(tris.size())});
    }
  }
  if (tris.empty()) {
    log = "imposter bake: no triangles";
    return false;
  }

  constexpr u32 D = ImposterAtlas::kDirs;
  constexpr u32 N = ImposterAtlas::kTexels;
  out.depth.assign(usize(D) * D * N * N, 0);
  out.tri.assign(usize(D) * D * N * N, 0xFFFFFFFFu);
  std::vector<TileHit> tile(usize(N) * N);
  for (u32 dv = 0; dv < D; ++dv) {
    for (u32 du = 0; du < D; ++du) {
      // Direction at the texel CENTER (the runtime samples the same
      // centers; the octahedral fold keeps the seam continuous).
      const glm::vec2 uv((du + 0.5f) / D * 2.0f - 1.0f,
                         (dv + 0.5f) / D * 2.0f - 1.0f);
      const glm::vec3 dir = octDecode(glm::vec2(uv.x, uv.y));
      std::fill(tile.begin(), tile.end(), TileHit{});
      for (const Tri& t : tris)
        rasterTri(t.a, t.b, t.c, t.index, dir, mn, mx, tile.data());
      const usize base = (usize(dv) * D + du) * N * N;
      for (usize i = 0; i < tile.size(); ++i) {
        if (tile[i].tri == 0xFFFFFFFFu) continue;
        const float z = std::clamp(tile[i].depth, 0.0f, 1.0f);
        out.depth[base + i] = u32((1.0 - z) * 4294967295.0);
        out.tri[base + i] = tile[i].tri;
      }
    }
  }
  return true;
}

}  // namespace geometry

KENSHIN_NS_END
