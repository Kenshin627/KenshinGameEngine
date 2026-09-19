#include "kenshin/geometry/ClusterQuant.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>

#include "kenshin/core/Log.h"
#include "meshoptimizer/meshoptimizer.h"

KENSHIN_NS_BEGIN

namespace geometry {

QuantMeshGrid chooseQuantGrid(const ClusterDag& dag) {
  // UE5's density heuristic: clusters whose extent hovers around one
  // unit land at P = 7 (step = 2^-7); the average cluster extent shifts
  // P so typical clusters use ~6-9 bits per axis.
  double avgLog2 = 0.0;
  double maxExtent = 0.0;
  for (const DagCluster& cl : dag.clusters) {
    glm::vec3 mn(FLT_MAX), mx(-FLT_MAX);
    for (u32 v = 0; v < cl.vertexCount; ++v) {
      const f32* p = dag.positions.data() + (usize(cl.vertexOffset + v) * 3);
      mn = glm::min(mn, glm::vec3(p[0], p[1], p[2]));
      mx = glm::max(mx, glm::vec3(p[0], p[1], p[2]));
    }
    const glm::vec3 ext = mx - mn;
    const double extent = double(std::max({ext.x, ext.y, ext.z}));
    avgLog2 += std::log2(std::max(extent, 1e-9));
    maxExtent = std::max(maxExtent, extent);
  }
  avgLog2 /= double(std::max<usize>(dag.clusters.size(), 1));

  i32 P = i32(7.0 - std::round(avgLog2));
  // Every cluster's integer extent must fit 11 bits (worst extent *
  // 2^P <= 2047); halving the grid (smaller P) shrinks the integers.
  if (maxExtent > 0.0) {
    const i32 Pfit = i32(std::floor(std::log2(2047.0 / maxExtent)));
    P = std::min(P, Pfit);
  }
  P = std::clamp(P, -16, 20);

  QuantMeshGrid grid;
  grid.P = P;
  grid.step = std::exp2(-double(P));  // exact f32 power of two
  return grid;
}

namespace {
/// Round-to-nearest integer coordinate on the global grid (double
/// domain - f32 * 2^P can lose the +0.5 at large |I|).
inline i64 gridInt(f32 v, const QuantMeshGrid& g) {
  return i64(std::floor(double(v) * std::exp2(double(g.P)) + 0.5));
}
inline u32 axisWidth(i64 extent) {
  // ceil(log2(extent+1)) exactly: bits to represent values in [0, extent].
  return extent <= 0 ? 0 : std::bit_width(u64(extent));
}
}  // namespace

bool quantizeCluster(const ClusterDag& dag, u32 cluster,
                     const QuantMeshGrid& grid, bool hasNormals, bool hasUvs,
                     QuantCluster& out, std::string& log) {
  const DagCluster& cl = dag.clusters[cluster];
  out = {};
  out.vertexCount = cl.vertexCount;
  out.indexCount = cl.indexCount;
  if (cl.vertexCount > 255) {
    log = "cluster " + std::to_string(cluster) +
          " has more than 255 vertices (packed-index baseBits limit)";
    return false;
  }
  const u32 nV = cl.vertexCount;

  // 1) Integer coordinates + per-axis anchor/extent.
  std::vector<i64> I(usize(nV) * 3);
  i64 mn[3] = {INT64_MAX, INT64_MAX, INT64_MAX};
  i64 mx[3] = {INT64_MIN, INT64_MIN, INT64_MIN};
  for (u32 v = 0; v < nV; ++v) {
    const f32* p = dag.positions.data() + (usize(cl.vertexOffset + v) * 3);
    for (int a = 0; a < 3; ++a) {
      const i64 i = gridInt(p[a], grid);
      I[usize(v) * 3 + a] = i;
      mn[a] = std::min(mn[a], i);
      mx[a] = std::max(mx[a], i);
      if (std::abs(i) > (1 << 24)) {
        log = "cluster " + std::to_string(cluster) +
              " integer coord exceeds 2^24 (grid too coarse)";
        return false;
      }
    }
  }
  u32 b[3];
  for (int a = 0; a < 3; ++a) {
    b[a] = axisWidth(mx[a] - mn[a]);
    if (b[a] > 11) {
      log = "cluster " + std::to_string(cluster) + " axis " +
            std::to_string(a) + " needs " + std::to_string(b[a]) +
            " bits (max 11) - the grid heuristic must clamp harder";
      return false;
    }
    out.meta.posStart[a] = i32(mn[a]);
  }
  out.meta.widths = b[0] | (b[1] << 4) | (b[2] << 8);

  // 2) UV ranges (per component; range 0 -> all lanes 0).
  f32 uvMn[2] = {FLT_MAX, FLT_MAX}, uvMx[2] = {-FLT_MAX, -FLT_MAX};
  if (hasUvs) {
    for (u32 v = 0; v < nV; ++v) {
      const f32* t = dag.uvs.data() + (usize(cl.vertexOffset + v) * 2);
      for (int a = 0; a < 2; ++a) {
        uvMn[a] = std::min(uvMn[a], t[a]);
        uvMx[a] = std::max(uvMx[a], t[a]);
      }
    }
  } else {
    uvMn[0] = uvMn[1] = 0.0f;
    uvMx[0] = uvMx[1] = 0.0f;
  }

  // 3) Pack: positions SOA per axis (LSB-first bit stream), oct
  // normals, uv lanes, M7.2 packed indices (cyclic min-first rotation
  // + per-cluster declared widths - see the header comment).
  BitWriter w(out.posWords);
  for (int a = 0; a < 3; ++a)
    for (u32 v = 0; v < nV; ++v)
      w.write(u32(I[usize(v) * 3 + a] - mn[a]), b[a]);

  out.nrmOct.resize(usize(nV) * 2);
  if (hasNormals)
    for (u32 v = 0; v < nV; ++v) {
      const f32* n = dag.normals.data() + (usize(cl.vertexOffset + v) * 3);
      octEncode(n[0], n[1], n[2], &out.nrmOct[usize(v) * 2]);
    }

  out.uvQ.resize(usize(nV) * 2);
  for (int a = 0; a < 2; ++a) {
    const f32 range = uvMx[a] - uvMn[a];
    out.meta.uvMin[a] = uvMn[a];
    out.meta.uvRange[a] = range;
    const f32 inv = range > 0.0f ? 255.0f / range : 0.0f;
    for (u32 v = 0; v < nV; ++v) {
      const f32 t = hasUvs ? dag.uvs[usize(cl.vertexOffset + v) * 2 + a] : 0.0f;
      const i32 q = i32((t - uvMn[a]) * inv + 0.5f);
      out.uvQ[usize(v) * 2 + a] = u8(std::clamp(q, 0, 255));
    }
  }

  // Indices: rotate each triangle so its minimal index (first
  // occurrence) leads - a CYCLIC rotation preserves winding - then
  // store [base | off1 | off2] with per-cluster widths. The worst
  // triangle's span bounds the offsets; the builder's first-use vertex
  // ordering (measured 2026-09-19) keeps most clusters at
  // offsetBits 5-6 (Morton/triangle-reorder variants were worse).
  const u32 baseBits = nV <= 1 ? 0u : u32(std::bit_width(u64(nV - 1)));
  if (baseBits > 8) {
    log = "cluster " + std::to_string(cluster) +
          " vertexCount needs baseBits > 8";
    return false;
  }
  u32 maxSpan = 0;
  const u32 triCount = cl.indexCount / 3;
  for (u32 t = 0; t < triCount; ++t) {
    const u32 i0 = dag.indices[usize(cl.indexOffset) + t * 3 + 0];
    const u32 i1 = dag.indices[usize(cl.indexOffset) + t * 3 + 1];
    const u32 i2 = dag.indices[usize(cl.indexOffset) + t * 3 + 2];
    if (i0 >= nV || i1 >= nV || i2 >= nV) {
      log = "cluster " + std::to_string(cluster) + " index out of range";
      return false;
    }
    maxSpan =
        std::max({maxSpan, std::max({i0, i1, i2}) - std::min({i0, i1, i2})});
  }
  const u32 offsetBits = maxSpan == 0 ? 0u : u32(std::bit_width(u64(maxSpan)));
  out.meta.idxBits = baseBits | (offsetBits << 4);
  const u32 triW = baseBits + 2 * offsetBits;
  BitWriter iw(out.idxWords);
  for (u32 t = 0; t < triCount; ++t) {
    const u32 src[3] = {dag.indices[usize(cl.indexOffset) + t * 3 + 0],
                        dag.indices[usize(cl.indexOffset) + t * 3 + 1],
                        dag.indices[usize(cl.indexOffset) + t * 3 + 2]};
    const u32 r = (src[0] <= src[1] && src[0] <= src[2])
                      ? 0u
                      : (src[1] <= src[2] ? 1u : 2u);
    const u32 base = src[r];
    iw.write(base, baseBits);
    iw.write(src[(r + 1) % 3] - base, offsetBits);
    iw.write(src[(r + 2) % 3] - base, offsetBits);
  }
  out.idxBitCount = triCount * triW;
  return true;
}

void decodeQuantCluster(QuantCluster& q, f32 step, u64 idxBitBase) {
  // D5 mirror - the shader runs these exact f32 ops in this order.
  const u32 nV = q.vertexCount;
  const u32 b[3] = {q.meta.widths & 0xFu, (q.meta.widths >> 4) & 0xFu,
                    (q.meta.widths >> 8) & 0xFu};
  const u64 axisBase[3] = {0, u64(nV) * b[0], u64(nV) * (b[0] + b[1])};

  q.decPos.assign(usize(nV) * 3, 0.0f);
  for (int a = 0; a < 3; ++a)
    for (u32 v = 0; v < nV; ++v) {
      const u32 bits =
          readBits(q.posWords.data(), axisBase[a] + u64(v) * b[a], b[a]);
      const i32 coord = i32(bits) + q.meta.posStart[a];
      q.decPos[usize(v) * 3 + a] = f32(coord) * step;
    }

  q.decNrm.assign(usize(nV) * 3, 0.0f);
  for (u32 v = 0; v < nV; ++v) {
    u8 pair[2] = {q.nrmOct[usize(v) * 2], q.nrmOct[usize(v) * 2 + 1]};
    const glm::vec3 n = octDecode(pair);
    q.decNrm[usize(v) * 3 + 0] = n.x;
    q.decNrm[usize(v) * 3 + 1] = n.y;
    q.decNrm[usize(v) * 3 + 2] = n.z;
  }

  q.decUv.assign(usize(nV) * 2, 0.0f);
  for (int a = 0; a < 2; ++a) {
    const f32 rangeOver255 = q.meta.uvRange[a] / 255.0f;
    for (u32 v = 0; v < nV; ++v)
      q.decUv[usize(v) * 2 + a] =
          q.meta.uvMin[a] + f32(q.uvQ[usize(v) * 2 + a]) * rangeOver255;
  }

  // M7.2 mirror: the shader reads the SAME fields in the SAME order
  // (triangle t at meta-absolute bit idxBitBase + t*W; the per-cluster
  // stream here starts at bit 0 - packPages concatenates bit-exactly).
  const u32 baseBits = q.meta.idxBits & 0xFu;
  const u32 offsetBits = (q.meta.idxBits >> 4) & 0xFu;
  const u32 triW = baseBits + 2 * offsetBits;
  const u32 triCount = q.indexCount / 3;
  q.decIdx.resize(q.indexCount);
  for (u32 t = 0; t < triCount; ++t) {
    const u64 bit = idxBitBase + u64(t) * triW;
    const u32 base = readBits(q.idxWords.data(), bit, baseBits);
    const u32 o1 = readBits(q.idxWords.data(), bit + baseBits, offsetBits);
    const u32 o2 =
        readBits(q.idxWords.data(), bit + baseBits + offsetBits, offsetBits);
    q.decIdx[t * 3 + 0] = base;
    q.decIdx[t * 3 + 1] = base + o1;
    q.decIdx[t * 3 + 2] = base + o2;
  }
}

void octEncode(f32 nx, f32 ny, f32 nz, u8 out[2]) {
  // meshopt_encodeFilterOct's scalar formula at 8 bits.
  const f32 nl = std::fabs(nx) + std::fabs(ny) + std::fabs(nz);
  const f32 ns = nl == 0.0f ? 0.0f : 1.0f / nl;
  nx *= ns;
  ny *= ns;
  const f32 u =
      (nz >= 0.0f) ? nx : (1.0f - std::fabs(ny)) * (nx >= 0.0f ? 1.0f : -1.0f);
  const f32 v =
      (nz >= 0.0f) ? ny : (1.0f - std::fabs(nx)) * (ny >= 0.0f ? 1.0f : -1.0f);
  const int fu = meshopt_quantizeSnorm(u, 8);
  const int fv = meshopt_quantizeSnorm(v, 8);
  out[0] = u8(fu);
  out[1] = u8(fv);
}

glm::vec3 octDecode(u8 q[2]) {
  // meshopt decodeFilterOct's int stage with fo = 127 (snorm-8), then
  // the shader's normalize.
  const f32 x = f32(i8(q[0]));
  const f32 y = f32(i8(q[1]));
  f32 z = 127.0f - std::fabs(x) - std::fabs(y);
  const f32 t = z >= 0.0f ? 0.0f : z;
  const f32 xe = x + (x >= 0.0f ? t : -t);
  const f32 ye = y + (y >= 0.0f ? t : -t);
  return glm::normalize(glm::vec3(xe, ye, z));
}

}  // namespace geometry

KENSHIN_NS_END
