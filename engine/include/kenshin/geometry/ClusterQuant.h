#pragma once

// ── ClusterQuant — M7.1: per-cluster geometry quantization ──────────
// nanite.md §17/D8/D9 (UE5-scheme): one GLOBAL power-of-two integer
// grid per mesh (step = 2^-P; NEVER bounds-normalized - D9.2's crack
// discipline), per-cluster integer anchors + per-axis bit widths,
// bit-packed cluster-local positions, octahedral 2x8 normals, per-
// cluster-range 8-bit UVs, flat u8 indices (cluster vertexCount <= 128
// by the builder's config). Decode contract (D5): the CPU mirror and
// the shader perform the SAME f32 ops in the SAME order, so a decoded
// vertex is bit-identical on both sides - the GPU differential tests
// keep their mismatches==0 assertions.
//
//   pos = float3(int3(localBits) + posStart) * step   (exact ints, one
//        f32 multiply by a power of two)
//   nrm = octDecode(i8 pair)  (z = 127 - |x| - |y| fixup + normalize)
//   uv  = uvMin + float2(q) * (uvRange / 255)
//   idx = u8 lane

#include <glm/glm.hpp>

#include <string>
#include <vector>

#include "kenshin/Common.h"
#include "kenshin/geometry/ClusterDag.h"

KENSHIN_NS_BEGIN

namespace geometry {

/// The mesh-wide integer grid. P: positions live on multiples of step.
struct QuantMeshGrid {
  i32 P = 0;
  f32 step = 1.0f;  // 2^-P (exact f32 power of two)
};

/// Mirrors the GPU 10-u32 per-cluster record (see GpuClusterQuant in
/// the shaders - copied, not shared, per project convention).
struct ClusterQuantMeta {
  i32 posStart[3] = {0, 0, 0};  // integer anchor = cluster int AABB min
  u32 widths = 0;               // bx | by<<4 | bz<<8   (each 0..11)
  u32 posWordOff = 0;           // cluster pos block, u32 words into the
                                // page's posBits section
  f32 uvMin[2] = {0.0f, 0.0f};
  f32 uvRange[2] = {0.0f, 0.0f};
  u32 reserved = 0;
};

/// One cluster's packed payload (page-section slices) plus the D5-exact
/// decoded mirror (fills after decodeQuantCluster).
struct QuantCluster {
  u32 vertexCount = 0;
  u32 indexCount = 0;
  ClusterQuantMeta meta;

  // posBits: SOA axis blocks [x: nV*bx | y: nV*by | z: nV*bz] bits,
  // LSB-first, bit-contiguous within the block (axis bases are bit
  // offsets; loadBits handles arbitrary alignment). The PAGE appends
  // one zero sentinel word after the whole section - the two-word
  // combine read may touch it at the tail.
  std::vector<u32> posWords;
  std::vector<u8> nrmOct;  // 2 bytes/vertex
  std::vector<u8> uvQ;     // 2 bytes/vertex
  std::vector<u8> idx;     // 1 byte/index

  // D5-exact decoded mirror (3/vertex, 3/vertex, 2/vertex, raw u32).
  std::vector<f32> decPos;
  std::vector<f32> decNrm;
  std::vector<f32> decUv;
  std::vector<u32> decIdx;
};

/// Choose the mesh grid: UE5's density heuristic (P0 = 7 - round(avg
/// log2(cluster extent))), clamped so every cluster's integer extent
/// fits 11 bits and P lands in [-16, 20].
QuantMeshGrid chooseQuantGrid(const ClusterDag& dag);

/// Quantize one cluster of `dag` under `grid`. Fills the packed fields;
/// call decodeQuantCluster to fill the decoded mirror (kept separate so
/// the roundtrip test decodes from packed bits alone).
bool quantizeCluster(const ClusterDag& dag, u32 cluster,
                     const QuantMeshGrid& grid, bool hasNormals,
                     bool hasUvs, QuantCluster& out, std::string& log);

/// The D5-exact inverse - the CPU mirror of the shader decode. `step`
/// is the mesh grid's 2^-P (carried by the page header, not the
/// per-cluster record).
void decodeQuantCluster(QuantCluster& q, f32 step);

// ---- bit-stream primitives (LSB-first u32 words) --------------------

/// Write `bits` of `value` at the writer's position (0..11 bits; may
/// straddle words; `words` must be sized to cover the final bit).
struct BitWriter {
  std::vector<u32>& words;
  u64 bit = 0;
  explicit BitWriter(std::vector<u32>& w) : words(w) {}
  void write(u32 value, u32 bits) {
    if (bits == 0) return;
    const u64 end = bit + bits;
    // +1 word of headroom: readBits' two-word combine may touch the
    // word after a straddling field.
    const u64 needWords = (end + 31) / 32 + 1;
    if (u64(words.size()) < needWords) words.resize(usize(needWords), 0);
    const usize w = usize(bit >> 5);
    const u32 shift = u32(bit & 31);
    words[w] |= value << shift;
    if (shift + bits > 32) words[w + 1] |= value >> (32 - shift);
    bit = end;
  }
};

/// Read `bits` (0..24) at absolute bit index (two-word combine).
inline u32 readBits(const u32* words, u64 bitIdx, u32 bits) {
  if (bits == 0) return 0;
  const u64 v = u64(words[bitIdx >> 5]) | (u64(words[(bitIdx >> 5) + 1]) << 32);
  return u32((v >> (bitIdx & 31)) & ((1ull << bits) - 1));
}

/// Octahedral encode of a unit normal to a snorm-8 pair (mirrors
/// meshopt_encodeFilterOct's scalar formula, 8 bits).
void octEncode(f32 nx, f32 ny, f32 nz, u8 out[2]);

/// Octahedral decode of a snorm-8 pair (the int-stage mirrors meshopt's
/// decodeFilterOct with fo = 127; normalize is the final f32 op).
glm::vec3 octDecode(u8 q[2]);

}  // namespace geometry

KENSHIN_NS_END
