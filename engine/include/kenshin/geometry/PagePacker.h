#pragma once

// ── PagePacker — M6.1/M7.1/M7.2: pack a ClusterDag into fixed pages ─
// nanite.md D14 + §17/D9: fixed 128 KB slots, cluster-granularity
// packing, coarsest-first page order (D14.3's root-page safety net).
// M7.1 v5: the page payload is QUANTIZED (see ClusterQuant) -
//
//   [clusterMeta 10 u32/cluster][posBits u32 words][nrmOct 2 B/vertex]
//   [uvQ 2 B/vertex][idxBits: M7.2 packed index bit stream, words
//    4B-padded + 1 zero sentinel word]
//
// all section bases in uniform u32 units. Positions live on the mesh's
// GLOBAL 2^-P integer grid with per-cluster integer anchors (D9.2's
// anti-crack discipline - shared border vertices decode bit-identically
// across clusters), so the dag's cull bounds are padded by one grid
// step at pack time and maxEdge is computed from the DECODED positions
// (routing sees what actually rasterizes).
//
// `PackedGeometry` also carries the D5-exact decoded mirror pools in
// DAG order - the v2-shaped reference the tests' reference chain and
// the writer consume.

#include <string>
#include <vector>

#include "kenshin/Common.h"
#include "kenshin/geometry/ClusterDag.h"
#include "kenshin/geometry/ClusterQuant.h"

KENSHIN_NS_BEGIN

namespace geometry {

/// D14.1 fixed page size (tunable constant).
constexpr u32 kPageSlotSize = 128u * 1024u;

struct PackedGeometry {
  u32 pageCount = 0;
  bool hasNormals = false;
  bool hasUvs = false;
  std::vector<u32> clusterPage;        // per cluster -> page index
  std::vector<u32> localVertexOffset;  // page vertex ordinal (nrm/uv)
  std::vector<u32> localIndexOffset;   // page idx-section BIT base
                                       // (v6; the shaders read the bit
                                       // base from the meta word)
  std::vector<ClusterDag::DagPageMetaV5> pageMeta;
  std::vector<std::vector<u8>> pages;  // GPU-layout blobs

  // ---- v5 quantization ----------------------------------------------
  QuantMeshGrid grid;
  /// cluster -> ordinal within its page's clusterMeta array.
  std::vector<u32> clusterMetaIndex;
  /// D5-exact decoded mirror pools in DAG cluster order (v2-shaped
  /// reference: identical slicing discipline to dag.positions etc.).
  std::vector<f32> decodedPositions;  // 3/vertex
  std::vector<f32> decodedNormals;    // 3/vertex
  std::vector<f32> decodedUvs;        // 2/vertex
  std::vector<u32> decodedIndices;
};

/// Pack + quantize `dag`. The dag is MUTATED: cull bounds (cluster and
/// group spheres, hierarchy AABBs) padded by one grid step so they
/// provably cover the decoded geometry; positions/indices untouched.
/// Page-local offsets land in `out` (indexOffset semantics: BIT base
/// into the page's packed idx section).
bool packPages(ClusterDag& dag, PackedGeometry& out, std::string& log);

/// Dequantize-aware verification (tests + the load path): every
/// cluster's payload, re-decoded from its PAGE BLOB, must match the
/// decoded mirror bit-for-bit and the dag pools within the quantizer's
/// error bounds (position <= step/2, normal <= 1.5 deg, uv <=
/// range/508, indices exact under the winding-preserving cyclic
/// rotation) - plus conservation and watertight checks.
bool verifyPacking(const ClusterDag& dag, const PackedGeometry& pg,
                   std::string& log);

}  // namespace geometry

KENSHIN_NS_END
