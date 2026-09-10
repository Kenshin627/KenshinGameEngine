#pragma once

// ── PagePacker — M6.1: pack a ClusterDag's pools into fixed-size pages ──
// nanite.md D14 (self-designed per the milestone plan): fixed 128 KB
// pages, cluster-granularity packing (groups may split across pages),
// the GPU-ready layout inside each page
//   [positions vec4 | normals vec4 | uvs vec2 | indices u32]
// with per-page element bases for the four sections. The v2 pools are
// already cluster-local (per-cluster dedup at assembly), so packing is
// slice concatenation - no cross-cluster dedup exists to do. Clusters
// are assigned to pages in lodLevel-DESCENDING order, so the first
// pages hold the coarsest data: those become the always-resident root
// pages (D14.3's streaming safety net).

#include <array>
#include <string>
#include <vector>

#include "kenshin/Common.h"
#include "kenshin/geometry/ClusterDag.h"

KENSHIN_NS_BEGIN

namespace geometry {

/// D14.1 fixed page size (tunable constant).
constexpr u32 kPageSlotSize = 128u * 1024u;

struct PackedGeometry {
  u32 pageCount = 0;
  bool hasNormals = false;
  bool hasUvs = false;
  std::vector<u32> clusterPage;         // per cluster -> page index
  std::vector<u32> localVertexOffset;   // per cluster, page-local (elements)
  std::vector<u32> localIndexOffset;    // per cluster, page-local (elements)
  std::vector<ClusterDag::DagPageMeta> pageMeta;
  std::vector<std::vector<u8>> pages;   // GPU-layout blobs
};

/// Pack `dag`'s pools into pages. The dag is unchanged (its offsets
/// stay pool-global); the page-local offsets land in `out`.
bool packPages(const ClusterDag& dag, PackedGeometry& out, std::string& log);

/// Structural verification of a packing against its dag (tests + the
/// v3 load path): every cluster's page-local slice, re-extracted from
/// its page blob, must equal the dag pool slice bit-for-bit.
bool verifyPacking(const ClusterDag& dag, const PackedGeometry& pg,
                   std::string& log);

}  // namespace geometry

KENSHIN_NS_END
