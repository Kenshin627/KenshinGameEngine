#pragma once

// ── .kgeo file format (S3.3, v2 = M5.1, v3 = M6.1) ──────────────
// Serialized ClusterDag. Little-endian, 16-byte-aligned sections,
// version-tagged. Layout (v2, "KGO2"):
//
//   [Header]
//   [DagSectionHeader] counts
//   [DagGroup]      * groupCount
//   [DagClusterDisk] * clusterCount     (children via spans, see below)
//   [u32 childIndex] * totalChildEdges
//   [u32 memberIndex] * totalMemberEdges
//   [u32 parentIndex] * totalParentEdges
//   [DagHierarchyNode] * hierarchyNodeCount   (v2: the group BVH8, 64 B each)
//   [u32 rootIndex]  * rootCount
//   [f32 positions]  * poolVertexCount * 3
//   [f32 normals]    * poolVertexCount * 3   (omitted when hasNormals=0)
//   [f32 uvs]        * poolVertexCount * 2   (omitted when hasUvs=0)
//   [u32 indices]    * poolIndexCount
//
// Cluster children are stored as (childSpanOffset, childSpanCount) slices
// into one concatenated child-edge array (span offsets are end-inclusive
// packed: offset[i] = start, count[i] = length).
// v1 files are rejected outright - assets are regenerable via kgeo_builder.
//
// v3 ("KGO3", M6.1 - paged): the same table/hierarchy sections, then the
// page machinery INSTEAD of the pools:
//
//   [Header v3]      (+ pageCount, pageSlotSize)
//   [DagGroup] / [DagClusterDisk] / edges / hierarchy / roots (as v2)
//   [u32 clusterPage]        * clusterCount   (cluster -> page)
//   [u32 pageDir]            * pageCount * 2  (fileOffset, byteLen)
//   [u32 pageMeta]           * pageCount * 6  (pos/nrm/uv/idx element
//                                               bases + vertex/index counts)
//   [page blobs]             GPU layout per page:
//                             [vec4 positions | vec4 normals | float2 uvs |
//                              u32 indices], coarsest pages first
//
// ClusterDisk.vertexOffset/indexOffset are PAGE-LOCAL in v3 (element
// offsets within the page's sections). The culling tables are complete
// without the pages - the resident set drives everything the traversal
// needs (nanite.md D14.4). Pages are read on demand via KGeoPageReader.

#include <string>
#include <vector>

#include "kenshin/geometry/ClusterDag.h"
#include "kenshin/geometry/PagePacker.h"

KENSHIN_NS_BEGIN

namespace geometry {

bool writeKGeo(const std::string& path, const ClusterDag& dag,
               std::string& log);
bool readKGeo(const std::string& path, ClusterDag& dag, std::string& log);

/// v3 writer: `pg` from packPages over `dag` (whose pools must still be
/// populated - the packer reads them; the file gets the page-local
/// offsets).
/// M7.2b: `compress` rides the meshopt codec over the page blobs on
/// disk (v8; the reader transparently transcodes back to the v7 byte
/// layout). A page the codec cannot shrink is stored raw.
bool writeKGeoPaged(const std::string& path, const ClusterDag& dag,
                     const PackedGeometry& pg, std::string& log,
                     bool compress = true);

/// On-demand page reader for v3 files (the streaming IO side). Holds
/// the FILE* open; readPage seeks to the page's directory offset.
class KGeoPageReader {
 public:
  ~KGeoPageReader() { close(); }

  bool open(const std::string& path, std::string& log);
  /// Reads page `page` into `out` (resized to the page's byteLen).
  bool readPage(u32 page, std::vector<u8>& out, std::string& log);
  void close();

  u32 pageCount() const { return m_pageCount; }
  u32 pageSlotSize() const { return m_slotSize; }
  bool hasNormals() const { return m_hasNormals; }
  bool hasUvs() const { return m_hasUvs; }

 private:
  std::string m_path;  // v1 reopens per readPage (tests); the M6.3 IO
                       // thread keeps its own long-lived handle
  u32 m_pageCount = 0;
  u32 m_slotSize = 0;
  bool m_hasNormals = false;
  bool m_hasUvs = false;
  std::vector<u64> m_offsets;  // per page: fileOffset, byteLen pairs
  std::vector<u64> m_rawLen;  // M7.2b: decoded length (v7: == byteLen)
  bool m_compressed = false;
};

}  // namespace geometry

KENSHIN_NS_END
