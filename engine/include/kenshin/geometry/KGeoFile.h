#pragma once

// ── .kgeo file format (S3.3, v2 = M5.1) ──────────────────────────
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

#include <string>

#include "kenshin/geometry/ClusterDag.h"

KENSHIN_NS_BEGIN

namespace geometry {

bool writeKGeo(const std::string& path, const ClusterDag& dag,
               std::string& log);
bool readKGeo(const std::string& path, ClusterDag& dag, std::string& log);

}  // namespace geometry

KENSHIN_NS_END
