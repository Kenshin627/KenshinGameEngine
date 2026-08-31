#pragma once

// ── .kgeo file format (S3.3) ─────────────────────────────────────
// Serialized ClusterDag. Little-endian, 16-byte-aligned sections,
// version-tagged. Layout:
//
//   [Header]
//   [DagSectionHeader] counts
//   [DagGroup]      * groupCount
//   [DagClusterDisk] * clusterCount     (children via spans, see below)
//   [u32 childIndex] * totalChildEdges
//   [u32 rootIndex]  * rootCount
//   [f32 positions]  * poolVertexCount * 3
//   [f32 normals]    * poolVertexCount * 3   (omitted when hasNormals=0)
//   [f32 uvs]        * poolVertexCount * 2   (omitted when hasUvs=0)
//   [u32 indices]    * poolIndexCount
//
// Cluster children are stored as (childSpanOffset, childSpanCount) slices
// into one concatenated child-edge array (span offsets are end-inclusive
// packed: offset[i] = start, count[i] = length).

#include "kenshin/geometry/ClusterDag.h"

#include <string>

KENSHIN_NS_BEGIN

namespace geometry
{

    bool writeKGeo(const std::string& path, const ClusterDag& dag,
                   std::string& log);
    bool readKGeo(const std::string& path, ClusterDag& dag, std::string& log);

} // namespace geometry

KENSHIN_NS_END
