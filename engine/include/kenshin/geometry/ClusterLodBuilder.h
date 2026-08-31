#pragma once

// ── Cluster LOD builder (S3.1) ───────────────────────────────────
// Thin wrapper over the vendored clusterlod.h micro-library (meshoptimizer
// demo) which runs the full build loop: clusterize -> partition -> lock
// borders -> simplify -> reclusterize -> recurse, emitting one group per DAG
// level through a callback. Raw clusterlod types stay in the .cpp.

#include "kenshin/assets/MeshData.h"

#include <functional>
#include <string>
#include <vector>

KENSHIN_NS_BEGIN

namespace geometry
{

    /// One cluster emitted by the builder. Indices refer to the ORIGINAL
    /// mesh vertex buffer (SOA arrays below copied by the caller).
    struct Cluster
    {
        int parentGroupId = -1;      // clodCluster::refined
        float boundsCenter[4] = {};  // xyz + radius
        float error = 0.0f;          // object-space simplification error
        std::vector<u32> indices;    // meshopt triangle indices (3 per tri)
        u32 vertexCount = 0;         // unique vertices referenced by indices
    };

    /// One group = one partition at one DAG level.
    struct ClusterGroup
    {
        int groupId = -1;            // value our callback returned
        int depth = 0;               // DAG level
        float simplifiedCenter[4] = {};
        float simplifiedError = 0.0f;
        std::vector<Cluster> clusters;
    };

    struct BuildStats
    {
        u32 groupCount = 0;
        u32 clusterCount = 0;
        u32 maxDepth = 0;
        u32 totalTriangles = 0;      // across all levels
        float maxError = 0.0f;
        double seconds = 0.0;
    };

    /// Run the cluster-LOD build over `mesh`. Calls `emit` per group.
    /// Returns false on failure with a message in `log`.
    bool buildClusterLod(const assets::MeshData& mesh,
                         const std::function<void(const ClusterGroup&)>& emit,
                         BuildStats& stats, std::string& log,
                         u32 maxTriangles = 128);

} // namespace geometry

KENSHIN_NS_END
