#pragma once

// ── Cluster DAG (S3.2) ───────────────────────────────────────────
// Engine-side metadata layer over the clusterlod output: parent/child edges
// (multi-parent by construction), per-cluster bounds/error with the
// monotonic-error enforcement (unique LOD cut guarantee), group-level shared
// decision data (same input => same output across the group), and
// cluster-local vertex/index pools ready for GPU upload and .kgeo
// serialization.

#include <glm/glm.hpp>
#include <string>
#include <vector>

#include "kenshin/Common.h"
#include "kenshin/assets/MeshData.h"

KENSHIN_NS_BEGIN

namespace geometry {

struct ClusterDag;

/// One cluster in the DAG. Vertices/indices are cluster-local: the pools
/// in the owning ClusterDag hold the actual data, this node slices them.
struct DagCluster {
  // Bounds (object space), shared by every cluster of the same group so
  // parallel LOD decisions agree (same input => same output).
  glm::vec4 boundsSphere = {};  // xyz center, w radius
  f32 error = 0.0f;             // own simplification error
  f32 parentError = 0.0f;       // max(error, children errors) - monotonic

  // Cluster-local geometry slices into the DAG pools.
  u32 vertexOffset = 0;
  u32 vertexCount = 0;
  u32 indexOffset = 0;
  u32 indexCount = 0;  // 3 per triangle

  // Group membership and DAG edges.
  u32 group = 0;              // -> ClusterDag::groups
  std::vector<u32> children;  // finer clusters this one replaces
  bool isRoot = false;        // no parent (level-0 refinement target)

  // clodCluster::refined: THE group this cluster was produced from during
  // simplification (-1 = original geometry). The cut's second condition
  // checks ONLY this source (clusterlod.h contract) - per-cluster, fully
  // local/parallel. The all-parents union lives on the group (parents) for
  // traversal only; judging a cluster by ALL parents' thresholds blocks it
  // for regions its own source could still cover (the missing-strand bug).
  i32 refinedGroup = -1;
};

/// One group: clusters that must make identical LOD decisions.
struct DagGroup {
  i32 lodLevel = 0;             // 0 = original geometry
  glm::vec4 boundsSphere = {};  // union of member bounds (decision space)
  f32 maxError = 0.0f;          // monotonic decision error for the group
  /// clusterlod semantics: error of coarsening FROM this group
  /// (FLT_MAX on terminal groups - nothing coarser exists).
  f32 simplifyError = 0.0f;
  std::vector<u32> clusters;  // member cluster indices
  std::vector<u32> parents;   // groups whose clusters refined into us
};

/// M5.1: BVH8 node over ALL groups - the spatial acceleration view for
/// hierarchical culling (Nanite builds its BVH over ClusterGroupParts; the
/// DAG itself stays untouched - "any grouping structure works" since the LOD
/// test only needs each node's own error data). Leaf = one group:
/// childCount == 0 and children[0] = the group index; that group's clusters
/// are a contiguous range of ClusterDag::clusters (assembly appends them
/// per group, verifyDagInvariants enforces it) - M5.2's traversal emits
/// pairs from the range without any member-index upload.
/// The AABB conservatively boxes the subtree's group bounds SPHERES, and
/// maxSimplifyError is the max over the subtree (terminal groups hold
/// FLT_MAX, so subtrees containing them can never be LOD-pruned - exactly
/// the "cut must reach the roots" safety). Flat 64 B, GPU-upload ready;
/// bit packing is M7's business.
struct DagHierarchyNode {
  f32 aabbMin[3] = {0.0f, 0.0f, 0.0f};
  f32 aabbMax[3] = {0.0f, 0.0f, 0.0f};
  f32 maxSimplifyError = 0.0f;
  u32 childCount = 0;    // 0 = leaf
  u32 children[8] = {};  // node indices; leaf: children[0] = group index
};

struct ClusterDag {
  std::vector<DagCluster> clusters;
  std::vector<DagGroup> groups;
  std::vector<u32> roots;  // root cluster indices

  // M5.1: the BVH8 over `groups` (single root; empty before M5.1 assembly).
  std::vector<DagHierarchyNode> hierarchyNodes;
  u32 hierarchyRoot = 0xFFFFFFFFu;

  // Cluster-local pools (positions/normals/uvs interleaved per vertex).
  std::vector<f32> positions;  // 3 per vertex
  std::vector<f32> normals;    // 3 per vertex
  std::vector<f32> uvs;        // 2 per vertex
  std::vector<u32> indices;    // cluster-local, 3 per triangle

  // Source mesh stats.
  u32 sourceVertexCount = 0;
  u32 sourceTriangleCount = 0;

  // ---- queries -------------------------------------------------------
  u32 triangleCount(u32 cluster) const {
    return clusters[cluster].indexCount / 3;
  }
};

/// Assembly result.
struct DagBuildStats {
  u32 clusterCount = 0;
  u32 groupCount = 0;
  u32 rootCount = 0;
  u32 maxLevel = 0;
  u32 monotonicFixes = 0;  // clusters whose parentError we raised
  double seconds = 0.0;
};

/// Assemble a ClusterDag from a built group stream (from
/// buildClusterLod's emit callback order: groups arrive in build order,
/// groupId >= 0 identifies each; parent linkage via Cluster::parentGroupId).
/// `mesh` supplies the vertex attributes referenced by cluster indices.
/// Also builds the group BVH8 (M5.1). On failure returns false with a
/// message in `log`.
bool assembleClusterDag(const kenshin::assets::MeshData& mesh,
                        const std::vector<struct ClusterGroup>& builtGroups,
                        ClusterDag& dag, DagBuildStats& stats,
                        std::string& log);

/// Build the BVH8 over `dag.groups` into `dag.hierarchyNodes/hierarchyRoot`
/// (called by assembleClusterDag after the dominance pass; exposed for
/// tests). Longest-axis median split into up to 8 contiguous chunks per
/// node, leaf = single group. No SAH yet - measure first (M5 plan).
bool buildGroupHierarchy(ClusterDag& dag, std::string& log);

/// Structural invariants (used by tests and the LOD preview gate).
bool verifyDagInvariants(const ClusterDag& dag, std::string& log);

}  // namespace geometry

KENSHIN_NS_END
