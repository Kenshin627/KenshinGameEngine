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

  // ---- M6.1: the paged variant (.kgeo v3) -----------------------------
  // `paged` dags carry the full table/hierarchy set (culling data is
  // complete) but EMPTY pools - the geometry lives in pages on disk,
  // read on demand. cluster.vertexOffset/indexOffset are PAGE-LOCAL in
  // this mode (pageMeta[clusterPage[c]] carries the section bases);
  // they are pool-global in the unpaged v2 mode.
  bool paged = false;
  /// v4 paged: per-cluster max triangle edge in object space (the
  /// HW/SW routing's EdgeLength, computed by the writer from the full
  /// pools; empty = absent (v3) - fall back to the sphere 2r bound).
  std::vector<f32> clusterMaxEdge;
  u32 pageCount = 0;
  u32 pageSlotSize = 0;
  bool pagesHaveNormals = false;
  bool pagesHaveUvs = false;
  std::vector<u32> clusterPage;  // per cluster -> page index
  struct DagPageMeta {
    u32 posBase = 0;  // float4 elements
    u32 nrmBase = 0;  // float4 elements
    u32 uvBase = 0;   // float2 elements
    u32 idxBase = 0;  // u32 elements
    u32 vertexCount = 0;
    u32 indexCount = 0;
  };
  std::vector<DagPageMeta> pageMeta;
  /// v5 (M7.1): the quantized page header (12 u32, uniform u32 units).
  struct DagPageMetaV5 {
    u32 posBase = 0, nrmBase = 0, uvBase = 0, idxBase = 0, metaBase = 0;
    u32 vertexCount = 0, indexCount = 0, clusterCount = 0;
    f32 posStep = 1.0f;
    u32 flags = 0, pad0 = 0, pad1 = 0;
  };
  std::vector<DagPageMetaV5> pageMetaV5;
  /// v5: true when the paged blobs are quantized (the runtime gate);
  /// v3/v4 assets (false) are rejected by the scene loader.
  bool pagesQuantized = false;
  /// v6 (M7.2): true when the idx sections are PACKED index bit
  /// streams (per-cluster widths in the meta word 10). v5 pages are
  /// quantized but carry u8-flat indices - the paged RUNTIME requires
  /// this flag; the shaders only decode the packed format.
  bool pageIdxPacked = false;
  /// v5: per-cluster ordinal into its page's clusterMeta array.
  std::vector<u32> clusterMetaIndex;
  /// v7 (M8.2): the material table. texPath is the disk-side albedo
  /// reference ("" = untextured); texLayer is RUNTIME-assigned by the
  /// TextureManager's load order (-1 until the app resolves it).
  struct DagMaterial {
    f32 kd[3] = {0.75f, 0.75f, 0.78f};
    f32 texLayer = -1.0f;
    f32 roughness = 0.5f;
    f32 metallic = 0.0f;
    std::string texPath;
  };
  std::vector<DagMaterial> materials;
  /// Per-cluster material id into `materials` (u8 range; stamped into
  /// the clusterMeta widths word's free high bits at pack time).
  std::vector<u32> clusterMaterial;
  /// v7: true when the file carried a material table (the runtime
  /// paged gate; pre-v7 assets are rejected with a rebuild hint).
  bool hasMaterialTable = false;

  // ---- M9.2: the far-field imposter atlas ---------------------------
  /// v9: the root cut's 12x12x12x12 (depth-bits, triangle-id) bake.
  /// Empty when the file predates v9 or the bake was skipped - the
  /// runtime falls back to the cluster path (no gate; Stage B treats
  /// an empty atlas as "never imposter this mesh").
  std::vector<u32> imposterDepth;  // reversed depth bits, 0 = miss
  std::vector<u32> imposterTri;    // triangle id, 0xFFFFFFFF = miss
  /// The bake's cut cluster (single-cluster cuts only, v1): the
  /// payload's pair target. SINGLE SOURCE OF TRUTH - the runtime must
  /// not recompute the cut (the builder/runtime definitions drifted:
  /// dag.roots vs deepest-level, 9-cluster vs 1-cluster bakes).
  u32 imposterCluster = 0xFFFFFFFFu;

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
