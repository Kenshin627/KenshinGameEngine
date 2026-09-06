#include "kenshin/geometry/ClusterDag.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <queue>

#include "kenshin/geometry/ClusterLodBuilder.h"

KENSHIN_NS_BEGIN

namespace geometry {

namespace {
glm::vec4 unionSphere(const glm::vec4& a, const glm::vec4& b) {
  if (b.w < 0.0f) return a;
  if (a.w < 0.0f) return b;
  const glm::vec3 pa(a), pb(b);
  const f32 d = glm::distance(pa, pb);
  if (d + b.w <= a.w) return a;  // b inside a
  if (d + a.w <= b.w) return b;  // a inside b
  const f32 t = (d + a.w - b.w) * 0.5f;
  const glm::vec3 dir = d > 0.0f ? (pb - pa) / d : glm::vec3(0.0f);
  const glm::vec3 center = pa + dir * t;
  return glm::vec4(center, (a.w + b.w + d) * 0.5f);
}
}  // namespace

namespace {
/// M5.1 BVH8 builder: longest-axis median split into up to 8 contiguous
/// chunks, recursing to single-group leaves. Returns the new node's index
/// (children land at lower indices, the root last).
u32 buildHierarchyNode(std::vector<u32> groupIdx, const ClusterDag& dag,
                       std::vector<DagHierarchyNode>& nodes) {
  const f32 kMaxF = std::numeric_limits<f32>::max();
  glm::vec3 bmin(kMaxF), bmax(-kMaxF);
  f32 maxErr = 0.0f;
  for (u32 g : groupIdx) {
    const glm::vec4& s = dag.groups[g].boundsSphere;
    bmin = glm::min(bmin, glm::vec3(s) - s.w);
    bmax = glm::max(bmax, glm::vec3(s) + s.w);
    maxErr = std::max(maxErr, dag.groups[g].simplifyError);
  }

  DagHierarchyNode node;
  node.aabbMin[0] = bmin.x, node.aabbMin[1] = bmin.y, node.aabbMin[2] = bmin.z;
  node.aabbMax[0] = bmax.x, node.aabbMax[1] = bmax.y, node.aabbMax[2] = bmax.z;
  node.maxSimplifyError = maxErr;

  if (groupIdx.size() == 1) {
    node.childCount = 0;
    node.children[0] = groupIdx[0];
  } else {
    int axis = 0;
    const f32 extent[3] = {bmax.x - bmin.x, bmax.y - bmin.y, bmax.z - bmin.z};
    if (extent[1] > extent[axis]) axis = 1;
    if (extent[2] > extent[axis]) axis = 2;
    std::sort(groupIdx.begin(), groupIdx.end(), [&](u32 a, u32 b) {
      return dag.groups[a].boundsSphere[axis] <
             dag.groups[b].boundsSphere[axis];
    });
    const usize n = groupIdx.size();
    const u32 k = static_cast<u32>(std::min<usize>(8, n));
    node.childCount = k;
    for (u32 i = 0; i < k; ++i) {
      const usize begin = i * n / k, end = (i + 1) * n / k;
      node.children[i] = buildHierarchyNode(
          std::vector<u32>(groupIdx.begin() + begin, groupIdx.begin() + end),
          dag, nodes);
    }
  }
  nodes.push_back(node);
  return static_cast<u32>(nodes.size() - 1);
}
}  // namespace

bool buildGroupHierarchy(ClusterDag& dag, std::string& log) {
  dag.hierarchyNodes.clear();
  dag.hierarchyRoot = 0xFFFFFFFFu;
  if (dag.groups.empty()) {
    log = "no groups to build the hierarchy from";
    return false;
  }
  std::vector<u32> all(dag.groups.size());
  std::iota(all.begin(), all.end(), 0u);
  dag.hierarchyRoot =
      buildHierarchyNode(std::move(all), dag, dag.hierarchyNodes);
  return true;
}

bool assembleClusterDag(const assets::MeshData& mesh,
                        const std::vector<ClusterGroup>& builtGroups,
                        ClusterDag& dag, DagBuildStats& stats,
                        std::string& log) {
  if (builtGroups.empty()) {
    log = "no groups provided";
    return false;
  }

  dag = {};
  dag.sourceVertexCount = mesh.vertexCount();
  dag.sourceTriangleCount = mesh.triangleCount();

  // Pass 1: clusters + groups; geometry into cluster-local pools with
  // per-cluster vertex dedup.
  dag.groups.resize(builtGroups.size());
  for (u32 g = 0; g < builtGroups.size(); ++g) {
    const ClusterGroup& src = builtGroups[g];
    DagGroup& dst = dag.groups[g];
    dst.lodLevel = src.depth;
    dst.simplifyError = src.simplifiedError;
    for (int k = 0; k < 4; ++k) dst.boundsSphere[k] = src.simplifiedCenter[k];
    for (usize ci = 0; ci < src.clusters.size(); ++ci) {
      const Cluster& cluster = src.clusters[ci];
      const u32 clusterIndex = static_cast<u32>(dag.clusters.size());

      DagCluster node;
      node.group = g;
      node.refinedGroup = cluster.parentGroupId;  // clodCluster::refined
      for (int k = 0; k < 4; ++k)
        node.boundsSphere[k] = cluster.boundsCenter[k];
      node.error = cluster.error;
      node.vertexOffset = static_cast<u32>(dag.positions.size() / 3);
      node.indexOffset = static_cast<u32>(dag.indices.size());

      std::map<u32, u32> localOf;  // source vertex -> local index
      for (u32 index : cluster.indices) {
        auto it = localOf.find(index);
        if (it == localOf.end()) {
          it = localOf.emplace(index, static_cast<u32>(localOf.size())).first;
          const usize v = index;
          for (int k = 0; k < 3; ++k)
            dag.positions.push_back(mesh.positions[v * 3 + k]);
          if (mesh.normals.size() == mesh.positions.size())
            for (int k = 0; k < 3; ++k)
              dag.normals.push_back(mesh.normals[v * 3 + k]);
          if (mesh.uvs.size() / 2 == mesh.vertexCount()) {
            dag.uvs.push_back(mesh.uvs[v * 2]);
            dag.uvs.push_back(mesh.uvs[v * 2 + 1]);
          } else {
            dag.uvs.push_back(0.0f);
            dag.uvs.push_back(0.0f);
          }
          node.vertexCount += 1;
        }
        dag.indices.push_back(it->second);
      }
      node.indexCount = static_cast<u32>(cluster.indices.size());
      dag.clusters.push_back(std::move(node));
      dst.clusters.push_back(clusterIndex);
    }
  }

  // Pass 2: DAG edges. Cluster::parentGroupId holds the group id OUR
  // callback returned (= index in builtGroups). Per the clusterlod
  // contract every cluster of a child group descends from the whole
  // parent group, so link parent group's every cluster -> each child
  // (Nanite "all siblings connect to all parents").
  for (u32 g = 0; g < builtGroups.size(); ++g) {
    const ClusterGroup& src = builtGroups[g];
    for (usize ci = 0; ci < src.clusters.size(); ++ci) {
      const Cluster& childMeta = src.clusters[ci];
      if (childMeta.parentGroupId < 0) continue;
      const u32 childCluster = dag.groups[g].clusters[ci];
      const u32 parentGroup = static_cast<u32>(childMeta.parentGroupId);
      for (u32 parentCluster : dag.groups[parentGroup].clusters)
        dag.clusters[parentCluster].children.push_back(childCluster);
      dag.groups[g].parents.push_back(parentGroup);
    }
  }
  for (auto& group : dag.groups) {
    std::sort(group.parents.begin(), group.parents.end());
    group.parents.erase(std::unique(group.parents.begin(), group.parents.end()),
                        group.parents.end());
  }

  // Roots: terminal clusters (nobody refines them further).
  std::vector<bool> isRefined(dag.clusters.size(), false);
  for (const auto& cluster : dag.clusters)
    for (u32 child : cluster.children) isRefined[child] = true;
  for (u32 c = 0; c < dag.clusters.size(); ++c)
    if (!isRefined[c]) {
      dag.clusters[c].isRoot = true;
      dag.roots.push_back(c);
    }

  // Pass 3: group decision data (union bounds + max member error).
  for (auto& group : dag.groups) {
    glm::vec4 bounds(0.0f, 0.0f, 0.0f, -1.0f);  // invalid seed
    f32 maxErr = 0.0f;
    for (u32 c : group.clusters) {
      bounds = unionSphere(bounds, dag.clusters[c].boundsSphere);
      maxErr = std::max(maxErr, dag.clusters[c].error);
    }
    group.boundsSphere = bounds;
    group.maxError = maxErr;
  }

  // Pass 4: monotonic error propagation, deepest groups first (build
  // order is non-decreasing depth, so reverse iterate):
  //   parentError(c) = max(error(c), parentError(children), group.maxError)
  // Members of a group all end with the same parentError (shared
  // decision data => identical parallel LOD decisions).
  for (usize gi = dag.groups.size(); gi-- > 0;) {
    DagGroup& group = dag.groups[gi];
    f32 subtreeError = group.maxError;
    for (u32 c : group.clusters) {
      f32 childMax = dag.clusters[c].error;
      for (u32 child : dag.clusters[c].children)
        childMax = std::max(childMax, dag.clusters[child].parentError);
      subtreeError = std::max(subtreeError, childMax);
    }
    group.maxError = std::max(group.maxError, subtreeError);
    for (u32 c : group.clusters) {
      const f32 raised = std::max(dag.clusters[c].error, group.maxError);
      if (raised > dag.clusters[c].parentError) ++stats.monotonicFixes;
      dag.clusters[c].parentError = raised;
    }
  }

  // Pass 5 (Nanite build-time dominance, Karis SIGGRAPH 2021: "modify the
  // parent's stored error AND projected bounds, both >= child's"): a
  // coarser group's bounds sphere and simplify error must dominate every
  // finer group's (DagGroup.parents points to the FINER groups that
  // refined into this one). The runtime cut evaluates each node against
  // ITS OWN projected error/threshold; monotone stored data makes that
  // projection monotone along chains for ANY view, so the first-fireable
  // level exists everywhere (full coverage) by construction. Pass 3 only
  // unions SAME-LEVEL member spheres - a parent sphere that misses a
  // child's pokes breaks the monotonicity and left whole strands
  // unselected. Groups are emitted in non-decreasing coarsening depth and
  // parent edges point to lower indices, so one forward pass reads
  // already-raised spheres.
  for (u32 g = 0; g < dag.groups.size(); ++g) {
    DagGroup& group = dag.groups[g];
    for (u32 parentGroup : group.parents) {
      const DagGroup& finer = dag.groups[parentGroup];
      group.boundsSphere = unionSphere(group.boundsSphere, finer.boundsSphere);
      group.simplifyError = std::max(group.simplifyError, finer.simplifyError);
      group.maxError = std::max(group.maxError, finer.maxError);
    }
  }

  // Pass 6 (M5.1): the BVH8 over all groups - spatial acceleration view
  // for hierarchical culling. Built after Pass 5 so the dominance-raised
  // bounds/simplifyErrors are what the node AABBs and error bounds wrap.
  if (!buildGroupHierarchy(dag, log)) return false;

  stats.clusterCount = static_cast<u32>(dag.clusters.size());
  stats.groupCount = static_cast<u32>(dag.groups.size());
  stats.rootCount = static_cast<u32>(dag.roots.size());
  for (const auto& group : dag.groups)
    stats.maxLevel =
        std::max(stats.maxLevel, static_cast<u32>(group.lodLevel + 1));
  return verifyDagInvariants(dag, log);
}

bool verifyDagInvariants(const ClusterDag& dag, std::string& log) {
  // 1) Acyclic (Kahn over the cluster graph).
  {
    std::vector<u32> indegree(dag.clusters.size(), 0);
    for (const auto& cluster : dag.clusters)
      for (u32 child : cluster.children) ++indegree[child];
    std::queue<u32> queue;
    for (u32 c = 0; c < dag.clusters.size(); ++c)
      if (indegree[c] == 0) queue.push(c);
    u32 visited = 0;
    while (!queue.empty()) {
      const u32 c = queue.front();
      queue.pop();
      ++visited;
      for (u32 child : dag.clusters[c].children)
        if (--indegree[child] == 0) queue.push(child);
    }
    if (visited != dag.clusters.size()) {
      log = "cycle detected: visited " + std::to_string(visited) + " of " +
            std::to_string(dag.clusters.size()) + " clusters";
      return false;
    }
  }

  // 2) Monotonic error along every parent->child edge.
  for (const auto& cluster : dag.clusters) {
    if (cluster.parentError + 1e-6f < cluster.error) {
      log = "parentError < own error";
      return false;
    }
    for (u32 child : cluster.children)
      if (cluster.parentError + 1e-6f < dag.clusters[child].parentError) {
        log = "parentError(parent) < parentError(child)";
        return false;
      }
  }

  // 3) Same-group clusters share the decision data.
  for (const auto& group : dag.groups) {
    if (group.clusters.empty()) continue;
    const DagCluster& first = dag.clusters[group.clusters.front()];
    for (u32 c : group.clusters)
      if (std::fabs(dag.clusters[c].parentError - first.parentError) > 1e-6f) {
        log = "group members disagree on parentError";
        return false;
      }
  }

  // 4) Pools consistent with the cluster slices.
  if (dag.positions.size() % 3 != 0 || dag.indices.size() % 3 != 0 ||
      (!dag.normals.empty() && dag.normals.size() != dag.positions.size()) ||
      dag.uvs.size() / 2 != dag.positions.size() / 3) {
    log = "pool size mismatch";
    return false;
  }
  for (const auto& cluster : dag.clusters) {
    if (cluster.vertexOffset + cluster.vertexCount > dag.positions.size() / 3 ||
        cluster.indexOffset + cluster.indexCount > dag.indices.size()) {
      log = "cluster slice out of pool bounds";
      return false;
    }
    for (usize i = cluster.indexOffset;
         i < cluster.indexOffset + cluster.indexCount; ++i)
      if (dag.indices[i] >= cluster.vertexCount) {
        log = "cluster-local index out of vertex range";
        return false;
      }
  }
  // 5) Clusters are contiguous per group (Pass 1 appends them per group;
  // M5.2's traversal derives group -> cluster ranges from this).
  for (u32 c = 1; c < dag.clusters.size(); ++c)
    if (dag.clusters[c].group < dag.clusters[c - 1].group) {
      log = "clusters not contiguous per group at " + std::to_string(c);
      return false;
    }

  // 6) Group hierarchy (M5.1), when present.
  if (!dag.hierarchyNodes.empty()) {
    if (dag.hierarchyRoot >= dag.hierarchyNodes.size()) {
      log = "hierarchy root out of range";
      return false;
    }
    std::vector<u8> visited(dag.hierarchyNodes.size(), 0);
    std::vector<u32> leafGroupHits(dag.groups.size(), 0);
    // Recursive validator: returns the subtree's max simplifyError and
    // checks bounds/children/leaves. f32 pair -> struct to carry both.
    struct SubtreeInfo {
      f32 maxErr;
    };
    std::function<SubtreeInfo(u32)> validate = [&](u32 nodeIdx) -> SubtreeInfo {
      const DagHierarchyNode& node = dag.hierarchyNodes[nodeIdx];
      visited[nodeIdx] = 1;
      if (node.childCount == 0) {
        const u32 g = node.children[0];
        if (g >= dag.groups.size()) throw std::string("leaf group OOR");
        ++leafGroupHits[g];
        const glm::vec4& s = dag.groups[g].boundsSphere;
        for (int k = 0; k < 3; ++k) {
          if (node.aabbMin[k] > s[k] - s.w + 1e-6f ||
              node.aabbMax[k] < s[k] + s.w - 1e-6f)
            throw std::string("leaf aabb misses group sphere");
        }
        return {dag.groups[g].simplifyError};
      }
      if (node.childCount > 8) throw std::string("childCount > 8");
      f32 maxErr = 0.0f;
      for (u32 i = 0; i < node.childCount; ++i) {
        const u32 child = node.children[i];
        if (child >= dag.hierarchyNodes.size()) throw std::string("child OOR");
        if (visited[child]) throw std::string("not a tree (revisit)");
        const DagHierarchyNode& cn = dag.hierarchyNodes[child];
        for (int k = 0; k < 3; ++k) {
          if (node.aabbMin[k] > cn.aabbMin[k] + 1e-6f ||
              node.aabbMax[k] < cn.aabbMax[k] - 1e-6f)
            throw std::string("node aabb misses child aabb");
        }
        maxErr = std::max(maxErr, validate(child).maxErr);
      }
      if (std::fabs(node.maxSimplifyError - maxErr) > 1e-6f &&
          !(node.maxSimplifyError == maxErr))  // FLT_MAX exact match
        throw std::string("maxSimplifyError mismatch");
      return {maxErr};
    };
    try {
      validate(dag.hierarchyRoot);
      for (u32 n = 0; n < visited.size(); ++n)
        if (!visited[n]) {
          log = "hierarchy node unreachable: " + std::to_string(n);
          return false;
        }
      for (u32 g = 0; g < leafGroupHits.size(); ++g)
        if (leafGroupHits[g] != 1) {
          log = "group " + std::to_string(g) + " in " +
                std::to_string(leafGroupHits[g]) + " leaves (want 1)";
          return false;
        }
    } catch (const std::string& err) {
      log = "hierarchy: " + err;
      return false;
    }
  }
  return true;
}

}  // namespace geometry

KENSHIN_NS_END
