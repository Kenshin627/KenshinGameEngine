#include "kenshin/geometry/ClusterLodBuilder.h"

#include "kenshin/core/Log.h"
#include "kenshin/core/Timer.h"

#pragma warning(push, 0)
// Vendored meshoptimizer (master) - NOT the vcpkg copy (too old for
// SimplifyPreserveFolds that clusterlod.h uses).
#include "meshoptimizer/meshoptimizer.h"
#define CLUSTERLOD_IMPLEMENTATION
#include "../../ThirdParty/clusterlod.h"
#pragma warning(pop)

#include <algorithm>

KENSHIN_NS_BEGIN

namespace geometry {

namespace {
/// Collector fed by the clodBuild callback.
struct Collector {
  std::vector<ClusterGroup>* groups = nullptr;
  BuildStats* stats = nullptr;
};
}  // namespace

bool buildClusterLod(const assets::MeshData& mesh,
                     const std::function<void(const ClusterGroup&)>& emit,
                     BuildStats& stats, std::string& log, u32 maxTriangles) {
  if (mesh.positions.empty() || mesh.indices.empty()) {
    log = "empty mesh";
    return false;
  }
  // v1 (staged plan): position-only simplification so clusterlod's
  // error is a pure world-space geometric distance - exactly what the
  // pixel-threshold LOD rule needs. Attribute-aware simplification
  // returns unitless mixed errors (position + normals + uvs) that
  // cannot be projected to pixels; revisit with calibrated weights.
  const usize vertexCount = mesh.vertexCount();

  clodMesh input = {};
  input.indices = mesh.indices.data();
  input.index_count = mesh.indices.size();
  input.vertex_count = vertexCount;
  input.vertex_positions = mesh.positions.data();
  input.vertex_positions_stride = sizeof(f32) * 3;
  input.vertex_attributes = nullptr;
  input.vertex_attributes_stride = 0;
  input.attribute_weights = nullptr;
  input.attribute_count = 0;

  clodConfig config = clodDefaultConfig(maxTriangles);
  // Conservative v1: permissive/sloppy modes collapse aggressively and
  // return large errors (first-level error saturates the whole DAG via
  // the max() merge); regular QEM keeps errors proportional and useful
  // for the pixel-threshold rule. Revisit when calibrating attributes.
  config.simplify_permissive = false;
  config.simplify_fallback_sloppy = false;

  std::vector<ClusterGroup> groups;
  Collector collector{&groups, &stats};

  // Material boundaries are CLUSTER boundaries: one clodBuild run per
  // material over that material's triangles (same vertex pool). A
  // cluster can then never straddle two materials - the old single run
  // let simplification/partitioning mix them and the per-cluster vote
  // painted the minority material's triangles with the majority's (the
  // Vulture's left eye lens rendered half helmet-gray). Each run's
  // groups form an independent sub-DAG; the callback hands out GLOBAL
  // group ids (indices into the shared collector), so cross-run edges
  // never exist and every per-run ordering invariant the DAG assembly
  // relies on (parents at lower indices, non-decreasing depth) holds
  // run by run. The BVH8 is built over all groups afterwards.
  std::vector<std::vector<u32>> runs;
  if (!mesh.triMaterial.empty() &&
      mesh.triMaterial.size() == mesh.indices.size() / 3) {
    u32 maxMat = 0;
    for (u32 m : mesh.triMaterial) maxMat = std::max(maxMat, m);
    runs.resize(usize(maxMat) + 1);
    for (usize t = 0; t < mesh.triMaterial.size(); ++t)
      for (int k = 0; k < 3; ++k)
        runs[mesh.triMaterial[t]].push_back(mesh.indices[t * 3 + k]);
    runs.erase(
        std::remove_if(runs.begin(), runs.end(),
                       [](const std::vector<u32>& r) { return r.empty(); }),
        runs.end());
  }
  if (runs.size() <= 1) runs.assign(1, mesh.indices);

  kenshin::Timer timer;
  size_t built = 0;
  for (const auto& run : runs) {
    input.indices = run.data();
    input.index_count = run.size();
    built += clodBuild(
        config, input, &collector,
        [](void* context, clodGroup group, const clodCluster* clusters,
           size_t clusterCount) -> int {
          Collector* self = static_cast<Collector*>(context);
          ClusterGroup out;
          out.groupId = static_cast<int>(self->groups->size());
          out.depth = group.depth;
          for (int k = 0; k < 3; ++k)
            out.simplifiedCenter[k] = group.simplified.center[k];
          out.simplifiedCenter[3] = group.simplified.radius;
          out.simplifiedError = group.simplified.error;
          for (size_t c = 0; c < clusterCount; ++c) {
            Cluster cluster;
            cluster.parentGroupId = clusters[c].refined;
            for (int k = 0; k < 3; ++k)
              cluster.boundsCenter[k] = clusters[c].bounds.center[k];
            cluster.boundsCenter[3] = clusters[c].bounds.radius;
            cluster.error = clusters[c].bounds.error;
            cluster.indices.assign(
                clusters[c].indices,
                clusters[c].indices + clusters[c].index_count);
            cluster.vertexCount = static_cast<u32>(clusters[c].vertex_count);
            out.clusters.push_back(std::move(cluster));
          }
          self->stats->groupCount += 1;
          self->stats->clusterCount += static_cast<u32>(clusterCount);
          self->stats->maxDepth = std::max(self->stats->maxDepth,
                                           static_cast<u32>(group.depth + 1));
          self->stats->maxError =
              std::max(self->stats->maxError, group.simplified.error);
          self->groups->push_back(std::move(out));
          return static_cast<int>(self->groups->size()) - 1;
        });
  }
  if (runs.size() > 1)
    KENSHIN_INFO("[clod] {} per-material runs (no cluster mixes materials)",
                 runs.size());
  stats.seconds = timer.elapsedSeconds();

  if (built == 0) {
    log = "clodBuild produced no output";
    return false;
  }

  for (const auto& group : groups) {
    for (const auto& cluster : group.clusters)
      stats.totalTriangles += static_cast<u32>(cluster.indices.size() / 3);
    emit(group);
  }

  KENSHIN_INFO(
      "[clod] built: {} groups, {} clusters, depth {}, {} tris, "
      "max error {:.4g}, {:.0f} ms",
      stats.groupCount, stats.clusterCount, stats.maxDepth,
      stats.totalTriangles, stats.maxError, stats.seconds * 1000.0);
  return true;
}

}  // namespace geometry

KENSHIN_NS_END
