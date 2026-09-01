#include "kenshin/geometry/LodCut.h"

#include <algorithm>
#include <cmath>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/constants.hpp>

KENSHIN_NS_BEGIN

namespace geometry {

namespace {
/// Convert a screen-space pixel error into an object-space world error
/// at a given distance (same math the GPU cull will use):
///   worldError = pixels * (2 * tan(fov/2) / viewHeight) * distance
f32 worldErrorAt(f32 pixelError, u32 viewHeight, f32 fovYDeg, f32 distance) {
  const f32 worldPerPixel = 2.0f * std::tan(glm::radians(fovYDeg) * 0.5f) /
                            static_cast<f32>(viewHeight);
  return pixelError * worldPerPixel * distance;
}
}  // namespace

void selectLodCut(const ClusterDag& dag, f32 pixelThreshold, u32 viewHeight,
                  f32 fovYDeg, const float cameraPos[3],
                  std::vector<u32>& outClusters) {
  outClusters.clear();

  // Camera-relative world threshold per group (the Phase 4 GPU rule
  // shape): worldErrorAt(distance(groupBounds, camera)). Groups near
  // the camera demand finer LOD, far groups coarsen earlier.
  const glm::vec3 cam(cameraPos[0], cameraPos[1], cameraPos[2]);
  std::vector<f32> thresholdOf(dag.groups.size());
  for (usize g = 0; g < dag.groups.size(); ++g) {
    const DagGroup& group = dag.groups[g];
    const glm::vec3 center(group.boundsSphere);
    const f32 distance =
        std::max(glm::distance(center, cam) - group.boundsSphere.w, 0.1f);
    thresholdOf[g] =
        worldErrorAt(pixelThreshold, viewHeight, fovYDeg, distance);
  }

  // Render rule (Nanite SIGGRAPH 2021 "cutting the DAG", Karis slides
  // 62-66; clusterlod semantics): a cluster renders when it is the FIRST
  // (finest) level on its chains that fires:
  //   fireable(G) = simplifyError(G) > T_G   [node property: own stored
  //                                     error, own distance-scaled threshold]
  //   select(G) = fireable(G) AND every direct parent !fireable
  // ("parent says no, child says yes" - fully local, no path context.)
  // EVERY comparison uses its OWN node's threshold: fireable stays a pure
  // node property, so multi-parent DAG nodes cannot be judged
  // inconsistently by different children. Build-time monotonicity (parent
  // error AND bounds >= child's) makes fireable monotone along chains ->
  // the first-fireable level exists on every chain (terminals hold
  // FLT_MAX) -> full coverage, exactly-once selection, by construction.
  // [Bug this fixed: judging parents against the CHILD's threshold let a
  //  parent fail its child's check while passing its own -> neither
  //  selected -> a whole strand of the knot vanished at threshold 1.0.]
  std::vector<bool> fireable(dag.groups.size());
  for (usize g = 0; g < dag.groups.size(); ++g)
    fireable[g] = dag.groups[g].simplifyError > thresholdOf[g];

  for (u32 c = 0; c < dag.clusters.size(); ++c) {
    const DagCluster& cluster = dag.clusters[c];
    if (!fireable[cluster.group]) continue;
    // Condition 2 is PER-CLUSTER and checks ONLY this cluster's source
    // group (clusterlod.h: "cluster.refined is -1 or simplified for
    // groups[cluster.refined] is at or under threshold"). Judging by ALL
    // parent groups blocks a cluster for regions its own source covers -
    // sibling parents straddling the threshold left whole strands
    // unrendered.
    if (cluster.refinedGroup >= 0 && fireable[cluster.refinedGroup])
      continue;
    outClusters.push_back(c);
  }

  // Fallback: pathological thresholds never select - render roots.
  if (outClusters.empty()) outClusters = dag.roots;
}
CutStats summarizeCut(const ClusterDag& dag, const std::vector<u32>& clusters) {
  CutStats stats;
  stats.clusterCount = static_cast<u32>(clusters.size());
  stats.minLevel = 0x7FFFFFFF;
  for (u32 c : clusters) {
    stats.triangleCount += dag.triangleCount(c);
    const i32 level = dag.groups[dag.clusters[c].group].lodLevel;
    stats.minLevel = std::min(stats.minLevel, level);
    stats.maxLevel = std::max(stats.maxLevel, level);
  }
  if (clusters.empty()) stats.minLevel = 0;
  return stats;
}

}  // namespace geometry

KENSHIN_NS_END
