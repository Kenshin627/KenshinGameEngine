#pragma once

// ── LOD cut selection (S3.4) ─────────────────────────────────────
// CPU-side reference of the runtime LOD rule (same formula the GPU culling
// will use in Phase 4): a cluster renders iff its group's parent decision
// error is above threshold and its own decision error is at/below it.
// Produces a crack-free cut thanks to the monotonic error enforcement and
// the group-shared decision data from S3.2.

#include <vector>

#include "kenshin/geometry/ClusterDag.h"

KENSHIN_NS_BEGIN

namespace geometry {

/// Select the LOD cut for a pixel-error threshold (camera-relative:
/// each group's threshold scales with its distance from `cameraPos` -
/// the same shape as the Phase 4 GPU rule).
/// @param pixelThreshold  screen-space error threshold in pixels
/// @param viewSize        viewport height in pixels (pixels -> world scale)
/// @param fovYDeg         vertical FOV (pixels -> world scale)
/// @param cameraPos       world-space camera position (distance per group)
/// @param outClusters     selected cluster indices (render these)
void selectLodCut(const ClusterDag& dag, f32 pixelThreshold, u32 viewHeight,
                  f32 fovYDeg, const float cameraPos[3],
                  std::vector<u32>& outClusters);

/// LOD statistics of a cut.
struct CutStats {
  u32 clusterCount = 0;
  u32 triangleCount = 0;
  i32 minLevel = 0;
  i32 maxLevel = 0;
};
CutStats summarizeCut(const ClusterDag& dag, const std::vector<u32>& clusters);

}  // namespace geometry

KENSHIN_NS_END
