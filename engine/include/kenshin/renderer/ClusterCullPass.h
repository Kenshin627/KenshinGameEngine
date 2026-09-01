#pragma once

// ── ClusterCullPass (S4.3) ───────────────────────────────────────
// The LOD cut on the GPU, instance aware. Two phases (v1
// scan/compaction shape; the persistent-traversal rewrite in a later
// milestone absorbs them):
//   1. expandWorkItemsOnGpu: visible instances x their meshes' cluster
//      ranges -> (instanceIdx, clusterIdx) work items (one atomic
//      reservation per instance).
//   2. cullClustersOnGpu: one thread per work item; Karis p.70 rule
//      (ParentError > T && ClusterError <= T) with the threshold scaled
//      by the GROUP's instance-transformed bounds distance - identical
//      math to the CPU selectLodCut; the differential test pins them.

#include <glm/glm.hpp>

#include <span>
#include <vector>

#include "kenshin/Common.h"
#include "kenshin/renderer/InstanceCullPass.h"

KENSHIN_NS_BEGIN

namespace backend {
class IRenderBackend;
}
namespace manager {
class ShaderManager;
}

namespace renderer {

/// Mesh registry entry on the GPU: (clusterBegin, clusterEnd) packed as
/// u32 pairs (mirrors VgMeshRange's cluster span).
struct GpuMeshRangePair {
  u32 clusterBegin = 0;
  u32 clusterEnd = 0;
};

/// Phase 1: expand visible instances into work items. `visible` are
/// indices into `instances`; `meshRanges` per meshId. On success
/// `workItems` holds count pairs (order nondeterministic across
/// instances; contiguous per instance).
bool expandWorkItemsOnGpu(backend::IRenderBackend& backend,
                          manager::ShaderManager& shaders,
                          std::span<const u32> visible,
                          std::span<const CullGpuInstance> instances,
                          std::span<const GpuMeshRangePair> meshRanges,
                          std::vector<glm::uvec2>& workItems);

/// Phase 2: the cluster cut. `clusterData`/`groupData` are the packed
/// std430 blobs (packGpuCluster/packGpuGroup). `worldPerPixel` must be
/// computed on the CPU (2*tan(fov/2)/viewHeight) so both sides share
/// identical values. On success `visiblePairs` holds the survivors and
/// `stats[3]` = (visible, culledTooCoarse, culledSourceTooCoarse).
bool cullClustersOnGpu(backend::IRenderBackend& backend,
                       manager::ShaderManager& shaders,
                       const glm::vec3& camPos, f32 pixelThreshold,
                       f32 worldPerPixel, const std::vector<u8>& clusterData,
                       const std::vector<u8>& groupData,
                       std::span<const glm::uvec2> workItems,
                       std::span<const CullGpuInstance> instances,
                       std::vector<glm::uvec2>& visiblePairs, u32 stats[3]);

}  // namespace renderer

KENSHIN_NS_END
