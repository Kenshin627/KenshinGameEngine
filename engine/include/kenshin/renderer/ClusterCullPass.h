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
///
/// Per-cluster AABBs (M3.2b occlusion bound): derived at runtime from
/// the cluster table's vertex spans + the shared position pool (48B
/// entries, offsets rebased) - two vec4s per cluster (min.xyz, max.xyz).
/// Not serialized: the .kgeo format and GpuCluster stay untouched.
std::vector<glm::vec4> computeClusterAabbs(
    std::span<const glm::vec4> positions4, const std::vector<u8>& clusterData);

/// M3.2 occlusion inputs (optional; disabled by default): when enabled,
/// the survivors additionally test against the PREV frame's HZB (built
/// by buildHzbOneshot) projected with the prev frame's viewProj;
/// occluded clusters drop out; stats is 4 wide with
/// [3] = occlusion-culled (0 when disabled).
struct OcclusionInputs {
  bool enabled = false;
  glm::mat4 prevViewProj{1.0f};  // the frame the HZB was built in
  glm::vec3 prevCamPos{0.0f};
  const u32* hzbTexels = nullptr;  // hzbLayout(w,h) packing
  u32 hzbCount = 0;
  u32 screenW = 0, screenH = 0;  // the HZB's frame dimensions
  f32 tanHalfFov = 0.5f;
  const glm::vec4* clusterAabbs = nullptr;  // 2 vec4 per cluster (min,max)
};

bool cullClustersOnGpu(backend::IRenderBackend& backend,
                       manager::ShaderManager& shaders, const glm::vec3& camPos,
                       f32 pixelThreshold, f32 worldPerPixel,
                       const std::vector<u8>& clusterData,
                       const std::vector<u8>& groupData,
                       std::span<const glm::uvec2> workItems,
                       std::span<const CullGpuInstance> instances,
                       std::vector<glm::uvec2>& visiblePairs, u32 stats[4],
                       const OcclusionInputs& occlusion = {});

}  // namespace renderer

KENSHIN_NS_END
