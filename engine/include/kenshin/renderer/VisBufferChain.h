#pragma once

// ── VisBufferChain — the in-frame GPU chain (链入帧化 B2) ─────────
// Instance cull -> work-item expand -> cluster cull -> software
// raster, all recorded into the CURRENT frame's command buffer with
// persistent pipelines/buffers and Nanite-style indirect dispatches
// (count_to_indirect converts each stage's GPU-written count into the
// next stage's dispatch command - no CPU readback, no blocking).
//
// Every intermediate lives PER FRAME SLOT (two frames in flight write
// their own copies); small per-frame inputs are CPU-written through
// persistently-mapped host-visible buffers (legal + coherent for the
// GPU-side atomics). The result: the decode pass consumes the SAME
// frame's VisBuffer - zero camera lag, no inter-frame stall.
//
// The one-shot functions (cullInstancesOnGpu etc.) stay as the
// differential reference; this class must produce identical bits.

#include <glm/glm.hpp>
#include <span>
#include <vector>

#include "kenshin/Common.h"
#include "kenshin/renderer/ClusterCullPass.h"
#include "kenshin/renderer/InstanceCullPass.h"

KENSHIN_NS_BEGIN

namespace backend {
class IRenderBackend;
}
namespace manager {
class ShaderManager;
}

namespace renderer {

struct ChainInputs {
  glm::mat4 view = glm::mat4(1.0f);
  glm::mat4 proj = glm::mat4(1.0f);  // camera's own (Y-flipped)
  f32 nearZ = 0.01f;
  glm::vec3 camPos{0.0f};
  f32 pixelThreshold = 1.0f;
  f32 worldPerPixel = 0.001f;
  u32 width = 0;
  u32 height = 0;
  glm::vec4 planes[6]{};
  bool occlusionEnabled = true;  // A/B + first-run safety toggle
  u32 hwMode = 1;                // M4.2: 0 = all SW, 1 = route by
                                 // threshold, 2 = force all HW
  f32 hwThresholdPx = 18.0f;     // UE's GNaniteMinPixelsPerEdgeHW
  /// M5.2: 0 = linear expand+per-cluster cull (differential reference),
  /// 1 = BVH8 hierarchy traversal (level-wise dispatch). Forced to 0 at
  /// init when no hierarchy data was provided.
  u32 cullMode = 1;
};

class VisBufferChain {
 public:
  static constexpr u32 kSlots = 2;  // = backend frames in flight

  /// Compile the five shaders, create the persistent pipelines and
  /// buffers. `maxPairs` bounds work items AND pairs (worst case
  /// visible instances x total clusters). The M5.2 hierarchy arguments
  /// may be empty - the chain then forces cullMode 0 (linear).
  bool init(backend::IRenderBackend& backend, manager::ShaderManager& shaders,
            std::span<const glm::vec4> positions4, std::span<const u32> indices,
            const std::vector<u8>& clusterData,
            const std::vector<u8>& groupData,
            const std::vector<u8>& hierarchyData,
            std::span<const glm::uvec2> groupRanges,
            std::span<const u32> meshHierRoots,
            std::span<const CullGpuInstance> instances,
            std::span<const GpuMeshRangePair> meshRanges, u32 maxPairs,
            u32 width, u32 height);

  /// (Re)create the per-slot VisBuffers on resolution change.
  void resize(backend::IRenderBackend& backend, u32 width, u32 height);

  /// Record the whole chain into the CURRENT frame (between beginFrame
  /// and endFrame). CPU-writes this slot's params first, then five
  /// dispatches + count conversions, each behind barriers.
  void execute(backend::IRenderBackend& backend, const ChainInputs& in);

  /// The CURRENT recording frame's VisBuffer - the in-frame decode
  /// consumer's handle (the same frame's chain just produced it).
  u32 visBufferCurrent(backend::IRenderBackend& backend) const;
  /// The most recently ENDED frame's VisBuffer (post-frame readbacks;
  /// before the first endFrame this is invalid).
  u32 visBuffer(backend::IRenderBackend& backend) const;

  /// Blocking stats readback (test/debug), per chain stage:
  /// [0] visible instances, [1] work items, [2] visible pairs (clusters
  /// cut), [3] too-coarse, [4] source-too-coarse, [5] px writes,
  /// [6] backface, [7] near-clipped, [8] occlusion-culled (pending),
  /// [9] post-pass revived, [10] post-raster px writes,
  /// [11] cut tris (main + revived).
  bool readStats(backend::IRenderBackend& backend, u32 out[13]) const;
  /// NON-BLOCKING stats snapshot for panels: memcpy straight off the
  /// per-slot HOST-VISIBLE counter maps (no submit, no idle - the
  /// values trail the GPU by however much the driver's writes lag,
  /// which a debug panel cannot tell apart). The blocking readStats
  /// stays for tests; calling IT periodically from the frame loop
  /// degrades presentation progressively (measured: 6 queued idles
  /// every 8 frames -> 5x frame-time growth over ~2k frames).
  bool readStatsMapped(backend::IRenderBackend& backend, u32 out[13]) const;
  /// Debug probe: the three indirect dispatch commands as written by
  /// the conversions (x = group count).
  bool readIndirectArgs(backend::IRenderBackend& backend, u32 expandX[3],
                        u32 cullX[1], u32 rasterX[1]) const;

  /// Test/debug: read back the most recently ENDED frame's visiblePairs
  /// (count from readStats[2]) - the M5.2 hierarchy-vs-selectLodCut set
  /// equality assertions read this.
  bool readVisiblePairs(backend::IRenderBackend& backend,
                        std::vector<glm::uvec2>& out) const;

  void onShutdown(backend::IRenderBackend& backend);

 private:
  u32 m_pipeCull = ~0u, m_pipeExpand = ~0u, m_pipeCluster = ~0u,
      m_pipeRaster = ~0u;
  // Three SEPARATE conversion pipelines (same shader): a descriptor set
  // updated mid-command-buffer invalidates already-recorded uses, so each
  // conversion site needs its own set - hence its own pipeline. The 4th
  // (M3.3) feeds the post raster from the cull counters' revived slot.
  u32 m_pipeConvert[5] = {~0u, ~0u, ~0u, ~0u, ~0u};
  u32 m_pipePostCull = ~0u;  // M3.3: pending retest vs current HZB
  // Same SPIR-V, SEPARATE pipeline objects: a pipeline's per-slot
  // descriptor set may only be updated once per frame slot, and the
  // post raster / final HZB build run in the SAME frame as their
  // main-pass twins with different bindings.
  u32 m_pipeRasterPost = ~0u;   // M3.3: revived pairs' re-raster
  u32 m_pipeHzbFinal = ~0u;     // M3.3: HZB #2 (post-pass-inclusive)
  u32 m_pipeHwRaster = ~0u;     // M4.2: null-attachment graphics pass
  u32 m_pipeCountToDraw = ~0u;  // M4.2: counters[7] -> draw command
  // M4 follow-up: pending-gated post section (the flag conversion)
  // + the level-by-level HZB reduction (the atomic climb's
  // same-address apex contention cost milliseconds at real sizes).
  // One REDUCE pipeline per LEVEL-slot pair usage: #1 (flag-gated,
  // indirect) and #2 (always-on, direct) each need their OWN set per
  // LEVEL (the per-level params buffer differs within one frame).
  static constexpr u32 kMaxHzbLevels = 16;
  u32 m_pipeCountToFlag = ~0u;  // counters[4] -> {G|0,1,1}
  u32 m_pipeReduce1[kMaxHzbLevels] = {};
  u32 m_pipeReduce2[kMaxHzbLevels] = {};
  u32 m_instanceBuffer = ~0u, m_meshRangeBuffer = ~0u;
  u32 m_clusterBuffer = ~0u, m_groupBuffer = ~0u;
  u32 m_positionBuffer = ~0u, m_indexBuffer = ~0u;
  // Per-slot intermediates (host-visible where CPU-written).
  u32 m_planes[kSlots]{}, m_cntA[kSlots]{}, m_visibleList[kSlots]{};
  u32 m_workItems[kSlots]{}, m_expCnt[kSlots]{};
  u32 m_cullParams[kSlots]{}, m_cullCnt[kSlots]{}, m_pairs[kSlots]{};
  u32 m_rasterParams[kSlots]{}, m_rasterCnt[kSlots]{}, m_vis[kSlots]{};
  // Static conversion configs + per-slot indirect args.
  u32 m_cfgExpand = ~0u, m_cfgCull = ~0u, m_cfgRaster = ~0u;
  u32 m_cfgPost = ~0u;    // post convert: {1 group/item, fwd 0, src 5}
  u32 m_cfgHwDraw = ~0u;  // M4.2: {384 verts, 0, src = counters[7]}
  u32 m_cfgFlag = ~0u;    // {fullscreenGroups, 0, src = counters[4]}
  u32 m_argsExpand[kSlots]{}, m_argsCull[kSlots]{}, m_argsRaster[kSlots]{};
  u32 m_argsPostRaster[kSlots]{};
  u32 m_argsHzb1[kSlots]{};  // the flag: {fullscreenGroups|0,1,1}
  // Per-level reduce params (baked per resolution in resize): one
  // vec4 per level = (childW, childH, childOffset, dstOffset); the
  // L0 pass uses (0, 0, 0, 0).
  u32 m_hzbLevelParams[kMaxHzbLevels]{};
  u32 m_hzbLevelTexels[kMaxHzbLevels]{};
  u32 m_hzbLevelCount = 0;
  u32 m_hzbL0Groups = 0;  // the flag size (largest dispatch)
  // M3.3 post pass (per slot): the Main cull's occluded items, the
  // revived subset, its raster counters, and the post cull's params
  // (current-frame viewProj - the preliminary HZB's space).
  u32 m_pendingPairs[kSlots]{}, m_postPairs[kSlots]{};
  u32 m_postRasterCnt[kSlots]{}, m_postParams[kSlots]{};
  // M4.2 hardware raster: the routed pair list, its draw command
  // (VkDrawIndirectCommand via count_to_draw) and the pass params.
  u32 m_hwPairs[kSlots]{}, m_hwDrawArgs[kSlots]{}, m_hwParams[kSlots]{};
  // M3.2b/M3.3 occlusion: per-slot HZB pyramids. SEMANTICS (M3.3):
  // m_hzb[s] holds the pyramid built at the END of slot s's most
  // recent frame (build #1 after the Main raster feeds the same
  // frame's post cull; build #2 after the post raster - from the
  // COMPLETE VisBuffer - is what the NEXT frame's Main cull tests
  // against, indexed by the OTHER slot). m_hzbValid gates the first
  // run per slot (and a resize reset).
  u32 m_pipeHzb = ~0u;
  u32 m_hzb[kSlots]{};
  u32 m_hzbParams[kSlots]{};  // (w,h) upload for build_hzb
  bool m_hzbValid[kSlots]{};
  glm::mat4 m_prevVP[kSlots]{};
  glm::vec3 m_prevCam[kSlots]{};
  u32 m_aabbBuffer = ~0u;  // runtime-derived per-cluster AABBs (static)

  // ---- M5.2: the BVH8 hierarchy path ----------------------------------
  static constexpr u32 kMaxHierLevels = 16;
  // Static: the node pool (flat 64B DagHierarchyNodes), per-group
  // (clusterBegin, count) spans, per-mesh root node indices.
  u32 m_hierBuffer = ~0u, m_groupRangeBuffer = ~0u, m_meshHierRootBuffer = ~0u;
  // Per-slot: ping-pong active-node lists (uint2 = instance, node), the
  // two 4B ping-pong count buffers (in/out cursors per level parity),
  // and per-level indirect args.
  u32 m_activeNodes[kSlots][2]{};
  u32 m_hierCnt[kSlots][2]{};
  u32 m_hierParams[kSlots]{};  // cullParams copy + [1].y/[7].zw counts
  u32 m_argsHier[kMaxHierLevels][kSlots]{};
  // Pipelines: the seed, the two level PARITIES (in/out list bindings
  // differ - one descriptor-set update per site per frame), and one
  // convert site per level (the args dst differs per level).
  u32 m_pipeHierSeed = ~0u;
  u32 m_pipeHierLevel[kMaxHierLevels] = {};
  u32 m_pipeConvertHier[kMaxHierLevels] = {};
  u32 m_cfgHier[kMaxHierLevels]{};
  u32 m_hierDepth = 0;  // levels actually dispatched (1 = root only)
  bool m_hierReady = false;
  u32 m_hierNodeCount = 0;  // for the shader-side index clamps
  u32 m_argsPostCull[kSlots]{};
  u32 m_cfgPostCull = ~0u;
  u32 m_groupCount = 0;
  u32 m_instanceCount = 0, m_maxPairs = 0;
  u32 m_clusterCount = 0, m_indexCount = 0, m_positionCount = 0;
  u32 m_width = 0, m_height = 0;
};

}  // namespace renderer

KENSHIN_NS_END
