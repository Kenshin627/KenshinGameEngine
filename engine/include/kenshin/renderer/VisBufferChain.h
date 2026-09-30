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
  /// M9.1 Stage C: inclusive pixel bounds [x0,y0,x1,y1] clamping the SW
  /// raster's bounding-box loops (the page-cache dirty region). The
  /// sentinel below = the full viewport, so every existing caller keeps
  /// bit-identical behavior; the shadow clipmaps pass live tile rects.
  u32 scissorRect[4] = {0, 0, 0xFFFFFFFFu, 0xFFFFFFFFu};
  glm::vec4 planes[6]{};
  bool occlusionEnabled = true;  // A/B + first-run safety toggle
  /// M9.2: instances flagged imposterable (castShadows==2) with
  /// projected bounds under this many pixels skip the cluster
  /// pipeline at the instance cull (imposter quads carry them).
  /// <= 0 disables (every non-M9 chain).
  f32 imposterThresholdPx = 0.0f;
  /// M9.1: shadow chains set this to skip receiver-only instances
  /// (castShadows == 0 in the chain's instance buffer) at emit time.
  u32 cullNonCasters = 0;
  u32 hwMode = 0;                // M4.2: 0 = all SW (the default since
                                 // the 2026-09-19 threshold sweep - the
                                 // mode-1 rasterizer beat every routing
                                 // config on all tested scene shapes),
                                 // 1 = route by threshold, 2 = force HW
  f32 hwThresholdPx = 1.0f;      // SW/HW routing edge threshold (mode 1
                                 // only; UE's r.NaniteMinPixelsPerEdgeHW
                                 // twin - our LOD keeps triangle edges far
                                 // below the HW crossover, so routing lost
                                 // its workload)
  /// M5.2/M5.3: 0 = linear expand+per-cluster cull (differential
  /// reference), 1 = BVH8 level-wise dispatch (the default), 2 = the
  /// M5.3 persistent-threads traversal (single dispatch + MPMC ticket
  /// queue -- at parity on light scenes, -7..-27% on chain-heavy ones
  /// in the v1 measurements; the M5.5 worker redesign decides the
  /// final default). Forced to 0 at init without hierarchy data.
  u32 cullMode = 1;
  /// SW-raster scheduling mode (the nanite.md-verified optimizations):
  /// 0 = the original per-pair walk (permanent differential baseline),
  /// 1 = exact-int64 row-span clamp + per-thread pixel-count
  /// aggregation. Both modes produce BIT-IDENTICAL VisBuffers and
  /// counter totals; 1 only changes which pixels get tested and how
  /// the stats are accumulated.
  u32 rasterMode = 0;
  /// Coverage histogram probe (params[11].y): classify each rasterized
  /// triangle by covered-pixel count into probeStats buckets. Debug
  /// tooling - adds two atomics per triangle; leave off for
  /// benchmarks.
  u32 rasterProbe = 0;
};

/// M6.3 paged geometry: the buffer handles the raster/decode passes
/// address through instead of the v2 pools. `slotPool` is a
/// createPageSlotPool buffer; `clusterPage`/`pageHeader` are static
/// tables; `pageTable` is HOST-VISIBLE (the streaming manager writes
/// page -> slot mappings, 0xFFFFFFFF = not resident). The element
/// totals feed the HW path's clamp bounds.
struct ChainPagedInputs {
  u32 slotPool = 0xFFFFFFFFu;
  u32 clusterPage = 0xFFFFFFFFu;
  u32 pageTable = 0xFFFFFFFFu;
  u32 pageHeader = 0xFFFFFFFFu;
  u32 pageCount = 0;
  // M7.1 v5: every page section is u32-addressed - ONE u32 capacity
  // for the whole pool (raster_hw's clamp bound).
  u32 slotU32Total = 0;
  // v5: per-cluster ordinal into its page's clusterMeta array.
  u32 clusterMeta = 0xFFFFFFFFu;
};

class VisBufferChain {
 public:
  static constexpr u32 kSlots = 2;  // = backend frames in flight

  /// Compile the five shaders, create the persistent pipelines and
  /// buffers. `maxPairs` bounds work items AND pairs (worst case
  /// visible instances x total clusters). The M5.2 hierarchy arguments
  /// may be empty - the chain then forces cullMode 0 (linear).
  /// M6.3: `paged` non-null switches the raster/decode addressing to
  /// the page slot pool (the positions/indices spans are then unused;
  /// `clusterAabbs` carries the precomputed occlusion bounds the v2
  /// path derives from the pools).
  bool init(backend::IRenderBackend& backend, manager::ShaderManager& shaders,
            std::span<const glm::vec4> positions4, std::span<const u32> indices,
            const std::vector<u8>& clusterData,
            const std::vector<u8>& groupData,
            const std::vector<u8>& hierarchyData,
            std::span<const glm::uvec2> groupRanges,
            std::span<const u32> meshHierRoots,
            std::span<const CullGpuInstance> instances,
            std::span<const GpuMeshRangePair> meshRanges, u32 maxPairs,
            std::span<const glm::vec4> clusterAabbs,
            const ChainPagedInputs* paged, u32 width, u32 height);

  /// (Re)create the per-slot VisBuffers on resolution change.
  void resize(backend::IRenderBackend& backend, u32 width, u32 height);

  /// M9.1: per-chain GPU-timestamp toggle. The per-slot timing pool
  /// is SHARED - a second same-frame chain would overwrite the first
  /// chain's segments (and the per-slot armed guard would skip its
  /// reset). The shadow chain runs with timing off; the main chain
  /// keeps the [chainms] segments.
  void setTimingEnabled(bool enabled) { m_timingEnabled = enabled; }

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
  /// The CURRENT frame's pair lists - the wide-instance decode's
  /// indirection tables (the VisBuffer payload indexes these).
  u32 pairsCurrent(backend::IRenderBackend& backend) const;
  u32 hwPairsCurrent(backend::IRenderBackend& backend) const;
  u32 postPairsCurrent(backend::IRenderBackend& backend) const;
  /// The pair-list capacity all three lists were sized to (the
  /// decode's address clamp bound).
  u32 maxPairs() const { return m_maxPairs; }

  /// Blocking stats readback (test/debug), per chain stage:
  /// [0] visible instances, [1] work items, [2] visible pairs (clusters
  /// cut), [3] too-coarse, [4] source-too-coarse, [5] px writes,
  /// [6] backface, [7] near-clipped, [8] occlusion-culled (pending),
  /// [9] post-pass revived, [10] post-raster px writes,
  /// [11] cut tris (main + revived), [12] HW-routed clusters,
  /// [13] hierarchy visited, [14] hierarchy pruned,
  /// [15] pending-resident skips (M6.3: SW workgroups + HW clusters
  /// whose page was not resident that frame).
  bool readStats(backend::IRenderBackend& backend, u32 out[16]) const;
  /// NON-BLOCKING stats snapshot for panels: memcpy straight off the
  /// per-slot HOST-VISIBLE counter maps (no submit, no idle - the
  /// values trail the GPU by however much the driver's writes lag,
  /// which a debug panel cannot tell apart). The blocking readStats
  /// stays for tests; calling IT periodically from the frame loop
  /// degrades presentation progressively (measured: 6 queued idles
  /// every 8 frames -> 5x frame-time growth over ~2k frames).
  bool readStatsMapped(backend::IRenderBackend& backend, u32 out[16]) const;
  /// NON-BLOCKING copy of the touched-page bitmap (one u32 per 32
  /// pages; M6.3 page_touch output - the streaming manager's residency
  /// feedback). The readStatsMapped contract: call between endFrame
  /// and the NEXT execute (the current slot's bitmap is then fully
  /// written and not yet cleared).
  bool readTouchedPages(backend::IRenderBackend& backend,
                        std::vector<u32>& out) const;
  /// Debug probe: the three indirect dispatch commands as written by
  /// the conversions (x = group count).
  bool readIndirectArgs(backend::IRenderBackend& backend, u32 expandX[3],
                        u32 cullX[1], u32 rasterX[1]) const;

  /// NON-BLOCKING coverage-histogram snapshot (the readStatsMapped
  /// contract: current slot, before this frame's execute). out[8]:
  /// [0..5] = triangles covering 0 / 1 / 2-4 / 5-16 / 17-64 / >64
  /// pixels, [6] = covered pixels summed, [7] = triangles that reached
  /// rasterization (main + post dispatches share the buffer). Only
  /// meaningful with ChainInputs::rasterProbe = 1. `lastCompleted`
  /// reads the most recently ENDED frame's slot instead (tests, after
  /// a blocking readback has proven the frame drained).
  bool readRasterProbe(backend::IRenderBackend& backend, u32 out[8],
                       bool lastCompleted = false) const;

  /// Test/debug: read back the most recently ENDED frame's visiblePairs
  /// (count from readStats[2]) - the M5.2 hierarchy-vs-selectLodCut set
  /// equality assertions read this.
  bool readVisiblePairs(backend::IRenderBackend& backend,
                        std::vector<glm::uvec2>& out) const;
  /// Test/debug: the same for the HW-routed list (count: stats[12])
  /// and the post-revived list (count: stats[9]) - the wide-instance
  /// payload resolves through all three.
  bool readHwPairs(backend::IRenderBackend& backend,
                   std::vector<glm::uvec2>& out) const;
  bool readPostPairs(backend::IRenderBackend& backend,
                     std::vector<glm::uvec2>& out) const;

  void onShutdown(backend::IRenderBackend& backend);

 private:
  bool m_timingEnabled = true;  // M9.1: see setTimingEnabled
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
  // Coverage histogram (rasterMode tooling): 8 u32, written by the
  // raster passes when ChainInputs::rasterProbe is set.
  u32 m_rasterProbe[kSlots]{};
  // The post raster's payload listSel (=2) needs its own params copy -
  // one host buffer cannot hold two values for the same frame's two
  // dispatches.
  u32 m_postRasterParams[kSlots]{};
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
  /// M6.4 saturation: the static pair/list capacity word the saturating
  /// writers read (expand binding 5, hier level binding 17).
  u32 m_pairCapBuffer = ~0u;
  /// Persistent mode requires queue capacity >= instances x (nodes+1)
  /// (the no-wraparound proof); beyond the memory cap it falls back to
  /// level-wise with a one-time warning.
  bool m_persistUsable = true;

  // ---- M5.3: the persistent-threads path --------------------------------
  // The ticket ring: capacity = pow2(instances x (nodes+1)) - every node
  // enters at most once per instance per frame, so indices never wrap
  // within a frame. EMPTY-initialized once at init (consumers re-empty
  // their slots, keeping the invariant without per-frame clears).
  static constexpr u32 kPersistGroups = 8;  // measured sweep 8>16>32>...>512
      // (per-group async-engine overhead dominates
      // at current scene scales; M5.5 re-tunes)
  u32 m_persistQueue[kSlots]{};
  u32 m_persistState
      [kSlots]{};  // {head,tail,outstanding,done,visited,pruned,pad,pad}
  u32 m_pipeSeedQ = ~0u;
  u32 m_pipePersist = ~0u;
  u32 m_persistCapacity = 0;  // pow2 entry count (mask = capacity-1)
  u32 m_argsPostCull[kSlots]{};
  u32 m_cfgPostCull = ~0u;
  u32 m_groupCount = 0;
  u32 m_instanceCount = 0, m_maxPairs = 0;
  u32 m_clusterCount = 0, m_indexCount = 0, m_positionCount = 0;
  u32 m_width = 0, m_height = 0;

  // ---- M6.3: paged geometry (slot-pool addressing) -------------------
  // When m_paged, bindings 1/2 of the raster passes (and the decode
  // pool bindings) are the PAGE SLOT POOL: cluster offsets are
  // page-local, each read adds the slot-strided section base, and a
  // pageTable miss skips the cluster (counted in rasterCnt[4]). The
  // v2 path binds tiny dummies at the paged table slots so ONE shader
  // serves both modes (params-gated reads only).
  bool m_paged = false;
  ChainPagedInputs m_pagedInputs;
  u32 m_dummyU32[3] = {~0u, ~0u, ~0u};  // v2 stand-ins: clusterPage,
                                        // pageTable, raster_hw pending
  u32 m_dummyPageHeader = ~0u;          // v2 stand-in: pageHeader
  // page_touch feedback (paged only; compiled in every run to keep the
  // shader regression-gated): two pipeline instances (the SW pair list
  // and the HW pair list each need their one set update per frame),
  // the per-slot touched bitmap, and the HW-list dispatch args convert.
  u32 m_pipeTouchPairs = ~0u, m_pipeTouchHw = ~0u;
  u32 m_pipeConvertTouch = ~0u;  // cull counters[7] -> touch dispatch args
  u32 m_touched[kSlots]{};       // host-visible bitmap, pageCount/32 words
  u32 m_argsTouch[kSlots]{};
  u32 m_cfgTouch = ~0u;  // {0, 0, 7, 0}: groupsX = cull counters[7]
  u32 m_pageBitmapWords = 0;
};

}  // namespace renderer

KENSHIN_NS_END
