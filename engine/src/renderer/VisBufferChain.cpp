#include "kenshin/renderer/VisBufferChain.h"

#include <cmath>
#include <cstring>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/geometry/ClusterDag.h"
#include "kenshin/manager/ShaderManager.h"
#include "kenshin/renderer/HzbPass.h"

KENSHIN_NS_BEGIN

namespace renderer {

namespace {
constexpr u32 kInvalid = backend::IRenderBackend::kInvalidBufferHandle;

/// Host-visible persistent buffer (CPU-written params / counters).
/// TransferSrc: the stats readbacks copy these back (readStats).
u32 makeHostBuffer(backend::IRenderBackend& backend, usize bytes,
                   const void* initial, const char* name) {
  backend::IRenderBackend::BufferDesc desc;
  desc.size = bytes;
  desc.usage = backend::IRenderBackend::BufferStorage |
               backend::IRenderBackend::BufferHostVisible |
               backend::IRenderBackend::BufferTransferSrc;
  desc.initialData = initial;
  desc.debugName = name;
  return backend.createBuffer(desc);
}
/// Device-local persistent buffer (GPU-only intermediates / pools).
u32 makeDeviceBuffer(backend::IRenderBackend& backend, usize bytes,
                     const void* initial, const char* name) {
  backend::IRenderBackend::BufferDesc desc;
  desc.size = bytes;
  desc.usage = backend::IRenderBackend::BufferStorage;
  desc.initialData = initial;
  desc.debugName = name;
  return backend.createBuffer(desc);
}
bool compilePipe(manager::ShaderManager& shaders, const char* file,
                 std::vector<u32>& spirv) {
  std::string log;
  if (!shaders.compileFile(file, spirv, log, "main")) {
    KENSHIN_ERROR("[vgchain] {} compile failed: {}", file, log);
    return false;
  }
  return true;
}
}  // namespace

bool VisBufferChain::init(
    backend::IRenderBackend& backend, manager::ShaderManager& shaders,
    std::span<const glm::vec4> positions4, std::span<const u32> indices,
    const std::vector<u8>& clusterData, const std::vector<u8>& groupData,
    const std::vector<u8>& hierarchyData,
    std::span<const glm::uvec2> groupRanges, std::span<const u32> meshHierRoots,
    std::span<const CullGpuInstance> instances,
    std::span<const GpuMeshRangePair> meshRanges, u32 maxPairs,
    std::span<const glm::vec4> clusterAabbs, const ChainPagedInputs* paged,
    u32 width, u32 height) {
  std::vector<u32> spirv;
  backend::IRenderBackend::ComputePipelineInfo info;
  const auto makePipe = [&](const char* file, u32& out) {
    if (!compilePipe(shaders, file, spirv)) return false;
    info.csSpirv = spirv.data();
    info.csBytes = spirv.size() * sizeof(u32);
    out = backend.createComputePipeline(info);
    if (out == backend::IRenderBackend::kInvalidPipelineHandle) {
      KENSHIN_ERROR("[vgchain] {} pipeline failed", file);
      return false;
    }
    return true;
  };
  if (!makePipe("cull_instances.slang", m_pipeCull)) return false;
  if (!makePipe("expand_workitems.slang", m_pipeExpand)) return false;
  if (!makePipe("cull_clusters.slang", m_pipeCluster)) return false;
  if (!makePipe("post_cull.slang", m_pipePostCull)) return false;
  if (!makePipe("build_hzb.slang", m_pipeHzb)) return false;
  if (!makePipe("build_hzb.slang", m_pipeHzbFinal)) return false;
  if (!makePipe("rasterize_sw.slang", m_pipeRaster)) return false;
  if (!makePipe("rasterize_sw.slang", m_pipeRasterPost)) return false;
  if (!makePipe("count_to_draw.slang", m_pipeCountToDraw)) return false;
  if (!makePipe("count_to_flag.slang", m_pipeCountToFlag)) return false;
  for (u32 k = 0; k < kMaxHzbLevels; ++k) {
    if (!makePipe("hzb_reduce.slang", m_pipeReduce1[k])) return false;
    if (!makePipe("hzb_reduce.slang", m_pipeReduce2[k])) return false;
  }
  {
    // raster_hw: one file, two entry points; the null-attachment
    // graphics pipeline (the FS reflection covers the VS's bindings -
    // both stages share the file-scope globals).
    std::vector<u32> vs, fs;
    std::string log;
    if (!shaders.compileFile("raster_hw.slang", vs, log, "vsMain") ||
        !shaders.compileFile("raster_hw.slang", fs, log, "psMain")) {
      KENSHIN_ERROR("[vgchain] raster_hw compile failed: {}", log);
      return false;
    }
    backend::IRenderBackend::FullscreenPipelineInfo fi;
    fi.vsSpirv = vs.data();
    fi.vsBytes = vs.size() * sizeof(u32);
    fi.fsSpirv = fs.data();
    fi.fsBytes = fs.size() * sizeof(u32);
    m_pipeHwRaster = backend.createNullPipeline(fi);
    if (m_pipeHwRaster == backend::IRenderBackend::kInvalidPipelineHandle) {
      KENSHIN_ERROR("[vgchain] raster_hw pipeline failed");
      return false;
    }
  }
  for (u32 k = 0; k < 5; ++k)
    if (!makePipe("count_to_indirect.slang", m_pipeConvert[k])) return false;
  // M5.2: hierarchy shaders (compiled unconditionally so every test run
  // also regression-gates them) + the per-level convert sites.
  if (!makePipe("cull_hier_seed.slang", m_pipeHierSeed)) return false;
  // ONE PIPELINE PER LEVEL: the backend updates a pipeline's descriptor
  // set on EVERY dispatch, so a parity pair gets its set updated twice
  // in one frame-recording once the tree is deeper than two levels,
  // invalidating the earlier recording (the M3.3 discipline).
  for (u32 k = 0; k < kMaxHierLevels; ++k)
    if (!makePipe("cull_hier_level.slang", m_pipeHierLevel[k])) return false;
  for (u32 k = 0; k < kMaxHierLevels; ++k)
    if (!makePipe("count_to_indirect.slang", m_pipeConvertHier[k]))
      return false;
  // M5.3: the persistent-threads pair (compiled unconditionally, same
  // regression-gating rationale).
  if (!makePipe("cull_hier_seedq.slang", m_pipeSeedQ)) return false;
  if (!makePipe("cull_hier_persist.slang", m_pipePersist)) return false;

  m_instanceCount = static_cast<u32>(instances.size());
  m_maxPairs = maxPairs;
  m_clusterCount = static_cast<u32>(clusterData.size() / 48);
  m_paged = paged != nullptr;
  if (m_paged) m_pagedInputs = *paged;
  m_width = width;
  m_height = height;

  // Static pools (uploaded once).
  m_instanceBuffer =
      makeDeviceBuffer(backend, instances.size() * sizeof(CullGpuInstance),
                       instances.data(), "chain-instances");
  m_meshRangeBuffer =
      makeDeviceBuffer(backend, meshRanges.size() * sizeof(GpuMeshRangePair),
                       meshRanges.data(), "chain-meshranges");
  m_clusterBuffer = makeDeviceBuffer(backend, clusterData.size(),
                                     clusterData.data(), "chain-clusters");
  m_groupBuffer = makeDeviceBuffer(backend, groupData.size(), groupData.data(),
                                   "chain-groups");
  m_groupCount = static_cast<u32>(groupData.size() / 32);
  const u32 cfgPostCull[4] = {64, 0, 4, m_maxPairs};  // pending-sized envelope
  m_cfgPostCull =
      makeHostBuffer(backend, sizeof(cfgPostCull), cfgPostCull, "cfgPC");
  // The saturating writers' capacity word (static - lists never resize).
  m_pairCapBuffer = makeDeviceBuffer(backend, 4, &m_maxPairs, "chain-pairCap");
  // M5.2: hierarchy static pools. The 64B DagHierarchyNodes are repacked
  // into the GPU layout GpuHierNode (float4 min/max + error + count +
  // children[8], std430 stride 80B) - flat f32 triplets would straddle
  // std430 vec3 alignment seams.
  m_hierReady = !hierarchyData.empty();
  if (m_hierReady) {
    const usize nodeCount =
        hierarchyData.size() / sizeof(geometry::DagHierarchyNode);
    std::vector<u8> gpuNodes(nodeCount * 80);
    const auto* nodes = reinterpret_cast<const geometry::DagHierarchyNode*>(
        hierarchyData.data());
    for (usize n = 0; n < nodeCount; ++n) {
      u8* p = gpuNodes.data() + n * 80;
      const f32 mn[4] = {nodes[n].aabbMin[0], nodes[n].aabbMin[1],
                         nodes[n].aabbMin[2], 0.0f};
      const f32 mx[4] = {nodes[n].aabbMax[0], nodes[n].aabbMax[1],
                         nodes[n].aabbMax[2], 0.0f};
      std::memcpy(p + 0, mn, 16);
      std::memcpy(p + 16, mx, 16);
      std::memcpy(p + 32, &nodes[n].maxSimplifyError, 4);
      std::memcpy(p + 36, &nodes[n].childCount, 4);
      std::memcpy(p + 40, nodes[n].children, 32);
    }
    m_hierBuffer = makeDeviceBuffer(backend, gpuNodes.size(), gpuNodes.data(),
                                    "chain-hierNodes");
    m_groupRangeBuffer =
        makeDeviceBuffer(backend, groupRanges.size() * sizeof(glm::uvec2),
                         groupRanges.data(), "chain-groupRanges");
    m_meshHierRootBuffer =
        makeDeviceBuffer(backend, meshHierRoots.size() * sizeof(u32),
                         meshHierRoots.data(), "chain-meshHierRoots");

    // Depth from each mesh root (level count to dispatch, root = 1).
    u32 depth = 1;
    for (const u32 root : meshHierRoots) {
      if (root >= nodeCount) {
        m_hierReady = false;
        break;
      }
      std::vector<std::pair<u32, u32>> dfs{{root, 1}};
      while (!dfs.empty()) {
        const auto [n, d] = dfs.back();
        dfs.pop_back();
        depth = std::max(depth, d);
        if (nodes[n].childCount == 0 || d >= kMaxHierLevels) continue;
        for (u32 i = 0; i < nodes[n].childCount && i < 8; ++i)
          if (nodes[n].children[i] < nodeCount)
            dfs.emplace_back(nodes[n].children[i], d + 1);
      }
    }
    m_hierNodeCount = static_cast<u32>(nodeCount);
    m_hierDepth = m_hierReady ? std::min(depth, kMaxHierLevels) : 0;

    // M5.3: the ticket ring. Capacity = pow2(instances x (nodes+1)) -
    // every node enters the queue at most once per instance per frame,
    // so monotonic ticket indices never wrap within a frame.
    u32 entries = m_instanceCount * (m_hierNodeCount + 1u);
    // M6.4: the ring's no-wraparound proof needs capacity >= entries;
    // beyond the memory cap (4M entries = 64 MB across slots) the
    // persistent mode falls back to level-wise instead of corrupting.
    constexpr u32 kPersistMaxEntries = 4u << 20;
    m_persistUsable = entries <= kPersistMaxEntries;
    if (!m_persistUsable) entries = kPersistMaxEntries;
    m_persistCapacity = 1;
    while (m_persistCapacity < entries && m_persistCapacity < (1u << 30))
      m_persistCapacity <<= 1;
    if (!m_persistUsable)
      KENSHIN_WARN(
          "[vgchain] persistent cull needs {} queue entries (> {} cap) - "
          "falling back to level-wise",
          m_instanceCount * (m_hierNodeCount + 1u), kPersistMaxEntries);
    // One-time EMPTY init via initialData (the HZB 0xFF house pattern -
    // fillBuffer needs an active recording): consumers re-empty their
    // slots every frame, so the invariant holds without per-frame clears.
    const std::vector<u32> empties(usize(m_persistCapacity) * 2, 0xFFFFFFFFu);
    for (u32 sl = 0; sl < kSlots; ++sl) {
      m_persistQueue[sl] =
          makeDeviceBuffer(backend, empties.size() * sizeof(u32),
                           empties.data(), "chain-persistQ");
      m_persistState[sl] =
          makeHostBuffer(backend, 32, nullptr, "chain-persistState");
    }
    KENSHIN_INFO("[vgchain] persistent queue: {} entries (mask {:#x})",
                 m_persistCapacity, m_persistCapacity - 1u);
  }
  if (!m_hierReady)
    KENSHIN_WARN("[vgchain] no hierarchy data - cullMode falls back to linear");
  if (m_paged) {
    // M6.3: no v2 pools - the raster/decode pool bindings are the page
    // slot pool (page-local cluster offsets + slot-strided section
    // bases; see rasterize_sw.slang). The slot pool itself and the
    // page tables are CALLER-owned (the scene / streaming manager).
    m_positionBuffer = m_pagedInputs.slotPool;
    m_indexBuffer = m_pagedInputs.slotPool;
    m_positionCount = m_pagedInputs.slotU32Total;
    m_indexCount = m_pagedInputs.slotU32Total;
  } else {
    m_positionBuffer =
        makeDeviceBuffer(backend, positions4.size() * sizeof(glm::vec4),
                         positions4.data(), "chain-positions");
    m_indexBuffer = makeDeviceBuffer(backend, indices.size() * sizeof(u32),
                                     indices.data(), "chain-indices");
    // The HW VS clamps into U32 capacities (the shader reads the pools
    // as u32 views): positions are 4 u32 per float4, indices 1.
    m_positionCount = static_cast<u32>(positions4.size()) * 4u;
    m_indexCount = static_cast<u32>(indices.size());
  }
  // Runtime-derived per-cluster AABBs (the occlusion bound; static).
  // Paged mode receives them precomputed (the pools are empty there).
  const std::vector<glm::vec4> aabbs =
      m_paged ? std::vector<glm::vec4>(clusterAabbs.begin(), clusterAabbs.end())
              : computeClusterAabbs(positions4, indices, clusterData);
  m_aabbBuffer = makeDeviceBuffer(backend, aabbs.size() * sizeof(glm::vec4),
                                  aabbs.data(), "chain-aabbs");
  // M6.3: v2-mode stand-ins for the paged table bindings (params-gated
  // reads; the descriptors must still be VALID).
  const u32 zero4[1] = {0};
  m_dummyU32[0] = makeDeviceBuffer(backend, 4, zero4, "chain-dummyU32a");
  m_dummyU32[1] = makeDeviceBuffer(backend, 4, zero4, "chain-dummyU32b");
  m_dummyU32[2] = makeDeviceBuffer(backend, 4, zero4, "chain-dummyPend");
  const u32 zero24[6] = {};
  m_dummyPageHeader = makeDeviceBuffer(backend, 24, zero24, "chain-dummyPh");

  // M6.3 page_touch feedback: two pipeline instances (SW / HW pair
  // lists), per-slot host-visible bitmaps, the HW-list args convert.
  // Compiled/created in v2 runs too - page_touch.slang stays
  // regression-gated and the dispatch is a cheap no-op over dummies.
  m_pageBitmapWords =
      m_paged ? (m_pagedInputs.pageCount + 31u) / 32u : 1u;
  if (!makePipe("page_touch.slang", m_pipeTouchPairs)) return false;
  if (!makePipe("page_touch.slang", m_pipeTouchHw)) return false;
  if (!makePipe("count_to_indirect.slang", m_pipeConvertTouch)) return false;
  const u32 cfgTouch[4] = {0, 0, 7, m_maxPairs};  // groupsX = cull counters[7]
  m_cfgTouch = makeHostBuffer(backend, sizeof(cfgTouch), cfgTouch, "cfgT");
  for (u32 s = 0; s < kSlots; ++s) {
    m_touched[s] = makeHostBuffer(
        backend, usize(m_pageBitmapWords) * sizeof(u32), nullptr,
        "chain-touched");
    // Indirect dispatch args need the INDIRECT usage bit (the makeArgs
    // discipline - storage-only buffers validation-reject here and the
    // driver eventually wedges).
    backend::IRenderBackend::BufferDesc at;
    at.size = 12;
    at.usage = backend::IRenderBackend::BufferStorage |
               backend::IRenderBackend::BufferIndirect;
    at.debugName = "chain-argsTouch";
    m_argsTouch[s] = backend.createBuffer(at);
  }
  // Conversion configs (static): {groupSize, fwdSlot, srcSlot, pad}.
  const u32 cfgExpand[4] = {64, 1, 0, m_maxPairs};
  const u32 cfgCull[4] = {64, 3, 0, m_maxPairs};
  const u32 cfgRaster[4] = {0, 0, 0, m_maxPairs};
  const u32 cfgPost[4] = {0, 0, 5, m_maxPairs};      // post raster: src = cull [5]
  const u32 cfgHwDraw[4] = {384, 0, 7, m_maxPairs};  // 3*128 verts, src = cull [7]
  m_cfgExpand = makeHostBuffer(backend, sizeof(cfgExpand), cfgExpand, "cfgE");
  m_cfgCull = makeHostBuffer(backend, sizeof(cfgCull), cfgCull, "cfgC");
  m_cfgRaster = makeHostBuffer(backend, sizeof(cfgRaster), cfgRaster, "cfgR");
  m_cfgPost = makeHostBuffer(backend, sizeof(cfgPost), cfgPost, "cfgP");
  m_cfgHwDraw = makeHostBuffer(backend, sizeof(cfgHwDraw), cfgHwDraw, "cfgHD");
  // M5.2: hierarchy conversion configs. Level 1's source is expCnt[1]
  // (the visible-instance count convert #0 forwarded); deeper levels
  // read the ping-pong count buffers (slot 0 of each). The post-cull
  // envelope switch (both modes): pending-count sized via cull [4].
  const u32 cfgHierL1[4] = {64, 0, 1, m_maxPairs};
  const u32 cfgHierLn[4] = {64, 0, 0, m_maxPairs};
  m_cfgHier[0] = makeHostBuffer(backend, sizeof(cfgHierL1), cfgHierL1, "cfgH1");
  for (u32 l = 1; l < kMaxHierLevels; ++l)
    m_cfgHier[l] =
        makeHostBuffer(backend, sizeof(cfgHierLn), cfgHierLn, "cfgHL");
  // Post-section gate: full dispatch size when pending > 0 (the L0
  // reduce pass is the largest level dispatch; coarser levels guard
  // on their dst texel counts).
  const u32 cfgFlag[4] = {(width * height + 63u) / 64u, 0, 4, 0};
  m_cfgFlag = makeHostBuffer(backend, sizeof(cfgFlag), cfgFlag, "cfgF");

  const usize listBytes = m_instanceCount * sizeof(u32);
  const usize pairBytes = usize(maxPairs) * sizeof(glm::uvec2);
  for (u32 s = 0; s < kSlots; ++s) {
    m_planes[s] =
        makeHostBuffer(backend, 6 * sizeof(glm::vec4), nullptr, "chain-planes");
    m_cntA[s] = makeHostBuffer(backend, 8, nullptr, "chain-cntA");
    m_visibleList[s] =
        makeDeviceBuffer(backend, listBytes, nullptr, "chain-visible");
    m_workItems[s] =
        makeDeviceBuffer(backend, pairBytes, nullptr, "chain-work");
    m_expCnt[s] = makeHostBuffer(backend, 8, nullptr, "chain-expCnt");
    m_cullParams[s] = makeHostBuffer(backend, 8 * sizeof(glm::vec4), nullptr,
                                     "chain-cullParams");
    m_hzbParams[s] =
        makeHostBuffer(backend, sizeof(glm::vec4), nullptr, "chain-hzbParams");
    // [5] = post-revived count, [6] = cut-triangle accumulation (the
    // cull counters and the post convert share this buffer via cfg[2]
    // src slots; post_cull adds the revived share into [6]).
    m_cullCnt[s] = makeHostBuffer(backend, 32, nullptr, "chain-cullCnt");
    m_pairs[s] = makeDeviceBuffer(backend, pairBytes, nullptr, "chain-pairs");
    m_rasterParams[s] = makeHostBuffer(backend, 12 * sizeof(glm::vec4), nullptr,
                                       "chain-rasterParams");
    m_postRasterParams[s] = makeHostBuffer(backend, 12 * sizeof(glm::vec4),
                                           nullptr, "chain-postRasterParams");
    // 6 u32: [4] = the M6.3 pending-resident skip count (SW + HW).
    m_rasterCnt[s] = makeHostBuffer(backend, 24, nullptr, "chain-rasterCnt");
    // M3.3 post pass: pending (Main-occluded) / revived pair lists,
    // the post raster's counters, and its params (current matrices).
    m_pendingPairs[s] =
        makeDeviceBuffer(backend, pairBytes, nullptr, "chain-pending");
    m_postPairs[s] =
        makeDeviceBuffer(backend, pairBytes, nullptr, "chain-postPairs");
    m_postParams[s] = makeHostBuffer(backend, 8 * sizeof(glm::vec4), nullptr,
                                     "chain-postParams");
    m_postRasterCnt[s] =
        makeHostBuffer(backend, 16, nullptr, "chain-postRasterCnt");
    m_hwPairs[s] =
        makeDeviceBuffer(backend, pairBytes, nullptr, "chain-hwPairs");
    m_hwParams[s] = makeHostBuffer(backend, 8 * sizeof(glm::vec4), nullptr,
                                   "chain-hwParams");
    // M5.2: ping-pong active-node lists + count cursors (capacity
    // maxPairs - visited nodes per instance ~1.55x groups, far under the
    // clusters-per-instance the same bound already covers).
    m_activeNodes[s][0] =
        makeDeviceBuffer(backend, pairBytes, nullptr, "chain-hierA");
    m_activeNodes[s][1] =
        makeDeviceBuffer(backend, pairBytes, nullptr, "chain-hierB");
    m_hierCnt[s][0] = makeHostBuffer(backend, 4, nullptr, "chain-hierCntA");
    m_hierCnt[s][1] = makeHostBuffer(backend, 4, nullptr, "chain-hierCntB");
    m_hierParams[s] = makeHostBuffer(backend, 8 * sizeof(glm::vec4), nullptr,
                                     "chain-hierParams");
    // Indirect dispatch reads these at the DRAW_INDIRECT stage: the
    // usage bit is mandatory (validation rejects storage-only buffers).
    const auto makeArgs = [&](const char* name) {
      backend::IRenderBackend::BufferDesc d;
      d.size = 16;
      d.usage = backend::IRenderBackend::BufferStorage |
                backend::IRenderBackend::BufferIndirect;
      d.debugName = name;
      return backend.createBuffer(d);
    };
    m_argsExpand[s] = makeArgs("argsE");
    m_argsCull[s] = makeArgs("argsC");
    m_argsRaster[s] = makeArgs("argsR");
    m_argsPostRaster[s] = makeArgs("argsPR");
    m_argsPostCull[s] = makeArgs("argsPC");
    m_hwDrawArgs[s] = makeArgs("argsHW");
    for (u32 l = 1; l < kMaxHierLevels; ++l)
      m_argsHier[l][s] = makeArgs("argsH");
    // The pending-gated HZB#1 flag args. Missing since 087c551: the zero
    // handle made the backend silently drop the flag conversion AND the
    // indirect HZB#1 dispatches, leaving the post pass testing against a
    // two-frame-old pyramid instead of the intended same-frame preliminary
    // one (M5.2 pre-fix).
    m_argsHzb1[s] = makeArgs("argsHzb1");
  }
  resize(backend, width, height);
  return true;
}

void VisBufferChain::resize(backend::IRenderBackend& backend, u32 width,
                            u32 height) {
  m_width = width;
  m_height = height;
  for (u32 s = 0; s < kSlots; ++s) {
    if (m_vis[s] != kInvalid && m_vis[s] != 0) backend.destroyBuffer(m_vis[s]);
    // fillBuffer (per-frame zeroing) requires TRANSFER_DST; the tests'
    // VisBuffer readback requires TRANSFER_SRC.
    backend::IRenderBackend::BufferDesc vd;
    vd.size = usize(width) * height * sizeof(u64);
    vd.usage = backend::IRenderBackend::BufferStorage |
               backend::IRenderBackend::BufferTransferDst |
               backend::IRenderBackend::BufferTransferSrc;
    vd.debugName = "chain-vis";
    m_vis[s] = backend.createBuffer(vd);
    if (m_hzb[s] != kInvalid && m_hzb[s] != 0) backend.destroyBuffer(m_hzb[s]);
    // 0xFF init: the atomic climb's contract (ancestors start at the
    // max so every write wins).
    const HzbLayout hz = hzbLayout(width, height);
    const std::vector<u32> ones(hz.totalTexels, 0xFFFFFFFFu);
    backend::IRenderBackend::BufferDesc hd;
    hd.size = hz.totalTexels * sizeof(u32);
    hd.usage = backend::IRenderBackend::BufferStorage |
               backend::IRenderBackend::BufferTransferDst;
    hd.initialData = ones.data();
    hd.debugName = "chain-hzb";
    m_hzb[s] = backend.createBuffer(hd);
    m_hzbValid[s] = false;
  }

  // Bake the per-level reduce params (static per resolution; the L0
  // pass reads the VisBuffer via its (0,0,0,0) marker). The flag's
  // group count rides the host-mapped cfg.
  const HzbLayout hz = hzbLayout(width, height);
  m_hzbLevelCount = hz.levelCount;
  for (u32 l = 0; l < m_hzbLevelCount && l < kMaxHzbLevels; ++l) {
    const glm::vec4 lp(l == 0 ? 0.0f : f32(hz.levelW[l - 1]),
                       l == 0 ? 0.0f : f32(hz.levelH[l - 1]),
                       l == 0 ? 0.0f : f32(hz.levelOffset[l - 1]),
                       f32(hz.levelOffset[l]));
    if (m_hzbLevelParams[l] != kInvalid && m_hzbLevelParams[l] != 0)
      backend.destroyBuffer(m_hzbLevelParams[l]);
    m_hzbLevelParams[l] =
        makeHostBuffer(backend, sizeof(glm::vec4), &lp, "chain-hzbLvl");
  }
  for (u32 l = 0; l < m_hzbLevelCount && l < kMaxHzbLevels; ++l)
    m_hzbLevelTexels[l] = hz.levelW[l] * hz.levelH[l];
  m_hzbL0Groups = (width * height + 63u) / 64u;
  const u32 flagVal[1] = {m_hzbL0Groups};
  if (m_cfgFlag != kInvalid && m_cfgFlag != 0)
    std::memcpy(backend.mappedPtr(m_cfgFlag), flagVal, sizeof(flagVal));
}

void VisBufferChain::execute(backend::IRenderBackend& backend,
                             const ChainInputs& in) {
  if (m_pipeRaster == ~0u || in.width == 0 || in.height == 0) return;
  if (in.width != m_width || in.height != m_height)
    resize(backend, in.width, in.height);
  const u32 slot = backend.currentFrameSlot();
  const u32 prevSlot = 1u - slot;
  // M3.3: the Main cull tests against the HZB the PREVIOUS frame
  // (other slot) left at its end - built from that frame's COMPLETE
  // (post-pass-inclusive) VisBuffer, one frame stale instead of two.
  // Warmup: the final build below runs even while occActive is off,
  // so the flag turns true after the very first frame per slot.
  const bool occActive = in.occlusionEnabled && m_hzbValid[prevSlot];

  // ---- CPU-side per-frame writes (this slot's host-visible memory is
  // free: beginFrame's pacing wait proved its prior frame completed).
  std::memcpy(backend.mappedPtr(m_planes[slot]), in.planes,
              6 * sizeof(glm::vec4));
  const u32 cntA[2] = {0, m_instanceCount};
  std::memcpy(backend.mappedPtr(m_cntA[slot]), cntA, sizeof(cntA));
  std::memset(backend.mappedPtr(m_expCnt[slot]), 0, 8);
  // Ping-pong cursors reset EVERY frame (all modes - the hierarchy path
  // appends into them; leaving them stale accumulates across frames and
  // the level guards read garbage entries).
  std::memset(backend.mappedPtr(m_hierCnt[slot][0]), 0, 4);
  std::memset(backend.mappedPtr(m_hierCnt[slot][1]), 0, 4);
  // Traversal stats (visited/pruned) reset every frame - both drivers
  // count into the state buffer's words [4]/[5] (the level shader binds
  // it as its 18th handle; the persistent shader's state IS this buffer).
  std::memset(backend.mappedPtr(m_persistState[slot]), 0, 32);

  glm::vec4 cullParams[8];
  cullParams[0] = glm::vec4(in.camPos, in.pixelThreshold);
  cullParams[1] = glm::vec4(in.worldPerPixel, 0.0f, in.nearZ, f32(m_maxPairs));
  for (int c = 0; c < 4; ++c)
    cullParams[2 + c] =
        occActive ? m_prevVP[prevSlot][c] : glm::vec4(c == 0 ? 1.0f : 0.0f);
  cullParams[6] =
      glm::vec4(occActive ? 1.0f : 0.0f, f32(in.width), f32(in.height),
                1.0f / std::abs(in.proj[1][1]));  // tanHalf
  cullParams[7] = glm::vec4(f32(in.hwMode), in.hwThresholdPx, 0.0f, 0.0f);
  std::memcpy(backend.mappedPtr(m_cullParams[slot]), cullParams,
              sizeof(cullParams));
  if (in.cullMode != 0 && m_hierReady) {
    // M5.2: the level shader's params = cullParams + the counts its
    // index clamps need (Slang buffers carry no length).
    auto* hp =
        reinterpret_cast<glm::vec4*>(backend.mappedPtr(m_hierParams[slot]));
    const auto* cp = reinterpret_cast<const glm::vec4*>(
        backend.mappedPtr(m_cullParams[slot]));
    for (int k = 0; k < 8; ++k) hp[k] = cp[k];
    hp[1].y = static_cast<f32>(m_hierNodeCount);
    hp[1].w = static_cast<f32>(m_persistCapacity - 1u);  // queue mask
    hp[7].z = static_cast<f32>(m_clusterCount);
    hp[7].w = static_cast<f32>(m_groupCount);
  }

  std::memset(backend.mappedPtr(m_cullCnt[slot]), 0,
              32);  // [7] too: the HW cursor
  std::memset(backend.mappedPtr(m_rasterCnt[slot]), 0, 24);
  std::memset(backend.mappedPtr(m_postRasterCnt[slot]), 0, 24);
  // M6.3: clear this slot's touched-page bitmap (the pacing wait at
  // beginFrame proves the slot's previous frame drained).
  std::memset(backend.mappedPtr(m_touched[slot]), 0,
              usize(m_pageBitmapWords) * sizeof(u32));
  // post_cull's params: the CURRENT viewProj - the preliminary HZB
  // built after the Main raster lives in this frame's space.
  glm::vec4* pp =
      static_cast<glm::vec4*>(backend.mappedPtr(m_postParams[slot]));
  for (int c = 0; c < 8; ++c) pp[c] = glm::vec4(0.0f);
  pp[1].w = f32(m_maxPairs);  // the saturating-append capacity
  pp[0] = glm::vec4(in.camPos, 0.0f);
  {
    const glm::mat4 vp = in.proj * in.view;
    for (int c = 0; c < 4; ++c) pp[2 + c] = vp[c];
  }
  pp[6] = glm::vec4(1.0f, f32(in.width), f32(in.height),
                    1.0f / std::abs(in.proj[1][1]));
  // raster_hw's params: [0..3] the CURRENT viewProj (the VS's clip
  // space), [4] = (w, h) - the VisBuffer/viewport the PS indexes.
  glm::vec4* hp = static_cast<glm::vec4*>(backend.mappedPtr(m_hwParams[slot]));
  for (int c = 0; c < 8; ++c) hp[c] = glm::vec4(0.0f);
  {
    const glm::mat4 vpHw = in.proj * in.view;
    for (int c = 0; c < 4; ++c) hp[c] = vpHw[c];
  }
  // [4].zw / [5].xz: the pool bounds every payload-driven index in the
  // HW VS clamps into (the decode-pass device-lost workaround). Paged
  // mode: the SLOT POOL totals (coarser, still in-bounds). [5].y: the
  // M6.3 paged addressing flag; [5].z: the instance count (the
  // wide-instance VS clamp).
  hp[4] = glm::vec4(f32(in.width), f32(in.height), f32(m_clusterCount),
                    f32(m_indexCount));
  hp[5] = glm::vec4(f32(m_positionCount), m_paged ? 1.0f : 0.0f,
                    f32(m_instanceCount), 0.0f);
  {
    // rasterize_sw's params (float4[12]): [0..3] view, [4..7] proj
    // (glm raw), [8] = (w, h, pairCount(unused by the guard), nearZ),
    // [9].x = yFlip, [9].y = the M6.3 paged addressing flag. The post
    // raster's copy differs ONLY in [9].z = listSel 2 (its payload
    // indexes the revived-pairs table).
    glm::vec4* p =
        static_cast<glm::vec4*>(backend.mappedPtr(m_rasterParams[slot]));
    for (int c = 0; c < 4; ++c) p[c] = in.view[c];
    for (int c = 0; c < 4; ++c) p[4 + c] = in.proj[c];
    p[8] = glm::vec4(f32(in.width), f32(in.height), 0.0f, in.nearZ);
    p[9] = glm::vec4(1.0f, m_paged ? 1.0f : 0.0f, 0.0f, f32(m_maxPairs));
    for (int c = 10; c < 12; ++c) p[c] = glm::vec4(0.0f);
    glm::vec4* pp =
        static_cast<glm::vec4*>(backend.mappedPtr(m_postRasterParams[slot]));
    for (int c = 0; c < 12; ++c) pp[c] = p[c];
    pp[9].z = 2.0f;
  }

  // ---- Record: cull (direct size; N is CPU-known) -> convert ->
  // expand (indirect) -> convert -> cluster cull (indirect) ->
  // convert -> raster (indirect), then the M3.3 post half: HZB #1 ->
  // post cull -> convert -> post raster -> HZB #2. Barriers between
  // every handoff.
  const auto b = [&](u32 handle) { backend.bufferBarrier(handle, false); };

  // M4.3: the WHOLE chain records onto the async compute queue when
  // available (single submission; the graphics frame waits its
  // completion before the HW draw + decode, so frame N's chain
  // overlaps frame N-1's graphics work). Without an async queue the
  // same recording lands on the graphics buffer exactly as before.
  const bool asyncChain = backend.beginChain();

  backend.fillBuffer(m_vis[slot], 0);

  const u32 cullHandles[4] = {m_planes[slot], m_instanceBuffer,
                              m_visibleList[slot], m_cntA[slot]};
  backend.dispatchCompute(m_pipeCull, cullHandles, 4,
                          (m_instanceCount + 63) / 64, 1, 1);
  b(m_cntA[slot]);

  const u32 cv1[4] = {m_cntA[slot], m_cfgExpand, m_argsExpand[slot],
                      m_expCnt[slot]};
  backend.dispatchCompute(m_pipeConvert[0], cv1, 4, 1, 1, 1);
  b(m_argsExpand[slot]);
  b(m_expCnt[slot]);

  // ---- M5.2: the cull stage. Linear (the differential reference) or
  // the level-wise BVH8 traversal - both leave counters[0..7] and the
  // three pair lists in IDENTICAL semantics; the raster stage onward
  // is shared.
  const bool hier = in.cullMode != 0 && m_hierReady;
  // The persistent driver's queue is capacity-capped at init; past the
  // cap the no-wraparound proof breaks, so level-wise takes over.
  const bool persist = hier && in.cullMode == 2u && m_persistUsable;
  if (!hier) {
    const u32 expandHandles[6] = {m_workItems[slot], m_expCnt[slot],
                                  m_visibleList[slot], m_instanceBuffer,
                                  m_meshRangeBuffer, m_pairCapBuffer};
    backend.dispatchComputeIndirect(m_pipeExpand, expandHandles, 6,
                                    m_argsExpand[slot], 0);
    b(m_workItems[slot]);
    b(m_expCnt[slot]);

    const u32 cv2[4] = {m_expCnt[slot], m_cfgCull, m_argsCull[slot],
                        m_cullCnt[slot]};
    backend.dispatchCompute(m_pipeConvert[1], cv2, 4, 1, 1, 1);
    b(m_argsCull[slot]);
    b(m_cullCnt[slot]);

    const u32 clusterHandles[11] = {
        m_cullParams[slot],   m_clusterBuffer,  m_groupBuffer,
        m_workItems[slot],    m_instanceBuffer, m_pairs[slot],
        m_cullCnt[slot],      m_hzb[prevSlot],  m_aabbBuffer,
        m_pendingPairs[slot], m_hwPairs[slot]};
    backend.dispatchComputeIndirect(m_pipeCluster, clusterHandles, 11,
                                    m_argsCull[slot], 0);
    b(m_pairs[slot]);
    b(m_pendingPairs[slot]);
    b(m_hwPairs[slot]);
    b(m_cullCnt[slot]);
  } else if (persist) {
    // ---- M5.3: the persistent-threads traversal. Seed the ticket
    // queue (same argsExpand envelope - the seed count is the visible
    // instance count), then ONE fixed direct dispatch: the workers
    // drain the queue until the outstanding counter hits zero.
    const u32 seedqHandles[6] = {m_visibleList[slot],  m_instanceBuffer,
                                 m_meshHierRootBuffer, m_persistQueue[slot],
                                 m_persistState[slot], m_expCnt[slot]};
    const u32 seedGroups = (m_instanceCount + 63u) / 64u;
    backend.dispatchCompute(m_pipeSeedQ, seedqHandles, 6, seedGroups, 1, 1);
    b(m_persistQueue[slot]);
    b(m_persistState[slot]);

    const u32 persistHandles[15] = {
        m_hierParams[slot], m_clusterBuffer,      m_groupBuffer,
        m_instanceBuffer,   m_pairs[slot],        m_cullCnt[slot],
        m_hzb[prevSlot],    m_aabbBuffer,         m_pendingPairs[slot],
        m_hwPairs[slot],    m_planes[slot],       m_hierBuffer,
        m_groupRangeBuffer, m_persistQueue[slot], m_persistState[slot]};
    backend.dispatchCompute(m_pipePersist, persistHandles, 15, kPersistGroups,
                            1, 1);

    b(m_pairs[slot]);
    b(m_pendingPairs[slot]);
    b(m_hwPairs[slot]);
    b(m_cullCnt[slot]);
    b(m_persistQueue[slot]);
    b(m_persistState[slot]);
  } else {
    // ---- M5.2: level-wise. Seed: visible instances -> (instance,
    // mesh root) entries in the level-0 list (sequential writes; the
    // count is expCnt[1]).
    const u32 seedHandles[6] = {m_visibleList[slot],  m_instanceBuffer,
                                m_meshHierRootBuffer, m_activeNodes[slot][0],
                                m_expCnt[slot],       m_hierCnt[slot][0]};
    backend.dispatchComputeIndirect(m_pipeHierSeed, seedHandles, 6,
                                    m_argsExpand[slot], 0);
    b(m_activeNodes[slot][0]);
    b(m_hierCnt[slot][0]);

    // Level 1's dispatch args come from expCnt[1] (the visible count).
    const u32 cvh1[4] = {m_expCnt[slot], m_cfgHier[0], m_argsHier[1][slot],
                         m_expCnt[slot]};
    backend.dispatchCompute(m_pipeConvertHier[0], cvh1, 4, 1, 1, 1);
    b(m_argsHier[1][slot]);

    // Levels 1..depth: parity ping-pong of the in/out lists; the
    // per-level convert sites each own their pipeline (one set update
    // per site per frame - the M3.3 discipline).
    for (u32 level = 1; level <= m_hierDepth; ++level) {
      const u32 parity = (level - 1) & 1;
      const u32 levelHandles[19] = {m_hierParams[slot],
                                    m_clusterBuffer,
                                    m_groupBuffer,
                                    m_instanceBuffer,
                                    m_pairs[slot],
                                    m_cullCnt[slot],
                                    m_hzb[prevSlot],
                                    m_aabbBuffer,
                                    m_pendingPairs[slot],
                                    m_hwPairs[slot],
                                    m_planes[slot],
                                    m_hierBuffer,
                                    m_groupRangeBuffer,
                                    m_activeNodes[slot][parity],
                                    m_activeNodes[slot][parity ^ 1],
                                    m_hierCnt[slot][parity],
                                    m_hierCnt[slot][parity ^ 1],
                                    m_pairCapBuffer,
                                    m_persistState[slot]};
      backend.dispatchComputeIndirect(m_pipeHierLevel[level - 1], levelHandles,
                                      19, m_argsHier[level][slot], 0);
      b(m_pairs[slot]);
      b(m_pendingPairs[slot]);
      b(m_hwPairs[slot]);
      b(m_cullCnt[slot]);
      b(m_activeNodes[slot][parity ^ 1]);
      b(m_hierCnt[slot][parity ^ 1]);
      if (level < m_hierDepth) {
        const u32 cvh[4] = {m_hierCnt[slot][level & 1], m_cfgHier[1],
                            m_argsHier[level + 1][slot], m_expCnt[slot]};
        backend.dispatchCompute(m_pipeConvertHier[level], cvh, 4, 1, 1, 1);
        b(m_argsHier[level + 1][slot]);
      }
    }
  }

  const u32 cv3[4] = {m_cullCnt[slot], m_cfgRaster, m_argsRaster[slot],
                      m_rasterCnt[slot]};
  backend.dispatchCompute(m_pipeConvert[2], cv3, 4, 1, 1, 1);
  b(m_argsRaster[slot]);
  b(m_rasterCnt[slot]);

  // M6.3: the paged tables at bindings 8/9/10 (dummies in v2 - the
  // shader's reads are params-gated there). Bindings 1/2 are the slot
  // pool itself in paged mode (m_positionBuffer aliases it).
  const u32 pagedClusterPage =
      m_paged ? m_pagedInputs.clusterPage : m_dummyU32[0];
  const u32 pagedPageTable = m_paged ? m_pagedInputs.pageTable : m_dummyU32[1];
  const u32 pagedPageHeader =
      m_paged ? m_pagedInputs.pageHeader : m_dummyPageHeader;
  const u32 pagedClusterMeta =
      m_paged ? m_pagedInputs.clusterMeta : m_dummyU32[0];
  const u32 rasterHandles[12] = {
      m_rasterParams[slot], m_positionBuffer, m_indexBuffer, m_clusterBuffer,
      m_instanceBuffer,     m_pairs[slot],    m_vis[slot],   m_rasterCnt[slot],
      pagedClusterPage,     pagedPageTable,   pagedPageHeader,
      pagedClusterMeta};
  backend.dispatchComputeIndirect(m_pipeRaster, rasterHandles, 12,
                                  m_argsRaster[slot], 0);

  // ---- M3.3 post half + the final HZB stay on the CHAIN side (they
  // are pure compute); the M4.2 HW draw moved AFTER the chain submit
  // (it needs the graphics queue's rasterizer - running it there is
  // the whole point of the async split).
  if (occActive) {
    // ---- M3.3 post half (Karis Slide 75). The pyramid builds LEVEL
    // BY LEVEL (plain min-of-4). NOTE: the 087c551 pending-gating
    // (flag conversion + one shared indirect args buffer) never
    // actually ran - the args buffer was never created until the M5.2
    // pre-fix, and reviving it crashed occlusion-active views: the
    // never-tested path was broken. Direct per-level dispatches like
    // HZB #2; the gating optimization can return when done correctly.
    b(m_vis[slot]);
    for (u32 l = 0; l < m_hzbLevelCount; ++l) {
      const u32 reduceHandles[3] = {m_hzbLevelParams[l], m_vis[slot],
                                    m_hzb[slot]};
      const u32 groups =
          l == 0 ? m_hzbL0Groups : (m_hzbLevelTexels[l] + 63) / 64;
      backend.dispatchCompute(m_pipeReduce1[l], reduceHandles, 3, groups, 1, 1);
      b(m_hzb[slot]);
    }

    // M5.2: the post cull's envelope is PENDING-sized (counters[4]) -
    // REQUIRED in hierarchy mode: argsCull is only written by the
    // linear branch's convert, so reusing it there reads a stale
    // envelope and revives die. (The historical "flag args broke
    // revives" note was the never-created argsHzb1 masking, not this.)
    const u32 cvPostCull[4] = {m_cullCnt[slot], m_cfgPostCull,
                               m_argsPostCull[slot], m_expCnt[slot]};
    backend.dispatchCompute(m_pipeConvert[4], cvPostCull, 4, 1, 1, 1);
    backend.barrierIndirectArgs(m_argsPostCull[slot]);

    const u32 postCullHandles[8] = {m_postParams[slot], m_pendingPairs[slot],
                                    m_postPairs[slot],  m_instanceBuffer,
                                    m_hzb[slot],        m_aabbBuffer,
                                    m_cullCnt[slot],    m_clusterBuffer};
    backend.dispatchComputeIndirect(m_pipePostCull, postCullHandles, 8,
                                    m_argsPostCull[slot], 0);
    b(m_postPairs[slot]);
    b(m_cullCnt[slot]);

    const u32 cv4[4] = {m_cullCnt[slot], m_cfgPost, m_argsPostRaster[slot],
                        m_postRasterCnt[slot]};
    backend.dispatchCompute(m_pipeConvert[3], cv4, 4, 1, 1, 1);
    b(m_argsPostRaster[slot]);
    b(m_postRasterCnt[slot]);

    const u32 postRasterHandles[12] = {
        m_postRasterParams[slot], m_positionBuffer,   m_indexBuffer,
        m_clusterBuffer,          m_instanceBuffer,   m_postPairs[slot],
        m_vis[slot],              m_postRasterCnt[slot], pagedClusterPage,
        pagedPageTable,           pagedPageHeader,    pagedClusterMeta};
    backend.dispatchComputeIndirect(m_pipeRasterPost, postRasterHandles, 12,
                                    m_argsPostRaster[slot], 0);
    b(m_vis[slot]);
  }

  // Order EVERY prior write of the vis (the main SW raster - and, on
  // the pre-M4.3 layout, the HW pass) before HZB #2's L0 read. The
  // post section's leading b(m_vis) only covers the occActive path;
  // warmup frames (occ off, post skipped) previously leaned on the
  // HW section's trailing barrier - the M4.3 restructure moved that
  // section after this loop and the race surfaced as a zero pyramid.
  b(m_vis[slot]);

  if (in.occlusionEnabled) {
    // HZB #2: from the COMPLETE VisBuffer (post pass included), for
    // the next frame's Main cull - always-on (the bootstrap that
    // turns m_hzbValid on), DIRECT dispatches. Its own reduce
    // pipeline family: #1 already used each level's slot sets.
    for (u32 l = 0; l < m_hzbLevelCount; ++l) {
      const u32 groups =
          l == 0 ? m_hzbL0Groups : (m_hzbLevelTexels[l] + 63u) / 64u;
      const u32 reduceHandles[3] = {m_hzbLevelParams[l], m_vis[slot],
                                    m_hzb[slot]};
      backend.dispatchCompute(m_pipeReduce2[l], reduceHandles, 3, groups, 1, 1);
      b(m_hzb[slot]);
    }
  }

  // Compute write -> fragment read (the decode pass consumes the vis).
  // On the async path this in-cb barrier is advisory only - the real
  // cross-queue handoff is the compute timeline wait on the graphics
  // submission (see endFrame); harmless to keep for the fallback.
  backend.bufferBarrier(m_vis[slot], true);
  backend.bufferBarrier(m_vis[slot], false);

  // ---- M6.3 page_touch feedback: mark the pages of every rasterized
  // (SW + HW) pair in this slot's bitmap. Paged chains only - the
  // shader reads clusterPage unconditionally and v2's dummy would be
  // out of bounds. The SW list reuses the raster's own dispatch args
  // (same {pairCount,1,1} shape); the HW list gets its own convert
  // from cull counters[7]. Post-pass revived pairs are NOT walked (a
  // documented v1 gap - occluded-cluster pages under-refresh in LRU).
  if (m_paged) {
    const u32 cvTouch[4] = {m_cullCnt[slot], m_cfgTouch, m_argsTouch[slot],
                            m_expCnt[slot]};
    backend.dispatchCompute(m_pipeConvertTouch, cvTouch, 4, 1, 1, 1);
    backend.barrierIndirectArgs(m_argsTouch[slot]);
    const u32 touchHandles[3] = {m_pairs[slot], m_pagedInputs.clusterPage,
                                 m_touched[slot]};
    backend.dispatchComputeIndirect(m_pipeTouchPairs, touchHandles, 3,
                                    m_argsRaster[slot], 0);
    const u32 touchHwHandles[3] = {m_hwPairs[slot], m_pagedInputs.clusterPage,
                                   m_touched[slot]};
    backend.dispatchComputeIndirect(m_pipeTouchHw, touchHwHandles, 3,
                                    m_argsTouch[slot], 0);
  }

  // Submit the chain (no-op on the graphics fallback path) BEFORE
  // recording the graphics-side HW draw.
  backend.submitChain();
  backend.submitChain();

  // ---- M4.2 hardware raster (graphics queue): the routed clusters
  // draw through the fixed function (zero attachments; the PS re-runs
  // the SW coverage test on the quantized triangle and atomically
  // writes the SAME VisBuffer). The draw's indirect args are
  // GPU-written INSIDE the chain - count_to_draw runs at the chain's
  // tail, and the graphics submission's timeline wait orders the
  // DRAW_INDIRECT/vertex reads.
  if (in.hwMode > 0) {
    const u32 cvHw[3] = {m_cullCnt[slot], m_cfgHwDraw, m_hwDrawArgs[slot]};
    backend.dispatchCompute(m_pipeCountToDraw, cvHw, 3, 1, 1, 1);
    backend.barrierIndirectArgs(m_hwDrawArgs[slot]);
    // M6.3: bindings 7/8/9/10 = the paged tables + the pending counter
    // (the same rasterCnt[4] word the SW path skips count into; v2
    // binds dummies - the VS's paged reads are params-gated).
    const u32 hwHandles[12] = {m_hwParams[slot],
                               m_hwPairs[slot],
                               m_clusterBuffer,
                               m_positionBuffer,
                               m_indexBuffer,
                               m_instanceBuffer,
                               m_vis[slot],
                               pagedClusterPage,
                               pagedPageTable,
                               pagedPageHeader,
                               m_rasterCnt[slot],
                               pagedClusterMeta};
    backend.drawNullIndirect(m_pipeHwRaster, hwHandles, 11, m_hwDrawArgs[slot],
                             0, in.width, in.height);
  }
  (void)asyncChain;

  // This frame's matrices become the slot's stored prev: the final
  // HZB just built from this slot's VisBuffer was rendered with
  // exactly these.
  m_prevVP[slot] = in.proj * in.view;
  m_prevCam[slot] = in.camPos;
  m_hzbValid[slot] = in.occlusionEnabled;
}

u32 VisBufferChain::visBufferCurrent(backend::IRenderBackend& backend) const {
  return m_vis[backend.currentFrameSlot()];
}

u32 VisBufferChain::visBuffer(backend::IRenderBackend& backend) const {
  const u32 slot = backend.lastCompletedFrameSlot();
  return slot == 0xFFFFFFFFu ? kInvalid : m_vis[slot];
}

u32 VisBufferChain::pairsCurrent(backend::IRenderBackend& backend) const {
  return m_pairs[backend.currentFrameSlot()];
}

u32 VisBufferChain::hwPairsCurrent(backend::IRenderBackend& backend) const {
  return m_hwPairs[backend.currentFrameSlot()];
}

u32 VisBufferChain::postPairsCurrent(backend::IRenderBackend& backend) const {
  return m_postPairs[backend.currentFrameSlot()];
}

bool VisBufferChain::readHwPairs(backend::IRenderBackend& backend,
                                 std::vector<glm::uvec2>& out) const {
  const u32 slot = backend.lastCompletedFrameSlot();
  if (slot == 0xFFFFFFFFu) return false;
  u32 stats[16] = {};
  if (!readStats(backend, stats)) return false;
  out.clear();
  if (stats[12] == 0) return true;
  std::vector<u8> bytes;
  if (!backend.readbackBuffer(m_hwPairs[slot], 0,
                              stats[12] * sizeof(glm::uvec2), bytes))
    return false;
  out.resize(stats[12]);
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return true;
}

bool VisBufferChain::readPostPairs(backend::IRenderBackend& backend,
                                   std::vector<glm::uvec2>& out) const {
  const u32 slot = backend.lastCompletedFrameSlot();
  if (slot == 0xFFFFFFFFu) return false;
  u32 stats[16] = {};
  if (!readStats(backend, stats)) return false;
  out.clear();
  if (stats[9] == 0) return true;
  std::vector<u8> bytes;
  if (!backend.readbackBuffer(m_postPairs[slot], 0,
                              stats[9] * sizeof(glm::uvec2), bytes))
    return false;
  out.resize(stats[9]);
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return true;
}

bool VisBufferChain::readStats(backend::IRenderBackend& backend,
                               u32 out[16]) const {
  const u32 slot = backend.lastCompletedFrameSlot();
  if (slot == 0xFFFFFFFFu) return false;  // no completed frame yet
  std::vector<u8> a(8), c(32), r(24);
  if (!backend.readbackBuffer(m_cntA[slot], 0, 8, a)) return false;
  if (!backend.readbackBuffer(m_cullCnt[slot], 0, 32, c)) return false;
  if (!backend.readbackBuffer(m_rasterCnt[slot], 0, 24, r)) return false;
  std::vector<u8> e(8);
  if (!backend.readbackBuffer(m_expCnt[slot], 0, 8, e)) return false;
  std::vector<u8> pr(16);
  if (!backend.readbackBuffer(m_postRasterCnt[slot], 0, 16, pr)) return false;
  const u32* ac = reinterpret_cast<const u32*>(a.data());
  const u32* cc = reinterpret_cast<const u32*>(c.data());
  const u32* rc = reinterpret_cast<const u32*>(r.data());
  const u32* prc = reinterpret_cast<const u32*>(pr.data());
  out[0] = ac[0];  // visible instances
  out[1] =
      *reinterpret_cast<const u32*>(e.data());  // work items (expand cursor)
  out[2] = cc[0];                               // visible pairs (clusters cut)
  out[3] = cc[1];                               // too-coarse
  out[4] = cc[2];                               // source-too-coarse
  out[5] = rc[1];                               // px writes (main raster)
  out[6] = rc[2];                               // backface culls
  out[7] = rc[3];                               // near-clipped
  out[8] = cc[4];                               // occlusion-culled (pending)
  out[9] = cc[5];                               // post-pass revived
  out[10] = prc[1];                             // post-raster px writes
  out[11] = cc[6];                              // cut tris (main + revived)
  out[12] = cc[7];                              // HW-routed clusters
  {
    std::vector<u8> ps(32);
    if (backend.readbackBuffer(m_persistState[slot], 0, 32, ps)) {
      const u32* psw = reinterpret_cast<const u32*>(ps.data());
      out[13] = psw[4];  // visited nodes
      out[14] = psw[5];  // pruned nodes
    } else {
      out[13] = out[14] = 0;
    }
  }
  out[15] = rc[4];  // pending-resident skips (M6.3)
  return true;
}

bool VisBufferChain::readStatsMapped(backend::IRenderBackend& backend,
                                     u32 out[16]) const {
  // The counter buffers are HOST_VISIBLE + persistently mapped. Read
  // the CURRENT slot, called BEFORE this frame's execute: its last
  // use was the frame-before-last, which beginFrame's pacing wait
  // has PROVEN complete - the values are settled. (The previously
  // submitted frame is NOT safe: a submitted frame may still be
  // executing - single-queue submissions overlap - and reading its
  // counters races the GPU's writes.)
  const u32 slot = backend.currentFrameSlot();
  if (m_cullCnt[slot] == kInvalid || m_cullCnt[slot] == 0) return false;
  const u32* cc = static_cast<const u32*>(backend.mappedPtr(m_cullCnt[slot]));
  const u32* rc = static_cast<const u32*>(backend.mappedPtr(m_rasterCnt[slot]));
  const u32* prc =
      static_cast<const u32*>(backend.mappedPtr(m_postRasterCnt[slot]));
  const u32* ac = static_cast<const u32*>(backend.mappedPtr(m_cntA[slot]));
  const u32* e = static_cast<const u32*>(backend.mappedPtr(m_expCnt[slot]));
  out[0] = ac ? ac[0] : 0;
  out[1] = e ? e[0] : 0;
  out[2] = cc ? cc[0] : 0;
  out[3] = cc ? cc[1] : 0;
  out[4] = cc ? cc[2] : 0;
  out[5] = rc ? rc[1] : 0;
  out[6] = rc ? rc[2] : 0;
  out[7] = rc ? rc[3] : 0;
  out[8] = cc ? cc[4] : 0;
  out[9] = cc ? cc[5] : 0;
  out[10] = prc ? prc[1] : 0;
  out[11] = cc ? cc[6] : 0;
  out[12] = cc ? cc[7] : 0;
  const u32* ps =
      static_cast<const u32*>(backend.mappedPtr(m_persistState[slot]));
  out[13] = ps ? ps[4] : 0;  // visited nodes
  out[14] = ps ? ps[5] : 0;  // pruned nodes
  out[15] = rc ? rc[4] : 0;  // pending-resident skips (M6.3)
  return cc != nullptr;
}

bool VisBufferChain::readTouchedPages(backend::IRenderBackend& backend,
                                      std::vector<u32>& out) const {
  // The readStatsMapped contract: call between endFrame and the NEXT
  // execute - the current slot's bitmap was fully written by the
  // frame-before-last (beginFrame's pacing wait proved it complete)
  // and execute has not yet cleared it for this frame.
  const u32 slot = backend.currentFrameSlot();
  if (m_touched[slot] == kInvalid || m_touched[slot] == 0) return false;
  const auto* words =
      static_cast<const u32*>(backend.mappedPtr(m_touched[slot]));
  if (!words) return false;
  out.assign(words, words + m_pageBitmapWords);
  return true;
}

bool VisBufferChain::readVisiblePairs(backend::IRenderBackend& backend,
                                      std::vector<glm::uvec2>& out) const {
  const u32 slot = backend.lastCompletedFrameSlot();
  if (slot == 0xFFFFFFFFu) return false;  // no completed frame yet
  u32 stats[16] = {};
  if (!readStats(backend, stats)) return false;
  const usize count = stats[2];
  out.clear();
  if (count == 0) return true;
  std::vector<u8> bytes;
  if (!backend.readbackBuffer(m_pairs[slot], 0, count * sizeof(glm::uvec2),
                              bytes))
    return false;
  out.resize(count);
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return true;
}

bool VisBufferChain::readIndirectArgs(backend::IRenderBackend& backend,
                                      u32 expandX[3], u32 cullX[1],
                                      u32 rasterX[1]) const {
  const u32 slot = backend.lastCompletedFrameSlot();
  if (slot == 0xFFFFFFFFu) return false;  // no completed frame yet
  std::vector<u8> e(4), c(4), r(4), ec(8);
  if (!backend.readbackBuffer(m_argsExpand[slot], 0, 4, e)) return false;
  if (!backend.readbackBuffer(m_argsCull[slot], 0, 4, c)) return false;
  if (!backend.readbackBuffer(m_argsRaster[slot], 0, 4, r)) return false;
  if (!backend.readbackBuffer(m_expCnt[slot], 0, 8, ec)) return false;
  expandX[0] = *reinterpret_cast<const u32*>(e.data());
  expandX[1] = reinterpret_cast<const u32*>(ec.data())[0];  // cursor
  expandX[2] = reinterpret_cast<const u32*>(ec.data())[1];  // fwd slot
  cullX[0] = *reinterpret_cast<const u32*>(c.data());
  rasterX[0] = *reinterpret_cast<const u32*>(r.data());
  return true;
}

void VisBufferChain::onShutdown(backend::IRenderBackend& backend) {
  const u32 pipes[10] = {m_pipeCull,       m_pipeExpand,     m_pipeCluster,
                         m_pipeRaster,     m_pipeRasterPost, m_pipePostCull,
                         m_pipeConvert[0], m_pipeConvert[1], m_pipeConvert[2],
                         m_pipeCountToDraw};
  const u32 pipes2[4 + 2 * kMaxHzbLevels] = {
      m_pipeConvert[3],  m_pipeCountToFlag, m_pipeHzb,
      m_pipeHzbFinal,    m_pipeReduce1[0],  m_pipeReduce1[1],
      m_pipeReduce1[2],  m_pipeReduce1[3],  m_pipeReduce1[4],
      m_pipeReduce1[5],  m_pipeReduce1[6],  m_pipeReduce1[7],
      m_pipeReduce1[8],  m_pipeReduce1[9],  m_pipeReduce1[10],
      m_pipeReduce1[11], m_pipeReduce1[12], m_pipeReduce1[13],
      m_pipeReduce1[14], m_pipeReduce1[15], m_pipeReduce2[0],
      m_pipeReduce2[1],  m_pipeReduce2[2],  m_pipeReduce2[3],
      m_pipeReduce2[4],  m_pipeReduce2[5],  m_pipeReduce2[6],
      m_pipeReduce2[7],  m_pipeReduce2[8],  m_pipeReduce2[9],
      m_pipeReduce2[10], m_pipeReduce2[11], m_pipeReduce2[12],
      m_pipeReduce2[13], m_pipeReduce2[14], m_pipeReduce2[15]};
  for (const u32 p : pipes)
    if (p != ~0u) backend.destroyComputePipeline(p);
  for (const u32 p : pipes2)
    if (p != ~0u) backend.destroyComputePipeline(p);
  if (m_pipeHwRaster != ~0u) backend.destroyNullPipeline(m_pipeHwRaster);
  {
    const u32 hierPipes[3 + 2 * kMaxHierLevels] = {
        m_pipeHierSeed,        m_pipeSeedQ,           m_pipePersist,
        m_pipeHierLevel[0],    m_pipeHierLevel[1],    m_pipeHierLevel[2],
        m_pipeHierLevel[3],    m_pipeHierLevel[4],    m_pipeHierLevel[5],
        m_pipeHierLevel[6],    m_pipeHierLevel[7],    m_pipeHierLevel[8],
        m_pipeHierLevel[9],    m_pipeHierLevel[10],   m_pipeHierLevel[11],
        m_pipeHierLevel[12],   m_pipeHierLevel[13],   m_pipeHierLevel[14],
        m_pipeHierLevel[15],   m_pipeConvertHier[0],  m_pipeConvertHier[1],
        m_pipeConvertHier[2],  m_pipeConvertHier[3],  m_pipeConvertHier[4],
        m_pipeConvertHier[5],  m_pipeConvertHier[6],  m_pipeConvertHier[7],
        m_pipeConvertHier[8],  m_pipeConvertHier[9],  m_pipeConvertHier[10],
        m_pipeConvertHier[11], m_pipeConvertHier[12], m_pipeConvertHier[13],
        m_pipeConvertHier[14], m_pipeConvertHier[15]};
    for (const u32 p : hierPipes)
      if (p != ~0u && p != 0) backend.destroyComputePipeline(p);
  }
  for (u32 s = 0; s < kSlots; ++s) {
    for (u32 l = 1; l < kMaxHierLevels; ++l)
      if (m_argsHier[l][s] != kInvalid && m_argsHier[l][s] != 0)
        backend.destroyBuffer(m_argsHier[l][s]);
    for (const u32 h : {m_activeNodes[s][0], m_activeNodes[s][1],
                        m_hierCnt[s][0], m_hierCnt[s][1], m_hierParams[s],
                        m_persistQueue[s], m_persistState[s]})
      if (h != kInvalid && h != 0) backend.destroyBuffer(h);
    for (const u32 h :
         {m_planes[s],     m_cntA[s],          m_visibleList[s],
          m_workItems[s],  m_expCnt[s],        m_cullParams[s],
          m_cullCnt[s],    m_pairs[s],         m_rasterParams[s],
          m_rasterCnt[s],  m_vis[s],           m_hzb[s],
          m_hzbParams[s],  m_pendingPairs[s],  m_postPairs[s],
          m_postParams[s], m_postRasterCnt[s], m_argsExpand[s],
          m_argsCull[s],   m_argsRaster[s],    m_argsPostRaster[s],
          m_hwPairs[s],    m_hwParams[s],      m_hwDrawArgs[s],
          m_argsHzb1[s],   m_argsPostCull[s],  m_touched[s],
          m_argsTouch[s],  m_postRasterParams[s]})
      if (h != kInvalid && h != 0) backend.destroyBuffer(h);
  }
  // M6.3: in paged mode the pool/table handles are CALLER-owned - only
  // the v2 pools (created here) die with the chain. The v2 dummies and
  // the touch pipelines are ours either way.
  for (const u32 h :
       {m_instanceBuffer, m_meshRangeBuffer, m_clusterBuffer, m_groupBuffer,
        m_paged ? m_dummyU32[0] : m_positionBuffer,
        m_paged ? m_dummyU32[1] : m_indexBuffer, m_dummyU32[2],
        m_dummyPageHeader, m_cfgExpand, m_cfgCull, m_cfgRaster,
        m_cfgPost, m_cfgHwDraw, m_cfgPostCull, m_aabbBuffer, m_hierBuffer,
        m_groupRangeBuffer, m_meshHierRootBuffer, m_cfgTouch})
    if (h != kInvalid && h != 0) backend.destroyBuffer(h);
  if (m_pairCapBuffer != kInvalid && m_pairCapBuffer != 0)
    backend.destroyBuffer(m_pairCapBuffer);
  for (const u32 p : {m_pipeTouchPairs, m_pipeTouchHw, m_pipeConvertTouch})
    if (p != ~0u && p != 0) backend.destroyComputePipeline(p);
  for (u32 l = 0; l < kMaxHierLevels; ++l)
    if (m_cfgHier[l] != kInvalid && m_cfgHier[l] != 0)
      backend.destroyBuffer(m_cfgHier[l]);
  for (u32 l = 0; l < kMaxHzbLevels; ++l)
    if (m_hzbLevelParams[l] != kInvalid && m_hzbLevelParams[l] != 0)
      backend.destroyBuffer(m_hzbLevelParams[l]);
  if (m_cfgFlag != kInvalid && m_cfgFlag != 0) backend.destroyBuffer(m_cfgFlag);
}

}  // namespace renderer

KENSHIN_NS_END
