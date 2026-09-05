#include "kenshin/renderer/VisBufferChain.h"

#include <cmath>
#include <cstring>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
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

bool VisBufferChain::init(backend::IRenderBackend& backend,
                          manager::ShaderManager& shaders,
                          std::span<const glm::vec4> positions4,
                          std::span<const u32> indices,
                          const std::vector<u8>& clusterData,
                          const std::vector<u8>& groupData,
                          std::span<const CullGpuInstance> instances,
                          std::span<const GpuMeshRangePair> meshRanges,
                          u32 maxPairs, u32 width, u32 height) {
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
  for (u32 k = 0; k < 4; ++k)
    if (!makePipe("count_to_indirect.slang", m_pipeConvert[k])) return false;

  m_instanceCount = static_cast<u32>(instances.size());
  m_maxPairs = maxPairs;
  m_clusterCount = static_cast<u32>(clusterData.size() / 48);
  m_indexCount = static_cast<u32>(indices.size());
  m_positionCount = static_cast<u32>(positions4.size());
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
  m_positionBuffer =
      makeDeviceBuffer(backend, positions4.size() * sizeof(glm::vec4),
                       positions4.data(), "chain-positions");
  m_indexBuffer = makeDeviceBuffer(backend, indices.size() * sizeof(u32),
                                   indices.data(), "chain-indices");
  // Runtime-derived per-cluster AABBs (the occlusion bound; static).
  const std::vector<glm::vec4> aabbs =
      computeClusterAabbs(positions4, indices, clusterData);
  m_aabbBuffer = makeDeviceBuffer(backend, aabbs.size() * sizeof(glm::vec4),
                                  aabbs.data(), "chain-aabbs");
  // Conversion configs (static): {groupSize, fwdSlot, srcSlot, pad}.
  const u32 cfgExpand[4] = {64, 1, 0, 0};
  const u32 cfgCull[4] = {64, 3, 0, 0};
  const u32 cfgRaster[4] = {0, 0, 0, 0};
  const u32 cfgPost[4] = {0, 0, 5, 0};      // post raster: src = cull [5]
  const u32 cfgHwDraw[4] = {384, 0, 7, 0};  // 3*128 verts, src = cull [7]
  m_cfgExpand = makeHostBuffer(backend, sizeof(cfgExpand), cfgExpand, "cfgE");
  m_cfgCull = makeHostBuffer(backend, sizeof(cfgCull), cfgCull, "cfgC");
  m_cfgRaster = makeHostBuffer(backend, sizeof(cfgRaster), cfgRaster, "cfgR");
  m_cfgPost = makeHostBuffer(backend, sizeof(cfgPost), cfgPost, "cfgP");
  m_cfgHwDraw = makeHostBuffer(backend, sizeof(cfgHwDraw), cfgHwDraw, "cfgHD");
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
    m_rasterCnt[s] = makeHostBuffer(backend, 16, nullptr, "chain-rasterCnt");
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
    m_hwDrawArgs[s] = makeArgs("argsHW");
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
  glm::vec4 cullParams[8];
  cullParams[0] = glm::vec4(in.camPos, in.pixelThreshold);
  cullParams[1] = glm::vec4(in.worldPerPixel, 0.0f, in.nearZ, 0.0f);
  for (int c = 0; c < 4; ++c)
    cullParams[2 + c] =
        occActive ? m_prevVP[prevSlot][c] : glm::vec4(c == 0 ? 1.0f : 0.0f);
  cullParams[6] =
      glm::vec4(occActive ? 1.0f : 0.0f, f32(in.width), f32(in.height),
                1.0f / std::abs(in.proj[1][1]));  // tanHalf
  cullParams[7] = glm::vec4(f32(in.hwMode), in.hwThresholdPx, 0.0f, 0.0f);
  std::memcpy(backend.mappedPtr(m_cullParams[slot]), cullParams,
              sizeof(cullParams));
  std::memset(backend.mappedPtr(m_cullCnt[slot]), 0, 28);
  std::memset(backend.mappedPtr(m_rasterCnt[slot]), 0, 16);
  std::memset(backend.mappedPtr(m_postRasterCnt[slot]), 0, 16);
  // post_cull's params: the CURRENT viewProj - the preliminary HZB
  // built after the Main raster lives in this frame's space.
  glm::vec4* pp =
      static_cast<glm::vec4*>(backend.mappedPtr(m_postParams[slot]));
  for (int c = 0; c < 8; ++c) pp[c] = glm::vec4(0.0f);
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
  // [4].zw / [5].x: the pool bounds every payload-driven index in the
  // HW VS clamps into (the decode-pass device-lost workaround).
  hp[4] = glm::vec4(f32(in.width), f32(in.height), f32(m_clusterCount),
                    f32(m_indexCount));
  hp[5] = glm::vec4(f32(m_positionCount), 0.0f, 0.0f, 0.0f);
  {
    // rasterize_sw's params (float4[12]): [0..3] view, [4..7] proj
    // (glm raw), [8] = (w, h, pairCount(unused by the guard), nearZ),
    // [9].x = yFlip.
    glm::vec4* p =
        static_cast<glm::vec4*>(backend.mappedPtr(m_rasterParams[slot]));
    for (int c = 0; c < 4; ++c) p[c] = in.view[c];
    for (int c = 0; c < 4; ++c) p[4 + c] = in.proj[c];
    p[8] = glm::vec4(f32(in.width), f32(in.height), 0.0f, in.nearZ);
    p[9] = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
    for (int c = 10; c < 12; ++c) p[c] = glm::vec4(0.0f);
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

  const u32 expandHandles[5] = {m_workItems[slot], m_expCnt[slot],
                                m_visibleList[slot], m_instanceBuffer,
                                m_meshRangeBuffer};
  backend.dispatchComputeIndirect(m_pipeExpand, expandHandles, 5,
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

  const u32 cv3[4] = {m_cullCnt[slot], m_cfgRaster, m_argsRaster[slot],
                      m_rasterCnt[slot]};
  backend.dispatchCompute(m_pipeConvert[2], cv3, 4, 1, 1, 1);
  b(m_argsRaster[slot]);
  b(m_rasterCnt[slot]);

  const u32 rasterHandles[8] = {
      m_rasterParams[slot], m_positionBuffer, m_indexBuffer, m_clusterBuffer,
      m_instanceBuffer,     m_pairs[slot],    m_vis[slot],   m_rasterCnt[slot]};
  backend.dispatchComputeIndirect(m_pipeRaster, rasterHandles, 8,
                                  m_argsRaster[slot], 0);

  // ---- M3.3 post half + the final HZB stay on the CHAIN side (they
  // are pure compute); the M4.2 HW draw moved AFTER the chain submit
  // (it needs the graphics queue's rasterizer - running it there is
  // the whole point of the async split).
  if (occActive) {
    // ---- M3.3 post half (Karis Slide 75), PENDING-GATED: nothing
    // occluded (the common case) -> the whole section dispatches ZERO
    // groups (a flag conversion turns counters[4] into {G|0}). The
    // pyramid builds LEVEL BY LEVEL (plain min-of-4, one writer per
    // texel - no atomics; the old single-dispatch atomic climb had
    // every L0 thread InterlockedMin the 1x1 apex: ~295k serialized
    // same-address atomics at 640x461 = multiple milliseconds).
    b(m_vis[slot]);
    const u32 cvFlag[3] = {m_cullCnt[slot], m_cfgFlag, m_argsHzb1[slot]};
    backend.dispatchCompute(m_pipeCountToFlag, cvFlag, 3, 1, 1, 1);
    backend.barrierIndirectArgs(m_argsHzb1[slot]);

    for (u32 l = 0; l < m_hzbLevelCount; ++l) {
      const u32 reduceHandles[3] = {m_hzbLevelParams[l], m_vis[slot],
                                    m_hzb[slot]};
      backend.dispatchComputeIndirect(m_pipeReduce1[l], reduceHandles, 3,
                                      m_argsHzb1[slot], 0);
      b(m_hzb[slot]);
    }

    // The post cull keeps the MAIN cull's dispatch size (argsCull):
    // its threads guard on the pending count, so the oversize costs
    // nothing but idle early-outs. (Routing it through the FLAG args
    // measurably broke revives - kept on argsCull, the proven shape.)
    const u32 postCullHandles[8] = {m_postParams[slot], m_pendingPairs[slot],
                                    m_postPairs[slot],  m_instanceBuffer,
                                    m_hzb[slot],        m_aabbBuffer,
                                    m_cullCnt[slot],    m_clusterBuffer};
    backend.dispatchComputeIndirect(m_pipePostCull, postCullHandles, 8,
                                    m_argsCull[slot], 0);
    b(m_postPairs[slot]);
    b(m_cullCnt[slot]);

    const u32 cv4[4] = {m_cullCnt[slot], m_cfgPost, m_argsPostRaster[slot],
                        m_postRasterCnt[slot]};
    backend.dispatchCompute(m_pipeConvert[3], cv4, 4, 1, 1, 1);
    b(m_argsPostRaster[slot]);
    b(m_postRasterCnt[slot]);

    const u32 postRasterHandles[8] = {
        m_rasterParams[slot], m_positionBuffer,     m_indexBuffer,
        m_clusterBuffer,      m_instanceBuffer,     m_postPairs[slot],
        m_vis[slot],          m_postRasterCnt[slot]};
    backend.dispatchComputeIndirect(m_pipeRasterPost, postRasterHandles, 8,
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

  // Submit the chain (no-op on the graphics fallback path) BEFORE
  // recording the graphics-side HW draw.
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
    const u32 hwHandles[7] = {
        m_hwParams[slot], m_hwPairs[slot],  m_clusterBuffer, m_positionBuffer,
        m_indexBuffer,    m_instanceBuffer, m_vis[slot]};
    backend.drawNullIndirect(m_pipeHwRaster, hwHandles, 7, m_hwDrawArgs[slot],
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

bool VisBufferChain::readStats(backend::IRenderBackend& backend,
                               u32 out[13]) const {
  const u32 slot = backend.lastCompletedFrameSlot();
  if (slot == 0xFFFFFFFFu) return false;  // no completed frame yet
  std::vector<u8> a(8), c(32), r(16);
  if (!backend.readbackBuffer(m_cntA[slot], 0, 8, a)) return false;
  if (!backend.readbackBuffer(m_cullCnt[slot], 0, 32, c)) return false;
  if (!backend.readbackBuffer(m_rasterCnt[slot], 0, 16, r)) return false;
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
  return true;
}

bool VisBufferChain::readStatsMapped(backend::IRenderBackend& backend,
                                     u32 out[13]) const {
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
  return cc != nullptr;
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
  for (u32 s = 0; s < kSlots; ++s) {
    for (const u32 h :
         {m_planes[s],     m_cntA[s],          m_visibleList[s],
          m_workItems[s],  m_expCnt[s],        m_cullParams[s],
          m_cullCnt[s],    m_pairs[s],         m_rasterParams[s],
          m_rasterCnt[s],  m_vis[s],           m_hzb[s],
          m_hzbParams[s],  m_pendingPairs[s],  m_postPairs[s],
          m_postParams[s], m_postRasterCnt[s], m_argsExpand[s],
          m_argsCull[s],   m_argsRaster[s],    m_argsPostRaster[s],
          m_hwPairs[s],    m_hwParams[s],      m_hwDrawArgs[s]})
      if (h != kInvalid && h != 0) backend.destroyBuffer(h);
  }
  for (const u32 h :
       {m_instanceBuffer, m_meshRangeBuffer, m_clusterBuffer, m_groupBuffer,
        m_positionBuffer, m_indexBuffer, m_cfgExpand, m_cfgCull, m_cfgRaster,
        m_cfgPost, m_cfgHwDraw, m_aabbBuffer})
    if (h != kInvalid && h != 0) backend.destroyBuffer(h);
  for (u32 l = 0; l < kMaxHzbLevels; ++l)
    if (m_hzbLevelParams[l] != kInvalid && m_hzbLevelParams[l] != 0)
      backend.destroyBuffer(m_hzbLevelParams[l]);
  if (m_cfgFlag != kInvalid && m_cfgFlag != 0) backend.destroyBuffer(m_cfgFlag);
}

}  // namespace renderer

KENSHIN_NS_END
