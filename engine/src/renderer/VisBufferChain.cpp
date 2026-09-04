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
  if (!makePipe("build_hzb.slang", m_pipeHzb)) return false;
  if (!makePipe("rasterize_sw.slang", m_pipeRaster)) return false;
  for (u32 k = 0; k < 3; ++k)
    if (!makePipe("count_to_indirect.slang", m_pipeConvert[k])) return false;

  m_instanceCount = static_cast<u32>(instances.size());
  m_maxPairs = maxPairs;
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
      computeClusterAabbs(positions4, clusterData);
  m_aabbBuffer = makeDeviceBuffer(backend, aabbs.size() * sizeof(glm::vec4),
                                  aabbs.data(), "chain-aabbs");
  // Conversion configs (static): {groupSize, fwdSlot}.
  const u32 cfgExpand[4] = {64, 1, 0, 0};
  const u32 cfgCull[4] = {64, 3, 0, 0};
  const u32 cfgRaster[4] = {0, 0, 0, 0};
  m_cfgExpand = makeHostBuffer(backend, sizeof(cfgExpand), cfgExpand, "cfgE");
  m_cfgCull = makeHostBuffer(backend, sizeof(cfgCull), cfgCull, "cfgC");
  m_cfgRaster = makeHostBuffer(backend, sizeof(cfgRaster), cfgRaster, "cfgR");

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
    m_cullCnt[s] = makeHostBuffer(backend, 20, nullptr, "chain-cullCnt");
    m_pairs[s] = makeDeviceBuffer(backend, pairBytes, nullptr, "chain-pairs");
    m_rasterParams[s] = makeHostBuffer(backend, 12 * sizeof(glm::vec4), nullptr,
                                       "chain-rasterParams");
    m_rasterCnt[s] = makeHostBuffer(backend, 16, nullptr, "chain-rasterCnt");
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
}

void VisBufferChain::execute(backend::IRenderBackend& backend,
                             const ChainInputs& in) {
  if (m_pipeRaster == ~0u || in.width == 0 || in.height == 0) return;
  if (in.width != m_width || in.height != m_height)
    resize(backend, in.width, in.height);
  const u32 slot = backend.currentFrameSlot();

  // ---- CPU-side per-frame writes (this slot's host-visible memory is
  // free: beginFrame's pacing wait proved its prior frame completed).
  std::memcpy(backend.mappedPtr(m_planes[slot]), in.planes,
              6 * sizeof(glm::vec4));
  const u32 cntA[2] = {0, m_instanceCount};
  std::memcpy(backend.mappedPtr(m_cntA[slot]), cntA, sizeof(cntA));
  std::memset(backend.mappedPtr(m_expCnt[slot]), 0, 8);
  glm::vec4 cullParams[8];
  cullParams[0] = glm::vec4(in.camPos, in.pixelThreshold);
  cullParams[1] = glm::vec4(in.worldPerPixel, 0.0f, 0.0f, 0.0f);
  for (int c = 0; c < 4; ++c)
    cullParams[2 + c] =
        m_hzbValid[slot] ? m_prevVP[slot][c] : glm::vec4(c == 0 ? 1.0f : 0.0f);
  cullParams[6] =
      glm::vec4(m_hzbValid[slot] ? 1.0f : 0.0f, f32(in.width), f32(in.height),
                1.0f / std::abs(in.proj[1][1]));  // tanHalf
  cullParams[7] = glm::vec4(m_prevCam[slot], 0.0f);
  std::memcpy(backend.mappedPtr(m_cullParams[slot]), cullParams,
              sizeof(cullParams));
  std::memset(backend.mappedPtr(m_cullCnt[slot]), 0, 20);
  std::memset(backend.mappedPtr(m_rasterCnt[slot]), 0, 16);
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
  // convert -> raster (indirect). Barriers between every handoff.
  const auto b = [&](u32 handle) { backend.bufferBarrier(handle, false); };

  // ---- M3.2b: build this slot's HZB from the VisBuffer the SAME
  // slot wrote two frames ago (provably complete), so the cull below
  // tests against the previously rendered frame. MUST run BEFORE this
  // frame's vis zeroing. The first run per slot / after a resize skips
  // the build (the occ flag stays off).
  if (m_hzbValid[slot]) {
    const glm::vec4 hp(f32(in.width), f32(in.height), 0.0f, 0.0f);
    std::memcpy(backend.mappedPtr(m_hzbParams[slot]), &hp, sizeof(hp));
    backend.fillBuffer(m_hzb[slot], 0xFFFFFFFFu);
    const u32 hzbHandles[3] = {m_hzbParams[slot], m_vis[slot], m_hzb[slot]};
    backend.dispatchCompute(m_pipeHzb, hzbHandles, 3,
                            (in.width * in.height + 63) / 64, 1, 1);
    backend.bufferBarrier(m_hzb[slot], false);
  }

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

  const u32 clusterHandles[9] = {
      m_cullParams[slot], m_clusterBuffer,  m_groupBuffer,
      m_workItems[slot],  m_instanceBuffer, m_pairs[slot],
      m_cullCnt[slot],    m_hzb[slot],      m_aabbBuffer};
  backend.dispatchComputeIndirect(m_pipeCluster, clusterHandles, 9,
                                  m_argsCull[slot], 0);
  b(m_pairs[slot]);
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
  // Compute write -> fragment read (the decode pass consumes the vis);
  // also compute-read for the NEXT frame's HZB build on this slot.
  backend.bufferBarrier(m_vis[slot], true);
  backend.bufferBarrier(m_vis[slot], false);

  // This frame's matrices become the slot's stored prev: the HZB the
  // next frame builds from this slot's VisBuffer was rendered with
  // exactly these.
  m_prevVP[slot] = in.proj * in.view;
  m_prevCam[slot] = in.camPos;
  m_hzbValid[slot] = true;
}

u32 VisBufferChain::visBufferCurrent(backend::IRenderBackend& backend) const {
  return m_vis[backend.currentFrameSlot()];
}

u32 VisBufferChain::visBuffer(backend::IRenderBackend& backend) const {
  const u32 slot = backend.lastCompletedFrameSlot();
  return slot == 0xFFFFFFFFu ? kInvalid : m_vis[slot];
}

bool VisBufferChain::readStats(backend::IRenderBackend& backend,
                               u32 out[9]) const {
  const u32 slot = backend.lastCompletedFrameSlot();
  if (slot == 0xFFFFFFFFu) return false;  // no completed frame yet
  std::vector<u8> a(8), c(16), r(16);
  if (!backend.readbackBuffer(m_cntA[slot], 0, 8, a)) return false;
  if (!backend.readbackBuffer(m_cullCnt[slot], 0, 20, c)) return false;
  if (!backend.readbackBuffer(m_rasterCnt[slot], 0, 16, r)) return false;
  std::vector<u8> e(8);
  if (!backend.readbackBuffer(m_expCnt[slot], 0, 8, e)) return false;
  const u32* ac = reinterpret_cast<const u32*>(a.data());
  const u32* cc = reinterpret_cast<const u32*>(c.data());
  const u32* rc = reinterpret_cast<const u32*>(r.data());
  out[0] = ac[0];  // visible instances
  out[1] =
      *reinterpret_cast<const u32*>(e.data());  // work items (expand cursor)
  out[2] = cc[0];                               // visible pairs (clusters cut)
  out[3] = cc[1];                               // too-coarse
  out[4] = cc[2];                               // source-too-coarse
  out[5] = rc[1];                               // px writes
  out[6] = rc[2];                               // backface culls
  out[7] = rc[3];                               // near-clipped
  out[8] = cc[4];                               // occlusion-culled
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
  const u32 pipes[8] = {m_pipeCull,       m_pipeExpand,    m_pipeCluster,
                        m_pipeRaster,     m_pipeHzb,       m_pipeConvert[0],
                        m_pipeConvert[1], m_pipeConvert[2]};
  for (const u32 p : pipes)
    if (p != ~0u) backend.destroyComputePipeline(p);
  for (u32 s = 0; s < kSlots; ++s) {
    for (const u32 h :
         {m_planes[s], m_cntA[s], m_visibleList[s], m_workItems[s], m_expCnt[s],
          m_cullParams[s], m_cullCnt[s], m_pairs[s], m_rasterParams[s],
          m_rasterCnt[s], m_vis[s], m_hzb[s], m_hzbParams[s], m_argsExpand[s],
          m_argsCull[s], m_argsRaster[s]})
      if (h != kInvalid && h != 0) backend.destroyBuffer(h);
  }
  for (const u32 h : {m_instanceBuffer, m_meshRangeBuffer, m_clusterBuffer,
                      m_groupBuffer, m_positionBuffer, m_indexBuffer,
                      m_cfgExpand, m_cfgCull, m_cfgRaster, m_aabbBuffer})
    if (h != kInvalid && h != 0) backend.destroyBuffer(h);
}

}  // namespace renderer

KENSHIN_NS_END
