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
};

class VisBufferChain {
 public:
  static constexpr u32 kSlots = 2;  // = backend frames in flight

  /// Compile the five shaders, create the persistent pipelines and
  /// buffers. `maxPairs` bounds work items AND pairs (worst case
  /// visible instances x total clusters).
  bool init(backend::IRenderBackend& backend, manager::ShaderManager& shaders,
            std::span<const glm::vec4> positions4, std::span<const u32> indices,
            const std::vector<u8>& clusterData,
            const std::vector<u8>& groupData,
            std::span<const CullGpuInstance> instances,
            std::span<const GpuMeshRangePair> meshRanges, u32 maxPairs,
            u32 width, u32 height);

  /// (Re)create the per-slot VisBuffers on resolution change.
  void resize(backend::IRenderBackend& backend, u32 width, u32 height);

  /// Record the whole chain into the CURRENT frame (between beginFrame
  /// and endFrame). CPU-writes this slot's params first, then five
  /// dispatches + count conversions, each behind barriers.
  void execute(backend::IRenderBackend& backend, const ChainInputs& in);

  /// The CURRENT frame slot's VisBuffer (for the decode draw; the same
  /// frame's chain produced it).
  u32 visBuffer(backend::IRenderBackend& backend) const;

  /// Blocking stats readback (test/debug), per chain stage:
  /// [0] visible instances, [1] work items, [2] visible pairs (clusters
  /// cut), [3] too-coarse, [4] source-too-coarse, [5] px writes,
  /// [6] backface, [7] near-clipped.
  bool readStats(backend::IRenderBackend& backend, u32 out[8]) const;
  /// Debug probe: the three indirect dispatch commands as written by
  /// the conversions (x = group count).
  bool readIndirectArgs(backend::IRenderBackend& backend, u32 expandX[3],
                        u32 cullX[1], u32 rasterX[1]) const;

  void onShutdown(backend::IRenderBackend& backend);

 private:
  u32 m_pipeCull = ~0u, m_pipeExpand = ~0u, m_pipeCluster = ~0u,
      m_pipeRaster = ~0u, m_pipeConvert = ~0u;
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
  u32 m_argsExpand[kSlots]{}, m_argsCull[kSlots]{}, m_argsRaster[kSlots]{};
  u32 m_instanceCount = 0, m_maxPairs = 0, m_width = 0, m_height = 0;
};

}  // namespace renderer

KENSHIN_NS_END
