#pragma once

// ── PipelineStore — graphics pipeline table + disk cache ────────
// Owns the pipeline slots (handle = index + 1, freed slots keep their
// place with null handles) and the VkPipelineCache persisted next to
// the exe. The renderer layer defines state; this store translates it
// (the enum helpers live in the .cpp).

#include <volk.h>

#include <vector>

#include "FrameSync.h"
#include "kenshin/Common.h"
#include "kenshin/backend/RenderBackend.h"

KENSHIN_NS_BEGIN

namespace backend {

class PipelineStore {
 public:
  struct Pipeline {
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
  };

  /// Compute entry: persistent compute pipeline + per-frame-slot SSBO
  /// descriptor sets (layout from SPIR-V reflection; the in-frame chain
  /// dispatches these every frame).
  struct Compute {
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet sets[kMaxFramesInFlight] = {};
    bool valid = false;
  };

  /// Fullscreen entry: graphics pipeline + its SSBO descriptor sets.
  /// Layout comes from FS reflection; one set per frame-in-flight slot,
  /// re-written by that slot's draw (the slot's prior frame has
  /// provably completed by then - a single shared set re-written every
  /// frame would race in-flight command buffers).
  struct Fullscreen {
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet sets[kMaxFramesInFlight] = {};
    bool valid = false;
  };

  /// Create the VkPipelineCache, loading prior data from disk.
  bool init(VkDevice device);
  /// Destroy every live pipeline, persist the cache, destroy it.
  void shutdown(VkDevice device);

  /// Translate + create. `colorFormat` is the render-target format the
  /// pipeline renders into (the swapchain's). Returns the handle or
  /// kInvalidPipelineHandle.
  u32 create(VkDevice device, VkFormat colorFormat,
             const IRenderBackend::GraphicsPipelineInfo& info);

  /// Fullscreen triangle + FS SSBO set (VisBuffer decode path). The
  /// handle indexes the SAME slot table as create().
  u32 createFullscreen(VkDevice device, VkFormat colorFormat,
                       const IRenderBackend::FullscreenPipelineInfo& info);
  /// M4.2 hardware raster path: NO attachments at all (the PS writes
  /// the VisBuffer SSBO atomically, UE's null-targets form - one output
  /// buffer, async-overlap friendly). Cull BACK / front CCW, matching
  /// the SW rasterizer's convention under the Y-flipped projection.
  u32 createNullRaster(VkDevice device,
                       const IRenderBackend::FullscreenPipelineInfo& info);
  /// Persistent compute pipeline (in-frame chain passes).
  u32 createCompute(VkDevice device,
                    const IRenderBackend::ComputePipelineInfo& info);
  const Compute& compute(u32 handle) const { return m_compute[handle - 1]; }
  u32 computeCount() const { return static_cast<u32>(m_compute.size()); }
  const Fullscreen& fullscreen(u32 handle) const {
    return m_fullscreen[handle - 1];
  }
  u32 fullscreenCount() const { return static_cast<u32>(m_fullscreen.size()); }
  const Fullscreen& nullRaster(u32 handle) const {
    return m_nullRaster[handle - 1];
  }
  u32 nullRasterCount() const { return static_cast<u32>(m_nullRaster.size()); }

  /// Zero the slot and return its raw handles (for a deletion-queue
  /// lambda that destroys them when the frame completes).
  Pipeline detach(u32 handle);

  bool valid(u32 handle) const;
  VkPipeline pipeline(u32 handle) const;
  VkPipelineLayout layout(u32 handle) const;

 private:
  std::vector<Pipeline> m_slots;
  std::vector<Fullscreen> m_fullscreen;
  std::vector<Fullscreen> m_nullRaster;
  std::vector<Compute> m_compute;
  VkPipelineCache m_cache = VK_NULL_HANDLE;
};

}  // namespace backend

KENSHIN_NS_END
