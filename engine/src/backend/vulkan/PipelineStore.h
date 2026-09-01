#pragma once

// ── PipelineStore — graphics pipeline table + disk cache ────────
// Owns the pipeline slots (handle = index + 1, freed slots keep their
// place with null handles) and the VkPipelineCache persisted next to
// the exe. The renderer layer defines state; this store translates it
// (the enum helpers live in the .cpp).

#include <volk.h>

#include <vector>

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

  /// Create the VkPipelineCache, loading prior data from disk.
  bool init(VkDevice device);
  /// Destroy every live pipeline, persist the cache, destroy it.
  void shutdown(VkDevice device);

  /// Translate + create. `colorFormat` is the render-target format the
  /// pipeline renders into (the swapchain's). Returns the handle or
  /// kInvalidPipelineHandle.
  u32 create(VkDevice device, VkFormat colorFormat,
             const IRenderBackend::GraphicsPipelineInfo& info);

  /// Zero the slot and return its raw handles (for a deletion-queue
  /// lambda that destroys them when the frame completes).
  Pipeline detach(u32 handle);

  bool valid(u32 handle) const;
  VkPipeline pipeline(u32 handle) const;
  VkPipelineLayout layout(u32 handle) const;

 private:
  std::vector<Pipeline> m_slots;
  VkPipelineCache m_cache = VK_NULL_HANDLE;
};

}  // namespace backend

KENSHIN_NS_END
