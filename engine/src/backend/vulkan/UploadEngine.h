#pragma once

// ── UploadEngine — host-to-device uploads ────────────────────────
// P0-1: the one-shot device-local upload (staging copy + queue submit +
// wait). P0-3 grows the per-frame staging ring and stagePerFrameData()
// here (Raptor dynamic_allocate shape).

#include <volk.h>

#include <vma/vk_mem_alloc.h>

#include <vector>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace backend {

/// One-shot upload: creates a device-local buffer with TRANSFER_DST usage,
/// copies through a mapped staging buffer, submits + waits. `pool` must be
/// a command pool whose buffers are not concurrently recorded.
bool uploadDeviceBuffer(VmaAllocator allocator, VkDevice device, VkQueue queue,
                        VkCommandPool pool, const void* data, VkDeviceSize bytes,
                        VkBufferUsageFlags usage, VkBuffer& outBuffer,
                        VmaAllocation& outAlloc);

/// Per-frame-slot staging ring (Raptor dynamic_allocate shape): one large
/// mapped host-visible buffer per frame slot; stage() hands out
/// (buffer, offset) slices written straight through the mapping. The used
/// offset resets when the slot's frame begins (fence-proven reuse), so the
/// previous frame's GPU reads are safe. M2's per-frame data (culling
/// inputs, instance tables, LOD params) flows through here; the renderer
/// layer receives slices with the FrameContext when M2 lands.
class StagingRing {
 public:
  bool init(VmaAllocator allocator, VkDeviceSize bytesPerSlot, u32 slotCount);
  void shutdown(VmaAllocator allocator);

  /// Start a new frame's slice region on the given slot.
  void beginFrame(u32 slotIndex);

  /// Stage `bytes` (aligned up to `alignment`). Returns false when the
  /// slot's region is full - callers fall back to uploadDeviceBuffer.
  bool stage(const void* data, VkDeviceSize bytes, VkDeviceSize alignment,
             VkBuffer& outBuffer, VkDeviceSize& outOffset);

 private:
  struct Slot {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize capacity = 0;
    VkDeviceSize used = 0;
  };
  std::vector<Slot> m_slots;
  u32 m_current = 0xFFFFFFFFu;
};

}  // namespace backend

KENSHIN_NS_END
