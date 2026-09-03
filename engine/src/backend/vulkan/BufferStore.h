#pragma once

// ── BufferStore — general device buffers behind u32 handles ──────
// The virtual-geometry currency: cluster tables / vertex pools / index
// pools / instance data are arbitrary-size device-local buffers, not the
// PNU mesh format. Creation may one-shot-upload initial data; destruction
// rides the frame slots' deletion queues (deferred) or destroyAll at
// shutdown; readbackBuffer serves the CRC verification paths.

#include <vma/vk_mem_alloc.h>
#include <volk.h>

#include <vector>

#include "kenshin/Common.h"
#include "kenshin/backend/RenderBackend.h"

KENSHIN_NS_BEGIN

namespace backend {

struct VulkanDevice;

class BufferStore {
 public:
  struct Entry {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    usize size = 0;
    void* mapped = nullptr;  // host-visible buffers stay mapped
  };

  /// Device-local buffer (+ optional one-shot initial upload). Returns the
  /// handle or kInvalidBufferHandle. `pool` must not be concurrently
  /// recorded (frame slot 0's is the norm).
  u32 create(const VulkanDevice& device, VkCommandPool pool,
             const IRenderBackend::BufferDesc& desc);

  bool valid(u32 handle) const {
    return handle != IRenderBackend::kInvalidBufferHandle && handle != 0 &&
           handle <= m_slots.size() &&
           m_slots[handle - 1].buffer != VK_NULL_HANDLE;
  }
  VkBuffer buffer(u32 handle) const { return m_slots[handle - 1].buffer; }
  usize size(u32 handle) const { return m_slots[handle - 1].size; }
  /// Persistently mapped host pointer (host-visible buffers; null for
  /// device-local ones). CPU writes before submit are visible to the GPU.
  void* mapped(u32 handle) const { return m_slots[handle - 1].mapped; }

  /// Zero the slot and return the raw entry (deletion-queue lambdas
  /// destroy it when the frame completes).
  Entry detach(u32 handle);
  static void destroyRaw(VmaAllocator allocator, const Entry& raw);
  void destroyAll(VmaAllocator allocator);

  /// Blocking staged readback of a byte range (queue submit + wait).
  static bool readback(const VulkanDevice& device, VkCommandPool pool,
                       const Entry& entry, usize offset, usize bytes,
                       std::vector<u8>& out);

 private:
  std::vector<Entry> m_slots;
};

}  // namespace backend

KENSHIN_NS_END
