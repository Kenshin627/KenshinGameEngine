#pragma once

// ── UploadEngine — host-to-device uploads ────────────────────────
// P0-1: the one-shot device-local upload (staging copy + queue submit +
// wait). P0-3 grows the per-frame staging ring and stagePerFrameData()
// here (Raptor dynamic_allocate shape).

#include <volk.h>

#include <vma/vk_mem_alloc.h>

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

}  // namespace backend

KENSHIN_NS_END
