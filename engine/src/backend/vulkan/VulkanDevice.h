#pragma once

// ── Vulkan logical device (internal to the backend) ─────────────
// Owns: VkDevice, queues, VMA allocator.
// This header is internal to src/backend/vulkan; VMA types do not
// leak past the backend boundary.

#include <volk.h>

#include "kenshin/Common.h"

#define VMA_VULKAN_VERSION 1004000
#include <vma/vk_mem_alloc.h>

KENSHIN_NS_BEGIN

namespace backend {

struct VulkanDevice {
  VkDevice device = VK_NULL_HANDLE;
  u32 graphicsFamily = 0xFFFFFFFFu;
  u32 computeFamily = 0xFFFFFFFFu;
  u32 transferFamily = 0xFFFFFFFFu;
  VkQueue graphicsQueue = VK_NULL_HANDLE;
  VkQueue computeQueue = VK_NULL_HANDLE;
  VkQueue transferQueue =
      VK_NULL_HANDLE;  // aliases graphicsQueue when same family
  VmaAllocator allocator = VK_NULL_HANDLE;

  // Features actually enabled at device creation
  bool enabledBufferInt64Atomics = false;
  bool enabledTimelineSemaphore = false;
  bool enabledSynchronization2 = false;
  bool enabledDynamicRendering = false;

  /// @param context  initialized VulkanContext (physical device + families)
  bool init(const struct VulkanContext& context);
  void shutdown();
};

}  // namespace backend

KENSHIN_NS_END
