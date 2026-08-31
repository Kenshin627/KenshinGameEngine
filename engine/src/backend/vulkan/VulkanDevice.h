#pragma once

// ── Vulkan logical device (internal to the backend) ─────────────
// Owns: VkDevice, queues, VMA allocator.
// This header is internal to src/backend/vulkan; VMA types do not
// leak past the backend boundary.

#include "kenshin/Common.h"

#include <volk.h>

#define VMA_VULKAN_VERSION 1004000
#include <vma/vk_mem_alloc.h>

KENSHIN_NS_BEGIN

namespace backend
{

    struct VulkanDevice
    {
        VkDevice device = VK_NULL_HANDLE;
        VkQueue graphicsQueue = VK_NULL_HANDLE;
        VkQueue computeQueue = VK_NULL_HANDLE;
        VkQueue transferQueue = VK_NULL_HANDLE; // aliases graphicsQueue when same family
        VmaAllocator allocator = VK_NULL_HANDLE;

        // Features actually enabled at device creation
        bool enabledBufferInt64Atomics = false;
        bool enabledTimelineSemaphore  = false;

        /// @param context  initialized VulkanContext (physical device + families)
        bool init(const struct VulkanContext& context);
        void shutdown();
    };

} // namespace backend

KENSHIN_NS_END
