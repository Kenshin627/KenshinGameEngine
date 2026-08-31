#pragma once

// ── Per-frame resources (internal to the backend) ───────────────
// kMaxFramesInFlight slots, each owning: command pool+buffer, acquire/present
// semaphores, in-flight fence, and its own offscreen color target (so two
// in-flight frames never touch the same image).

#include "GpuImage.h"
#include "kenshin/Common.h"

#include <volk.h>

KENSHIN_NS_BEGIN

namespace backend
{

    struct VulkanDevice;

    constexpr u32 kMaxFramesInFlight = 2;

    struct FrameSlot
    {
        VkCommandPool commandPool = VK_NULL_HANDLE;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkSemaphore renderFinished = VK_NULL_HANDLE;
        VkFence inFlight = VK_NULL_HANDLE;
        GpuImage offscreen;

        bool initialized() const { return commandPool != VK_NULL_HANDLE; }
    };

    struct FrameSync
    {
        FrameSlot slots[kMaxFramesInFlight];
        u32 frameIndex = 0;   // rotating slot
        u32 lastCompleted = 0xFFFFFFFFu;
        bool anyCompleted = false;

        bool init(const VulkanDevice& device);
        void shutdown(const VulkanDevice& device);

        /// (Re)create offscreen targets for the given size/format when they
        /// do not match. Destroys old ones first.
        bool ensureOffscreens(const VulkanDevice& device, u32 width, u32 height,
                              VkFormat format, VkImageUsageFlags usage);

        void advance() { frameIndex = (frameIndex + 1) % kMaxFramesInFlight; }
    };

} // namespace backend

KENSHIN_NS_END
