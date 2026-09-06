#include "FrameSync.h"

#include "VulkanDevice.h"
#include "kenshin/core/Log.h"

KENSHIN_NS_BEGIN

namespace backend
{

    bool FrameSync::init(const VulkanDevice& device)
    {
        for (u32 i = 0; i < kMaxFramesInFlight; ++i)
        {
            FrameSlot& slot = slots[i];

            const VkCommandPoolCreateInfo poolInfo {
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                .queueFamilyIndex = device.graphicsFamily,
            };
            if (vkCreateCommandPool(device.device, &poolInfo, nullptr, &slot.commandPool) != VK_SUCCESS)
                return false;

            const VkCommandBufferAllocateInfo allocInfo {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool = slot.commandPool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1,
            };
            if (vkAllocateCommandBuffers(device.device, &allocInfo, &slot.commandBuffer) != VK_SUCCESS)
                return false;

            const VkSemaphoreCreateInfo semInfo {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            };
            if (vkCreateSemaphore(device.device, &semInfo, nullptr, &slot.imageAvailable) != VK_SUCCESS)
                return false;

            const VkFenceCreateInfo fenceInfo {
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .flags = VK_FENCE_CREATE_SIGNALED_BIT, // first wait is a no-op
            };
            if (vkCreateFence(device.device, &fenceInfo, nullptr, &slot.inFlight) != VK_SUCCESS)
                return false;
        }
        return true;
    }

    void FrameSync::shutdown(const VulkanDevice& device)
    {
        for (u32 i = 0; i < kMaxFramesInFlight; ++i)
        {
            FrameSlot& slot = slots[i];
            slot.offscreen.destroy(device.allocator, device.device);
            if (slot.inFlight != VK_NULL_HANDLE)
            {
                vkDestroyFence(device.device, slot.inFlight, nullptr);
                slot.inFlight = VK_NULL_HANDLE;
            }
            if (slot.imageAvailable != VK_NULL_HANDLE)
            {
                vkDestroySemaphore(device.device, slot.imageAvailable, nullptr);
                slot.imageAvailable = VK_NULL_HANDLE;
            }
            if (slot.commandPool != VK_NULL_HANDLE)
            {
                vkDestroyCommandPool(device.device, slot.commandPool, nullptr);
                slot.commandPool = VK_NULL_HANDLE;
            }
        }
    }

    bool FrameSync::ensureOffscreens(const VulkanDevice& device, u32 width, u32 height,
                                     VkFormat format, VkImageUsageFlags usage)
    {
        // Does anything need (re)creation?
        bool anyStale = false;
        for (u32 i = 0; i < kMaxFramesInFlight; ++i)
            if (!slots[i].offscreen.matches(width, height, format))
                anyStale = true;
        if (!anyStale)
            return true;
        if (std::getenv("KENSHIN_PROBE_NORESIZE") &&  // PROBE: keep the first-created offscreen forever
            slots[0].offscreen.image != VK_NULL_HANDLE)
            return true;

        // The other in-flight slot's command buffer may still reference the
        // old targets; a resize is rare, so a full idle wait is fine here.
        vkDeviceWaitIdle(device.device);

        for (u32 i = 0; i < kMaxFramesInFlight; ++i)
        {
            FrameSlot& slot = slots[i];
            if (slot.offscreen.matches(width, height, format))
                continue;
            // PROBE leak: old offscreen intentionally kept alive (no destroy)
            if (!slot.offscreen.create(device.allocator, device.device, width, height,
                                       format, usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            {
                KENSHIN_ERROR("[vulkan] offscreen target creation failed (slot {})", i);
                return false;
            }
        }
        return true;
    }

} // namespace backend

KENSHIN_NS_END
