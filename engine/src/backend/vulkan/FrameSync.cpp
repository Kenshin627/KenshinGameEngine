#include "FrameSync.h"

#include "VulkanDevice.h"
#include "kenshin/core/Log.h"
#include "Vk.h"

KENSHIN_NS_BEGIN

namespace backend {

bool FrameSync::init(const VulkanDevice& device) {
  for (u32 i = 0; i < kMaxFramesInFlight; ++i) {
    FrameSlot& slot = slots[i];

    const VkCommandPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = device.graphicsFamily,
    };
    if (!KE_VK(vkCreateCommandPool(device.device, &poolInfo, nullptr,
                            &slot.commandPool)))
      return false;

    const VkCommandBufferAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = slot.commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (!KE_VK(vkAllocateCommandBuffers(device.device, &allocInfo,
                                 &slot.commandBuffer)))
      return false;

    const VkSemaphoreCreateInfo semInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    if (!KE_VK(vkCreateSemaphore(device.device, &semInfo, nullptr,
                          &slot.imageAvailable)))
      return false;
  }

  // The single device timeline: submit #k signals value k (see FrameSync).
  const VkSemaphoreTypeCreateInfo timelineType{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
      .initialValue = 0,
  };
  const VkSemaphoreCreateInfo timelineInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &timelineType,
  };
  return KE_VK(vkCreateSemaphore(device.device, &timelineInfo, nullptr,
                                 &frameTimeline));
}

void FrameSync::shutdown(const VulkanDevice& device) {
  for (u32 i = 0; i < kMaxFramesInFlight; ++i) {
    FrameSlot& slot = slots[i];
    slot.offscreen.destroy(device.allocator, device.device);
    if (slot.imageAvailable != VK_NULL_HANDLE) {
      vkDestroySemaphore(device.device, slot.imageAvailable, nullptr);
      slot.imageAvailable = VK_NULL_HANDLE;
    }
    if (slot.commandPool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device.device, slot.commandPool, nullptr);
      slot.commandPool = VK_NULL_HANDLE;
    }
  }
  if (frameTimeline != VK_NULL_HANDLE) {
    vkDestroySemaphore(device.device, frameTimeline, nullptr);
    frameTimeline = VK_NULL_HANDLE;
  }
}

bool FrameSync::waitTimeline(const VulkanDevice& device, u64 value) const {
  const VkSemaphoreWaitInfo wait{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .semaphoreCount = 1,
      .pSemaphores = &frameTimeline,
      .pValues = &value,
  };
  return KE_VK(vkWaitSemaphores(device.device, &wait, UINT64_MAX));
}

bool FrameSync::ensureOffscreens(const VulkanDevice& device, u32 width,
                                 u32 height, VkFormat format,
                                 VkImageUsageFlags usage) {
  // Does anything need (re)creation?
  bool anyStale = false;
  for (u32 i = 0; i < kMaxFramesInFlight; ++i)
    if (!slots[i].offscreen.matches(width, height, format)) anyStale = true;
  if (!anyStale) return true;

  // The other in-flight slot's command buffer may still reference the
  // old targets; a resize is rare, so a full idle wait is fine here.
  KE_VK(vkDeviceWaitIdle(device.device));

  for (u32 i = 0; i < kMaxFramesInFlight; ++i) {
    FrameSlot& slot = slots[i];
    if (slot.offscreen.matches(width, height, format)) continue;
    slot.offscreen.destroy(device.allocator, device.device);
    if (!slot.offscreen.create(device.allocator, device.device, width, height,
                               format, usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      KENSHIN_ERROR("[vulkan] offscreen target creation failed (slot {})", i);
      return false;
    }
  }
  return true;
}

}  // namespace backend

KENSHIN_NS_END
