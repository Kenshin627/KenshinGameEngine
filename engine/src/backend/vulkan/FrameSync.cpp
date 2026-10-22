#include "FrameSync.h"

#include "Vk.h"
#include "VulkanDevice.h"
#include "kenshin/core/Log.h"

KENSHIN_NS_BEGIN

namespace backend {

bool FrameSync::init(const VulkanDevice& device, bool timestamps) {
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

    if (timestamps) {
      const VkQueryPoolCreateInfo queryInfo{
          .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
          .queryType = VK_QUERY_TYPE_TIMESTAMP,
          .queryCount = 3,
      };
      if (!KE_VK(vkCreateQueryPool(device.device, &queryInfo, nullptr,
                                   &slot.timingQueries)))
        return false;
      const VkQueryPoolCreateInfo chainQueryInfo{
          .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
          .queryType = VK_QUERY_TYPE_TIMESTAMP,
          .queryCount = 7,
      };
      if (!KE_VK(vkCreateQueryPool(device.device, &chainQueryInfo, nullptr,
                                   &slot.chainTimingQueries)))
        return false;
    }
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
  return KE_VK(
      vkCreateSemaphore(device.device, &timelineInfo, nullptr, &frameTimeline));
}

bool FrameSync::initAsyncCompute(const VulkanDevice& device) {
  if (device.asyncComputeQueue == VK_NULL_HANDLE) return false;
  for (u32 i = 0; i < kMaxFramesInFlight; ++i) {
    FrameSlot& slot = slots[i];
    const VkCommandPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = device.asyncComputeFamily,
    };
    if (!KE_VK(vkCreateCommandPool(device.device, &poolInfo, nullptr,
                                   &slot.computePool)))
      return false;
    const VkCommandBufferAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = slot.computePool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (!KE_VK(vkAllocateCommandBuffers(device.device, &allocInfo,
                                        &slot.computeCullCb)))
      return false;
    if (!KE_VK(vkAllocateCommandBuffers(device.device, &allocInfo,
                                        &slot.computeRasterCb)))
      return false;
  }
  const VkSemaphoreTypeCreateInfo timelineType{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
      .initialValue = 0,
  };
  const VkSemaphoreCreateInfo timelineInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &timelineType,
  };
  if (!KE_VK(vkCreateSemaphore(device.device, &timelineInfo, nullptr,
                               &computeTimeline)))
    return false;
  asyncComputeActive = true;
  return true;
}

bool FrameSync::waitCompute(const VulkanDevice& device, u64 value) const {
  if (computeTimeline == VK_NULL_HANDLE || value == 0) return true;
  const VkSemaphoreWaitInfo wait{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .semaphoreCount = 1,
      .pSemaphores = &computeTimeline,
      .pValues = &value,
  };
  return KE_VK(vkWaitSemaphores(device.device, &wait, UINT64_MAX));
}

void FrameSync::shutdown(const VulkanDevice& device) {
  for (auto& r : retired) r.image.destroy(device.allocator, device.device);
  retired.clear();
  for (u32 i = 0; i < kMaxFramesInFlight; ++i) {
    FrameSlot& slot = slots[i];
    slot.offscreen.destroy(device.allocator, device.device);
    for (auto& g : slot.gbuffer) g.destroy(device.allocator, device.device);
    slot.hdr.destroy(device.allocator, device.device);
    if (slot.imageAvailable != VK_NULL_HANDLE) {
      vkDestroySemaphore(device.device, slot.imageAvailable, nullptr);
      slot.imageAvailable = VK_NULL_HANDLE;
    }
    if (slot.computePool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device.device, slot.computePool, nullptr);
      slot.computePool = VK_NULL_HANDLE;
      slot.computeCullCb = slot.computeRasterCb = VK_NULL_HANDLE;
    }
    if (slot.commandPool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device.device, slot.commandPool, nullptr);
      slot.commandPool = VK_NULL_HANDLE;
    }
    if (slot.timingQueries != VK_NULL_HANDLE) {
      vkDestroyQueryPool(device.device, slot.timingQueries, nullptr);
      slot.timingQueries = VK_NULL_HANDLE;
    }
    if (slot.chainTimingQueries != VK_NULL_HANDLE) {
      vkDestroyQueryPool(device.device, slot.chainTimingQueries, nullptr);
      slot.chainTimingQueries = VK_NULL_HANDLE;
    }
  }
  if (computeTimeline != VK_NULL_HANDLE) {
    vkDestroySemaphore(device.device, computeTimeline, nullptr);
    computeTimeline = VK_NULL_HANDLE;
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

void FrameSync::drainRetired(const VulkanDevice& device) {
  ++frameCounter;
  constexpr u64 kRetireDelay = 8;
  for (u64 k = 0; k < retired.size();) {
    if (frameCounter - retired[k].retireFrame >= kRetireDelay) {
      retired[k].image.destroy(device.allocator, device.device);
      retired.erase(retired.begin() + k);
    } else {
      ++k;
    }
  }
}

bool FrameSync::ensureOffscreens(const VulkanDevice& device, u32 width,
                                 u32 height, VkFormat format,
                                 VkImageUsageFlags usage) {
  // Does anything need (re)creation?
  bool anyStale = false;
  for (u32 i = 0; i < kMaxFramesInFlight; ++i)
    if (!slots[i].offscreen.matches(width, height, format)) anyStale = true;
  if (!anyStale) return true;

  // Old targets are RETIRED, not destroyed: in-flight frames may still
  // sample them, and destroying a recently-sampled image mid-loop wedges
  // the NVIDIA no-validation present path (see RetiredTarget in the
  // header). drainRetired() destroys them once aged past the delay.
  for (u32 i = 0; i < kMaxFramesInFlight; ++i) {
    FrameSlot& slot = slots[i];
    if (slot.offscreen.matches(width, height, format)) continue;
    retired.push_back({slot.offscreen, frameCounter});
    if (!slot.offscreen.create(device.allocator, device.device, width, height,
                               format, usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      KENSHIN_ERROR("[vulkan] offscreen target creation failed (slot {})", i);
      return false;
    }
  }
  return true;
}

bool FrameSync::ensureGBuffer(const VulkanDevice& device, u32 width,
                              u32 height) {
  // GBuffer B is sRGB-encoded: fsGBuffer writes LINEAR base color, and
  // an 8-bit UNORM store of linear albedo bands the darks (linear 0.02
  // sits 5 codes above black - the dark camo). The alpha channel (the
  // packed roughness/metallic bits) is linear in an _SRGB format.
  static const VkFormat kFormats[2] = {VK_FORMAT_R16G16B16A16_SFLOAT,
                                       VK_FORMAT_R8G8B8A8_SRGB};
  bool anyStale = false;
  for (u32 i = 0; i < kMaxFramesInFlight; ++i)
    for (u32 g = 0; g < 2; ++g)
      if (!slots[i].gbuffer[g].matches(width, height, kFormats[g]))
        anyStale = true;
  if (!anyStale) return true;
  // The images transition UNDEFINED -> COLOR_ATTACHMENT on the frame's
  // first GBuffer pass and COLOR_ATTACHMENT -> SHADER_READ_ONLY after
  // it; creation leaves them UNDEFINED (no initial barrier needed).
  const VkImageUsageFlags usage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  for (u32 i = 0; i < kMaxFramesInFlight; ++i) {
    FrameSlot& slot = slots[i];
    for (u32 g = 0; g < 2; ++g) {
      if (slot.gbuffer[g].matches(width, height, kFormats[g])) continue;
      retired.push_back({slot.gbuffer[g], frameCounter});
      if (!slot.gbuffer[g].create(device.allocator, device.device, width,
                                  height, kFormats[g], usage,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
        KENSHIN_ERROR("[vulkan] gbuffer target creation failed (slot {})", i);
        return false;
      }
    }
  }
  return true;
}

bool FrameSync::ensureHdr(const VulkanDevice& device, u32 width, u32 height) {
  static const VkFormat kFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
  bool anyStale = false;
  for (u32 i = 0; i < kMaxFramesInFlight; ++i)
    if (!slots[i].hdr.matches(width, height, kFormat)) anyStale = true;
  if (!anyStale) return true;
  // Same lifecycle as the GBuffer pair: UNDEFINED -> COLOR_ATTACHMENT
  // (the lighting pass) -> SHADER_READ_ONLY (the tonemap sample).
  const VkImageUsageFlags usage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  for (u32 i = 0; i < kMaxFramesInFlight; ++i) {
    FrameSlot& slot = slots[i];
    if (slot.hdr.matches(width, height, kFormat)) continue;
    retired.push_back({slot.hdr, frameCounter});
    if (!slot.hdr.create(device.allocator, device.device, width, height,
                         kFormat, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      KENSHIN_ERROR("[vulkan] hdr target creation failed (slot {})", i);
      return false;
    }
  }
  return true;
}

}  // namespace backend

KENSHIN_NS_END
