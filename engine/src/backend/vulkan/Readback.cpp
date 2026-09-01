#include "Readback.h"

#include "CommandContext.h"
#include "FrameSync.h"
#include "Vk.h"
#include "VulkanDevice.h"

#include <cstring>

KENSHIN_NS_BEGIN

namespace backend {

bool Readback::readViewport(const VulkanDevice& device,
                            const FrameSync& frameSync, VkFormat swapchainFormat,
                            std::vector<u8>& rgbaOut, u32& width, u32& height) {
  if (!frameSync.anyCompleted) return false;

  const FrameSlot& slot = frameSync.slots[frameSync.lastCompleted];
  width = slot.offscreen.width;
  height = slot.offscreen.height;

  frameSync.waitTimeline(device, frameSync.completedValue);

  // Staging buffer (host visible) for the copy.
  const VkBufferCreateInfo bufferInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = static_cast<VkDeviceSize>(width) * height * 4,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  const VmaAllocationCreateInfo stagingAlloc{
      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
      .usage = VMA_MEMORY_USAGE_AUTO,
      .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
  };
  VkBuffer staging = VK_NULL_HANDLE;
  VmaAllocation stagingAllocHandle = VK_NULL_HANDLE;
  void* mapped = nullptr;
  if (vmaCreateBuffer(device.allocator, &bufferInfo, &stagingAlloc,
                      &staging, &stagingAllocHandle, nullptr) != VK_SUCCESS)
    return false;
  vmaMapMemory(device.allocator, stagingAllocHandle, &mapped);

  // One-shot copy on the graphics queue (slot 0's pool; drained by the fence wait).
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  const VkCommandBufferAllocateInfo allocInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = frameSync.slots[0].commandPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };
  KE_VK(vkAllocateCommandBuffers(device.device, &allocInfo, &cmd));
  const VkCommandBufferBeginInfo beginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  KE_VK(vkBeginCommandBuffer(cmd, &beginInfo));
  CommandContext::pipelineBarrier(cmd, CommandContext::imageBarrier(slot.offscreen.image,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                  VK_ACCESS_2_TRANSFER_READ_BIT));
  const VkBufferImageCopy region{
      .bufferOffset = 0,
      .bufferRowLength = 0,  // tightly packed
      .bufferImageHeight = 0,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageOffset = {0, 0, 0},
      .imageExtent = {width, height, 1},
  };
  vkCmdCopyImageToBuffer(cmd, slot.offscreen.image,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1,
                         &region);
  KE_VK(vkEndCommandBuffer(cmd));
  const VkCommandBufferSubmitInfo cmdSubmit{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = cmd,
  };
  const VkSubmitInfo2 submit{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &cmdSubmit,
  };
  KE_VK(vkQueueSubmit2(device.graphicsQueue, 1, &submit, VK_NULL_HANDLE));
  KE_VK(vkQueueWaitIdle(device.graphicsQueue));
  vkFreeCommandBuffers(device.device, frameSync.slots[0].commandPool,
                       1, &cmd);

  // Swizzle BGRA -> RGBA when the surface format is BGRA-ordered.
  rgbaOut.resize(static_cast<usize>(width) * height * 4);
  const bool bgra = swapchainFormat == VK_FORMAT_B8G8R8A8_UNORM;
  const u8* src = static_cast<const u8*>(mapped);
  if (bgra) {
    for (usize i = 0; i < rgbaOut.size(); i += 4) {
      rgbaOut[i + 0] = src[i + 2];
      rgbaOut[i + 1] = src[i + 1];
      rgbaOut[i + 2] = src[i + 0];
      rgbaOut[i + 3] = src[i + 3];
    }
  } else
    std::memcpy(rgbaOut.data(), src, rgbaOut.size());

  vmaUnmapMemory(device.allocator, stagingAllocHandle);
  vmaDestroyBuffer(device.allocator, staging, stagingAllocHandle);
  return true;
}

bool Readback::readFullFrame(const VulkanDevice& device,
                             const FrameSync& frameSync, VkFormat swapchainFormat,
                             std::vector<u8>& rgbaOut, u32& width, u32& height) {
  if (!m_fullCaptured || m_captureStaging == VK_NULL_HANDLE) return false;
  frameSync.waitTimeline(device, frameSync.completedValue);

  width = m_captureWidth;
  height = m_captureHeight;
  void* mapped = nullptr;
  if (vmaMapMemory(device.allocator, m_captureAlloc, &mapped) !=
      VK_SUCCESS)
    return false;

  rgbaOut.resize(static_cast<usize>(width) * height * 4);
  const u8* src = static_cast<const u8*>(mapped);
  const bool bgra = swapchainFormat == VK_FORMAT_B8G8R8A8_UNORM;
  if (bgra) {
    for (usize i = 0; i < rgbaOut.size(); i += 4) {
      rgbaOut[i + 0] = src[i + 2];
      rgbaOut[i + 1] = src[i + 1];
      rgbaOut[i + 2] = src[i + 0];
      rgbaOut[i + 3] = src[i + 3];
    }
  } else
    std::memcpy(rgbaOut.data(), src, rgbaOut.size());
  vmaUnmapMemory(device.allocator, m_captureAlloc);
  m_fullCaptured = false;
  return true;
}

void Readback::ensureCaptureStaging(VmaAllocator allocator, u32 width,
                                    u32 height) {
  if (m_captureStaging != VK_NULL_HANDLE && m_captureWidth == width &&
      m_captureHeight == height)
    return;
  if (m_captureStaging != VK_NULL_HANDLE)
    vmaDestroyBuffer(allocator, m_captureStaging, m_captureAlloc);
  const VkBufferCreateInfo bufferInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = static_cast<VkDeviceSize>(width) * height * 4,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  const VmaAllocationCreateInfo allocInfo{
      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
      .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
      .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
  };
  vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &m_captureStaging,
                  &m_captureAlloc, nullptr);
  m_captureWidth = width;
  m_captureHeight = height;
}

void Readback::destroy(VmaAllocator allocator) {
  if (m_captureStaging != VK_NULL_HANDLE)
    vmaDestroyBuffer(allocator, m_captureStaging, m_captureAlloc);
  m_captureStaging = VK_NULL_HANDLE;
  m_captureAlloc = VK_NULL_HANDLE;
}

}  // namespace backend

KENSHIN_NS_END
