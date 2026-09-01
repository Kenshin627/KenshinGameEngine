#include "UploadEngine.h"

#include "Vk.h"
#include "kenshin/core/Log.h"

#include <cstring>

KENSHIN_NS_BEGIN

namespace backend {

bool StagingRing::init(VmaAllocator allocator, VkDeviceSize bytesPerSlot,
                       u32 slotCount) {
  m_slots.resize(slotCount);
  for (Slot& slot : m_slots) {
    const VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytesPerSlot,
        // Broad usage so M2 can bind slices as SSBO/UBO or copy out.
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo allocInfo{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
        .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    };
    VmaAllocationInfo out{};
    if (vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &slot.buffer,
                        &slot.alloc, &out) != VK_SUCCESS) {
      KENSHIN_ERROR("[upload] staging ring allocation failed");
      shutdown(allocator);
      return false;
    }
    slot.mapped = out.pMappedData;
    slot.capacity = bytesPerSlot;
    slot.used = 0;
  }
  return true;
}

void StagingRing::shutdown(VmaAllocator allocator) {
  for (Slot& slot : m_slots) {
    if (slot.buffer != VK_NULL_HANDLE)
      vmaDestroyBuffer(allocator, slot.buffer, slot.alloc);
  }
  m_slots.clear();
  m_current = 0xFFFFFFFFu;
}

void StagingRing::beginFrame(u32 slotIndex) {
  m_current = slotIndex;
  if (slotIndex < m_slots.size()) m_slots[slotIndex].used = 0;
}

bool StagingRing::stage(const void* data, VkDeviceSize bytes,
                        VkDeviceSize alignment, VkBuffer& outBuffer,
                        VkDeviceSize& outOffset) {
  if (m_current >= m_slots.size()) return false;
  Slot& slot = m_slots[m_current];
  const VkDeviceSize aligned =
      (slot.used + alignment - 1) / alignment * alignment;
  if (aligned + bytes > slot.capacity) return false;
  std::memcpy(static_cast<u8*>(slot.mapped) + aligned, data,
              static_cast<usize>(bytes));
  slot.used = aligned + bytes;
  outBuffer = slot.buffer;
  outOffset = aligned;
  return true;
}

bool uploadDeviceBuffer(VmaAllocator allocator, VkDevice device, VkQueue queue,
                        VkCommandPool pool, const void* data, VkDeviceSize bytes,
                        VkBufferUsageFlags usage, VkBuffer& outBuffer,
                        VmaAllocation& outAlloc) {
  const VkBufferCreateInfo bufferInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes,
      .usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  const VmaAllocationCreateInfo allocInfo{
      .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
      .preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
  };
  if (vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &outBuffer, &outAlloc,
                      nullptr) != VK_SUCCESS)
    return false;

  const VkBufferCreateInfo stagingInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = bytes,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  const VmaAllocationCreateInfo stagingAllocInfo{
      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
               VMA_ALLOCATION_CREATE_MAPPED_BIT,
      .usage = VMA_MEMORY_USAGE_AUTO,
      .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
  };
  VkBuffer staging = VK_NULL_HANDLE;
  VmaAllocation stagingAlloc = VK_NULL_HANDLE;
  VmaAllocationInfo stagingInfoOut{};
  if (vmaCreateBuffer(allocator, &stagingInfo, &stagingAllocInfo, &staging,
                      &stagingAlloc, &stagingInfoOut) != VK_SUCCESS) {
    vmaDestroyBuffer(allocator, outBuffer, outAlloc);
    outBuffer = VK_NULL_HANDLE;
    outAlloc = VK_NULL_HANDLE;
    return false;
  }
  std::memcpy(stagingInfoOut.pMappedData, data, static_cast<usize>(bytes));

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  const VkCommandBufferAllocateInfo allocCmd{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };
  KE_VK(vkAllocateCommandBuffers(device, &allocCmd, &cmd));
  const VkCommandBufferBeginInfo beginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  KE_VK(vkBeginCommandBuffer(cmd, &beginInfo));
  const VkBufferCopy region{.size = bytes};
  vkCmdCopyBuffer(cmd, staging, outBuffer, 1, &region);
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
  KE_VK(vkQueueSubmit2(queue, 1, &submit, VK_NULL_HANDLE));
  KE_VK(vkQueueWaitIdle(queue));
  vkFreeCommandBuffers(device, pool, 1, &cmd);
  vmaDestroyBuffer(allocator, staging, stagingAlloc);
  return true;
}

}  // namespace backend

KENSHIN_NS_END
