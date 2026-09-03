#include "BufferStore.h"

#include "UploadEngine.h"
#include "Vk.h"
#include "VulkanDevice.h"
#include "kenshin/core/Log.h"

KENSHIN_NS_BEGIN

namespace backend {

u32 BufferStore::create(const VulkanDevice& device, VkCommandPool pool,
                        const IRenderBackend::BufferDesc& desc) {
  if (desc.size == 0) return IRenderBackend::kInvalidBufferHandle;

  VkBufferUsageFlags usage{};
  if (desc.usage & IRenderBackend::BufferStorage)
    usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (desc.usage & IRenderBackend::BufferUniform)
    usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  if (desc.usage & IRenderBackend::BufferTransferSrc)
    usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  if (desc.initialData) usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  const VkBufferCreateInfo bufferInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = static_cast<VkDeviceSize>(desc.size),
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  // Host-visible buffers (in-frame chain params, updated by CPU writes
  // between submits) stay persistently mapped; everything else is
  // device-local (AUTO_PREFER_DEVICE + one-shot initial upload).
  const bool hostVisible =
      (desc.usage & IRenderBackend::BufferHostVisible) != 0;
  Entry entry;
  if (hostVisible) {
    const VmaAllocationCreateInfo hostAllocInfo{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
        .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    };
    VmaAllocationInfo out{};
    if (vmaCreateBuffer(device.allocator, &bufferInfo, &hostAllocInfo,
                        &entry.buffer, &entry.alloc, &out) != VK_SUCCESS ||
        out.pMappedData == nullptr) {
      KENSHIN_ERROR("[buffer] host-visible creation failed ({} bytes)",
                    desc.size);
      return IRenderBackend::kInvalidBufferHandle;
    }
    entry.mapped = out.pMappedData;
    if (desc.initialData)
      std::memcpy(entry.mapped, desc.initialData, desc.size);
  } else {
    const VmaAllocationCreateInfo allocInfo{
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    };
    // uploadDeviceBuffer is a create+upload helper (it allocates the device
    // buffer itself into the out params) - calling vmaCreateBuffer first
    // here would orphan that buffer. No-initial-data path allocates
    // directly.
    if (desc.initialData) {
      if (!uploadDeviceBuffer(device.allocator, device.device,
                              device.graphicsQueue, pool, desc.initialData,
                              static_cast<VkDeviceSize>(desc.size), usage,
                              entry.buffer, entry.alloc)) {
        KENSHIN_ERROR("[buffer] initial upload failed ({} bytes)", desc.size);
        return IRenderBackend::kInvalidBufferHandle;
      }
    } else if (vmaCreateBuffer(device.allocator, &bufferInfo, &allocInfo,
                               &entry.buffer, &entry.alloc,
                               nullptr) != VK_SUCCESS) {
      KENSHIN_ERROR("[buffer] creation failed ({} bytes)", desc.size);
      return IRenderBackend::kInvalidBufferHandle;
    }
  }
  entry.size = desc.size;
  if (desc.debugName)
    vkutil::setName(device.device, VK_OBJECT_TYPE_BUFFER,
                    reinterpret_cast<u64>(entry.buffer), desc.debugName);

  m_slots.push_back(entry);
  return static_cast<u32>(m_slots.size());
}

BufferStore::Entry BufferStore::detach(u32 handle) {
  if (!valid(handle)) return {};
  Entry raw = m_slots[handle - 1];
  m_slots[handle - 1] = {};
  return raw;
}

void BufferStore::destroyRaw(VmaAllocator allocator, const Entry& raw) {
  if (raw.buffer != VK_NULL_HANDLE)
    vmaDestroyBuffer(allocator, raw.buffer, raw.alloc);
}

void BufferStore::destroyAll(VmaAllocator allocator) {
  for (const Entry& e : m_slots)
    if (e.buffer != VK_NULL_HANDLE) destroyRaw(allocator, e);
  m_slots.clear();
}

bool BufferStore::readback(const VulkanDevice& device, VkCommandPool pool,
                           const Entry& entry, usize offset, usize bytes,
                           std::vector<u8>& out) {
  if (entry.buffer == VK_NULL_HANDLE || offset + bytes > entry.size)
    return false;

  // Host-visible staging.
  const VkBufferCreateInfo stagingInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = static_cast<VkDeviceSize>(bytes),
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
  if (vmaCreateBuffer(device.allocator, &stagingInfo, &stagingAlloc, &staging,
                      &stagingAllocHandle, nullptr) != VK_SUCCESS)
    return false;
  vmaMapMemory(device.allocator, stagingAllocHandle, &mapped);

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  const VkCommandBufferAllocateInfo allocInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };
  KE_VK(vkAllocateCommandBuffers(device.device, &allocInfo, &cmd));
  const VkCommandBufferBeginInfo beginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  KE_VK(vkBeginCommandBuffer(cmd, &beginInfo));
  const VkBufferCopy region{.srcOffset = static_cast<VkDeviceSize>(offset),
                            .dstOffset = 0,
                            .size = static_cast<VkDeviceSize>(bytes)};
  vkCmdCopyBuffer(cmd, entry.buffer, staging, 1, &region);
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
  vkFreeCommandBuffers(device.device, pool, 1, &cmd);

  out.resize(bytes);
  std::memcpy(out.data(), mapped, bytes);
  vmaUnmapMemory(device.allocator, stagingAllocHandle);
  vmaDestroyBuffer(device.allocator, staging, stagingAllocHandle);
  return true;
}

}  // namespace backend

KENSHIN_NS_END
