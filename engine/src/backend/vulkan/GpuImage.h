#pragma once

// ── GPU image RAII (internal to the backend) ────────────────────
// Image + VMA allocation + optional view. Internal to src/backend/vulkan;
// Vulkan types never leak past the backend boundary.

#include <volk.h>

#include "kenshin/Common.h"

#define VMA_VULKAN_VERSION 1004000
#include <vma/vk_mem_alloc.h>

KENSHIN_NS_BEGIN

namespace backend {

struct GpuImage {
  VkImage image = VK_NULL_HANDLE;
  VmaAllocation allocation = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  u32 width = 0;
  u32 height = 0;
  VkFormat format = VK_FORMAT_UNDEFINED;

  bool create(VmaAllocator allocator, VkDevice device, u32 w, u32 h,
              VkFormat fmt, VkImageUsageFlags usage,
              VkMemoryPropertyFlags memoryFlags);
  void destroy(VmaAllocator allocator, VkDevice device);

  /// True when it can serve a request of the given size/format.
  bool matches(u32 w, u32 h, VkFormat fmt) const {
    return image != VK_NULL_HANDLE && width == w && height == h &&
           format == fmt;
  }
};

}  // namespace backend

KENSHIN_NS_END
