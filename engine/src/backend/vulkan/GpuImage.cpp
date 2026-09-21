#include "GpuImage.h"

#include "Vk.h"

KENSHIN_NS_BEGIN

namespace backend {

bool GpuImage::create(VmaAllocator allocator, VkDevice device, u32 w, u32 h,
                      VkFormat fmt, VkImageUsageFlags usage,
                      VkMemoryPropertyFlags memoryFlags, u32 mips, u32 layers,
                      bool arrayedView) {
  const VkImageCreateInfo imageInfo{
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = fmt,
      .extent = {w, h, 1},
      .mipLevels = mips,
      .arrayLayers = layers,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };
  const VmaAllocationCreateInfo allocInfo{
      .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
      .requiredFlags = memoryFlags,
  };
  if (vmaCreateImage(allocator, &imageInfo, &allocInfo, &image, &allocation,
                     nullptr) != VK_SUCCESS)
    return false;

  const VkImageViewCreateInfo viewInfo{
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
      // arrayedView: a 1-layer image viewed as a Texture2DArray (the
      // shader's declared view type - a plain 2D view on the array
      // slot is a spec violation that happens to work on this driver).
      .viewType = (layers > 1 || arrayedView) ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                                              : VK_IMAGE_VIEW_TYPE_2D,
      .format = fmt,
      .subresourceRange =
          {
              .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
              .levelCount = mips,
              .layerCount = layers,
          },
  };
  if (!KE_VK(vkCreateImageView(device, &viewInfo, nullptr, &view))) {
    vmaDestroyImage(allocator, image, allocation);
    image = VK_NULL_HANDLE;
    allocation = VK_NULL_HANDLE;
    return false;
  }

  width = w;
  height = h;
  mipLevels = mips;
  arrayLayers = layers;
  format = fmt;
  return true;
}

void GpuImage::destroy(VmaAllocator allocator, VkDevice device) {
  if (view != VK_NULL_HANDLE) {
    vkDestroyImageView(device, view, nullptr);
    view = VK_NULL_HANDLE;
  }
  if (image != VK_NULL_HANDLE) {
    vmaDestroyImage(allocator, image, allocation);
    image = VK_NULL_HANDLE;
    allocation = VK_NULL_HANDLE;
  }
  width = height = 0;
  mipLevels = arrayLayers = 1;
  format = VK_FORMAT_UNDEFINED;
}

}  // namespace backend

KENSHIN_NS_END
