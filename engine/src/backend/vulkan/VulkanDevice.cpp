#include "VulkanDevice.h"

#include "VulkanContext.h"
#include "kenshin/core/Log.h"

#pragma warning(push, 0)
#define VMA_IMPLEMENTATION
#include <vma/vk_mem_alloc.h>
#pragma warning(pop)

#include <vector>

KENSHIN_NS_BEGIN

namespace backend {

namespace {
/// Fill VMA's function table explicitly from volk's loaded pointers.
///
/// Why not rely on VMA's default static import: in this vcpkg VMA 3.3 +
/// volk combination, ImportVulkanFunctions_Static ended up calling a null
/// pointer during vmaCreateAllocator (observed: execute violation at 0x0).
/// Passing an explicit table sidesteps the interop path and documents the
/// dependency on volkLoadInstance()/volkLoadDevice() having run first.
void fillVmaVulkanFunctions(VmaVulkanFunctions& fns) {
  fns = {};
  fns.vkGetPhysicalDeviceProperties =
      (PFN_vkGetPhysicalDeviceProperties)vkGetPhysicalDeviceProperties;
  fns.vkGetPhysicalDeviceMemoryProperties =
      (PFN_vkGetPhysicalDeviceMemoryProperties)
          vkGetPhysicalDeviceMemoryProperties;
  fns.vkAllocateMemory = (PFN_vkAllocateMemory)vkAllocateMemory;
  fns.vkFreeMemory = (PFN_vkFreeMemory)vkFreeMemory;
  fns.vkMapMemory = (PFN_vkMapMemory)vkMapMemory;
  fns.vkUnmapMemory = (PFN_vkUnmapMemory)vkUnmapMemory;
  fns.vkFlushMappedMemoryRanges =
      (PFN_vkFlushMappedMemoryRanges)vkFlushMappedMemoryRanges;
  fns.vkInvalidateMappedMemoryRanges =
      (PFN_vkInvalidateMappedMemoryRanges)vkInvalidateMappedMemoryRanges;
  fns.vkBindBufferMemory = (PFN_vkBindBufferMemory)vkBindBufferMemory;
  fns.vkBindImageMemory = (PFN_vkBindImageMemory)vkBindImageMemory;
  fns.vkGetBufferMemoryRequirements =
      (PFN_vkGetBufferMemoryRequirements)vkGetBufferMemoryRequirements;
  fns.vkGetImageMemoryRequirements =
      (PFN_vkGetImageMemoryRequirements)vkGetImageMemoryRequirements;
  fns.vkCreateBuffer = (PFN_vkCreateBuffer)vkCreateBuffer;
  fns.vkDestroyBuffer = (PFN_vkDestroyBuffer)vkDestroyBuffer;
  fns.vkCreateImage = (PFN_vkCreateImage)vkCreateImage;
  fns.vkDestroyImage = (PFN_vkDestroyImage)vkDestroyImage;
  fns.vkCmdCopyBuffer = (PFN_vkCmdCopyBuffer)vkCmdCopyBuffer;
  fns.vkGetBufferMemoryRequirements2KHR =
      (PFN_vkGetBufferMemoryRequirements2)vkGetBufferMemoryRequirements2;
  fns.vkGetImageMemoryRequirements2KHR =
      (PFN_vkGetImageMemoryRequirements2)vkGetImageMemoryRequirements2;
  fns.vkBindBufferMemory2KHR = (PFN_vkBindBufferMemory2)vkBindBufferMemory2;
  fns.vkBindImageMemory2KHR = (PFN_vkBindImageMemory2)vkBindImageMemory2;
  fns.vkGetPhysicalDeviceMemoryProperties2KHR =
      (PFN_vkGetPhysicalDeviceMemoryProperties2)
          vkGetPhysicalDeviceMemoryProperties2;
  fns.vkGetDeviceBufferMemoryRequirements =
      (PFN_vkGetDeviceBufferMemoryRequirements)
          vkGetDeviceBufferMemoryRequirements;
  fns.vkGetDeviceImageMemoryRequirements =
      (PFN_vkGetDeviceImageMemoryRequirements)
          vkGetDeviceImageMemoryRequirements;
}
}  // namespace

bool VulkanDevice::init(const VulkanContext& context) {
  // Enable only what the physical device reports as supported.
  enabledBufferInt64Atomics = context.supportsBufferInt64Atomics;
  enabledTimelineSemaphore = context.supportsTimelineSemaphore;
  enabledSynchronization2 = context.supportsSynchronization2;
  enabledDynamicRendering = context.supportsDynamicRendering;

  // Chain: features2 (core) <- 1.2 <- 1.3 (declare tail-first; the
  // device create info points at features2). Core Int64 rides with its
  // atomics sibling: the S4.x shaders use uint64 VisBuffer pixels
  // (SPIR-V CapabilityInt64) even where no atomic touches them - the
  // validation layer rejects the module without the core feature, while
  // Int64Atomics alone covers only the atomic ops.
  VkPhysicalDeviceVulkan13Features features13{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .synchronization2 = enabledSynchronization2 ? VK_TRUE : VK_FALSE,
      .dynamicRendering = enabledDynamicRendering ? VK_TRUE : VK_FALSE,
  };
  VkPhysicalDeviceVulkan12Features features12{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &features13,
      .shaderBufferInt64Atomics =
          enabledBufferInt64Atomics ? VK_TRUE : VK_FALSE,
      .timelineSemaphore = enabledTimelineSemaphore ? VK_TRUE : VK_FALSE,
  };
  // SV_VertexID (the decode pass's fullscreen triangle) lowers to
  // SPIR-V CapabilityDrawParameters.
  VkPhysicalDeviceVulkan11Features features11{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
      .pNext = &features12,
      .shaderDrawParameters =
          context.supportsDrawParameters ? VK_TRUE : VK_FALSE,
  };
  VkPhysicalDeviceFeatures coreFeatures{};
  vkGetPhysicalDeviceFeatures(context.physicalDevice, &coreFeatures);
  VkPhysicalDeviceFeatures2 features2{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &features11,
  };
  features2.features.shaderInt64 =
      coreFeatures.shaderInt64 ? VK_TRUE : VK_FALSE;

  const float priority = 1.0f;
  std::vector<VkDeviceQueueCreateInfo> queueInfos;
  const bool sameFamily = context.graphicsFamily == context.computeFamily;
  const u32 candidateFamilies[3] = {
      context.graphicsFamily,
      context.computeFamily,
      context.transferFamily,
  };
  for (const u32 family : candidateFamilies) {
    bool alreadyCreated = false;
    for (const auto& info : queueInfos)
      if (info.queueFamilyIndex == family) alreadyCreated = true;
    if (!alreadyCreated)
      queueInfos.push_back(VkDeviceQueueCreateInfo{
          .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
          .queueFamilyIndex = family,
          .queueCount = 1,
          .pQueuePriorities = &priority,
      });
  }

  graphicsFamily = context.graphicsFamily;
  computeFamily = context.computeFamily;
  transferFamily = context.transferFamily;

  std::vector<const char*> deviceExtensions;
  if (context.surfaceEnabled)
    deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

  const VkDeviceCreateInfo createInfo{
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &features2,
      .queueCreateInfoCount = static_cast<u32>(queueInfos.size()),
      .pQueueCreateInfos = queueInfos.data(),
      .enabledExtensionCount = static_cast<u32>(deviceExtensions.size()),
      .ppEnabledExtensionNames =
          deviceExtensions.empty() ? nullptr : deviceExtensions.data(),
  };

  if (const VkResult result =
          vkCreateDevice(context.physicalDevice, &createInfo, nullptr, &device);
      result != VK_SUCCESS) {
    KENSHIN_ERROR("[vulkan] vkCreateDevice failed ({})",
                  static_cast<int>(result));
    return false;
  }
  volkLoadDevice(device);

  vkGetDeviceQueue(device, context.graphicsFamily, 0, &graphicsQueue);
  vkGetDeviceQueue(device, context.computeFamily, 0, &computeQueue);
  vkGetDeviceQueue(device, context.transferFamily, 0, &transferQueue);

  VmaVulkanFunctions vmaFns;
  fillVmaVulkanFunctions(vmaFns);
  const VmaAllocatorCreateInfo allocatorInfo{
      .flags = 0,
      .physicalDevice = context.physicalDevice,
      .device = device,
      .pVulkanFunctions = &vmaFns,
      .instance = context.instance,
      .vulkanApiVersion = VK_API_VERSION_1_4,
  };
  if (const VkResult result = vmaCreateAllocator(&allocatorInfo, &allocator);
      result != VK_SUCCESS) {
    KENSHIN_ERROR("[vulkan] vmaCreateAllocator failed ({})",
                  static_cast<int>(result));
    vkDestroyDevice(device, nullptr);
    device = VK_NULL_HANDLE;
    return false;
  }

  const bool dedicatedTransfer =
      context.transferFamily != context.graphicsFamily &&
      context.transferFamily != context.computeFamily;
  KENSHIN_INFO("[vulkan] features13: synchronization2={}, dynamicRendering={}",
               enabledSynchronization2 ? "enabled" : "OFF",
               enabledDynamicRendering ? "enabled" : "OFF");
  KENSHIN_INFO(
      "[vulkan] device created (graphics family {}, compute family {}{}, "
      "transfer family {}{})",
      context.graphicsFamily, context.computeFamily,
      sameFamily ? ", same family" : "", context.transferFamily,
      dedicatedTransfer ? " (dedicated DMA)" : " (shared with graphics)");
  return true;
}

void VulkanDevice::shutdown() {
  if (allocator != VK_NULL_HANDLE) {
    vmaDestroyAllocator(allocator);
    allocator = VK_NULL_HANDLE;
  }
  if (device != VK_NULL_HANDLE) {
    vkDestroyDevice(device, nullptr);
    device = VK_NULL_HANDLE;
  }
}

}  // namespace backend

KENSHIN_NS_END
