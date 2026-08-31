#include "VulkanDevice.h"

#include "VulkanContext.h"
#include "kenshin/core/Log.h"

#pragma warning(push, 0)
#define VMA_IMPLEMENTATION
#include <vma/vk_mem_alloc.h>
#pragma warning(pop)

#include <vector>

KENSHIN_NS_BEGIN

namespace backend
{

    namespace
    {
        /// Fill VMA's function table explicitly from volk's loaded pointers.
        ///
        /// Why not rely on VMA's default static import: in this vcpkg VMA 3.3 +
        /// volk combination, ImportVulkanFunctions_Static ended up calling a null
        /// pointer during vmaCreateAllocator (observed: execute violation at 0x0).
        /// Passing an explicit table sidesteps the interop path and documents the
        /// dependency on volkLoadInstance()/volkLoadDevice() having run first.
        void fillVmaVulkanFunctions(VmaVulkanFunctions& fns)
        {
            fns = {};
            fns.vkGetPhysicalDeviceProperties           = (PFN_vkGetPhysicalDeviceProperties)vkGetPhysicalDeviceProperties;
            fns.vkGetPhysicalDeviceMemoryProperties     = (PFN_vkGetPhysicalDeviceMemoryProperties)vkGetPhysicalDeviceMemoryProperties;
            fns.vkAllocateMemory                        = (PFN_vkAllocateMemory)vkAllocateMemory;
            fns.vkFreeMemory                            = (PFN_vkFreeMemory)vkFreeMemory;
            fns.vkMapMemory                             = (PFN_vkMapMemory)vkMapMemory;
            fns.vkUnmapMemory                           = (PFN_vkUnmapMemory)vkUnmapMemory;
            fns.vkFlushMappedMemoryRanges               = (PFN_vkFlushMappedMemoryRanges)vkFlushMappedMemoryRanges;
            fns.vkInvalidateMappedMemoryRanges          = (PFN_vkInvalidateMappedMemoryRanges)vkInvalidateMappedMemoryRanges;
            fns.vkBindBufferMemory                      = (PFN_vkBindBufferMemory)vkBindBufferMemory;
            fns.vkBindImageMemory                       = (PFN_vkBindImageMemory)vkBindImageMemory;
            fns.vkGetBufferMemoryRequirements           = (PFN_vkGetBufferMemoryRequirements)vkGetBufferMemoryRequirements;
            fns.vkGetImageMemoryRequirements            = (PFN_vkGetImageMemoryRequirements)vkGetImageMemoryRequirements;
            fns.vkCreateBuffer                          = (PFN_vkCreateBuffer)vkCreateBuffer;
            fns.vkDestroyBuffer                         = (PFN_vkDestroyBuffer)vkDestroyBuffer;
            fns.vkCreateImage                           = (PFN_vkCreateImage)vkCreateImage;
            fns.vkDestroyImage                          = (PFN_vkDestroyImage)vkDestroyImage;
            fns.vkCmdCopyBuffer                         = (PFN_vkCmdCopyBuffer)vkCmdCopyBuffer;
            fns.vkGetBufferMemoryRequirements2KHR       = (PFN_vkGetBufferMemoryRequirements2)vkGetBufferMemoryRequirements2;
            fns.vkGetImageMemoryRequirements2KHR        = (PFN_vkGetImageMemoryRequirements2)vkGetImageMemoryRequirements2;
            fns.vkBindBufferMemory2KHR                  = (PFN_vkBindBufferMemory2)vkBindBufferMemory2;
            fns.vkBindImageMemory2KHR                   = (PFN_vkBindImageMemory2)vkBindImageMemory2;
            fns.vkGetPhysicalDeviceMemoryProperties2KHR = (PFN_vkGetPhysicalDeviceMemoryProperties2)vkGetPhysicalDeviceMemoryProperties2;
            fns.vkGetDeviceBufferMemoryRequirements      = (PFN_vkGetDeviceBufferMemoryRequirements)vkGetDeviceBufferMemoryRequirements;
            fns.vkGetDeviceImageMemoryRequirements       = (PFN_vkGetDeviceImageMemoryRequirements)vkGetDeviceImageMemoryRequirements;
        }
    } // namespace

    bool VulkanDevice::init(const VulkanContext& context)
    {
        // Enable only what the physical device reports as supported.
        enabledBufferInt64Atomics = context.supportsBufferInt64Atomics;
        enabledTimelineSemaphore  = context.supportsTimelineSemaphore;

        const VkPhysicalDeviceVulkan12Features features12 {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .shaderBufferInt64Atomics = enabledBufferInt64Atomics ? VK_TRUE : VK_FALSE,
            .timelineSemaphore = enabledTimelineSemaphore ? VK_TRUE : VK_FALSE,
        };

        const float priority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> queueInfos;
        const bool sameFamily = context.graphicsFamily == context.computeFamily;
        const u32 usedFamilies[2] = { context.graphicsFamily, context.computeFamily };
        const u32 familyCount = sameFamily ? 1 : 2;
        for (u32 i = 0; i < familyCount; ++i)
        {
            queueInfos.push_back(VkDeviceQueueCreateInfo {
                .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = usedFamilies[i],
                .queueCount = 1,
                .pQueuePriorities = &priority,
            });
        }

        const VkDeviceCreateInfo createInfo {
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = &features12,
            .queueCreateInfoCount = static_cast<u32>(queueInfos.size()),
            .pQueueCreateInfos = queueInfos.data(),
        };

        if (const VkResult result = vkCreateDevice(context.physicalDevice, &createInfo,
                                                   nullptr, &device);
            result != VK_SUCCESS)
        {
            KENSHIN_ERROR("[vulkan] vkCreateDevice failed ({})", static_cast<int>(result));
            return false;
        }
        volkLoadDevice(device);

        vkGetDeviceQueue(device, context.graphicsFamily, 0, &graphicsQueue);
        vkGetDeviceQueue(device, context.computeFamily, 0, &computeQueue);

        VmaVulkanFunctions vmaFns;
        fillVmaVulkanFunctions(vmaFns);
        const VmaAllocatorCreateInfo allocatorInfo {
            .flags = 0,
            .physicalDevice = context.physicalDevice,
            .device = device,
            .pVulkanFunctions = &vmaFns,
            .instance = context.instance,
            .vulkanApiVersion = VK_API_VERSION_1_4,
        };
        if (const VkResult result = vmaCreateAllocator(&allocatorInfo, &allocator);
            result != VK_SUCCESS)
        {
            KENSHIN_ERROR("[vulkan] vmaCreateAllocator failed ({})", static_cast<int>(result));
            vkDestroyDevice(device, nullptr);
            device = VK_NULL_HANDLE;
            return false;
        }

        KENSHIN_INFO("[vulkan] device created (graphics family {}, compute family {}{})",
                     context.graphicsFamily, context.computeFamily,
                     sameFamily ? ", same family" : "");
        return true;
    }

    void VulkanDevice::shutdown()
    {
        if (allocator != VK_NULL_HANDLE)
        {
            vmaDestroyAllocator(allocator);
            allocator = VK_NULL_HANDLE;
        }
        if (device != VK_NULL_HANDLE)
        {
            vkDestroyDevice(device, nullptr);
            device = VK_NULL_HANDLE;
        }
    }

} // namespace backend

KENSHIN_NS_END
