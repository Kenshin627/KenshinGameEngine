#include "kenshin/backend/VulkanBackend.h"

#include "vulkan/FrameSync.h"
#include "vulkan/Swapchain.h"
#include "vulkan/VulkanContext.h"
#include "vulkan/VulkanDevice.h"
#include "kenshin/core/Log.h"

#include <cstdio>
#include <cstring>

KENSHIN_NS_BEGIN

namespace backend
{

    struct VulkanBackend::Impl
    {
        VulkanContext context;
        VulkanDevice device;
        bool presentSupported = false;
        void* hwnd = nullptr;

        Swapchain swapchain;
        FrameSync frameSync;
        bool frameActive = false;
        u32 activeWidth = 0;
        u32 activeHeight = 0;
    };

    VulkanBackend::VulkanBackend()
        : m_impl(std::make_unique<Impl>())
    {
    }

    VulkanBackend::~VulkanBackend()
    {
        shutdown();
    }

    // ── lifecycle ────────────────────────────────────────────────

    bool VulkanBackend::initialize(const BackendInitParams& params)
    {
        const bool needSurface = params.nativeWindowHandle != nullptr;
        if (!m_impl->context.init(params.enableValidation, needSurface))
            return false;

        if (params.nativeWindowHandle)
            m_impl->presentSupported =
                m_impl->context.checkPresentSupport(params.nativeWindowHandle);

        if (!m_impl->device.init(m_impl->context))
            return false;

        m_impl->hwnd = params.nativeWindowHandle;

        // ---- capability report -------------------------------------------
        BackendCaps& caps = m_caps;
        std::snprintf(caps.deviceName, sizeof(caps.deviceName), "%s",
                      m_impl->context.props.deviceName);
        caps.gpuType = [](VkPhysicalDeviceType type) {
            switch (type)
            {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return GpuType::Discrete;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return GpuType::Integrated;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return GpuType::Virtual;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:            return GpuType::Cpu;
            default:                                     return GpuType::Unknown;
            }
        }(m_impl->context.props.deviceType);

        caps.apiVersionMajor = VK_API_VERSION_MAJOR(m_impl->context.props.apiVersion);
        caps.apiVersionMinor = VK_API_VERSION_MINOR(m_impl->context.props.apiVersion);
        caps.apiVersionPatch = VK_API_VERSION_PATCH(m_impl->context.props.apiVersion);
        caps.driverVersionRaw = m_impl->context.props.driverVersion;

        u64 localMB = 0;
        const auto& memProps = m_impl->context.memProps;
        for (u32 i = 0; i < memProps.memoryHeapCount; ++i)
            if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                localMB += memProps.memoryHeaps[i].size / (1024 * 1024);
        caps.deviceLocalMemoryMB = localMB;

        caps.supportsShaderBufferInt64Atomics = m_impl->context.supportsBufferInt64Atomics;
        caps.enabledShaderBufferInt64Atomics  = m_impl->device.enabledBufferInt64Atomics;
        caps.supportsTimelineSemaphore = m_impl->context.supportsTimelineSemaphore;
        caps.enabledTimelineSemaphore  = m_impl->device.enabledTimelineSemaphore;

        caps.presentSupported = m_impl->presentSupported;
        caps.graphicsFamily = m_impl->context.graphicsFamily;
        caps.computeFamily  = m_impl->context.computeFamily;
        caps.transferFamily = m_impl->context.transferFamily;
        caps.graphicsAndComputeSameFamily =
            m_impl->context.graphicsFamily == m_impl->context.computeFamily;
        caps.dedicatedTransferFamily =
            caps.transferFamily != caps.graphicsFamily &&
            caps.transferFamily != caps.computeFamily;

        static const char* kGpuTypeNames[] = { "unknown", "integrated", "discrete", "virtual", "cpu" };
        KENSHIN_INFO("[vulkan] GPU: {} (api {}.{}, {}, {}MB device-local)",
                     caps.deviceName, caps.apiVersionMajor, caps.apiVersionMinor,
                     kGpuTypeNames[static_cast<int>(caps.gpuType)], caps.deviceLocalMemoryMB);
        KENSHIN_INFO("[vulkan] feature shaderBufferInt64Atomics = {}{}",
                     caps.supportsShaderBufferInt64Atomics ? "supported" : "NOT supported",
                     caps.enabledShaderBufferInt64Atomics ? " (enabled)" : "");
        KENSHIN_INFO("[vulkan] feature timelineSemaphore = {}{}",
                     caps.supportsTimelineSemaphore ? "supported" : "NOT supported",
                     caps.enabledTimelineSemaphore ? " (enabled)" : "");
        KENSHIN_INFO("[vulkan] queues: graphics={} compute={}{} transfer={} ({}), {}",
                     caps.graphicsFamily, caps.computeFamily,
                     caps.graphicsAndComputeSameFamily ? " (same family)" : "",
                     caps.transferFamily,
                     caps.dedicatedTransferFamily ? "dedicated DMA" : "shared",
                     params.nativeWindowHandle
                         ? (caps.presentSupported ? "present=ok" : "present=NOT supported")
                         : "present=unchecked (no window)");

        // ---- presentation resources --------------------------------------
        if (params.nativeWindowHandle)
        {
            if (!m_impl->presentSupported)
            {
                KENSHIN_ERROR("[vulkan] graphics family does not support present");
                return false;
            }
            if (!m_impl->frameSync.init(m_impl->device))
            {
                KENSHIN_ERROR("[vulkan] frame resources init failed");
                return false;
            }
        }
        return true;
    }

    void VulkanBackend::shutdown()
    {
        if (!m_impl)
            return;
        if (m_impl->device.device != VK_NULL_HANDLE)
            vkDeviceWaitIdle(m_impl->device.device);
        m_impl->frameSync.shutdown(m_impl->device);
        m_impl->swapchain.shutdown(m_impl->context.instance, m_impl->device.device);
        m_impl->device.shutdown();
        m_impl->context.shutdown();
    }

    std::unique_ptr<IRenderBackend> createVulkanBackend()
    {
        return std::make_unique<VulkanBackend>();
    }

    // ── frame loop (S1.3) ─────────────────────────────────────────

    namespace
    {
        VkImageMemoryBarrier2 imageBarrier(VkImage image, VkImageLayout oldLayout,
                                           VkImageLayout newLayout,
                                           VkPipelineStageFlags2 dstStage,
                                           VkAccessFlags2 dstAccess)
        {
            return VkImageMemoryBarrier2 {
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask = dstStage,
                .dstAccessMask = dstAccess,
                .oldLayout = oldLayout,
                .newLayout = newLayout,
                .image = image,
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
            };
        }

        void recordBarrier(VkCommandBuffer cmd, const VkImageMemoryBarrier2& barrier)
        {
            const VkDependencyInfo dep {
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers = &barrier,
            };
            vkCmdPipelineBarrier2(cmd, &dep);
        }
    } // namespace

    bool VulkanBackend::beginFrame(const FrameBeginInfo& info)
    {
        Impl& impl = *m_impl;
        if (impl.swapchain.swapchain == VK_NULL_HANDLE)
        {
            if (!impl.swapchain.init(impl.context.instance, impl.context.physicalDevice,
                                     impl.device.device, impl.hwnd, info.width, info.height))
                return false;
        }

        FrameSlot& slot = impl.frameSync.slots[impl.frameSync.frameIndex];

        vkWaitForFences(impl.device.device, 1, &slot.inFlight, VK_TRUE, UINT64_MAX);
        vkResetFences(impl.device.device, 1, &slot.inFlight);

        // (Re)create per-frame offscreen targets on size/format change.
        if (!impl.frameSync.ensureOffscreens(
                impl.device, info.width, info.height, impl.swapchain.format,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT))
            return false;

        // Acquire; rebuild once on out-of-date (e.g. window resize).
        VkResult acquireResult =
            impl.swapchain.acquire(impl.device.device, slot.imageAvailable);
        if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR)
        {
            if (!impl.swapchain.recreate(impl.context.physicalDevice, impl.device.device,
                                         info.width, info.height))
                return false;
            acquireResult = impl.swapchain.acquire(impl.device.device, slot.imageAvailable);
        }
        if (acquireResult < 0)
        {
            KENSHIN_ERROR("[vulkan] acquire failed ({})", static_cast<int>(acquireResult));
            return false;
        }

        const VkCommandBufferBeginInfo beginInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkResetCommandBuffer(slot.commandBuffer, 0);
        vkBeginCommandBuffer(slot.commandBuffer, &beginInfo);

        // Clear the offscreen target (the "rendering" of S1.3).
        recordBarrier(slot.commandBuffer,
                      imageBarrier(slot.offscreen.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                   VK_ACCESS_2_TRANSFER_WRITE_BIT));
        VkClearColorValue clear {};
        clear.float32[0] = info.clearColor[0];
        clear.float32[1] = info.clearColor[1];
        clear.float32[2] = info.clearColor[2];
        clear.float32[3] = info.clearColor[3];
        const VkImageSubresourceRange range {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(slot.commandBuffer, slot.offscreen.image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);

        impl.frameActive = true;
        impl.activeWidth = info.width;
        impl.activeHeight = info.height;
        return true;
    }

    void VulkanBackend::endFrame()
    {
        Impl& impl = *m_impl;
        if (!impl.frameActive)
            return;
        impl.frameActive = false;

        FrameSlot& slot = impl.frameSync.slots[impl.frameSync.frameIndex];
        const VkImage dstImage = impl.swapchain.images[impl.swapchain.imageIndex];

        // offscreen: TRANSFER_DST -> TRANSFER_SRC
        recordBarrier(slot.commandBuffer,
                      imageBarrier(slot.offscreen.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT));
        // swapchain image: UNDEFINED -> TRANSFER_DST
        recordBarrier(slot.commandBuffer,
                      imageBarrier(dstImage, VK_IMAGE_LAYOUT_UNDEFINED,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT));

        const VkImageBlit blit {
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffsets = {{0, 0, 0},
                           {static_cast<i32>(slot.offscreen.width),
                            static_cast<i32>(slot.offscreen.height), 1}},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffsets = {{0, 0, 0},
                           {static_cast<i32>(impl.swapchain.extent.width),
                            static_cast<i32>(impl.swapchain.extent.height), 1}},
        };
        vkCmdBlitImage(slot.commandBuffer,
                       slot.offscreen.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &blit, VK_FILTER_LINEAR);

        // swapchain image: TRANSFER_DST -> PRESENT
        recordBarrier(slot.commandBuffer,
                      imageBarrier(dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                   VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                   VK_ACCESS_2_MEMORY_READ_BIT));

        if (vkEndCommandBuffer(slot.commandBuffer) != VK_SUCCESS)
        {
            KENSHIN_ERROR("[vulkan] vkEndCommandBuffer failed");
            return;
        }

        const VkCommandBufferSubmitInfo cmdSubmit {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .commandBuffer = slot.commandBuffer,
        };
        const VkSemaphoreSubmitInfo waitSubmit {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = slot.imageAvailable,
            .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        };
        const VkSemaphoreSubmitInfo signalSubmit {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = impl.swapchain.presentSemaphores[impl.swapchain.imageIndex],
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        };
        const VkSubmitInfo2 submit {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .waitSemaphoreInfoCount = 1,
            .pWaitSemaphoreInfos = &waitSubmit,
            .commandBufferInfoCount = 1,
            .pCommandBufferInfos = &cmdSubmit,
            .signalSemaphoreInfoCount = 1,
            .pSignalSemaphoreInfos = &signalSubmit,
        };
        if (vkQueueSubmit2(impl.device.graphicsQueue, 1, &submit, slot.inFlight) != VK_SUCCESS)
        {
            KENSHIN_ERROR("[vulkan] vkQueueSubmit2 failed");
            return;
        }

        const VkSemaphore presentWait = impl.swapchain.presentSemaphores[impl.swapchain.imageIndex];
        const VkPresentInfoKHR present {
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &presentWait,
            .swapchainCount = 1,
            .pSwapchains = &impl.swapchain.swapchain,
            .pImageIndices = &impl.swapchain.imageIndex,
        };
        const VkResult presentResult = vkQueuePresentKHR(impl.device.graphicsQueue, &present);
        if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR)
        {
            // Rebuild immediately; nothing else is pending (fences drained above).
            impl.swapchain.recreate(impl.context.physicalDevice, impl.device.device,
                                    impl.activeWidth, impl.activeHeight);
        }

        impl.frameSync.lastCompleted = impl.frameSync.frameIndex;
        impl.frameSync.anyCompleted = true;
        impl.frameSync.advance();
    }

    bool VulkanBackend::readbackFrame(std::vector<u8>& rgbaOut, u32& width, u32& height)
    {
        Impl& impl = *m_impl;
        if (!impl.frameSync.anyCompleted)
            return false;

        const FrameSlot& slot = impl.frameSync.slots[impl.frameSync.lastCompleted];
        width = slot.offscreen.width;
        height = slot.offscreen.height;

        vkWaitForFences(impl.device.device, 1, &slot.inFlight, VK_TRUE, UINT64_MAX);

        // Staging buffer (host visible) for the copy.
        const VkBufferCreateInfo bufferInfo {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = static_cast<VkDeviceSize>(width) * height * 4,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        const VmaAllocationCreateInfo stagingAlloc {
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
            .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        };
        VkBuffer staging = VK_NULL_HANDLE;
        VmaAllocation stagingAllocHandle = VK_NULL_HANDLE;
        void* mapped = nullptr;
        if (vmaCreateBuffer(impl.device.allocator, &bufferInfo, &stagingAlloc,
                            &staging, &stagingAllocHandle, nullptr) != VK_SUCCESS)
            return false;
        vmaMapMemory(impl.device.allocator, stagingAllocHandle, &mapped);

        // One-shot copy on the graphics queue (slot 0's pool; drained by the fence wait).
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        const VkCommandBufferAllocateInfo allocInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = impl.frameSync.slots[0].commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        vkAllocateCommandBuffers(impl.device.device, &allocInfo, &cmd);
        const VkCommandBufferBeginInfo beginInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkBeginCommandBuffer(cmd, &beginInfo);
        const VkBufferImageCopy region {
            .bufferOffset = 0,
            .bufferRowLength = 0,   // tightly packed
            .bufferImageHeight = 0,
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, 1},
        };
        vkCmdCopyImageToBuffer(cmd, slot.offscreen.image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1, &region);
        vkEndCommandBuffer(cmd);
        const VkCommandBufferSubmitInfo cmdSubmit {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .commandBuffer = cmd,
        };
        const VkSubmitInfo2 submit {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .commandBufferInfoCount = 1,
            .pCommandBufferInfos = &cmdSubmit,
        };
        vkQueueSubmit2(impl.device.graphicsQueue, 1, &submit, VK_NULL_HANDLE);
        vkQueueWaitIdle(impl.device.graphicsQueue);
        vkFreeCommandBuffers(impl.device.device, impl.frameSync.slots[0].commandPool, 1, &cmd);

        // Swizzle BGRA -> RGBA when the surface format is BGRA-ordered.
        rgbaOut.resize(static_cast<usize>(width) * height * 4);
        const bool bgra = impl.swapchain.format == VK_FORMAT_B8G8R8A8_UNORM;
        const u8* src = static_cast<const u8*>(mapped);
        if (bgra)
        {
            for (usize i = 0; i < rgbaOut.size(); i += 4)
            {
                rgbaOut[i + 0] = src[i + 2];
                rgbaOut[i + 1] = src[i + 1];
                rgbaOut[i + 2] = src[i + 0];
                rgbaOut[i + 3] = src[i + 3];
            }
        }
        else
            std::memcpy(rgbaOut.data(), src, rgbaOut.size());

        vmaUnmapMemory(impl.device.allocator, stagingAllocHandle);
        vmaDestroyBuffer(impl.device.allocator, staging, stagingAllocHandle);
        return true;
    }

} // namespace backend

KENSHIN_NS_END
