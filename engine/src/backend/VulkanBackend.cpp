#include "kenshin/backend/VulkanBackend.h"

#include "vulkan/FrameSync.h"
#include "vulkan/Swapchain.h"
#include "vulkan/VulkanContext.h"
#include "vulkan/VulkanDevice.h"
#include "kenshin/core/Log.h"

#pragma warning(push, 0)
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <imgui_impl_win32.h>
#pragma warning(pop)

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

        // ImGui integration (S1.5)
        VkDescriptorPool imguiPool = VK_NULL_HANDLE;
        bool imguiInited = false;
        f32 bgColor[4] = {0.055f, 0.055f, 0.075f, 1.0f};

        // Full-frame capture (UI included; dev/visual checks only)
        bool captureFull = false;
        bool fullCaptured = false;
        VkBuffer captureStaging = VK_NULL_HANDLE;
        VmaAllocation captureAlloc = VK_NULL_HANDLE;
        u32 captureWidth = 0, captureHeight = 0;
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

        // Frame resources (command pools etc.) are device-level and needed by
        // windowless compute paths too.
        if (!m_impl->frameSync.init(m_impl->device))
        {
            KENSHIN_ERROR("[vulkan] frame resources init failed");
            return false;
        }

        if (params.nativeWindowHandle && !m_impl->presentSupported)
        {
            KENSHIN_ERROR("[vulkan] graphics family does not support present");
            return false;
        }
        return true;
    }

    void VulkanBackend::shutdown()
    {
        if (!m_impl)
            return;
        if (m_impl->imguiInited)
        {
            ImGui_ImplVulkan_Shutdown();
            ImGui_ImplWin32_Shutdown();
            m_impl->imguiInited = false;
        }
        if (m_impl->device.device != VK_NULL_HANDLE)
            vkDeviceWaitIdle(m_impl->device.device);
        if (m_impl->imguiPool != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(m_impl->device.device, m_impl->imguiPool, nullptr);
            m_impl->imguiPool = VK_NULL_HANDLE;
        }
        if (m_impl->captureStaging != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(m_impl->device.allocator,
                             m_impl->captureStaging, m_impl->captureAlloc);
            m_impl->captureStaging = VK_NULL_HANDLE;
        }
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
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                VK_IMAGE_USAGE_SAMPLED_BIT))
            return false;
        ensureViewportTextures();

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

        // offscreen: TRANSFER_DST -> SHADER_READ_ONLY (sampled by the ImGui
        // viewport panel later in this same command buffer).
        recordBarrier(slot.commandBuffer,
                      imageBarrier(slot.offscreen.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_READ_BIT));

        if (impl.imguiInited)
        {
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplWin32_NewFrame();
            ImGui::NewFrame();
        }

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

        vkDeviceWaitIdle(impl.device.device);  // PROBE C: drain before ImGui writes
        ImDrawData* drawData = nullptr;
        if (impl.imguiInited)
        {
            ImGui::Render();
            drawData = ImGui::GetDrawData();
        }

        // swapchain image: UNDEFINED -> COLOR_ATTACHMENT (ImGui draws the
        // whole window through dynamic rendering).
        recordBarrier(slot.commandBuffer,
                      imageBarrier(dstImage, VK_IMAGE_LAYOUT_UNDEFINED,
                                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                   VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT));

        const VkClearValue uiClear {};
        const VkRenderingAttachmentInfo colorAttachment {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = impl.swapchain.imageViews[impl.swapchain.imageIndex],
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = uiClear,
        };
        const VkRenderingInfo renderingInfo {
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = {0, 0, impl.swapchain.extent.width, impl.swapchain.extent.height},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachment,
        };
        vkCmdBeginRendering(slot.commandBuffer, &renderingInfo);
        if (drawData && drawData->CmdListsCount > 0)
            // PROBE A: ImGui draw disabled - empty UI pass only
            ImGui_ImplVulkan_RenderDrawData(drawData, slot.commandBuffer);
        vkCmdEndRendering(slot.commandBuffer);

        // swapchain image: COLOR_ATTACHMENT -> PRESENT (via a capture copy
        // when a full-frame screenshot was requested).
        if (impl.captureFull)
        {
            ensureCaptureStaging(impl.swapchain.extent.width, impl.swapchain.extent.height);
            recordBarrier(slot.commandBuffer,
                          imageBarrier(dstImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                       VK_ACCESS_2_TRANSFER_READ_BIT));
            const VkBufferImageCopy captureRegion {
                .bufferOffset = 0,
                .bufferRowLength = 0,
                .bufferImageHeight = 0,
                .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                .imageOffset = {0, 0, 0},
                .imageExtent = {impl.swapchain.extent.width, impl.swapchain.extent.height, 1},
            };
            vkCmdCopyImageToBuffer(slot.commandBuffer, dstImage,
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   impl.captureStaging, 1, &captureRegion);
            recordBarrier(slot.commandBuffer,
                          imageBarrier(dstImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                       VK_ACCESS_2_MEMORY_READ_BIT));
            impl.captureFull = false;
            impl.fullCaptured = true;
        }
        else
        {
            recordBarrier(slot.commandBuffer,
                          imageBarrier(dstImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                       VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                       VK_ACCESS_2_MEMORY_READ_BIT));
        }

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
        recordBarrier(cmd,
                      imageBarrier(slot.offscreen.image,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                   VK_ACCESS_2_TRANSFER_READ_BIT));
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

    // ── one-shot compute (S1.4) ───────────────────────────────────

    bool VulkanBackend::runComputeOneshot(const Job& job,
                                          std::vector<u8> outputs[Job::kMaxBuffers])
    {
        Impl& impl = *m_impl;
        VkDevice device = impl.device.device;
        if (!device || !job.spirv || job.spirvBytes == 0)
            return false;

        // Count used buffers.
        u32 bufferCount = 0;
        for (u32 i = 0; i < Job::kMaxBuffers; ++i)
            if (job.buffers[i].size > 0)
                bufferCount = i + 1;

        // Shader module + descriptor set layout + pipeline layout + pipeline.
        const VkShaderModuleCreateInfo moduleInfo {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = job.spirvBytes,
            .pCode = job.spirv,
        };
        VkShaderModule module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(device, &moduleInfo, nullptr, &module) != VK_SUCCESS)
        {
            KENSHIN_ERROR("[compute] shader module creation failed");
            return false;
        }

        VkDescriptorSetLayoutBinding layoutBindings[Job::kMaxBuffers] = {};
        for (u32 i = 0; i < bufferCount; ++i)
        {
            layoutBindings[i] = VkDescriptorSetLayoutBinding {
                .binding = i,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            };
        }
        const VkDescriptorSetLayoutCreateInfo setLayoutInfo {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = bufferCount,
            .pBindings = layoutBindings,
        };
        VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
        vkCreateDescriptorSetLayout(device, &setLayoutInfo, nullptr, &setLayout);

        const VkPipelineLayoutCreateInfo pipelineLayoutInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = bufferCount ? 1u : 0u,
            .pSetLayouts = &setLayout,
        };
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);

        const VkComputePipelineCreateInfo pipelineInfo {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = VkPipelineShaderStageCreateInfo {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = module,
                .pName = job.entryPoint,
            },
            .layout = pipelineLayout,
        };
        VkPipeline pipeline = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                     nullptr, &pipeline) != VK_SUCCESS)
        {
            KENSHIN_ERROR("[compute] pipeline creation failed");
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
            vkDestroyShaderModule(device, module, nullptr);
            return false;
        }

        // Host-visible, host-coherent buffers (smoke-test scale: simple over fast).
        VkBuffer buffers[Job::kMaxBuffers] = {};
        VmaAllocation allocations[Job::kMaxBuffers] = {};
        void* mapped[Job::kMaxBuffers] = {};
        bool ok = true;
        for (u32 i = 0; i < bufferCount && ok; ++i)
        {
            const VkBufferCreateInfo bufferInfo {
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size = static_cast<VkDeviceSize>(job.buffers[i].size),
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            };
            const VmaAllocationCreateInfo allocInfo {
                .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                         VMA_ALLOCATION_CREATE_MAPPED_BIT,
                .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            };
            VmaAllocationInfo allocInfoOut {};
            if (vmaCreateBuffer(impl.device.allocator, &bufferInfo, &allocInfo,
                                &buffers[i], &allocations[i], &allocInfoOut) != VK_SUCCESS)
            {
                KENSHIN_ERROR("[compute] buffer creation failed (slot {})", i);
                ok = false;
                break;
            }
            mapped[i] = allocInfoOut.pMappedData;
            if (job.buffers[i].initData)
                std::memcpy(mapped[i], job.buffers[i].initData, job.buffers[i].size);
        }

        // Descriptor set from a transient pool.
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (ok && bufferCount > 0)
        {
            const VkDescriptorPoolSize poolSize {
                .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = Job::kMaxBuffers,
            };
            const VkDescriptorPoolCreateInfo poolInfo {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = 1,
                .poolSizeCount = 1,
                .pPoolSizes = &poolSize,
            };
            vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool);

            const VkDescriptorSetAllocateInfo setInfo {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &setLayout,
            };
            if (vkAllocateDescriptorSets(device, &setInfo, &set) != VK_SUCCESS)
            {
                KENSHIN_ERROR("[compute] descriptor set allocation failed");
                ok = false;
            }
        }
        if (ok && bufferCount > 0)
        {
            VkDescriptorBufferInfo bufferInfos[Job::kMaxBuffers] = {};
            VkWriteDescriptorSet writes[Job::kMaxBuffers] = {};
            for (u32 i = 0; i < bufferCount; ++i)
            {
                bufferInfos[i] = VkDescriptorBufferInfo {
                    .buffer = buffers[i],
                    .range = VK_WHOLE_SIZE,
                };
                writes[i] = VkWriteDescriptorSet {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = set,
                    .dstBinding = i,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &bufferInfos[i],
                };
            }
            vkUpdateDescriptorSets(device, bufferCount, writes, 0, nullptr);
        }

        // Record + submit the dispatch, then wait (host-coherent: submit and
        // queueWaitIdle provide the necessary host sync boundaries).
        if (ok)
        {
            VkCommandBuffer cmd = VK_NULL_HANDLE;
            const VkCommandBufferAllocateInfo allocInfo {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool = impl.frameSync.slots[0].commandPool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1,
            };
            vkAllocateCommandBuffers(device, &allocInfo, &cmd);
            const VkCommandBufferBeginInfo beginInfo {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };
            vkBeginCommandBuffer(cmd, &beginInfo);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            if (bufferCount > 0)
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
                                        0, 1, &set, 0, nullptr);
            vkCmdDispatch(cmd, job.groupCountX, 1, 1);
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
            vkFreeCommandBuffers(device, impl.frameSync.slots[0].commandPool, 1, &cmd);
        }

        // Readback.
        if (ok)
        {
            for (u32 i = 0; i < bufferCount; ++i)
            {
                if (!job.buffers[i].readBack || !mapped[i])
                    continue;
                outputs[i].resize(job.buffers[i].size);
                std::memcpy(outputs[i].data(), mapped[i], job.buffers[i].size);
            }
        }

        // Teardown.
        if (pool != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(device, pool, nullptr);
        for (u32 i = 0; i < bufferCount; ++i)
            if (buffers[i] != VK_NULL_HANDLE)
                vmaDestroyBuffer(impl.device.allocator, buffers[i], allocations[i]);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
        vkDestroyShaderModule(device, module, nullptr);
        return ok;
    }


    // ── ImGui integration (S1.5) ──────────────────────────────────

    void VulkanBackend::ensureViewportTextures()
    {
        if (!m_impl->imguiInited)
            return;
        for (u32 i = 0; i < kMaxFramesInFlight; ++i)
        {
            FrameSlot& slot = m_impl->frameSync.slots[i];
            if (slot.offscreen.image == VK_NULL_HANDLE ||
                slot.imguiTextureImage == slot.offscreen.image)
                continue;
            if (slot.imguiTexture)
            {
                ImGui_ImplVulkan_RemoveTexture(
                    static_cast<VkDescriptorSet>(slot.imguiTexture));
                slot.imguiTexture = nullptr;
            }
            slot.imguiTexture = ImGui_ImplVulkan_AddTexture(
                slot.offscreen.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            slot.imguiTextureImage = slot.offscreen.image;
        }
    }

    bool VulkanBackend::initImGuiBackend(void* hwnd, u32 width, u32 height)
    {
        Impl& impl = *m_impl;
        if (impl.imguiInited || !hwnd)
            return impl.imguiInited;

        // The swapchain must exist so its format drives the ImGui pipeline.
        if (impl.swapchain.swapchain == VK_NULL_HANDLE &&
            !impl.swapchain.init(impl.context.instance, impl.context.physicalDevice,
                                 impl.device.device, impl.hwnd, width, height))
            return false;

        // NOTE: the validation layer (vcpkg build) warns that this pool lacks
        // VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE even though it is listed below —
        // a layer bookkeeping gap for the Vulkan 1.4 standalone image
        // descriptor enums, not a real pool mismatch (allocation succeeds,
        // counts are far below limits). Verified against source and pool handle.
        const VkDescriptorPoolSize poolSizes[] = {
            { VK_DESCRIPTOR_TYPE_SAMPLER, 16 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64 },
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 64 },
        };
        const VkDescriptorPoolCreateInfo poolInfo {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 64,
            .poolSizeCount = 3,  // PROBE D: all three sizes counted
            .pPoolSizes = poolSizes,
        };
        if (vkCreateDescriptorPool(impl.device.device, &poolInfo, nullptr,
                                   &impl.imguiPool) != VK_SUCCESS)
        {
            KENSHIN_ERROR("[imgui] descriptor pool creation failed");
            return false;
        }

        VkFormat colorFormat = impl.swapchain.format;
        ImGui_ImplVulkan_InitInfo initInfo {};
        initInfo.ApiVersion = VK_API_VERSION_1_4;
        initInfo.Instance = impl.context.instance;
        initInfo.PhysicalDevice = impl.context.physicalDevice;
        initInfo.Device = impl.device.device;
        initInfo.QueueFamily = impl.device.graphicsFamily;
        initInfo.Queue = impl.device.graphicsQueue;
        initInfo.DescriptorPool = impl.imguiPool;
        initInfo.MinImageCount = 2;
        initInfo.ImageCount = 3;
        initInfo.UseDynamicRendering = true;
        initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = VkPipelineRenderingCreateInfoKHR {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &colorFormat,
        };
        initInfo.CheckVkResultFn = [](VkResult err) {
            if (err != VK_SUCCESS)
                KENSHIN_ERROR("[imgui][vulkan] VkResult = {}", static_cast<int>(err));
        };
        initInfo.MinAllocationSize = 1024 * 1024;

        if (!ImGui_ImplVulkan_Init(&initInfo))
        {
            KENSHIN_ERROR("[imgui] ImGui_ImplVulkan_Init failed");
            vkDestroyDescriptorPool(impl.device.device, impl.imguiPool, nullptr);
            impl.imguiPool = VK_NULL_HANDLE;
            return false;
        }
        if (!ImGui_ImplWin32_Init(hwnd))
        {
            KENSHIN_ERROR("[imgui] ImGui_ImplWin32_Init failed");
            ImGui_ImplVulkan_Shutdown();
            vkDestroyDescriptorPool(impl.device.device, impl.imguiPool, nullptr);
            impl.imguiPool = VK_NULL_HANDLE;
            return false;
        }

        impl.imguiInited = true;
        ensureViewportTextures();
        KENSHIN_INFO("[imgui] backend ready (dynamic rendering, docking UI)");
        return true;
    }

    void* VulkanBackend::viewportImGuiTexture()
    {
        FrameSlot& slot = m_impl->frameSync.slots[m_impl->frameSync.frameIndex];
        return slot.imguiTexture;
    }


    void VulkanBackend::requestFullFrameCapture()
    {
        m_impl->captureFull = true;
    }

    bool VulkanBackend::readbackFullFrame(std::vector<u8>& rgbaOut, u32& width, u32& height)
    {
        Impl& impl = *m_impl;
        if (!impl.fullCaptured || impl.captureStaging == VK_NULL_HANDLE)
            return false;
        vkWaitForFences(impl.device.device, 1,
                        &impl.frameSync.slots[impl.frameSync.lastCompleted].inFlight,
                        VK_TRUE, UINT64_MAX);

        width = impl.captureWidth;
        height = impl.captureHeight;
        void* mapped = nullptr;
        if (vmaMapMemory(impl.device.allocator, impl.captureAlloc, &mapped) != VK_SUCCESS)
            return false;

        rgbaOut.resize(static_cast<usize>(width) * height * 4);
        const u8* src = static_cast<const u8*>(mapped);
        const bool bgra = impl.swapchain.format == VK_FORMAT_B8G8R8A8_UNORM;
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
        vmaUnmapMemory(impl.device.allocator, impl.captureAlloc);
        impl.fullCaptured = false;
        return true;
    }

    void VulkanBackend::ensureCaptureStaging(u32 width, u32 height)
    {
        Impl& impl = *m_impl;
        if (impl.captureStaging != VK_NULL_HANDLE && impl.captureWidth == width &&
            impl.captureHeight == height)
            return;
        if (impl.captureStaging != VK_NULL_HANDLE)
            vmaDestroyBuffer(impl.device.allocator, impl.captureStaging, impl.captureAlloc);
        const VkBufferCreateInfo bufferInfo {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = static_cast<VkDeviceSize>(width) * height * 4,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        const VmaAllocationCreateInfo allocInfo {
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        };
        vmaCreateBuffer(impl.device.allocator, &bufferInfo, &allocInfo,
                        &impl.captureStaging, &impl.captureAlloc, nullptr);
        impl.captureWidth = width;
        impl.captureHeight = height;
    }

} // namespace backend

KENSHIN_NS_END
