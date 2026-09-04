#include "kenshin/backend/VulkanBackend.h"

#include "kenshin/core/Log.h"
#include "kenshin/manager/ShaderManager.h"
#include "vulkan/BufferStore.h"
#include "vulkan/CommandContext.h"
#include "vulkan/ComputeOneShot.h"
#include "vulkan/DescriptorCache.h"
#include "vulkan/FrameSync.h"
#include "vulkan/MeshStore.h"
#include "vulkan/PipelineStore.h"
#include "vulkan/Readback.h"
#include "vulkan/Swapchain.h"
#include "vulkan/UploadEngine.h"
#include "vulkan/Vk.h"
#include "vulkan/VulkanContext.h"
#include "vulkan/VulkanDevice.h"

#pragma warning(push, 0)
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <imgui_impl_win32.h>
#pragma warning(pop)

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

KENSHIN_NS_BEGIN

namespace backend {

struct VulkanBackend::Impl {
  VulkanContext context;
  VulkanDevice device;
  bool presentSupported = false;
  void* hwnd = nullptr;

  Swapchain swapchain;
  FrameSync frameSync;
  StagingRing staging;  // per-frame dynamic data slices (M2 consumer)
  bool frameActive = false;
  u32 activeWidth = 0;
  u32 activeHeight = 0;

  // ImGui integration (S1.5)
  DescriptorCache descriptors;
  CommandContext cmd;  // the current frame slot's recording
  bool imguiInited = false;
  f32 bgColor[4] = {0.055f, 0.055f, 0.075f, 1.0f};

  // Frame-content capture + host readback (Readback unit)
  Readback readback;

  // Uploaded meshes (MeshStore: SOA buffers behind handles)
  MeshStore meshes;
  BufferStore buffers;  // general device buffers (GPU Scene paths)

  VkImage depthImage = VK_NULL_HANDLE;
  VmaAllocation depthAlloc = VK_NULL_HANDLE;
  VkImageView depthView = VK_NULL_HANDLE;
  u32 depthWidth = 0, depthHeight = 0;

  // GPU timestamps + graphics pipelines (PipelineStore: table + disk cache)
  bool gpuTimestamps = false;
  f32 timestampPeriodNs = 0.0f;
  PipelineStore pipelines;
};

VulkanBackend::VulkanBackend() : m_impl(std::make_unique<Impl>()) {}

VulkanBackend::~VulkanBackend() { shutdown(); }

// ── lifecycle ────────────────────────────────────────────────

bool VulkanBackend::initialize(const BackendInitParams& params) {
  // If a validation layer ships next to the exe (vcpkg dev build),
  // point the loader at it before instance creation.
  if (params.enableValidation) {
    wchar_t exePath[MAX_PATH]{};
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    const auto exeDir = std::filesystem::path(exePath).parent_path();
    if (std::filesystem::exists(exeDir / "VkLayer_khronos_validation.json")) {
      SetEnvironmentVariableW(L"VK_LAYER_PATH", exeDir.c_str());
      KENSHIN_INFO("validation layer path -> {}", exeDir.string());
    }
  }

  const bool needSurface = params.nativeWindowHandle != nullptr;
  if (!m_impl->context.init(params.enableValidation, needSurface)) return false;

  if (params.nativeWindowHandle)
    m_impl->presentSupported =
        m_impl->context.checkPresentSupport(params.nativeWindowHandle);

  if (!m_impl->device.init(m_impl->context)) return false;

  m_impl->hwnd = params.nativeWindowHandle;

  // ---- capability report -------------------------------------------
  BackendCaps& caps = m_caps;
  std::snprintf(caps.deviceName, sizeof(caps.deviceName), "%s",
                m_impl->context.props.deviceName);
  caps.gpuType = [](VkPhysicalDeviceType type) {
    switch (type) {
      case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        return GpuType::Discrete;
      case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        return GpuType::Integrated;
      case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        return GpuType::Virtual;
      case VK_PHYSICAL_DEVICE_TYPE_CPU:
        return GpuType::Cpu;
      default:
        return GpuType::Unknown;
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

  caps.supportsShaderBufferInt64Atomics =
      m_impl->context.supportsBufferInt64Atomics;
  caps.enabledShaderBufferInt64Atomics =
      m_impl->device.enabledBufferInt64Atomics;
  caps.supportsTimelineSemaphore = m_impl->context.supportsTimelineSemaphore;
  caps.enabledTimelineSemaphore = m_impl->device.enabledTimelineSemaphore;

  caps.presentSupported = m_impl->presentSupported;
  caps.graphicsFamily = m_impl->context.graphicsFamily;
  caps.computeFamily = m_impl->context.computeFamily;
  caps.transferFamily = m_impl->context.transferFamily;
  caps.graphicsAndComputeSameFamily =
      m_impl->context.graphicsFamily == m_impl->context.computeFamily;
  caps.dedicatedTransferFamily = caps.transferFamily != caps.graphicsFamily &&
                                 caps.transferFamily != caps.computeFamily;

  static const char* kGpuTypeNames[] = {"unknown", "integrated", "discrete",
                                        "virtual", "cpu"};
  KENSHIN_INFO("[vulkan] GPU: {} (api {}.{}, {}, {}MB device-local)",
               caps.deviceName, caps.apiVersionMajor, caps.apiVersionMinor,
               kGpuTypeNames[static_cast<int>(caps.gpuType)],
               caps.deviceLocalMemoryMB);
  KENSHIN_INFO(
      "[vulkan] feature shaderBufferInt64Atomics = {}{}",
      caps.supportsShaderBufferInt64Atomics ? "supported" : "NOT supported",
      caps.enabledShaderBufferInt64Atomics ? " (enabled)" : "");
  KENSHIN_INFO("[vulkan] feature timelineSemaphore = {}{}",
               caps.supportsTimelineSemaphore ? "supported" : "NOT supported",
               caps.enabledTimelineSemaphore ? " (enabled)" : "");
  KENSHIN_INFO(
      "[vulkan] queues: graphics={} compute={}{} transfer={} ({}), {}",
      caps.graphicsFamily, caps.computeFamily,
      caps.graphicsAndComputeSameFamily ? " (same family)" : "",
      caps.transferFamily,
      caps.dedicatedTransferFamily ? "dedicated DMA" : "shared",
      params.nativeWindowHandle
          ? (caps.presentSupported ? "present=ok" : "present=NOT supported")
          : "present=unchecked (no window)");

  // GPU timestamp support: the graphics queue family's timestampValidBits
  // (period comes from device limits).
  {
    u32 familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_impl->context.physicalDevice,
                                             &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_impl->context.physicalDevice,
                                             &familyCount, families.data());
    m_impl->gpuTimestamps =
        familyCount > m_impl->context.graphicsFamily &&
        families[m_impl->context.graphicsFamily].timestampValidBits > 0;
    m_impl->timestampPeriodNs = m_impl->context.props.limits.timestampPeriod;
  }

  if (!m_impl->pipelines.init(m_impl->device.device)) return false;

  // Frame resources (command pools etc.) are device-level and needed by
  // windowless compute paths too.
  if (!m_impl->frameSync.init(m_impl->device, m_impl->gpuTimestamps)) {
    KENSHIN_ERROR("[vulkan] frame resources init failed");
    return false;
  }
  // 16MB per in-flight frame of mapped staging; M2's per-frame data
  // (culling inputs / instance tables) will slice out of this.
  if (!m_impl->staging.init(m_impl->device.allocator, 16ull * 1024 * 1024,
                            kMaxFramesInFlight)) {
    KENSHIN_ERROR("[vulkan] staging ring init failed");
    return false;
  }

  if (params.nativeWindowHandle && !m_impl->presentSupported) {
    KENSHIN_ERROR("[vulkan] graphics family does not support present");
    return false;
  }
  return true;
}

void VulkanBackend::shutdown() {
  if (!m_impl || m_shutdownDone) return;
  m_shutdownDone = true;
  // The final frame was submitted and presented but never waited on (the
  // timeline wait only happens at the NEXT beginFrame, which never comes) -
  // idle the device BEFORE ImGui teardown destroys its pipelines/samplers,
  // or validation flags them destroyed-while-in-use.
  if (m_impl->device.device != VK_NULL_HANDLE)
    KE_VK(vkDeviceWaitIdle(m_impl->device.device));
  if (m_impl->imguiInited) {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplWin32_Shutdown();
    m_impl->imguiInited = false;
  }
  // Deferred destroys are safe now (device idle); run them before the
  // objects they reference (pools, device) go away.
  for (u32 i = 0; i < kMaxFramesInFlight; ++i) {
    m_impl->frameSync.slots[i].deletionQueue.flush();
  }
  m_impl->staging.shutdown(m_impl->device.allocator);
  m_impl->descriptors.shutdown(m_impl->device.device);
  m_impl->readback.destroy(m_impl->device.allocator);
  // Frame resources first: their command buffers still reference the
  // mesh/depth objects below (validation: objects stay "in use" until
  // the referencing command buffer is freed with its pool).
  m_impl->frameSync.shutdown(m_impl->device);
  m_impl->buffers.destroyAll(m_impl->device.allocator);
  m_impl->meshes.destroyAll(m_impl->device.allocator);
  m_impl->pipelines.shutdown(m_impl->device.device);
  if (m_impl->depthImage != VK_NULL_HANDLE) {
    vkDestroyImageView(m_impl->device.device, m_impl->depthView, nullptr);
    vmaDestroyImage(m_impl->device.allocator, m_impl->depthImage,
                    m_impl->depthAlloc);
  }
  m_impl->swapchain.shutdown(m_impl->context.instance, m_impl->device.device);
  m_impl->device.shutdown();
  m_impl->context.shutdown();
}

std::unique_ptr<IRenderBackend> createVulkanBackend() {
  return std::make_unique<VulkanBackend>();
}

// ── frame loop (S1.3) ─────────────────────────────────────────

// Barrier/clear recording helpers live in CommandContext.

bool VulkanBackend::beginFrame(const FrameBeginInfo& info) {
  Impl& impl = *m_impl;
  // Headless (no window): no swapchain - record offscreen-only frames
  // (in-frame compute tests, CI). Everything else (pacing, offscreen
  // targets, per-slot recording) runs unchanged.
  const bool headless = impl.hwnd == nullptr;
  if (!headless && impl.swapchain.swapchain == VK_NULL_HANDLE) {
    if (!impl.swapchain.init(impl.context.instance, impl.context.physicalDevice,
                             impl.device.device, impl.hwnd, info.width,
                             info.height))
      return false;
  }

  FrameSlot& slot = impl.frameSync.slots[impl.frameSync.frameIndex];

  // Frame pacing: wait for the submit kMaxFramesInFlight frames ago - this
  // slot's command buffer and deletion queue are then provably free. (The
  // single timeline replaces the old per-slot fences.)
  const u64 nextTimelineValue = impl.frameSync.timelineValue + 1;
  if (nextTimelineValue > kMaxFramesInFlight)
    impl.frameSync.waitTimeline(impl.device,
                                nextTimelineValue - kMaxFramesInFlight);

  // Harvest this slot's previous timing queries (the wait above proves
  // they completed). Guarded until every slot has recorded at least one
  // frame: the in-frame vkCmdResetQueryPool is the first legal reset of a
  // fresh pool, and reading never-reset queries is a validation error.
  if (impl.gpuTimestamps && slot.timingQueries != VK_NULL_HANDLE &&
      impl.frameSync.timelineValue >= kMaxFramesInFlight) {
    u64 q[3] = {};
    if (vkGetQueryPoolResults(impl.device.device, slot.timingQueries, 0, 3,
                              sizeof(q), q, sizeof(u64),
                              VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
      m_gpuTimings.frameMs =
          f32(f64(q[2] - q[0]) * impl.timestampPeriodNs / 1e6);
      m_gpuTimings.opaqueMs =
          f32(f64(q[1] - q[0]) * impl.timestampPeriodNs / 1e6);
    }
  }

  // (Re)create per-frame offscreen targets on size/format change.
  const VkFormat targetFormat =
      headless ? VK_FORMAT_B8G8R8A8_UNORM : impl.swapchain.format;
  if (!impl.frameSync.ensureOffscreens(
          impl.device, info.width, info.height, targetFormat,
          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
              VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT))
    return false;
  ensureViewportTextures();

  // Acquire; rebuild once on out-of-date (e.g. window resize). The
  // headless path has nothing to acquire.
  VkResult acquireResult = VK_SUCCESS;
  if (!headless) {
    acquireResult =
        impl.swapchain.acquire(impl.device.device, slot.imageAvailable);
    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
      if (!impl.swapchain.recreate(impl.context.physicalDevice,
                                   impl.device.device, info.width, info.height))
        return false;
      acquireResult =
          impl.swapchain.acquire(impl.device.device, slot.imageAvailable);
    }
    if (acquireResult < 0) {
      KENSHIN_ERROR("[vulkan] acquire failed ({})",
                    static_cast<int>(acquireResult));
      return false;
    }
  }

  // Begin this slot's recording; destroys queued on this slot's last use
  // are now provably finished (the fence wait above). The staging ring's
  // slice region restarts for this frame.
  slot.deletionQueue.flush();
  impl.staging.beginFrame(impl.frameSync.frameIndex);
  impl.cmd.begin(slot.commandBuffer);
  if (impl.gpuTimestamps && slot.timingQueries != VK_NULL_HANDLE) {
    vkCmdResetQueryPool(impl.cmd.cmd(), slot.timingQueries, 0, 3);
    vkCmdWriteTimestamp2(impl.cmd.cmd(), VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                         slot.timingQueries, 0);
  }

  // Clear the offscreen target (the "rendering" of S1.3).
  CommandContext::pipelineBarrier(
      impl.cmd.cmd(),
      CommandContext::imageBarrier(
          slot.offscreen.image, VK_IMAGE_LAYOUT_UNDEFINED,
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT));
  CommandContext::clearColorImage(impl.cmd.cmd(), slot.offscreen.image,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  info.clearColor);

  // offscreen: TRANSFER_DST -> COLOR_ATTACHMENT (passes between
  // begin/end render here; endFrame moves it to SHADER_READ_ONLY).
  CommandContext::pipelineBarrier(
      impl.cmd.cmd(),
      CommandContext::imageBarrier(
          slot.offscreen.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
          VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
              VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT));

  // Depth companion target: reset to DEPTH_ATTACHMENT each frame.
  ensureViewportDepth(info.width, info.height);
  if (impl.depthImage != VK_NULL_HANDLE) {
    CommandContext::pipelineBarrier(
        impl.cmd.cmd(),
        VkImageMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
            .dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                             VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .image = impl.depthImage,
            .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
        });
  }

  if (impl.imguiInited) {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
  }

  impl.frameActive = true;
  impl.activeWidth = info.width;
  impl.activeHeight = info.height;
  return true;
}

void VulkanBackend::endFrame() {
  Impl& impl = *m_impl;
  if (!impl.frameActive) return;
  impl.frameActive = false;

  FrameSlot& slot = impl.frameSync.slots[impl.frameSync.frameIndex];
  const VkImage dstImage =
      impl.hwnd == nullptr ? VK_NULL_HANDLE
                           : impl.swapchain.images[impl.swapchain.imageIndex];

  // Close the frame's viewport scope before transitioning its target (and
  // before the UI scope opens - render passes never nest). endPass is also
  // called inside cmd.end() as a backstop; this call is the real one.
  impl.cmd.endPass();
  if (impl.gpuTimestamps && slot.timingQueries != VK_NULL_HANDLE)
    vkCmdWriteTimestamp2(impl.cmd.cmd(), VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                         slot.timingQueries, 1);

  // Headless: no swapchain to composite into - end the recording, submit
  // with only the timeline signal and advance. (The offscreen stays in
  // COLOR_ATTACHMENT layout; readbackFrame handles its own transitions.)
  if (impl.hwnd == nullptr) {
    if (!impl.cmd.end()) {
      KENSHIN_ERROR("[vulkan] vkEndCommandBuffer failed (headless)");
      return;
    }
    const VkCommandBufferSubmitInfo cmdSubmit{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = impl.cmd.cmd(),
    };
    const u64 signalValue = impl.frameSync.timelineValue + 1;
    const VkSemaphoreSubmitInfo timelineSignal{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = impl.frameSync.frameTimeline,
        .value = signalValue,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    };
    const VkSubmitInfo2 submit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &cmdSubmit,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &timelineSignal,
    };
    if (KE_VK(vkQueueSubmit2(impl.device.graphicsQueue, 1, &submit,
                             VK_NULL_HANDLE))) {
      impl.frameSync.timelineValue = signalValue;
      impl.frameSync.lastCompleted = impl.frameSync.frameIndex;
      impl.frameSync.anyCompleted = true;
      impl.frameSync.completedValue = signalValue;
      impl.frameSync.advance();
    } else {
      KENSHIN_ERROR("[vulkan] headless submit failed");
    }
    return;
  }

  // offscreen: COLOR_ATTACHMENT -> SHADER_READ_ONLY for the ImGui sample.
  CommandContext::pipelineBarrier(
      impl.cmd.cmd(),
      CommandContext::imageBarrier(slot.offscreen.image,
                                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_READ_BIT));

  ImDrawData* drawData = nullptr;
  if (impl.imguiInited) {
    ImGui::Render();
    drawData = ImGui::GetDrawData();
  }

  // swapchain image: UNDEFINED -> COLOR_ATTACHMENT (ImGui draws the
  // whole window through dynamic rendering).
  CommandContext::pipelineBarrier(
      impl.cmd.cmd(), CommandContext::imageBarrier(
                          dstImage, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                          VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                              VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT));

  const VkClearValue uiClear{};
  const VkRenderingAttachmentInfo colorAttachment{
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = impl.swapchain.imageViews[impl.swapchain.imageIndex],
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue = uiClear,
  };
  const VkRenderingInfo renderingInfo{
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {0, 0, impl.swapchain.extent.width,
                     impl.swapchain.extent.height},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &colorAttachment,
  };
  vkCmdBeginRendering(impl.cmd.cmd(), &renderingInfo);
  {
    const vkutil::ScopedDebugLabel uiLabel{impl.cmd.cmd(), "ui-composite"};
    if (drawData && drawData->CmdListsCount > 0)
      ImGui_ImplVulkan_RenderDrawData(drawData, impl.cmd.cmd());
  }
  vkCmdEndRendering(impl.cmd.cmd());

  // swapchain image: COLOR_ATTACHMENT -> PRESENT (via a capture copy
  // when a full-frame screenshot was requested).
  if (impl.readback.captureRequested()) {
    impl.readback.ensureCaptureStaging(impl.device.allocator,
                                       impl.swapchain.extent.width,
                                       impl.swapchain.extent.height);
    CommandContext::pipelineBarrier(
        impl.cmd.cmd(),
        CommandContext::imageBarrier(
            dstImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT));
    const VkBufferImageCopy captureRegion{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {impl.swapchain.extent.width,
                        impl.swapchain.extent.height, 1},
    };
    vkCmdCopyImageToBuffer(impl.cmd.cmd(), dstImage,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           impl.readback.captureStaging(), 1, &captureRegion);
    CommandContext::pipelineBarrier(
        impl.cmd.cmd(),
        CommandContext::imageBarrier(
            dstImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT));
    impl.readback.markCaptured();
  } else {
    CommandContext::pipelineBarrier(
        impl.cmd.cmd(),
        CommandContext::imageBarrier(
            dstImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT));
  }

  if (impl.gpuTimestamps && slot.timingQueries != VK_NULL_HANDLE)
    vkCmdWriteTimestamp2(impl.cmd.cmd(), VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                         slot.timingQueries, 2);

  if (!impl.cmd.end()) {
    KENSHIN_ERROR("[vulkan] vkEndCommandBuffer failed");
    return;
  }

  const VkCommandBufferSubmitInfo cmdSubmit{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = impl.cmd.cmd(),
  };
  const VkSemaphoreSubmitInfo waitSubmit{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = slot.imageAvailable,
      .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
  };
  const VkSemaphoreSubmitInfo presentSignal{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = impl.swapchain.presentSemaphores[impl.swapchain.imageIndex],
      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
  };
  // Timeline signal = frame counter (replaces the per-submit fence).
  const u64 signalValue = impl.frameSync.timelineValue + 1;
  const VkSemaphoreSubmitInfo timelineSignal{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = impl.frameSync.frameTimeline,
      .value = signalValue,
      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
  };
  const VkSemaphoreSubmitInfo signals[2] = {presentSignal, timelineSignal};
  const VkSubmitInfo2 submit{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = 1,
      .pWaitSemaphoreInfos = &waitSubmit,
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &cmdSubmit,
      .signalSemaphoreInfoCount = 2,
      .pSignalSemaphoreInfos = signals,
  };
  if (!KE_VK(vkQueueSubmit2(impl.device.graphicsQueue, 1, &submit,
                            VK_NULL_HANDLE))) {
    KENSHIN_ERROR("[vulkan] vkQueueSubmit2 failed");
    return;
  }
  impl.frameSync.timelineValue = signalValue;

  const VkSemaphore presentWait =
      impl.swapchain.presentSemaphores[impl.swapchain.imageIndex];
  const VkPresentInfoKHR present{
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &presentWait,
      .swapchainCount = 1,
      .pSwapchains = &impl.swapchain.swapchain,
      .pImageIndices = &impl.swapchain.imageIndex,
  };
  const VkResult presentResult =
      vkQueuePresentKHR(impl.device.graphicsQueue, &present);
  if (presentResult == VK_ERROR_OUT_OF_DATE_KHR ||
      presentResult == VK_SUBOPTIMAL_KHR) {
    // Rebuild immediately; nothing else is pending (fences drained above).
    impl.swapchain.recreate(impl.context.physicalDevice, impl.device.device,
                            impl.activeWidth, impl.activeHeight);
  }

  impl.frameSync.lastCompleted = impl.frameSync.frameIndex;
  impl.frameSync.anyCompleted = true;
  impl.frameSync.completedValue = signalValue;
  impl.frameSync.advance();
}

// One-shot compute lives in ComputeOneShot.

bool VulkanBackend::runComputeOneshot(
    const Job& job, std::vector<u8> outputs[Job::kMaxBuffers]) {
  return runComputeOneshotJob(
      m_impl->device, m_impl->frameSync.slots[0].commandPool, job, outputs);
}

// ── ImGui integration (S1.5) ──────────────────────────────────

void VulkanBackend::ensureViewportTextures() {
  if (!m_impl->imguiInited) return;
  for (u32 i = 0; i < kMaxFramesInFlight; ++i) {
    FrameSlot& slot = m_impl->frameSync.slots[i];
    if (slot.offscreen.image == VK_NULL_HANDLE ||
        slot.imguiTextureImage == slot.offscreen.image)
      continue;
    if (slot.imguiTexture) {
      ImGui_ImplVulkan_RemoveTexture(
          static_cast<VkDescriptorSet>(slot.imguiTexture));
      slot.imguiTexture = nullptr;
    }
    slot.imguiTexture = ImGui_ImplVulkan_AddTexture(
        slot.offscreen.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    slot.imguiTextureImage = slot.offscreen.image;
  }
}

bool VulkanBackend::initImGuiBackend(void* hwnd, u32 width, u32 height) {
  Impl& impl = *m_impl;
  if (impl.imguiInited || !hwnd) return impl.imguiInited;

  // The swapchain must exist so its format drives the ImGui pipeline.
  if (impl.swapchain.swapchain == VK_NULL_HANDLE &&
      !impl.swapchain.init(impl.context.instance, impl.context.physicalDevice,
                           impl.device.device, impl.hwnd, width, height))
    return false;

  // NOTE: the pool config (incl. the SAMPLED_IMAGE poolSizeCount fix) lives
  // in DescriptorCache.
  if (!impl.descriptors.initImGuiPool(impl.device.device)) return false;

  VkFormat colorFormat = impl.swapchain.format;
  ImGui_ImplVulkan_InitInfo initInfo{};
  initInfo.ApiVersion = VK_API_VERSION_1_4;
  initInfo.Instance = impl.context.instance;
  initInfo.PhysicalDevice = impl.context.physicalDevice;
  initInfo.Device = impl.device.device;
  initInfo.QueueFamily = impl.device.graphicsFamily;
  initInfo.Queue = impl.device.graphicsQueue;
  initInfo.DescriptorPool = impl.descriptors.imguiPool();
  initInfo.MinImageCount = 2;
  initInfo.ImageCount = 3;
  initInfo.UseDynamicRendering = true;
  initInfo.PipelineInfoMain.PipelineRenderingCreateInfo =
      VkPipelineRenderingCreateInfoKHR{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
          .colorAttachmentCount = 1,
          .pColorAttachmentFormats = &colorFormat,
      };
  initInfo.CheckVkResultFn = [](VkResult err) {
    if (err != VK_SUCCESS)
      KENSHIN_ERROR("[imgui][vulkan] VkResult = {}", static_cast<int>(err));
  };
  initInfo.MinAllocationSize = 1024 * 1024;

  if (!ImGui_ImplVulkan_Init(&initInfo)) {
    KENSHIN_ERROR("[imgui] ImGui_ImplVulkan_Init failed");
    impl.descriptors.shutdown(impl.device.device);
    return false;
  }
  if (!ImGui_ImplWin32_Init(hwnd)) {
    KENSHIN_ERROR("[imgui] ImGui_ImplWin32_Init failed");
    ImGui_ImplVulkan_Shutdown();
    impl.descriptors.shutdown(impl.device.device);
    return false;
  }

  impl.imguiInited = true;
  ensureViewportTextures();
  KENSHIN_INFO("[imgui] backend ready (dynamic rendering, docking UI)");
  return true;
}

void* VulkanBackend::viewportImGuiTexture() {
  FrameSlot& slot = m_impl->frameSync.slots[m_impl->frameSync.frameIndex];
  return slot.imguiTexture;
}

// ── readback forwards ─────────────────────────────────────────

bool VulkanBackend::readbackFrame(std::vector<u8>& rgbaOut, u32& width,
                                  u32& height) {
  return m_impl->readback.readViewport(m_impl->device, m_impl->frameSync,
                                       m_impl->swapchain.format, rgbaOut, width,
                                       height);
}

void VulkanBackend::requestFullFrameCapture() {
  m_impl->readback.requestCapture();
}

bool VulkanBackend::readbackFullFrame(std::vector<u8>& rgbaOut, u32& width,
                                      u32& height) {
  return m_impl->readback.readFullFrame(m_impl->device, m_impl->frameSync,
                                        m_impl->swapchain.format, rgbaOut,
                                        width, height);
}

// ── baseline mesh path (S2.2) ─────────────────────────────────

// Mesh upload lives in UploadEngine (one-shot path; P0-3 adds the ring).

// ── mesh path forwards ─────────────────────────────────────────

u32 VulkanBackend::createMeshBuffer(const MeshUpload& upload) {
  return m_impl->meshes.create(m_impl->device,
                               m_impl->frameSync.slots[0].commandPool, upload);
}

void VulkanBackend::destroyMeshBuffer(u32 mesh) {
  Impl& impl = *m_impl;
  // Deferred: the slot is invalidated now (handle unusable), the actual
  // buffer destruction runs when the current frame slot is reused - its
  // timeline wait proves no in-flight frame still binds these buffers.
  const MeshStore::MeshResource raw = impl.meshes.detach(mesh);
  if (raw.positions == VK_NULL_HANDLE && raw.indices == VK_NULL_HANDLE) return;
  VmaAllocator allocator = impl.device.allocator;
  impl.frameSync.slots[impl.frameSync.frameIndex].deletionQueue.push(
      [allocator, raw] { MeshStore::destroyRaw(allocator, raw); });
}

void VulkanBackend::ensureViewportDepth(u32 width, u32 height) {
  Impl& impl = *m_impl;
  if (impl.depthImage != VK_NULL_HANDLE && impl.depthWidth == width &&
      impl.depthHeight == height)
    return;
  if (impl.depthImage != VK_NULL_HANDLE) {
    KE_VK(vkDeviceWaitIdle(impl.device.device));
    vkDestroyImageView(impl.device.device, impl.depthView, nullptr);
    vmaDestroyImage(impl.device.allocator, impl.depthImage, impl.depthAlloc);
    impl.depthImage = VK_NULL_HANDLE;
    impl.depthView = VK_NULL_HANDLE;
  }
  const VkImageCreateInfo imageInfo{
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_D32_SFLOAT,
      .extent = {width, height, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };
  const VmaAllocationCreateInfo allocInfo{
      .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
      .preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
  };
  if (vmaCreateImage(impl.device.allocator, &imageInfo, &allocInfo,
                     &impl.depthImage, &impl.depthAlloc,
                     nullptr) != VK_SUCCESS) {
    KENSHIN_ERROR("[baseline] depth image creation failed");
    return;
  }
  const VkImageViewCreateInfo viewInfo{
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = impl.depthImage,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_D32_SFLOAT,
      .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
  };
  KE_VK(vkCreateImageView(impl.device.device, &viewInfo, nullptr,
                          &impl.depthView));
  vkutil::setName(impl.device.device, VK_OBJECT_TYPE_IMAGE,
                  (uint64_t)impl.depthImage, "viewport-depth");
  vkutil::setName(impl.device.device, VK_OBJECT_TYPE_IMAGE_VIEW,
                  (uint64_t)impl.depthView, "viewport-depth-view");
  impl.depthWidth = width;
  impl.depthHeight = height;
}

u32 VulkanBackend::createGraphicsPipeline(const GraphicsPipelineInfo& info) {
  return m_impl->pipelines.create(m_impl->device.device,
                                  m_impl->swapchain.format, info);
}

void VulkanBackend::destroyGraphicsPipeline(u32 pipeline) {
  Impl& impl = *m_impl;
  // Deferred like mesh buffers: destroy at slot reuse (fence-proven idle).
  const PipelineStore::Pipeline raw = impl.pipelines.detach(pipeline);
  if (raw.pipeline == VK_NULL_HANDLE && raw.layout == VK_NULL_HANDLE) return;
  VkDevice device = impl.device.device;
  impl.frameSync.slots[impl.frameSync.frameIndex].deletionQueue.push(
      [device, raw] {
        if (raw.pipeline != VK_NULL_HANDLE)
          vkDestroyPipeline(device, raw.pipeline, nullptr);
        if (raw.layout != VK_NULL_HANDLE)
          vkDestroyPipelineLayout(device, raw.layout, nullptr);
      });
}

// ── general buffer forwards ────────────────────────────────────

u32 VulkanBackend::createBuffer(const BufferDesc& desc) {
  return m_impl->buffers.create(m_impl->device,
                                m_impl->frameSync.slots[0].commandPool, desc);
}

void VulkanBackend::destroyBuffer(u32 buffer) {
  Impl& impl = *m_impl;
  const BufferStore::Entry raw = impl.buffers.detach(buffer);
  if (raw.buffer == VK_NULL_HANDLE) return;
  VmaAllocator allocator = impl.device.allocator;
  impl.frameSync.slots[impl.frameSync.frameIndex].deletionQueue.push(
      [allocator, raw] { BufferStore::destroyRaw(allocator, raw); });
}

bool VulkanBackend::readbackBuffer(u32 buffer, usize offset, usize bytes,
                                   std::vector<u8>& out) {
  if (!m_impl->buffers.valid(buffer)) return false;
  return BufferStore::readback(
      m_impl->device, m_impl->frameSync.slots[0].commandPool,
      {m_impl->buffers.buffer(buffer), {}, m_impl->buffers.size(buffer)},
      offset, bytes, out);
}

// ── fullscreen pass (VisBuffer decode) ─────────────────────────

u32 VulkanBackend::createFullscreenPipeline(
    const FullscreenPipelineInfo& info) {
  return m_impl->pipelines.createFullscreen(m_impl->device.device,
                                            m_impl->swapchain.format, info);
}

void VulkanBackend::drawFullscreen(u32 pipeline, const u32* bufferHandles,
                                   u32 bufferCount, const void* push,
                                   usize pushBytes) {
  Impl& impl = *m_impl;
  if (!impl.frameActive || pipeline == kInvalidPipelineHandle ||
      pipeline > impl.pipelines.fullscreenCount())
    return;
  const PipelineStore::Fullscreen& fs = impl.pipelines.fullscreen(pipeline);
  if (!fs.valid) return;

  FrameSlot& slot = impl.frameSync.slots[impl.frameSync.frameIndex];

  // Validate + collect the buffer infos FIRST - bailing out mid-way
  // would leave the pass opened with nothing drawn.
  std::vector<VkDescriptorBufferInfo> infos(bufferCount);
  for (u32 i = 0; i < bufferCount; ++i) {
    if (!impl.buffers.valid(bufferHandles[i])) return;
    infos[i] = VkDescriptorBufferInfo{
        .buffer = impl.buffers.buffer(bufferHandles[i]),
        .offset = 0,
        .range = impl.buffers.size(bufferHandles[i]),
    };
  }

  const vkutil::ScopedDebugLabel passLabel{impl.cmd.cmd(), "decode"};
  if (!impl.cmd.passOpen())
    impl.cmd.beginViewportPass(slot.offscreen.view, impl.depthView,
                               slot.offscreen.width, slot.offscreen.height);

  // Write THIS frame slot's set (the slot's prior frame has provably
  // completed at beginFrame - its buffer contents may change freely).
  const VkDescriptorSet set = fs.sets[impl.frameSync.frameIndex];
  std::vector<VkWriteDescriptorSet> writes(bufferCount);
  for (u32 i = 0; i < bufferCount; ++i)
    writes[i] = VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &infos[i],
    };
  vkUpdateDescriptorSets(impl.device.device, bufferCount, writes.data(), 0,
                         nullptr);

  vkCmdBindDescriptorSets(impl.cmd.cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                          fs.layout, 0, 1, &set, 0, nullptr);
  vkCmdBindPipeline(impl.cmd.cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                    fs.pipeline);
  vkCmdPushConstants(impl.cmd.cmd(), fs.layout,
                     VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                     0, static_cast<u32>(pushBytes), push);
  vkCmdDraw(impl.cmd.cmd(), 3, 1, 0, 0);
}

// ── in-frame compute (链入帧化) ────────────────────────────────

u32 VulkanBackend::createComputePipeline(const ComputePipelineInfo& info) {
  return m_impl->pipelines.createCompute(m_impl->device.device, info);
}

void VulkanBackend::destroyComputePipeline(u32 pipeline) {
  Impl& impl = *m_impl;
  if (pipeline == 0 || pipeline > impl.pipelines.computeCount()) return;
  PipelineStore::Compute& c =
      const_cast<PipelineStore::Compute&>(impl.pipelines.compute(pipeline));
  if (!c.valid) return;
  vkDestroyPipeline(impl.device.device, c.pipeline, nullptr);
  vkDestroyPipelineLayout(impl.device.device, c.layout, nullptr);
  vkDestroyDescriptorPool(impl.device.device, c.pool, nullptr);
  vkDestroyDescriptorSetLayout(impl.device.device, c.setLayout, nullptr);
  c = {};
}

void VulkanBackend::dispatchCompute(u32 pipeline, const u32* bufferHandles,
                                    u32 bufferCount, u32 groupsX, u32 groupsY,
                                    u32 groupsZ) {
  Impl& impl = *m_impl;
  if (!impl.frameActive || pipeline == kInvalidPipelineHandle ||
      pipeline > impl.pipelines.computeCount() || groupsX == 0)
    return;
  const PipelineStore::Compute& c = impl.pipelines.compute(pipeline);
  if (!c.valid) return;

  // Validate + collect BEFORE recording (bail-out must not leave partial
  // state; no pass is opened here so there is nothing to unwind).
  std::vector<VkDescriptorBufferInfo> infos(bufferCount);
  for (u32 i = 0; i < bufferCount; ++i) {
    if (!impl.buffers.valid(bufferHandles[i])) return;
    infos[i] = VkDescriptorBufferInfo{
        .buffer = impl.buffers.buffer(bufferHandles[i]),
        .offset = 0,
        .range = impl.buffers.size(bufferHandles[i]),
    };
  }
  // Write THIS frame slot's set (the slot's prior frame has provably
  // completed at beginFrame - the fullscreen-draw discipline).
  const VkDescriptorSet set = c.sets[impl.frameSync.frameIndex];
  std::vector<VkWriteDescriptorSet> writes(bufferCount);
  for (u32 i = 0; i < bufferCount; ++i)
    writes[i] = VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &infos[i],
    };
  vkUpdateDescriptorSets(impl.device.device, bufferCount, writes.data(), 0,
                         nullptr);

  vkCmdBindDescriptorSets(impl.cmd.cmd(), VK_PIPELINE_BIND_POINT_COMPUTE,
                          c.layout, 0, 1, &set, 0, nullptr);
  vkCmdBindPipeline(impl.cmd.cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, c.pipeline);
  vkCmdDispatch(impl.cmd.cmd(), groupsX, groupsY, groupsZ);
}

void VulkanBackend::dispatchComputeIndirect(u32 pipeline,
                                            const u32* bufferHandles,
                                            u32 bufferCount, u32 countBuffer,
                                            usize countOffset) {
  Impl& impl = *m_impl;
  if (!impl.frameActive || pipeline == kInvalidPipelineHandle ||
      pipeline > impl.pipelines.computeCount())
    return;
  const PipelineStore::Compute& c = impl.pipelines.compute(pipeline);
  if (!c.valid || !impl.buffers.valid(countBuffer)) return;

  // The dispatch-count fetch happens at the DRAW_INDIRECT stage, not in
  // a shader - a compute->compute barrier does NOT order it. Without
  // this the fetch can see stale args (zero) and silently dispatch
  // nothing while later readbacks (post-idle) look correct.
  {
    const VkBufferMemoryBarrier2 argsBarrier{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
        .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
        .buffer = impl.buffers.buffer(countBuffer),
        .offset = 0,
        .size = impl.buffers.size(countBuffer),
    };
    const VkDependencyInfo dep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &argsBarrier,
    };
    vkCmdPipelineBarrier2(impl.cmd.cmd(), &dep);
  }

  std::vector<VkDescriptorBufferInfo> infos(bufferCount);
  for (u32 i = 0; i < bufferCount; ++i) {
    if (!impl.buffers.valid(bufferHandles[i])) return;
    infos[i] = VkDescriptorBufferInfo{
        .buffer = impl.buffers.buffer(bufferHandles[i]),
        .offset = 0,
        .range = impl.buffers.size(bufferHandles[i]),
    };
  }
  const VkDescriptorSet set = c.sets[impl.frameSync.frameIndex];
  std::vector<VkWriteDescriptorSet> writes(bufferCount);
  for (u32 i = 0; i < bufferCount; ++i)
    writes[i] = VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .dstBinding = i,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &infos[i],
    };
  vkUpdateDescriptorSets(impl.device.device, bufferCount, writes.data(), 0,
                         nullptr);
  vkCmdBindDescriptorSets(impl.cmd.cmd(), VK_PIPELINE_BIND_POINT_COMPUTE,
                          c.layout, 0, 1, &set, 0, nullptr);
  vkCmdBindPipeline(impl.cmd.cmd(), VK_PIPELINE_BIND_POINT_COMPUTE, c.pipeline);
  vkCmdDispatchIndirect(impl.cmd.cmd(), impl.buffers.buffer(countBuffer),
                        static_cast<VkDeviceSize>(countOffset));
}

void* VulkanBackend::mappedPtr(u32 buffer) {
  return m_impl->buffers.valid(buffer) ? m_impl->buffers.mapped(buffer)
                                       : nullptr;
}

usize VulkanBackend::bufferSize(u32 buffer) const {
  return m_impl->buffers.valid(buffer) ? m_impl->buffers.size(buffer) : 0;
}

u32 VulkanBackend::currentFrameSlot() const {
  return m_impl->frameSync.frameIndex;
}

u32 VulkanBackend::lastCompletedFrameSlot() const {
  return m_impl->frameSync.lastCompleted;
}

void VulkanBackend::bufferBarrier(u32 buffer, bool toFragment) {
  Impl& impl = *m_impl;
  if (!impl.frameActive || !impl.buffers.valid(buffer)) return;
  const VkPipelineStageFlags2 dstStage =
      toFragment ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                 : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  const VkBufferMemoryBarrier2 barrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      .dstStageMask = dstStage,
      .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
      .buffer = impl.buffers.buffer(buffer),
      .offset = 0,
      .size = impl.buffers.size(buffer),
  };
  const VkDependencyInfo dep{
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .bufferMemoryBarrierCount = 1,
      .pBufferMemoryBarriers = &barrier,
  };
  vkCmdPipelineBarrier2(impl.cmd.cmd(), &dep);
}

void VulkanBackend::fillBuffer(u32 buffer, u32 value) {
  Impl& impl = *m_impl;
  if (!impl.frameActive || !impl.buffers.valid(buffer)) return;
  const VkBufferMemoryBarrier2 barrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      .buffer = impl.buffers.buffer(buffer),
      .offset = 0,
      .size = impl.buffers.size(buffer),
  };
  const VkDependencyInfo dep{
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .bufferMemoryBarrierCount = 1,
      .pBufferMemoryBarriers = &barrier,
  };
  vkCmdFillBuffer(impl.cmd.cmd(), impl.buffers.buffer(buffer), 0, VK_WHOLE_SIZE,
                  value);
  // Self-contained: the fill's TRANSFER write is ordered before any
  // subsequent compute read/write of the buffer (callers never need a
  // manual barrier after a fill).
  vkCmdPipelineBarrier2(impl.cmd.cmd(), &dep);
}

void VulkanBackend::draw(u32 pipeline, u32 meshHandle, const DrawData& data) {
  Impl& impl = *m_impl;
  if (!impl.frameActive || impl.depthView == VK_NULL_HANDLE) return;
  if (!impl.meshes.valid(meshHandle)) return;
  if (pipeline == kInvalidPipelineHandle || !impl.pipelines.valid(pipeline))
    return;

  FrameSlot& slot = impl.frameSync.slots[impl.frameSync.frameIndex];
  const auto& mesh = impl.meshes.mesh(meshHandle);
  const VkPipeline vkPipeline = impl.pipelines.pipeline(pipeline);
  const VkPipelineLayout vkLayout = impl.pipelines.layout(pipeline);
  const vkutil::ScopedDebugLabel passLabel{impl.cmd.cmd(), "opaque-draw"};

  // The frame's single viewport scope opens lazily on the first draw and
  // stays open; the render pass's implicit subpass ordering makes the old
  // per-draw scopes + hand-rolled inter-draw depth barrier unnecessary.
  if (!impl.cmd.passOpen())
    impl.cmd.beginViewportPass(slot.offscreen.view, impl.depthView,
                               slot.offscreen.width, slot.offscreen.height);

  // Push constants: viewProj + kd + lightDir + camPos.
  struct Push {
    f32 viewProj[16];
    f32 kd[4];
    f32 lightDir[4];
    f32 camPos[4];
  } pushData{};
  std::memcpy(pushData.viewProj, data.viewProj, sizeof(pushData.viewProj));
  std::memcpy(pushData.kd, data.kd, sizeof(pushData.kd));
  std::memcpy(pushData.lightDir, data.lightDir, sizeof(pushData.lightDir));
  std::memcpy(pushData.camPos, data.camPos, sizeof(pushData.camPos));

  vkCmdBindPipeline(impl.cmd.cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                    vkPipeline);
  const VkBuffer vertexBuffers[3] = {mesh.positions, mesh.normals, mesh.uvs};
  const VkDeviceSize vertexOffsets[3] = {0, 0, 0};
  vkCmdBindVertexBuffers(impl.cmd.cmd(), 0, 3, vertexBuffers, vertexOffsets);
  vkCmdBindIndexBuffer(impl.cmd.cmd(), mesh.indices, 0, VK_INDEX_TYPE_UINT32);
  vkCmdPushConstants(impl.cmd.cmd(), vkLayout,
                     VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                     0, sizeof(pushData), &pushData);
  vkCmdDrawIndexed(impl.cmd.cmd(), mesh.indexCount, 1, 0, 0, 0);
}

}  // namespace backend

KENSHIN_NS_END
