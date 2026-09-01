#include "kenshin/backend/VulkanBackend.h"

#include "vulkan/Vk.h"
#include "vulkan/CommandContext.h"
#include "vulkan/DescriptorCache.h"
#include "vulkan/UploadEngine.h"

#include "kenshin/core/Log.h"
#include "kenshin/manager/ShaderManager.h"
#include "vulkan/FrameSync.h"
#include "vulkan/Swapchain.h"
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
#include <fstream>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

KENSHIN_NS_BEGIN

namespace backend {

// SOA mesh GPU buffers (backend-internal; opaque to the public header).
struct VulkanBackend::MeshResourceOpaque {
  VkBuffer positions = VK_NULL_HANDLE;
  VmaAllocation positionsAlloc = VK_NULL_HANDLE;
  VkBuffer normals = VK_NULL_HANDLE;
  VmaAllocation normalsAlloc = VK_NULL_HANDLE;
  VkBuffer uvs = VK_NULL_HANDLE;
  VmaAllocation uvsAlloc = VK_NULL_HANDLE;
  VkBuffer indices = VK_NULL_HANDLE;
  VmaAllocation indicesAlloc = VK_NULL_HANDLE;
  u32 vertexCount = 0;
  u32 indexCount = 0;
};

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

  // Full-frame capture (UI included; dev/visual checks only)
  bool captureFull = false;
  bool fullCaptured = false;
  VkBuffer captureStaging = VK_NULL_HANDLE;
  VmaAllocation captureAlloc = VK_NULL_HANDLE;
  u32 captureWidth = 0, captureHeight = 0;

  // Baseline mesh path (S2.2)
  using MeshResource = VulkanBackend::MeshResourceOpaque;
  std::vector<MeshResource> meshes;  // slot = handle - 1

  VkImage depthImage = VK_NULL_HANDLE;
  VmaAllocation depthAlloc = VK_NULL_HANDLE;
  VkImageView depthView = VK_NULL_HANDLE;
  u32 depthWidth = 0, depthHeight = 0;

  // GPU timestamps + pipeline disk cache
  bool gpuTimestamps = false;
  f32 timestampPeriodNs = 0.0f;
  VkPipelineCache pipelineCache = VK_NULL_HANDLE;

  // Graphics pipelines created through createGraphicsPipeline
  // (renderer layer defines state, this layer translates). Handle =
  // index + 1; a freed slot keeps its place with null handles.
  struct Pipeline {
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
  };
  std::vector<Pipeline> pipelines;
};

void VulkanBackend::destroyMeshResourceInternal(
    VulkanBackend::MeshResourceOpaque& raw) {
  Impl& impl = *m_impl;
  if (raw.positions != VK_NULL_HANDLE)
    vmaDestroyBuffer(impl.device.allocator, raw.positions, raw.positionsAlloc);
  if (raw.normals != VK_NULL_HANDLE)
    vmaDestroyBuffer(impl.device.allocator, raw.normals, raw.normalsAlloc);
  if (raw.uvs != VK_NULL_HANDLE)
    vmaDestroyBuffer(impl.device.allocator, raw.uvs, raw.uvsAlloc);
  if (raw.indices != VK_NULL_HANDLE)
    vmaDestroyBuffer(impl.device.allocator, raw.indices, raw.indicesAlloc);
  raw = {};
}

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

  // Pipeline disk cache: loads prior driver compilations, saved at
  // shutdown next to the exe.
  {
    std::vector<char> cacheData;
    wchar_t exePath[MAX_PATH]{};
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    const auto cachePath = std::filesystem::path(exePath).parent_path() /
                           "kenshin_pipeline.cache";
    if (std::filesystem::exists(cachePath)) {
      std::ifstream in(cachePath, std::ios::binary);
      in.seekg(0, std::ios::end);
      const std::streamsize size = in.tellg();
      in.seekg(0, std::ios::beg);
      if (size > 0) {
        cacheData.resize(static_cast<usize>(size));
        in.read(cacheData.data(), size);
      }
    }
    const VkPipelineCacheCreateInfo cacheInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        .initialDataSize = cacheData.size(),
        .pInitialData = cacheData.data(),
    };
    if (!KE_VK(vkCreatePipelineCache(m_impl->device.device, &cacheInfo,
                                     nullptr, &m_impl->pipelineCache)))
      return false;
  }

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
  if (m_impl->imguiInited) {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplWin32_Shutdown();
    m_impl->imguiInited = false;
  }
  if (m_impl->device.device != VK_NULL_HANDLE)
    KE_VK(vkDeviceWaitIdle(m_impl->device.device));
  // Deferred destroys are safe now (device idle); run them before the
  // objects they reference (pools, device) go away.
  for (u32 i = 0; i < kMaxFramesInFlight; ++i)
    m_impl->frameSync.slots[i].deletionQueue.flush();
  m_impl->staging.shutdown(m_impl->device.allocator);
  m_impl->descriptors.shutdown(m_impl->device.device);
  if (m_impl->captureStaging != VK_NULL_HANDLE) {
    vmaDestroyBuffer(m_impl->device.allocator, m_impl->captureStaging,
                     m_impl->captureAlloc);
    m_impl->captureStaging = VK_NULL_HANDLE;
  }
  // Frame resources first: their command buffers still reference the
  // mesh/depth objects below (validation: objects stay "in use" until
  // the referencing command buffer is freed with its pool).
  m_impl->frameSync.shutdown(m_impl->device);
  for (auto& mesh : m_impl->meshes) destroyMeshResourceInternal(mesh);
  m_impl->meshes.clear();
  for (const auto& p : m_impl->pipelines) {
    if (p.pipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(m_impl->device.device, p.pipeline, nullptr);
    if (p.layout != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(m_impl->device.device, p.layout, nullptr);
  }
  m_impl->pipelines.clear();
  if (m_impl->depthImage != VK_NULL_HANDLE) {
    vkDestroyImageView(m_impl->device.device, m_impl->depthView, nullptr);
    vmaDestroyImage(m_impl->device.allocator, m_impl->depthImage,
                    m_impl->depthAlloc);
  }
  m_impl->swapchain.shutdown(m_impl->context.instance, m_impl->device.device);
  // Persist the pipeline cache (this session's driver compilations) next
  // to the exe; rebuilt pipelines hit the cache on the next launch.
  if (m_impl->pipelineCache != VK_NULL_HANDLE) {
    size_t bytes = 0;
    if (KE_VK(vkGetPipelineCacheData(m_impl->device.device,
                                     m_impl->pipelineCache, &bytes,
                                     nullptr)) &&
        bytes > 0) {
      std::vector<char> data(bytes);
      if (KE_VK(vkGetPipelineCacheData(m_impl->device.device,
                                       m_impl->pipelineCache, &bytes,
                                       data.data()))) {
        wchar_t exePath[MAX_PATH]{};
        GetModuleFileNameW(nullptr, exePath, MAX_PATH);
        const auto cachePath =
            std::filesystem::path(exePath).parent_path() /
            "kenshin_pipeline.cache";
        std::ofstream out(cachePath, std::ios::binary);
        out.write(data.data(), static_cast<std::streamsize>(bytes));
      }
    }
    vkDestroyPipelineCache(m_impl->device.device, m_impl->pipelineCache,
                           nullptr);
    m_impl->pipelineCache = VK_NULL_HANDLE;
  }
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
  if (impl.swapchain.swapchain == VK_NULL_HANDLE) {
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
  if (!impl.frameSync.ensureOffscreens(
          impl.device, info.width, info.height, impl.swapchain.format,
          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
              VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT))
    return false;
  ensureViewportTextures();

  // Acquire; rebuild once on out-of-date (e.g. window resize).
  VkResult acquireResult =
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
  CommandContext::pipelineBarrier(impl.cmd.cmd(),
                CommandContext::imageBarrier(slot.offscreen.image, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                             VK_ACCESS_2_TRANSFER_WRITE_BIT));
  CommandContext::clearColorImage(impl.cmd.cmd(), slot.offscreen.image,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  info.clearColor);

  // offscreen: TRANSFER_DST -> COLOR_ATTACHMENT (passes between
  // begin/end render here; endFrame moves it to SHADER_READ_ONLY).
  CommandContext::pipelineBarrier(
      impl.cmd.cmd(),
      CommandContext::imageBarrier(slot.offscreen.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
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
  const VkImage dstImage = impl.swapchain.images[impl.swapchain.imageIndex];

  // Close the frame's viewport scope before transitioning its target (and
  // before the UI scope opens - render passes never nest). endPass is also
  // called inside cmd.end() as a backstop; this call is the real one.
  impl.cmd.endPass();
  if (impl.gpuTimestamps && slot.timingQueries != VK_NULL_HANDLE)
    vkCmdWriteTimestamp2(impl.cmd.cmd(), VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                        slot.timingQueries, 1);

  // offscreen: COLOR_ATTACHMENT -> SHADER_READ_ONLY for the ImGui sample.
  CommandContext::pipelineBarrier(impl.cmd.cmd(),
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
  CommandContext::pipelineBarrier(impl.cmd.cmd(),
                CommandContext::imageBarrier(dstImage, VK_IMAGE_LAYOUT_UNDEFINED,
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
  if (impl.captureFull) {
    ensureCaptureStaging(impl.swapchain.extent.width,
                         impl.swapchain.extent.height);
    CommandContext::pipelineBarrier(
        impl.cmd.cmd(),
        CommandContext::imageBarrier(dstImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     VK_ACCESS_2_TRANSFER_READ_BIT));
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
                           impl.captureStaging, 1, &captureRegion);
    CommandContext::pipelineBarrier(impl.cmd.cmd(),
                  CommandContext::imageBarrier(dstImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                               VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               VK_ACCESS_2_MEMORY_READ_BIT));
    impl.captureFull = false;
    impl.fullCaptured = true;
  } else {
    CommandContext::pipelineBarrier(
        impl.cmd.cmd(),
        CommandContext::imageBarrier(dstImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                     VK_ACCESS_2_MEMORY_READ_BIT));
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

bool VulkanBackend::readbackFrame(std::vector<u8>& rgbaOut, u32& width,
                                  u32& height) {
  Impl& impl = *m_impl;
  if (!impl.frameSync.anyCompleted) return false;

  const FrameSlot& slot = impl.frameSync.slots[impl.frameSync.lastCompleted];
  width = slot.offscreen.width;
  height = slot.offscreen.height;

  impl.frameSync.waitTimeline(impl.device, impl.frameSync.completedValue);

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
  if (vmaCreateBuffer(impl.device.allocator, &bufferInfo, &stagingAlloc,
                      &staging, &stagingAllocHandle, nullptr) != VK_SUCCESS)
    return false;
  vmaMapMemory(impl.device.allocator, stagingAllocHandle, &mapped);

  // One-shot copy on the graphics queue (slot 0's pool; drained by the fence wait).
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  const VkCommandBufferAllocateInfo allocInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = impl.frameSync.slots[0].commandPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };
  KE_VK(vkAllocateCommandBuffers(impl.device.device, &allocInfo, &cmd));
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
  KE_VK(vkQueueSubmit2(impl.device.graphicsQueue, 1, &submit, VK_NULL_HANDLE));
  KE_VK(vkQueueWaitIdle(impl.device.graphicsQueue));
  vkFreeCommandBuffers(impl.device.device, impl.frameSync.slots[0].commandPool,
                       1, &cmd);

  // Swizzle BGRA -> RGBA when the surface format is BGRA-ordered.
  rgbaOut.resize(static_cast<usize>(width) * height * 4);
  const bool bgra = impl.swapchain.format == VK_FORMAT_B8G8R8A8_UNORM;
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

  vmaUnmapMemory(impl.device.allocator, stagingAllocHandle);
  vmaDestroyBuffer(impl.device.allocator, staging, stagingAllocHandle);
  return true;
}

// ── one-shot compute (S1.4) ───────────────────────────────────

bool VulkanBackend::runComputeOneshot(
    const Job& job, std::vector<u8> outputs[Job::kMaxBuffers]) {
  Impl& impl = *m_impl;
  VkDevice device = impl.device.device;
  if (!device || !job.spirv || job.spirvBytes == 0) return false;

  // Count used buffers.
  u32 bufferCount = 0;
  for (u32 i = 0; i < Job::kMaxBuffers; ++i)
    if (job.buffers[i].size > 0) bufferCount = i + 1;

  // Shader module + descriptor set layout + pipeline layout + pipeline.
  const VkShaderModuleCreateInfo moduleInfo{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = job.spirvBytes,
      .pCode = job.spirv,
  };
  VkShaderModule module = VK_NULL_HANDLE;
  if (!KE_VK(vkCreateShaderModule(device, &moduleInfo, nullptr, &module))) {
    KENSHIN_ERROR("[compute] shader module creation failed");
    return false;
  }

  // Descriptor layout from shader reflection: the compiled SPIR-V is the
  // authority on bindings (spirv-reflect via ShaderManager), replacing a
  // hand-maintained copy. Job convention: buffers[i] feeds binding i; if
  // reflection is unavailable or disagrees with that, fall back to it.
  std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
  std::vector<manager::ReflectedBinding> reflected;
  const std::vector<u32> spirvWords(job.spirv,
                                    job.spirv + job.spirvBytes / sizeof(u32));
  bool reflectedOk = false;
  if (manager::reflectDescriptorBindings(spirvWords, reflected)) {
    u32 storageCount = 0;
    for (const auto& r : reflected)
      if (r.type == manager::ReflectedBinding::Type::StorageBuffer &&
          r.set == 0)
        ++storageCount;
    reflectedOk = storageCount == bufferCount;
  }
  if (reflectedOk) {
    for (const auto& r : reflected) {
      if (r.type != manager::ReflectedBinding::Type::StorageBuffer) continue;
      layoutBindings.push_back(VkDescriptorSetLayoutBinding{
          .binding = r.binding,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      });
    }
  } else {
    for (u32 i = 0; i < bufferCount; ++i) {
      layoutBindings.push_back(VkDescriptorSetLayoutBinding{
          .binding = i,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      });
    }
  }
  const VkDescriptorSetLayoutCreateInfo setLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = static_cast<u32>(layoutBindings.size()),
      .pBindings = layoutBindings.data(),
  };
  VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
  KE_VK(vkCreateDescriptorSetLayout(device, &setLayoutInfo, nullptr, &setLayout));

  const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = bufferCount ? 1u : 0u,
      .pSetLayouts = &setLayout,
  };
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  KE_VK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout));

  const VkComputePipelineCreateInfo pipelineInfo{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage =
          VkPipelineShaderStageCreateInfo{
              .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
              .stage = VK_SHADER_STAGE_COMPUTE_BIT,
              .module = module,
              .pName = job.entryPoint,
          },
      .layout = pipelineLayout,
  };
  VkPipeline pipeline = VK_NULL_HANDLE;
  if (!KE_VK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                               nullptr, &pipeline))) {
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
  for (u32 i = 0; i < bufferCount && ok; ++i) {
    const VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = static_cast<VkDeviceSize>(job.buffers[i].size),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo allocInfo{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    };
    VmaAllocationInfo allocInfoOut{};
    if (vmaCreateBuffer(impl.device.allocator, &bufferInfo, &allocInfo,
                        &buffers[i], &allocations[i],
                        &allocInfoOut) != VK_SUCCESS) {
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
  if (ok && bufferCount > 0) {
    const VkDescriptorPoolSize poolSize{
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = Job::kMaxBuffers,
    };
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };
    KE_VK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool));

    const VkDescriptorSetAllocateInfo setInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &setLayout,
    };
    if (!KE_VK(vkAllocateDescriptorSets(device, &setInfo, &set))) {
      KENSHIN_ERROR("[compute] descriptor set allocation failed");
      ok = false;
    }
  }
  if (ok && bufferCount > 0) {
    VkDescriptorBufferInfo bufferInfos[Job::kMaxBuffers] = {};
    VkWriteDescriptorSet writes[Job::kMaxBuffers] = {};
    for (u32 i = 0; i < bufferCount; ++i) {
      bufferInfos[i] = VkDescriptorBufferInfo{
          .buffer = buffers[i],
          .range = VK_WHOLE_SIZE,
      };
      writes[i] = VkWriteDescriptorSet{
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = set,
          .dstBinding = layoutBindings[i].binding,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &bufferInfos[i],
      };
    }
    vkUpdateDescriptorSets(device, bufferCount, writes, 0, nullptr);
  }

  // Record + submit the dispatch, then wait (host-coherent: submit and
  // queueWaitIdle provide the necessary host sync boundaries).
  if (ok) {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    const VkCommandBufferAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = impl.frameSync.slots[0].commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    KE_VK(vkAllocateCommandBuffers(device, &allocInfo, &cmd));
    const VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    KE_VK(vkBeginCommandBuffer(cmd, &beginInfo));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    if (bufferCount > 0)
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdDispatch(cmd, job.groupCountX, 1, 1);
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
    KE_VK(vkQueueSubmit2(impl.device.graphicsQueue, 1, &submit, VK_NULL_HANDLE));
    KE_VK(vkQueueWaitIdle(impl.device.graphicsQueue));
    vkFreeCommandBuffers(device, impl.frameSync.slots[0].commandPool, 1, &cmd);
  }

  // Readback.
  if (ok) {
    for (u32 i = 0; i < bufferCount; ++i) {
      if (!job.buffers[i].readBack || !mapped[i]) continue;
      outputs[i].resize(job.buffers[i].size);
      std::memcpy(outputs[i].data(), mapped[i], job.buffers[i].size);
    }
  }

  // Teardown.
  if (pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, pool, nullptr);
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

void VulkanBackend::requestFullFrameCapture() { m_impl->captureFull = true; }

bool VulkanBackend::readbackFullFrame(std::vector<u8>& rgbaOut, u32& width,
                                      u32& height) {
  Impl& impl = *m_impl;
  if (!impl.fullCaptured || impl.captureStaging == VK_NULL_HANDLE) return false;
  impl.frameSync.waitTimeline(impl.device, impl.frameSync.completedValue);

  width = impl.captureWidth;
  height = impl.captureHeight;
  void* mapped = nullptr;
  if (vmaMapMemory(impl.device.allocator, impl.captureAlloc, &mapped) !=
      VK_SUCCESS)
    return false;

  rgbaOut.resize(static_cast<usize>(width) * height * 4);
  const u8* src = static_cast<const u8*>(mapped);
  const bool bgra = impl.swapchain.format == VK_FORMAT_B8G8R8A8_UNORM;
  if (bgra) {
    for (usize i = 0; i < rgbaOut.size(); i += 4) {
      rgbaOut[i + 0] = src[i + 2];
      rgbaOut[i + 1] = src[i + 1];
      rgbaOut[i + 2] = src[i + 0];
      rgbaOut[i + 3] = src[i + 3];
    }
  } else
    std::memcpy(rgbaOut.data(), src, rgbaOut.size());
  vmaUnmapMemory(impl.device.allocator, impl.captureAlloc);
  impl.fullCaptured = false;
  return true;
}

void VulkanBackend::ensureCaptureStaging(u32 width, u32 height) {
  Impl& impl = *m_impl;
  if (impl.captureStaging != VK_NULL_HANDLE && impl.captureWidth == width &&
      impl.captureHeight == height)
    return;
  if (impl.captureStaging != VK_NULL_HANDLE)
    vmaDestroyBuffer(impl.device.allocator, impl.captureStaging,
                     impl.captureAlloc);
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
  vmaCreateBuffer(impl.device.allocator, &bufferInfo, &allocInfo,
                  &impl.captureStaging, &impl.captureAlloc, nullptr);
  impl.captureWidth = width;
  impl.captureHeight = height;
}

// ── baseline mesh path (S2.2) ─────────────────────────────────

// Mesh upload lives in UploadEngine (one-shot path; P0-3 adds the ring).

u32 VulkanBackend::createMeshBuffer(const MeshUpload& upload) {
  Impl& impl = *m_impl;
  if (!upload.positions || !upload.indices || upload.vertexCount == 0 ||
      upload.indexCount == 0)
    return kInvalidMeshHandle;

  Impl::MeshResource mesh;
  const auto uploadF32 = [&](const f32* data, usize floatCount,
                             VkBufferUsageFlags usage, VkBuffer& buffer,
                             VmaAllocation& alloc) {
    return uploadDeviceBuffer(
        impl.device.allocator, impl.device.device, impl.device.graphicsQueue,
        impl.frameSync.slots[0].commandPool, data,
        VkDeviceSize(floatCount) * sizeof(f32), usage, buffer, alloc);
  };
  const bool ok =
      uploadF32(upload.positions, usize(upload.vertexCount) * 3,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.positions,
                mesh.positionsAlloc) &&
      (!upload.normals ||
       uploadF32(upload.normals, usize(upload.vertexCount) * 3,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.normals,
                 mesh.normalsAlloc)) &&
      (!upload.uvs ||
       uploadF32(upload.uvs, usize(upload.vertexCount) * 2,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.uvs, mesh.uvsAlloc)) &&
      uploadDeviceBuffer(
          impl.device.allocator, impl.device.device, impl.device.graphicsQueue,
          impl.frameSync.slots[0].commandPool, upload.indices,
          VkDeviceSize(upload.indexCount) * sizeof(u32),
          VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh.indices, mesh.indicesAlloc);
  if (!ok) {
    destroyMeshResourceInternal(mesh);
    KENSHIN_ERROR("[baseline] mesh buffer upload failed");
    return kInvalidMeshHandle;
  }
  mesh.vertexCount = upload.vertexCount;
  mesh.indexCount = upload.indexCount;
  impl.meshes.push_back(mesh);
  const u32 handle = static_cast<u32>(impl.meshes.size());
  char name[40];
  snprintf(name, sizeof(name), "mesh-%u-positions", handle);
  vkutil::setName(impl.device.device, VK_OBJECT_TYPE_BUFFER,
                  (uint64_t)impl.meshes[handle - 1].positions, name);
  snprintf(name, sizeof(name), "mesh-%u-indices", handle);
  vkutil::setName(impl.device.device, VK_OBJECT_TYPE_BUFFER,
                  (uint64_t)impl.meshes[handle - 1].indices, name);
  return handle;
}

void VulkanBackend::destroyMeshBuffer(u32 mesh) {
  Impl& impl = *m_impl;
  if (mesh == kInvalidMeshHandle || mesh == 0 || mesh > impl.meshes.size())
    return;
  // Deferred: the slot is invalidated now (handle unusable), the actual
  // buffer destruction runs when the current frame slot is reused - its
  // fence wait proves no in-flight frame still binds these buffers.
  Impl::MeshResource& res = impl.meshes[mesh - 1];
  const Impl::MeshResource raw = res;
  res = {};
  VmaAllocator allocator = impl.device.allocator;
  impl.frameSync.slots[impl.frameSync.frameIndex].deletionQueue.push(
      [allocator, raw] {
        if (raw.positions != VK_NULL_HANDLE)
          vmaDestroyBuffer(allocator, raw.positions, raw.positionsAlloc);
        if (raw.normals != VK_NULL_HANDLE)
          vmaDestroyBuffer(allocator, raw.normals, raw.normalsAlloc);
        if (raw.uvs != VK_NULL_HANDLE)
          vmaDestroyBuffer(allocator, raw.uvs, raw.uvsAlloc);
        if (raw.indices != VK_NULL_HANDLE)
          vmaDestroyBuffer(allocator, raw.indices, raw.indicesAlloc);
      });
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
  KE_VK(vkCreateImageView(impl.device.device, &viewInfo, nullptr, &impl.depthView));
  vkutil::setName(impl.device.device, VK_OBJECT_TYPE_IMAGE, (uint64_t)impl.depthImage,
                  "viewport-depth");
  vkutil::setName(impl.device.device, VK_OBJECT_TYPE_IMAGE_VIEW,
                  (uint64_t)impl.depthView, "viewport-depth-view");
  impl.depthWidth = width;
  impl.depthHeight = height;
}

namespace {
VkCompareOp toVk(IRenderBackend::CompareOp op) {
  using Op = IRenderBackend::CompareOp;
  switch (op) {
    case Op::Never:
      return VK_COMPARE_OP_NEVER;
    case Op::Less:
      return VK_COMPARE_OP_LESS;
    case Op::Equal:
      return VK_COMPARE_OP_EQUAL;
    case Op::LessOrEqual:
      return VK_COMPARE_OP_LESS_OR_EQUAL;
    case Op::Greater:
      return VK_COMPARE_OP_GREATER;
    case Op::NotEqual:
      return VK_COMPARE_OP_NOT_EQUAL;
    case Op::GreaterOrEqual:
      return VK_COMPARE_OP_GREATER_OR_EQUAL;
    default:
      return VK_COMPARE_OP_ALWAYS;
  }
}
VkCullModeFlags toVk(IRenderBackend::CullMode mode) {
  using M = IRenderBackend::CullMode;
  switch (mode) {
    case M::None:
      return VK_CULL_MODE_NONE;
    case M::Front:
      return VK_CULL_MODE_FRONT_BIT;
    default:
      return VK_CULL_MODE_BACK_BIT;
  }
}
VkFrontFace toVk(IRenderBackend::FrontFace face) {
  return face == IRenderBackend::FrontFace::Clockwise
             ? VK_FRONT_FACE_CLOCKWISE
             : VK_FRONT_FACE_COUNTER_CLOCKWISE;
}
}  // namespace

u32 VulkanBackend::createGraphicsPipeline(const GraphicsPipelineInfo& info) {
  Impl& impl = *m_impl;
  if (info.vsSpirv == nullptr || info.vsBytes == 0 || info.fsSpirv == nullptr ||
      info.fsBytes == 0)
    return kInvalidPipelineHandle;

  const VkPushConstantRange pushRange{
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      .offset = 0,
      .size =
          112,  // mat4 viewProj (64) + kd (16) + lightDir (16) + camPos (16)
  };
  const VkPipelineLayoutCreateInfo layoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pushRange,
  };
  VkPipelineLayout layout = VK_NULL_HANDLE;
  KE_VK(vkCreatePipelineLayout(impl.device.device, &layoutInfo, nullptr, &layout));

  const VkShaderModuleCreateInfo vsInfo{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = info.vsBytes,
      .pCode = info.vsSpirv,
  };
  VkShaderModule vs = VK_NULL_HANDLE;
  if (!KE_VK(vkCreateShaderModule(impl.device.device, &vsInfo, nullptr, &vs))) {
    vkDestroyPipelineLayout(impl.device.device, layout, nullptr);
    return kInvalidPipelineHandle;
  }
  const VkShaderModuleCreateInfo fsInfo{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = info.fsBytes,
      .pCode = info.fsSpirv,
  };
  VkShaderModule fs = VK_NULL_HANDLE;
  if (!KE_VK(vkCreateShaderModule(impl.device.device, &fsInfo, nullptr, &fs))) {
    vkDestroyShaderModule(impl.device.device, vs, nullptr);
    vkDestroyPipelineLayout(impl.device.device, layout, nullptr);
    return kInvalidPipelineHandle;
  }

  const VkPipelineShaderStageCreateInfo stages[2] = {
      VkPipelineShaderStageCreateInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT,
          .module = vs,
          .pName =
              "main",  // Slang emits OpEntryPoint "main" regardless of function name
      },
      VkPipelineShaderStageCreateInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
          .module = fs,
          .pName = "main",
      },
  };

  // SOA vertex input: three bindings (positions / normals / uvs).
  const VkVertexInputBindingDescription bindings[3] = {
      {0, 12, VK_VERTEX_INPUT_RATE_VERTEX},
      {1, 12, VK_VERTEX_INPUT_RATE_VERTEX},
      {2, 8, VK_VERTEX_INPUT_RATE_VERTEX},
  };
  const VkVertexInputAttributeDescription attributes[3] = {
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
      {1, 1, VK_FORMAT_R32G32B32_SFLOAT, 0},
      {2, 2, VK_FORMAT_R32G32_SFLOAT, 0},
  };
  const VkFormat colorFormat = impl.swapchain.format;
  const VkPipelineRenderingCreateInfoKHR renderingInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &colorFormat,
      .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT,
  };
  const VkDynamicState dynamicStates[2] = {VK_DYNAMIC_STATE_VIEWPORT,
                                           VK_DYNAMIC_STATE_SCISSOR};
  const VkPipelineDynamicStateCreateInfo dynamicState{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2,
      .pDynamicStates = dynamicStates,
  };

  // State structs need named locals (designated initializers of
  // temporaries can't be address-taken).
  const VkPipelineVertexInputStateCreateInfo vertexInput{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 3,
      .pVertexBindingDescriptions = bindings,
      .vertexAttributeDescriptionCount = 3,
      .pVertexAttributeDescriptions = attributes,
  };
  const VkPipelineInputAssemblyStateCreateInfo inputAssembly{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
  };
  const VkPipelineViewportStateCreateInfo viewportState{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .scissorCount = 1,
  };
  const VkPipelineRasterizationStateCreateInfo rasterization{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = toVk(info.cullMode),
      .frontFace = toVk(info.frontFace),
      .lineWidth = 1.0f,
  };
  const VkPipelineMultisampleStateCreateInfo multisample{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
  };
  const VkPipelineDepthStencilStateCreateInfo depthStencil{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = info.depthTest ? VK_TRUE : VK_FALSE,
      .depthWriteEnable = info.depthWrite ? VK_TRUE : VK_FALSE,
      .depthCompareOp = toVk(info.depthCompare),
      .minDepthBounds = 0.0f,
      .maxDepthBounds = 1.0f,
  };
  const VkPipelineColorBlendAttachmentState blendAttachment{
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
  };
  const VkPipelineColorBlendStateCreateInfo colorBlend{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &blendAttachment,
  };

  const VkGraphicsPipelineCreateInfo createInfo{
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &renderingInfo,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vertexInput,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState = &viewportState,
      .pRasterizationState = &rasterization,
      .pMultisampleState = &multisample,
      .pDepthStencilState = &depthStencil,
      .pColorBlendState = &colorBlend,
      .pDynamicState = &dynamicState,
      .layout = layout,
  };
  VkPipeline pipeline = VK_NULL_HANDLE;
  if (!KE_VK(vkCreateGraphicsPipelines(impl.device.device, impl.pipelineCache,
                                1, &createInfo, nullptr,
                                &pipeline))) {
    KENSHIN_ERROR("[vulkan] graphics pipeline creation failed");
    vkDestroyShaderModule(impl.device.device, vs, nullptr);
    vkDestroyShaderModule(impl.device.device, fs, nullptr);
    vkDestroyPipelineLayout(impl.device.device, layout, nullptr);
    return kInvalidPipelineHandle;
  }
  vkDestroyShaderModule(impl.device.device, vs, nullptr);
  vkDestroyShaderModule(impl.device.device, fs, nullptr);

  // Reuse a freed slot when one exists; handle = index + 1.
  u32 handle = 0;
  for (usize i = 0; i < impl.pipelines.size(); ++i) {
    if (impl.pipelines[i].pipeline == VK_NULL_HANDLE) {
      impl.pipelines[i] = {layout, pipeline};
      handle = static_cast<u32>(i + 1);
      break;
    }
  }
  if (handle == 0) {
    impl.pipelines.push_back({layout, pipeline});
    handle = static_cast<u32>(impl.pipelines.size());
  }
  char name[32];
  snprintf(name, sizeof(name), "graphics-pipeline-%u", handle);
  vkutil::setName(impl.device.device, VK_OBJECT_TYPE_PIPELINE,
                  (uint64_t)pipeline, name);
  return handle;
}

void VulkanBackend::destroyGraphicsPipeline(u32 pipeline) {
  Impl& impl = *m_impl;
  if (pipeline == 0 || pipeline > impl.pipelines.size()) return;
  Impl::Pipeline& p = impl.pipelines[pipeline - 1];
  if (p.pipeline == VK_NULL_HANDLE && p.layout == VK_NULL_HANDLE) return;
  // Deferred like mesh buffers: destroy at slot reuse (fence-proven idle).
  const Impl::Pipeline raw = p;
  p = {};
  VkDevice device = impl.device.device;
  impl.frameSync.slots[impl.frameSync.frameIndex].deletionQueue.push(
      [device, raw] {
        if (raw.pipeline != VK_NULL_HANDLE)
          vkDestroyPipeline(device, raw.pipeline, nullptr);
        if (raw.layout != VK_NULL_HANDLE)
          vkDestroyPipelineLayout(device, raw.layout, nullptr);
      });
}

void VulkanBackend::draw(u32 pipeline, u32 meshHandle, const DrawData& data) {
  Impl& impl = *m_impl;
  if (!impl.frameActive || impl.depthView == VK_NULL_HANDLE) return;
  if (meshHandle == kInvalidMeshHandle || meshHandle == 0 ||
      meshHandle > impl.meshes.size())
    return;
  if (pipeline == kInvalidPipelineHandle || pipeline == 0 ||
      pipeline > impl.pipelines.size() ||
      impl.pipelines[pipeline - 1].pipeline == VK_NULL_HANDLE)
    return;

  FrameSlot& slot = impl.frameSync.slots[impl.frameSync.frameIndex];
  const Impl::MeshResource& mesh = impl.meshes[meshHandle - 1];
  const Impl::Pipeline& pipe = impl.pipelines[pipeline - 1];
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
                    pipe.pipeline);
  const VkBuffer vertexBuffers[3] = {mesh.positions, mesh.normals, mesh.uvs};
  const VkDeviceSize vertexOffsets[3] = {0, 0, 0};
  vkCmdBindVertexBuffers(impl.cmd.cmd(), 0, 3, vertexBuffers, vertexOffsets);
  vkCmdBindIndexBuffer(impl.cmd.cmd(), mesh.indices, 0, VK_INDEX_TYPE_UINT32);
  vkCmdPushConstants(impl.cmd.cmd(), pipe.layout,
                     VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                     0, sizeof(pushData), &pushData);
  vkCmdDrawIndexed(impl.cmd.cmd(), mesh.indexCount, 1, 0, 0, 0);
}

}  // namespace backend

KENSHIN_NS_END
