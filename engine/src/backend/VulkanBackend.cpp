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

#include <algorithm>
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

  // M4.3 async compute: when set, dispatch/barrier/fill calls record
  // into this command buffer instead of the frame's graphics one (the
  // chain's cull/raster segments). chainValueA/B are this frame's
  // compute-timeline values for the graphics submission's waits.
  VkCommandBuffer computeOverride = VK_NULL_HANDLE;
  u64 chainValueA = 0, chainValueB = 0;
  bool asyncActive = false;
  // One chain submission per frame slot: a second beginChain before
  // endFrame would re-begin the SAME command buffer whose previous
  // submission may still be executing (two-chain test setups).
  bool chainSubmittedThisFrame = false;

  // The command buffer every chain-recording call targets (the
  // compute override during beginChainCull/beginChainRaster, the
  // frame's graphics buffer otherwise).
  VkCommandBuffer rc() const {
    return computeOverride != VK_NULL_HANDLE ? computeOverride : cmd.cmd();
  }

  Swapchain swapchain;
  FrameSync frameSync;
  StagingRing staging;  // per-frame dynamic data slices (M2 consumer)

  // M6.2 page streaming: upload batches on the transfer queue (or the
  // graphics queue in fallback). The backend owns the command pools,
  // the timeline and the staging arena - callers only record batches.
  struct PageStreaming {
    VkCommandPool transferPool = VK_NULL_HANDLE;  // family: transfer
    VkCommandPool graphicsPool = VK_NULL_HANDLE;  // family: graphics
    VkCommandBuffer transferCb = VK_NULL_HANDLE;
    VkCommandBuffer graphicsCb = VK_NULL_HANDLE;
    VkSemaphore timeline = VK_NULL_HANDLE;
    u64 signaledValue = 0;   // value of the last submitted batch
    u64 completedValue = 0;  // host-waited watermark
    bool batchOpen = false;
    bool batchEmpty = true;
    bool transferEnabled = true;  // A/B fallback switch
    // LATCHED AT beginPageUploads: the open batch's queue choice and
    // target pool geometry. setTransferQueueEnabled takes effect only
    // on the NEXT batch (the documented contract), and the offset math
    // must use THIS pool's slotBytes even when several pools exist.
    bool batchOnTransfer = false;
    u32 dstPool = 0xFFFFFFFFu;
    u32 dstSlotBytes = 0;
    // createPageSlotPool records (handle -> slotBytes) here; only
    // handles with an entry are valid beginPageUploads targets.
    std::vector<std::pair<u32, u32>> poolSlotBytes;
    // Staging arena: one persistently-mapped host-visible buffer,
    // reused only after the previous batch's completion wait (armed in
    // beginPageUploads) - its bytes are read by the copies in flight.
    static constexpr usize kArenaBytes = 8u * 1024 * 1024;  // 64 x 128KB
    VkBuffer arenaBuffer = VK_NULL_HANDLE;
    VmaAllocation arenaAlloc = VK_NULL_HANDLE;
    void* arenaMapped = nullptr;
    usize arenaUsed = 0;
  } page;
  // ImGui texture sets of retired offscreens, freed after a delay in
  // beginFrame (see FrameSync::retired for the driver hazard).
  std::vector<std::pair<void*, u64>> retiredImguiSets;
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
  // M4.3: async compute chain resources (optional - the chain falls
  // back to the graphics queue without them).
  m_impl->asyncActive = m_impl->frameSync.initAsyncCompute(m_impl->device);
  if (m_impl->asyncActive)
    KENSHIN_INFO("[vulkan] async compute chain: enabled");
  // 16MB per in-flight frame of mapped staging; M2's per-frame data
  // (culling inputs / instance tables) will slice out of this.
  if (!m_impl->staging.init(m_impl->device.allocator, 16ull * 1024 * 1024,
                            kMaxFramesInFlight)) {
    KENSHIN_ERROR("[vulkan] staging ring init failed");
    return false;
  }
  // M6.2: page-upload resources (transfer queue + timeline + arena).
  if (!initPageStreaming()) {
    KENSHIN_ERROR("[vulkan] page streaming init failed");
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
  for (auto& [set, frame] : m_impl->retiredImguiSets)
    ImGui_ImplVulkan_RemoveTexture(static_cast<VkDescriptorSet>(set));
  m_impl->retiredImguiSets.clear();
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
  // M6.2 page streaming (device idled above; the slot POOL buffer rode
  // buffers.destroyAll, these are the batch-side resources).
  if (m_impl->page.arenaBuffer != VK_NULL_HANDLE) {
    vmaDestroyBuffer(m_impl->device.allocator, m_impl->page.arenaBuffer,
                     m_impl->page.arenaAlloc);
    m_impl->page.arenaBuffer = VK_NULL_HANDLE;
  }
  if (m_impl->page.transferPool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(m_impl->device.device, m_impl->page.transferPool,
                         nullptr);
    m_impl->page.transferPool = VK_NULL_HANDLE;
  }
  if (m_impl->page.graphicsPool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(m_impl->device.device, m_impl->page.graphicsPool,
                         nullptr);
    m_impl->page.graphicsPool = VK_NULL_HANDLE;
  }
  if (m_impl->page.timeline != VK_NULL_HANDLE) {
    vkDestroySemaphore(m_impl->device.device, m_impl->page.timeline, nullptr);
    m_impl->page.timeline = VK_NULL_HANDLE;
  }
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
  // M4.3: the same pacing for the slot's CHAIN command buffers - the
  // compute timeline trails the graphics one by however long the
  // compute queue lags, so its (kMaxFramesInFlight-ago) valueB must
  // be waited separately before this slot's chain cbs are rewritten.
  if (impl.asyncActive && impl.frameSync.computeValue >= 2 * kMaxFramesInFlight)
    impl.frameSync.waitCompute(
        impl.device, impl.frameSync.computeValue - 2 * kMaxFramesInFlight + 2);
  // Per-frame chain value bookkeeping (chainOnComputeQueue gates the
  // frame submission's waits; zeroed until this frame submits a chain).
  impl.chainValueA = impl.chainValueB = 0;
  impl.chainSubmittedThisFrame = false;

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
  impl.frameSync.drainRetired(impl.device);
  {
    // ImGui sets of retired offscreens are freed a few frames later too
    // (same driver hazard as FrameSync::retired).
    constexpr u64 kSetRetireDelay = 8;
    for (u64 k = 0; k < impl.retiredImguiSets.size();) {
      if (impl.frameSync.frameCounter - impl.retiredImguiSets[k].second >=
          kSetRetireDelay) {
        ImGui_ImplVulkan_RemoveTexture(
            static_cast<VkDescriptorSet>(impl.retiredImguiSets[k].first));
        impl.retiredImguiSets.erase(impl.retiredImguiSets.begin() + k);
      } else {
        ++k;
      }
    }
  }

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
    // M4.3: the chain's compute-timeline waits. valueB waits at
    // ALL_COMMANDS here (headless frames have no decode draws whose
    // stage mask could order it, and the test readbacks that follow
    // must observe the completed chain).
    VkSemaphoreSubmitInfo chainWaits[2] = {};
    u32 chainWaitCount = 0;
    if (impl.chainValueB != 0) {
      chainWaits[chainWaitCount++] = VkSemaphoreSubmitInfo{
          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
          .semaphore = impl.frameSync.computeTimeline,
          .value = impl.chainValueB,
          .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      };
    }
    const VkSubmitInfo2 submit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount = chainWaitCount,
        .pWaitSemaphoreInfos = chainWaits,
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
  // M4.3: the chain's compute-timeline waits, stage-masked to keep
  // the overlap real - valueA gates only the HW draw's indirect
  // command + vertex fetches (it may then run CONCURRENTLY with the
  // still-executing SW raster; both atomically max the same
  // VisBuffer), valueB gates the decode's fragment-stage vis read.
  VkSemaphoreSubmitInfo waits[3] = {waitSubmit, {}, {}};
  u32 waitCount = 1;
  if (impl.chainValueB != 0) {
    waits[waitCount++] = VkSemaphoreSubmitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = impl.frameSync.computeTimeline,
        .value = impl.chainValueA,
        .stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                     VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
    };
    waits[waitCount++] = VkSemaphoreSubmitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = impl.frameSync.computeTimeline,
        .value = impl.chainValueB,
        .stageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
    };
  }
  const VkSubmitInfo2 submit{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = waitCount,
      .pWaitSemaphoreInfos = waits,
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
    // Retire the old set (freed in beginFrame once aged) instead of freeing
    // it here: same recently-sampled-resource driver hazard as the retired
    // offscreen images in FrameSync.
    if (slot.imguiTexture) {
      m_impl->retiredImguiSets.push_back(
          {slot.imguiTexture, m_impl->frameSync.frameCounter});
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
  // The frame-timeline pacing of the deletion queue does not cover
  // page-upload batches on the transfer queue (they may still hold the
  // LAST reference to this buffer) - drain them before the deferred
  // destroy becomes eligible.
  if (impl.page.signaledValue > impl.page.completedValue)
    waitPageUploads(impl.page.signaledValue);
  VmaAllocator allocator = impl.device.allocator;
  impl.frameSync.slots[impl.frameSync.frameIndex].deletionQueue.push(
      [allocator, raw] { BufferStore::destroyRaw(allocator, raw); });
}

bool VulkanBackend::readbackBuffer(u32 buffer, usize offset, usize bytes,
                                   std::vector<u8>& out) {
  if (!m_impl->buffers.valid(buffer)) return false;
  // M4.3: the chain may still be executing on the compute queue -
  // BufferStore's vkQueueWaitIdle targets the GRAPHICS queue only.
  // Host-wait the chain timeline so the copy observes final values.
  if (m_impl->asyncActive)
    m_impl->frameSync.waitCompute(m_impl->device,
                                  m_impl->frameSync.computeValue);
  // M6.2: same exposure for page-upload batches on the transfer queue.
  if (m_impl->page.signaledValue > m_impl->page.completedValue)
    waitPageUploads(m_impl->page.signaledValue);
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

u32 VulkanBackend::createNullPipeline(const FullscreenPipelineInfo& info) {
  FullscreenPipelineInfo patched = info;
  patched.conservativeRaster = m_impl->context.supportsConservativeRaster;
  return m_impl->pipelines.createNullRaster(m_impl->device.device, patched);
}

void VulkanBackend::destroyNullPipeline(u32 pipeline) {
  Impl& impl = *m_impl;
  if (pipeline == 0 || pipeline > impl.pipelines.nullRasterCount()) return;
  PipelineStore::Fullscreen& e = const_cast<PipelineStore::Fullscreen&>(
      impl.pipelines.nullRaster(pipeline));
  if (!e.valid) return;
  // Same deferred discipline as destroyComputePipeline.
  VkDevice device = impl.device.device;
  const VkPipeline pipelineRaw = e.pipeline;
  const VkPipelineLayout layoutRaw = e.layout;
  const VkDescriptorPool poolRaw = e.pool;
  const VkDescriptorSetLayout setLayoutRaw = e.setLayout;
  impl.frameSync.slots[impl.frameSync.frameIndex].deletionQueue.push(
      [device, pipelineRaw, layoutRaw, poolRaw, setLayoutRaw] {
        vkDestroyPipeline(device, pipelineRaw, nullptr);
        vkDestroyPipelineLayout(device, layoutRaw, nullptr);
        vkDestroyDescriptorPool(device, poolRaw, nullptr);
        vkDestroyDescriptorSetLayout(device, setLayoutRaw, nullptr);
      });
  e = {};
}

void VulkanBackend::drawNullIndirect(u32 pipeline, const u32* bufferHandles,
                                     u32 bufferCount, u32 argsBuffer,
                                     usize argsOffset, u32 width, u32 height) {
  Impl& impl = *m_impl;
  if (!impl.frameActive || pipeline == 0 ||
      pipeline > impl.pipelines.nullRasterCount())
    return;
  const PipelineStore::Fullscreen& nr = impl.pipelines.nullRaster(pipeline);
  if (!nr.valid || !impl.buffers.valid(argsBuffer)) return;

  // Validate + collect BEFORE opening the scope (bail-out must not
  // leave a pass dangling).
  std::vector<VkDescriptorBufferInfo> infos(bufferCount);
  for (u32 i = 0; i < bufferCount; ++i) {
    if (!impl.buffers.valid(bufferHandles[i])) return;
    infos[i] = VkDescriptorBufferInfo{
        .buffer = impl.buffers.buffer(bufferHandles[i]),
        .offset = 0,
        .range = impl.buffers.size(bufferHandles[i]),
    };
  }
  const VkBuffer argBuf = impl.buffers.buffer(argsBuffer);

  if (impl.cmd.passOpen()) impl.cmd.endPass();
  impl.cmd.beginNullPass(width, height);

  const VkDescriptorSet set = nr.sets[impl.frameSync.frameIndex];
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
                          nr.layout, 0, 1, &set, 0, nullptr);
  vkCmdBindPipeline(impl.cmd.cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                    nr.pipeline);
  const VkDeviceSize stride = 4 * sizeof(u32);  // VkDrawIndirectCommand
  vkCmdDrawIndirect(impl.cmd.cmd(), argBuf,
                    static_cast<VkDeviceSize>(argsOffset), 1, stride);
  impl.cmd.endPass();
}

bool VulkanBackend::beginChain() {
  Impl& impl = *m_impl;
  if (!impl.asyncActive || !impl.frameActive || impl.chainSubmittedThisFrame)
    return false;
  FrameSlot& slot = impl.frameSync.slots[impl.frameSync.frameIndex];
  vkResetCommandBuffer(slot.computeCullCb, 0);
  const VkCommandBufferBeginInfo beginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  if (!KE_VK(vkBeginCommandBuffer(slot.computeCullCb, &beginInfo)))
    return false;
  impl.computeOverride = slot.computeCullCb;
  return true;
}

void VulkanBackend::submitChain() {
  Impl& impl = *m_impl;
  if (!impl.asyncActive || impl.computeOverride == VK_NULL_HANDLE) return;
  FrameSlot& slot = impl.frameSync.slots[impl.frameSync.frameIndex];
  if (!KE_VK(vkEndCommandBuffer(slot.computeCullCb))) return;

  // The chain waits the PREVIOUS chain's completion (it reads the
  // HZB the last frame's tail built - same queue, but submissions
  // carry no implicit dependencies) and signals this frame's value,
  // which the graphics submission waits before the HW draw + decode.
  const u64 prev = impl.frameSync.computeValue;
  const u64 value = prev + 1;
  const VkSemaphoreSubmitInfo waitInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = impl.frameSync.computeTimeline,
      .value = prev,
      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
  };
  const VkSemaphoreSubmitInfo signalInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = impl.frameSync.computeTimeline,
      .value = value,
      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
  };
  const VkCommandBufferSubmitInfo cmdSubmit{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = slot.computeCullCb,
  };
  const VkSubmitInfo2 submit{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = prev > 0 ? 1u : 0u,
      .pWaitSemaphoreInfos = &waitInfo,
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &cmdSubmit,
      .signalSemaphoreInfoCount = 1,
      .pSignalSemaphoreInfos = &signalInfo,
  };
  if (KE_VK(vkQueueSubmit2(impl.device.asyncComputeQueue, 1, &submit,
                           VK_NULL_HANDLE))) {
    impl.frameSync.computeValue = value;
    impl.chainValueA = impl.chainValueB = value;
    impl.chainSubmittedThisFrame = true;
  }
  impl.computeOverride = VK_NULL_HANDLE;
}

bool VulkanBackend::chainOnComputeQueue() const {
  return m_impl->asyncActive && m_impl->chainValueB != 0;
}

void VulkanBackend::setAsyncChainEnabled(bool enabled) {
  // Takes effect from the NEXT beginChain; an in-flight chain's
  // waits stay armed on the frame submission regardless.
  m_impl->asyncActive = m_impl->frameSync.asyncComputeActive && enabled;
}

// ── page streaming (M6.2) ─────────────────────────────────────

bool VulkanBackend::initPageStreaming() {
  Impl& impl = *m_impl;
  VulkanDevice& device = impl.device;
  auto& page = impl.page;

  // One command pool per submission family (the fallback switches
  // families; pools are family-bound). RESET flag: one reused
  // command buffer per pool.
  const VkCommandPoolCreateInfo transferPoolInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = device.transferFamily,
  };
  const VkCommandPoolCreateInfo graphicsPoolInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = device.graphicsFamily,
  };
  if (!KE_VK(vkCreateCommandPool(device.device, &transferPoolInfo, nullptr,
                                 &page.transferPool)) ||
      !KE_VK(vkCreateCommandPool(device.device, &graphicsPoolInfo, nullptr,
                                 &page.graphicsPool)))
    return false;
  const VkCommandBufferAllocateInfo allocTransfer{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = page.transferPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };
  const VkCommandBufferAllocateInfo allocGraphics{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = page.graphicsPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };
  if (!KE_VK(vkAllocateCommandBuffers(device.device, &allocTransfer,
                                      &page.transferCb)) ||
      !KE_VK(vkAllocateCommandBuffers(device.device, &allocGraphics,
                                      &page.graphicsCb)))
    return false;

  // The page-upload timeline (independent of frame/chain pacing).
  const VkSemaphoreTypeCreateInfo timelineType{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
      .initialValue = 0,
  };
  const VkSemaphoreCreateInfo timelineInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &timelineType,
  };
  if (!KE_VK(
          vkCreateSemaphore(device.device, &timelineInfo, nullptr,
                            &page.timeline)))
    return false;

  // Staging arena: copies are submitted from the transfer OR graphics
  // queue depending on the fallback - CONCURRENT over both families.
  const bool sameFamily = device.transferFamily == device.graphicsFamily;
  const u32 arenaFamilies[2] = {device.graphicsFamily, device.transferFamily};
  const VkBufferCreateInfo arenaInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = static_cast<VkDeviceSize>(page.kArenaBytes),
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      .sharingMode = sameFamily ? VK_SHARING_MODE_EXCLUSIVE
                                : VK_SHARING_MODE_CONCURRENT,
      .queueFamilyIndexCount = sameFamily ? 0u : 2u,
      .pQueueFamilyIndices = arenaFamilies,
  };
  const VmaAllocationCreateInfo arenaAlloc{
      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
               VMA_ALLOCATION_CREATE_MAPPED_BIT,
      .usage = VMA_MEMORY_USAGE_AUTO,
      .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
  };
  VmaAllocationInfo out{};
  if (vmaCreateBuffer(device.allocator, &arenaInfo, &arenaAlloc,
                      &page.arenaBuffer, &page.arenaAlloc, &out) !=
          VK_SUCCESS ||
      out.pMappedData == nullptr)
    return false;
  page.arenaMapped = out.pMappedData;

  KENSHIN_INFO(
      "[vulkan] page uploads: {} ({}B staging arena)",
      pageUploadsOnTransferQueue() ? "dedicated transfer queue" : "graphics queue",
      page.kArenaBytes);
  return true;
}

u32 VulkanBackend::createPageSlotPool(u32 slotCount, u32 slotBytes) {
  Impl& impl = *m_impl;
  if (slotCount == 0 || slotBytes == 0)
    return kInvalidBufferHandle;
  // CONCURRENT over every family that touches slots: graphics (frame
  // reads), async compute (the chain) and transfer (the uploads).
  // TRANSFER_SRC: readbackBuffer copies OUT of the pool (the M6.2
  // verification path) - the copy-source usage bit is a VU requirement.
  const u32 allFamilies[3] = {impl.device.graphicsFamily,
                              impl.device.asyncComputeFamily,
                              impl.device.transferFamily};
  u32 families[3];
  u32 familyCount = 0;
  for (const u32 f : allFamilies) {
    if (f == 0xFFFFFFFFu) continue;
    bool dup = false;
    for (u32 k = 0; k < familyCount; ++k) dup |= families[k] == f;
    if (!dup) families[familyCount++] = f;
  }
  const u32 handle = impl.buffers.createShared(
      impl.device, usize(slotCount) * slotBytes,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      families, familyCount, "pageSlotPool");
  if (handle != kInvalidBufferHandle)
    impl.page.poolSlotBytes.emplace_back(handle, slotBytes);
  return handle;
}

bool VulkanBackend::beginPageUploads(u32 slotPool) {
  Impl& impl = *m_impl;
  auto& page = impl.page;
  if (page.batchOpen || page.timeline == VK_NULL_HANDLE ||
      !impl.buffers.valid(slotPool))
    return false;
  // Only createPageSlotPool handles carry the slot geometry.
  u32 slotBytes = 0;
  for (const auto& [handle, bytes] : page.poolSlotBytes)
    if (handle == slotPool) slotBytes = bytes;
  if (slotBytes == 0) return false;
  // Arena reuse proof: the previous batch's copies may still be reading
  // the staging bytes - drain them before overwriting.
  if (page.signaledValue > page.completedValue &&
      !waitPageUploads(page.signaledValue))
    return false;
  // Latch the batch's queue choice (see PageStreaming): a mid-batch
  // setTransferQueueEnabled must not reroute the recording.
  page.batchOnTransfer = pageUploadsOnTransferQueue();
  const VkCommandBuffer cb =
      page.batchOnTransfer ? page.transferCb : page.graphicsCb;
  vkResetCommandBuffer(cb, 0);
  const VkCommandBufferBeginInfo beginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  if (!KE_VK(vkBeginCommandBuffer(cb, &beginInfo))) return false;
  page.batchOpen = true;
  page.batchEmpty = true;
  page.arenaUsed = 0;
  page.dstPool = slotPool;
  page.dstSlotBytes = slotBytes;
  return true;
}

bool VulkanBackend::copyPageToSlot(const void* hostData, u32 slot, u32 bytes) {
  Impl& impl = *m_impl;
  auto& page = impl.page;
  if (!page.batchOpen || hostData == nullptr || bytes == 0 ||
      page.dstSlotBytes == 0 || bytes > page.dstSlotBytes)
    return false;
  if (page.arenaUsed + bytes > page.kArenaBytes) return false;
  const usize slotBytes = page.dstSlotBytes;
  const usize dstOffset64 = usize(slot) * slotBytes;
  if (dstOffset64 + bytes > impl.buffers.size(page.dstPool)) return false;
  std::memcpy(static_cast<u8*>(page.arenaMapped) + page.arenaUsed, hostData,
              bytes);
  const VkCommandBuffer cb =
      page.batchOnTransfer ? page.transferCb : page.graphicsCb;
  const VkBufferCopy region{
      .srcOffset = static_cast<VkDeviceSize>(page.arenaUsed),
      .dstOffset = static_cast<VkDeviceSize>(dstOffset64),
      .size = static_cast<VkDeviceSize>(bytes),
  };
  vkCmdCopyBuffer(cb, page.arenaBuffer, impl.buffers.buffer(page.dstPool), 1,
                  &region);
  page.arenaUsed = (page.arenaUsed + bytes + 15) & ~usize(15);
  page.batchEmpty = false;
  return true;
}

u64 VulkanBackend::submitPageUploads() {
  Impl& impl = *m_impl;
  auto& page = impl.page;
  if (!page.batchOpen) return 0;
  page.batchOpen = false;
  if (page.batchEmpty) return 0;
  const bool useTransfer = page.batchOnTransfer;
  const VkCommandBuffer cb = useTransfer ? page.transferCb : page.graphicsCb;
  if (!KE_VK(vkEndCommandBuffer(cb))) return 0;

  // Drain frames submitted so far before the copies run: an evicted
  // slot may still be read by an in-flight frame. The wait value is
  // the last SUBMITTED frame's - a batch recorded mid-frame N drains
  // everything through frame N-1 (frame N is not submitted yet, and
  // the manager flips the old page's table entry before issuing the
  // upload, so frame N sees neither page - it skips and counts
  // pending). Recorded mid-frame (after beginFrame's pacing wait) the
  // wait is already satisfied.
  const u64 frameWait = impl.frameSync.timelineValue;
  const u64 value = page.signaledValue + 1;
  const VkSemaphoreSubmitInfo frameWaitInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = impl.frameSync.frameTimeline,
      .value = frameWait,
      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
  };
  const VkSemaphoreSubmitInfo signalInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = page.timeline,
      .value = value,
      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
  };
  const VkCommandBufferSubmitInfo cmdSubmit{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = cb,
  };
  const VkSubmitInfo2 submit{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = frameWait > 0 ? 1u : 0u,
      .pWaitSemaphoreInfos = &frameWaitInfo,
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &cmdSubmit,
      .signalSemaphoreInfoCount = 1,
      .pSignalSemaphoreInfos = &signalInfo,
  };
  const VkQueue queue =
      useTransfer ? impl.device.transferQueue : impl.device.graphicsQueue;
  if (KE_VK(vkQueueSubmit2(queue, 1, &submit, VK_NULL_HANDLE))) {
    page.signaledValue = value;
    return value;
  }
  return 0;
}

bool VulkanBackend::waitPageUploads(u64 value) {
  Impl& impl = *m_impl;
  auto& page = impl.page;
  if (value == 0) return true;
  if (page.timeline == VK_NULL_HANDLE || value > page.signaledValue)
    return false;
  const VkSemaphoreWaitInfo wait{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .semaphoreCount = 1,
      .pSemaphores = &page.timeline,
      .pValues = &value,
  };
  if (!KE_VK(vkWaitSemaphores(impl.device.device, &wait, UINT64_MAX)))
    return false;
  if (value > page.completedValue) page.completedValue = value;
  return true;
}

void VulkanBackend::setTransferQueueEnabled(bool enabled) {
  // Takes effect from the NEXT beginPageUploads; an open batch keeps
  // the queue it was opened with.
  m_impl->page.transferEnabled = enabled;
}

bool VulkanBackend::pageUploadsOnTransferQueue() const {
  const Impl& impl = *m_impl;
  return impl.page.transferEnabled &&
         impl.device.transferQueue != VK_NULL_HANDLE &&
         impl.device.transferFamily != impl.device.graphicsFamily;
}

void VulkanBackend::barrierGraphicsToCompute(u32 buffer) {
  Impl& impl = *m_impl;
  if (!impl.frameActive || !impl.buffers.valid(buffer)) return;
  const VkBufferMemoryBarrier2 barrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      // The HW raster pass's VERTEX reads + FRAGMENT atomics.
      .srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
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
  vkCmdPipelineBarrier2(impl.rc(), &dep);
}

void VulkanBackend::barrierIndirectArgs(u32 buffer) {
  Impl& impl = *m_impl;
  if (!impl.frameActive || !impl.buffers.valid(buffer)) return;
  const VkBufferMemoryBarrier2 barrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
      .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
      .buffer = impl.buffers.buffer(buffer),
      .offset = 0,
      .size = impl.buffers.size(buffer),
  };
  const VkDependencyInfo dep{
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .bufferMemoryBarrierCount = 1,
      .pBufferMemoryBarriers = &barrier,
  };
  vkCmdPipelineBarrier2(impl.rc(), &dep);
}

void VulkanBackend::barrierGraphicsRw(u32 buffer) {
  Impl& impl = *m_impl;
  if (!impl.frameActive || !impl.buffers.valid(buffer)) return;
  const VkBufferMemoryBarrier2 barrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      // The HW pass reads the pair lists in its VERTEX shader and
      // atomically writes the VisBuffer in its FRAGMENT shader.
      .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
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
  vkCmdPipelineBarrier2(impl.rc(), &dep);
}

u32 VulkanBackend::createComputePipeline(const ComputePipelineInfo& info) {
  return m_impl->pipelines.createCompute(m_impl->device.device, info);
}

void VulkanBackend::destroyComputePipeline(u32 pipeline) {
  Impl& impl = *m_impl;
  if (pipeline == 0 || pipeline > impl.pipelines.computeCount()) return;
  PipelineStore::Compute& c =
      const_cast<PipelineStore::Compute&>(impl.pipelines.compute(pipeline));
  if (!c.valid) return;
  // DEFERRED destroy (the immediate form died with vkDestroyPipeline
  // "currently in use" warnings at app teardown - the last frame's
  // command buffer references the pipeline): push onto this slot's
  // deletion queue (flushed after the shutdown device-idle; also at
  // slot reuse), clear the entry so the store's own shutdown skips
  // it (no double destroy).
  VkDevice device = impl.device.device;
  const VkPipeline pipelineRaw = c.pipeline;
  const VkPipelineLayout layoutRaw = c.layout;
  const VkDescriptorPool poolRaw = c.pool;
  const VkDescriptorSetLayout setLayoutRaw = c.setLayout;
  impl.frameSync.slots[impl.frameSync.frameIndex].deletionQueue.push(
      [device, pipelineRaw, layoutRaw, poolRaw, setLayoutRaw] {
        vkDestroyPipeline(device, pipelineRaw, nullptr);
        vkDestroyPipelineLayout(device, layoutRaw, nullptr);
        vkDestroyDescriptorPool(device, poolRaw, nullptr);
        vkDestroyDescriptorSetLayout(device, setLayoutRaw, nullptr);
      });
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

  vkCmdBindDescriptorSets(impl.rc(), VK_PIPELINE_BIND_POINT_COMPUTE, c.layout,
                          0, 1, &set, 0, nullptr);
  vkCmdBindPipeline(impl.rc(), VK_PIPELINE_BIND_POINT_COMPUTE, c.pipeline);
  vkCmdDispatch(impl.rc(), groupsX, groupsY, groupsZ);
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
    vkCmdPipelineBarrier2(impl.rc(), &dep);
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
  vkCmdBindDescriptorSets(impl.rc(), VK_PIPELINE_BIND_POINT_COMPUTE, c.layout,
                          0, 1, &set, 0, nullptr);
  vkCmdBindPipeline(impl.rc(), VK_PIPELINE_BIND_POINT_COMPUTE, c.pipeline);
  vkCmdDispatchIndirect(impl.rc(), impl.buffers.buffer(countBuffer),
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
  // A toFragment barrier inside the async chain's command buffer is
  // illegal (the compute-only family has no fragment stage) and
  // meaningless - the chain-to-graphics handoff is the compute
  // timeline wait on the frame submission. Drop it there; the
  // fragment ordering still rides the timeline.
  if (toFragment && impl.computeOverride != VK_NULL_HANDLE) return;
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
  vkCmdPipelineBarrier2(impl.rc(), &dep);
}

void VulkanBackend::fillBuffer(u32 buffer, u32 value) {
  Impl& impl = *m_impl;
  if (!impl.frameActive || !impl.buffers.valid(buffer)) return;
  const VkBufferMemoryBarrier2 warBarrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      // Order any prior compute read/write of the buffer (e.g. the
      // post cull consuming the pyramid) BEFORE this fill's write.
      // Barrier scopes span the whole queue's submission order, so
      // this also covers the other in-flight frame's reads.
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .buffer = impl.buffers.buffer(buffer),
      .offset = 0,
      .size = impl.buffers.size(buffer),
  };
  const VkDependencyInfo warDep{
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .bufferMemoryBarrierCount = 1,
      .pBufferMemoryBarriers = &warBarrier,
  };
  vkCmdPipelineBarrier2(impl.rc(), &warDep);
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
  vkCmdFillBuffer(impl.rc(), impl.buffers.buffer(buffer), 0, VK_WHOLE_SIZE,
                  value);
  // Self-contained: the fill's TRANSFER write is ordered after every
  // prior compute access and before any subsequent compute
  // read/write of the buffer (callers never need a manual barrier
  // around a fill).
  vkCmdPipelineBarrier2(impl.rc(), &dep);
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
