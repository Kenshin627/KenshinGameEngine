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

#ifdef _MSC_VER

#pragma warning(push, 0)

#endif
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "kenshin/core/Window.h"
#include "kenshin/platform/Paths.h"

KENSHIN_NS_BEGIN

namespace backend {

struct VulkanBackend::Impl {
  VulkanContext context;
  VulkanDevice device;
  bool presentSupported = false;
  const core::Window* window = nullptr;  // null = headless
  VkSurfaceKHR surface = VK_NULL_HANDLE;  // owned; outlives the swapchain
  std::function<void()> imguiPlatformNewFrame;

  // M4.3 async compute: when set, dispatch/barrier/fill calls record
  // into this command buffer instead of the frame's graphics one (the
  // chain's cull/raster segments). chainValueA/B are this frame's
  // compute-timeline values for the graphics submission's waits.
  VkCommandBuffer computeOverride = VK_NULL_HANDLE;
  u64 chainValueA = 0, chainValueB = 0;
  // chainWaitGraphics(): the frame-timeline value this frame's chain
  // submission must wait (0 = none).
  u64 chainGraphicsWait = 0;
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

  // M8.1: the one engine sampler (linear, clamp-to-edge) for the
  // lighting pass's GBuffer reads.
  VkSampler linearSampler = VK_NULL_HANDLE;
  // M8.2: albedo sampling (mip-filtered, repeat for tiling UVs) and the
  // immutable texture-array table (static scene resources; destroyed at
  // shutdown). dummyTexture keeps mixed descriptor sets validly written
  // even when a scene carries no albedo.
  VkSampler albedoSampler = VK_NULL_HANDLE;
  // M9.1: the directional light's shadow target (persistent, fixed
  // size - not per-slot) + the NEAREST sampler for depth comparison
  // (a linear filter would average DEPTHS, not comparisons).
  GpuImage shadowMap[3];  // M9.1 Stage B: clipmap levels 0..2
  VkSampler shadowSampler = VK_NULL_HANDLE;
  VkImageLayout shadowMapLayout[3] = {};
  u32 shadowMapSize = 2048;
  // M9 TAA: the resolved-HDR ping-pong pair (viewport-sized, RGBA16F
  // like the slot HDR targets). drawTaa writes taa[parity] and reads
  // taa[parity ^ 1] as history; drawTonemap(fromTaa) samples the
  // freshly written one.
  GpuImage taa[2];
  VkImageLayout taaLayout[2] = {};
  // Per-slot copies of the TAA params block (drawTaa memcpys the
  // pass's shared host-mapped buffer into the CURRENT slot's copy at
  // record time): the shared block is rewritten every frame - its
  // jitter word differs per frame - while the previous frame's shader
  // read may still be in flight; slot ownership is the only pacing
  // that covers a 1-frame-reuse CPU write.
  u32 taaParamsSlot[kMaxFramesInFlight] = {kInvalidBufferHandle,
                                           kInvalidBufferHandle};
  // snapshotHostBuffer mirrors: source handle -> per-slot copies
  // (created lazily; few entries - one per hot shared params buffer).
  std::unordered_map<u32, u32> snapMirrors[kMaxFramesInFlight];
  u32 taaParity = 0;
  u32 taaSize = 0;
  u32 taaHeight = 0;  // both extents key the pair (height-only resizes
                      // used to keep a stale-height pair)
  bool taaActive = false;
  // Bloom chain targets (half, quarter, ... resolution; RGBA16F) +
  // whether this frame recorded the chain (drawTonemap binds level 0).
  static constexpr u32 kBloomLevels = 6;
  GpuImage bloom[kBloomLevels];
  u32 bloomW = 0, bloomH = 0;
  bool bloomActive = false;
  // Cross-submission TAA history dependency (see drawTaa): the frame
  // submit waits this timeline value before the resolve's history read.
  u64 taaHistWaitValue = 0;
  bool taaHistWaitNeeded = false;
  GpuImage dummyTexture;
  std::vector<GpuImage> textures;
  std::vector<bool> textureAlive;

  // GPU timestamps + graphics pipelines (PipelineStore: table + disk cache)
  bool gpuTimestamps = false;
  // Chain segment timestamps: valid on the async compute family (the
  // chain's normal home) and armed-once-per-slot reset tracking.
  bool chainTimestampsCompute = false;
  bool chainQueriesArmed[kMaxFramesInFlight] = {};
  // Per-slot "this pool was reset at least once by a chain write" -
  // unlike the frame pool (written unconditionally every frame), a
  // slot's chain pool only ever sees a reset when the chain actually
  // executes on it; harvesting a never-reset pool is a validation
  // error. Chain-idle frames (window not drawable) still leave STALE
  // (complete) stamps in an armed pool - documented telemetry caveat,
  // not corrected for.
  bool chainPoolEverWritten[kMaxFramesInFlight] = {};
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
    const auto exeDir = platform::executableDir();
    if (!exeDir.empty() &&
        std::filesystem::exists(exeDir / "VkLayer_khronos_validation.json")) {
      platform::setEnvironmentVariable("VK_LAYER_PATH", exeDir.string());
      KENSHIN_INFO("validation layer path -> {}", exeDir.string());
    }
  }

  const std::vector<std::string> surfaceExtensions =
      params.window ? params.window->vulkanInstanceExtensions()
                    : std::vector<std::string>{};
  if (params.window && surfaceExtensions.empty()) {
    KENSHIN_ERROR("[vulkan] the window reports no Vulkan surface extensions");
    return false;
  }
  if (!m_impl->context.init(params.enableValidation, surfaceExtensions))
    return false;

  if (params.window) {
    u64 surfaceBits = params.window->createVulkanSurface(m_impl->context.instance);
    std::memcpy(&m_impl->surface, &surfaceBits, sizeof(m_impl->surface));
    if (m_impl->surface == VK_NULL_HANDLE) return false;
    m_impl->presentSupported =
        m_impl->context.checkPresentSupport(m_impl->surface);
  }

  if (!m_impl->device.init(m_impl->context)) return false;

  m_impl->window = params.window;

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
      params.window != nullptr
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
    m_impl->chainTimestampsCompute =
        m_impl->context.asyncComputeFamily != 0xFFFFFFFFu &&
        familyCount > m_impl->context.asyncComputeFamily &&
        families[m_impl->context.asyncComputeFamily].timestampValidBits > 0;
    m_impl->timestampPeriodNs = m_impl->context.props.limits.timestampPeriod;
  }

  if (!m_impl->pipelines.init(m_impl->device.device)) return false;

  // M8.1: the shared linear/clamp sampler (GBuffer reads + future
  // material textures).
  {
    const VkSamplerCreateInfo samplerInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .maxLod = VK_LOD_CLAMP_NONE,
    };
    if (!KE_VK(vkCreateSampler(m_impl->device.device, &samplerInfo, nullptr,
                               &m_impl->linearSampler)))
      return false;
  }
  // M9.1: shadow support gate + the comparison sampler (NEAREST -
  // PCF does its own kernel; linear-filtered depths are wrong).
  {
    VkFormatProperties fp{};
    vkGetPhysicalDeviceFormatProperties(m_impl->context.physicalDevice,
                                        VK_FORMAT_R32_SFLOAT, &fp);
    const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                                      VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    if ((fp.optimalTilingFeatures & need) == need) {
      const VkSamplerCreateInfo samplerInfo{
          .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
          // LINEAR: the receiver-side compare samples DEPTH manually
          // (not a PCF compare sampler) - filtering the value before
          // the compare is what kills the minification moire bands on
          // grazing receivers (the cube's side faces, the far ground):
          // NEAREST row-snaps the depth a fore-shortened face spans
          // (each map row = a world-horizontal line).
          .magFilter = VK_FILTER_LINEAR,
          .minFilter = VK_FILTER_LINEAR,
          .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
          .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
          .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
          .maxLod = VK_LOD_CLAMP_NONE,
      };
      if (KE_VK(vkCreateSampler(m_impl->device.device, &samplerInfo, nullptr,
                                &m_impl->shadowSampler)))
        KENSHIN_INFO("[vulkan] shadow map R32F: supported");
    } else {
      KENSHIN_WARN("[vulkan] shadow map R32F unsupported - shadows off");
    }
  }
  // M8.2: the albedo sampler - mip-filtered LINEAR (analytic-lod
  // SampleLevel snaps between mips with the default NEAREST mipmap
  // mode) and REPEAT addressing (tiling UVs; the knot wraps).
  {
    const VkSamplerCreateInfo samplerInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .maxLod = VK_LOD_CLAMP_NONE,
    };
    if (!KE_VK(vkCreateSampler(m_impl->device.device, &samplerInfo, nullptr,
                               &m_impl->albedoSampler)))
      return false;
  }
  // The always-valid binding target for mixed sets without a scene
  // texture (fsGBuffer's SPIR-V always carries the sample opcode, so
  // the binding is always live - dead-strip discipline in reverse).
  {
    const u8 white[4] = {255, 255, 255, 255};
    TextureArrayDesc dummy;
    dummy.width = 1;
    dummy.height = 1;
    dummy.rgba = white;
    dummy.bytes = 4;
    dummy.debugName = "dummy-texture";
    if (createTextureArray(dummy) == kInvalidTextureHandle) return false;
    m_impl->dummyTexture = m_impl->textures.back();
    m_impl->textures.pop_back();
    m_impl->textureAlive.pop_back();
  }
  // M8.1 format gate: the deferred GBuffer needs RGBA16F/RGBA8 as
  // color attachment + sampled (the engine never queried format
  // support before - report instead of trusting the driver blind).
  {
    const VkFormat formats[2] = {VK_FORMAT_R16G16B16A16_SFLOAT,
                                 VK_FORMAT_R8G8B8A8_UNORM};
    for (const VkFormat f : formats) {
      VkFormatProperties props{};
      vkGetPhysicalDeviceFormatProperties(m_impl->context.physicalDevice, f,
                                          &props);
      const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                                        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
      if ((props.optimalTilingFeatures & need) != need) {
        KENSHIN_ERROR(
            "[vulkan] format {} lacks color-attachment+sampled support - "
            "the deferred GBuffer cannot run",
            int(f));
        return false;
      }
    }
    KENSHIN_INFO("[vulkan] GBuffer formats RGBA16F/RGBA8: supported");
  }

  // Frame resources (command pools etc.) are device-level and needed by
  // windowless compute paths too. Timing pools are created when EITHER
  // the graphics or the async compute family reports timestamp bits
  // (the chain stamps ride whichever queue family runs the chain).
  if (!m_impl->frameSync.init(
          m_impl->device,
          m_impl->gpuTimestamps || m_impl->chainTimestampsCompute)) {
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

  if (params.window != nullptr && !m_impl->presentSupported) {
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
    ImGui_ImplVulkan_Shutdown();  // the platform half is the window layer's
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
  // M9 TAA ping-pong pair (created lazily; the shadow maps go the same
  // way through their recreate path or leak into validation reports).
  for (u32 i = 0; i < 2; ++i)
    if (m_impl->taa[i].image != VK_NULL_HANDLE)
      m_impl->taa[i].destroy(m_impl->device.allocator, m_impl->device.device);
  for (auto& b : m_impl->bloom)
    if (b.image != VK_NULL_HANDLE)
      b.destroy(m_impl->device.allocator, m_impl->device.device);
  // Frame resources first: their command buffers still reference the
  // mesh/depth objects below (validation: objects stay "in use" until
  // the referencing command buffer is freed with its pool).
  m_impl->frameSync.shutdown(m_impl->device);
  if (m_impl->linearSampler != VK_NULL_HANDLE) {
    vkDestroySampler(m_impl->device.device, m_impl->linearSampler, nullptr);
    m_impl->linearSampler = VK_NULL_HANDLE;
  }
  if (m_impl->albedoSampler != VK_NULL_HANDLE) {
    vkDestroySampler(m_impl->device.device, m_impl->albedoSampler, nullptr);
    m_impl->albedoSampler = VK_NULL_HANDLE;
  }
  if (m_impl->shadowSampler != VK_NULL_HANDLE) {
    vkDestroySampler(m_impl->device.device, m_impl->shadowSampler, nullptr);
    m_impl->shadowSampler = VK_NULL_HANDLE;
  }
  for (GpuImage& t : m_impl->shadowMap)
    if (t.image != VK_NULL_HANDLE)
      t.destroy(m_impl->device.allocator, m_impl->device.device);
  for (GpuImage& t : m_impl->textures)
    if (t.image != VK_NULL_HANDLE)
      t.destroy(m_impl->device.allocator, m_impl->device.device);
  m_impl->textures.clear();
  m_impl->textureAlive.clear();
  if (m_impl->dummyTexture.image != VK_NULL_HANDLE)
    m_impl->dummyTexture.destroy(m_impl->device.allocator,
                                 m_impl->device.device);
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
  m_impl->swapchain.shutdown(m_impl->device.device);
  if (m_impl->surface != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(m_impl->context.instance, m_impl->surface, nullptr);
    m_impl->surface = VK_NULL_HANDLE;
  }
  m_impl->device.shutdown();
  m_impl->context.shutdown();
}

std::unique_ptr<IRenderBackend> createVulkanBackend() {
  return std::make_unique<VulkanBackend>();
}

// ── frame loop (S1.3) ─────────────────────────────────────────

// Barrier/clear recording helpers live in CommandContext.

FrameStatus VulkanBackend::beginFrame(const FrameBeginInfo& info) {
  Impl& impl = *m_impl;
  impl.taaActive = false;  // drawTaa re-arms it; drawTonemap reads it
  impl.bloomActive = false;  // drawBloom re-arms it
  // Headless (no window): no swapchain - record offscreen-only frames
  // (in-frame compute tests, CI). Everything else (pacing, offscreen
  // targets, per-slot recording) runs unchanged.
  const bool headless = impl.window == nullptr;
  if (!headless && impl.swapchain.swapchain == VK_NULL_HANDLE) {
    if (!impl.swapchain.init(impl.context.physicalDevice, impl.device.device,
                             impl.surface, info.width, info.height))
      return FrameStatus::Error;
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

  // Harvest this slot's CHAIN segment stamps under the same completion
  // proof (the frame's fence covers the chain: the graphics submission
  // waits the chain's timeline value before the HW draw + decode).
  // Segments: 0 cull, 1 SW raster, 2 HZB#1, 3 post, 4 HZB#2, 5 tail.
  m_gpuTimings.chainValid = false;
  for (u32 k = 0; k < 6; ++k) m_gpuTimings.chainMs[k] = 0.0f;
  if (impl.chainPoolEverWritten[impl.frameSync.frameIndex] &&
      slot.chainTimingQueries != VK_NULL_HANDLE &&
      impl.frameSync.timelineValue >= kMaxFramesInFlight) {
    u64 cq[7] = {};
    if (vkGetQueryPoolResults(impl.device.device, slot.chainTimingQueries, 0, 7,
                              sizeof(cq), cq, sizeof(u64),
                              VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
      bool allWritten = cq[0] != 0;
      for (u32 k = 1; k < 7; ++k) allWritten = allWritten && cq[k] != 0;
      if (allWritten) {
        for (u32 k = 0; k < 6; ++k)
          m_gpuTimings.chainMs[k] =
              f32(f64(cq[k + 1] - cq[k]) * impl.timestampPeriodNs / 1e6);
        m_gpuTimings.chainValid = true;
      }
    }
  }
  // The chain re-arms (resets) its pool on the first write of the new
  // frame - vkCmdResetQueryPool must not race in-flight reads.
  impl.chainQueriesArmed[impl.frameSync.frameIndex] = false;

  // (Re)create per-frame offscreen targets on size/format change.
  const VkFormat targetFormat =
      headless ? VK_FORMAT_B8G8R8A8_UNORM : impl.swapchain.format;
  if (!impl.frameSync.ensureOffscreens(
          impl.device, info.width, info.height, targetFormat,
          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
              VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT))
    return FrameStatus::Error;
  // M8.1: the per-slot GBuffer pair rides the same retire discipline
  // (format support was gated once at initialize).
  if (!impl.frameSync.ensureGBuffer(impl.device, info.width, info.height))
    return FrameStatus::Error;
  // M8.3: the per-slot HDR intermediate (same retire discipline; the
  // format matches the gate-checked GBuffer A's RGBA16F).
  if (!impl.frameSync.ensureHdr(impl.device, info.width, info.height))
    return FrameStatus::Error;
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
    // A recreate deferred while the window was minimized runs first;
    // still no area = nothing to present this frame (Skipped, not an
    // error: the window system can minimize between the app's own
    // minimized check and this acquire).
    if (impl.swapchain.recreatePending &&
        !impl.swapchain.recreate(impl.context.physicalDevice,
                                 impl.device.device, info.width, info.height))
      return FrameStatus::Error;
    if (impl.swapchain.recreatePending) return FrameStatus::Skipped;
    acquireResult =
        impl.swapchain.acquire(impl.device.device, slot.imageAvailable);
    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
      if (!impl.swapchain.recreate(impl.context.physicalDevice,
                                   impl.device.device, info.width, info.height))
        return FrameStatus::Error;
      if (impl.swapchain.recreatePending) return FrameStatus::Skipped;
      acquireResult =
          impl.swapchain.acquire(impl.device.device, slot.imageAvailable);
    }
    if (acquireResult < 0) {
      KENSHIN_ERROR("[vulkan] acquire failed ({})",
                    static_cast<int>(acquireResult));
      return FrameStatus::Error;
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
    if (impl.imguiPlatformNewFrame) impl.imguiPlatformNewFrame();
    ImGui::NewFrame();
  }

  impl.frameActive = true;
  impl.activeWidth = info.width;
  impl.activeHeight = info.height;
  return FrameStatus::Ready;
}

void VulkanBackend::endFrame() {
  Impl& impl = *m_impl;
  if (!impl.frameActive) return;
  impl.frameActive = false;

  FrameSlot& slot = impl.frameSync.slots[impl.frameSync.frameIndex];
  const VkImage dstImage =
      impl.window == nullptr ? VK_NULL_HANDLE
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
  if (impl.window == nullptr) {
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
    VkSemaphoreSubmitInfo chainWaits[3] = {};
    u32 chainWaitCount = 0;
    if (impl.chainValueB != 0) {
      chainWaits[chainWaitCount++] = VkSemaphoreSubmitInfo{
          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
          .semaphore = impl.frameSync.computeTimeline,
          .value = impl.chainValueB,
          .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      };
    }
    // Same TAA history cross-submission wait as the windowed submit
    // (drawTaa runs headless too - the resolve is window-independent).
    if (impl.taaHistWaitNeeded) {
      chainWaits[chainWaitCount++] = VkSemaphoreSubmitInfo{
          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
          .semaphore = impl.frameSync.frameTimeline,
          .value = impl.taaHistWaitValue,
          .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      };
      impl.taaHistWaitNeeded = false;
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
  VkSemaphoreSubmitInfo waits[4] = {waitSubmit, {}, {}, {}};
  u32 waitCount = 1;
  if (impl.chainValueB != 0) {
    waits[waitCount++] = VkSemaphoreSubmitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = impl.frameSync.computeTimeline,
        .value = impl.chainValueA,
        // COMPUTE_SHADER: count_to_draw (the convert that fills
        // hwDrawArgs from the chain-written cullCnt[7]) runs as a
        // compute dispatch on this submission - without the stage it
        // read ungated per spec (NVIDIA's submit-level blocking made
        // it work in practice; a theory-race that surfacing under
        // driver-marginal states would zero every HW draw).
        .stageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                     VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    };
    waits[waitCount++] = VkSemaphoreSubmitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = impl.frameSync.computeTimeline,
        .value = impl.chainValueB,
        .stageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
    };
  }
  // TAA history cross-submission dependency (armed by drawTaa): wait
  // the previous frame's completion before any FRAGMENT-stage work in
  // THIS submission - the resolve reads the ping-pong image that frame
  // wrote. Waits the same timeline this submit signals next (legal for
  // timeline semaphores: wait value < signal value).
  if (impl.taaHistWaitNeeded) {
    waits[waitCount++] = VkSemaphoreSubmitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = impl.frameSync.frameTimeline,
        .value = impl.taaHistWaitValue,
        .stageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
    };
    impl.taaHistWaitNeeded = false;
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

bool VulkanBackend::initImGuiBackend(u32 width, u32 height,
                                     std::function<void()> platformNewFrame) {
  Impl& impl = *m_impl;
  if (impl.imguiInited || impl.window == nullptr) return impl.imguiInited;

  // The swapchain must exist so its format drives the ImGui pipeline.
  if (impl.swapchain.swapchain == VK_NULL_HANDLE &&
      !impl.swapchain.init(impl.context.physicalDevice, impl.device.device,
                           impl.surface, width, height))
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
  impl.imguiPlatformNewFrame = std::move(platformNewFrame);
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

u32 VulkanBackend::snapshotHostBuffer(u32 buffer) {
  Impl& impl = *m_impl;
  const usize bytes = impl.buffers.size(buffer);
  if (!impl.buffers.valid(buffer) || bytes == 0) return buffer;
  const u32 slotIdx = impl.frameSync.frameIndex;
  auto& mirror = impl.snapMirrors[slotIdx];
  auto it = mirror.find(buffer);
  if (it == mirror.end()) {
    BufferDesc desc;
    desc.size = bytes;
    desc.usage =
        IRenderBackend::BufferStorage | IRenderBackend::BufferHostVisible;
    desc.debugName = "host-snapshot-slot";
    const u32 created = createBuffer(desc);
    if (created == kInvalidBufferHandle) return buffer;  // passthrough
    it = mirror.emplace(buffer, created).first;
  }
  // The slot's prior frame has provably completed at beginFrame - the
  // memcpy cannot tear an in-flight read of THIS slot's mirror.
  std::memcpy(impl.buffers.mapped(it->second), impl.buffers.mapped(buffer),
              bytes);
  return it->second;
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

  // Binding i <- handle i, clamped to the pipeline's reflected binding
  // count (extra handles are inert). FEWER handles than bindings leaves
  // statically used bindings unwritten - undefined data, a validation
  // error every frame - so it is reported, once per pipeline.
  if (fs.bindingCount > bufferCount) {
    static std::vector<u32> reported;
    if (std::find(reported.begin(), reported.end(), pipeline) ==
        reported.end()) {
      reported.push_back(pipeline);
      KENSHIN_ERROR(
          "[vulkan] drawFullscreen: pipeline {} reflects {} bindings, only {} "
          "handles given",
          pipeline, fs.bindingCount, bufferCount);
    }
  }
  bufferCount = fs.bindingCount > 0 ? (std::min)(bufferCount, fs.bindingCount)
                                    : bufferCount;

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

// ── M8.1 deferred shading: GBuffer decode + lighting ────────────

// M8.2 texture upload: one-shot (own pool + cb, submit, wait) - the
// uploadDeviceBuffer pattern applied to images. Mip chain via linear
// blits; layouts per subresource tracked explicitly (a barrier's
// oldLayout must match reality):
//   copy stage : whole image UNDEFINED -> TRANSFER_DST
//   blit stage : before blitting m-1 -> m, mip m-1 goes DST -> SRC
//                (invariant: mip m-1 is in DST when the loop reaches
//                it - mip 0 from the copy, deeper mips from the blit
//                that produced them)
//   final      : [0, mips-1) SRC -> SHADER_READ_ONLY, last mip DST ->
//                SHADER_READ_ONLY
u32 VulkanBackend::createTextureArray(const TextureArrayDesc& desc) {
  Impl& impl = *m_impl;
  const usize texelBytes = desc.hdr ? usize(8) : usize(4);  // RGBA16F : RGBA8
  if (desc.width == 0 || desc.height == 0 || desc.layers == 0 ||
      desc.rgba == nullptr ||
      desc.bytes < static_cast<usize>(desc.width) * desc.height * texelBytes *
                       desc.layers) {
    KENSHIN_ERROR("[vulkan] createTextureArray: bad desc ({}x{}x{} bytes={})",
                  desc.width, desc.height, desc.layers, desc.bytes);
    return kInvalidTextureHandle;
  }
  u32 mips = 1;
  while ((desc.width | desc.height) >> mips != 0) ++mips;

  GpuImage image;
  if (!image.create(
          impl.device.allocator, impl.device.device, desc.width, desc.height,
          desc.hdr ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM,
          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
              VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mips, desc.layers,
          /*arrayedView=*/true)) {
    KENSHIN_ERROR("[vulkan] createTextureArray: image creation failed");
    return kInvalidTextureHandle;
  }

  // Staging: every layer's mip 0, consecutive.
  const usize layerBytes =
      static_cast<usize>(desc.width) * desc.height * texelBytes;
  const VkBufferCreateInfo stagingInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = static_cast<VkDeviceSize>(layerBytes * desc.layers),
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  const VmaAllocationCreateInfo stagingAlloc{
      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
               VMA_ALLOCATION_CREATE_MAPPED_BIT,
      .usage = VMA_MEMORY_USAGE_AUTO,
      .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
  };
  VkBuffer staging = VK_NULL_HANDLE;
  VmaAllocation stagingAllocHandle = VK_NULL_HANDLE;
  VmaAllocationInfo mapped{};
  if (vmaCreateBuffer(impl.device.allocator, &stagingInfo, &stagingAlloc,
                      &staging, &stagingAllocHandle, &mapped) != VK_SUCCESS) {
    image.destroy(impl.device.allocator, impl.device.device);
    return kInvalidTextureHandle;
  }
  std::memcpy(mapped.pMappedData, desc.rgba, layerBytes * desc.layers);

  const VkCommandPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      .queueFamilyIndex = impl.device.graphicsFamily,
  };
  VkCommandPool pool = VK_NULL_HANDLE;
  if (!KE_VK(
          vkCreateCommandPool(impl.device.device, &poolInfo, nullptr, &pool))) {
    vmaDestroyBuffer(impl.device.allocator, staging, stagingAllocHandle);
    image.destroy(impl.device.allocator, impl.device.device);
    return kInvalidTextureHandle;
  }
  const VkCommandBufferAllocateInfo cbInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  KE_VK(vkAllocateCommandBuffers(impl.device.device, &cbInfo, &cmd));
  const VkCommandBufferBeginInfo begin{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  KE_VK(vkBeginCommandBuffer(cmd, &begin));

  // Every barrier spans ALL array layers (the review's layer-0-only
  // bug: copies/blits/draws touch layer 1..N-1 too).
  const auto barrier = [&](u32 baseMip, u32 mipCount, VkImageLayout oldL,
                           VkImageLayout newL) {
    CommandContext::pipelineBarrier(
        cmd, CommandContext::imageBarrier(image.image, oldL, newL,
                                          VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                          VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                          baseMip, mipCount, 0, desc.layers));
  };
  barrier(0, mips, VK_IMAGE_LAYOUT_UNDEFINED,
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

  std::vector<VkBufferImageCopy> copies(desc.layers);
  for (u32 l = 0; l < desc.layers; ++l) {
    copies[l] = VkBufferImageCopy{
        .bufferOffset = static_cast<VkDeviceSize>(layerBytes * l),
        .imageSubresource =
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = l,
                .layerCount = 1,
            },
        .imageExtent = {desc.width, desc.height, 1},
    };
  }
  vkCmdCopyBufferToImage(cmd, staging, image.image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         static_cast<u32>(copies.size()), copies.data());

  for (u32 m = 1; m < mips; ++m) {
    barrier(m - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    const u32 sw = (std::max)(desc.width >> (m - 1), 1u);
    const u32 sh = (std::max)(desc.height >> (m - 1), 1u);
    const u32 dw = (std::max)(desc.width >> m, 1u);
    const u32 dh = (std::max)(desc.height >> m, 1u);
    for (u32 l = 0; l < desc.layers; ++l) {
      const VkImageBlit region{
          .srcSubresource =
              {
                  .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                  .mipLevel = m - 1,
                  .baseArrayLayer = l,
                  .layerCount = 1,
              },
          .srcOffsets = {{0, 0, 0},
                         {static_cast<i32>(sw), static_cast<i32>(sh), 1}},
          .dstSubresource =
              {
                  .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                  .mipLevel = m,
                  .baseArrayLayer = l,
                  .layerCount = 1,
              },
          .dstOffsets = {{0, 0, 0},
                         {static_cast<i32>(dw), static_cast<i32>(dh), 1}},
      };
      vkCmdBlitImage(cmd, image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                     &region, VK_FILTER_LINEAR);
    }
  }
  // Final read transitions order against FRAGMENT sampling (the
  // upload's queue-wait below already covers it host-side; this keeps
  // the barrier scopes semantically correct for any future reuse).
  if (mips > 1)
    CommandContext::pipelineBarrier(
        cmd,
        CommandContext::imageBarrier(
            image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, 0, mips - 1, 0, desc.layers));
  CommandContext::pipelineBarrier(
      cmd,
      CommandContext::imageBarrier(
          image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
          VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
          VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, mips - 1, 1, 0, desc.layers));

  KE_VK(vkEndCommandBuffer(cmd));
  const VkCommandBufferSubmitInfo cbSubmit{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = cmd,
  };
  const VkSubmitInfo2 submit{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &cbSubmit,
  };
  KE_VK(vkQueueSubmit2(impl.device.graphicsQueue, 1, &submit, VK_NULL_HANDLE));
  KE_VK(vkQueueWaitIdle(impl.device.graphicsQueue));
  vkDestroyCommandPool(impl.device.device, pool, nullptr);
  vmaDestroyBuffer(impl.device.allocator, staging, stagingAllocHandle);

  if (desc.debugName)
    vkutil::setName(impl.device.device, VK_OBJECT_TYPE_IMAGE,
                    reinterpret_cast<uint64_t>(image.image), desc.debugName);
  for (u32 i = 0; i < impl.textures.size(); ++i) {
    if (!impl.textureAlive[i]) {
      impl.textures[i] = std::move(image);
      impl.textureAlive[i] = true;
      return i;
    }
  }
  impl.textures.push_back(std::move(image));
  impl.textureAlive.push_back(true);
  return static_cast<u32>(impl.textures.size() - 1);
}

void VulkanBackend::destroyTexture(u32 texture) {
  Impl& impl = *m_impl;
  if (texture >= impl.textures.size() || !impl.textureAlive[texture]) {
    KENSHIN_ERROR("[vulkan] destroyTexture: invalid handle {}", texture);
    return;
  }
  // Static scene resources only (v1): draining the device is heavy but
  // correct - textures are destroyed at scene teardown, not per frame.
  vkDeviceWaitIdle(impl.device.device);
  impl.textures[texture].destroy(impl.device.allocator, impl.device.device);
  impl.textureAlive[texture] = false;
}

u32 VulkanBackend::createGBufferPipeline(const FullscreenPipelineInfo& info) {
  return m_impl->pipelines.createGBuffer(m_impl->device.device, info);
}

void VulkanBackend::drawGBuffer(u32 pipeline, const u32* bufferHandles,
                                u32 bufferCount, u32 albedoTexture,
                                const void* push, usize pushBytes, u32 width,
                                u32 height) {
  Impl& impl = *m_impl;
  if (!impl.frameActive || pipeline == kInvalidPipelineHandle ||
      pipeline > impl.pipelines.fullscreenCount())
    return;
  const PipelineStore::Fullscreen& fs = impl.pipelines.fullscreen(pipeline);
  if (!fs.valid) return;
  FrameSlot& slot = impl.frameSync.slots[impl.frameSync.frameIndex];
  if (slot.gbuffer[0].view == VK_NULL_HANDLE) return;

  std::vector<VkDescriptorBufferInfo> infos(bufferCount);
  for (u32 i = 0; i < bufferCount; ++i) {
    if (!impl.buffers.valid(bufferHandles[i])) return;
    infos[i] = VkDescriptorBufferInfo{
        .buffer = impl.buffers.buffer(bufferHandles[i]),
        .offset = 0,
        .range = impl.buffers.size(bufferHandles[i]),
    };
  }
  // The albedo binding target: the scene texture when valid, the
  // backend's 1x1 dummy otherwise (the mixed set must always be
  // validly written - the SPIR-V sample opcode exists regardless of
  // whether any material references a texture).
  const GpuImage* albedo = &impl.dummyTexture;
  if (albedoTexture < impl.textures.size() && impl.textureAlive[albedoTexture])
    albedo = &impl.textures[albedoTexture];
  else if (albedoTexture != kInvalidTextureHandle)
    KENSHIN_ERROR("[vulkan] drawGBuffer: bad albedo texture {} (dummy)",
                  albedoTexture);

  // The GBuffer targets cycle UNDEFINED/COLOR_ATTACHMENT (this pass) ->
  // SHADER_READ_ONLY (after it) -> COLOR_ATTACHMENT (next frame).
  // DONT_CARE-style: discard last frame's contents on the way in.
  impl.cmd.endPass();
  for (u32 g = 0; g < 3; ++g)
    CommandContext::pipelineBarrier(
        impl.cmd.cmd(), CommandContext::imageBarrier(
                            slot.gbuffer[g].image, VK_IMAGE_LAYOUT_UNDEFINED,
                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT));
  {
    const vkutil::ScopedDebugLabel label{impl.cmd.cmd(), "gbuffer-decode"};
    impl.cmd.beginGBufferPass(slot.gbuffer[0].view, slot.gbuffer[1].view,
                              slot.gbuffer[2].view, width, height);
    // Mixed binding writes (M8.2): STORAGE_BUFFER bindings take the
    // buffer handles in ascending order; the SAMPLED_IMAGE binding
    // gets the albedo view and the SAMPLER binding the albedo sampler.
    const VkDescriptorSet set = fs.sets[impl.frameSync.frameIndex];
    VkDescriptorImageInfo albedoImage{
        .sampler = VK_NULL_HANDLE,
        .imageView = albedo->view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    VkDescriptorImageInfo albedoSamplerInfo{
        .sampler = impl.albedoSampler,
        .imageView = VK_NULL_HANDLE,
        .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    std::vector<VkWriteDescriptorSet> writes(fs.bindingCount);
    u32 nextBuffer = 0;
    bool bindMismatch = false;
    for (u32 b = 0; b < fs.bindingCount && !bindMismatch; ++b) {
      writes[b] = VkWriteDescriptorSet{
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = set,
          .dstBinding = b,
          .descriptorCount = 1,
          .descriptorType = fs.bindingTypes[b],
      };
      switch (fs.bindingTypes[b]) {
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
          writes[b].pImageInfo = &albedoImage;
          break;
        case VK_DESCRIPTOR_TYPE_SAMPLER:
          writes[b].pImageInfo = &albedoSamplerInfo;
          break;
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
          // fsGBuffer uses separate image+sampler; a combined binding
          // (if one ever appears) gets view + albedo sampler.
          albedoImage.sampler = impl.albedoSampler;
          writes[b].pImageInfo = &albedoImage;
          break;
        default:
          // Never early-return here: the GBuffer pass is OPEN - a
          // mismatch means a programming error; log once, skip the
          // draw, and let the scope close below.
          if (nextBuffer >= bufferCount) {
            KENSHIN_ERROR("[vulkan] drawGBuffer: {} handles < {} bindings",
                          bufferCount, fs.bindingCount);
            bindMismatch = true;
            break;
          }
          writes[b].pBufferInfo = &infos[nextBuffer++];
          break;
      }
    }
    if (!bindMismatch) {
      vkUpdateDescriptorSets(impl.device.device, fs.bindingCount, writes.data(),
                             0, nullptr);
      vkCmdBindDescriptorSets(impl.cmd.cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                              fs.layout, 0, 1, &set, 0, nullptr);
      vkCmdBindPipeline(impl.cmd.cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                        fs.pipeline);
      vkCmdPushConstants(
          impl.cmd.cmd(), fs.layout,
          VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
          static_cast<u32>(pushBytes), push);
      vkCmdDraw(impl.cmd.cmd(), 3, 1, 0, 0);
    }
    impl.cmd.endPass();
  }
  for (u32 g = 0; g < 3; ++g)
    CommandContext::pipelineBarrier(
        impl.cmd.cmd(),
        CommandContext::imageBarrier(slot.gbuffer[g].image,
                                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT));
}

u32 VulkanBackend::createLightingPipeline(const FullscreenPipelineInfo& info) {
  // M8.3: linear HDR target (headless frames never create these
  // pipelines).
  return m_impl->pipelines.createLighting(m_impl->device.device, info);
}

u32 VulkanBackend::createTonemapPipeline(const FullscreenPipelineInfo& info) {
  return m_impl->pipelines.createTonemap(m_impl->device.device,
                                         m_impl->swapchain.format, info);
}

u32 VulkanBackend::createTaaPipeline(const FullscreenPipelineInfo& info) {
  return m_impl->pipelines.createTaa(m_impl->device.device, info);
}

u32 VulkanBackend::createTranslucentPipeline(
    const FullscreenPipelineInfo& info) {
  return m_impl->pipelines.createTranslucent(m_impl->device.device, info);
}

void VulkanBackend::drawTranslucent(u32 pipeline, const u32* bufferHandles,
                                    u32 bufferCount, u32 envTexture,
                                    u32 albedoTexture, const void* push,
                                    usize pushBytes, u32 vertexCount,
                                    u32 instanceCount, u32 width, u32 height) {
  Impl& impl = *m_impl;
  if (!impl.frameActive || pipeline == kInvalidPipelineHandle ||
      pipeline > impl.pipelines.fullscreenCount() || vertexCount == 0 ||
      instanceCount == 0 || width == 0 || height == 0)
    return;
  const PipelineStore::Fullscreen& fs = impl.pipelines.fullscreen(pipeline);
  if (!fs.valid || fs.bindingCount == 0) return;
  FrameSlot& slot = impl.frameSync.slots[impl.frameSync.frameIndex];
  if (slot.hdr.view == VK_NULL_HANDLE) return;
  std::vector<VkDescriptorBufferInfo> infos(bufferCount);
  for (u32 i = 0; i < bufferCount; ++i) {
    if (!impl.buffers.valid(bufferHandles[i])) return;
    infos[i] = VkDescriptorBufferInfo{
        .buffer = impl.buffers.buffer(bufferHandles[i]),
        .offset = 0,
        .range = impl.buffers.size(bufferHandles[i]),
    };
  }
  const VkDescriptorImageInfo envInfo{
      .sampler = impl.albedoSampler,
      .imageView =
          envTexture < impl.textures.size() && impl.textureAlive[envTexture]
              ? impl.textures[envTexture].view
              : impl.dummyTexture.view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
  };
  // The scene's albedo/MR array (or the dummy - the set must always be
  // validly written) as SAMPLED_IMAGE + SAMPLER, decode's pairing.
  const VkDescriptorImageInfo albedoInfo{
      .sampler = VK_NULL_HANDLE,
      .imageView = albedoTexture < impl.textures.size() &&
                           impl.textureAlive[albedoTexture]
                       ? impl.textures[albedoTexture].view
                       : impl.dummyTexture.view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
  };
  const VkDescriptorImageInfo albedoSamplerInfo{
      .sampler = impl.albedoSampler,
      .imageView = VK_NULL_HANDLE,
      .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };
  // Binding 10: GBuffer C (the opaque emission, SHADER_READ since
  // drawLighting) for the emissive pass-through.
  if (slot.gbuffer[2].view == VK_NULL_HANDLE) return;
  const VkDescriptorImageInfo gbufferCInfo{
      .sampler = impl.linearSampler,
      .imageView = slot.gbuffer[2].view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
  };
  // Validate the binding map BEFORE opening the pass.
  std::vector<VkWriteDescriptorSet> writes(fs.bindingCount);
  const VkDescriptorSet set = fs.sets[impl.frameSync.frameIndex];
  u32 nextBuffer = 0;
  for (u32 b = 0; b < fs.bindingCount; ++b) {
    writes[b] = VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .dstBinding = b,
        .descriptorCount = 1,
        .descriptorType = fs.bindingTypes[b],
    };
    if (fs.bindingTypes[b] == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
      writes[b].pImageInfo = b == 10 ? &gbufferCInfo : &envInfo;
    } else if (fs.bindingTypes[b] == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) {
      writes[b].pImageInfo = &albedoInfo;
    } else if (fs.bindingTypes[b] == VK_DESCRIPTOR_TYPE_SAMPLER) {
      writes[b].pImageInfo = &albedoSamplerInfo;
    } else {
      if (nextBuffer >= bufferCount) {
        KENSHIN_ERROR("[vulkan] drawTranslucent: {} handles < {} bindings",
                      bufferCount, fs.bindingCount);
        return;
      }
      writes[b].pBufferInfo = &infos[nextBuffer++];
    }
  }

  // The HDR target sits in SHADER_READ after drawLighting: LOAD it (the
  // lit opaque frame is the blend destination), then park it back for
  // TAA/tonemap.
  impl.cmd.endPass();
  CommandContext::pipelineBarrier(
      impl.cmd.cmd(),
      CommandContext::imageBarrier(
          slot.hdr.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
          VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
              VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT));
  {
    const vkutil::ScopedDebugLabel label{impl.cmd.cmd(), "translucent"};
    impl.cmd.beginColorPass(slot.hdr.view, width, height,
                            VK_ATTACHMENT_LOAD_OP_LOAD,
                            VkRect2D{{0, 0}, {width, height}}, 0.0f);
    vkUpdateDescriptorSets(impl.device.device, fs.bindingCount, writes.data(),
                           0, nullptr);
    vkCmdBindDescriptorSets(impl.cmd.cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                            fs.layout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(impl.cmd.cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                      fs.pipeline);
    vkCmdPushConstants(
        impl.cmd.cmd(), fs.layout,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
        static_cast<u32>(pushBytes), push);
    vkCmdDraw(impl.cmd.cmd(), vertexCount, instanceCount, 0, 0);
    impl.cmd.endPass();
  }
  CommandContext::pipelineBarrier(
      impl.cmd.cmd(),
      CommandContext::imageBarrier(slot.hdr.image,
                                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_SAMPLED_READ_BIT));
}

u32 VulkanBackend::createBloomPipeline(const FullscreenPipelineInfo& info,
                                       bool additive) {
  return m_impl->pipelines.createBloom(m_impl->device.device, info, additive);
}

bool VulkanBackend::drawBloom(const u32* downPipelines,
                              const u32* upPipelines, u32 levels, bool fromTaa,
                              u32 width, u32 height) {
  Impl& impl = *m_impl;
  levels = (std::min)(levels, Impl::kBloomLevels);
  if (!impl.frameActive || levels < 2 || width < 4 || height < 4) return false;
  if (fromTaa && !impl.taaActive) return false;
  FrameSlot& slot = impl.frameSync.slots[impl.frameSync.frameIndex];
  if (slot.hdr.view == VK_NULL_HANDLE) return false;
  for (u32 k = 0; k < levels; ++k) {
    const u32 d = downPipelines[k];
    if (d == kInvalidPipelineHandle || d > impl.pipelines.fullscreenCount() ||
        !impl.pipelines.fullscreen(d).valid)
      return false;
    if (k + 1 < levels) {
      const u32 u = upPipelines[k];
      if (u == kInvalidPipelineHandle || u > impl.pipelines.fullscreenCount() ||
          !impl.pipelines.fullscreen(u).valid)
        return false;
    }
  }
  // (Re)create the chain on size change - deferred destruction, the
  // TAA pair's discipline (an in-flight frame may still sample it).
  if (impl.bloomW != width || impl.bloomH != height ||
      impl.bloom[0].image == VK_NULL_HANDLE) {
    for (u32 k = 0; k < Impl::kBloomLevels; ++k) {
      if (impl.bloom[k].image != VK_NULL_HANDLE) {
        const VkImage deadImage = impl.bloom[k].image;
        const VkImageView deadView = impl.bloom[k].view;
        const VmaAllocation deadAlloc = impl.bloom[k].allocation;
        const VmaAllocator alloc = impl.device.allocator;
        const VkDevice dev = impl.device.device;
        slot.deletionQueue.push([deadImage, deadView, deadAlloc, alloc, dev] {
          vkDestroyImage(dev, deadImage, nullptr);
          vkDestroyImageView(dev, deadView, nullptr);
          vmaFreeMemory(alloc, deadAlloc);
        });
        impl.bloom[k] = GpuImage{};
      }
      const u32 w = (std::max)(width >> (k + 1), 1u);
      const u32 h = (std::max)(height >> (k + 1), 1u);
      if (!impl.bloom[k].create(
              impl.device.allocator, impl.device.device, w, h,
              VK_FORMAT_R16G16B16A16_SFLOAT,
              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
        KENSHIN_ERROR("[vulkan] bloom target creation failed");
        impl.bloomW = impl.bloomH = 0;
        return false;
      }
    }
    impl.bloomW = width;
    impl.bloomH = height;
  }

  const auto levelW = [&](u32 k) { return (std::max)(width >> (k + 1), 1u); };
  const auto levelH = [&](u32 k) { return (std::max)(height >> (k + 1), 1u); };
  const VkImageView sceneView =
      fromTaa ? impl.taa[impl.taaParity].view : slot.hdr.view;
  // One pass: target view/extent, source view, pipeline, push, load op.
  const auto pass = [&](u32 pipeline, VkImageView srcView, u32 srcW, u32 srcH,
                        u32 dstK, bool firstDown, bool additive,
                        float radius) {
    const PipelineStore::Fullscreen& fs = impl.pipelines.fullscreen(pipeline);
    const VkDescriptorSet set = fs.sets[impl.frameSync.frameIndex];
    const VkDescriptorImageInfo srcInfo{
        .sampler = impl.linearSampler,
        .imageView = srcView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    std::vector<VkWriteDescriptorSet> writes;
    for (u32 b = 0; b < fs.bindingCount; ++b)
      if (fs.bindingTypes[b] == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
        writes.push_back(VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = b,
            .descriptorCount = 1,
            .descriptorType = fs.bindingTypes[b],
            .pImageInfo = &srcInfo,
        });
    vkUpdateDescriptorSets(impl.device.device, static_cast<u32>(writes.size()),
                           writes.data(), 0, nullptr);
    // Down: the target's old contents are irrelevant (UNDEFINED); up:
    // the target keeps its downsample - the additive blend accumulates.
    CommandContext::pipelineBarrier(
        impl.cmd.cmd(),
        CommandContext::imageBarrier(
            impl.bloom[dstK].image,
            additive ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                     : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT));
    const u32 w = levelW(dstK), h = levelH(dstK);
    impl.cmd.beginColorPass(impl.bloom[dstK].view, w, h,
                            additive ? VK_ATTACHMENT_LOAD_OP_LOAD
                                     : VK_ATTACHMENT_LOAD_OP_CLEAR,
                            VkRect2D{{0, 0}, {w, h}}, 0.0f);
    vkCmdBindDescriptorSets(impl.cmd.cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                            fs.layout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(impl.cmd.cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                      fs.pipeline);
    const float push[8] = {1.0f / float(srcW), 1.0f / float(srcH),
                           firstDown ? 1.0f : 0.0f, 0.0f,
                           1.0f / float(w), 1.0f / float(h), radius, 0.0f};
    vkCmdPushConstants(
        impl.cmd.cmd(), fs.layout,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
        sizeof(push), push);
    vkCmdDraw(impl.cmd.cmd(), 3, 1, 0, 0);
    impl.cmd.endPass();
    CommandContext::pipelineBarrier(
        impl.cmd.cmd(),
        CommandContext::imageBarrier(impl.bloom[dstK].image,
                                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT));
  };

  impl.cmd.endPass();
  const vkutil::ScopedDebugLabel label{impl.cmd.cmd(), "bloom"};
  for (u32 k = 0; k < levels; ++k)
    pass(downPipelines[k], k == 0 ? sceneView : impl.bloom[k - 1].view,
         k == 0 ? width : levelW(k - 1), k == 0 ? height : levelH(k - 1), k,
         k == 0, false, 1.0f);
  for (u32 k = levels - 1; k >= 1; --k)
    pass(upPipelines[k - 1], impl.bloom[k].view, levelW(k), levelH(k), k - 1,
         false, true, 1.0f);
  impl.bloomActive = true;
  return true;
}

u32 VulkanBackend::createExposurePipeline(const ComputePipelineInfo& info) {
  return m_impl->pipelines.createCompute(m_impl->device.device, info, true);
}

bool VulkanBackend::drawExposure(u32 histogramPipeline, u32 adaptPipeline,
                                 u32 histogram, u32 state, u32 envTexture,
                                 bool fromTaa, u32 width, u32 height,
                                 const void* push, usize pushBytes) {
  Impl& impl = *m_impl;
  if (!impl.frameActive || width == 0 || height == 0 || pushBytes > 128)
    return false;
  if (fromTaa && !impl.taaActive) return false;
  const auto computeOk = [&](u32 h) {
    return h != kInvalidPipelineHandle && h != 0 &&
           h <= impl.pipelines.computeCount() &&
           impl.pipelines.compute(h).valid &&
           impl.pipelines.compute(h).bindingCount > 0;
  };
  if (!computeOk(histogramPipeline) || !computeOk(adaptPipeline) ||
      !impl.buffers.valid(histogram) || !impl.buffers.valid(state))
    return false;
  FrameSlot& slot = impl.frameSync.slots[impl.frameSync.frameIndex];
  if (slot.hdr.view == VK_NULL_HANDLE) return false;
  const VkImage sceneImage =
      fromTaa ? impl.taa[impl.taaParity].image : slot.hdr.image;
  const VkImageView sceneView =
      fromTaa ? impl.taa[impl.taaParity].view : slot.hdr.view;

  impl.cmd.endPass();
  const VkCommandBuffer cb = impl.cmd.cmd();
  const vkutil::ScopedDebugLabel label{cb, "auto exposure"};
  const auto memoryBarrier = [&](VkPipelineStageFlags2 srcStage,
                                 VkAccessFlags2 srcAccess,
                                 VkPipelineStageFlags2 dstStage,
                                 VkAccessFlags2 dstAccess) {
    const VkMemoryBarrier2 mb{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = srcStage,
        .srcAccessMask = srcAccess,
        .dstStageMask = dstStage,
        .dstAccessMask = dstAccess,
    };
    const VkDependencyInfo dep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &mb,
    };
    vkCmdPipelineBarrier2(cb, &dep);
  };
  // The scene HDR reached SHADER_READ with a FRAGMENT-stage destination
  // (TAA / lighting); widen the visibility to this compute read.
  CommandContext::pipelineBarrier(
      cb, CommandContext::imageBarrier(
              sceneImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
              VK_ACCESS_2_SHADER_SAMPLED_READ_BIT));
  // Clear the histogram: WAR vs the previous frame's adapt read (one
  // queue - barrier scopes span submission order), then fill -> compute.
  memoryBarrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT);
  vkCmdFillBuffer(cb, impl.buffers.buffer(histogram), 0, VK_WHOLE_SIZE, 0u);
  memoryBarrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

  const VkDescriptorImageInfo sceneInfo{
      .sampler = impl.linearSampler,
      .imageView = sceneView,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
  };
  // Binding 3 (Sampler2DArray envMap) needs an ARRAYED view - a 2D view
  // there is a view-type mismatch (the IBL cache-poison lesson). Without
  // an env texture the auto exposure does not run.
  if (!(envTexture < impl.textures.size() && impl.textureAlive[envTexture]))
    return false;
  const VkDescriptorImageInfo envInfo{
      .sampler = impl.linearSampler,
      .imageView = impl.textures[envTexture].view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
  };
  const u32 bindingBuffers[3] = {histogram, histogram, state};
  const auto dispatch = [&](u32 handle, u32 gx, u32 gy) {
    const PipelineStore::Compute& cp = impl.pipelines.compute(handle);
    const VkDescriptorSet set = cp.sets[impl.frameSync.frameIndex];
    std::vector<VkDescriptorBufferInfo> bufInfos(cp.bindingCount);
    std::vector<VkWriteDescriptorSet> writes(cp.bindingCount);
    for (u32 b = 0; b < cp.bindingCount; ++b) {
      writes[b] = VkWriteDescriptorSet{
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = set,
          .dstBinding = b,
          .descriptorCount = 1,
          .descriptorType = cp.bindingTypes[b],
      };
      if (cp.bindingTypes[b] == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
        writes[b].pImageInfo = b == 0 ? &sceneInfo : &envInfo;
      } else {
        const u32 buf = bindingBuffers[(std::min)(b, 2u)];
        bufInfos[b] = VkDescriptorBufferInfo{
            .buffer = impl.buffers.buffer(buf),
            .offset = 0,
            .range = impl.buffers.size(buf),
        };
        writes[b].pBufferInfo = &bufInfos[b];
      }
    }
    vkUpdateDescriptorSets(impl.device.device, cp.bindingCount, writes.data(),
                           0, nullptr);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cp.pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, cp.layout, 0,
                            1, &set, 0, nullptr);
    vkCmdPushConstants(cb, cp.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       static_cast<u32>(pushBytes), push);
    vkCmdDispatch(cb, gx, gy, 1);
  };
  dispatch(histogramPipeline, (width + 15) / 16, (height + 15) / 16);
  memoryBarrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
  dispatch(adaptPipeline, 1, 1);
  // state -> the tonemap's fragment read (and the host's panel readout
  // after the frame fence).
  memoryBarrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                    VK_PIPELINE_STAGE_2_HOST_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                    VK_ACCESS_2_HOST_READ_BIT);
  return true;
}

u32 VulkanBackend::createShadowDecodePipeline(
    const FullscreenPipelineInfo& info) {
  return m_impl->pipelines.createShadowDecode(m_impl->device.device, info);
}

void VulkanBackend::drawLighting(u32 pipeline, const u32* bufferHandles,
                                 u32 bufferCount, const void* push,
                                 usize pushBytes, u32 width, u32 height,
                                 u32 envTexture) {
  Impl& impl = *m_impl;
  if (!impl.frameActive || pipeline == kInvalidPipelineHandle ||
      pipeline > impl.pipelines.fullscreenCount())
    return;
  const PipelineStore::Fullscreen& fs = impl.pipelines.fullscreen(pipeline);
  if (!fs.valid || fs.bindingCount == 0) return;
  FrameSlot& slot = impl.frameSync.slots[impl.frameSync.frameIndex];
  if (slot.gbuffer[0].view == VK_NULL_HANDLE ||
      slot.gbuffer[1].view == VK_NULL_HANDLE ||
      slot.gbuffer[2].view == VK_NULL_HANDLE || slot.hdr.view == VK_NULL_HANDLE)
    return;

  // Buffer infos for the StorageBuffer bindings (consumed in ascending
  // binding order after the image bindings): lights, materials, env.
  std::vector<VkDescriptorBufferInfo> infos(bufferCount);
  for (u32 i = 0; i < bufferCount; ++i) {
    if (!impl.buffers.valid(bufferHandles[i])) return;
    infos[i] = VkDescriptorBufferInfo{
        .buffer = impl.buffers.buffer(bufferHandles[i]),
        .offset = 0,
        .range = impl.buffers.size(bufferHandles[i]),
    };
  }

  // Frame-1 guard: the shadow targets are created lazily by the first
  // drawShadowDecode; on the very first frame the shadow chain hasn't
  // produced a VisBuffer yet, so no target exists and the sampler
  // bindings below would be null. Bail BEFORE touching the HDR target
  // so its layout cycle stays consistent for the tonemap pass - one
  // unlit frame at startup.
  if (impl.shadowMap[0].image == VK_NULL_HANDLE) {
    // Park the HDR target in SHADER_READ (the tonemap pass samples it
    // unconditionally) - the layout cycle below is what normally does
    // this; the skipped frame must not leave it UNDEFINED.
    CommandContext::pipelineBarrier(
        impl.cmd.cmd(),
        CommandContext::imageBarrier(slot.hdr.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT));
    return;
  }
  // M8.3: the HDR target cycles UNDEFINED -> COLOR_ATTACHMENT (this
  // pass) -> SHADER_READ_ONLY (the tonemap sample right after);
  // depth-less fullscreen scope.
  impl.cmd.endPass();
  CommandContext::pipelineBarrier(
      impl.cmd.cmd(), CommandContext::imageBarrier(
                          slot.hdr.image, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                          VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT));
  {
    const vkutil::ScopedDebugLabel label{impl.cmd.cmd(), "lighting"};
    impl.cmd.beginColorPass(slot.hdr.view, width, height);

    const VkDescriptorSet set = fs.sets[impl.frameSync.frameIndex];
    // Image infos: the reflected Sampler2D bindings (ascending) get
    // this slot's GBuffer views in order (A, then B), and the third
    // the shadow map (NEAREST - depth comparisons must not filter).
    const VkImageView shadowView0 = impl.shadowMap[0].view;
    VkDescriptorImageInfo imageInfos[7] = {
        VkDescriptorImageInfo{
            .sampler = impl.linearSampler,
            .imageView = slot.gbuffer[0].view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        },
        VkDescriptorImageInfo{
            .sampler = impl.linearSampler,
            .imageView = slot.gbuffer[1].view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        },
        // The three clipmap levels, all on the NEAREST sampler.
        // Un-created levels (levelCount < 3, first frame) fall back
        // to level 0's view - the set must be VALIDLY written.
        VkDescriptorImageInfo{
            .sampler = impl.shadowSampler,
            .imageView = shadowView0,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        },
        VkDescriptorImageInfo{
            .sampler = impl.shadowSampler,
            .imageView = impl.shadowMap[1].image != VK_NULL_HANDLE
                             ? impl.shadowMap[1].view
                             : shadowView0,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        },
        VkDescriptorImageInfo{
            .sampler = impl.shadowSampler,
            .imageView = impl.shadowMap[2].image != VK_NULL_HANDLE
                             ? impl.shadowMap[2].view
                             : shadowView0,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        },
        // The IBL environment probe (mip-filtered sampler - the mip
        // chain is the prefiltered radiance LOD ladder; the dummy
        // keeps the set valid when no env was created).
        VkDescriptorImageInfo{
            .sampler = impl.albedoSampler,
            .imageView = envTexture < impl.textures.size() &&
                                 impl.textureAlive[envTexture]
                             ? impl.textures[envTexture].view
                             : impl.dummyTexture.view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        },
        // GBuffer C: emitted radiance (binding 9 - after the env probe in
        // ascending binding order).
        VkDescriptorImageInfo{
            .sampler = impl.linearSampler,
            .imageView = slot.gbuffer[2].view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        },
    };
    std::vector<VkWriteDescriptorSet> writes(fs.bindingCount);
    u32 nextImage = 0, nextBuffer = 0;
    bool bindMismatch = false;
    for (u32 b = 0; b < fs.bindingCount && !bindMismatch; ++b) {
      writes[b] = VkWriteDescriptorSet{
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = set,
          .dstBinding = b,
          .descriptorCount = 1,
          .descriptorType = fs.bindingTypes[b],
      };
      if (fs.bindingTypes[b] == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
        // GBuffer A/B, the three clipmap shadow maps, the IBL
        // environment probe, then GBuffer C (emissive) - the shader
        // declares exactly these seven sampler bindings.
        if (nextImage >= 7) {
          bindMismatch = true;
          break;
        }
        writes[b].pImageInfo = &imageInfos[nextImage++];
      } else {
        if (nextBuffer >= bufferCount) {
          bindMismatch = true;
          break;
        }
        writes[b].pBufferInfo = &infos[nextBuffer++];
      }
    }
    if (bindMismatch)
      KENSHIN_ERROR("[vulkan] drawLighting: {} handles < {} bindings",
                    bufferCount, fs.bindingCount);
    else {
      vkUpdateDescriptorSets(impl.device.device, fs.bindingCount, writes.data(),
                             0, nullptr);
      vkCmdBindDescriptorSets(impl.cmd.cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                              fs.layout, 0, 1, &set, 0, nullptr);
      vkCmdBindPipeline(impl.cmd.cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                        fs.pipeline);
      vkCmdPushConstants(
          impl.cmd.cmd(), fs.layout,
          VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
          static_cast<u32>(pushBytes), push);
      vkCmdDraw(impl.cmd.cmd(), 3, 1, 0, 0);
    }
    impl.cmd.endPass();
  }
  CommandContext::pipelineBarrier(
      impl.cmd.cmd(),
      CommandContext::imageBarrier(slot.hdr.image,
                                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_SAMPLED_READ_BIT));
}

// M9.1: the shadow depth decode - the shadow chain's VisBuffer
// rendered (fsShadowDepth) into the persistent R32F shadow target.
// When shadows are disabled the target still transitions to
// SHADER_READ (cleared to one - lighting's PCF then reads "far"
// everywhere and multiplies by 1.0), so drawLighting's binding stays
// valid either way.
void VulkanBackend::drawShadowDecode(u32 pipeline, const u32* bufferHandles,
                                     u32 bufferCount, const void* push,
                                     usize pushBytes, u32 width, u32 height,
                                     u32 level, bool shadowEnabled,
                                     const u32 scissorRect[4]) {
  Impl& impl = *m_impl;
  if (!impl.frameActive || pipeline == kInvalidPipelineHandle ||
      pipeline > impl.pipelines.fullscreenCount())
    return;
  const PipelineStore::Fullscreen& fs = impl.pipelines.fullscreen(pipeline);
  if (!fs.valid || fs.bindingCount == 0) return;
  if (impl.shadowSampler == VK_NULL_HANDLE) return;  // unsupported format
  if (width == 0 || height == 0) return;
  if (level >= 3) return;

  // (Re)create the fixed-size target on first use or size change.
  if (!impl.shadowMap[level].matches(impl.shadowMapSize, impl.shadowMapSize,
                                     VK_FORMAT_R32_SFLOAT)) {
    if (impl.shadowMap[level].image != VK_NULL_HANDLE)
      impl.shadowMap[level].destroy(impl.device.allocator, impl.device.device);
    if (!impl.shadowMap[level].create(
            impl.device.allocator, impl.device.device, impl.shadowMapSize,
            impl.shadowMapSize, VK_FORMAT_R32_SFLOAT,
            // TRANSFER_DST: the shadows-off path clears to "far" with
            // vkCmdClearColorImage (missing, it was a validation error
            // whenever shadows were disabled).
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      KENSHIN_ERROR("[vulkan] shadow target creation failed");
      return;
    }
    vkutil::setName(impl.device.device, VK_OBJECT_TYPE_IMAGE,
                    reinterpret_cast<uint64_t>(impl.shadowMap[level].image),
                    "shadow-map");
    impl.shadowMapLayout[level] = VK_IMAGE_LAYOUT_UNDEFINED;
    // Region-limited rewrites (the page cache) LOAD the prior content;
    // a fresh target must start "far" (= 1.0, no occluder -> lit) so
    // never-rendered tiles sample lit, matching the clear-to-far path.
    {
      CommandContext::pipelineBarrier(
          impl.cmd.cmd(),
          CommandContext::imageBarrier(
              impl.shadowMap[level].image, VK_IMAGE_LAYOUT_UNDEFINED,
              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
              VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
              VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT));
      impl.cmd.beginColorPass(impl.shadowMap[level].view, width, height,
                              VK_ATTACHMENT_LOAD_OP_CLEAR,
                              VkRect2D{{0, 0}, {width, height}}, 1.0f);
      impl.cmd.endPass();
      impl.shadowMapLayout[level] = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
  }

  // Invalid inputs (the shadow chain has not run - e.g. --no-shadows)
  // take the clear-to-far path instead of returning: the target must
  // still end in SHADER_READ, which drawLighting binds unconditionally
  // (an early return here left a fresh target in COLOR_ATTACHMENT - a
  // validation error every frame with shadows off).
  std::vector<VkDescriptorBufferInfo> infos(bufferCount);
  for (u32 i = 0; i < bufferCount && shadowEnabled; ++i) {
    if (!impl.buffers.valid(bufferHandles[i])) {
      shadowEnabled = false;
      break;
    }
    infos[i] = VkDescriptorBufferInfo{
        .buffer = impl.buffers.buffer(bufferHandles[i]),
        .offset = 0,
        .range = impl.buffers.size(bufferHandles[i]),
    };
  }

  impl.cmd.endPass();
  if (shadowEnabled) {
    CommandContext::pipelineBarrier(
        impl.cmd.cmd(),
        CommandContext::imageBarrier(
            impl.shadowMap[level].image, impl.shadowMapLayout[level],
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT));
    {
      const vkutil::ScopedDebugLabel label{impl.cmd.cmd(), "shadow-decode"};
      // Stage C page cache: a scissor limits the rewrite to the dirty
      // region; everything else LOADs its cached content. Null rect =
      // full target (CLEAR-to-far keeps the old whole-map semantics -
      // the disabled/toggle path relies on it).
      const bool full = scissorRect == nullptr;
      const VkAttachmentLoadOp loadOp =
          full ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
      const VkRect2D rect{
          {full ? 0 : int(scissorRect[0]), full ? 0 : int(scissorRect[1])},
          {full ? width : scissorRect[2] - scissorRect[0],
           full ? height : scissorRect[3] - scissorRect[1]}};
      impl.cmd.beginColorPass(impl.shadowMap[level].view, width, height, loadOp,
                              rect, 1.0f);

      const VkDescriptorSet set = fs.sets[impl.frameSync.frameIndex];
      std::vector<VkWriteDescriptorSet> writes(fs.bindingCount);
      for (u32 b = 0; b < fs.bindingCount; ++b) {
        // The dense SSBO layout consumes handles in order (gaps in the
        // reflection stay harmless unwritten SSBO slots).
        if (b < bufferCount) {
          writes[b] = VkWriteDescriptorSet{
              .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
              .dstSet = set,
              .dstBinding = b,
              .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .pBufferInfo = &infos[b],
          };
        }
      }
      // The dense layout spans only the REFLECTED bindings (16 for
      // fsShadowDepth) - the callers' 17-handle array (the GBuffer
      // shape) must not overflow the write count.
      const u32 writeCount = (std::min)(bufferCount, fs.bindingCount);
      vkUpdateDescriptorSets(impl.device.device, writeCount, writes.data(), 0,
                             nullptr);
      vkCmdBindDescriptorSets(impl.cmd.cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                              fs.layout, 0, 1, &set, 0, nullptr);
      vkCmdBindPipeline(impl.cmd.cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                        fs.pipeline);
      vkCmdPushConstants(
          impl.cmd.cmd(), fs.layout,
          VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
          static_cast<u32>(pushBytes), push);
      vkCmdDraw(impl.cmd.cmd(), 3, 1, 0, 0);
      impl.cmd.endPass();
    }
  } else {
    // Disabled: clear the whole target to 1.0 ("far" = lit) through a
    // dummy color pass loadOp instead of a transfer clear (keeps the
    // layout cycle in graphics; beginColorPass clears to zero so
    // render one fullscreen discard... simplest correct: a transfer
    // clear then barrier - the map is not being sampled this frame).
    CommandContext::pipelineBarrier(
        impl.cmd.cmd(),
        CommandContext::imageBarrier(
            impl.shadowMap[level].image, impl.shadowMapLayout[level],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT));
    const f32 farOne[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    CommandContext::clearColorImage(impl.cmd.cmd(), impl.shadowMap[level].image,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    farOne);
    impl.shadowMapLayout[level] = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  }
  CommandContext::pipelineBarrier(
      impl.cmd.cmd(),
      CommandContext::imageBarrier(
          impl.shadowMap[level].image,
          shadowEnabled ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                        : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
          VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
          VK_ACCESS_2_SHADER_SAMPLED_READ_BIT));
  impl.shadowMapLayout[level] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

// M8.3: the tonemap pass - samples this slot's HDR target (bound to
// the reflected Sampler2D bindings in ascending order) and writes the
// exposure/tonemapped LDR into the offscreen. The offscreen sits in
// COLOR_ATTACHMENT since beginFrame's clear (endFrame moves it to
// SHADER_READ for the ImGui composite as before); the pass overwrites
// every pixel, so the clear's contents are irrelevant here.
void VulkanBackend::drawTaa(u32 pipeline, const u32* bufferHandles,
                            u32 bufferCount, const void* push, usize pushBytes,
                            u32 width, u32 height) {
  Impl& impl = *m_impl;
  if (!impl.frameActive || pipeline == kInvalidPipelineHandle ||
      pipeline > impl.pipelines.fullscreenCount())
    return;
  const PipelineStore::Fullscreen& fs = impl.pipelines.fullscreen(pipeline);
  if (!fs.valid || fs.bindingCount == 0) return;
  // The dense mixed layout leaves binding gaps as SSBO when SPIR-V
  // dead-strips an unused sampler (the M7.2 reflection lesson) - the
  // binding loop clamps instead of skipping so every write stays
  // valid whatever the reflection produced.
  if (fs.bindingCount < 1) return;
  FrameSlot& slot = impl.frameSync.slots[impl.frameSync.frameIndex];
  if (slot.hdr.view == VK_NULL_HANDLE || slot.gbuffer[0].view == VK_NULL_HANDLE)
    return;
  if (width == 0 || height == 0) return;

  // (Re)create the ping-pong pair on size change; the fresh history is
  // garbage either way - the app's blend ramp covers the first frames.
  static const VkFormat kTaaFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
  bool taaPairFresh = false;
  if (impl.taa[0].image == VK_NULL_HANDLE || impl.taaSize != width ||
      impl.taaHeight != height || impl.taa[1].image == VK_NULL_HANDLE) {
    // Deferred destruction (the mesh-buffer idiom): the old pair may
    // still be bound by an in-flight frame - freeing it now lets the
    // allocator hand the memory to the new pair while frame k's
    // resolve still writes it (torn initial history).
    for (u32 i = 0; i < 2; ++i) {
      if (impl.taa[i].image != VK_NULL_HANDLE) {
        const VkImage deadImage = impl.taa[i].image;
        const VkImageView deadView = impl.taa[i].view;
        const VmaAllocation deadAlloc = impl.taa[i].allocation;
        const VmaAllocator alloc = impl.device.allocator;
        const VkDevice dev = impl.device.device;
        impl.frameSync.slots[impl.frameSync.frameIndex].deletionQueue.push(
            [deadImage, deadView, deadAlloc, alloc, dev] {
              vkDestroyImage(dev, deadImage, nullptr);
              vkDestroyImageView(dev, deadView, nullptr);
              vmaFreeMemory(alloc, deadAlloc);
            });
        impl.taa[i] = GpuImage{};  // detached: the slot queue frees it
      }
      if (!impl.taa[i].create(
              impl.device.allocator, impl.device.device, width, height,
              kTaaFormat,
              VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
        KENSHIN_ERROR("[vulkan] taa target creation failed");
        return;
      }
      vkutil::setName(impl.device.device, VK_OBJECT_TYPE_IMAGE,
                      reinterpret_cast<uint64_t>(impl.taa[i].image),
                      i == 0 ? "taa-a" : "taa-b");
      impl.taaLayout[i] = VK_IMAGE_LAYOUT_UNDEFINED;
    }
    impl.taaSize = width;
    impl.taaHeight = height;
    impl.taaParity = 0;
    taaPairFresh = true;
    KENSHIN_INFO("[vulkan] taa pair resized {}x{}", width, height);
  }

  impl.taaParity ^= 1u;
  const u32 writeIdx = impl.taaParity;
  const u32 histIdx = writeIdx ^ 1u;

  std::vector<VkDescriptorBufferInfo> infos(bufferCount);
  for (u32 i = 0; i < bufferCount; ++i) {
    if (!impl.buffers.valid(bufferHandles[i])) return;
    infos[i] = VkDescriptorBufferInfo{
        .buffer = impl.buffers.buffer(bufferHandles[i]),
        .offset = 0,
        .range = impl.buffers.size(bufferHandles[i]),
    };
  }
  // Snapshot the params into this slot's private buffer BEFORE the
  // descriptor bind: TaaPass rewrites its shared mapped block every
  // frame (the jitter word changes per frame), so binding it directly
  // lets the NEXT frame's CPU write tear THIS frame's in-flight read
  // (the run-to-run nondeterminism with TAA on; everything upstream
  // is byte-exact). The slot's pacing proves its previous submission
  // finished, so the memcpy lands while the GPU cannot be reading it.
  if (bufferCount > 0) {
    const u32 slotIdx = impl.frameSync.frameIndex;
    const usize paramBytes = impl.buffers.size(bufferHandles[0]);
    if (!impl.buffers.valid(impl.taaParamsSlot[slotIdx])) {
      BufferDesc desc;
      desc.size = paramBytes;
      desc.usage =
          IRenderBackend::BufferStorage | IRenderBackend::BufferHostVisible;
      desc.initialData = nullptr;
      desc.debugName = "taa-params-slot";
      impl.taaParamsSlot[slotIdx] = createBuffer(desc);
    }
    if (!impl.buffers.valid(impl.taaParamsSlot[slotIdx]))
      KENSHIN_ERROR(
          "[vulkan] taa slot-params mirror unavailable (allocation "
          "failure?) - binding the shared block; per-frame param "
          "rewrites can tear the in-flight read");
    if (impl.buffers.valid(impl.taaParamsSlot[slotIdx])) {
      std::memcpy(impl.buffers.mapped(impl.taaParamsSlot[slotIdx]),
                  impl.buffers.mapped(bufferHandles[0]), paramBytes);
      // A fresh pair's history is UNINITIALIZED - blending it at full
      // strength seeds the accumulation with allocation garbage (the
      // run-to-run nondeterminism). The app's warmup ramp only covers
      // startup, not mid-run resizes, so zero the blend word (offset
      // 140 = TaaParams.camBlend.w) in the slot copy: this frame seeds
      // the history with the raw current sample.
      if (taaPairFresh)
        static_cast<f32*>(
            impl.buffers.mapped(impl.taaParamsSlot[slotIdx]))[35] = 0.0f;
      infos[0] = VkDescriptorBufferInfo{
          .buffer = impl.buffers.buffer(impl.taaParamsSlot[slotIdx]),
          .offset = 0,
          .range = paramBytes,
      };
    }
  }

  impl.cmd.endPass();
  // history: park in SHADER_READ; the write target: -> COLOR_ATTACHMENT.
  CommandContext::pipelineBarrier(
      impl.cmd.cmd(), CommandContext::imageBarrier(
                          impl.taa[histIdx].image, impl.taaLayout[histIdx],
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                          VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                          VK_ACCESS_2_SHADER_READ_BIT));
  CommandContext::pipelineBarrier(
      impl.cmd.cmd(), CommandContext::imageBarrier(
                          impl.taa[writeIdx].image, impl.taaLayout[writeIdx],
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                          VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT));
  {
    const vkutil::ScopedDebugLabel label{impl.cmd.cmd(), "taa"};
    impl.cmd.beginColorPass(impl.taa[writeIdx].view, width, height);

    const VkDescriptorSet set = fs.sets[impl.frameSync.frameIndex];
    const VkDescriptorImageInfo imageInfos[3] = {
        VkDescriptorImageInfo{
            .sampler = impl.linearSampler,
            .imageView = slot.hdr.view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        VkDescriptorImageInfo{
            .sampler = impl.linearSampler,
            .imageView = impl.taa[histIdx].view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        VkDescriptorImageInfo{
            .sampler = impl.linearSampler,
            .imageView = slot.gbuffer[0].view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
    };
    std::vector<VkWriteDescriptorSet> writes(fs.bindingCount);
    u32 nextImage = 0, nextBuffer = 0;
    for (u32 b = 0; b < fs.bindingCount; ++b) {
      if (fs.bindingTypes[b] == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
        const u32 k = (std::min)(nextImage++, 2u);
        writes[b] = VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = b,
            .descriptorCount = 1,
            .descriptorType = fs.bindingTypes[b],
            .pImageInfo = &imageInfos[k],
        };
      } else {
        const u32 k =
            bufferCount == 0 ? 0 : (std::min)(nextBuffer++, bufferCount - 1u);
        writes[b] = VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = b,
            .descriptorCount = 1,
            .descriptorType = fs.bindingTypes[b],
            .pBufferInfo = &infos[k],
        };
      }
    }
    vkUpdateDescriptorSets(impl.device.device, fs.bindingCount, writes.data(),
                           0, nullptr);
    vkCmdBindDescriptorSets(impl.cmd.cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                            fs.layout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(impl.cmd.cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                      fs.pipeline);
    if (pushBytes > 0)
      vkCmdPushConstants(
          impl.cmd.cmd(), fs.layout,
          VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
          static_cast<u32>(pushBytes), push);
    vkCmdDraw(impl.cmd.cmd(), 3, 1, 0, 0);
  }
  impl.cmd.endPass();
  // The fresh resolve becomes the tonemap input (and next frame's
  // history) - park it in SHADER_READ.
  CommandContext::pipelineBarrier(
      impl.cmd.cmd(),
      CommandContext::imageBarrier(impl.taa[writeIdx].image,
                                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_READ_BIT));
  impl.taaLayout[writeIdx] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  impl.taaLayout[histIdx] = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  impl.taaActive = true;
  // The resolve reads the ping-pong image the PREVIOUS frame's resolve
  // wrote - a one-frame-reuse image OUTSIDE slot ownership. Slot pacing
  // only proves kMaxFramesInFlight-ago submits complete, and a pipeline
  // barrier cannot order across submissions: without a queue-level wait
  // the history read races the previous frame's resolve write (observed
  // as run-to-run nondeterministic output pixels with TAA on, byte-exact
  // with it off). The frame submit waits the previous frame's timeline
  // value, FRAGMENT-stage masked - only the history-reading passes gate;
  // the async chain still overlaps frames.
  impl.taaHistWaitValue = impl.frameSync.timelineValue;
  impl.taaHistWaitNeeded = true;
}

void VulkanBackend::drawTonemap(u32 pipeline, const void* push, usize pushBytes,
                                u32 width, u32 height, bool fromTaa,
                                u32 exposureState, u32 envTexture) {
  Impl& impl = *m_impl;
  if (!impl.frameActive || pipeline == kInvalidPipelineHandle ||
      pipeline > impl.pipelines.fullscreenCount())
    return;
  const PipelineStore::Fullscreen& fs = impl.pipelines.fullscreen(pipeline);
  if (!fs.valid || fs.bindingCount == 0) return;
  FrameSlot& slot = impl.frameSync.slots[impl.frameSync.frameIndex];
  if (slot.hdr.view == VK_NULL_HANDLE || slot.offscreen.view == VK_NULL_HANDLE)
    return;
  // fromTaa: sample the freshly written resolve instead of the raw
  // slot HDR (impl.taaParity points at it after drawTaa).
  if (fromTaa && !impl.taaActive) return;
  // The SSBO slot (exposure state) must be written - a draw with an
  // unwritten statically-used binding is invalid.
  for (u32 b = 0; b < fs.bindingCount; ++b)
    if (fs.bindingTypes[b] == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER &&
        !impl.buffers.valid(exposureState)) {
      KENSHIN_ERROR("[vulkan] drawTonemap: no exposure state buffer");
      return;
    }

  impl.cmd.endPass();
  {
    const vkutil::ScopedDebugLabel label{impl.cmd.cmd(), "tonemap"};
    impl.cmd.beginColorPass(slot.offscreen.view, width, height);

    const VkDescriptorSet set = fs.sets[impl.frameSync.frameIndex];
    VkDescriptorImageInfo imageInfo{
        .sampler = impl.linearSampler,
        .imageView = fromTaa ? impl.taa[impl.taaParity].view : slot.hdr.view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    // Second sampler (binding 1): the bloom chain's top level when this
    // frame recorded it; otherwise the scene view again (the push sets
    // the bloom strength to 0 then - the set must stay validly written).
    VkDescriptorImageInfo bloomInfo = imageInfo;
    if (impl.bloomActive) bloomInfo.imageView = impl.bloom[0].view;
    // Third sampler (binding 3, Sampler2DArray): the sky's env texture -
    // an ARRAYED view either way (the dummy array when absent).
    VkDescriptorImageInfo envInfo = imageInfo;
    envInfo.imageView = envTexture < impl.textures.size() &&
                                impl.textureAlive[envTexture]
                            ? impl.textures[envTexture].view
                            : impl.dummyTexture.view;
    u32 nextImage = 0;
    VkDescriptorBufferInfo stateInfo{};
    if (impl.buffers.valid(exposureState))
      stateInfo = VkDescriptorBufferInfo{
          .buffer = impl.buffers.buffer(exposureState),
          .offset = 0,
          .range = impl.buffers.size(exposureState),
      };
    std::vector<VkWriteDescriptorSet> writes(fs.bindingCount);
    for (u32 b = 0; b < fs.bindingCount; ++b) {
      writes[b] = VkWriteDescriptorSet{
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = set,
          .dstBinding = b,
          .descriptorCount = 1,
          .descriptorType = fs.bindingTypes[b],
      };
      if (fs.bindingTypes[b] == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
        const u32 img = nextImage++;
        writes[b].pImageInfo =
            img == 0 ? &imageInfo : (img == 1 ? &bloomInfo : &envInfo);
      } else {
        writes[b].pBufferInfo = &stateInfo;  // binding 2: exposure state
      }
    }
    vkUpdateDescriptorSets(impl.device.device, fs.bindingCount, writes.data(),
                           0, nullptr);
    vkCmdBindDescriptorSets(impl.cmd.cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                            fs.layout, 0, 1, &set, 0, nullptr);
    vkCmdBindPipeline(impl.cmd.cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                      fs.pipeline);
    vkCmdPushConstants(
        impl.cmd.cmd(), fs.layout,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
        static_cast<u32>(pushBytes), push);
    vkCmdDraw(impl.cmd.cmd(), 3, 1, 0, 0);
    impl.cmd.endPass();
  }
  // The offscreen stays COLOR_ATTACHMENT - endFrame's SHADER_READ
  // barrier for the ImGui composite picks it up unchanged.
}

u32 VulkanBackend::createNullPipeline(const FullscreenPipelineInfo& info) {
  FullscreenPipelineInfo patched = info;
  // raster_hw's VS emits the 1/256-SNAPPED corners, so a rasterizer
  // with >= 8 subpixel bits reproduces the SW-agreed triangle exactly
  // and needs no overestimation (which cost ~40% of the HW path in
  // extra fragment invocations). Coarser snaps keep the conservative
  // fallback - the PS's exact test still rejects the extra pixels.
  patched.conservativeRaster =
      m_impl->context.supportsConservativeRaster &&
      m_impl->context.props.limits.subPixelPrecisionBits < 8;
  static bool logged = false;
  if (!logged) {
    logged = true;
    KENSHIN_INFO("[vulkan] HW raster: {} subpixel bits -> conservative {}",
                 m_impl->context.props.limits.subPixelPrecisionBits,
                 patched.conservativeRaster ? "ON (fallback)" : "off");
  }
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
  VkSemaphoreSubmitInfo waits[2] = {waitInfo, {}};
  u32 waitCount = 0;
  if (prev > 0) waits[waitCount++] = waitInfo;
  if (impl.chainGraphicsWait > 0) {
    waits[waitCount++] = VkSemaphoreSubmitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = impl.frameSync.frameTimeline,
        .value = impl.chainGraphicsWait,
        .stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    };
    impl.chainGraphicsWait = 0;
  }
  const VkSubmitInfo2 submit{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = waitCount,
      .pWaitSemaphoreInfos = waits,
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

void VulkanBackend::chainWaitGraphics() {
  Impl& impl = *m_impl;
  // During recording, timelineValue is the value the PREVIOUS frame's
  // graphics submission signals (this frame's is assigned at endFrame).
  if (impl.asyncActive) impl.chainGraphicsWait = impl.frameSync.timelineValue;
}

bool VulkanBackend::chainOnComputeQueue() const {
  return m_impl->asyncActive && m_impl->chainValueB != 0;
}

void VulkanBackend::writeChainTimestamp(u32 index) {
  Impl& impl = *m_impl;
  if (index >= 7) return;
  FrameSlot& slot = impl.frameSync.slots[impl.frameSync.frameIndex];
  if (slot.chainTimingQueries == VK_NULL_HANDLE) return;
  // The chain records onto the async compute buffer while a chain is
  // open; without one (async disabled) the same recording lands on the
  // graphics frame buffer. Timestamp validity is per queue FAMILY.
  const bool onCompute = impl.computeOverride != VK_NULL_HANDLE;
  if (onCompute ? !impl.chainTimestampsCompute : !impl.gpuTimestamps) return;
  const VkCommandBuffer cb = onCompute ? impl.computeOverride : impl.cmd.cmd();
  if (!impl.chainQueriesArmed[impl.frameSync.frameIndex]) {
    vkCmdResetQueryPool(cb, slot.chainTimingQueries, 0, 7);
    impl.chainQueriesArmed[impl.frameSync.frameIndex] = true;
    impl.chainPoolEverWritten[impl.frameSync.frameIndex] = true;
  }
  vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       slot.chainTimingQueries, index);
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
  if (!KE_VK(vkCreateSemaphore(device.device, &timelineInfo, nullptr,
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
      .sharingMode =
          sameFamily ? VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT,
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
                      &page.arenaBuffer, &page.arenaAlloc,
                      &out) != VK_SUCCESS ||
      out.pMappedData == nullptr)
    return false;
  page.arenaMapped = out.pMappedData;

  KENSHIN_INFO("[vulkan] page uploads: {} ({}B staging arena)",
               pageUploadsOnTransferQueue() ? "dedicated transfer queue"
                                            : "graphics queue",
               page.kArenaBytes);
  return true;
}

u32 VulkanBackend::createPageSlotPool(u32 slotCount, u32 slotBytes) {
  Impl& impl = *m_impl;
  if (slotCount == 0 || slotBytes == 0) return kInvalidBufferHandle;
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
