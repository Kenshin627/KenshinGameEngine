#pragma once

// ── Per-frame resources (internal to the backend) ───────────────
// kMaxFramesInFlight slots, each owning: command pool+buffer, acquire/present
// semaphores, in-flight fence, and its own offscreen color target (so two
// in-flight frames never touch the same image).

#include <volk.h>

#include <vector>

#include "DeletionQueue.h"
#include "GpuImage.h"
#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace backend {

struct VulkanDevice;

constexpr u32 kMaxFramesInFlight = 2;

struct FrameSlot {
  VkCommandPool commandPool = VK_NULL_HANDLE;
  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
  // Binary on purpose: WSI requires binary semaphores for acquire/present
  // (Khronos swapchain-semaphore-reuse guidance). All non-WSI frame sync
  // rides the single device timeline semaphore below.
  VkSemaphore imageAvailable =
      VK_NULL_HANDLE;  // signaled by acquire, consumed by submit
  GpuImage offscreen;
  // Deferred destroys queued while this slot was recording run when the
  // slot is reused (the timeline wait proves the frame completed by then).
  DeletionQueue deletionQueue;
  // ImGui descriptor set sampling `offscreen` (VkImage it was created for
  // tracked so a resize recreates the set). Opaque in this header.
  void* imguiTexture = nullptr;
  VkImage imguiTextureImage = VK_NULL_HANDLE;

  // GPU timing queries for this slot's frame: 0 = submission start,
  // 1 = viewport scope close, 2 = submission end. VK_NULL_HANDLE when
  // the queue family lacks timestamp support.
  VkQueryPool timingQueries = VK_NULL_HANDLE;
  // Chain segment timestamps (7 stamps/frame; written on the chain's
  // command buffer - async compute or the graphics fallback).
  VkQueryPool chainTimingQueries = VK_NULL_HANDLE;

  // M4.3 async compute: this slot's chain command buffers on the
  // compute queue. cbA = the cull chain (cull -> expand -> cluster
  // cull -> conversions -> count_to_draw), cbB = the raster chain
  // (SW raster -> post section -> final HZB). Split so the graphics
  // queue's HW draw can wait ONLY on A and run CONCURRENTLY with B
  // (both atomically write the same VisBuffer - order-free max).
  VkCommandPool computePool = VK_NULL_HANDLE;
  VkCommandBuffer computeCullCb = VK_NULL_HANDLE;
  VkCommandBuffer computeRasterCb = VK_NULL_HANDLE;

  bool initialized() const { return commandPool != VK_NULL_HANDLE; }
};

struct FrameSync {
  FrameSlot slots[kMaxFramesInFlight];
  u32 frameIndex = 0;  // rotating slot
  u32 lastCompleted = 0xFFFFFFFFu;
  bool anyCompleted = false;

  // CPU<->GPU frame pacing: ONE timeline semaphore, submit #k signals
  // value k; a frame waits for (k - kMaxFramesInFlight) before reusing a
  // slot's resources. Replaces the per-slot fences (Raptor shape).
  VkSemaphore frameTimeline = VK_NULL_HANDLE;
  u64 timelineValue = 0;   // value of the last signaled submit
  u64 completedValue = 0;  // timeline value of `lastCompleted`

  // M4.3: the async-compute chain's OWN timeline (independent of the
  // graphics frame pacing - a chain's completion is not the frame's).
  // Per frame N: valueA = 2N+1 (cull cb), valueB = 2N+2 (raster cb).
  // The graphics frame waits valueA before the HW draw (vertex/
  // indirect stages) and valueB before the decode (fragment stage).
  VkSemaphore computeTimeline = VK_NULL_HANDLE;
  u64 computeValue = 0;             // value of the last signaled chain submit
  bool asyncComputeActive = false;  // queue + pools created

  // Resized-out offscreen targets are retired, not destroyed: destroying a
  // recently-sampled image mid-loop wedges the NVIDIA no-validation present
  // path even after vkDeviceWaitIdle (isolated 2026-09-06: destroy+free at
  // the frame-1 viewport resize = DEVICE_LOST; deferring both = clean).
  // Destroyed once kRetireDelay frames have passed.
  struct RetiredTarget {
    GpuImage image;
    u64 retireFrame = 0;
  };
  std::vector<RetiredTarget> retired;
  u64 frameCounter = 0;

  bool init(const VulkanDevice& device, bool timestamps);
  void shutdown(const VulkanDevice& device);

  /// Destroy retired offscreen targets that have aged past the delay.
  void drainRetired(const VulkanDevice& device);

  /// M4.3: per-slot compute pools + cull/raster command buffers and
  /// the compute timeline. No-op (false) when the device lacks an
  /// async compute queue - the chain then stays on the graphics
  /// queue exactly as before.
  bool initAsyncCompute(const VulkanDevice& device);

  /// Block the host until the compute timeline reaches `value`.
  bool waitCompute(const VulkanDevice& device, u64 value) const;

  /// Block the host until the timeline reaches `value`.
  bool waitTimeline(const VulkanDevice& device, u64 value) const;

  /// (Re)create offscreen targets for the given size/format when they do
  /// not match. Old targets are retired (see RetiredTarget).
  bool ensureOffscreens(const VulkanDevice& device, u32 width, u32 height,
                        VkFormat format, VkImageUsageFlags usage);

  void advance() { frameIndex = (frameIndex + 1) % kMaxFramesInFlight; }
};

}  // namespace backend

KENSHIN_NS_END
