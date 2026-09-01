#pragma once

// ── Per-frame resources (internal to the backend) ───────────────
// kMaxFramesInFlight slots, each owning: command pool+buffer, acquire/present
// semaphores, in-flight fence, and its own offscreen color target (so two
// in-flight frames never touch the same image).

#include <volk.h>

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
  u64 timelineValue = 0;      // value of the last signaled submit
  u64 completedValue = 0;     // timeline value of `lastCompleted`

  bool init(const VulkanDevice& device, bool timestamps);
  void shutdown(const VulkanDevice& device);

  /// Block the host until the timeline reaches `value`.
  bool waitTimeline(const VulkanDevice& device, u64 value) const;

  /// (Re)create offscreen targets for the given size/format when they
  /// do not match. Destroys old ones first.
  bool ensureOffscreens(const VulkanDevice& device, u32 width, u32 height,
                        VkFormat format, VkImageUsageFlags usage);

  void advance() { frameIndex = (frameIndex + 1) % kMaxFramesInFlight; }
};

}  // namespace backend

KENSHIN_NS_END
