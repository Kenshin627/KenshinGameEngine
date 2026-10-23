#pragma once

// ── Swapchain (internal to the backend) ─────────────────────────
// The swapchain with its image views over a surface the BACKEND owns
// (created by the platform window, destroyed after the swapchain).
// FIFO present mode; format picks an 8-bit UNORM variant for deterministic
// readback comparisons (no SRGB conversion in the blit path).

#include <volk.h>

#include <vector>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace backend {

struct Swapchain {
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkExtent2D extent{};
  std::vector<VkImage> images;
  std::vector<VkImageView> imageViews;

  /// One present-wait semaphore per swapchain image: a semaphore stays
  /// "in use by the swapchain" until its present completes, so reusing
  /// it is only safe when the SAME image is presented again.
  /// Canonical pattern per the Khronos guide "Swapchain Semaphore Reuse"
  /// (https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html):
  /// semaphores indexed by swapchain image, fences/cmd buffers by frame
  /// slot. Violations trip VUID-vkQueueSubmit-pSignalSemaphores-00067.
  /// Future option: VK_EXT_swapchain_maintenance1 present fences.
  std::vector<VkSemaphore> presentSemaphores;

  /// Index of the last acquired image (valid after acquire() == SUCCESS/SUBOPTIMAL).
  u32 imageIndex = 0;

  bool init(VkPhysicalDevice physicalDevice, VkDevice device,
            VkSurfaceKHR windowSurface, u32 width, u32 height);
  /// Destroys the swapchain objects; the surface stays (backend-owned).
  void shutdown(VkDevice device);

  /// Acquire the next image. @return VK_SUCCESS, VK_SUBOPTIMAL_KHR (image
  /// still valid), VK_ERROR_OUT_OF_DATE_KHR (call recreate), or an error.
  VkResult acquire(VkDevice device, VkSemaphore imageAvailable);

  /// Recreate the swapchain for a new size (waits for the device idle).
  /// While the surface reports a 0x0 extent (window minimized) nothing
  /// is created - Vulkan forbids zero extents - and the recreate is
  /// deferred (recreatePending) to the next frame with a real area.
  bool recreate(VkPhysicalDevice physicalDevice, VkDevice device, u32 width,
                u32 height);
  bool surfaceHasArea(VkPhysicalDevice physicalDevice) const;
  bool recreatePending = false;

 private:
  const char* m_presentModeName = "FIFO";
  // graphicsFamily unused while sharing stays EXCLUSIVE (single queue family)
  bool createSwapchain(VkPhysicalDevice physicalDevice, VkDevice device,
                       u32 width, u32 height);
  void destroySwapchainObjects(VkDevice device);
};

}  // namespace backend

KENSHIN_NS_END
