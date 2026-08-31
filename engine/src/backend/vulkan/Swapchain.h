#pragma once

// ── Swapchain (internal to the backend) ─────────────────────────
// Owns the persistent Win32 surface and the swapchain with its image views.
// FIFO present mode; format picks an 8-bit UNORM variant for deterministic
// readback comparisons (no SRGB conversion in the blit path).

#include "kenshin/Common.h"

#include <volk.h>

#include <vector>

KENSHIN_NS_BEGIN

namespace backend
{

    struct Swapchain
    {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        VkSwapchainKHR swapchain = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkExtent2D extent {};
        std::vector<VkImage> images;
        std::vector<VkImageView> imageViews;

        /// Index of the last acquired image (valid after acquire() == SUCCESS/SUBOPTIMAL).
        u32 imageIndex = 0;

        bool init(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device,
                  void* hwnd, u32 width, u32 height);
        void shutdown(VkInstance instance, VkDevice device);

        /// Acquire the next image. @return VK_SUCCESS, VK_SUBOPTIMAL_KHR (image
        /// still valid), VK_ERROR_OUT_OF_DATE_KHR (call recreate), or an error.
        VkResult acquire(VkDevice device, VkSemaphore imageAvailable);

        /// Recreate the swapchain for a new size (waits for the device idle).
        bool recreate(VkPhysicalDevice physicalDevice, VkDevice device,
                      u32 width, u32 height);

    private:
        const char* m_presentModeName = "FIFO";
        // graphicsFamily unused while sharing stays EXCLUSIVE (single queue family)
        bool createSwapchain(VkPhysicalDevice physicalDevice, VkDevice device,
                             u32 width, u32 height);
        void destroySwapchainObjects(VkDevice device);
    };

} // namespace backend

KENSHIN_NS_END
