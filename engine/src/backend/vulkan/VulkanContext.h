#pragma once

// ── Vulkan context (internal to the backend) ────────────────────
// Owns: volk bootstrap, VkInstance, debug messenger, physical device
// selection and queue family selection.

#include "kenshin/Common.h"

#include <volk.h>

KENSHIN_NS_BEGIN

namespace backend
{

    struct VulkanContext
    {
        VkInstance instance = VK_NULL_HANDLE;
        VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

        VkPhysicalDeviceProperties props {};
        VkPhysicalDeviceMemoryProperties memProps {};

        // Feature support (queried from the selected physical device)
        bool supportsBufferInt64Atomics = false;
        bool supportsTimelineSemaphore  = false;

        // Queue family selection. transferFamily prefers a dedicated DMA
        // family (TRANSFER without GRAPHICS/COMPUTE) for async streaming
        // uploads (M6); falls back to compute-only, then graphics.
        u32 graphicsFamily = 0xFFFFFFFFu;
        u32 computeFamily  = 0xFFFFFFFFu;
        u32 transferFamily = 0xFFFFFFFFu;

        /// @param enableValidation  add VK_LAYER_KHRONOS_validation + debug messenger
        /// @param needSurface       add instance-level surface extensions (Win32)
        bool init(bool enableValidation, bool needSurface);
        void shutdown();

        /// Create a temporary surface from hwnd and test present support on the
        /// graphics family. The surface is destroyed again.
        bool checkPresentSupport(void* hwnd);

    private:
        bool createInstance(bool enableValidation, bool needSurface);
        bool pickPhysicalDevice();
        void selectQueueFamilies();
    };

} // namespace backend

KENSHIN_NS_END
