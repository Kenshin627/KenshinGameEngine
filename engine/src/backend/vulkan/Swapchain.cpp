#include "Swapchain.h"

#include "kenshin/core/Log.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#include <cstring>

KENSHIN_NS_BEGIN

namespace backend
{

    namespace
    {
        // vkCreateWin32SurfaceKHR is platform-gated in volk.c; fetch at runtime.
        PFN_vkCreateWin32SurfaceKHR win32SurfaceFn(VkInstance instance)
        {
            return reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(
                vkGetInstanceProcAddr(instance, "vkCreateWin32SurfaceKHR"));
        }
    } // namespace

    bool Swapchain::init(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device,
                         void* hwnd, u32 width, u32 height)
    {
        const auto createSurface = win32SurfaceFn(instance);
        if (!createSurface)
        {
            KENSHIN_ERROR("[vulkan] vkCreateWin32SurfaceKHR unavailable");
            return false;
        }
        const VkWin32SurfaceCreateInfoKHR surfaceInfo {
            .sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
            .hinstance = GetModuleHandleW(nullptr),
            .hwnd = static_cast<HWND>(hwnd),
        };
        if (createSurface(instance, &surfaceInfo, nullptr, &surface) != VK_SUCCESS)
        {
            KENSHIN_ERROR("[vulkan] surface creation failed");
            return false;
        }

        if (!createSwapchain(physicalDevice, device, width, height))
            return false;

        KENSHIN_INFO("[vulkan] swapchain {}x{}, {} images, {}",
                     extent.width, extent.height, static_cast<u32>(images.size()),
                     m_presentModeName);
        return true;
    }

    void Swapchain::shutdown(VkInstance instance, VkDevice device)
    {
        destroySwapchainObjects(device);
        if (surface != VK_NULL_HANDLE)
        {
            vkDestroySurfaceKHR(instance, surface, nullptr);
            surface = VK_NULL_HANDLE;
        }
    }

    VkResult Swapchain::acquire(VkDevice device, VkSemaphore imageAvailable)
    {
        u32 nextIndex = 0;
        const VkResult result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
                                                      imageAvailable, VK_NULL_HANDLE,
                                                      &nextIndex);
        if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR)
            imageIndex = nextIndex;
        return result;
    }

    bool Swapchain::recreate(VkPhysicalDevice physicalDevice, VkDevice device,
                             u32 width, u32 height)
    {
        vkDeviceWaitIdle(device);
        destroySwapchainObjects(device);
        if (!createSwapchain(physicalDevice, device, width, height))
            return false;
        KENSHIN_INFO("[vulkan] swapchain recreated {}x{}", extent.width, extent.height);
        return true;
    }

    bool Swapchain::createSwapchain(VkPhysicalDevice physicalDevice, VkDevice device,
                                    u32 width, u32 height)
    {
        VkSurfaceCapabilitiesKHR caps {};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &caps);

        // Deterministic format choice: prefer B8G8R8A8_UNORM, then R8G8B8A8_UNORM,
        // then the first UNORM format. UNORM keeps the readback path free of any
        // SRGB conversion for byte-exact screenshot comparisons.
        u32 formatCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, formats.data());

        VkSurfaceFormatKHR chosen = formats.front();
        if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED)
            chosen = {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
        else
        {
            for (const auto& f : formats)
                if (f.format == VK_FORMAT_B8G8R8A8_UNORM) { chosen = f; break; }
            if (chosen.format != VK_FORMAT_B8G8R8A8_UNORM)
                for (const auto& f : formats)
                    if (f.format == VK_FORMAT_R8G8B8A8_UNORM) { chosen = f; break; }
        }

        VkExtent2D newExtent = caps.currentExtent;
        if (extent.width == 0xFFFFFFFFu)
            newExtent = {width, height}; // some window systems report "match the window"

        // Present mode: prefer MAILBOX (low-latency vsync); FIFO is the
        // guaranteed fallback. User decision 2026-08-31: default MAILBOX.
        u32 modeCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &modeCount, nullptr);
        std::vector<VkPresentModeKHR> modes(modeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &modeCount, modes.data());
        VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
        for (const auto mode : modes)
            if (mode == VK_PRESENT_MODE_MAILBOX_KHR) { presentMode = mode; break; }
        m_presentModeName = presentMode == VK_PRESENT_MODE_MAILBOX_KHR ? "MAILBOX" : "FIFO";

        const u32 imageCount = (caps.minImageCount + 1 <= caps.maxImageCount || caps.maxImageCount == 0)
                             ? caps.minImageCount + 1
                             : caps.maxImageCount;

        const VkSwapchainCreateInfoKHR createInfo {
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = surface,
            .minImageCount = imageCount,
            .imageFormat = chosen.format,
            .imageColorSpace = chosen.colorSpace,
            .imageExtent = newExtent,
            .imageArrayLayers = 1,
            .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT, // blit target (S1.3)
            .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .preTransform = caps.currentTransform,
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = presentMode,
            .clipped = VK_TRUE,
            .oldSwapchain = VK_NULL_HANDLE,
        };
        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain) != VK_SUCCESS)
        {
            KENSHIN_ERROR("[vulkan] vkCreateSwapchainKHR failed");
            return false;
        }

        u32 actualCount = 0;
        vkGetSwapchainImagesKHR(device, swapchain, &actualCount, nullptr);
        images.resize(actualCount);
        vkGetSwapchainImagesKHR(device, swapchain, &actualCount, images.data());

        imageViews.resize(actualCount);
        for (u32 i = 0; i < actualCount; ++i)
        {
            const VkImageViewCreateInfo viewInfo {
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = images[i],
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = chosen.format,
                .subresourceRange = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .levelCount = 1,
                    .layerCount = 1,
                },
            };
            if (vkCreateImageView(device, &viewInfo, nullptr, &imageViews[i]) != VK_SUCCESS)
            {
                KENSHIN_ERROR("[vulkan] swapchain image view creation failed");
                return false;
            }
        }

        format = chosen.format;
        this->extent = newExtent;
        return true;
    }

    void Swapchain::destroySwapchainObjects(VkDevice device)
    {
        for (auto view : imageViews)
            if (view != VK_NULL_HANDLE)
                vkDestroyImageView(device, view, nullptr);
        imageViews.clear();
        images.clear();
        if (swapchain != VK_NULL_HANDLE)
        {
            vkDestroySwapchainKHR(device, swapchain, nullptr);
            swapchain = VK_NULL_HANDLE;
        }
    }

} // namespace backend

KENSHIN_NS_END
