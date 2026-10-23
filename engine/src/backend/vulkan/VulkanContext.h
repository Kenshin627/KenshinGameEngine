#pragma once

// ── Vulkan context (internal to the backend) ────────────────────
// Owns: volk bootstrap, VkInstance, debug messenger, physical device
// selection and queue family selection.

#include <volk.h>

#include <string>
#include <vector>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace backend {

struct VulkanContext {
  VkInstance instance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

  VkPhysicalDeviceProperties props{};
  VkPhysicalDeviceMemoryProperties memProps{};

  // Feature support (queried from the selected physical device)
  bool supportsBufferInt64Atomics = false;
  bool supportsTimelineSemaphore = false;
  bool supportsDrawParameters = false;  // SV_VertexID (fullscreen passes)
  bool supportsSynchronization2 = false;
  bool supportsDynamicRendering = false;
  // P0 seam fix: VK_EXT_conservative_rasterization (the HW raster
  // pipeline's overestimate mode - see createNullRaster).
  bool supportsConservativeRaster = false;

  // True when surface extensions were enabled at instance creation.
  bool surfaceEnabled = false;

  // Queue family selection. transferFamily prefers a dedicated DMA
  // family (TRANSFER without GRAPHICS/COMPUTE) for async streaming
  // uploads (M6); falls back to compute-only, then graphics.
  u32 graphicsFamily = 0xFFFFFFFFu;
  u32 computeFamily = 0xFFFFFFFFu;
  u32 transferFamily = 0xFFFFFFFFu;
  // M4.3 async compute: a compute-only family, or - when the
  // graphics family also exposes compute with queueCount >= 2 - a
  // second queue from it (sameFamilyAsyncCompute).
  u32 asyncComputeFamily = 0xFFFFFFFFu;
  bool sameFamilyAsyncCompute = false;

  /// @param enableValidation  add VK_LAYER_KHRONOS_validation + debug messenger
  /// @param surfaceExtensions the window system's instance extensions
  ///                          (empty = headless)
  bool init(bool enableValidation,
            const std::vector<std::string>& surfaceExtensions);
  void shutdown();

  /// Present support of the graphics family for `surface`.
  bool checkPresentSupport(VkSurfaceKHR surface) const;

 private:
  bool createInstance(bool enableValidation,
                      const std::vector<std::string>& surfaceExtensions);
  bool pickPhysicalDevice();
  void selectQueueFamilies();
};

}  // namespace backend

KENSHIN_NS_END
