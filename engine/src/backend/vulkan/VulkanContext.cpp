#include "VulkanContext.h"

#include "Vk.h"
#include "kenshin/core/Log.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#include <cstdlib>
#include <cstring>
#include <vector>

KENSHIN_NS_BEGIN

namespace backend {

namespace {
VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* data, void* /*userData*/) {
  if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    KENSHIN_ERROR("[vulkan][validation] {}", data->pMessage);
  else if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    KENSHIN_WARN("[vulkan][validation] {}", data->pMessage);
  else
    KENSHIN_INFO("[vulkan][validation] {}", data->pMessage);
  return VK_FALSE;
}

bool hasLayer(const std::vector<VkLayerProperties>& layers, const char* name) {
  for (const auto& layer : layers)
    if (std::strcmp(layer.layerName, name) == 0) return true;
  return false;
}

bool hasExtension(const std::vector<VkExtensionProperties>& extensions,
                  const char* name) {
  for (const auto& ext : extensions)
    if (std::strcmp(ext.extensionName, name) == 0) return true;
  return false;
}

const char* gpuTypeName(VkPhysicalDeviceType type) {
  switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      return "discrete";
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      return "integrated";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      return "virtual";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      return "cpu";
    default:
      return "unknown";
  }
}
}  // namespace

bool VulkanContext::init(bool enableValidation, bool needSurface) {
  if (const VkResult result = volkInitialize(); result != VK_SUCCESS) {
    KENSHIN_ERROR(
        "[vulkan] volkInitialize failed ({}) — is a Vulkan loader present?",
        static_cast<int>(result));
    return false;
  }

  u32 loaderVersion = 0;
  vkEnumerateInstanceVersion(&loaderVersion);
  if (loaderVersion < VK_API_VERSION_1_4) {
    KENSHIN_ERROR("[vulkan] loader only supports {}.{} (need 1.4)",
                  VK_API_VERSION_MAJOR(loaderVersion),
                  VK_API_VERSION_MINOR(loaderVersion));
    return false;
  }

  if (!createInstance(enableValidation, needSurface)) return false;

  if (!pickPhysicalDevice()) return false;

  selectQueueFamilies();
  return true;
}

void VulkanContext::shutdown() {
  if (messenger != VK_NULL_HANDLE) {
    vkDestroyDebugUtilsMessengerEXT(instance, messenger, nullptr);
    messenger = VK_NULL_HANDLE;
  }
  if (instance != VK_NULL_HANDLE) {
    vkDestroyInstance(instance, nullptr);
    instance = VK_NULL_HANDLE;
  }
}

bool VulkanContext::createInstance(bool enableValidation, bool needSurface) {
  u32 layerCount = 0;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
  std::vector<VkLayerProperties> layers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, layers.data());

  u32 extCount = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
  std::vector<VkExtensionProperties> extensions(extCount);
  vkEnumerateInstanceExtensionProperties(nullptr, &extCount, extensions.data());

  std::vector<const char*> instanceLayers;
  bool validationOn = false;
  if (enableValidation) {
    if (hasLayer(layers, "VK_LAYER_KHRONOS_validation")) {
      instanceLayers.push_back("VK_LAYER_KHRONOS_validation");
      validationOn = true;
    } else
      KENSHIN_WARN(
          "[vulkan] validation requested but layer not found; continuing "
          "without");
  }

  std::vector<const char*> instanceExtensions;
  if (hasExtension(extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME))
    instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  if (needSurface) {
    instanceExtensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
    instanceExtensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
    surfaceEnabled = true;
  }

  const VkApplicationInfo app{
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "KENSHIN",
      .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
      .pEngineName = "KENSHIN",
      .engineVersion = VK_MAKE_VERSION(0, 1, 0),
      .apiVersion = VK_API_VERSION_1_4,
  };
  const VkInstanceCreateInfo createInfo{
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
      .enabledLayerCount = static_cast<u32>(instanceLayers.size()),
      .ppEnabledLayerNames =
          instanceLayers.empty() ? nullptr : instanceLayers.data(),
      .enabledExtensionCount = static_cast<u32>(instanceExtensions.size()),
      .ppEnabledExtensionNames =
          instanceExtensions.empty() ? nullptr : instanceExtensions.data(),
  };

  if (const VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
      result != VK_SUCCESS) {
    KENSHIN_ERROR("[vulkan] vkCreateInstance failed ({})",
                  static_cast<int>(result));
    return false;
  }
  volkLoadInstance(instance);

  if (!instanceExtensions.empty() &&
      hasExtension(extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
    const VkDebugUtilsMessengerCreateInfoEXT messengerInfo{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debugCallback,
    };
    if (!KE_VK(vkCreateDebugUtilsMessengerEXT(instance, &messengerInfo, nullptr,
                                              &messenger)))
      KENSHIN_WARN("[vulkan] debug messenger creation failed (non-fatal)");
    else
      vkutil::g_debugUtilsEnabled = true;  // object names/labels become live
  }

  KENSHIN_INFO("[vulkan] instance {}.{} ok, validation={}",
               VK_API_VERSION_MAJOR(VK_API_VERSION_1_4),
               VK_API_VERSION_MINOR(VK_API_VERSION_1_4),
               validationOn ? "on" : "off");
  return true;
}

bool VulkanContext::pickPhysicalDevice() {
  u32 deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
  if (deviceCount == 0) {
    KENSHIN_ERROR("[vulkan] no physical device found");
    return false;
  }
  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

  auto scoreOf = [](const VkPhysicalDeviceProperties& props,
                    const VkPhysicalDeviceMemoryProperties& mem) {
    int score = 0;
    switch (props.deviceType) {
      case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        score = 300;
        break;
      case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        score = 200;
        break;
      case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        score = 100;
        break;
      default:
        score = 0;
        break;
    }
    // Tiebreak on device-local memory (heuristic only; fits in int)
    u64 localMB = 0;
    for (u32 i = 0; i < mem.memoryHeapCount; ++i)
      if (mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
        localMB += mem.memoryHeaps[i].size / (1024 * 1024);
    return score * 1000 + static_cast<int>(localMB % 1000);
  };

  int bestScore = -1;
  for (const auto device : devices) {
    VkPhysicalDeviceProperties devProps{};
    VkPhysicalDeviceMemoryProperties devMemProps{};
    vkGetPhysicalDeviceProperties(device, &devProps);
    vkGetPhysicalDeviceMemoryProperties(device, &devMemProps);

    KENSHIN_INFO("[vulkan] GPU candidate: {} (api {}.{}.{}, {}, driver {})",
                 devProps.deviceName, VK_API_VERSION_MAJOR(devProps.apiVersion),
                 VK_API_VERSION_MINOR(devProps.apiVersion),
                 VK_API_VERSION_PATCH(devProps.apiVersion),
                 gpuTypeName(devProps.deviceType), devProps.driverVersion);

    const int score = scoreOf(devProps, devMemProps);
    if (score > bestScore) {
      bestScore = score;
      physicalDevice = device;
      props = devProps;
      memProps = devMemProps;
    }
  }

  if (physicalDevice == VK_NULL_HANDLE) {
    KENSHIN_ERROR("[vulkan] no usable physical device");
    return false;
  }

  // Query feature support on the selected device (1.1 + 1.2 + 1.3 chains)
  VkPhysicalDeviceVulkan13Features features13{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
  };
  VkPhysicalDeviceVulkan12Features features12{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &features13,
  };
  VkPhysicalDeviceVulkan11Features features11{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
      .pNext = &features12,
  };
  VkPhysicalDeviceFeatures2 features2{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &features11,
  };
  vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);
  supportsBufferInt64Atomics = features12.shaderBufferInt64Atomics != VK_FALSE;
  supportsTimelineSemaphore = features12.timelineSemaphore != VK_FALSE;
  supportsSynchronization2 = features13.synchronization2 != VK_FALSE;
  supportsDynamicRendering = features13.dynamicRendering != VK_FALSE;
  supportsDrawParameters = features11.shaderDrawParameters != VK_FALSE;

  // Device extensions (P0 seam fix: conservative rasterization for the
  // HW path's null pipeline).
  {
    u32 devExtCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &devExtCount,
                                         nullptr);
    std::vector<VkExtensionProperties> devExts(devExtCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &devExtCount,
                                         devExts.data());
    for (const auto& e : devExts)
      if (std::strcmp(e.extensionName,
                      VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME) == 0) {
        supportsConservativeRaster = true;
        break;
      }
  }

  KENSHIN_INFO("[vulkan] GPU selected: {} (api {}.{})", props.deviceName,
               VK_API_VERSION_MAJOR(props.apiVersion),
               VK_API_VERSION_MINOR(props.apiVersion));
  return true;
}

void VulkanContext::selectQueueFamilies() {
  u32 familyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount,
                                           nullptr);
  std::vector<VkQueueFamilyProperties> families(familyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount,
                                           families.data());

  // Prefer one family exposing both graphics and compute.
  for (u32 i = 0; i < familyCount; ++i) {
    if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
        (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
      graphicsFamily = computeFamily = i;
      break;
    }
  }
  if (graphicsFamily == 0xFFFFFFFFu) {
    for (u32 i = 0; i < familyCount; ++i) {
      if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        graphicsFamily = i;
        break;
      }
    }
    for (u32 i = 0; i < familyCount; ++i) {
      if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        computeFamily = i;
        break;
      }
    }
  }

  // Dedicated async transfer family (typical on NVIDIA: extra families
  // with TRANSFER only) > compute-only family > graphics family.
  u32 dedicated = 0xFFFFFFFFu, asyncCompute = 0xFFFFFFFFu;
  for (u32 i = 0; i < familyCount; ++i) {
    const auto flags = families[i].queueFlags;
    if (!(flags & VK_QUEUE_TRANSFER_BIT)) continue;
    if (!(flags & VK_QUEUE_GRAPHICS_BIT) && !(flags & VK_QUEUE_COMPUTE_BIT))
      dedicated = (dedicated == 0xFFFFFFFFu) ? i : dedicated;
    else if (!(flags & VK_QUEUE_GRAPHICS_BIT))
      asyncCompute = (asyncCompute == 0xFFFFFFFFu) ? i : asyncCompute;
  }
  transferFamily = dedicated != 0xFFFFFFFFu      ? dedicated
                   : asyncCompute != 0xFFFFFFFFu ? asyncCompute
                                                 : graphicsFamily;
  // M4.3 async compute: a compute-only family (true second engine on
  // some hardware) or a second queue from the graphics family. The
  // KENSHIN_SAME_FAMILY_ASYNC env var forces the same-family variant -
  // the A/B against the cross-family CONCURRENT-sharing default (the
  // todoList risk-point review).
  const bool forceSameFamily =
      std::getenv("KENSHIN_SAME_FAMILY_ASYNC") != nullptr;
  asyncComputeFamily =
      (asyncCompute != 0xFFFFFFFFu && !forceSameFamily) ? asyncCompute
                                                        : 0xFFFFFFFFu;
  sameFamilyAsyncCompute =
      asyncComputeFamily == 0xFFFFFFFFu &&
      (families[graphicsFamily].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
      families[graphicsFamily].queueCount >= 2;
  for (u32 i = 0; i < familyCount; ++i) {
    const auto& f = families[i];
    KENSHIN_INFO("[vulkan] queue family {}: flags{}{}{} count {}", i,
                 f.queueFlags & VK_QUEUE_GRAPHICS_BIT ? " graphics" : "",
                 f.queueFlags & VK_QUEUE_COMPUTE_BIT ? " compute" : "",
                 f.queueFlags & VK_QUEUE_TRANSFER_BIT ? " transfer" : "",
                 f.queueCount);
  }
  KENSHIN_INFO(
      "[vulkan] async compute: {}",
      asyncComputeFamily != 0xFFFFFFFFu
          ? "dedicated compute-only family"
          : (sameFamilyAsyncCompute ? "second queue from the graphics family"
                                    : "unavailable"));
}

bool VulkanContext::checkPresentSupport(void* hwnd) {
  if (graphicsFamily == 0xFFFFFFFFu || hwnd == nullptr) return false;

  // vkCreateWin32SurfaceKHR is platform-gated in volk.c; fetch it at
  // runtime instead of linking against a symbol that may not exist.
  const auto pfnCreateWin32Surface =
      reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(
          vkGetInstanceProcAddr(instance, "vkCreateWin32SurfaceKHR"));
  if (pfnCreateWin32Surface == nullptr) {
    KENSHIN_WARN("[vulkan] vkCreateWin32SurfaceKHR not available");
    return false;
  }

  const VkWin32SurfaceCreateInfoKHR surfaceInfo{
      .sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
      .hinstance = GetModuleHandleW(nullptr),
      .hwnd = static_cast<HWND>(hwnd),
  };
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  if (pfnCreateWin32Surface(instance, &surfaceInfo, nullptr, &surface) !=
      VK_SUCCESS)
    return false;

  VkBool32 supported = VK_FALSE;
  vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, graphicsFamily, surface,
                                       &supported);
  vkDestroySurfaceKHR(instance, surface, nullptr);
  return supported == VK_TRUE;
}

}  // namespace backend

KENSHIN_NS_END
