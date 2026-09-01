#pragma once

// ── Vulkan API utilities (backend-internal) ─────────────────────
// Thin wrapper over the C API (docs/vulkan-rhi-plan.md §2):
//   KE_VK(call)      - result discipline: log file:line + result name,
//                      assert-abort in debug, returns bool for release paths
//   resultName()     - VkResult -> human readable
//   setName()        - debug-utils object naming (RenderDoc/NSight)
//   ScopedDebugLabel - command buffer region markers
//   xxxCreate factories - defaults for high-frequency structs; one-off
//                      structs stay at call sites with designated
//                      initializers (policy must stay visible)
//
// No UniqueHandle wrappers by design: setup-phase objects live as Impl
// members, runtime resources live in the (planned) deletion queue.

#include <volk.h>

#include <cassert>
#include <cstdint>

#include "kenshin/core/Log.h"

KENSHIN_NS_BEGIN

namespace backend {
namespace vkutil {

/// VkResult -> "VK_ERROR_OUT_OF_DEVICE_MEMORY" (nullptr-safe fallback).
inline const char* resultName(VkResult result) {
  switch (result) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_EVENT_SET: return "VK_EVENT_SET";
    case VK_EVENT_RESET: return "VK_EVENT_RESET";
    case VK_INCOMPLETE: return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:
      return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT:
      return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED:
      return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
    case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
    case VK_ERROR_INVALID_EXTERNAL_HANDLE:
      return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
    case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
    case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
    case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
    default: return "VK_ERROR_<unknown>";
  }
}

/// KE_VK backs here. Debug: a failed call aborts at the exact site.
/// Release: logs and returns false so init paths can bail out.
inline bool check(VkResult result, const char* call, const char* file,
                  int line) {
  if (result == VK_SUCCESS) return true;
  KENSHIN_ERROR("[vulkan] {} failed with {} at {}:{}", call, resultName(result),
                file, line);
  assert(result == VK_SUCCESS && "Vulkan call failed (see log above)");
  return false;
}

// Set by VulkanContext when VK_EXT_debug_utils is actually enabled; naming
// and labels are no-ops otherwise (zero release cost).
inline bool g_debugUtilsEnabled = false;

/// Name a Vulkan object for debugging tools (buffer/pipeline/image/...).
inline void setName(VkDevice device, VkObjectType type, uint64_t handle,
                    const char* name) {
  if (!g_debugUtilsEnabled || device == VK_NULL_HANDLE) return;
  const VkDebugUtilsObjectNameInfoEXT info{
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
      .objectType = type,
      .objectHandle = handle,
      .pObjectName = name,
  };
  vkSetDebugUtilsObjectNameEXT(device, &info);
}

/// Command buffer region marker ("pass name" in RenderDoc's event browser).
class ScopedDebugLabel {
 public:
  ScopedDebugLabel(VkCommandBuffer cmd, const char* name)
      : m_cmd(vkutil::g_debugUtilsEnabled ? cmd : VK_NULL_HANDLE) {
    if (m_cmd != VK_NULL_HANDLE) {
      const VkDebugUtilsLabelEXT label{
          .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
          .pLabelName = name,
      };
      vkCmdBeginDebugUtilsLabelEXT(m_cmd, &label);
    }
  }
  ~ScopedDebugLabel() {
    if (m_cmd != VK_NULL_HANDLE) vkCmdEndDebugUtilsLabelEXT(m_cmd);
  }
  ScopedDebugLabel(const ScopedDebugLabel&) = delete;
  ScopedDebugLabel& operator=(const ScopedDebugLabel&) = delete;

 private:
  VkCommandBuffer m_cmd;
};

// ---- high-frequency struct factories (non-zero engineering defaults) --
// Only for structs reused across many sites; one-off state stays at the
// call site in designated-initializer form.

/// Full-resource image view for color/depth attachments and sampled reads.
inline VkImageViewCreateInfo imageView(VkImage image, VkFormat format,
                                       VkImageAspectFlags aspect) {
  return VkImageViewCreateInfo{
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = format,
      .subresourceRange = {aspect, 0, 1, 0, 1},
  };
}

/// sync2 image barrier between two layouts (identity layout transitions
/// included - only the stage/access masks differ in that case).
inline VkImageMemoryBarrier2 imageBarrier(
    VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
    VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
    VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
  return VkImageMemoryBarrier2{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = srcStage,
      .srcAccessMask = srcAccess,
      .dstStageMask = dstStage,
      .dstAccessMask = dstAccess,
      .oldLayout = oldLayout,
      .newLayout = newLayout,
      .image = image,
      .subresourceRange = {aspect, 0, 1, 0, 1},
  };
}

/// One command buffer recording pass (single submit per frame slot).
inline VkCommandBufferBeginInfo commandBufferBegin() {
  return VkCommandBufferBeginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
}

// No submitInfo() factory on purpose: the struct stores pointers to its
// arguments, so a returning factory would dangle. Submit sites build it
// with named locals in designated-initializer form.

}  // namespace vkutil
}  // namespace backend

KENSHIN_NS_END

/// Wrap every VkResult-returning call: KE_VK(vkCreateBuffer(...)).
#define KE_VK(call)                                              \
  ::kenshin::backend::vkutil::check((call), #call, __FILE__, __LINE__)
