#pragma once

// ── Readback — frame content capture + host readback ────────────
// Owns the full-frame capture staging buffer and implements both
// readback paths: the viewport target (deterministic frame content)
// and the composited swapchain image (UI included). The capture-copy
// recording itself stays in endFrame (it is frame recording); this unit
// provides the staging buffer and the host-side waits/swizzles.

#include <volk.h>

#include <vma/vk_mem_alloc.h>

#include <vector>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace backend {

struct VulkanDevice;
struct FrameSync;

class Readback {
 public:
  /// (Re)create the capture staging buffer on size change.
  void ensureCaptureStaging(VmaAllocator allocator, u32 width, u32 height);

  /// The frame-recording side: request/mark capture, staging access.
  void requestCapture() { m_captureFull = true; }
  bool captureRequested() const { return m_captureFull; }
  void markCaptured() {
    m_captureFull = false;
    m_fullCaptured = true;
  }
  VkBuffer captureStaging() const { return m_captureStaging; }
  u32 captureWidth() const { return m_captureWidth; }
  u32 captureHeight() const { return m_captureHeight; }

  /// Copy the last completed frame's offscreen target to the host.
  /// Output RGBA8, rows top-down, BGRA swizzled when needed.
  bool readViewport(const VulkanDevice& device, const FrameSync& frameSync,
                    VkFormat swapchainFormat, std::vector<u8>& rgbaOut,
                    u32& width, u32& height);

  /// Copy the previously requested full-frame capture (UI included).
  bool readFullFrame(const VulkanDevice& device, const FrameSync& frameSync,
                     VkFormat swapchainFormat, std::vector<u8>& rgbaOut,
                     u32& width, u32& height);

  void destroy(VmaAllocator allocator);

 private:
  bool m_captureFull = false;
  bool m_fullCaptured = false;
  VkBuffer m_captureStaging = VK_NULL_HANDLE;
  VmaAllocation m_captureAlloc = VK_NULL_HANDLE;
  u32 m_captureWidth = 0, m_captureHeight = 0;
};

}  // namespace backend

KENSHIN_NS_END
