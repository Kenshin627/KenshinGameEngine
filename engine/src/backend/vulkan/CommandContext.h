#pragma once

// ── CommandContext — the frame's command recording ───────────────
// P0-1: owns the current frame slot's command buffer lifecycle and the
// shared barrier/clear recording helpers. P0-2 grows the single
// per-frame rendering scope + pass switching here.

#include <volk.h>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace backend {

class CommandContext {
 public:
  /// Reset + begin the slot's command buffer (one-time submit).
  bool begin(VkCommandBuffer cmd);
  /// End recording; false on failure (caller drops the frame).
  bool end();

  VkCommandBuffer cmd() const { return m_cmd; }

  // ---- recording helpers (shared by the frame paths) ─────────────

  /// sync2 pipeline barrier with a single image barrier.
  static void pipelineBarrier(VkCommandBuffer cmd,
                              const VkImageMemoryBarrier2& barrier);

  /// Frame-loop image transition: "everything written so far" (src side is
  /// ALL_COMMANDS + MEMORY_WRITE) into the given dst stage/access + layout.
  static VkImageMemoryBarrier2 imageBarrier(
      VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
      VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess);

  /// Clear a color image (RGBA floats) in the given (already transitioned)
  /// layout; full mip 0 / layer 0 / single aspect range.
  static void clearColorImage(VkCommandBuffer cmd, VkImage image,
                              VkImageLayout layout, const f32 color[4]);

 private:
  VkCommandBuffer m_cmd = VK_NULL_HANDLE;
};

}  // namespace backend

KENSHIN_NS_END
