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
  /// Reset + begin the slot's command buffer (one-time submit). Closes any
  /// stale pass state from the previous use of this slot.
  bool begin(VkCommandBuffer cmd);
  /// End recording; false on failure (caller drops the frame). m_cmd
  /// stays valid until the next begin(): the submit path still reads it.
  bool end();

  VkCommandBuffer cmd() const { return m_cmd; }

  // ---- the frame's single viewport rendering scope ────────────────
  /// Open it (color LOAD - beginFrame already cleared the image; depth
  /// CLEAR once here) and set the viewport/scissor. All draws of the
  /// frame record into this one scope: the render pass's implicit subpass
  /// ordering replaces the hand-rolled inter-draw barriers.
  void beginViewportPass(VkImageView color, VkImageView depth, u32 width,
                         u32 height);
  /// Close the viewport scope (no-op when it never opened).
  void endPass();
  bool passOpen() const { return m_passOpen; }

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
  bool m_passOpen = false;
};

}  // namespace backend

KENSHIN_NS_END
