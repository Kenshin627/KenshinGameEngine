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
  /// M4.2 hardware raster: a rendering scope with NO attachments (the
  /// only output is the VisBuffer SSBO atomic written by the PS).
  void beginNullPass(u32 width, u32 height);
  /// M8.1 deferred shading: the decode pass's GBuffer scope - TWO color
  /// attachments (A = normal+distance, B = base color), no depth (the
  /// VisBuffer already resolved visibility), CLEAR to zero (the
  /// lighting pass reads alpha==0 as "background").
  void beginGBufferPass(VkImageView colorA, VkImageView colorB, u32 width,
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
  /// Subresource window defaults to mip 0 / layer 0 / one each (the
  /// frame targets are all single-mip single-layer); the M8.2 texture
  /// upload drives per-mip-range transitions through the overrides.
  static VkImageMemoryBarrier2 imageBarrier(
      VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
      VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess, u32 baseMip = 0,
      u32 mipCount = 1, u32 baseLayer = 0, u32 layerCount = 1);

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
