#pragma once

// ── FrameStatus — the outcome of beginFrame ─────────────────────────
// Its own header so the renderer (and through it the app) can name it
// without including the backend API.

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace backend {

/// beginFrame outcome. Skipped is normal operation - the window has
/// nothing to present (minimized, swapchain out of date with a 0x0
/// surface): record nothing, call beginFrame again next frame.
enum class FrameStatus : u8 { Ready, Skipped, Error };

}  // namespace backend

KENSHIN_NS_END
