#pragma once

// ── Minimal 24-bit BMP writer (S1.3 screenshot pipeline) ────────
// Zero-dependency, deterministic output. Input: RGBA8 rows top-down.

#include "kenshin/Common.h"

#include <string>
#include <vector>

KENSHIN_NS_BEGIN

namespace util
{

    /// Write a 24-bit BMP (bottom-up BGR rows, 4-byte aligned).
    /// @param rgba      width*height*4 bytes, rows top-down.
    /// @return true on success.
    bool writeBmp24(const std::string& path, const std::vector<u8>& rgba,
                    u32 width, u32 height);

} // namespace util

KENSHIN_NS_END
