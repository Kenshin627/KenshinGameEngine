#pragma once

// ── Minimal 24-bit BMP reader (M8.2 texture loading) ─────────────
// The mirror of writeBmp24: accepts exactly what the repo produces
// (and any 24-bit BI_RGB BMP - positive height bottom-up or negative
// height top-down). Zero-dependency; the ObjLoader failure discipline
// applies (return false + log string, never throw).

#include <string>
#include <vector>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace util {

/// Decoded RGBA8 image, rows top-down (the writeBmp24 input contract,
/// so a writer roundtrip is byte-exact).
struct ImageRgba {
  u32 width = 0;
  u32 height = 0;
  std::vector<u8> rgba;  // width*height*4

  bool valid() const { return width > 0 && height > 0 && !rgba.empty(); }
};

/// Load a 24-bit uncompressed BMP (BI_RGB). @param log receives the
/// failure reason on false. @return true on success.
bool readBmp24(const std::string& path, ImageRgba& out, std::string& log);

}  // namespace util

KENSHIN_NS_END
