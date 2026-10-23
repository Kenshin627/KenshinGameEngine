#pragma once

// ── Image: decoded source pixels (roadmap 1.3a) ─────────────────
// What an image file holds before cooking: always 4 channels (RGBA,
// alpha 1 when the file has none), top-down rows, at the file's own
// precision - 8-bit, 16-bit (PNG) or 32-bit float (Radiance HDR).

#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN
namespace assets {

struct Image {
  enum class Depth : u8 { U8, U16, F32 };

  u32 width = 0;
  u32 height = 0;
  Depth depth = Depth::U8;
  /// The file carried an alpha channel (grey+alpha or RGBA).
  bool hasAlpha = false;
  /// width * height * 4 components of `depth` (native-endian for U16/F32).
  std::vector<u8> pixels;

  usize componentBytes() const {
    return depth == Depth::U8 ? 1 : depth == Depth::U16 ? 2 : 4;
  }
  usize byteSize() const { return usize(width) * height * 4 * componentBytes(); }
};

/// Decodes PNG, JPEG, TGA, BMP or HDR from memory (format sniffed from
/// the bytes). Bounded: images over 16384 px on a side are rejected.
bool decodeImage(std::span<const u8> bytes, Image& out, std::string& error);
bool loadImageFile(const std::filesystem::path& path, Image& out, std::string& error);

/// An 8-bit RGBA image (tests, procedural sources).
Image makeImageRgba8(u32 width, u32 height, std::vector<u8> rgba);

}  // namespace assets
KENSHIN_NS_END
