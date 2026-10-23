#pragma once

// ── Texture cooking (roadmap 1.3a) ──────────────────────────────
// Source Image -> GPU-ready texels with a full mip chain, computed on
// the CPU so every platform gets identical results:
//   * colour textures are filtered in LINEAR light (sRGB decoded,
//     filtered, re-encoded) - filtering sRGB codes darkens mips;
//   * colour is alpha-weighted (premultiplied while filtering), so
//     transparent texels do not bleed their colour into mips;
//   * normal maps are filtered as vectors and renormalized per mip;
//   * any size works: each mip texel averages exactly the source area
//     it covers (odd sizes included), down to 1x1.

#include <string>
#include <vector>

#include "kenshin/assets/Image.h"
#include "kenshin/backend/TextureFormat.h"

KENSHIN_NS_BEGIN
namespace assets {

struct TextureSettings {
  enum class ColorSpace : u8 { Srgb, Linear };
  /// Colour data (base colour, emissive) is sRGB; data maps are linear.
  ColorSpace colorSpace = ColorSpace::Srgb;
  /// Tangent-space normal map (implies Linear).
  bool normalMap = false;
  bool mipmaps = true;
  /// Largest allowed side; bigger sources keep only the mips that fit
  /// (0 = no limit).
  u32 maxSize = 0;
};

struct MipLevel {
  u32 width = 0;
  u32 height = 0;
  usize offset = 0;  // into CookedTexture::data
  usize size = 0;
};

struct CookedTexture {
  backend::TextureFormat format = backend::TextureFormat::RGBA8Srgb;
  u32 width = 0;   // mip 0
  u32 height = 0;
  std::vector<MipLevel> mips;  // mip 0 first, tightly packed in `data`
  std::vector<u8> data;
};

/// The storage format a source gets: 8-bit -> RGBA8 (sRGB or UNORM),
/// 16-bit linear -> RGBA16 UNORM, 16-bit colour and HDR -> RGBA16F.
backend::TextureFormat cookedFormat(const Image& image, const TextureSettings& settings);

bool cookTexture(const Image& image, const TextureSettings& settings, CookedTexture& out,
                 std::string& error);

// Exposed for tests: the transfer functions the cooker uses.
f32 srgbToLinear(f32 encoded);
f32 linearToSrgb(f32 linear);

}  // namespace assets
KENSHIN_NS_END
