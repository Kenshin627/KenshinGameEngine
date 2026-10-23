#pragma once

// ── Texture formats shared by the cooker, KTX2 files and the RHI ──
// Deliberately small: the uncompressed formats the engine cooks today
// (GPU block compression comes with export, roadmap Phase 8). Values
// map 1:1 to VkFormat for KTX2's vkFormat field.

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN
namespace backend {

enum class TextureFormat : u8 {
  RGBA8Unorm,  // linear data: normal, metallic-roughness, occlusion maps
  RGBA8Srgb,   // colour: base colour, emissive (hardware sRGB decode)
  RGBA16Unorm, // 16-bit linear data (16-bit PNG normal/height maps)
  RGBA16Float, // HDR / 16-bit colour decoded to linear
};

inline constexpr u32 kTextureFormatCount = 4;

/// Bytes per texel.
constexpr u32 texelBytes(TextureFormat f) {
  return f == TextureFormat::RGBA8Unorm || f == TextureFormat::RGBA8Srgb ? 4u : 8u;
}

/// The VkFormat enumerant (numeric; no Vulkan header needed).
constexpr u32 vkFormatOf(TextureFormat f) {
  switch (f) {
    case TextureFormat::RGBA8Unorm: return 37;   // VK_FORMAT_R8G8B8A8_UNORM
    case TextureFormat::RGBA8Srgb: return 43;    // VK_FORMAT_R8G8B8A8_SRGB
    case TextureFormat::RGBA16Unorm: return 91;  // VK_FORMAT_R16G16B16A16_UNORM
    case TextureFormat::RGBA16Float: return 97;  // VK_FORMAT_R16G16B16A16_SFLOAT
  }
  return 0;
}

/// Mip count of a full chain down to 1x1.
constexpr u32 fullMipCount(u32 width, u32 height) {
  u32 n = 1;
  for (u32 m = width > height ? width : height; m > 1; m >>= 1) ++n;
  return n;
}

// ── Samplers and bindless texture references ─────────────────────
// Material textures live in one bindless table; a material refers to
// one by a 32-bit reference = table slot (low 24 bits) | sampler index
// (high 8 bits). The sampler table is fixed: every wrap-mode pair x
// filter, so glTF samplers map onto it exactly. bindless.slang decodes
// the same layout - keep the two in sync.

enum class WrapMode : u8 { Repeat = 0, Clamp = 1, Mirror = 2 };
enum class FilterMode : u8 {
  Linear = 0,   // trilinear + anisotropic
  Nearest = 1,  // nearest texel and mip (pixel art)
};

struct SamplerDesc {
  WrapMode wrapU = WrapMode::Repeat;
  WrapMode wrapV = WrapMode::Repeat;
  FilterMode filter = FilterMode::Linear;
  friend bool operator==(const SamplerDesc&, const SamplerDesc&) = default;
};

inline constexpr u32 kSamplerCount = 18;

constexpr u32 samplerIndex(SamplerDesc d) {
  return u32(d.filter) * 9u + u32(d.wrapV) * 3u + u32(d.wrapU);
}
constexpr SamplerDesc samplerFromIndex(u32 index) {
  return {static_cast<WrapMode>(index % 3), static_cast<WrapMode>((index / 3) % 3),
          static_cast<FilterMode>(index / 9)};
}

/// "No texture" in a material slot.
inline constexpr u32 kNoTexture = 0xFFFFFFFFu;
inline constexpr u32 kMaxBindlessSlots = 1u << 24;

constexpr u32 textureRef(u32 slot, SamplerDesc sampler) {
  return (slot & (kMaxBindlessSlots - 1)) | (samplerIndex(sampler) << 24);
}
constexpr u32 textureRefSlot(u32 ref) { return ref & (kMaxBindlessSlots - 1); }
constexpr u32 textureRefSampler(u32 ref) { return ref >> 24; }

}  // namespace backend
KENSHIN_NS_END
