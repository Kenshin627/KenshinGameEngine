#pragma once

// ── Material assets (.kmat, roadmap 1.3c) ───────────────────────
// A metallic-roughness material as its own asset: factors, alpha mode,
// and texture slots that reference TEXTURE ASSETS by Guid with the
// sampler to use. Text JSON like scenes - canonical, strict, versioned:
//
//   {
//     "format": "kenshin.material",
//     "version": 1,
//     "base_color": [1.0, 1.0, 1.0, 1.0],
//     "metallic": 0.0,
//     "roughness": 0.5,
//     "emissive": [0.0, 0.0, 0.0],
//     "alpha_mode": "opaque",          // opaque | mask | blend
//     "alpha_cutoff": 0.5,
//     "double_sided": false,
//     "normal_scale": 1.0,
//     "occlusion_strength": 1.0,
//     "textures": {
//       "base_color": {"texture": "<guid>", "wrap_u": "repeat",
//                      "wrap_v": "repeat", "filter": "linear"}
//     }
//   }
//
// Texture slots: base_color, metallic_roughness, normal, occlusion,
// emissive (absent = none). Factors multiply their textures (glTF).

#include <glm/glm.hpp>
#include <string>
#include <string_view>

#include "kenshin/backend/TextureFormat.h"
#include "kenshin/core/Guid.h"
#include "kenshin/serialization/Json.h"

KENSHIN_NS_BEGIN
namespace assets {

inline constexpr std::string_view kMaterialFormat = "kenshin.material";
inline constexpr i32 kMaterialVersion = 1;

struct MaterialTextureSlot {
  core::Guid texture;  // null = no texture
  backend::SamplerDesc sampler;
  bool valid() const { return !texture.isNull(); }
  friend bool operator==(const MaterialTextureSlot&, const MaterialTextureSlot&) = default;
};

struct MaterialDesc {
  enum class AlphaMode : u8 { Opaque, Mask, Blend };

  glm::vec4 baseColor{1.0f};  // linear RGB + alpha
  f32 metallic = 0.0f;
  f32 roughness = 0.5f;
  glm::vec3 emissive{0.0f};  // linear radiance (strength folded in)
  AlphaMode alphaMode = AlphaMode::Opaque;
  f32 alphaCutoff = 0.5f;
  bool doubleSided = false;
  f32 normalScale = 1.0f;
  f32 occlusionStrength = 1.0f;
  MaterialTextureSlot baseColorTexture;
  MaterialTextureSlot metallicRoughnessTexture;
  MaterialTextureSlot normalTexture;
  MaterialTextureSlot occlusionTexture;
  MaterialTextureSlot emissiveTexture;

  friend bool operator==(const MaterialDesc&, const MaterialDesc&) = default;
};

const serialization::SchemaMigrator& materialMigrator();

serialization::Json materialToJson(const MaterialDesc& material);
std::string saveMaterial(const MaterialDesc& material);
/// Strict: unknown keys, wrong types, out-of-range factors, bad enum
/// labels or guids are errors; `out` is untouched on failure. Missing
/// keys keep their defaults.
bool materialFromJson(serialization::Json doc, MaterialDesc& out, std::string* error);
bool loadMaterial(std::string_view text, MaterialDesc& out, std::string* error);

}  // namespace assets
KENSHIN_NS_END
