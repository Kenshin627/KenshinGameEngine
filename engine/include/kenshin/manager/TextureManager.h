#pragma once

// ── TextureManager — material texture lifetime (1.3a) ───────────
// Decodes an image file (PNG/JPEG/TGA/BMP/HDR), cooks its mip chain on
// the CPU (assets::cookTexture: linear-light filtering, any size) and
// uploads it as its own GPU texture in the bindless table. Materials
// hold the returned 32-bit texture reference (bindless slot | sampler,
// backend::textureRef). Loads are deduplicated by (path, cook
// settings); the same image with two samplers shares one texture.
//
// The asset database (roadmap 1.3c) will feed cooked KTX2 artifacts
// through upload() instead of decoding sources at load time.

#include <string>
#include <unordered_map>
#include <vector>

#include "kenshin/Common.h"
#include "kenshin/assets/TextureCooker.h"
#include "kenshin/backend/TextureFormat.h"

KENSHIN_NS_BEGIN

class Engine;
namespace backend {
class IRenderBackend;
}

namespace manager {

class TextureManager {
 public:
  /// Subsystem hook: resolve the render backend from the registry.
  bool onInit(Engine& engine);
  /// Destroy every texture — runs before backend teardown (LIFO).
  void onShutdown();

  /// Load (or reuse) an image file. Returns its texture reference, or
  /// backend::kNoTexture on failure (logged; the material then renders
  /// with its factors only).
  u32 load(const std::string& path, const assets::TextureSettings& settings = {},
           backend::SamplerDesc sampler = {});
  /// Upload an already-cooked texture; returns the backend texture
  /// handle (kInvalidTextureHandle on failure). Not deduplicated.
  u32 upload(const assets::CookedTexture& texture, const char* debugName);
  /// The reference a material stores for an uploaded texture handle.
  u32 reference(u32 textureHandle, backend::SamplerDesc sampler) const;

  u32 textureCount() const { return static_cast<u32>(m_textures.size()); }

 private:
  backend::IRenderBackend* m_backend = nullptr;
  std::vector<u32> m_textures;                     // owned backend handles
  std::unordered_map<std::string, u32> m_byKey;    // load key -> handle
};

}  // namespace manager

KENSHIN_NS_END
