#pragma once

// ── TextureManager — albedo texture lifetime (M8.2) ─────────────
// Loads 24-bit BMPs through util::readBmp24 and uploads them as
// layers of ONE Texture2DArray (uniform dimensions - the array
// contract; mismatched sizes load into their own array handle
// instead). Scene material tables reference layers by index.

#include <string>
#include <vector>

#include "kenshin/Common.h"

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

  /// Load a BMP and append it as the next layer of the shared array
  /// (the FIRST load fixes the array's dimensions - the
  /// Texture2DArray contract; a mismatching image is REJECTED with a
  /// loud error). Returns the texture-array layer index, or
  /// kInvalidLayer on failure.
  static constexpr u32 kInvalidLayer = 0xFFFFFFFFu;
  u32 load(const std::string& path);

  /// The shared texture-array backend handle (kInvalidTextureHandle
  /// before the first successful load - the caller then binds the
  /// backend's dummy).
  u32 textureArray() const { return m_texture; }
  u32 layerCount() const { return m_layers; }

  /// Destroy every loaded array — runs before backend teardown (LIFO).
  void onShutdown();

 private:
  backend::IRenderBackend* m_backend = nullptr;
  u32 m_texture = 0xFFFFFFFFu;
  u32 m_layers = 0;
  u32 m_width = 0;
  u32 m_height = 0;
  std::vector<u8> m_pixels;  // width*height*4*layers, append-only
};

}  // namespace manager

KENSHIN_NS_END
