#include "kenshin/manager/TextureManager.h"

#include "kenshin/Engine.h"
#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/util/BmpReader.h"

KENSHIN_NS_BEGIN

namespace manager {

bool TextureManager::onInit(Engine& engine) {
  m_backend = &engine.get<backend::IRenderBackend>();
  return m_backend != nullptr;
}

u32 TextureManager::load(const std::string& path) {
  util::ImageRgba img;
  std::string log;
  if (!util::readBmp24(path, img, log)) {
    KENSHIN_ERROR("[texture] load failed: {}", log);
    return kInvalidLayer;
  }
  if (m_texture != 0xFFFFFFFFu &&
      (img.width != m_width || img.height != m_height)) {
    // One shared Texture2DArray = one dimension set (the descriptor
    // contract). v1 rejects instead of managing multiple arrays.
    KENSHIN_ERROR(
        "[texture] {} is {}x{} but the array is {}x{} - all albedo "
        "textures must share dimensions",
        path, img.width, img.height, m_width, m_height);
    return kInvalidLayer;
  }

  m_pixels.insert(m_pixels.end(), img.rgba.begin(), img.rgba.end());
  const u32 layer = m_layers;
  if (m_layers == 0) {
    m_width = img.width;
    m_height = img.height;
  }
  ++m_layers;

  // Re-upload the grown array (scene-load-time path: a handful of
  // one-shot uploads, the waitIdle inside destroy/create is fine).
  const u32 old = m_texture;
  backend::IRenderBackend::TextureArrayDesc desc;
  desc.width = m_width;
  desc.height = m_height;
  desc.layers = m_layers;
  desc.rgba = m_pixels.data();
  desc.bytes = m_pixels.size();
  desc.debugName = "albedo-array";
  m_texture = m_backend->createTextureArray(desc);
  if (m_texture == backend::IRenderBackend::kInvalidTextureHandle) {
    KENSHIN_ERROR("[texture] array upload failed ({} layers)", m_layers);
    m_texture = old;  // keep the previous handle alive if there was one
    --m_layers;
    m_pixels.resize(static_cast<usize>(m_width) * m_height * 4 * m_layers);
    return kInvalidLayer;
  }
  if (old != 0xFFFFFFFFu) m_backend->destroyTexture(old);
  KENSHIN_INFO("[texture] {} -> layer {} (array {}x{}x{})", path, layer,
               m_width, m_height, m_layers);
  return layer;
}

void TextureManager::onShutdown() {
  if (m_backend && m_texture != 0xFFFFFFFFu)
    m_backend->destroyTexture(m_texture);
  m_texture = 0xFFFFFFFFu;
  m_layers = 0;
  m_pixels.clear();
  m_pixels.shrink_to_fit();
  m_backend = nullptr;
}

}  // namespace manager

KENSHIN_NS_END
