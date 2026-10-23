#include "kenshin/manager/TextureManager.h"

#include <format>

#include "kenshin/Engine.h"
#include "kenshin/assets/Image.h"
#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/util/File.h"

KENSHIN_NS_BEGIN

namespace manager {

bool TextureManager::onInit(Engine& engine) {
  m_backend = &engine.get<backend::IRenderBackend>();
  return m_backend != nullptr;
}

u32 TextureManager::upload(const assets::CookedTexture& texture, const char* debugName) {
  backend::IRenderBackend::TextureDesc desc;
  desc.width = texture.width;
  desc.height = texture.height;
  desc.mipCount = static_cast<u32>(texture.mips.size());
  desc.format = texture.format;
  desc.data = texture.data.data();
  desc.bytes = texture.data.size();
  desc.debugName = debugName;
  const u32 handle = m_backend->createTexture(desc);
  if (handle != backend::IRenderBackend::kInvalidTextureHandle) m_textures.push_back(handle);
  return handle;
}

u32 TextureManager::reference(u32 textureHandle, backend::SamplerDesc sampler) const {
  const u32 slot = m_backend->bindlessSlot(textureHandle);
  return slot == backend::kNoTexture ? backend::kNoTexture : backend::textureRef(slot, sampler);
}

u32 TextureManager::load(const std::string& path, const assets::TextureSettings& settings,
                         backend::SamplerDesc sampler) {
  const std::string key =
      std::format("{}|{}{}|{}|{}", path, static_cast<int>(settings.colorSpace),
                  settings.normalMap ? "n" : "", settings.mipmaps ? 1 : 0, settings.maxSize);
  if (const auto it = m_byKey.find(key); it != m_byKey.end())
    return reference(it->second, sampler);

  assets::Image image;
  std::string error;
  if (!assets::loadImageFile(util::pathFromUtf8(path), image, error)) {
    KENSHIN_ERROR("[texture] {}", error);
    return backend::kNoTexture;
  }
  assets::CookedTexture cooked;
  if (!assets::cookTexture(image, settings, cooked, error)) {
    KENSHIN_ERROR("[texture] {}: {}", path, error);
    return backend::kNoTexture;
  }
  const u32 handle = upload(cooked, path.c_str());
  if (handle == backend::IRenderBackend::kInvalidTextureHandle) {
    KENSHIN_ERROR("[texture] {}: upload failed", path);
    return backend::kNoTexture;
  }
  m_byKey.emplace(key, handle);
  KENSHIN_INFO("[texture] {} -> {}x{}, {} mips, slot {}", path, cooked.width, cooked.height,
               cooked.mips.size(), m_backend->bindlessSlot(handle));
  return reference(handle, sampler);
}

void TextureManager::onShutdown() {
  if (m_backend)
    for (const u32 h : m_textures) m_backend->destroyTexture(h);
  m_textures.clear();
  m_byKey.clear();
  m_backend = nullptr;
}

}  // namespace manager

KENSHIN_NS_END
