#pragma once

// ── KTX 2.0 container (roadmap 1.3a) ────────────────────────────
// The cooked texture file format: Khronos KTX 2.0 with no
// supercompression, one layer, one face, the full mip chain. Written
// with a Basic Data Format Descriptor and a KTXwriter key so standard
// tools (ktx info/validate) can read it. The reader accepts exactly the
// uncompressed formats in backend::TextureFormat; Basis/UASTC sources
// (libktx) come with texture compression in Phase 8.

#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "kenshin/assets/TextureCooker.h"

KENSHIN_NS_BEGIN
namespace assets {

std::vector<u8> writeKtx2(const CookedTexture& texture);
/// Strict: identifier, header fields, level index bounds and sizes
/// are all validated; `out` is untouched on failure.
bool readKtx2(std::span<const u8> bytes, CookedTexture& out, std::string& error);

bool saveKtx2File(const std::filesystem::path& path, const CookedTexture& texture,
                  std::string* error = nullptr);
bool loadKtx2File(const std::filesystem::path& path, CookedTexture& out, std::string& error);

}  // namespace assets
KENSHIN_NS_END
