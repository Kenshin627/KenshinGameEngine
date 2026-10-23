#include "kenshin/assets/Image.h"

#include <cstring>
#include <format>

#include "kenshin/util/File.h"
#include "stb/stb_image.h"

KENSHIN_NS_BEGIN
namespace assets {

namespace {
constexpr int kMaxSide = 16384;

const char* reason() {
  const char* r = stbi_failure_reason();
  return r ? r : "unknown error";
}
}  // namespace

bool decodeImage(std::span<const u8> bytes, Image& out, std::string& error) {
  if (bytes.empty() || bytes.size() > static_cast<usize>(INT32_MAX)) {
    error = "empty or oversized image data";
    return false;
  }
  const auto* data = reinterpret_cast<const stbi_uc*>(bytes.data());
  const int len = static_cast<int>(bytes.size());
  int w = 0, h = 0, comp = 0;
  if (!stbi_info_from_memory(data, len, &w, &h, &comp)) {
    error = std::format("not a decodable image ({})", reason());
    return false;
  }
  if (w <= 0 || h <= 0 || w > kMaxSide || h > kMaxSide) {
    error = std::format("image is {}x{} (limit {} per side)", w, h, kMaxSide);
    return false;
  }
  Image img;
  img.width = static_cast<u32>(w);
  img.height = static_cast<u32>(h);
  img.hasAlpha = comp == 2 || comp == 4;
  void* pixels = nullptr;
  if (stbi_is_hdr_from_memory(data, len)) {
    img.depth = Image::Depth::F32;
    pixels = stbi_loadf_from_memory(data, len, &w, &h, &comp, 4);
  } else if (stbi_is_16_bit_from_memory(data, len)) {
    img.depth = Image::Depth::U16;
    pixels = stbi_load_16_from_memory(data, len, &w, &h, &comp, 4);
  } else {
    img.depth = Image::Depth::U8;
    pixels = stbi_load_from_memory(data, len, &w, &h, &comp, 4);
  }
  if (!pixels) {
    error = std::format("decode failed ({})", reason());
    return false;
  }
  img.pixels.resize(img.byteSize());
  std::memcpy(img.pixels.data(), pixels, img.pixels.size());
  stbi_image_free(pixels);
  out = std::move(img);
  return true;
}

bool loadImageFile(const std::filesystem::path& path, Image& out, std::string& error) {
  std::string bytes;
  if (!util::readFile(path, bytes)) {
    error = "cannot read " + util::pathToUtf8(path);
    return false;
  }
  if (!decodeImage({reinterpret_cast<const u8*>(bytes.data()), bytes.size()}, out, error)) {
    error = util::pathToUtf8(path) + ": " + error;
    return false;
  }
  return true;
}

Image makeImageRgba8(u32 width, u32 height, std::vector<u8> rgba) {
  Image img;
  img.width = width;
  img.height = height;
  img.depth = Image::Depth::U8;
  img.hasAlpha = true;
  img.pixels = std::move(rgba);
  img.pixels.resize(img.byteSize());
  return img;
}

}  // namespace assets
KENSHIN_NS_END
