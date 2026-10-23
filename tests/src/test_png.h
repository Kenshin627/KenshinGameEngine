#pragma once

// A minimal PNG encoder for tests (stored deflate blocks, zlib adler,
// CRC32 chunks): real PNG bytes - including 16-bit and grey+alpha
// variants - without binary fixtures. Used by texture_tests and
// gltf_import_tests.

#include <vector>

#include "kenshin/Common.h"
#include "kenshin/util/Crc32.h"

namespace testpng {

using kenshin::u32;
using kenshin::u8;
using kenshin::usize;

inline void be32(std::vector<u8>& b, u32 v) {
  for (int s = 24; s >= 0; s -= 8) b.push_back(static_cast<u8>(v >> s));
}
inline void chunk(std::vector<u8>& png, const char* type, const std::vector<u8>& data) {
  be32(png, static_cast<u32>(data.size()));
  std::vector<u8> body(type, type + 4);
  body.insert(body.end(), data.begin(), data.end());
  png.insert(png.end(), body.begin(), body.end());
  be32(png, kenshin::util::crc32Update(0, body.data(), body.size()));
}
/// colorType 0 grey, 2 RGB, 4 grey+alpha, 6 RGBA; `samples` holds the
/// channel values (0..255 or 0..65535), row-major.
inline std::vector<u8> makePng(u32 w, u32 h, int bitDepth, int colorType,
                        const std::vector<u32>& samples) {
  const int channels = colorType == 0 ? 1 : colorType == 2 ? 3 : colorType == 4 ? 2 : 4;
  std::vector<u8> raw;
  for (u32 y = 0; y < h; ++y) {
    raw.push_back(0);  // filter: none
    for (u32 x = 0; x < w * channels; ++x) {
      const u32 v = samples[y * w * channels + x];
      if (bitDepth == 16) raw.push_back(static_cast<u8>(v >> 8));
      raw.push_back(static_cast<u8>(v));
    }
  }
  std::vector<u8> z = {0x78, 0x01};
  u32 a = 1, b = 0;
  for (const u8 c : raw) {
    a = (a + c) % 65521;
    b = (b + a) % 65521;
  }
  for (usize off = 0; off < raw.size() || off == 0;) {
    const usize n = std::min<usize>(65535, raw.size() - off);
    const bool last = off + n >= raw.size();
    z.push_back(last ? 1 : 0);
    z.push_back(static_cast<u8>(n));
    z.push_back(static_cast<u8>(n >> 8));
    z.push_back(static_cast<u8>(~n));
    z.push_back(static_cast<u8>(~n >> 8));
    z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(off),
             raw.begin() + static_cast<std::ptrdiff_t>(off + n));
    off += n;
    if (last) break;
  }
  be32(z, (b << 16) | a);
  std::vector<u8> png = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
  std::vector<u8> ihdr;
  be32(ihdr, w);
  be32(ihdr, h);
  ihdr.insert(ihdr.end(), {static_cast<u8>(bitDepth), static_cast<u8>(colorType), 0, 0, 0});
  chunk(png, "IHDR", ihdr);
  chunk(png, "IDAT", z);
  chunk(png, "IEND", {});
  return png;
}

}  // namespace testpng
