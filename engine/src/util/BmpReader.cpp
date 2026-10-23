#include "kenshin/util/BmpReader.h"
#include "kenshin/util/File.h"

#include <cstdio>
#include <cstring>

KENSHIN_NS_BEGIN

namespace util {

bool readBmp24(const std::string& path, ImageRgba& out, std::string& log) {
  out = ImageRgba{};
  FILE* f = nullptr;
  if ((f = kenshin::util::openFile(path.c_str(), "rb")) == nullptr) {
    log = "cannot open: " + path;
    return false;
  }
  std::vector<u8> file;
  {
    char buf[4096];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0)
      file.insert(file.end(), buf, buf + n);
    std::fclose(f);
  }
  // File header (14B) + info header (40B) - the writeBmp24 layout. The
  // BMP parse offsets were double-checked against the spec: width at
  // 14+4, height at 14+8 (the M8.1 session's 40x640 mishap came from
  // mixing bfOffBits into this arithmetic - keep the +14 base explicit).
  if (file.size() < 54) {
    log = "truncated BMP (<54 bytes)";
    return false;
  }
  if (file[0] != 'B' || file[1] != 'M') {
    log = "not a BMP (magic)";
    return false;
  }
  const u32 dataOffset = *reinterpret_cast<const u32*>(&file[10]);
  const u32 infoSize = *reinterpret_cast<const u32*>(&file[14]);
  const i32 width = *reinterpret_cast<const i32*>(&file[18]);
  const i32 height = *reinterpret_cast<const i32*>(&file[22]);
  const u16 bitCount = *reinterpret_cast<const u16*>(&file[28]);
  const u32 compression = *reinterpret_cast<const u32*>(&file[30]);
  if (infoSize < 40) {
    log = "unsupported info header size " + std::to_string(infoSize);
    return false;
  }
  if (width <= 0 || height == 0) {
    log = "degenerate dimensions";
    return false;
  }
  if (bitCount != 24 || compression != 0) {
    log = "only 24-bit BI_RGB supported (got bitCount=" +
          std::to_string(bitCount) +
          " compression=" + std::to_string(compression) + ")";
    return false;
  }
  const bool bottomUp = height > 0;
  const u32 w = static_cast<u32>(width);
  const u32 h = static_cast<u32>(bottomUp ? height : -height);
  const usize stride = (static_cast<usize>(w) * 3 + 3) & ~usize(3);
  const usize pixelBytes = stride * h;
  if (dataOffset < 54 || dataOffset + pixelBytes > file.size()) {
    log = "pixel data out of file bounds";
    return false;
  }

  out.width = w;
  out.height = h;
  out.rgba.resize(static_cast<usize>(w) * h * 4);
  const u8* src = file.data() + dataOffset;
  for (u32 y = 0; y < h; ++y) {
    // Output rows are top-down; a bottom-up file stores them inverted.
    const u32 srcY = bottomUp ? h - 1 - y : y;
    const u8* row = src + srcY * stride;
    u8* dst = out.rgba.data() + static_cast<usize>(y) * w * 4;
    for (u32 x = 0; x < w; ++x) {
      dst[x * 4 + 0] = row[x * 3 + 2];  // BGR -> RGBA
      dst[x * 4 + 1] = row[x * 3 + 1];
      dst[x * 4 + 2] = row[x * 3 + 0];
      dst[x * 4 + 3] = 255;
    }
  }
  return true;
}

}  // namespace util

KENSHIN_NS_END
