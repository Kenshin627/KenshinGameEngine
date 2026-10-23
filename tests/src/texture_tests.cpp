// Texture pipeline (roadmap step 1.3a): image decoding, CPU mip
// cooking (linear-light filtering, alpha weighting, normal maps, odd
// sizes, size limits) and the KTX2 container.

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <random>

#include "kenshin/assets/Image.h"
#include "kenshin/assets/Ktx2.h"
#include "kenshin/assets/TextureCooker.h"
#include "kenshin/util/Crc32.h"
#include "kenshin/util/Half.h"

using kenshin::f32;
using kenshin::u16;
using kenshin::u32;
using kenshin::u8;
using kenshin::usize;
using kenshin::backend::TextureFormat;
using namespace kenshin::assets;

namespace {

// ── A minimal PNG encoder (stored deflate blocks) ──────────────
// Lets the decoder be tested on real PNG bytes - including 16-bit and
// grey+alpha variants - without binary fixtures.
void be32(std::vector<u8>& b, u32 v) {
  for (int s = 24; s >= 0; s -= 8) b.push_back(static_cast<u8>(v >> s));
}
void chunk(std::vector<u8>& png, const char* type, const std::vector<u8>& data) {
  be32(png, static_cast<u32>(data.size()));
  std::vector<u8> body(type, type + 4);
  body.insert(body.end(), data.begin(), data.end());
  png.insert(png.end(), body.begin(), body.end());
  be32(png, kenshin::util::crc32Update(0, body.data(), body.size()));
}
/// colorType 0 grey, 2 RGB, 4 grey+alpha, 6 RGBA; `samples` holds the
/// channel values (0..255 or 0..65535), row-major.
std::vector<u8> makePng(u32 w, u32 h, int bitDepth, int colorType,
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

Image rgba8(u32 w, u32 h, std::initializer_list<u8> px) {
  return makeImageRgba8(w, h, std::vector<u8>(px));
}

const u8* mip(const CookedTexture& t, usize level) { return t.data.data() + t.mips[level].offset; }

TextureSettings linear() {
  TextureSettings s;
  s.colorSpace = TextureSettings::ColorSpace::Linear;
  return s;
}

}  // namespace

// ── Numeric helpers ────────────────────────────────────────────

TEST_CASE("texture: half conversion round-trips every half", "[texture]") {
  for (u32 h = 0; h < 65536; ++h) {
    const f32 f = kenshin::util::halfToF32(static_cast<u16>(h));
    if (std::isnan(f)) continue;
    INFO(h);
    REQUIRE(kenshin::util::f32ToHalf(f) == h);
  }
  CHECK(kenshin::util::halfToF32(0x3C00) == 1.0f);
  CHECK(kenshin::util::halfToF32(0x0001) == std::ldexp(1.0f, -24));
}

TEST_CASE("texture: sRGB transfer round-trips all 8-bit codes", "[texture]") {
  for (int c = 0; c < 256; ++c) {
    const f32 l = srgbToLinear(c / 255.0f);
    REQUIRE(std::lround(linearToSrgb(l) * 255.0f) == c);
  }
  CHECK(std::abs(srgbToLinear(0.5f) - 0.21404f) < 1e-4f);
}

// ── Decoding ───────────────────────────────────────────────────

TEST_CASE("texture: PNG decoding keeps alpha and precision", "[texture]") {
  Image img;
  std::string err;
  const auto rgba = makePng(2, 1, 8, 6, {10, 20, 30, 40, 50, 60, 70, 80});
  REQUIRE(decodeImage(rgba, img, err));
  CHECK(img.width == 2);
  CHECK(img.height == 1);
  CHECK(img.depth == Image::Depth::U8);
  CHECK(img.hasAlpha);
  CHECK(img.pixels == std::vector<u8>{10, 20, 30, 40, 50, 60, 70, 80});

  const auto rgb = makePng(1, 2, 8, 2, {1, 2, 3, 4, 5, 6});
  REQUIRE(decodeImage(rgb, img, err));
  CHECK_FALSE(img.hasAlpha);
  CHECK(img.pixels == std::vector<u8>{1, 2, 3, 255, 4, 5, 6, 255});

  const auto greyAlpha = makePng(1, 1, 8, 4, {7, 9});
  REQUIRE(decodeImage(greyAlpha, img, err));
  CHECK(img.hasAlpha);
  CHECK(img.pixels == std::vector<u8>{7, 7, 7, 9});

  const auto deep = makePng(1, 1, 16, 6, {1000, 40000, 65535, 12345});
  REQUIRE(decodeImage(deep, img, err));
  CHECK(img.depth == Image::Depth::U16);
  u16 v[4];
  std::memcpy(v, img.pixels.data(), 8);
  CHECK(v[0] == 1000);
  CHECK(v[1] == 40000);
  CHECK(v[3] == 12345);

  std::vector<u8> large = makePng(300, 200, 8, 6, std::vector<u32>(300 * 200 * 4, 77));
  REQUIRE(decodeImage(large, img, err));  // multi-block stored deflate
  CHECK(img.width == 300);

  CHECK_FALSE(decodeImage(std::vector<u8>{1, 2, 3}, img, err));
  CHECK(err.find("not a decodable image") == 0);
  CHECK_FALSE(decodeImage({}, img, err));
}

TEST_CASE("texture: the committed BMP decodes through the same path", "[texture]") {
  Image img;
  std::string err;
  REQUIRE(loadImageFile(std::filesystem::path(KENSHIN_SOURCE_DIR) / "tests/assets/checker_64.bmp",
                        img, err));
  CHECK(img.width == 64);
  CHECK(img.height == 64);
  CHECK_FALSE(img.hasAlpha);
  CHECK_FALSE(loadImageFile("no/such/file.png", img, err));
  CHECK(err.find("cannot read") == 0);
}

// ── Cooking ────────────────────────────────────────────────────

TEST_CASE("texture: full mip chains for any size", "[texture][mips]") {
  const std::pair<u32, u32> sizes[] = {{1, 1}, {2, 2}, {5, 3}, {7, 1}, {1, 9},
                                       {64, 64}, {100, 37}, {513, 2}};
  for (const auto& [w, h] : sizes) {
    INFO(w << "x" << h);
    CookedTexture t;
    std::string err;
    REQUIRE(cookTexture(makeImageRgba8(w, h, std::vector<u8>(usize(w) * h * 4, 200)), {}, t, err));
    REQUIRE(t.mips.size() == kenshin::backend::fullMipCount(w, h));
    u32 ew = w, eh = h;
    usize offset = 0;
    for (const MipLevel& m : t.mips) {
      CHECK(m.width == ew);
      CHECK(m.height == eh);
      CHECK(m.offset == offset);
      CHECK(m.size == usize(ew) * eh * 4);
      offset += m.size;
      ew = std::max(1u, ew / 2);
      eh = std::max(1u, eh / 2);
    }
    CHECK(t.data.size() == offset);
    CHECK(t.mips.back().width == 1);
    CHECK(t.mips.back().height == 1);
    // A constant image stays exactly constant at every level.
    for (const u8 b : t.data) REQUIRE(b == 200);
  }
}

TEST_CASE("texture: mip 0 is the source, untouched", "[texture][mips]") {
  std::mt19937 rng(3);
  std::vector<u8> px(37 * 23 * 4);
  for (u8& b : px) b = static_cast<u8>(rng());
  const Image img = makeImageRgba8(37, 23, px);
  for (const bool srgb : {true, false}) {
    TextureSettings s;
    s.colorSpace = srgb ? TextureSettings::ColorSpace::Srgb : TextureSettings::ColorSpace::Linear;
    CookedTexture t;
    std::string err;
    REQUIRE(cookTexture(img, s, t, err));
    CHECK(t.format == (srgb ? TextureFormat::RGBA8Srgb : TextureFormat::RGBA8Unorm));
    CHECK(std::memcmp(mip(t, 0), px.data(), px.size()) == 0);
  }
}

TEST_CASE("texture: sRGB colour is averaged in linear light", "[texture][mips]") {
  // Opaque black + white. Averaging the codes would give 128; the
  // correct linear-light average 0.5 encodes to sRGB 188.
  const Image img = rgba8(2, 1, {0, 0, 0, 255, 255, 255, 255, 255});
  CookedTexture srgb, lin;
  std::string err;
  REQUIRE(cookTexture(img, {}, srgb, err));
  REQUIRE(cookTexture(img, linear(), lin, err));
  CHECK(mip(srgb, 1)[0] == 188);
  CHECK(mip(lin, 1)[0] == 128);  // linear data: plain average (127.5 rounds up)
  CHECK(mip(srgb, 1)[3] == 255);
}

TEST_CASE("texture: transparent texels do not bleed colour", "[texture][mips]") {
  // Opaque red next to fully transparent green.
  const Image img = rgba8(2, 1, {255, 0, 0, 255, 0, 255, 0, 0});
  CookedTexture t;
  std::string err;
  REQUIRE(cookTexture(img, {}, t, err));
  const u8* m1 = mip(t, 1);
  CHECK(m1[0] == 255);
  CHECK(m1[1] == 0);
  CHECK(m1[3] == 128);
  // All transparent: the plain average survives (no 0/0 black).
  const Image clear = rgba8(2, 1, {255, 0, 0, 0, 0, 0, 255, 0});
  REQUIRE(cookTexture(clear, linear(), t, err));
  CHECK(mip(t, 1)[0] == 128);
  CHECK(mip(t, 1)[2] == 128);
  CHECK(mip(t, 1)[3] == 0);
}

TEST_CASE("texture: odd sizes weight texels by covered area", "[texture][mips]") {
  // 3 -> 1: each texel covers a third. 5 -> 2: weights 1, 1, 0.5 / 2.5.
  CookedTexture t;
  std::string err;
  REQUIRE(cookTexture(rgba8(3, 1, {0, 0, 0, 255, 102, 0, 0, 255, 255, 0, 0, 255}), linear(), t, err));
  CHECK(mip(t, 1)[0] == 119);  // (0 + 102 + 255) / 3 = 119
  REQUIRE(cookTexture(rgba8(5, 1, {100, 0, 0, 255, 200, 0, 0, 255, 0, 0, 0, 255,
                                   50, 0, 0, 255, 250, 0, 0, 255}),
                      linear(), t, err));
  CHECK(mip(t, 1)[0] == 120);  // (100 + 200 + 0.5 * 0) / 2.5
  CHECK(mip(t, 1)[4] == 120);  // (0.5 * 0 + 50 + 250) / 2.5
}

TEST_CASE("texture: normal maps stay unit length", "[texture][mips]") {
  // +X and +Z unit normals, encoded n * 0.5 + 0.5.
  TextureSettings s;
  s.normalMap = true;
  CookedTexture t;
  std::string err;
  REQUIRE(cookTexture(rgba8(2, 1, {255, 128, 128, 255, 128, 128, 255, 255}), s, t, err));
  CHECK(t.format == TextureFormat::RGBA8Unorm);
  const u8* m = mip(t, 1);
  const f32 n[3] = {m[0] / 255.0f * 2 - 1, m[1] / 255.0f * 2 - 1, m[2] / 255.0f * 2 - 1};
  CHECK(std::abs(std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) - 1.0f) < 0.01f);
  CHECK(std::abs(n[0] - n[2]) < 0.01f);  // the 45 degree bisector
  // A naive average would have length 0.71.
}

TEST_CASE("texture: size limit and no-mip settings", "[texture][mips]") {
  CookedTexture t;
  std::string err;
  TextureSettings s;
  s.maxSize = 256;
  REQUIRE(cookTexture(makeImageRgba8(1024, 16, std::vector<u8>(1024 * 16 * 4, 9)), s, t, err));
  CHECK(t.width == 256);
  CHECK(t.height == 4);
  CHECK(t.mips.size() == 9);
  s.mipmaps = false;
  REQUIRE(cookTexture(makeImageRgba8(1024, 16, std::vector<u8>(1024 * 16 * 4, 9)), s, t, err));
  CHECK(t.mips.size() == 1);
  CHECK(t.width == 256);
  CHECK_FALSE(cookTexture(Image{}, {}, t, err));
}

TEST_CASE("texture: 16-bit and HDR sources keep their range", "[texture][mips]") {
  Image deep;
  deep.width = deep.height = 1;
  deep.depth = Image::Depth::U16;
  const u16 px[4] = {1000, 65535, 0, 65535};
  deep.pixels.resize(8);
  std::memcpy(deep.pixels.data(), px, 8);
  CookedTexture t;
  std::string err;
  REQUIRE(cookTexture(deep, linear(), t, err));
  CHECK(t.format == TextureFormat::RGBA16Unorm);
  CHECK(std::memcmp(mip(t, 0), px, 8) == 0);
  REQUIRE(cookTexture(deep, {}, t, err));  // sRGB colour: linear half
  CHECK(t.format == TextureFormat::RGBA16Float);
  u16 h;
  std::memcpy(&h, mip(t, 0) + 2, 2);
  CHECK(kenshin::util::halfToF32(h) == 1.0f);

  Image hdr;
  hdr.width = hdr.height = 1;
  hdr.depth = Image::Depth::F32;
  const f32 radiance[4] = {12.5f, 0.25f, 3000.0f, 1.0f};
  hdr.pixels.resize(16);
  std::memcpy(hdr.pixels.data(), radiance, 16);
  REQUIRE(cookTexture(hdr, {}, t, err));
  CHECK(t.format == TextureFormat::RGBA16Float);
  std::memcpy(&h, mip(t, 0), 2);
  CHECK(kenshin::util::halfToF32(h) == 12.5f);  // > 1: not clamped
}

// ── KTX2 ───────────────────────────────────────────────────────

TEST_CASE("ktx2: round trip in every format", "[texture][ktx2]") {
  std::mt19937 rng(11);
  for (const TextureFormat f : {TextureFormat::RGBA8Unorm, TextureFormat::RGBA8Srgb,
                                TextureFormat::RGBA16Unorm, TextureFormat::RGBA16Float}) {
    CookedTexture t;
    t.format = f;
    t.width = 13;
    t.height = 6;
    u32 w = 13, h = 6;
    for (u32 i = 0; i < kenshin::backend::fullMipCount(13, 6); ++i) {
      const usize n = usize(w) * h * kenshin::backend::texelBytes(f);
      t.mips.push_back({w, h, t.data.size(), n});
      for (usize k = 0; k < n; ++k) t.data.push_back(static_cast<u8>(rng()));
      w = std::max(1u, w / 2);
      h = std::max(1u, h / 2);
    }
    const std::vector<u8> file = writeKtx2(t);
    CookedTexture back;
    std::string err;
    INFO(static_cast<int>(f) << " " << err);
    REQUIRE(readKtx2(file, back, err));
    CHECK(back.format == f);
    CHECK(back.width == 13);
    CHECK(back.height == 6);
    CHECK(back.data == t.data);
    REQUIRE(back.mips.size() == t.mips.size());
    // Spec layout: vkFormat, smallest level stored first, aligned.
    u32 vk;
    std::memcpy(&vk, file.data() + 12, 4);
    CHECK(vk == kenshin::backend::vkFormatOf(f));
    kenshin::u64 off0, offLast;
    std::memcpy(&off0, file.data() + 80, 8);
    std::memcpy(&offLast, file.data() + 80 + 24 * (t.mips.size() - 1), 8);
    CHECK(offLast < off0);
    CHECK(off0 % kenshin::backend::texelBytes(f) == 0);
    CHECK(writeKtx2(back) == file);  // deterministic
  }
}

TEST_CASE("ktx2: malformed files are rejected", "[texture][ktx2]") {
  CookedTexture t;
  std::string err;
  REQUIRE(cookTexture(makeImageRgba8(8, 4, std::vector<u8>(8 * 4 * 4, 50)), {}, t, err));
  const std::vector<u8> good = writeKtx2(t);
  CookedTexture out;
  REQUIRE(readKtx2(good, out, err));

  auto patched = [&](usize at, u32 value) {
    std::vector<u8> b = good;
    std::memcpy(b.data() + at, &value, 4);
    return b;
  };
  const std::pair<std::vector<u8>, std::string> cases[] = {
      {std::vector<u8>(good.begin(), good.begin() + 40), "not a KTX 2.0 file"},
      {patched(0, 0), "not a KTX 2.0 file"},
      {patched(12, 999), "unsupported vkFormat 999"},
      {patched(16, 2), "typeSize"},
      {patched(44, 1), "supercompressed"},
      {patched(28, 4), "not a 2D texture"},
      {patched(36, 6), "cube maps"},
      {patched(40, 9), "too many mip levels"},
      {patched(80 + 8, 3), "level 0 has the wrong size"},
      {patched(80, 0xFFFFFF00u), "level 0 lies outside the file"},
      {patched(12, 37), "DFD transfer function contradicts vkFormat"},  // SRGB DFD, UNORM format
      {std::vector<u8>(good.begin(), good.end() - 4), "outside the file"},
  };
  for (const auto& [bytes, expected] : cases) {
    INFO(expected);
    CookedTexture keep;
    keep.width = 77;
    CHECK_FALSE(readKtx2(bytes, keep, err));
    INFO(err);
    CHECK(err.find(expected) != std::string::npos);
    CHECK(keep.width == 77);
  }
}
