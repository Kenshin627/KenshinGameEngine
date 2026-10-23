#include "kenshin/assets/TextureCooker.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>

#include "kenshin/util/Half.h"

KENSHIN_NS_BEGIN
namespace assets {

using backend::TextureFormat;

f32 srgbToLinear(f32 c) {
  return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

f32 linearToSrgb(f32 l) {
  return l <= 0.0031308f ? l * 12.92f : 1.055f * std::pow(l, 1.0f / 2.4f) - 0.055f;
}

TextureFormat cookedFormat(const Image& image, const TextureSettings& settings) {
  const bool colour = !settings.normalMap && settings.colorSpace == TextureSettings::ColorSpace::Srgb;
  switch (image.depth) {
    case Image::Depth::U8: return colour ? TextureFormat::RGBA8Srgb : TextureFormat::RGBA8Unorm;
    case Image::Depth::U16:
      // No 16-bit sRGB format exists: colour is decoded to linear half.
      return colour ? TextureFormat::RGBA16Float : TextureFormat::RGBA16Unorm;
    case Image::Depth::F32: return TextureFormat::RGBA16Float;
  }
  return TextureFormat::RGBA8Unorm;
}

namespace {

/// Float working image with C channels per texel.
struct Plane {
  u32 w = 0, h = 0, c = 0;
  std::vector<f32> v;
  f32* at(u32 x, u32 y) { return v.data() + (usize(y) * w + x) * c; }
  const f32* at(u32 x, u32 y) const { return v.data() + (usize(y) * w + x) * c; }
};

/// Per destination texel: the source texels it overlaps and the
/// overlap weights (summing to 1). Exact area coverage for any ratio.
struct Tap {
  u32 first;
  std::vector<f32> weights;
};

std::vector<Tap> footprints(u32 src, u32 dst) {
  std::vector<Tap> taps(dst);
  const double scale = double(src) / double(dst);
  for (u32 i = 0; i < dst; ++i) {
    const double lo = i * scale, hi = (i + 1) * scale;
    const u32 first = static_cast<u32>(std::floor(lo));
    const u32 last = std::min(src - 1, static_cast<u32>(std::ceil(hi)) - 1);
    taps[i].first = first;
    for (u32 s = first; s <= last; ++s) {
      const double overlap = std::min<double>(hi, s + 1.0) - std::max<double>(lo, s);
      taps[i].weights.push_back(static_cast<f32>(overlap / scale));
    }
  }
  return taps;
}

Plane downsample(const Plane& src) {
  const u32 w2 = std::max(1u, src.w / 2), h2 = std::max(1u, src.h / 2);
  const std::vector<Tap> tx = footprints(src.w, w2), ty = footprints(src.h, h2);
  Plane mid{w2, src.h, src.c, std::vector<f32>(usize(w2) * src.h * src.c, 0.0f)};
  for (u32 y = 0; y < src.h; ++y)
    for (u32 x = 0; x < w2; ++x) {
      f32* d = mid.at(x, y);
      for (usize k = 0; k < tx[x].weights.size(); ++k) {
        const f32* s = src.at(tx[x].first + static_cast<u32>(k), y);
        for (u32 ch = 0; ch < src.c; ++ch) d[ch] += s[ch] * tx[x].weights[k];
      }
    }
  Plane out{w2, h2, src.c, std::vector<f32>(usize(w2) * h2 * src.c, 0.0f)};
  for (u32 y = 0; y < h2; ++y)
    for (usize k = 0; k < ty[y].weights.size(); ++k) {
      const u32 sy = ty[y].first + static_cast<u32>(k);
      const f32 wgt = ty[y].weights[k];
      for (u32 x = 0; x < w2; ++x) {
        f32* d = out.at(x, y);
        const f32* s = mid.at(x, sy);
        for (u32 ch = 0; ch < src.c; ++ch) d[ch] += s[ch] * wgt;
      }
    }
  return out;
}

f32 component(const Image& img, usize i) {
  switch (img.depth) {
    case Image::Depth::U8: return img.pixels[i] / 255.0f;
    case Image::Depth::U16: {
      u16 v;
      std::memcpy(&v, img.pixels.data() + i * 2, 2);
      return v / 65535.0f;
    }
    case Image::Depth::F32: {
      f32 v;
      std::memcpy(&v, img.pixels.data() + i * 4, 4);
      return v;
    }
  }
  return 0.0f;
}

/// How the working plane is laid out and decoded back to RGBA.
struct Layout {
  bool normal = false;     // xyz in [-1, 1], renormalized per mip
  bool srgb = false;       // colour stored linear, encoded on output
  bool weighted = false;   // + alpha-weighted copy: [r*a g*a b*a a r g b]
};

Plane toPlane(const Image& img, const Layout& L) {
  Plane p{img.width, img.height, L.weighted ? 7u : 4u, {}};
  p.v.resize(usize(p.w) * p.h * p.c);
  const bool hdr = img.depth == Image::Depth::F32;
  for (usize t = 0, n = usize(p.w) * p.h; t < n; ++t) {
    f32 rgba[4];
    for (int ch = 0; ch < 4; ++ch) rgba[ch] = component(img, t * 4 + ch);
    for (int ch = 0; ch < 3; ++ch) {
      if (L.normal) rgba[ch] = rgba[ch] * 2.0f - 1.0f;
      else if (L.srgb && !hdr) rgba[ch] = srgbToLinear(rgba[ch]);
    }
    f32* d = p.v.data() + t * p.c;
    if (L.weighted) {
      for (int ch = 0; ch < 3; ++ch) d[ch] = rgba[ch] * rgba[3];
      d[3] = rgba[3];
      for (int ch = 0; ch < 3; ++ch) d[4 + ch] = rgba[ch];
    } else {
      std::memcpy(d, rgba, sizeof(rgba));
    }
  }
  return p;
}

/// Working texel -> straight RGBA in the format's value space
/// (linear for sRGB formats - encoded later).
std::array<f32, 4> texel(const Plane& p, u32 x, u32 y, const Layout& L) {
  const f32* s = p.at(x, y);
  std::array<f32, 4> o{s[0], s[1], s[2], s[3]};
  if (L.weighted) {
    // Fully transparent footprint: no weighted colour exists; keep the
    // plain average so the texel is still sensible if revealed.
    if (s[3] > 1e-6f)
      for (int ch = 0; ch < 3; ++ch) o[ch] = s[ch] / s[3];
    else
      for (int ch = 0; ch < 3; ++ch) o[ch] = s[4 + ch];
  }
  if (L.normal) {
    const f32 len = std::sqrt(o[0] * o[0] + o[1] * o[1] + o[2] * o[2]);
    if (len > 1e-8f)
      for (int ch = 0; ch < 3; ++ch) o[ch] /= len;
    else
      o = {0.0f, 0.0f, 1.0f, o[3]};
    for (int ch = 0; ch < 3; ++ch) o[ch] = o[ch] * 0.5f + 0.5f;
  }
  return o;
}

void encode(const Plane& p, const Layout& L, TextureFormat f, std::vector<u8>& out) {
  const usize start = out.size();
  out.resize(start + usize(p.w) * p.h * backend::texelBytes(f));
  u8* d = out.data() + start;
  auto unorm = [](f32 v, f32 scale) {
    return static_cast<u32>(std::lround(std::clamp(v, 0.0f, 1.0f) * scale));
  };
  for (u32 y = 0; y < p.h; ++y)
    for (u32 x = 0; x < p.w; ++x) {
      std::array<f32, 4> t = texel(p, x, y, L);
      switch (f) {
        case TextureFormat::RGBA8Srgb:
          for (int ch = 0; ch < 3; ++ch) t[ch] = linearToSrgb(std::clamp(t[ch], 0.0f, 1.0f));
          [[fallthrough]];
        case TextureFormat::RGBA8Unorm:
          for (int ch = 0; ch < 4; ++ch) *d++ = static_cast<u8>(unorm(t[ch], 255.0f));
          break;
        case TextureFormat::RGBA16Unorm:
          for (int ch = 0; ch < 4; ++ch) {
            const u16 v = static_cast<u16>(unorm(t[ch], 65535.0f));
            std::memcpy(d, &v, 2);
            d += 2;
          }
          break;
        case TextureFormat::RGBA16Float:
          for (int ch = 0; ch < 4; ++ch) {
            const u16 v = util::f32ToHalf(t[ch]);
            std::memcpy(d, &v, 2);
            d += 2;
          }
          break;
      }
    }
}

}  // namespace

bool cookTexture(const Image& image, const TextureSettings& settings, CookedTexture& out,
                 std::string& error) {
  if (image.width == 0 || image.height == 0 || image.pixels.size() != image.byteSize()) {
    error = "empty or inconsistent image";
    return false;
  }
  const TextureFormat format = cookedFormat(image, settings);
  Layout L;
  L.normal = settings.normalMap;
  L.srgb = !settings.normalMap && settings.colorSpace == TextureSettings::ColorSpace::Srgb;
  L.weighted = image.hasAlpha && !settings.normalMap;

  CookedTexture cooked;
  cooked.format = format;
  // Mip 0 of an unchanged-precision source is copied verbatim: exact
  // by construction, never a decode/encode round trip.
  const bool verbatim = (image.depth == Image::Depth::U8 &&
                         (format == TextureFormat::RGBA8Srgb || format == TextureFormat::RGBA8Unorm)) ||
                        (image.depth == Image::Depth::U16 && format == TextureFormat::RGBA16Unorm);

  Plane plane = toPlane(image, L);
  bool first = true;  // still the full-resolution level
  const u32 limit = settings.maxSize == 0 ? 0xFFFFFFFFu : settings.maxSize;
  for (;;) {
    const bool fits = std::max(plane.w, plane.h) <= limit;
    if (fits) {
      MipLevel level{plane.w, plane.h, cooked.data.size(), 0};
      if (first && verbatim)
        cooked.data.insert(cooked.data.end(), image.pixels.begin(), image.pixels.end());
      else
        encode(plane, L, format, cooked.data);
      level.size = cooked.data.size() - level.offset;
      cooked.mips.push_back(level);
      if (!settings.mipmaps) break;
    }
    if (plane.w == 1 && plane.h == 1) break;
    plane = downsample(plane);
    first = false;
  }
  cooked.width = cooked.mips.front().width;
  cooked.height = cooked.mips.front().height;
  out = std::move(cooked);
  return true;
}

}  // namespace assets
KENSHIN_NS_END
