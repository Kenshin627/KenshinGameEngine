#include "kenshin/assets/Ktx2.h"

#include <bit>
#include <cstring>
#include <format>
#include <numeric>

#include "kenshin/util/File.h"

KENSHIN_NS_BEGIN
namespace assets {

static_assert(std::endian::native == std::endian::little,
              "KTX2 files are little-endian; add byte swapping for this target");

using backend::TextureFormat;

namespace {

constexpr u8 kIdentifier[12] = {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32,
                                0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
constexpr usize kHeaderBytes = 12 + 9 * 4;      // identifier + header
constexpr usize kIndexBytes = 4 * 4 + 2 * 8;    // dfd/kvd/sgd index
constexpr usize kLevelEntryBytes = 3 * 8;

struct FormatInfo {
  TextureFormat format;
  u32 typeSize;     // bytes per component
  u32 bits;         // bits per component
  bool srgb;
  bool floating;
};

constexpr FormatInfo kFormats[] = {
    {TextureFormat::RGBA8Unorm, 1, 8, false, false},
    {TextureFormat::RGBA8Srgb, 1, 8, true, false},
    {TextureFormat::RGBA16Unorm, 2, 16, false, false},
    {TextureFormat::RGBA16Float, 2, 16, false, true},
};

const FormatInfo* infoOf(TextureFormat f) {
  for (const FormatInfo& i : kFormats)
    if (i.format == f) return &i;
  return nullptr;
}

const FormatInfo* infoOfVk(u32 vk) {
  for (const FormatInfo& i : kFormats)
    if (backend::vkFormatOf(i.format) == vk) return &i;
  return nullptr;
}

void put32(std::vector<u8>& b, u32 v) {
  const usize at = b.size();
  b.resize(at + 4);
  std::memcpy(b.data() + at, &v, 4);
}
void put64(std::vector<u8>& b, u64 v) {
  const usize at = b.size();
  b.resize(at + 8);
  std::memcpy(b.data() + at, &v, 8);
}
void set32(std::vector<u8>& b, usize at, u32 v) { std::memcpy(b.data() + at, &v, 4); }
void set64(std::vector<u8>& b, usize at, u64 v) { std::memcpy(b.data() + at, &v, 8); }
void padTo(std::vector<u8>& b, usize align) {
  while (b.size() % align != 0) b.push_back(0);
}
u32 get32(std::span<const u8> b, usize at) {
  u32 v;
  std::memcpy(&v, b.data() + at, 4);
  return v;
}
u64 get64(std::span<const u8> b, usize at) {
  u64 v;
  std::memcpy(&v, b.data() + at, 8);
  return v;
}

/// Khronos Basic Data Format Descriptor (KDF 1.3 section 5) for an
/// RGBA format: 4 samples R, G, B, A.
std::vector<u8> basicDfd(const FormatInfo& f) {
  std::vector<u8> d;
  constexpr u32 kSamples = 4;
  const u32 blockSize = 24 + 16 * kSamples;
  put32(d, 4 + blockSize);                 // dfdTotalSize
  put32(d, 0);                             // vendorId 0 (Khronos), type 0 (basic)
  put32(d, 2u | (blockSize << 16));        // version 1.3, block size
  const u32 transfer = f.srgb ? 2u : 1u;   // KHR_DF_TRANSFER_SRGB : LINEAR
  put32(d, 1u | (1u << 8) | (transfer << 16));  // RGBSDA, BT709, straight alpha
  put32(d, 0);                             // texel block 1x1x1x1
  put32(d, f.typeSize * 4);                // bytesPlane0
  put32(d, 0);                             // bytesPlane4..7
  const u32 channels[kSamples] = {0, 1, 2, 15};  // R G B A
  for (u32 s = 0; s < kSamples; ++s) {
    u32 qualifiers = 0;
    if (f.floating) qualifiers |= 0x80 | 0x40;  // FLOAT | SIGNED
    if (f.srgb && channels[s] == 15) qualifiers |= 0x10;  // alpha stays linear
    put32(d, (s * f.bits) | ((f.bits - 1) << 16) | ((channels[s] | qualifiers) << 24));
    put32(d, 0);  // sample position
    if (f.floating) {
      put32(d, 0xBF800000u);  // -1.0f
      put32(d, 0x3F800000u);  // 1.0f
    } else {
      put32(d, 0);
      put32(d, f.bits == 8 ? 255u : 65535u);
    }
  }
  return d;
}

std::vector<u8> keyValueData() {
  // One entry, KTXwriter (the spec recommends it). Deliberately free of
  // build/git versions: cooked files must not change between commits
  // unless the cooker itself does (bump kCookerVersion then).
  constexpr int kCookerVersion = 1;
  const std::string kv = std::string("KTXwriter") + '\0' + "KENSHIN texture cooker " +
                         std::to_string(kCookerVersion) + '\0';
  std::vector<u8> d;
  put32(d, static_cast<u32>(kv.size()));
  d.insert(d.end(), kv.begin(), kv.end());
  padTo(d, 4);
  return d;
}

}  // namespace

std::vector<u8> writeKtx2(const CookedTexture& t) {
  const FormatInfo& f = *infoOf(t.format);
  const u32 levels = static_cast<u32>(t.mips.size());
  std::vector<u8> b(kIdentifier, kIdentifier + 12);
  put32(b, backend::vkFormatOf(t.format));
  put32(b, f.typeSize);
  put32(b, t.width);
  put32(b, t.height);
  put32(b, 0);  // pixelDepth: 2D
  put32(b, 0);  // layerCount: not an array
  put32(b, 1);  // faceCount
  put32(b, levels);
  put32(b, 0);  // supercompressionScheme: none
  const usize indexAt = b.size();
  b.resize(b.size() + kIndexBytes, 0);
  const usize levelIndexAt = b.size();
  b.resize(b.size() + levels * kLevelEntryBytes, 0);

  const std::vector<u8> dfd = basicDfd(f);
  const usize dfdAt = b.size();
  b.insert(b.end(), dfd.begin(), dfd.end());
  const std::vector<u8> kvd = keyValueData();
  const usize kvdAt = b.size();
  b.insert(b.end(), kvd.begin(), kvd.end());
  set32(b, indexAt + 0, static_cast<u32>(dfdAt));
  set32(b, indexAt + 4, static_cast<u32>(dfd.size()));
  set32(b, indexAt + 8, static_cast<u32>(kvdAt));
  set32(b, indexAt + 12, static_cast<u32>(kvd.size()));
  // sgd offset/length stay 0 (no supercompression global data).

  // Mip data: smallest level first; each aligned to lcm(texel, 4).
  const usize align = std::lcm(usize(backend::texelBytes(t.format)), usize(4));
  for (u32 i = levels; i-- > 0;) {
    padTo(b, align);
    const MipLevel& m = t.mips[i];
    const usize at = b.size();
    b.insert(b.end(), t.data.begin() + static_cast<std::ptrdiff_t>(m.offset),
             t.data.begin() + static_cast<std::ptrdiff_t>(m.offset + m.size));
    const usize e = levelIndexAt + i * kLevelEntryBytes;
    set64(b, e, at);
    set64(b, e + 8, m.size);
    set64(b, e + 16, m.size);
  }
  return b;
}

bool readKtx2(std::span<const u8> b, CookedTexture& out, std::string& error) {
  auto fail = [&](std::string why) {
    error = "ktx2: " + std::move(why);
    return false;
  };
  if (b.size() < kHeaderBytes + kIndexBytes || std::memcmp(b.data(), kIdentifier, 12) != 0)
    return fail("not a KTX 2.0 file");
  const u32 vkFormat = get32(b, 12), typeSize = get32(b, 16), width = get32(b, 20),
            height = get32(b, 24), depth = get32(b, 28), layers = get32(b, 32),
            faces = get32(b, 36), levelField = get32(b, 40), scheme = get32(b, 44);
  const FormatInfo* f = infoOfVk(vkFormat);
  if (!f) return fail(std::format("unsupported vkFormat {}", vkFormat));
  if (typeSize != f->typeSize) return fail("typeSize does not match the format");
  if (scheme != 0) return fail("supercompressed files are not supported (yet)");
  if (width == 0 || height == 0 || depth != 0) return fail("not a 2D texture");
  if (layers > 1 || faces != 1) return fail("arrays and cube maps are not supported");
  if (width > 16384 || height > 16384) return fail("texture larger than 16384");
  const u32 levels = levelField == 0 ? 1 : levelField;
  if (levels > backend::fullMipCount(width, height)) return fail("too many mip levels");
  if (b.size() < kHeaderBytes + kIndexBytes + levels * kLevelEntryBytes)
    return fail("truncated level index");

  const u32 dfdAt = get32(b, 48), dfdLen = get32(b, 52);
  if (dfdLen < 4 + 24 || u64(dfdAt) + dfdLen > b.size()) return fail("bad data format descriptor");
  // The DFD must agree with vkFormat on the transfer function (a file
  // claiming SRGB data in a UNORM format would shade wrong silently).
  const u32 transfer = (get32(b, dfdAt + 12) >> 16) & 0xFF;
  if (transfer != (f->srgb ? 2u : 1u)) return fail("DFD transfer function contradicts vkFormat");

  CookedTexture t;
  t.format = f->format;
  t.width = width;
  t.height = height;
  const usize texel = backend::texelBytes(f->format);
  const usize align = std::lcm(texel, usize(4));
  for (u32 i = 0; i < levels; ++i) {
    const usize e = kHeaderBytes + kIndexBytes + i * kLevelEntryBytes;
    const u64 at = get64(b, e), len = get64(b, e + 8), raw = get64(b, e + 16);
    const u32 w = std::max(1u, width >> i), h = std::max(1u, height >> i);
    const u64 expect = u64(w) * h * texel;
    if (len != expect || raw != expect) return fail(std::format("level {} has the wrong size", i));
    if (at % align != 0 || at > b.size() || len > b.size() - at)
      return fail(std::format("level {} lies outside the file", i));
    t.mips.push_back({w, h, t.data.size(), static_cast<usize>(len)});
    t.data.insert(t.data.end(), b.begin() + static_cast<std::ptrdiff_t>(at),
                  b.begin() + static_cast<std::ptrdiff_t>(at + len));
  }
  out = std::move(t);
  return true;
}

bool saveKtx2File(const std::filesystem::path& path, const CookedTexture& texture,
                  std::string* error) {
  const std::vector<u8> bytes = writeKtx2(texture);
  return util::writeFileAtomic(
      path, std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()), error);
}

bool loadKtx2File(const std::filesystem::path& path, CookedTexture& out, std::string& error) {
  std::string bytes;
  if (!util::readFile(path, bytes)) {
    error = "cannot read " + util::pathToUtf8(path);
    return false;
  }
  return readKtx2({reinterpret_cast<const u8*>(bytes.data()), bytes.size()}, out, error);
}

}  // namespace assets
KENSHIN_NS_END
