#pragma once

// ── Crc32 — IEEE 802.3 CRC (zlib-compatible) ────────────────────
// Verification currency of the GPU Scene paths: data staged on the CPU
// is checksummed, the GPU copy is read back and checksummed again - a
// match proves the upload reached device memory intact.

#include <cstddef>
#include <cstdint>
#include <span>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace util {

/// Incremental CRC32 (initialize `crc` with 0 or a previous return value).
inline u32 crc32Update(u32 crc, const void* data, usize bytes) {
  static constexpr u32 kPolynomial = 0xEDB88320u;  // reflected
  crc = ~crc;
  const u8* p = static_cast<const u8*>(data);
  for (usize i = 0; i < bytes; ++i) {
    crc ^= p[i];
    for (int bit = 0; bit < 8; ++bit)
      crc = (crc >> 1) ^ (kPolynomial & (0u - (crc & 1u)));
  }
  return ~crc;
}

inline u32 crc32(std::span<const u8> data) {
  return crc32Update(0, data.data(), data.size());
}

}  // namespace util

KENSHIN_NS_END
