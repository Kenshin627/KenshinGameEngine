#pragma once

// ── SHA-1 (FIPS 180-4) ──────────────────────────────────────────
// Content hashing for the asset database (staleness keys) and the
// RFC 4122 version-5 name-based Guids. Not used for security.

#include <array>
#include <span>
#include <string>
#include <string_view>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN
namespace core {

class Sha1 {
 public:
  using Digest = std::array<u8, 20>;

  Sha1();
  void update(const void* data, usize bytes);
  void update(std::string_view text) { update(text.data(), text.size()); }
  Digest finish();  // the object is reset afterwards

  static Digest of(const void* data, usize bytes);
  static Digest of(std::string_view text) { return of(text.data(), text.size()); }
  static std::string hex(const Digest& d);

 private:
  void block(const u8* p);
  u32 m_h[5];
  u8 m_buffer[64];
  usize m_buffered = 0;
  u64 m_length = 0;
};

}  // namespace core
KENSHIN_NS_END
