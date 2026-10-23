#include "kenshin/core/Guid.h"

#include <chrono>
#include <random>
#include <thread>

KENSHIN_NS_BEGIN
namespace core {

namespace {
std::mt19937_64& threadRng() {
  // Seeded per thread from the OS entropy source, mixed with the clock
  // and thread id so two threads never share a sequence even where
  // random_device is weak.
  thread_local std::mt19937_64 rng = [] {
    std::random_device rd;
    const u64 clock = static_cast<u64>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const u64 tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
    std::seed_seq seq{rd(), rd(), rd(), rd(), static_cast<u32>(clock),
                      static_cast<u32>(clock >> 32), static_cast<u32>(tid),
                      static_cast<u32>(tid >> 32)};
    return std::mt19937_64(seq);
  }();
  return rng;
}

int hexValue(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

bool isDashPosition(usize i) { return i == 8 || i == 13 || i == 18 || i == 23; }
}  // namespace

Guid Guid::generate() {
  std::mt19937_64& rng = threadRng();
  Guid g;
  g.hi = rng();
  g.lo = rng();
  // RFC 4122: version 4 in the high nibble of byte 6, variant 10xx in
  // the top bits of byte 8. Both are non-zero, so never the null Guid.
  g.hi = (g.hi & ~0x000000000000F000ull) | 0x0000000000004000ull;
  g.lo = (g.lo & ~0xC000000000000000ull) | 0x8000000000000000ull;
  return g;
}

std::optional<Guid> Guid::parse(std::string_view text) {
  if (text.size() != 36) return std::nullopt;
  Guid g;
  int nibbles = 0;
  for (usize i = 0; i < text.size(); ++i) {
    if (isDashPosition(i)) {
      if (text[i] != '-') return std::nullopt;
      continue;
    }
    const int v = hexValue(text[i]);
    if (v < 0) return std::nullopt;
    u64& half = nibbles < 16 ? g.hi : g.lo;
    half = (half << 4) | static_cast<u64>(v);
    ++nibbles;
  }
  return g;
}

std::string Guid::toString() const {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(36);
  for (int n = 0; n < 32; ++n) {
    if (n == 8 || n == 12 || n == 16 || n == 20) out.push_back('-');
    const u64 half = n < 16 ? hi : lo;
    const int shift = 60 - 4 * (n % 16);
    out.push_back(kHex[(half >> shift) & 0xF]);
  }
  return out;
}

}  // namespace core
KENSHIN_NS_END
