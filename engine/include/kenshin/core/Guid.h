#pragma once

// ── Guid: 128-bit stable identity ───────────────────────────────
// Entities, assets and prefabs are referenced by Guid in scene files
// (roadmap 1.1/1.2): unlike entt handles, a Guid survives save/load and
// copy/paste between projects. Random RFC 4122 version-4 values; the
// text form is the canonical lower-case 8-4-4-4-12 hex string.

#include <compare>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN
namespace core {

struct Guid {
  u64 hi = 0;  // bytes 0..7 of the canonical form, big-endian
  u64 lo = 0;  // bytes 8..15

  /// A fresh random version-4 Guid (thread-safe; never the null Guid).
  static Guid generate();
  /// Parse the canonical form "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
  /// (hex digits in either case). Anything else -> nullopt.
  static std::optional<Guid> parse(std::string_view text);

  bool isNull() const { return hi == 0 && lo == 0; }
  /// Canonical lower-case form (36 chars).
  std::string toString() const;

  friend auto operator<=>(const Guid&, const Guid&) = default;
};

}  // namespace core
KENSHIN_NS_END

template <>
struct std::hash<kenshin::core::Guid> {
  size_t operator()(const kenshin::core::Guid& g) const noexcept {
    // Random bits already; the multiply folds both halves on 32-bit too.
    return static_cast<size_t>(g.hi ^ (g.lo * 0x9E3779B97F4A7C15ull));
  }
};
