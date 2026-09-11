#pragma once

// M6.4 wide-instance VisBuffer comparison: the payload is
// pairIndex 23b | listSel 2b | triangle 7b - PAIR-ENUMERATION-
// SENSITIVE (different cull paths order pairs differently). Raw u64
// equality only holds within one enumeration; cross-path/mode
// differentials compare RESOLVED attributes (depth + instance +
// cluster + triangle), which is exactly the decode-visible truth.

#include <glm/glm.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace viscmp {

struct Lists {
  std::vector<glm::uvec2> pairs, hw, post;
};

/// word -> {depth, instance, cluster, triangle}
inline void resolve(std::uint64_t w, const Lists& l,
                    std::uint32_t (&out)[4]) {
  const std::uint32_t lo = static_cast<std::uint32_t>(w & 0xFFFFFFFFull);
  out[0] = static_cast<std::uint32_t>(w >> 32);
  out[3] = lo & 0x7Fu;
  const std::uint32_t sel = (lo >> 7) & 3u;
  const std::uint32_t idx = (lo >> 9) & 0x7FFFFFu;
  const auto& list = sel == 1u ? l.hw : sel == 2u ? l.post : l.pairs;
  const glm::uvec2 pair = idx < list.size() ? list[idx] : glm::uvec2(0, 0);
  out[1] = pair.x;
  out[2] = pair.y;
}

inline std::size_t mismatches(const std::uint64_t* a, const Lists& la,
                              const std::uint64_t* b,
                              const Lists& lb, std::size_t n) {
  std::size_t m = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (a[i] == b[i]) continue;
    if (a[i] == 0 || b[i] == 0) {
      ++m;  // coverage difference (one side has no surface here)
      continue;
    }
    std::uint32_t ra[4], rb[4];
    resolve(a[i], la, ra);
    resolve(b[i], lb, rb);
    if (ra[0] != rb[0] || ra[1] != rb[1] || ra[2] != rb[2] || ra[3] != rb[3])
      ++m;
  }
  return m;
}

}  // namespace viscmp
