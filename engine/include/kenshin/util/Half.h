#pragma once

// ── Half — IEEE 754 binary16 conversion ─────────────────────────────

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace util {

/// f32 -> IEEE half bits (round-to-nearest-even, denormals kept,
/// overflow to inf, NaN preserved). An earlier inline version computed
/// the biased exponent as an UNSIGNED ex - 112: every value below 2^-14
/// wrapped to a huge exponent and became +INF - one dark texel in an
/// env.hdr poisoned its whole mip pyramid (inf/NaN reflections).
u16 f32ToHalf(float v);
/// IEEE half bits -> f32 (exact: every half is representable).
float halfToF32(u16 h);

}  // namespace util

KENSHIN_NS_END
