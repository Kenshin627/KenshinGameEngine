#include "kenshin/util/Half.h"

#include <cstring>

KENSHIN_NS_BEGIN

namespace util {

u16 f32ToHalf(float v) {
  u32 b;
  std::memcpy(&b, &v, sizeof(b));
  const u32 sign = (b >> 16) & 0x8000u;
  const u32 absBits = b & 0x7FFFFFFFu;
  if (absBits >= 0x7F800000u)  // inf / NaN
    return u16(sign | 0x7C00u | (absBits > 0x7F800000u ? 0x200u : 0u));
  if (absBits >= 0x477FF000u)  // rounds to >= 65520: overflow
    return u16(sign | 0x7C00u);
  if (absBits < 0x33000000u)  // < 2^-25: rounds to zero
    return u16(sign);
  const int ex = int(absBits >> 23) - 127;
  u32 man = (absBits & 0x7FFFFFu) | 0x800000u;  // implicit 1
  int shift;
  u32 he;
  if (ex < -14) {  // denormal half
    shift = 13 + (-14 - ex);
    he = 0;
  } else {
    shift = 13;
    he = u32(ex + 15);
    man &= 0x7FFFFFu;
  }
  u32 hm = man >> shift;
  const u32 rem = man & ((1u << shift) - 1u);
  const u32 half = 1u << (shift - 1);
  if (rem > half || (rem == half && (hm & 1u))) ++hm;  // RNE
  // A mantissa carry rolls into the exponent naturally.
  return u16(sign | ((he << 10) + hm));
}

float halfToF32(u16 h) {
  const u32 sign = u32(h & 0x8000u) << 16;
  const u32 ex = (h >> 10) & 0x1Fu;
  u32 man = h & 0x3FFu;
  u32 bits;
  if (ex == 0x1Fu) {
    bits = sign | 0x7F800000u | (man << 13);  // inf / NaN
  } else if (ex != 0) {
    bits = sign | ((ex + 112u) << 23) | (man << 13);
  } else if (man == 0) {
    bits = sign;  // +-0
  } else {  // denormal: normalize
    int e = -1;
    do {
      ++e;
      man <<= 1;
    } while ((man & 0x400u) == 0);
    bits = sign | (u32(112 - e) << 23) | ((man & 0x3FFu) << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

}  // namespace util

KENSHIN_NS_END
