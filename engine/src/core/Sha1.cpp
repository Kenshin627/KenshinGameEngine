#include "kenshin/core/Sha1.h"

#include <bit>
#include <cstring>

KENSHIN_NS_BEGIN
namespace core {

Sha1::Sha1() {
  m_h[0] = 0x67452301u;
  m_h[1] = 0xEFCDAB89u;
  m_h[2] = 0x98BADCFEu;
  m_h[3] = 0x10325476u;
  m_h[4] = 0xC3D2E1F0u;
}

void Sha1::block(const u8* p) {
  u32 w[80];
  for (int i = 0; i < 16; ++i)
    w[i] = (u32(p[i * 4]) << 24) | (u32(p[i * 4 + 1]) << 16) | (u32(p[i * 4 + 2]) << 8) |
           u32(p[i * 4 + 3]);
  for (int i = 16; i < 80; ++i) w[i] = std::rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
  u32 a = m_h[0], b = m_h[1], c = m_h[2], d = m_h[3], e = m_h[4];
  for (int i = 0; i < 80; ++i) {
    u32 f, k;
    if (i < 20) {
      f = (b & c) | (~b & d);
      k = 0x5A827999u;
    } else if (i < 40) {
      f = b ^ c ^ d;
      k = 0x6ED9EBA1u;
    } else if (i < 60) {
      f = (b & c) | (b & d) | (c & d);
      k = 0x8F1BBCDCu;
    } else {
      f = b ^ c ^ d;
      k = 0xCA62C1D6u;
    }
    const u32 t = std::rotl(a, 5) + f + e + k + w[i];
    e = d;
    d = c;
    c = std::rotl(b, 30);
    b = a;
    a = t;
  }
  m_h[0] += a;
  m_h[1] += b;
  m_h[2] += c;
  m_h[3] += d;
  m_h[4] += e;
}

void Sha1::update(const void* data, usize bytes) {
  const u8* p = static_cast<const u8*>(data);
  m_length += bytes;
  if (m_buffered > 0) {
    const usize take = std::min(bytes, usize(64) - m_buffered);
    std::memcpy(m_buffer + m_buffered, p, take);
    m_buffered += take;
    p += take;
    bytes -= take;
    if (m_buffered == 64) {
      block(m_buffer);
      m_buffered = 0;
    }
  }
  while (bytes >= 64) {
    block(p);
    p += 64;
    bytes -= 64;
  }
  if (bytes > 0) {
    std::memcpy(m_buffer, p, bytes);
    m_buffered = bytes;
  }
}

Sha1::Digest Sha1::finish() {
  const u64 bits = m_length * 8;
  const u8 one = 0x80;
  update(&one, 1);
  const u8 zero = 0;
  while (m_buffered != 56) update(&zero, 1);
  u8 len[8];
  for (int i = 0; i < 8; ++i) len[i] = u8(bits >> (56 - 8 * i));
  update(len, 8);
  Digest d;
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 4; ++j) d[usize(i * 4 + j)] = u8(m_h[i] >> (24 - 8 * j));
  *this = Sha1();
  return d;
}

Sha1::Digest Sha1::of(const void* data, usize bytes) {
  Sha1 s;
  s.update(data, bytes);
  return s.finish();
}

std::string Sha1::hex(const Digest& d) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(40);
  for (const u8 b : d) {
    out.push_back(kHex[b >> 4]);
    out.push_back(kHex[b & 15]);
  }
  return out;
}

}  // namespace core
KENSHIN_NS_END
