// CRC32 known vectors (IEEE 802.3, zlib-compatible).

#include <catch2/catch_test_macros.hpp>

#include <kenshin/util/Crc32.h>

#include <cstring>
#include <vector>

namespace {
kenshin::u32 crcOfString(const char* s) {
  const auto* p = reinterpret_cast<const kenshin::u8*>(s);
  return kenshin::util::crc32({p, std::strlen(s)});
}
}  // namespace

TEST_CASE("crc32 known vectors", "[util]") {
  // The canonical check value of CRC-32/ISO-HDLC.
  REQUIRE(crcOfString("123456789") == 0xCBF43926u);
  REQUIRE(crcOfString("") == 0x00000000u);
  REQUIRE(crcOfString("a") == 0xE8B7BE43u);
  REQUIRE(crcOfString("abc") == 0x352441C2u);

  // Incremental form must equal the one-shot form.
  const char* parts[] = {"kenshin", "-virtual-", "geometry"};
  kenshin::u32 incremental = 0;
  for (const char* part : parts)
    incremental = kenshin::util::crc32Update(
        incremental, part, std::strlen(part));
  std::string joined;
  for (const char* part : parts) joined += part;
  REQUIRE(incremental == crcOfString(joined.c_str()));

  // Byte-level difference must change the checksum (one flipped bit).
  const auto base = crcOfString("KENSHIN GPU SCENE UPLOAD");
  const auto flipped = crcOfString("KENSHIN GPU SCENE VPLOAD");
  REQUIRE(base != flipped);
}
