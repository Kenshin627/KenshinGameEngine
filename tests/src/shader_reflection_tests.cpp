// Slang compile + SPIR-V reflection integration (headless: Slang runs on
// the CPU, reflection is pure parsing - no GPU or window needed).

#include <catch2/catch_test_macros.hpp>

#include <kenshin/manager/ShaderManager.h>

#include <string>
#include <vector>

namespace {

const char* kShaderRoot = KENSHIN_SOURCE_DIR "/engine/shaders";

}  // namespace

TEST_CASE("reflect descriptor bindings from compiled slang", "[shader]") {
  kenshin::manager::ShaderManager shaders;
  REQUIRE(shaders.init(kShaderRoot));

  // test_atomic.slang: three RWStructuredBuffers at set 0 bindings 0..2.
  std::vector<kenshin::u32> spirv;
  std::string log;
  REQUIRE(shaders.compileFile("test_atomic.slang", spirv, log, "main"));

  std::vector<kenshin::manager::ReflectedBinding> bindings;
  REQUIRE(kenshin::manager::reflectDescriptorBindings(spirv, bindings));
  REQUIRE(bindings.size() == 3);
  for (kenshin::u32 i = 0; i < 3; ++i) {
    REQUIRE(bindings[i].set == 0);
    REQUIRE(bindings[i].binding == i);
    REQUIRE(bindings[i].type ==
            kenshin::manager::ReflectedBinding::Type::StorageBuffer);
  }

  // baseline.slang (vsMain): push constants only, no descriptor sets.
  REQUIRE(shaders.compileFile("baseline.slang", spirv, log, "vsMain"));
  REQUIRE(kenshin::manager::reflectDescriptorBindings(spirv, bindings));
  REQUIRE(bindings.empty());

  // Garbage input must fail cleanly, not crash.
  const std::vector<kenshin::u32> garbage{0x07230203, 0xdeadbeef};
  REQUIRE_FALSE(
      kenshin::manager::reflectDescriptorBindings(garbage, bindings));
}
