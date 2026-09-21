// Slang compile + SPIR-V reflection integration (headless: Slang runs on
// the CPU, reflection is pure parsing - no GPU or window needed).

#include <kenshin/manager/ShaderManager.h>

#include <catch2/catch_test_macros.hpp>
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
  REQUIRE_FALSE(kenshin::manager::reflectDescriptorBindings(garbage, bindings));
}

TEST_CASE("reflect texture array + separate sampler bindings", "[shader]") {
  // M8.2's albedo form: Texture2DArray + SamplerState must compile
  // through the vendored Slang and reflect as SAMPLED_IMAGE + SAMPLER
  // (two descriptors, not a combined image sampler) - the mixed
  // descriptor layout depends on distinguishing them.
  kenshin::manager::ShaderManager shaders;
  REQUIRE(shaders.init(kShaderRoot));

  std::vector<kenshin::u32> spirv;
  std::string log;
  REQUIRE(shaders.compileFile("texprobe.slang", spirv, log, "main"));

  std::vector<kenshin::manager::ReflectedBinding> bindings;
  REQUIRE(kenshin::manager::reflectDescriptorBindings(spirv, bindings));
  REQUIRE(bindings.size() == 4);
  REQUIRE(bindings[0].binding == 0);
  REQUIRE(bindings[0].type ==
          kenshin::manager::ReflectedBinding::Type::SampledTexture);
  REQUIRE(bindings[1].binding == 1);
  REQUIRE(bindings[1].type ==
          kenshin::manager::ReflectedBinding::Type::Sampler);
  REQUIRE(bindings[2].type ==
          kenshin::manager::ReflectedBinding::Type::StorageBuffer);
  REQUIRE(bindings[3].type ==
          kenshin::manager::ReflectedBinding::Type::StorageBuffer);
}

TEST_CASE("repeated compiles of one shader stay alive and cache cleanly",
          "[shader]") {
  // Regression for the module-cache use-after-free: loadModule caches by
  // name without holding its own reference, so a release() per compile
  // used to leave the NEXT compile walking freed memory. Compiling the
  // same file repeatedly must be stable - and with the SPIR-V cache,
  // repeats must return identical bytes.
  kenshin::manager::ShaderManager shaders;
  REQUIRE(shaders.init(kShaderRoot));

  std::vector<kenshin::u32> first, again;
  std::string log;
  REQUIRE(shaders.compileFile("test_atomic.slang", first, log, "main"));
  for (int i = 0; i < 5; ++i) {
    REQUIRE(shaders.compileFile("test_atomic.slang", again, log, "main"));
    REQUIRE(again == first);
  }

  // Two entry points from one file are distinct cache entries.
  std::vector<kenshin::u32> vs, fs;
  REQUIRE(shaders.compileFile("baseline.slang", vs, log, "vsMain"));
  REQUIRE(shaders.compileFile("baseline.slang", fs, log, "fsMain"));
  REQUIRE_FALSE(vs.empty());
  REQUIRE_FALSE(fs.empty());
  REQUIRE(vs != fs);
}
