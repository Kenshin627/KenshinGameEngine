// Bindless material textures on the GPU (roadmap 1.3a): real-format
// images with CPU-cooked mips, the slot table and the sampler table,
// probed by a compute shader through the same indexing the material
// shaders use (bindless_probe.slang).

#include <catch2/catch_test_macros.hpp>
#include <array>
#include <glm/glm.hpp>
#include <cmath>
#include <cstring>
#include <vector>

#include "kenshin/assets/TextureCooker.h"
#include "kenshin/backend/RenderBackend.h"
#include "kenshin/manager/ShaderManager.h"
#include "kenshin/util/Half.h"

using namespace kenshin;
using backend::FilterMode;
using backend::IRenderBackend;
using backend::SamplerDesc;
using backend::TextureFormat;
using backend::WrapMode;

namespace {

const char* kShaderRoot = KENSHIN_SOURCE_DIR "/engine/shaders";

/// A texture whose every mip level is one flat RGBA8 colour, so a probe
/// at level L proves level L landed where the upload put it.
assets::CookedTexture levels8(u32 w, u32 h, TextureFormat f,
                              const std::vector<std::array<u8, 4>>& colours) {
  assets::CookedTexture t;
  t.format = f;
  t.width = w;
  t.height = h;
  for (u32 m = 0; m < backend::fullMipCount(w, h); ++m) {
    const u32 mw = std::max(1u, w >> m), mh = std::max(1u, h >> m);
    t.mips.push_back({mw, mh, t.data.size(), usize(mw) * mh * 4});
    for (u32 i = 0; i < mw * mh; ++i)
      t.data.insert(t.data.end(), colours[m].begin(), colours[m].end());
  }
  return t;
}

u32 upload(IRenderBackend& be, const assets::CookedTexture& t) {
  IRenderBackend::TextureDesc d;
  d.width = t.width;
  d.height = t.height;
  d.mipCount = static_cast<u32>(t.mips.size());
  d.format = t.format;
  d.data = t.data.data();
  d.bytes = t.data.size();
  d.debugName = "bindless-test";
  return be.createTexture(d);
}

struct Probe {
  u32 ref;
  f32 u, v;
  f32 grad;  // isotropic UV gradient (see lod())
};

/// The UV gradient that selects mip `level` of a texture whose larger
/// side is `maxSide` texels (footprint = grad * maxSide = 2^level).
f32 lod(u32 level, u32 maxSide) { return std::ldexp(1.0f, static_cast<int>(level)) / maxSide; }
constexpr f32 kMip0 = 1e-6f;

class Prober {
 public:
  explicit Prober(IRenderBackend& be) : m_be(be) {
    manager::ShaderManager shaders;
    REQUIRE(shaders.init(kShaderRoot));
    std::vector<u32> spirv;
    std::string log;
    INFO(log);
    REQUIRE(shaders.compileFile("bindless_probe.slang", spirv, log, "main"));
    m_pipeline = be.createComputePipeline({spirv.data(), spirv.size() * sizeof(u32)});
    REQUIRE(m_pipeline != IRenderBackend::kInvalidPipelineHandle);
  }
  ~Prober() { m_be.destroyComputePipeline(m_pipeline); }

  std::vector<glm::vec4> run(const std::vector<Probe>& probes) {
    std::vector<glm::vec4> in(probes.size());
    for (usize i = 0; i < probes.size(); ++i) {
      std::memcpy(&in[i].x, &probes[i].ref, 4);
      in[i].y = probes[i].u;
      in[i].z = probes[i].v;
      in[i].w = probes[i].grad;
    }
    const usize bytes = in.size() * sizeof(glm::vec4);
    const u32 inBuf = m_be.createBuffer({bytes, IRenderBackend::BufferStorage, in.data(), "probes"});
    const u32 outBuf = m_be.createBuffer({bytes, IRenderBackend::BufferStorage, nullptr, "results"});
    REQUIRE(m_be.beginFrame(backend::FrameBeginInfo{.width = 64, .height = 64}) ==
            backend::FrameStatus::Ready);
    const u32 handles[2] = {inBuf, outBuf};
    m_be.dispatchCompute(m_pipeline, handles, 2, static_cast<u32>((probes.size() + 63) / 64), 1, 1);
    m_be.endFrame();
    std::vector<u8> raw;
    REQUIRE(m_be.readbackBuffer(outBuf, 0, bytes, raw));
    m_be.destroyBuffer(inBuf);
    m_be.destroyBuffer(outBuf);
    std::vector<glm::vec4> out(probes.size());
    std::memcpy(out.data(), raw.data(), bytes);
    return out;
  }

 private:
  IRenderBackend& m_be;
  u32 m_pipeline = IRenderBackend::kInvalidPipelineHandle;
};

bool near(const glm::vec4& a, const glm::vec4& b, f32 eps) {
  return glm::all(glm::lessThanEqual(glm::abs(a - b), glm::vec4(eps)));
}

glm::vec4 unorm(std::array<u8, 4> c) {
  return glm::vec4(c[0], c[1], c[2], c[3]) / 255.0f;
}
glm::vec4 srgb(std::array<u8, 4> c) {
  return glm::vec4(assets::srgbToLinear(c[0] / 255.0f), assets::srgbToLinear(c[1] / 255.0f),
                   assets::srgbToLinear(c[2] / 255.0f), c[3] / 255.0f);
}

}  // namespace

TEST_CASE("bindless: every mip level of every format samples back", "[gpu][bindless]") {
  auto be = backend::createVulkanBackend();
  REQUIRE(be->initialize({}));
  {
    Prober prober(*be);
    // 13x6: non-square, non-power-of-two (4 levels: 13x6 6x3 3x1 1x1).
    const std::vector<std::array<u8, 4>> colours = {
        {200, 10, 10, 255}, {10, 200, 10, 128}, {10, 10, 200, 64}, {90, 90, 90, 0}};
    const u32 lin = upload(*be, levels8(13, 6, TextureFormat::RGBA8Unorm, colours));
    const u32 col = upload(*be, levels8(13, 6, TextureFormat::RGBA8Srgb, colours));
    REQUIRE(lin != IRenderBackend::kInvalidTextureHandle);
    REQUIRE(col != IRenderBackend::kInvalidTextureHandle);
    // RGBA16F: values above 1 survive (HDR emissive / env data).
    assets::CookedTexture hdr;
    hdr.format = TextureFormat::RGBA16Float;
    hdr.width = hdr.height = 1;
    hdr.mips.push_back({1, 1, 0, 8});
    for (const f32 v : {12.5f, 0.25f, 3000.0f, 1.0f}) {
      const u16 h = util::f32ToHalf(v);
      hdr.data.insert(hdr.data.end(), reinterpret_cast<const u8*>(&h),
                      reinterpret_cast<const u8*>(&h) + 2);
    }
    const u32 half = upload(*be, hdr);
    // RGBA16 UNORM.
    assets::CookedTexture deep;
    deep.format = TextureFormat::RGBA16Unorm;
    deep.width = deep.height = 1;
    deep.mips.push_back({1, 1, 0, 8});
    const u16 d16[4] = {0, 65535, 32768, 65535};
    deep.data.resize(8);
    std::memcpy(deep.data.data(), d16, 8);
    const u32 deepTex = upload(*be, deep);

    const SamplerDesc nearest{WrapMode::Clamp, WrapMode::Clamp, FilterMode::Nearest};
    const u32 linRef = backend::textureRef(be->bindlessSlot(lin), nearest);
    const u32 colRef = backend::textureRef(be->bindlessSlot(col), nearest);
    std::vector<Probe> probes;
    for (u32 m = 0; m < 4; ++m) probes.push_back({linRef, 0.5f, 0.5f, lod(m, 13)});
    for (u32 m = 0; m < 4; ++m) probes.push_back({colRef, 0.5f, 0.5f, lod(m, 13)});
    probes.push_back({backend::textureRef(be->bindlessSlot(half), nearest), 0.5f, 0.5f, kMip0});
    probes.push_back({backend::textureRef(be->bindlessSlot(deepTex), nearest), 0.5f, 0.5f, kMip0});
    probes.push_back({backend::textureRef(0, nearest), 0.5f, 0.5f, kMip0});  // slot 0: white
    const std::vector<glm::vec4> r = prober.run(probes);
    for (u32 m = 0; m < 4; ++m) {
      INFO("mip " << m);
      CHECK(near(r[m], unorm(colours[m]), 1.0f / 512));
      // Hardware sRGB decode matches the cooker's transfer function.
      CHECK(near(r[4 + m], srgb(colours[m]), 2e-3f));
    }
    CHECK(near(r[8], glm::vec4(12.5f, 0.25f, 3000.0f, 1.0f), 1e-3f));
    CHECK(near(r[9], glm::vec4(0.0f, 1.0f, 32768 / 65535.0f, 1.0f), 1e-4f));
    CHECK(near(r[10], glm::vec4(1.0f), 1e-6f));
    for (const u32 t : {lin, col, half, deepTex}) be->destroyTexture(t);
  }
  be->shutdown();
}

TEST_CASE("bindless: the sampler table's wrap and filter modes", "[gpu][bindless]") {
  auto be = backend::createVulkanBackend();
  REQUIRE(be->initialize({}));
  {
    Prober prober(*be);
    // 2x1, no mips: red | green.
    assets::CookedTexture t;
    t.format = TextureFormat::RGBA8Unorm;
    t.width = 2;
    t.height = 1;
    t.mips.push_back({2, 1, 0, 8});
    t.data = {255, 0, 0, 255, 0, 255, 0, 255};
    const u32 tex = upload(*be, t);
    REQUIRE(tex != IRenderBackend::kInvalidTextureHandle);
    const u32 slot = be->bindlessSlot(tex);
    auto ref = [&](WrapMode w, FilterMode f) { return backend::textureRef(slot, {w, w, f}); };
    const glm::vec4 red(1, 0, 0, 1), green(0, 1, 0, 1), mix(0.5f, 0.5f, 0, 1);
    struct Case {
      Probe p;
      glm::vec4 expect;
      const char* what;
    };
    const Case cases[] = {
        {{ref(WrapMode::Repeat, FilterMode::Nearest), 1.25f, 0.5f, kMip0}, red, "repeat 1.25"},
        {{ref(WrapMode::Repeat, FilterMode::Nearest), 1.75f, 0.5f, kMip0}, green, "repeat 1.75"},
        {{ref(WrapMode::Clamp, FilterMode::Nearest), 1.25f, 0.5f, kMip0}, green, "clamp 1.25"},
        {{ref(WrapMode::Clamp, FilterMode::Nearest), -3.0f, 0.5f, kMip0}, red, "clamp -3"},
        {{ref(WrapMode::Mirror, FilterMode::Nearest), 1.25f, 0.5f, kMip0}, green, "mirror 1.25"},
        {{ref(WrapMode::Mirror, FilterMode::Nearest), 1.75f, 0.5f, kMip0}, red, "mirror 1.75"},
        {{ref(WrapMode::Clamp, FilterMode::Linear), 0.5f, 0.5f, kMip0}, mix, "linear centre"},
        {{ref(WrapMode::Clamp, FilterMode::Nearest), 0.49f, 0.5f, kMip0}, red, "nearest left"},
    };
    std::vector<Probe> probes;
    for (const Case& c : cases) probes.push_back(c.p);
    const std::vector<glm::vec4> r = prober.run(probes);
    for (usize i = 0; i < std::size(cases); ++i) {
      INFO(cases[i].what << ": " << r[i].x << " " << r[i].y << " " << r[i].z);
      CHECK(near(r[i], cases[i].expect, 1.0f / 256));
    }
    be->destroyTexture(tex);
  }
  be->shutdown();
}

TEST_CASE("bindless: slots are reused and released slots read white", "[gpu][bindless]") {
  auto be = backend::createVulkanBackend();
  REQUIRE(be->initialize({}));
  {
    Prober prober(*be);
    const std::vector<std::array<u8, 4>> black = {{0, 0, 0, 255}};
    const u32 base = be->bindlessUsed();
    const u32 a = upload(*be, levels8(1, 1, TextureFormat::RGBA8Unorm, black));
    const u32 b = upload(*be, levels8(1, 1, TextureFormat::RGBA8Unorm, black));
    CHECK(be->bindlessUsed() == base + 2);
    const u32 slotA = be->bindlessSlot(a);
    CHECK(slotA != 0);
    CHECK(slotA != be->bindlessSlot(b));
    be->destroyTexture(a);
    CHECK(be->bindlessSlot(a) == backend::kNoTexture);
    CHECK(be->bindlessUsed() == base + 1);
    // A stale reference to the freed slot reads the white default, not
    // the destroyed image.
    const SamplerDesc s{WrapMode::Clamp, WrapMode::Clamp, FilterMode::Nearest};
    std::vector<glm::vec4> r = prober.run({{backend::textureRef(slotA, s), 0.5f, 0.5f, kMip0}});
    CHECK(near(r[0], glm::vec4(1.0f), 1e-6f));
    const u32 c = upload(*be, levels8(1, 1, TextureFormat::RGBA8Unorm, black));
    CHECK(be->bindlessSlot(c) == slotA);  // reused
    r = prober.run({{backend::textureRef(slotA, s), 0.5f, 0.5f, kMip0}});
    CHECK(near(r[0], glm::vec4(0, 0, 0, 1), 1e-6f));
    be->destroyTexture(b);
    be->destroyTexture(c);
    CHECK(be->bindlessUsed() == base);
  }
  be->shutdown();
}

TEST_CASE("bindless: invalid texture descriptions are rejected", "[gpu][bindless]") {
  auto be = backend::createVulkanBackend();
  REQUIRE(be->initialize({}));
  std::vector<u8> px(4 * 4 * 4 + 2 * 2 * 4 + 4, 7);
  IRenderBackend::TextureDesc d;
  d.width = 4;
  d.height = 4;
  d.mipCount = 3;
  d.format = TextureFormat::RGBA8Unorm;
  d.data = px.data();
  d.bytes = px.size();
  const u32 ok = be->createTexture(d);
  CHECK(ok != IRenderBackend::kInvalidTextureHandle);
  be->destroyTexture(ok);
  IRenderBackend::TextureDesc bad = d;
  bad.bytes = px.size() - 1;
  CHECK(be->createTexture(bad) == IRenderBackend::kInvalidTextureHandle);
  bad = d;
  bad.mipCount = 4;  // a 4x4 has 3 levels
  CHECK(be->createTexture(bad) == IRenderBackend::kInvalidTextureHandle);
  bad = d;
  bad.width = 0;
  CHECK(be->createTexture(bad) == IRenderBackend::kInvalidTextureHandle);
  bad = d;
  bad.data = nullptr;
  CHECK(be->createTexture(bad) == IRenderBackend::kInvalidTextureHandle);
  CHECK(be->bindlessSlot(12345) == backend::kNoTexture);
  be->shutdown();
}
