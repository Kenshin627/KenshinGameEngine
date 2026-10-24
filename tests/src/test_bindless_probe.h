#pragma once

// The bindless table probe (bindless_probe.slang): samples texture
// references on the GPU through the material shaders' own
// sampleTextureGrad (bindless_tests, hot_reload_tests).

#include <catch2/catch_test_macros.hpp>
#include <glm/glm.hpp>
#include <cmath>
#include <cstring>
#include <vector>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/manager/ShaderManager.h"

namespace testprobe {

using namespace kenshin;
using backend::IRenderBackend;

inline const char* kShaderRoot = KENSHIN_SOURCE_DIR "/engine/shaders";

struct Probe {
  u32 ref;
  f32 u, v;
  f32 grad;  // isotropic UV gradient (see lod())
};

/// The UV gradient that selects mip `level` of a texture whose larger
/// side is `maxSide` texels (footprint = grad * maxSide = 2^level).
inline f32 lod(u32 level, u32 maxSide) { return std::ldexp(1.0f, static_cast<int>(level)) / maxSide; }
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

}  // namespace testprobe
