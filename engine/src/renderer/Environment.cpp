#include "kenshin/renderer/Environment.h"

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/util/File.h"
#include "kenshin/util/Half.h"
#include "stb/stb_image.h"  // impl: the engine's stb_image_impl

KENSHIN_NS_BEGIN

namespace renderer {

namespace {

u32 uploadProbe(backend::IRenderBackend& backend, const std::vector<u16>& half,
                u32 w, u32 h, const char* name) {
  backend::IRenderBackend::TextureArrayDesc ed;
  ed.width = w;
  ed.height = h;
  ed.layers = 1;
  ed.hdr = true;
  ed.rgba = reinterpret_cast<const u8*>(half.data());
  ed.bytes = half.size() * sizeof(u16);
  ed.debugName = name;
  return backend.createTextureArray(ed);
}

}  // namespace

bool loadHdrEnvironment(backend::IRenderBackend& backend,
                        const std::filesystem::path& path,
                        EnvironmentProbe& out) {
  if (!std::filesystem::exists(path)) return false;
  int ew = 0, eh = 0, ec = 0;
  // Decode from memory with the Radiance header normalized: stb_image
  // requires the resolution line to START with "-Y " and rejects writers
  // that emit whitespace before it (a blank line followed by
  // " -Y 512 +X 1024" -> "unsupported data layout"), which left the
  // env.hdr IBL silently on the procedural fallback.
  std::vector<unsigned char> hdrBytes;
  if (FILE* hf = util::openFile(path, "rb")) {
    std::fseek(hf, 0, SEEK_END);
    const long len = std::ftell(hf);
    std::fseek(hf, 0, SEEK_SET);
    if (len > 0) {
      hdrBytes.resize(static_cast<size_t>(len));
      if (std::fread(hdrBytes.data(), 1, hdrBytes.size(), hf) !=
          hdrBytes.size())
        hdrBytes.clear();
    }
    std::fclose(hf);
  }
  // The header ends at the first blank line; strip spaces/tabs between
  // it and the resolution string.
  for (size_t i = 0; i + 1 < hdrBytes.size() && i < 4096; ++i) {
    if (hdrBytes[i] == '\n' && hdrBytes[i + 1] == '\n') {
      size_t j = i + 2;
      while (j < hdrBytes.size() && (hdrBytes[j] == ' ' || hdrBytes[j] == '\t'))
        ++j;
      hdrBytes.erase(hdrBytes.begin() + static_cast<long>(i + 2),
                     hdrBytes.begin() + static_cast<long>(j));
      break;
    }
  }
  float* data = hdrBytes.empty()
                    ? nullptr
                    : stbi_loadf_from_memory(hdrBytes.data(),
                                             static_cast<int>(hdrBytes.size()),
                                             &ew, &eh, &ec, 4);
  if (data == nullptr) {
    KENSHIN_WARN("[ibl] {} failed to decode (stbi: {})", path.string(),
                 stbi_failure_reason() ? stbi_failure_reason() : "?");
    return false;
  }
  if (ew <= 0 || eh <= 0) {
    stbi_image_free(data);
    return false;
  }
  EnvironmentProbe probe;
  probe.captured = true;
  std::vector<u16> half(usize(ew) * eh * 4);
  for (usize i = 0; i < half.size(); ++i) {
    // Radiance is non-negative and finite by definition: clamp stray
    // negatives/NaN/inf out of the probe (one bad texel smears through
    // every mip).
    float v = data[i];
    if (!(v >= 0.0f)) v = 0.0f;
    if (v > 65000.0f) v = 65000.0f;
    data[i] = v;
  }
  // Sun extraction: an outdoor HDRI's sun disk is orders of magnitude
  // above the sky (kloofendal peaks at 5e4 vs a ~1 sky). Left in, it
  // lights everything a SECOND time - through the SH diffuse and the
  // blurred mips - from a direction the key light and its shadow map
  // know nothing about. Texels above kSun are summed into the sun's
  // irradiance (sum L * dw) and luminance-weighted direction, clamped to
  // kSun in the probe. Same equirect Z-up convention as the shader.
  {
    constexpr float kSun = 100.0f;
    const double kPi = 3.14159265358979323846;
    glm::dvec3 irr(0.0), dirSum(0.0);
    usize sunTexels = 0;
    for (int y = 0; y < eh; ++y) {
      const double theta = (y + 0.5) / eh * kPi;
      const double dw = (2.0 * kPi / ew) * (kPi / eh) * std::sin(theta);
      for (int x = 0; x < ew; ++x) {
        float* t = data + (usize(y) * ew + x) * 4;
        const float lum = 0.2126f * t[0] + 0.7152f * t[1] + 0.0722f * t[2];
        if (!(lum > kSun)) continue;
        const double a = ((x + 0.5) / ew - 0.5) * 2.0 * kPi;
        const glm::dvec3 d(std::cos(a) * std::sin(theta),
                           std::sin(a) * std::sin(theta), std::cos(theta));
        irr += glm::dvec3(t[0], t[1], t[2]) * dw;
        dirSum += d * (double(lum) * dw);
        const float s = kSun / lum;
        t[0] *= s;
        t[1] *= s;
        t[2] *= s;
        ++sunTexels;
      }
    }
    if (sunTexels > 0 && glm::length(dirSum) > 0.0) {
      const glm::vec3 sunDir = glm::normalize(glm::vec3(dirSum));
      const glm::vec3 e(irr);
      const float peak = std::max(e.r, std::max(e.g, e.b));
      probe.hasSun = true;
      probe.sunDir = sunDir;
      probe.sunColor = e / peak;
      probe.sunIntensity = peak;
      KENSHIN_INFO(
          "[ibl] sun extracted: {} texels, dir ({:.2f}, {:.2f}, {:.2f}), "
          "irradiance ({:.2f}, {:.2f}, {:.2f}) -> directional light",
          sunTexels, sunDir.x, sunDir.y, sunDir.z, e.r, e.g, e.b);
    }
  }
  for (usize i = 0; i < half.size(); ++i) half[i] = util::f32ToHalf(data[i]);
  probe.sh = LightingPass::projectIrradianceSH(data, u32(ew), u32(eh));
  probe.shValid = true;
  stbi_image_free(data);
  probe.texture = uploadProbe(backend, half, u32(ew), u32(eh), "ibl-env-hdr");
  probe.maxLod = std::floor(std::log2(float(ew))) + 1.0f - 1.0f;
  KENSHIN_INFO("[ibl] loaded {} ({}x{}, {} mips) -> handle {}", path.string(),
               ew, eh, int(probe.maxLod) + 1, probe.texture);
  out = probe;
  return true;
}

EnvironmentProbe createStudioProbe(backend::IRenderBackend& backend) {
  constexpr int kW = 512, kH = 256;
  std::vector<float> probe(usize(kW) * kH * 4);
  // The diffuse source: the same probe WITHOUT the key softbox - the
  // softbox is the key light's reflection stand-in, and the key's
  // diffuse is already the direct term (counting it in the irradiance
  // too would double-light every surface).
  std::vector<float> probeDiffuse(probe.size());
  const glm::vec3 kd = glm::normalize(glm::vec3(0.35f, 0.8f, 0.45f));
  // ONE scale for the whole ambient part: it lights both the diffuse (SH)
  // and the specular ambient, so it sits in the key light's range
  // (radiance ~2.4).
  const char* envScale = std::getenv("KENSHIN_ENV_SCALE");
  const float es = envScale ? float(std::atof(envScale)) : 0.5f;
  for (int y = 0; y < kH; ++y) {
    const float v = (y + 0.5f) / kH;
    const float theta = v * glm::pi<float>();
    const float z = std::cos(theta), lat = std::sin(theta);
    for (int x = 0; x < kW; ++x) {
      const float u = (x + 0.5f) / kW;
      const float a = (u - 0.5f) * 2.0f * glm::pi<float>();
      const glm::vec3 dir(std::cos(a) * lat, std::sin(a) * lat, z);
      const float t = z * 0.5f + 0.5f;
      glm::vec3 c = glm::mix(glm::vec3(0.35f, 0.34f, 0.32f) * es,
                             glm::vec3(1.6f, 1.65f, 1.75f) * es, t * t);
      const float dFill = std::max(0.0f, glm::dot(dir, -kd));
      c += glm::vec3(0.55f, 0.65f, 0.85f) * std::exp((dFill - 1.0f) * 20.0f) *
           1.2f * es;
      float* od = probeDiffuse.data() + (usize(y) * kW + x) * 4;
      od[0] = c.r;
      od[1] = c.g;
      od[2] = c.b;
      od[3] = 1.0f;
      const float dKey = std::max(0.0f, glm::dot(dir, kd));
      c += glm::vec3(1.0f, 0.97f, 0.92f) * std::exp((dKey - 1.0f) * 12.0f) *
           5.0f;
      float* o = probe.data() + (usize(y) * kW + x) * 4;
      o[0] = c.r;
      o[1] = c.g;
      o[2] = c.b;
      o[3] = 1.0f;
    }
  }
  // RGBA16F upload: half bits (feeding f32 bits reads as denormal
  // garbage - a black environment).
  std::vector<u16> half;
  half.reserve(probe.size());
  for (const float v : probe) half.push_back(util::f32ToHalf(v));
  EnvironmentProbe out;
  out.sh = LightingPass::projectIrradianceSH(probeDiffuse.data(), kW, kH);
  out.shValid = true;
  out.texture = uploadProbe(backend, half, kW, kH, "ibl-env");
  out.maxLod = std::floor(std::log2(float(kW))) + 1.0f - 1.0f;
  KENSHIN_INFO("[ibl] studio probe {}x{} ({} mips) -> handle {}", kW, kH,
               int(out.maxLod) + 1, out.texture);
  return out;
}

}  // namespace renderer

KENSHIN_NS_END
