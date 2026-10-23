#include "kenshin/renderer/ExposurePass.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <glm/glm.hpp>
#include <string>
#include <vector>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/manager/ShaderManager.h"

KENSHIN_NS_BEGIN

namespace renderer {

namespace {

constexpr u32 kBins = 64;
constexpr float kStartDistance = 1.5f;  // r.EyeAdaptation.Exponential
                                        // TransitionDistance

// exposure.slang's ExposurePush, packed identically.
struct ExposurePushC {
  float histo[4];
  float pct[4];
  float adapt[4];
  float curve[4];
  glm::mat4 invViewProj;
};
static_assert(sizeof(ExposurePushC) == 128, "ExposurePush layout");

}  // namespace

bool ExposurePass::init(backend::IRenderBackend& backend,
                        manager::ShaderManager& shaders) {
  std::vector<u32> hist, adapt;
  std::string log;
  if (!shaders.compileFile("exposure.slang", hist, log, "csHistogram") ||
      !shaders.compileFile("exposure.slang", adapt, log, "csAdapt")) {
    KENSHIN_ERROR("[exposure] shader compile failed: {}", log);
    return false;
  }
  backend::IRenderBackend::ComputePipelineInfo info;
  info.csSpirv = hist.data();
  info.csBytes = hist.size() * sizeof(u32);
  m_histogramPipeline = backend.createExposurePipeline(info);
  info.csSpirv = adapt.data();
  info.csBytes = adapt.size() * sizeof(u32);
  m_adaptPipeline = backend.createExposurePipeline(info);
  if (m_histogramPipeline == backend::IRenderBackend::kInvalidPipelineHandle ||
      m_adaptPipeline == backend::IRenderBackend::kInvalidPipelineHandle) {
    KENSHIN_ERROR("[exposure] pipeline creation failed");
    return false;
  }
  backend::IRenderBackend::BufferDesc desc;
  desc.size = kBins * sizeof(u32);
  desc.usage = backend::IRenderBackend::BufferStorage |
               backend::IRenderBackend::BufferTransferDst;
  desc.debugName = "exposure-histogram";
  m_histogram = backend.createBuffer(desc);
  // Host-visible: zeroed here (valid = 0 -> the first frame jumps to its
  // target) and read back by the panel.
  desc.size = 2 * 4 * sizeof(float);
  desc.usage = backend::IRenderBackend::BufferStorage |
               backend::IRenderBackend::BufferHostVisible;
  desc.debugName = "exposure-state";
  m_state = backend.createBuffer(desc);
  if (m_histogram == backend::IRenderBackend::kInvalidBufferHandle ||
      m_state == backend::IRenderBackend::kInvalidBufferHandle) {
    KENSHIN_ERROR("[exposure] buffer creation failed");
    return false;
  }
  std::memset(backend.mappedPtr(m_state), 0, desc.size);
  m_ready = true;
  return true;
}

bool ExposurePass::draw(backend::IRenderBackend& backend,
                        const ExposureSettings& s, const ExposureView& view,
                        bool fromTaa, u32 width, u32 height, float dt,
                        bool forceTarget) {
  if (!m_ready || width == 0 || height == 0) return false;
  // PostProcessEyeAdaptation.cpp's parameter setup.
  const float logMax = s.histogramLogMax;
  const float logMin = std::min(s.histogramLogMin, logMax - 1.0f);
  const float scale = 1.0f / (logMax - logMin);
  const float high = std::clamp(s.highPercent, 1.0f, 99.0f) * 0.01f;
  const float low =
      std::min(std::clamp(s.lowPercent, 1.0f, 99.0f) * 0.01f, high);
  const float minWhite = std::min(s.minBrightness, s.maxBrightness);
  const float speedUp = std::max(s.speedUp, 0.001f);
  const float speedDown = std::max(s.speedDown, 0.001f);
  // Exponential slope matched to the linear step at StartDistance.
  constexpr float kFrameTimeEps = 1.0f / 60.0f;
  const float upM =
      kFrameTimeEps / ((1.0f - std::exp2(-kFrameTimeEps * speedUp)) *
                       (kStartDistance / speedUp));
  const float downM =
      kFrameTimeEps / ((1.0f - std::exp2(-kFrameTimeEps * speedDown)) *
                       (kStartDistance / speedDown));
  ExposurePushC push{};
  push.histo[0] = scale;
  push.histo[1] = -logMin * scale;
  push.histo[2] = std::exp2(logMin);
  push.pct[0] = low;
  push.pct[1] = high;
  push.pct[2] = minWhite * 0.18f;
  push.pct[3] = s.maxBrightness * 0.18f;
  push.adapt[0] = speedUp;
  push.adapt[1] = speedDown;
  push.adapt[2] = std::clamp(dt, 0.0f, 0.25f);  // hitches must not snap
  push.adapt[3] = forceTarget ? 1.0f : 0.0f;
  push.curve[0] = std::exp2(s.biasEv);
  push.curve[1] = upM;
  push.curve[2] = downM;
  const bool env = view.envTexture != ~0u;
  push.histo[3] = view.envIntensity;
  push.curve[3] = env ? std::max(view.envMip, 0.0f) : -1.0f;
  push.invViewProj = view.invViewProj;
  return backend.drawExposure(m_histogramPipeline, m_adaptPipeline, m_histogram,
                              m_state, view.envTexture, fromTaa, width, height,
                              &push, sizeof(push));
}

bool ExposurePass::readState(backend::IRenderBackend& backend,
                             float out[4]) const {
  if (!m_ready) return false;
  const auto* p = static_cast<const float*>(backend.mappedPtr(m_state));
  if (p == nullptr) return false;
  for (int i = 0; i < 4; ++i) out[i] = p[i];
  return out[3] > 0.5f;
}

void ExposurePass::onShutdown(backend::IRenderBackend& backend) {
  if (m_histogram != ~0u) backend.destroyBuffer(m_histogram);
  if (m_state != ~0u) backend.destroyBuffer(m_state);
  m_histogram = m_state = ~0u;
  m_histogramPipeline = m_adaptPipeline = ~0u;  // die with PipelineStore
  m_ready = false;
}

}  // namespace renderer

KENSHIN_NS_END
