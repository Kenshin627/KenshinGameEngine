#pragma once

// ── TaaPass (M9) — the temporal antialiasing resolve ──────────────
// The geometry passes render with a per-frame subpixel jitter (the
// app offsets the projection's NDC xy); this pass sits between
// lighting and tonemap: reproject the previous frame's RESOLVED HDR
// through GBuffer A's euclidean view distance onto the current HDR,
// blend with a neighborhood clamp (taa.slang), and hand the fresh
// resolve to tonemap. The backend owns the ping-pong history pair;
// this class owns the pipeline + the per-frame matrices SSBO.

#include <glm/glm.hpp>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace backend {
class IRenderBackend;
}
namespace manager {
class ShaderManager;
}

namespace renderer {

struct TaaFrame {
  glm::mat4 invViewProj{1.0f};  // current UNJITTERED inverse
  glm::mat4 prevViewProj{1.0f}; // previous UNJITTERED
  glm::vec3 camPos{0.0f};
  float blend = 0.9f;           // 0 = no history (first frame/resize)
  glm::vec2 jitter{0.0f};       // THIS frame's jitter in UV units
};

class TaaPass {
 public:
  bool init(backend::IRenderBackend& backend, manager::ShaderManager& shaders);
  void onShutdown(backend::IRenderBackend& backend);

  void draw(backend::IRenderBackend& backend, const TaaFrame& frame,
            u32 width, u32 height);

 private:
  u32 m_pipeline = ~0u;
  u32 m_params = ~0u;  // mapped 160B: 2 mat4 + 2 vec4 (push caps at 128)
};

}  // namespace renderer

KENSHIN_NS_END
