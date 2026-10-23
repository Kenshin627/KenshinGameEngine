#pragma once

// ── TranslucentPass — glTF alphaMode BLEND, forward-lit ─────────────
// UE's split: Nanite never rasterizes translucent materials; they draw
// in a forward pass after the opaque lighting. kgeo_builder moves BLEND
// triangles out of the cluster DAG into <name>.tmesh (a non-indexed
// triangle list + the material factors); this pass loads that file and
// blends it into the slot's HDR target between LightingPass and TaaPass
// (translucent.slang - premultiplied UE Translucent blend, occlusion
// tested against the VisBuffer's depth word, no shadows in v1).

#include <glm/glm.hpp>
#include <span>
#include <string>
#include <vector>

#include "kenshin/Common.h"
#include "kenshin/renderer/LightingPass.h"

KENSHIN_NS_BEGIN

namespace backend {
class IRenderBackend;
}
namespace manager {
class ShaderManager;
}

namespace renderer {

struct TranslucentFrame {
  glm::mat4 viewProj{1.0f};  // the JITTERED matrix the opaque frame used
  glm::vec3 camPos{0.0f};
  float nearZ = 0.01f, farZ = 1000.0f;
  std::span<const GpuLightC> lights;
  const LightingPass::IrradianceSH* envSH = nullptr;
  float envIntensity = 1.0f;
  float emissiveStrength = 1.0f;  // lighting's env[11].x twin
  /// Opaque emission behind translucent layers is not dimmed by them
  /// (the Sketchfab look; translucent.slang's pass-through term).
  bool emissivePassThrough = true;
  u32 envTexture = ~0u;
  float envMaxLod = 1.0f;
  u32 visBuffer = ~0u;  // this frame's VisBuffer (u64 per pixel)
  /// The scene's runtime material table (bindless texture refs) - the
  /// opaque decode's own; ~0u = factors only.
  u32 sceneMaterials = ~0u;
  /// Instance transforms the translucent mesh draws under (the loaded
  /// asset's instances - mesh 0).
  std::span<const glm::mat4> instances;
};

class TranslucentPass {
 public:
  /// Load `path` (a .tmesh) and create the pipeline. Returns false (and
  /// logs) on a missing/invalid file - the frame then simply has no
  /// translucency.
  bool init(backend::IRenderBackend& backend, manager::ShaderManager& shaders,
            const std::string& path);
  void onShutdown(backend::IRenderBackend& backend);
  bool ready() const { return m_pipeline != ~0u && m_vertexCount > 0; }
  u32 triangleCount() const { return m_vertexCount / 3; }

  void draw(backend::IRenderBackend& backend, const TranslucentFrame& frame,
            u32 width, u32 height);

 private:
  u32 m_pipeline = ~0u;
  u32 m_verts = ~0u;      // 3 float4 per vertex
  u32 m_materials = ~0u;  // 2 float4 per material
  u32 m_instances = ~0u;  // 4 float4 per instance (host-visible)
  u32 m_lights = ~0u;     // host-visible light table
  u32 m_env = ~0u;        // host-visible env table (LightingPass layout)
  u32 m_vertexCount = 0;
  static constexpr u32 kMaxInstances = 256;
  static constexpr u32 kMaxLights = 32;
};

}  // namespace renderer

KENSHIN_NS_END
