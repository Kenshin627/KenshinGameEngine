#pragma once

// ── InstanceCullPass (S4.2) ──────────────────────────────────────
// Frustum-culls instances on the GPU (cull_instances.slang): one
// thread per instance, 6-plane test against the WORLD-space bounds
// sphere packed in the instance buffer, atomic-append survivors.
// Runs through the one-shot compute channel (reflection-driven
// layout); the in-frame dispatch joins when the VisBuffer pipeline
// assembles. The CPU reference in tests/ mirrors the math exactly.

#include <glm/glm.hpp>

#include <span>
#include <vector>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace backend {
class IRenderBackend;
}
namespace manager {
class ShaderManager;
}

namespace renderer {

/// Extract the 6 view-frustum planes from a viewProj matrix
/// (Gribb-Hartmann row combinations; Vulkan ZO NDC: z in [0, w]).
/// Planes are NORMALIZED (xyz unit length) so the sphere test's
/// distances are metric; positive side = inside. Order: left, right,
/// bottom, top, near, far. The CPU reference and the shader must use
/// this same extraction.
void extractFrustumPlanes(const glm::mat4& viewProj,
                          glm::vec4 planes[6]);

/// The 96B std430 GPU instance (mirrors VirtualGeometryScene's packing;
/// declared here for callers that stage synthetic instances for tests).
struct CullGpuInstance {
  glm::mat4 model{1.0f};
  u32 meshId = 0;
  u32 pad[3] = {};
  glm::vec4 bounds{0.0f, 0.0f, 0.0f, -1.0f};  // world-space cull sphere
};

/// Run the cull on the graphics queue (one-shot; blocks until done).
/// `planes` as from extractFrustumPlanes. Returns false on shader
/// compile or dispatch failure. On success `visibleOut` holds the
/// surviving instance indices (ORDER IS NONDETERMINISTIC - compare as
/// a set).
bool cullInstancesOnGpu(backend::IRenderBackend& backend,
                         manager::ShaderManager& shaders,
                         std::span<const CullGpuInstance> instances,
                         const glm::vec4 planes[6],
                         std::vector<u32>& visibleOut);

}  // namespace renderer

KENSHIN_NS_END
