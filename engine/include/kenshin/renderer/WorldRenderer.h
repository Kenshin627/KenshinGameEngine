#pragma once

// ── WorldRenderer — an ecs::World through the SceneRenderer (1.3d) ──
// The minimal STATIC bridge: build() turns every visible MeshRenderer
// (with its WorldTransform as of the last World::updateTransforms) into
// virtual-geometry instances, loading meshes, materials and textures by
// Guid through the AssetManager, and sets up a SceneRenderer over them.
// Nothing follows the World afterwards; the caller rebuilds after a
// change (1.4 makes the bridge dynamic).
//
// * Materials: a mesh slot uses its material asset (the import record's
//   slot list); a MeshRenderer's `material` overrides every slot of
//   that instance. A slot without a loadable material keeps the mesh's
//   own factor table.
// * Hot reload: textures are replaced in place by the AssetManager (no
//   work here); updateMaterials() rewrites the records of changed
//   materials; changed meshes need build() again. closeFiles() stops
//   the page streamer, which reopens the mesh files by path for every
//   page, before a re-import replaces them.
// * Not drawn yet (logged per build): translucent sub-meshes (BLEND),
//   occlusion maps not packed into the metallic-roughness map,
//   alpha-mask cutoffs.

#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "kenshin/Common.h"
#include "kenshin/core/Guid.h"
#include "kenshin/renderer/Environment.h"
#include "kenshin/renderer/SceneRenderer.h"

KENSHIN_NS_BEGIN

namespace backend {
class IRenderBackend;
}
namespace ecs {
class World;
}
namespace manager {
class AssetManager;
class ShaderManager;
}  // namespace manager

namespace renderer {

class VirtualGeometryScene;
class StreamingManager;

struct WorldRendererOptions {
  usize streamBudgetBytes = 0;  // resident page budget; 0 = unlimited
};

class WorldRenderer {
 public:
  WorldRenderer();
  ~WorldRenderer();
  WorldRenderer(const WorldRenderer&) = delete;
  WorldRenderer& operator=(const WorldRenderer&) = delete;

  bool init(backend::IRenderBackend& backend, manager::ShaderManager& shaders,
            manager::AssetManager& assets, const WorldRendererOptions& options = {});
  /// Builds the GPU scene from `world`, replacing the previous one. The
  /// new build acquires its resources before the old one releases its
  /// own, so shared resources stay resident. False (with `error`) when
  /// nothing could be built; the previous scene is gone either way.
  bool build(const ecs::World& world, std::string* error = nullptr);
  /// Drops the GPU scene and every resource reference.
  void clear();
  /// Closes the streamed mesh files; nothing renders until build().
  void closeFiles();
  /// Rewrites the material records that use `materials` (their new
  /// contents are already in the AssetManager). Returns how many
  /// records changed.
  usize updateMaterials(const std::vector<core::Guid>& materials);

  bool ready() const { return m_build != nullptr && !m_filesClosed; }
  bool usesMesh(const core::Guid& mesh) const;

  void setEnvironment(const EnvironmentProbe& probe);
  void render(const SceneRenderer::FrameInputs& in, const SceneRenderSettings& settings);
  void afterEndFrame();
  void logStats();
  /// Null without a scene.
  const SceneRenderStats* stats() const;
  StreamingManager* streamer() const;
  VirtualGeometryScene* scene() const;

  /// Union of the instances' world bounding spheres (w < 0: empty).
  glm::vec4 bounds() const;
  u32 instanceCount() const;
  u32 meshCount() const;
  u32 materialRecordCount() const;

  void shutdown();

 private:
  struct Build;
  std::unique_ptr<Build> makeBuild(const ecs::World& world, std::string& error);
  void destroy(std::unique_ptr<Build>& build);

  backend::IRenderBackend* m_backend = nullptr;
  manager::ShaderManager* m_shaders = nullptr;
  manager::AssetManager* m_assets = nullptr;
  WorldRendererOptions m_options;
  EnvironmentProbe m_env;
  bool m_hasEnv = false;
  bool m_filesClosed = false;
  std::unique_ptr<Build> m_build;
};

}  // namespace renderer
KENSHIN_NS_END
