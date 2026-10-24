#pragma once

// ── AssetManager — runtime assets by Guid + hot reload (roadmap 1.3d) ──
// The asset database's cooked artifacts, loaded on demand and shared:
//   texture  -> a GPU texture in the bindless table (KTX2 artifact)
//   material -> the parsed MaterialDesc; holds its textures
//   mesh     -> the cooked cluster DAG's tables (pages stream from
//               meshFile() through the renderer's StreamingManager)
// Every acquire adds a reference, every release drops one; the last
// release frees the resource. Guids may name an asset or a sub-asset
// (a model's mesh/N, material/N, texture/N.*).
//
// Hot reload, between frames:
//   update(dt)       polls assets/ (4 Hz by default); true once a burst
//                    of changes has settled (one quiet poll)
//   planReload()     rescans the database; the stale assets, and which
//                    resident meshes they own: stop streaming from
//                    those files first (the page streamer reopens a
//                    mesh file by path for every page - after the
//                    re-import it would read the new file against the
//                    old tables - and a read in flight blocks the
//                    folder swap on Windows)
//   applyReload(p)   re-imports and reloads every RESIDENT resource
//                    whose cooked bytes changed: textures in place (same
//                    handle and bindless slot, so material records stay
//                    valid), materials and meshes as new contents
// A failed re-import keeps the old artifacts and the loaded resources.

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "kenshin/Common.h"
#include "kenshin/assets/AssetDatabase.h"
#include "kenshin/assets/Material.h"
#include "kenshin/backend/TextureFormat.h"
#include "kenshin/core/FileWatcher.h"
#include "kenshin/core/Guid.h"

KENSHIN_NS_BEGIN

namespace backend {
class IRenderBackend;
}
namespace geometry {
struct ClusterDag;
}

namespace manager {

struct AssetManagerOptions {
  /// Seconds between watcher polls; 0 polls on every update() call
  /// (scripted runs and tests: reloads land a fixed number of frames
  /// after a change).
  f32 pollInterval = 0.25f;
  bool watch = true;
};

struct ReloadPlan {
  std::vector<core::Guid> assets;          // stale assets, by path
  std::vector<core::Guid> residentMeshes;  // resident meshes they own
  bool empty() const { return assets.empty(); }
};

/// What applyReload changed among the resident resources.
struct AssetReload {
  std::vector<core::Guid> reimported;  // asset Guids
  std::vector<std::string> failed;     // "path: reason"
  std::vector<core::Guid> textures;    // replaced in place
  std::vector<core::Guid> materials;   // new descs (texture refs may differ)
  std::vector<core::Guid> meshes;      // new DAGs (rebuild the scene)
};

class AssetManager {
 public:
  AssetManager();
  ~AssetManager();
  AssetManager(const AssetManager&) = delete;
  AssetManager& operator=(const AssetManager&) = delete;

  bool init(backend::IRenderBackend& backend, assets::AssetDatabase& database,
            const AssetManagerOptions& options = {});
  /// Frees everything still resident (and logs it: a leak elsewhere).
  void shutdown();

  // ---- textures ------------------------------------------------------
  /// +1 reference. Returns the backend texture handle, or
  /// kInvalidTextureHandle (unknown Guid, not a texture, load failure).
  u32 acquireTexture(const core::Guid& texture);
  void releaseTexture(const core::Guid& texture);
  /// The material-record reference of a resident texture (kNoTexture
  /// when it is not resident).
  u32 textureRef(const core::Guid& texture, backend::SamplerDesc sampler) const;

  // ---- materials -----------------------------------------------------
  /// +1 reference (its textures are acquired with it). The pointer is
  /// valid until the material is released or reloads.
  const assets::MaterialDesc* acquireMaterial(const core::Guid& material);
  void releaseMaterial(const core::Guid& material);

  // ---- meshes --------------------------------------------------------
  /// +1 reference. The pointer is valid until the mesh is released or
  /// reloads.
  const geometry::ClusterDag* acquireMesh(const core::Guid& mesh);
  void releaseMesh(const core::Guid& mesh);
  /// A resident mesh's cooked file and its slot materials.
  std::filesystem::path meshFile(const core::Guid& mesh) const;
  std::vector<core::Guid> meshMaterials(const core::Guid& mesh) const;

  // ---- hot reload ----------------------------------------------------
  bool update(f32 dt);
  ReloadPlan planReload();
  AssetReload applyReload(const ReloadPlan& plan);

  u32 references(const core::Guid& guid) const;
  usize residentTextures() const;
  usize residentMaterials() const;
  usize residentMeshes() const;
  /// Resources read from disk so far (loads and reloads; diagnostics).
  usize loads() const { return m_loads; }
  assets::AssetDatabase& database() { return *m_db; }

 private:
  struct Texture {
    u32 refs = 0;
    u32 handle = 0xFFFFFFFFu;
    core::Guid owner;  // the asset whose import produces it
    std::string sha1;  // of the loaded artifact
  };
  struct MaterialEntry {
    u32 refs = 0;
    assets::MaterialDesc desc;
    core::Guid owner;
    std::string sha1;
  };
  struct Mesh {
    u32 refs = 0;
    std::unique_ptr<geometry::ClusterDag> dag;
    std::filesystem::path file;
    std::vector<core::Guid> materials;
    core::Guid owner;
    std::string sha1;
  };

  bool loadTexture(const core::Guid& guid, Texture& out, bool replace);
  bool loadMaterialDesc(const core::Guid& guid, assets::MaterialDesc& desc, std::string& sha1);
  void acquireMaterialTextures(const assets::MaterialDesc& desc);
  void releaseMaterialTextures(const assets::MaterialDesc& desc);
  /// The artifact behind `guid` if it has `type` (logged otherwise).
  const assets::Artifact* artifactOf(const core::Guid& guid, assets::ArtifactType type,
                                     core::Guid* owner) const;

  backend::IRenderBackend* m_backend = nullptr;
  assets::AssetDatabase* m_db = nullptr;
  AssetManagerOptions m_options;
  core::FileWatcher m_watcher;
  f32 m_sincePoll = 0.0f;
  bool m_pending = false;
  usize m_loads = 0;
  std::unordered_map<core::Guid, Texture> m_textures;
  std::unordered_map<core::Guid, MaterialEntry> m_materials;
  std::unordered_map<core::Guid, Mesh> m_meshes;
};

}  // namespace manager
KENSHIN_NS_END
