#include "kenshin/manager/AssetManager.h"

#include <algorithm>

#include "kenshin/assets/Ktx2.h"
#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/geometry/ClusterDag.h"
#include "kenshin/geometry/KGeoFile.h"
#include "kenshin/util/File.h"

KENSHIN_NS_BEGIN
namespace manager {

namespace fs = std::filesystem;
using assets::ArtifactType;
using backend::IRenderBackend;

AssetManager::AssetManager() = default;
AssetManager::~AssetManager() { shutdown(); }

bool AssetManager::init(backend::IRenderBackend& backend, assets::AssetDatabase& database,
                        const AssetManagerOptions& options) {
  m_backend = &backend;
  m_db = &database;
  m_options = options;
  m_sincePoll = 0.0f;
  m_pending = false;
  if (m_options.watch) m_watcher.watch(database.projectDir() / "assets");
  return true;
}

void AssetManager::shutdown() {
  if (m_backend == nullptr) return;
  const usize left = m_textures.size() + m_materials.size() + m_meshes.size();
  if (left > 0)
    KENSHIN_WARN("[assets] shutdown with {} resident resources ({} textures, {} materials, {} meshes)",
                 left, m_textures.size(), m_materials.size(), m_meshes.size());
  for (auto& [guid, t] : m_textures) m_backend->destroyTexture(t.handle);
  m_textures.clear();
  m_materials.clear();
  m_meshes.clear();
  m_backend = nullptr;
  m_db = nullptr;
}

const assets::Artifact* AssetManager::artifactOf(const core::Guid& guid, ArtifactType type,
                                                 core::Guid* owner) const {
  const auto r = m_db->resolve(guid);
  if (!r) {
    KENSHIN_ERROR("[assets] {} is not an imported asset", guid.toString());
    return nullptr;
  }
  if (r->artifact->type != type) {
    KENSHIN_ERROR("[assets] {} ({}) is a {}, not a {}", guid.toString(), r->asset->path,
                  assets::artifactTypeName(r->artifact->type), assets::artifactTypeName(type));
    return nullptr;
  }
  if (owner) *owner = r->asset->guid;
  return r->artifact;
}

// ── textures ──────────────────────────────────────────────────

bool AssetManager::loadTexture(const core::Guid& guid, Texture& out, bool replace) {
  core::Guid owner;
  const assets::Artifact* art = artifactOf(guid, ArtifactType::Texture, &owner);
  if (art == nullptr) return false;
  const fs::path file = m_db->artifactPath(guid);
  assets::CookedTexture tex;
  std::string error;
  if (!assets::loadKtx2File(file, tex, error)) {
    KENSHIN_ERROR("[assets] texture {}: {}", art->name, error);
    return false;
  }
  IRenderBackend::TextureDesc desc;
  desc.width = tex.width;
  desc.height = tex.height;
  desc.mipCount = static_cast<u32>(tex.mips.size());
  desc.format = tex.format;
  desc.data = tex.data.data();
  desc.bytes = tex.data.size();
  desc.debugName = art->name.c_str();
  if (replace) {
    if (!m_backend->replaceTexture(out.handle, desc)) return false;
  } else {
    out.handle = m_backend->createTexture(desc);
    if (out.handle == IRenderBackend::kInvalidTextureHandle) return false;
  }
  out.owner = owner;
  out.sha1 = art->sha1;
  ++m_loads;
  KENSHIN_INFO("[assets] texture {} {}: {}x{}, {} mips, slot {}", art->name,
               replace ? "reloaded" : "loaded", tex.width, tex.height, tex.mips.size(),
               m_backend->bindlessSlot(out.handle));
  return true;
}

u32 AssetManager::acquireTexture(const core::Guid& guid) {
  if (const auto it = m_textures.find(guid); it != m_textures.end()) {
    ++it->second.refs;
    return it->second.handle;
  }
  Texture t;
  if (!loadTexture(guid, t, false)) return IRenderBackend::kInvalidTextureHandle;
  t.refs = 1;
  return m_textures.emplace(guid, std::move(t)).first->second.handle;
}

void AssetManager::releaseTexture(const core::Guid& guid) {
  const auto it = m_textures.find(guid);
  if (it == m_textures.end()) {
    KENSHIN_ERROR("[assets] releaseTexture: {} is not resident", guid.toString());
    return;
  }
  if (--it->second.refs > 0) return;
  m_backend->destroyTexture(it->second.handle);
  m_textures.erase(it);
}

u32 AssetManager::textureRef(const core::Guid& guid, backend::SamplerDesc sampler) const {
  const auto it = m_textures.find(guid);
  if (it == m_textures.end()) return backend::kNoTexture;
  const u32 slot = m_backend->bindlessSlot(it->second.handle);
  return slot == backend::kNoTexture ? backend::kNoTexture : backend::textureRef(slot, sampler);
}

// ── materials ─────────────────────────────────────────────────

namespace {

template <typename F>
void forEachSlot(const assets::MaterialDesc& d, F&& f) {
  for (const assets::MaterialTextureSlot* s :
       {&d.baseColorTexture, &d.metallicRoughnessTexture, &d.normalTexture, &d.occlusionTexture,
        &d.emissiveTexture})
    if (s->valid()) f(s->texture);
}

}  // namespace

bool AssetManager::loadMaterialDesc(const core::Guid& guid, assets::MaterialDesc& desc,
                                    std::string& sha1) {
  const assets::Artifact* art = artifactOf(guid, ArtifactType::Material, nullptr);
  if (art == nullptr) return false;
  std::string text, error;
  if (!util::readFile(m_db->artifactPath(guid), text) || !assets::loadMaterial(text, desc, &error)) {
    KENSHIN_ERROR("[assets] material {}: {}", art->name, error.empty() ? "unreadable" : error);
    return false;
  }
  sha1 = art->sha1;
  ++m_loads;
  return true;
}

void AssetManager::acquireMaterialTextures(const assets::MaterialDesc& desc) {
  // A texture that fails to load is left out (the record samples no
  // texture); its release is skipped to match.
  forEachSlot(desc, [&](const core::Guid& t) { acquireTexture(t); });
}

void AssetManager::releaseMaterialTextures(const assets::MaterialDesc& desc) {
  forEachSlot(desc, [&](const core::Guid& t) {
    if (m_textures.count(t)) releaseTexture(t);
  });
}

const assets::MaterialDesc* AssetManager::acquireMaterial(const core::Guid& guid) {
  if (const auto it = m_materials.find(guid); it != m_materials.end()) {
    ++it->second.refs;
    return &it->second.desc;
  }
  MaterialEntry m;
  if (!loadMaterialDesc(guid, m.desc, m.sha1)) return nullptr;
  m.owner = m_db->resolve(guid)->asset->guid;
  m.refs = 1;
  acquireMaterialTextures(m.desc);
  return &m_materials.emplace(guid, std::move(m)).first->second.desc;
}

void AssetManager::releaseMaterial(const core::Guid& guid) {
  const auto it = m_materials.find(guid);
  if (it == m_materials.end()) {
    KENSHIN_ERROR("[assets] releaseMaterial: {} is not resident", guid.toString());
    return;
  }
  if (--it->second.refs > 0) return;
  const assets::MaterialDesc desc = it->second.desc;
  m_materials.erase(it);
  releaseMaterialTextures(desc);
}

// ── meshes ────────────────────────────────────────────────────

const geometry::ClusterDag* AssetManager::acquireMesh(const core::Guid& guid) {
  if (const auto it = m_meshes.find(guid); it != m_meshes.end()) {
    ++it->second.refs;
    return it->second.dag.get();
  }
  core::Guid owner;
  const assets::Artifact* art = artifactOf(guid, ArtifactType::Mesh, &owner);
  if (art == nullptr) return nullptr;
  Mesh m;
  m.file = m_db->artifactPath(guid);
  m.dag = std::make_unique<geometry::ClusterDag>();
  std::string log;
  if (!geometry::readKGeo(util::pathToUtf8(m.file), *m.dag, log)) {
    KENSHIN_ERROR("[assets] mesh {}: {}", art->name, log);
    return nullptr;
  }
  m.materials = art->materials;
  m.owner = owner;
  m.sha1 = art->sha1;
  m.refs = 1;
  ++m_loads;
  KENSHIN_INFO("[assets] mesh {} loaded: {} clusters, {} slots", art->name, m.dag->clusters.size(),
               m.materials.size());
  return m_meshes.emplace(guid, std::move(m)).first->second.dag.get();
}

void AssetManager::releaseMesh(const core::Guid& guid) {
  const auto it = m_meshes.find(guid);
  if (it == m_meshes.end()) {
    KENSHIN_ERROR("[assets] releaseMesh: {} is not resident", guid.toString());
    return;
  }
  if (--it->second.refs == 0) m_meshes.erase(it);
}

fs::path AssetManager::meshFile(const core::Guid& guid) const {
  const auto it = m_meshes.find(guid);
  return it == m_meshes.end() ? fs::path() : it->second.file;
}

std::vector<core::Guid> AssetManager::meshMaterials(const core::Guid& guid) const {
  const auto it = m_meshes.find(guid);
  return it == m_meshes.end() ? std::vector<core::Guid>() : it->second.materials;
}

// ── hot reload ────────────────────────────────────────────────

bool AssetManager::update(f32 dt) {
  if (!m_options.watch || m_db == nullptr) return false;
  m_sincePoll += dt;
  if (m_options.pollInterval > 0.0f && m_sincePoll < m_options.pollInterval) return false;
  m_sincePoll = 0.0f;
  // Settling: an editor's save can span several polls (write, rename,
  // a second file) - act on the first quiet poll after the burst.
  if (!m_watcher.poll().empty()) {
    m_pending = true;
    return false;
  }
  const bool ready = m_pending;
  m_pending = false;
  return ready;
}

ReloadPlan AssetManager::planReload() {
  ReloadPlan plan;
  if (m_db == nullptr) return plan;
  const assets::ScanReport scan = m_db->refresh();
  for (const std::string& s : scan.createdMetas) KENSHIN_INFO("[assets] new asset {}", s);
  for (const std::string& s : scan.adoptedMoves) KENSHIN_INFO("[assets] moved {}", s);
  for (const std::string& s : scan.corruptMetas) KENSHIN_WARN("[assets] corrupt meta {}", s);
  for (const assets::AssetRecord* a : m_db->assets())
    if (m_db->isStale(a->guid)) plan.assets.push_back(a->guid);
  for (const auto& [guid, mesh] : m_meshes)
    if (std::find(plan.assets.begin(), plan.assets.end(), mesh.owner) != plan.assets.end())
      plan.residentMeshes.push_back(guid);
  std::sort(plan.residentMeshes.begin(), plan.residentMeshes.end());
  return plan;
}

AssetReload AssetManager::applyReload(const ReloadPlan& plan) {
  AssetReload out;
  if (m_db == nullptr) return out;
  for (const core::Guid& g : plan.assets) {
    const assets::AssetRecord* rec = m_db->find(g);
    if (rec == nullptr) continue;
    std::string why;
    if (m_db->reimport(g, &why)) {
      out.reimported.push_back(g);
      KENSHIN_INFO("[assets] re-imported {}", rec->path);
    } else {
      out.failed.push_back(rec->path + ": " + why);
      KENSHIN_ERROR("[assets] re-import failed, keeping the loaded version: {}: {}", rec->path, why);
    }
  }
  const auto reimported = [&](const core::Guid& owner) {
    return std::find(out.reimported.begin(), out.reimported.end(), owner) != out.reimported.end();
  };
  // Changed = the artifact's bytes differ (a re-import that reproduces
  // a sub-asset byte for byte leaves it alone).
  const auto changed = [&](const core::Guid& guid, const std::string& sha1, ArtifactType type) {
    const auto r = m_db->resolve(guid);
    if (!r || r->artifact->type != type) {
      KENSHIN_WARN("[assets] {} is no longer produced by its asset; keeping the loaded copy",
                   guid.toString());
      return false;
    }
    return sha1.empty() || r->artifact->sha1 != sha1;
  };

  for (auto& [guid, t] : m_textures)
    if (reimported(t.owner) && changed(guid, t.sha1, ArtifactType::Texture) &&
        loadTexture(guid, t, true))
      out.textures.push_back(guid);

  std::vector<core::Guid> materials;
  for (const auto& [guid, m] : m_materials)
    if (reimported(m.owner) && changed(guid, m.sha1, ArtifactType::Material))
      materials.push_back(guid);
  for (const core::Guid& guid : materials) {
    assets::MaterialDesc desc;
    std::string sha1;
    if (!loadMaterialDesc(guid, desc, sha1)) continue;
    MaterialEntry& m = m_materials.at(guid);
    // New textures first: one used before and after stays resident.
    acquireMaterialTextures(desc);
    releaseMaterialTextures(m.desc);
    m.desc = desc;
    m.sha1 = sha1;
    out.materials.push_back(guid);
  }

  for (auto& [guid, m] : m_meshes) {
    if (!reimported(m.owner) || !changed(guid, m.sha1, ArtifactType::Mesh)) continue;
    auto dag = std::make_unique<geometry::ClusterDag>();
    std::string log;
    if (!geometry::readKGeo(util::pathToUtf8(m_db->artifactPath(guid)), *dag, log)) {
      KENSHIN_ERROR("[assets] mesh reload {}: {}", guid.toString(), log);
      continue;
    }
    const auto r = m_db->resolve(guid);
    m.dag = std::move(dag);
    m.file = m_db->artifactPath(guid);
    m.materials = r->artifact->materials;
    m.sha1 = r->artifact->sha1;
    ++m_loads;
    out.meshes.push_back(guid);
  }
  for (auto* v : {&out.textures, &out.materials, &out.meshes}) std::sort(v->begin(), v->end());
  return out;
}

u32 AssetManager::references(const core::Guid& guid) const {
  if (const auto it = m_textures.find(guid); it != m_textures.end()) return it->second.refs;
  if (const auto it = m_materials.find(guid); it != m_materials.end()) return it->second.refs;
  if (const auto it = m_meshes.find(guid); it != m_meshes.end()) return it->second.refs;
  return 0;
}

usize AssetManager::residentTextures() const { return m_textures.size(); }
usize AssetManager::residentMaterials() const { return m_materials.size(); }
usize AssetManager::residentMeshes() const { return m_meshes.size(); }

}  // namespace manager
KENSHIN_NS_END
