#include "kenshin/renderer/WorldRenderer.h"

#include <algorithm>
#include <unordered_map>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/ecs/Components.h"
#include "kenshin/ecs/World.h"
#include "kenshin/geometry/PagePacker.h"
#include "kenshin/manager/AssetManager.h"
#include "kenshin/renderer/StreamingManager.h"
#include "kenshin/renderer/VirtualGeometryScene.h"
#include "kenshin/util/File.h"

KENSHIN_NS_BEGIN
namespace renderer {

using geometry::ClusterDag;
using DagMaterial = ClusterDag::DagMaterial;

/// One built scene and the references it holds.
struct WorldRenderer::Build {
  std::unique_ptr<VirtualGeometryScene> scene;
  std::unique_ptr<StreamingManager> streamer;
  std::unique_ptr<SceneRenderer> renderer;
  std::vector<core::Guid> meshes;     // acquired, one reference each
  std::vector<core::Guid> materials;  // acquired, one reference per entry
  /// Material record -> the material asset it was made from.
  struct Record {
    u32 index = 0;
    core::Guid material;
  };
  std::vector<Record> records;
  glm::vec4 bounds{0.0f, 0.0f, 0.0f, -1.0f};
  u32 instances = 0;
};

namespace {

/// The GPU record of a material asset (textures must be resident).
DagMaterial recordOf(const assets::MaterialDesc& m, const manager::AssetManager& assets) {
  DagMaterial d;
  for (int c = 0; c < 3; ++c) {
    d.kd[c] = m.baseColor[c];
    d.emissive[c] = m.emissive[c];
  }
  d.roughness = m.roughness;
  d.metallic = m.metallic;
  const auto ref = [&](const assets::MaterialTextureSlot& s) {
    return s.valid() ? assets.textureRef(s.texture, s.sampler) : backend::kNoTexture;
  };
  d.texRef = ref(m.baseColorTexture);
  d.mrTexRef = ref(m.metallicRoughnessTexture);
  d.nrmTexRef = ref(m.normalTexture);
  d.emTexRef = ref(m.emissiveTexture);
  // AO rides the MR map's R channel (ORM packing) - only then.
  d.occlusion = m.occlusionTexture.valid() &&
                        m.occlusionTexture.texture == m.metallicRoughnessTexture.texture
                    ? m.occlusionStrength
                    : 0.0f;
  return d;
}

glm::vec4 unionSphere(glm::vec4 a, const glm::vec4& b) {
  if (b.w < 0.0f) return a;
  if (a.w < 0.0f) return b;
  const f32 d = glm::distance(glm::vec3(a), glm::vec3(b));
  if (d + b.w <= a.w) return a;
  if (d + a.w <= b.w) return b;
  const f32 r = (a.w + b.w + d) * 0.5f;
  const glm::vec3 dir = d > 0.0f ? (glm::vec3(b) - glm::vec3(a)) / d : glm::vec3(0.0f);
  return glm::vec4(glm::vec3(a) + dir * (r - a.w), r);
}

}  // namespace

WorldRenderer::WorldRenderer() = default;
WorldRenderer::~WorldRenderer() { shutdown(); }

bool WorldRenderer::init(backend::IRenderBackend& backend, manager::ShaderManager& shaders,
                         manager::AssetManager& assets, const WorldRendererOptions& options) {
  m_backend = &backend;
  m_shaders = &shaders;
  m_assets = &assets;
  m_options = options;
  return true;
}

std::unique_ptr<WorldRenderer::Build> WorldRenderer::makeBuild(const ecs::World& world,
                                                               std::string& error) {
  auto build = std::make_unique<Build>();
  manager::AssetManager& assets = *m_assets;
  assets::AssetDatabase& db = assets.database();

  // ---- the instances, in hierarchy pre-order (deterministic)
  struct Item {
    u32 mesh;  // index into build->meshes
    glm::mat4 model;
    core::Guid material;  // override (null = the mesh's slots)
    bool castShadows;
  };
  std::vector<Item> items;
  std::unordered_map<core::Guid, u32> meshIndex;
  std::vector<const ClusterDag*> dags;
  usize notes[3] = {};  // translucent, separate AO, alpha mask
  for (const ecs::Entity e : world.preOrder()) {
    const ecs::MeshRenderer* mr = world.tryGet<ecs::MeshRenderer>(e);
    if (mr == nullptr || !mr->visible || mr->mesh.isNull()) continue;
    auto it = meshIndex.find(mr->mesh);
    if (it == meshIndex.end()) {
      const ClusterDag* dag = assets.acquireMesh(mr->mesh);
      if (dag == nullptr) {
        KENSHIN_WARN("[world] {}: mesh {} unavailable, entity skipped", world.get<ecs::Name>(e).value,
                     mr->mesh.toString());
        meshIndex.emplace(mr->mesh, 0xFFFFFFFFu);
        continue;
      }
      build->meshes.push_back(mr->mesh);
      dags.push_back(dag);
      it = meshIndex.emplace(mr->mesh, u32(build->meshes.size() - 1)).first;
      // The translucent part (BLEND) of the same source mesh.
      if (const auto r = db.resolve(mr->mesh);
          r && db.resolve(assets::AssetDatabase::subAssetGuid(r->asset->guid,
                                                              r->artifact->key + "#translucent")))
        ++notes[0];
    }
    if (it->second == 0xFFFFFFFFu) continue;
    items.push_back({it->second, world.computeWorldMatrix(e), mr->material, mr->castShadows});
  }
  if (items.empty()) {
    error = "no visible MeshRenderer with a loadable mesh";
    // The references taken so far go back with the build.
    return build;
  }

  // ---- materials: a mesh's slot table, and override blocks
  const auto materialRecord = [&](const core::Guid& g, DagMaterial& out) {
    const assets::MaterialDesc* m = assets.acquireMaterial(g);
    if (m == nullptr) return false;
    build->materials.push_back(g);
    out = recordOf(*m, assets);
    if (m->occlusionTexture.valid() && m->occlusionTexture.texture != m->metallicRoughnessTexture.texture)
      ++notes[1];
    if (m->alphaMode == assets::MaterialDesc::AlphaMode::Mask) ++notes[2];
    return true;
  };
  build->scene = std::make_unique<VirtualGeometryScene>();
  VirtualGeometryScene& scene = *build->scene;
  scene.init(*m_backend);
  std::vector<u32> meshIds(dags.size(), 0xFFFFFFFFu);
  for (usize i = 0; i < dags.size(); ++i) {
    const ClusterDag& dag = *dags[i];
    if (!dag.paged) {
      KENSHIN_WARN("[world] mesh {} is not paged; skipped", build->meshes[i].toString());
      continue;
    }
    const std::vector<core::Guid> slots = assets.meshMaterials(build->meshes[i]);
    std::vector<DagMaterial> mats = dag.materials;
    std::vector<Build::Record> recs;
    const u32 recordBase = scene.materialCount();  // addMeshPaged appends the slots here
    for (usize s = 0; s < mats.size() && s < slots.size(); ++s)
      if (!slots[s].isNull() && materialRecord(slots[s], mats[s]))
        recs.push_back({recordBase + u32(s), slots[s]});
    const u32 id = scene.addMeshPaged(dag, mats);
    if (id == 0xFFFFFFFFu) {
      KENSHIN_WARN("[world] mesh {} rejected by the scene (rebuild its asset)",
                   build->meshes[i].toString());
      // Its material references are already counted in `materials`;
      // they go back with the build. Records keep the table aligned.
      continue;
    }
    meshIds[i] = id;
    build->records.insert(build->records.end(), recs.begin(), recs.end());
  }
  // Override blocks, shared per (mesh, material).
  std::vector<VgInstance> vgInsts;
  std::unordered_map<core::Guid, std::vector<std::pair<u32, u32>>> overrideBase;  // -> (mesh, base)
  for (const Item& it : items) {
    if (meshIds[it.mesh] == 0xFFFFFFFFu) continue;
    VgInstance vi;
    vi.model = it.model;
    vi.meshId = meshIds[it.mesh];
    if (!it.material.isNull()) {
      auto& bases = overrideBase[it.material];
      const auto found = std::find_if(bases.begin(), bases.end(),
                                      [&](const auto& p) { return p.first == it.mesh; });
      if (found != bases.end()) {
        vi.materialBase = found->second;
      } else {
        DagMaterial rec;
        if (materialRecord(it.material, rec)) {
          const usize n = std::max<usize>(1, dags[it.mesh]->materials.size());
          const std::vector<DagMaterial> block(n, rec);
          vi.materialBase = scene.addMaterials(block);
          for (u32 s = 0; s < n; ++s) build->records.push_back({vi.materialBase + s, it.material});
          bases.emplace_back(it.mesh, vi.materialBase);
        } else {
          KENSHIN_WARN("[world] override material {} unavailable; the mesh's own slots are used",
                       it.material.toString());
          bases.emplace_back(it.mesh, 0xFFFFFFFFu);
        }
      }
    }
    vgInsts.push_back(vi);
  }
  if (vgInsts.empty()) {
    error = "no mesh could be added to the scene";
    return build;
  }

  const u32 slotCap = m_options.streamBudgetBytes > 0
                          ? u32(u64(m_options.streamBudgetBytes) / geometry::kPageSlotSize)
                          : 0xFFFFFFFFu;
  scene.flushPools(slotCap);
  scene.setInstances(vgInsts);

  // ---- the page streamer: every mesh streams from its cooked file
  build->streamer = std::make_unique<StreamingManager>();
  if (!build->streamer->init(*m_backend, scene, m_options.streamBudgetBytes)) {
    error = "page streamer init failed";
    build->streamer.reset();
    return build;
  }
  for (usize i = 0; i < dags.size(); ++i)
    if (meshIds[i] != 0xFFFFFFFFu)
      build->streamer->addAsset(meshIds[i], util::pathToUtf8(assets.meshFile(build->meshes[i])));
  if (!build->streamer->warmup())
    KENSHIN_WARN("[stream] warmup incomplete - early frames may be sparse");

  // ---- the renderer's description
  SceneRendererDesc desc;
  desc.scene = &scene;
  desc.streamer = build->streamer.get();
  desc.decodeParams.kd = glm::vec3(1.0f);  // material assets carry real factors
  const usize meshCount = scene.meshes().size();
  desc.imposterDepth.assign(meshCount, {});
  desc.imposterTri.assign(meshCount, {});
  desc.imposterRootCluster.assign(meshCount, 0xFFFFFFFFu);
  for (usize i = 0; i < dags.size(); ++i) {
    const u32 id = meshIds[i];
    if (id == 0xFFFFFFFFu || dags[i]->imposterDepth.empty() ||
        dags[i]->imposterCluster == 0xFFFFFFFFu)
      continue;
    desc.imposterDepth[id] = dags[i]->imposterDepth;
    desc.imposterTri[id] = dags[i]->imposterTri;
    desc.imposterRootCluster[id] = dags[i]->imposterCluster;
  }
  for (const VgMeshRange& m : scene.meshes()) desc.meshRanges.push_back({m.clusterBegin, m.clusterEnd});
  usize v = 0;
  for (const Item& it : items) {
    if (meshIds[it.mesh] == 0xFFFFFFFFu) continue;
    const VgInstance& vi = vgInsts[v++];
    const VgMeshRange& m = scene.meshes()[vi.meshId];
    CullGpuInstance inst;
    inst.model = vi.model;
    inst.meshId = vi.meshId;
    // 0 receiver-only, 1 caster, 2 caster with an imposter atlas.
    inst.pad[1] = !it.castShadows ? 0u : desc.imposterDepth[vi.meshId].empty() ? 1u : 2u;
    const f32 scale = std::max({glm::length(glm::vec3(vi.model[0])), glm::length(glm::vec3(vi.model[1])),
                                glm::length(glm::vec3(vi.model[2]))});
    inst.bounds = glm::vec4(glm::vec3(vi.model * glm::vec4(glm::vec3(m.boundsSphere), 1.0f)),
                            m.boundsSphere.w * scale);
    desc.instanceBounds.push_back(inst.bounds);
    build->bounds = unionSphere(build->bounds, inst.bounds);
    desc.instances.push_back(inst);
  }
  build->instances = u32(desc.instances.size());
  build->renderer = std::make_unique<SceneRenderer>();
  if (!build->renderer->setup(*m_backend, *m_shaders, std::move(desc))) {
    error = "scene renderer setup failed";
    build->renderer.reset();
    return build;
  }
  if (m_hasEnv) build->renderer->setEnvironment(m_env);

  if (notes[0]) KENSHIN_WARN("[world] {} translucent sub-mesh(es) not drawn yet (BLEND, 1.4)", notes[0]);
  if (notes[1]) KENSHIN_WARN("[world] {} material(s) with a separate occlusion map: AO ignored", notes[1]);
  if (notes[2]) KENSHIN_WARN("[world] {} alpha-mask material(s): cutoff ignored", notes[2]);
  KENSHIN_INFO("[world] built: {} instances, {} meshes, {} material records, {} streamed pages",
               build->instances, meshCount, scene.materialCount(), scene.pageCount());
  return build;
}

void WorldRenderer::destroy(std::unique_ptr<Build>& build) {
  if (!build) return;
  if (build->renderer) build->renderer->shutdown();
  if (build->streamer) build->streamer->shutdown();
  if (build->scene) build->scene->onShutdown();
  for (const core::Guid& g : build->materials) m_assets->releaseMaterial(g);
  for (const core::Guid& g : build->meshes) m_assets->releaseMesh(g);
  build.reset();
}

bool WorldRenderer::build(const ecs::World& world, std::string* error) {
  if (m_assets == nullptr) {
    if (error) *error = "not initialized";
    return false;
  }
  std::string why;
  std::unique_ptr<Build> next = makeBuild(world, why);
  const bool ok = next && next->renderer != nullptr;
  if (!ok) destroy(next);
  destroy(m_build);  // after the new build took its references
  m_build = std::move(next);
  m_filesClosed = false;
  if (!ok) {
    KENSHIN_ERROR("[world] build failed: {}", why);
    if (error) *error = why;
  }
  return ok;
}

void WorldRenderer::clear() {
  destroy(m_build);
  m_filesClosed = false;
}

void WorldRenderer::closeFiles() {
  if (m_build && m_build->streamer && !m_filesClosed) {
    m_build->streamer->shutdown();
    m_filesClosed = true;
  }
}

usize WorldRenderer::updateMaterials(const std::vector<core::Guid>& materials) {
  if (!m_build || !m_build->scene) return 0;
  std::vector<std::pair<u32, DagMaterial>> changes;
  for (const Build::Record& r : m_build->records) {
    if (std::find(materials.begin(), materials.end(), r.material) == materials.end()) continue;
    // Resident: this build holds a reference. acquire + release reads
    // the current contents without changing the count.
    const assets::MaterialDesc* m = m_assets->acquireMaterial(r.material);
    if (m == nullptr) continue;
    changes.emplace_back(r.index, recordOf(*m, *m_assets));
    m_assets->releaseMaterial(r.material);
  }
  if (!changes.empty() && !m_build->scene->updateMaterials(changes)) {
    KENSHIN_ERROR("[world] material table update failed");
    return 0;
  }
  return changes.size();
}

bool WorldRenderer::usesMesh(const core::Guid& mesh) const {
  return m_build && std::find(m_build->meshes.begin(), m_build->meshes.end(), mesh) !=
                        m_build->meshes.end();
}

void WorldRenderer::setEnvironment(const EnvironmentProbe& probe) {
  m_env = probe;
  m_hasEnv = true;
  if (m_build && m_build->renderer) m_build->renderer->setEnvironment(probe);
}

void WorldRenderer::render(const SceneRenderer::FrameInputs& in, const SceneRenderSettings& s) {
  if (ready()) m_build->renderer->render(in, s);
}

void WorldRenderer::afterEndFrame() {
  if (ready()) m_build->renderer->afterEndFrame();
}

void WorldRenderer::logStats() {
  if (ready()) m_build->renderer->logStats();
}

const SceneRenderStats* WorldRenderer::stats() const {
  return m_build && m_build->renderer ? &m_build->renderer->stats() : nullptr;
}

StreamingManager* WorldRenderer::streamer() const {
  return m_build ? m_build->streamer.get() : nullptr;
}

VirtualGeometryScene* WorldRenderer::scene() const {
  return m_build ? m_build->scene.get() : nullptr;
}

glm::vec4 WorldRenderer::bounds() const {
  return m_build ? m_build->bounds : glm::vec4(0.0f, 0.0f, 0.0f, -1.0f);
}

u32 WorldRenderer::instanceCount() const { return m_build ? m_build->instances : 0; }
u32 WorldRenderer::meshCount() const {
  return m_build && m_build->scene ? u32(m_build->scene->meshes().size()) : 0;
}
u32 WorldRenderer::materialRecordCount() const {
  return m_build && m_build->scene ? m_build->scene->materialCount() : 0;
}

void WorldRenderer::shutdown() {
  if (m_assets) destroy(m_build);
  m_assets = nullptr;
  m_backend = nullptr;
  m_shaders = nullptr;
}

}  // namespace renderer
KENSHIN_NS_END
