#include "KgeoScene.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <filesystem>

#include "kenshin/Engine.h"
#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/geometry/PagePacker.h"
#include "kenshin/manager/TextureManager.h"
#include "kenshin/renderer/VirtualGeometryScene.h"

namespace demo {

namespace {

namespace kr = kenshin::renderer;
using kenshin::u32;

/// Resolve a material texture reference: bare file names resolve
/// against the asset's directory (the mtllib rule), absolute / existing
/// relative paths stay as given.
std::filesystem::path resolveTexture(const std::filesystem::path& dir,
                                     const std::string& path) {
  return path.find(':') == std::string::npos &&
                 !std::filesystem::exists(dir / path)
             ? std::filesystem::path(path)
             : (dir / path);
}

}  // namespace

KgeoSceneResult buildKgeoScene(const Args& args, kenshin::Engine& engine,
                               kenshin::manager::TextureManager& textureManager,
                               KgeoScene& out) {
  std::string kgeoLog;
  if (!loadKgeo(args.kgeo, out.preview, kgeoLog)) {
    KENSHIN_WARN("[kgeo] load failed: {} (continuing with --mesh)", kgeoLog);
    return KgeoSceneResult::LoadFailed;
  }
  KgeoPreview& kgeoPreview = out.preview;
  kr::SceneRendererDesc& desc = out.desc;

  // The dag goes to the GPU-resident virtual geometry pools (cluster /
  // group tables + vertex/index pools or the page slot pool), CRC
  // round-trip verified at upload.
  auto& vgScene = engine.registerSubsystem<kr::VirtualGeometryScene>();
  desc.scene = &vgScene;
  // Multi-material assets carry their own base-color factors in the
  // material table - the studio kd (0.75) would wash them. Single-
  // material assets keep the studio kd (their material factor is 1).
  if (kgeoPreview.dag.materials.size() > 1)
    desc.decodeParams.kd = glm::vec3(1.0f);
  // v3+ assets (tables-only dag) take the paged branch - the vertex pools
  // are never uploaded, the pages stream via the StreamingManager.
  const bool pagedAsset = kgeoPreview.dag.paged;
  // Material texture references -> texture-array LAYERS before the scene
  // bakes them into its GPU table. A failed load keeps the layer at -1.
  if (pagedAsset && !kgeoPreview.dag.materials.empty()) {
    const auto kgeoDir = std::filesystem::path(args.kgeo).parent_path();
    for (auto& mat : kgeoPreview.dag.materials) {
      for (int slot = 0; slot < 3; ++slot) {
        std::string& path = slot == 0   ? mat.texPath
                            : slot == 1 ? mat.mrPath
                                        : mat.nrmPath;
        kenshin::f32& layerF = slot == 0   ? mat.texLayer
                               : slot == 1 ? mat.mrTexLayer
                                           : mat.nrmTexLayer;
        if (path.empty()) continue;
        const u32 layer =
            textureManager.load(resolveTexture(kgeoDir, path).string());
        if (layer != kenshin::manager::TextureManager::kInvalidLayer)
          layerF = static_cast<kenshin::f32>(layer);
      }
    }
    desc.albedoTexture = textureManager.textureArray();
  }
  const u32 vgMeshId = pagedAsset ? vgScene.addMeshPaged(kgeoPreview.dag)
                                  : vgScene.addMesh(kgeoPreview.dag);
  // Pre-v5 paged assets are rejected (their empty v5 tables would
  // silently drop every dispatch - black viewport, no warning).
  if (vgMeshId == 0xFFFFFFFFu) {
    KENSHIN_ERROR(
        "[main] asset rejected: rebuild {} with the current kgeo_builder",
        args.kgeo);
    return KgeoSceneResult::Fatal;
  }
  // An optional ground-plane asset (a second mesh; receiver-only).
  u32 groundMeshId = 0xFFFFFFFFu;
  bool groundPaged = false;
  if (!args.ground.empty()) {
    KgeoPreview groundPreview;
    std::string groundLog;
    if (!loadKgeo(args.ground, groundPreview, groundLog)) {
      KENSHIN_ERROR("[main] ground load failed: {}", groundLog);
      return KgeoSceneResult::Fatal;
    }
    if (groundPreview.dag.paged) {
      groundMeshId = vgScene.addMeshPaged(groundPreview.dag);
      groundPaged = true;
    } else {
      groundMeshId = vgScene.addMesh(groundPreview.dag);
    }
    if (groundMeshId == 0xFFFFFFFFu) {
      KENSHIN_ERROR("[main] ground asset rejected: rebuild {}", args.ground);
      return KgeoSceneResult::Fatal;
    }
    KENSHIN_INFO("[main] ground mesh loaded: {} clusters",
                 groundPreview.dag.clusters.size());
  }
  // --mesh: extra assets (the real-content gallery). Materials resolve
  // like the primary's; the scene's per-mesh material base rebases them.
  std::vector<u32> extraMeshIds;
  std::vector<std::string> extraPaths;
  std::vector<KgeoPreview> extraPreviews;
  for (const auto& path : args.meshes) {
    KgeoPreview mp;
    std::string mlog;
    if (!loadKgeo(path, mp, mlog)) {
      KENSHIN_ERROR("[main] --mesh load failed: {} ({})", path, mlog);
      return KgeoSceneResult::Fatal;
    }
    if (!mp.dag.paged) {
      KENSHIN_ERROR("[main] --mesh {} is not a paged asset; rebuild it", path);
      return KgeoSceneResult::Fatal;
    }
    if (!mp.dag.materials.empty()) {
      const auto dir = std::filesystem::path(path).parent_path();
      for (auto& mat : mp.dag.materials) {
        if (mat.texPath.empty()) continue;
        const u32 layer =
            textureManager.load(resolveTexture(dir, mat.texPath).string());
        if (layer != kenshin::manager::TextureManager::kInvalidLayer)
          mat.texLayer = static_cast<kenshin::f32>(layer);
      }
      desc.albedoTexture = textureManager.textureArray();
    }
    const u32 id = vgScene.addMeshPaged(mp.dag);
    if (id == 0xFFFFFFFFu) {
      KENSHIN_ERROR(
          "[main] --mesh {} rejected (rebuild with the current kgeo_builder)",
          path);
      return KgeoSceneResult::Fatal;
    }
    extraMeshIds.push_back(id);
    extraPaths.push_back(path);
    KENSHIN_INFO("[main] extra mesh {}: {} ({} clusters)", extraMeshIds.size() - 1,
                 path, mp.dag.clusters.size());
    extraPreviews.push_back(std::move(mp));
  }
  // Imposter atlases, indexed by MESH id (the ground sits between the
  // primary and the gallery - sizing these by the gallery count wrote
  // past the end with --ground + an imposter-capable --mesh).
  const kenshin::usize meshCount = vgScene.meshes().size();
  desc.imposterDepth.assign(meshCount, {});
  desc.imposterTri.assign(meshCount, {});
  desc.imposterRootCluster.assign(meshCount, 0xFFFFFFFFu);
  {
    const auto stage = [&](u32 id, const KgeoPreview& pv) {
      if (id >= meshCount || pv.dag.imposterDepth.empty()) return;
      if (pv.dag.imposterCluster == 0xFFFFFFFFu) return;
      desc.imposterDepth[id] = pv.dag.imposterDepth;
      desc.imposterTri[id] = pv.dag.imposterTri;
      desc.imposterRootCluster[id] = pv.dag.imposterCluster;
    };
    stage(0, kgeoPreview);
    for (u32 e = 0; e < extraMeshIds.size(); ++e)
      stage(extraMeshIds[e], extraPreviews[e]);
  }
  const u32 slotCap =
      pagedAsset && args.streamBudgetMb > 0
          ? u32((kenshin::u64(args.streamBudgetMb) * 1024 * 1024) /
                kenshin::geometry::kPageSlotSize)
          : 0xFFFFFFFFu;
  vgScene.flushPools(slotCap);
  vgScene.setInstances({kr::VgInstance{glm::mat4(1.0f), vgMeshId}});
  // The page streamer (paged assets only).
  if (pagedAsset) {
    out.streamer = std::make_unique<kr::StreamingManager>();
    if (!out.streamer->init(
            engine.get<kenshin::backend::IRenderBackend>(), vgScene,
            args.streamBudgetMb > 0
                ? kenshin::usize(args.streamBudgetMb) * 1024 * 1024
                : 0)) {
      KENSHIN_ERROR(
          "[stream] manager init failed - a paged asset cannot render "
          "without it, exiting");
      return KgeoSceneResult::Fatal;
    }
    out.streamer->addAsset(vgMeshId, args.kgeo);
    // The ground's (and gallery's) pages stream from THEIR files.
    if (groundMeshId != 0xFFFFFFFFu && groundPaged)
      out.streamer->addAsset(groundMeshId, args.ground);
    for (u32 e = 0; e < extraMeshIds.size(); ++e)
      out.streamer->addAsset(extraMeshIds[e], extraPaths[e]);
    // First frame coarse: the coarsest pages resident before the loop
    // starts - no black flash while the async thread spins up.
    if (!out.streamer->warmup())
      KENSHIN_WARN("[stream] warmup incomplete - early frames may be sparse");
    desc.streamer = out.streamer.get();
  }

  // ---- instance layout
  {
    const kr::VgMeshRange& mesh = vgScene.meshes()[vgMeshId];
    auto& chainInstances = desc.instances;
    auto& realBounds = desc.instanceBounds;
    // meshId + bounds come from the INSTANCE'S mesh.
    const auto makeInstFor = [&](u32 id, const glm::mat4& model,
                                 u32 casts = 1u) {
      // casts: 0 receiver-only, 1 caster, 2 imposterable
      const kr::VgMeshRange& m = vgScene.meshes()[id];
      kr::CullGpuInstance inst;
      inst.model = model;
      inst.meshId = id;
      inst.pad[1] = casts;  // 72: castShadows (receiver-only = 0)
      // WORLD bounds sphere (the cull input): center transformed, radius
      // scaled by the max column length - VirtualGeometryScene's
      // derivation.
      const kenshin::f32 scale =
          std::max({glm::length(model[0]), glm::length(model[1]),
                    glm::length(model[2])});
      inst.bounds = glm::vec4(
          glm::vec3(model * glm::vec4(m.boundsSphere.x, m.boundsSphere.y,
                                      m.boundsSphere.z, 1.0f)),
          m.boundsSphere.w * scale);
      return inst;
    };
    const auto makeInst = [&](const glm::mat4& model) {
      return makeInstFor(vgMeshId, model);
    };
    if (args.wall) {
      // Big-occluder scene: one 6x-scaled mesh at the origin as a wall,
      // 24 unit meshes in a 6-wide x 4-deep block BEHIND it.
      std::vector<kr::VgInstance> vgInsts;
      {
        const glm::mat4 m = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, 0)) *
                            glm::scale(glm::mat4(1.0f), glm::vec3(6.0f));
        chainInstances.push_back(makeInst(m));
        vgInsts.push_back({m, vgMeshId});
      }
      for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 6; ++col) {
          const glm::vec3 o(3.0f + 2.25f * row, 0.0f, (col - 2.5f) * 2.25f);
          const glm::mat4 m = glm::translate(glm::mat4(1.0f), o);
          chainInstances.push_back(makeInst(m));
          vgInsts.push_back({m, vgMeshId});
        }
      vgScene.setInstances(vgInsts);
    } else if (args.grid > 0) {
      // Occlusion stress carpet: DEPTH-major 8-deep lines along +x.
      const int n = std::clamp(args.grid, 1, 256);
      const int depth = std::max(1, std::min(8, (n + 3) / 4));
      const kenshin::f32 step = 2.25f;
      std::vector<kr::VgInstance> vgInsts;
      for (int i = 0; i < n; ++i) {
        const glm::vec3 o((i % depth) * step, 0.0f, (i / depth) * step);
        const glm::mat4 m = glm::translate(glm::mat4(1.0f), o);
        chainInstances.push_back(makeInst(m));
        vgInsts.push_back({m, vgMeshId});
      }
      vgScene.setInstances(vgInsts);
    } else if (args.multi) {
      // Near/mid/far along +x (one threshold, different LOD levels);
      // --dense pulls them close (front-back occluder stack).
      const std::vector<glm::vec3> offsets =
          args.dense ? std::vector<glm::vec3>{{0.0f, 0.0f, 0.0f},
                                              {2.2f, 0.0f, 0.0f},
                                              {4.4f, 0.0f, 0.0f}}
                     : std::vector<glm::vec3>{{0.0f, 0.0f, 0.0f},
                                              {7.0f, 0.0f, 0.0f},
                                              {59.0f, 0.0f, 0.0f}};
      std::vector<kr::VgInstance> vgInsts;
      for (const glm::vec3& o : offsets) {
        const glm::mat4 m = glm::translate(glm::mat4(1.0f), o);
        chainInstances.push_back(makeInst(m));
        vgInsts.push_back({m, vgMeshId});
      }
      if (groundMeshId != 0xFFFFFFFFu) {
        const glm::mat4 gm =
            glm::translate(glm::mat4(1.0f), glm::vec3(5.0f, 0.0f, -1.5f));
        chainInstances.push_back(makeInstFor(groundMeshId, gm));
        vgInsts.push_back({gm, groundMeshId});
      }
      vgScene.setInstances(vgInsts);
    } else {
      realBounds.clear();
      chainInstances = {makeInst(glm::mat4(1.0f))};
      realBounds.push_back(chainInstances.back().bounds);
      std::vector<kr::VgInstance> vgInsts = {
          kr::VgInstance{glm::mat4(1.0f), vgMeshId}};
      if (groundMeshId != 0xFFFFFFFFu) {
        // Z-up: the ground is a z=0 plane; drop it below the asset.
        const glm::mat4 gm =
            glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -1.5f));
        chainInstances.push_back(makeInstFor(groundMeshId, gm, 0u));
        vgInsts.push_back({gm, groundMeshId});
      }
      // --mesh gallery: spaced along -x, sitting on the same ground.
      for (u32 e = 0; e < extraMeshIds.size(); ++e) {
        const glm::vec3 o(-2.5f - 3.0f * float(e), 0.0f, 0.0f);
        const glm::mat4 m = glm::translate(glm::mat4(1.0f), o);
        const bool hasAtlas = e < extraPreviews.size() &&
                              !extraPreviews[e].dag.imposterDepth.empty();
        auto inst = makeInstFor(extraMeshIds[e], m, hasAtlas ? 2u : 1u);
        realBounds.push_back(inst.bounds);
        chainInstances.push_back(std::move(inst));
        vgInsts.push_back({m, extraMeshIds[e]});
      }
      vgScene.setInstances(vgInsts);
    }
    for (u32 i = 0; i < chainInstances.size(); ++i)
      desc.meshRanges.push_back({mesh.clusterBegin, mesh.clusterEnd});
  }
  desc.translucentMeshPath =
      std::filesystem::path(args.kgeo).replace_extension(".tmesh").string();

  KENSHIN_INFO("[kgeo] loaded: {} clusters / {} groups / {} levels",
               kgeoPreview.dag.clusters.size(), kgeoPreview.dag.groups.size(),
               [&] {
                 kenshin::i32 m = 0;
                 for (auto& g : kgeoPreview.dag.groups) m = std::max(m, g.lodLevel);
                 return m + 1;
               }());
  if (kgeoPreview.dag.hierarchyRoot < kgeoPreview.dag.hierarchyNodes.size()) {
    const auto& hn = kgeoPreview.dag.hierarchyNodes[kgeoPreview.dag.hierarchyRoot];
    KENSHIN_INFO(
        "[kgeo] hierarchy {} nodes, root {} aabb min({:.2f},{:.2f},{:.2f}) "
        "max({:.2f},{:.2f},{:.2f}) err {:.4f}",
        kgeoPreview.dag.hierarchyNodes.size(), kgeoPreview.dag.hierarchyRoot,
        hn.aabbMin[0], hn.aabbMin[1], hn.aabbMin[2], hn.aabbMax[0],
        hn.aabbMax[1], hn.aabbMax[2], hn.maxSimplifyError);
  }
  return KgeoSceneResult::Ok;
}

}  // namespace demo
