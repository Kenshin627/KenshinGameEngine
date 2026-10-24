#include "ProjectScene.h"

#include "kenshin/Engine.h"
#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/manager/ShaderManager.h"
#include "kenshin/scene/SceneAssets.h"
#include "kenshin/util/File.h"

namespace demo {

namespace {

namespace ka = kenshin::assets;

/// The sha1 of the document artifact (scene / prefab) behind `asset`.
std::string documentSha1(const ka::AssetDatabase& db, const kenshin::core::Guid& asset) {
  const kenshin::core::Guid doc = kenshin::scene::worldDocumentOf(db, asset);
  const auto r = db.resolve(doc);
  return r ? r->artifact->sha1 : std::string();
}

bool loadWorld(ProjectScene& ps) {
  kenshin::ecs::World world;
  std::string error;
  if (!kenshin::scene::loadWorldAsset(ps.database, ps.document, ps.registry, world, &error)) {
    KENSHIN_ERROR("[project] scene load failed: {}", error);
    return false;
  }
  ps.world = std::move(world);
  ps.documentSha1 = documentSha1(ps.database, ps.document);
  KENSHIN_INFO("[project] scene loaded: {} entities", ps.world.size());
  return true;
}

}  // namespace

bool openProjectScene(const Args& args, kenshin::Engine& engine, ProjectScene& ps) {
  kenshin::ecs::registerBuiltinComponents(ps.registry);
  ka::AssetDatabaseOptions options;
  options.registry = &ps.registry;
  std::string error;
  ka::ScanReport scan;
  if (!ps.database.open(kenshin::util::pathFromUtf8(args.project), options, &error, &scan)) {
    KENSHIN_ERROR("[project] {}", error);
    return false;
  }
  for (const std::string& s : scan.corruptMetas) KENSHIN_WARN("[project] corrupt meta {}", s);
  const ka::ImportReport imports = ps.database.importAll();
  KENSHIN_INFO("[project] {}: {} assets, {} imported, {} current, {} failed",
               ps.database.project().name, ps.database.assets().size(), imports.imported.size(),
               imports.upToDate, imports.failed.size());

  const std::string scenePath = args.scene.empty() ? ps.database.project().startupScene : args.scene;
  if (scenePath.empty()) {
    KENSHIN_ERROR("[project] no --scene and no startup scene in the project file");
    return false;
  }
  const ka::AssetRecord* rec = ps.database.findByPath(scenePath);
  if (rec == nullptr) {
    KENSHIN_ERROR("[project] {} is not an asset of the project", scenePath);
    return false;
  }
  ps.document = rec->guid;
  if (!loadWorld(ps)) return false;

  auto& backend = engine.get<kenshin::backend::IRenderBackend>();
  kenshin::manager::AssetManagerOptions am;
  am.watch = args.watchInterval >= 0.0f;
  am.pollInterval = am.watch ? args.watchInterval : 0.0f;
  ps.assets.init(backend, ps.database, am);
  kenshin::renderer::WorldRendererOptions wr;
  wr.streamBudgetBytes = args.streamBudgetMb > 0 ? kenshin::usize(args.streamBudgetMb) << 20 : 0;
  ps.renderer.init(backend, engine.get<kenshin::manager::ShaderManager>(), ps.assets, wr);
  if (!ps.renderer.build(ps.world, &error)) return false;
  return true;
}

void syncProjectScene(ProjectScene& ps, kenshin::f32 dt) {
  if (!ps.assets.update(dt)) return;
  const kenshin::manager::ReloadPlan plan = ps.assets.planReload();
  if (plan.empty()) return;
  bool rebuild = false;
  // A streamed mesh file is about to be replaced: stop streaming first
  // (the streamer reopens it by path per page - it would read the new
  // file against the old tables - and a read in flight blocks the
  // re-import's folder swap on Windows). The rebuild restarts it.
  if (!plan.residentMeshes.empty()) {
    ps.renderer.closeFiles();
    rebuild = true;
  }
  const kenshin::manager::AssetReload r = ps.assets.applyReload(plan);
  if (documentSha1(ps.database, ps.document) != ps.documentSha1) rebuild |= loadWorld(ps);
  if (!r.meshes.empty()) rebuild = true;
  kenshin::usize records = 0;
  if (rebuild)
    ps.renderer.build(ps.world);
  else if (!r.materials.empty())
    records = ps.renderer.updateMaterials(r.materials);
  ++ps.reloads;
  KENSHIN_INFO(
      "[hotreload] #{}: {} re-imported, {} failed; {} textures in place, {} materials ({} records), "
      "{} meshes{}",
      ps.reloads, r.reimported.size(), r.failed.size(), r.textures.size(), r.materials.size(),
      records, r.meshes.size(), rebuild ? "; scene rebuilt" : "");
}

void closeProjectScene(ProjectScene& ps) {
  ps.renderer.shutdown();
  ps.assets.shutdown();
}

}  // namespace demo
