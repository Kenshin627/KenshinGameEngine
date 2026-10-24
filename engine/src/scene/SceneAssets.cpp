#include "kenshin/scene/SceneAssets.h"

#include "kenshin/scene/SceneSerializer.h"
#include "kenshin/util/File.h"

KENSHIN_NS_BEGIN
namespace scene {

core::Guid worldDocumentOf(const assets::AssetDatabase& db, const core::Guid& asset) {
  // A model's own Guid names no artifact (only its sub-assets do): its
  // document is the prefab sub-asset.
  if (const assets::AssetRecord* rec = db.find(asset);
      rec && rec->type == assets::AssetType::Model) {
    for (const assets::Artifact& a : rec->artifacts)
      if (a.type == assets::ArtifactType::Prefab) return a.guid;
    return {};
  }
  const auto r = db.resolve(asset);
  if (!r) return {};
  switch (r->artifact->type) {
    case assets::ArtifactType::Scene:
    case assets::ArtifactType::Prefab: return r->artifact->guid;
    default: return {};
  }
}

bool loadWorldAsset(const assets::AssetDatabase& db, const core::Guid& asset,
                    const ecs::ComponentRegistry& registry, ecs::World& world,
                    std::string* error) {
  const auto fail = [&](const std::string& why) {
    if (error) *error = why;
    return false;
  };
  const core::Guid doc = worldDocumentOf(db, asset);
  if (doc.isNull()) return fail(asset.toString() + " is not an imported scene, prefab or model");
  const auto r = db.resolve(doc);
  const std::filesystem::path file = db.artifactPath(doc);
  if (r->artifact->type == assets::ArtifactType::Scene)
    return loadSceneFile(file, world, registry, error);
  std::string text;
  if (!util::readFile(file, text)) return fail("cannot read " + util::pathToUtf8(file));
  ecs::World fresh;
  if (instantiatePrefab(text, fresh, registry, ecs::kNullEntity, error) == ecs::kNullEntity)
    return false;
  world = std::move(fresh);
  return true;
}

}  // namespace scene
KENSHIN_NS_END
