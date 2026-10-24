#include "AssetImporters.h"

#include <algorithm>
#include <format>
#include <sstream>

#include "kenshin/assets/Image.h"
#include "kenshin/assets/Ktx2.h"
#include "kenshin/assets/Material.h"
#include "kenshin/assets/ObjLoader.h"
#include "kenshin/assets/TextureCooker.h"
#include "kenshin/ecs/World.h"
#include "kenshin/importer/GltfModel.h"
#include "kenshin/importer/MeshCooker.h"
#include "kenshin/scene/SceneSerializer.h"
#include "kenshin/util/File.h"

KENSHIN_NS_BEGIN
namespace assets::detail {

namespace fs = std::filesystem;
using serialization::Json;

u32 baseImporterVersion(AssetType type) {
  switch (type) {
    case AssetType::Texture: return 1;
    case AssetType::Model: return 3;  // 2: mesh slot materials; 3: one-primitive conditioning
    case AssetType::Material: return 1;
    case AssetType::Scene: return 1;
    case AssetType::Prefab: return 1;
  }
  return 1;
}

namespace {

bool fail(std::string& error, std::string why) {
  error = std::move(why);
  return false;
}

/// Reads `in[key]` as a bool / int into `out` (key absent = keep).
bool boolKey(const Json& in, const char* key, bool& out, std::string& error) {
  const auto it = in.find(key);
  if (it == in.end()) return true;
  if (!it->is_boolean()) return fail(error, std::format("settings.{} must be a boolean", key));
  out = it->get<bool>();
  return true;
}

bool checkKeys(const Json& in, std::initializer_list<const char*> allowed, std::string& error) {
  if (!in.is_object()) return fail(error, "settings must be an object");
  for (auto it = in.begin(); it != in.end(); ++it) {
    bool ok = false;
    for (const char* k : allowed) ok |= it.key() == k;
    if (!ok) {
      std::string list;
      for (const char* k : allowed) list += (list.empty() ? "" : ", ") + std::string(k);
      return fail(error, std::format("unknown setting \"{}\" (settings: {})", it.key(),
                                     list.empty() ? "none" : list));
    }
  }
  return true;
}

bool readBytes(const fs::path& p, std::string& out, std::string& error) {
  if (!util::readFile(p, out)) return fail(error, "cannot read " + util::pathToUtf8(p));
  return true;
}

bool writeText(const fs::path& p, const std::string& text, std::string& error) {
  std::string why;
  if (!util::writeFileAtomic(p, text, &why)) return fail(error, why);
  return true;
}

std::string stemOf(const fs::path& p) { return util::pathToUtf8(p.stem()); }

/// "texture/0.srgb" -> "texture_0_srgb" (cache file names).
std::string fileStem(std::string_view key) {
  std::string s(key);
  for (char& c : s)
    if (c == '/' || c == '.' || c == '#') c = '_';
  return s;
}

bool cookTextureFile(const Image& image, const TextureSettings& settings, const fs::path& out,
                     std::string& error) {
  CookedTexture cooked;
  if (!cookTexture(image, settings, cooked, error)) return false;
  std::string why;
  if (!saveKtx2File(out, cooked, &why)) return fail(error, why);
  return true;
}

// ── texture ────────────────────────────────────────────────────
bool importTexture(ImportContext& ctx, std::string& error) {
  const Json& s = *ctx.settings;
  const fs::path out = ctx.outDir / "texture.ktx2";
  std::string ext = util::pathToUtf8(ctx.source.extension());
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  if (ext == ".ktx2") {
    // Already cooked: validate and copy (settings cannot re-cook it).
    CookedTexture t;
    if (!loadKtx2File(ctx.source, t, error)) return false;
    std::string bytes;
    if (!readBytes(ctx.source, bytes, error) || !writeText(out, bytes, error)) return false;
    ctx.notes.push_back("KTX2 source used as is (cook settings do not apply)");
  } else {
    Image image;
    if (!loadImageFile(ctx.source, image, error)) return false;
    TextureSettings ts;
    ts.colorSpace = s["color_space"] == "linear" ? TextureSettings::ColorSpace::Linear
                                                 : TextureSettings::ColorSpace::Srgb;
    ts.normalMap = s["normal_map"].get<bool>();
    ts.mipmaps = s["mipmaps"].get<bool>();
    ts.maxSize = s["max_size"].get<u32>();
    if (!cookTextureFile(image, ts, out, error)) return false;
  }
  ctx.artifacts.push_back({"main", ctx.guid, ArtifactType::Texture, stemOf(ctx.source), "texture.ktx2"});
  return true;
}

// ── model ──────────────────────────────────────────────────────
MaterialDesc materialOf(const importer::ImportedMaterial& m, const core::Guid& asset) {
  const auto& src = m.source;
  MaterialDesc d;
  d.baseColor = src.baseColor;
  d.metallic = src.metallic;
  d.roughness = src.roughness;
  d.emissive = src.emissive;
  d.alphaMode = static_cast<MaterialDesc::AlphaMode>(src.alphaMode);
  d.alphaCutoff = src.alphaCutoff;
  d.doubleSided = src.doubleSided;
  d.normalScale = src.normalScale;
  d.occlusionStrength = src.occlusionStrength;
  auto slot = [&](const importer::ImportedTextureSlot& s) {
    MaterialTextureSlot t;
    if (s.valid()) {
      t.texture = AssetDatabase::subAssetGuid(asset, s.texture);
      t.sampler = s.sampler;
    }
    return t;
  };
  d.baseColorTexture = slot(m.baseColor);
  d.metallicRoughnessTexture = slot(m.metallicRoughness);
  d.normalTexture = slot(m.normal);
  d.occlusionTexture = slot(m.occlusion);
  d.emissiveTexture = slot(m.emissive);
  return d;
}

/// Writes every sub-asset of an imported model.
bool cookModel(ImportContext& ctx, importer::ImportedModel& model, std::string& error) {
  const Json& s = *ctx.settings;
  const core::Guid& asset = ctx.guid;
  auto sub = [&](const std::string& key) { return AssetDatabase::subAssetGuid(asset, key); };

  for (const importer::ImportedTexture& t : model.textures) {
    const GltfSceneImage& src = model.scene.images[usize(t.image)];
    Image image;
    std::string why;
    if (!decodeImage(src.encoded, image, why)) {
      ctx.notes.push_back(t.key + " skipped: " + why);
      continue;
    }
    const std::string file = fileStem(t.key) + ".ktx2";
    if (!cookTextureFile(image, t.settings, ctx.outDir / file, error)) return false;
    ctx.artifacts.push_back({t.key, sub(t.key), ArtifactType::Texture, t.name, file});
  }
  for (const importer::ImportedMaterial& m : model.materials) {
    const std::string file = fileStem(m.key) + ".kmat";
    if (!writeText(ctx.outDir / file, saveMaterial(materialOf(m, asset)), error)) return false;
    ctx.artifacts.push_back({m.key, sub(m.key), ArtifactType::Material, m.name, file});
  }
  importer::CookOptions cook;
  cook.compressPages = s["compress_pages"].get<bool>();
  for (importer::ImportedMesh& m : model.meshes) {
    // The per-slot factor table keeps the .kgeo self-describing; the
    // renderer binds the material ASSETS (and their textures) by slot.
    std::vector<GltfMaterial> slots;
    std::vector<core::Guid> slotGuids;
    for (const std::string& key : m.slotMaterials) {
      slotGuids.push_back(key.empty() ? core::Guid() : sub(key));  // no material: null
      GltfMaterial gm;
      for (const importer::ImportedMaterial& im : model.materials)
        if (im.key == key) {
          gm.kd = glm::vec3(im.source.baseColor);
          gm.alpha = im.source.alphaMode == GltfSceneMaterial::AlphaMode::Blend
                         ? im.source.baseColor.a
                         : 1.0f;
          gm.roughness = im.source.roughness;
          gm.metallic = im.source.metallic;
          gm.emissive = im.source.emissive;
        }
      slots.push_back(gm);
    }
    if (!m.mesh.indices.empty()) {
      geometry::ClusterDag dag;
      if (!importer::buildMeshDag(m.mesh, dag, error)) return fail(error, m.name + ": " + error);
      for (const GltfMaterial& gm : slots) {
        geometry::ClusterDag::DagMaterial dm;
        for (int c = 0; c < 3; ++c) {
          dm.kd[c] = gm.kd[c];
          dm.emissive[c] = gm.emissive[c];
        }
        dm.roughness = gm.roughness;
        dm.metallic = gm.metallic;
        dag.materials.push_back(dm);
      }
      if (dag.materials.empty()) dag.materials.emplace_back();
      if (dag.clusterMaterial.size() != dag.clusters.size())
        dag.clusterMaterial.assign(dag.clusters.size(), 0);
      const std::string file = fileStem(m.key) + ".kgeo";
      if (!importer::writeCookedMesh(ctx.outDir / file, dag, cook, error))
        return fail(error, m.name + ": " + error);
      ctx.artifacts.push_back({m.key, sub(m.key), ArtifactType::Mesh, m.name, file, slotGuids});
    }
    if (m.translucent.triangleCount() > 0) {
      const std::string key = m.key + "#translucent";
      const std::string file = fileStem(key) + ".tmesh";
      if (!importer::writeTranslucentMesh(ctx.outDir / file, slots, m.translucent, error))
        return false;
      ctx.artifacts.push_back({key, sub(key), ArtifactType::TranslucentMesh, m.name, file, slotGuids});
    }
  }
  // The prefab only uses built-in components.
  ecs::ComponentRegistry builtin;
  ecs::registerBuiltinComponents(builtin);
  std::string why;
  const Json prefab = importer::modelPrefab(
      model, [&](std::string_view key) { return sub(std::string(key)); }, builtin, &why);
  if (prefab.is_null()) return fail(error, "prefab: " + why);
  if (!writeText(ctx.outDir / "prefab.kprefab", serialization::writeJson(prefab), error)) return false;
  ctx.artifacts.push_back({"prefab", sub("prefab"), ArtifactType::Prefab, model.name, "prefab.kprefab"});
  for (const std::string& n : model.notes) ctx.notes.push_back(n);
  return true;
}

importer::ModelImportOptions modelOptions(const Json& s) {
  importer::ModelImportOptions o;
  o.splitTranslucent = s["split_translucent"].get<bool>();
  o.conditionLayers = s["condition_layers"].get<bool>();
  o.doubleSided = s["double_sided"].get<bool>();
  return o;
}

/// OBJ: one mesh, its MTL material (factors + map_Kd), one node.
bool loadObjModel(ImportContext& ctx, importer::ImportedModel& model, std::string& error) {
  MeshData mesh;
  std::string log;
  if (!loadObj(util::pathToUtf8(ctx.source), mesh, log)) return fail(error, log);
  // Dependencies: the mtllib files the OBJ names (the loader resolves
  // them next to the OBJ) and the base-colour image.
  std::string text;
  if (readBytes(ctx.source, text, error)) {
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
      if (line.rfind("mtllib", 0) != 0) continue;
      std::string name = line.substr(6);
      while (!name.empty() && (name.front() == ' ' || name.front() == '\t')) name.erase(0, 1);
      while (!name.empty() && (name.back() == '\r' || name.back() == ' ')) name.pop_back();
      if (!name.empty()) ctx.dependencies.push_back(ctx.source.parent_path() / util::pathFromUtf8(name));
    }
  }
  model.name = stemOf(ctx.source);
  importer::ImportedMaterial mat;
  mat.key = "material/0";
  mat.name = model.name;
  mat.source.baseColor = glm::vec4(mesh.material.kd, 1.0f);
  mat.source.roughness = mesh.material.roughness;
  mat.source.metallic = mesh.material.metallic;
  if (!mesh.material.diffuseTexture.empty()) {
    fs::path tex = util::pathFromUtf8(mesh.material.diffuseTexture);
    if (tex.is_relative()) tex = ctx.source.parent_path() / tex;
    GltfSceneImage img;
    std::string bytes;
    if (util::readFile(tex, bytes)) {
      img.name = stemOf(tex);
      img.encoded.assign(bytes.begin(), bytes.end());
      model.scene.images.push_back(std::move(img));
      importer::ImportedTexture t;
      t.key = "texture/0.srgb";
      t.name = model.scene.images.back().name;
      t.image = 0;
      model.textures.push_back(t);
      mat.baseColor.texture = t.key;
      ctx.dependencies.push_back(tex);
    } else {
      ctx.notes.push_back("map_Kd " + mesh.material.diffuseTexture + " not found");
    }
  }
  model.materials.push_back(mat);
  importer::ImportedMesh m;
  m.key = "mesh/0";
  m.name = model.name;
  m.mesh = std::move(mesh);
  m.slotMaterials = {mat.key};
  model.meshes.push_back(std::move(m));
  GltfSceneNode node;
  node.name = model.name;
  node.mesh = 0;
  model.scene.nodes.push_back(node);
  model.scene.roots = {0};
  return true;
}

bool importModel(ImportContext& ctx, std::string& error) {
  std::string ext = util::pathToUtf8(ctx.source.extension());
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  importer::ImportedModel model;
  if (ext == ".obj") {
    if (!loadObjModel(ctx, model, error)) return false;
  } else {
    std::string log;
    if (!importer::importGltfModel(util::pathToUtf8(ctx.source), modelOptions(*ctx.settings), model,
                                   log))
      return fail(error, log);
    for (const std::string& f : model.scene.externalFiles)
      ctx.dependencies.push_back(util::pathFromUtf8(f));
  }
  return cookModel(ctx, model, error);
}

// ── material / scene / prefab ──────────────────────────────────
bool importMaterial(ImportContext& ctx, std::string& error) {
  std::string text;
  if (!readBytes(ctx.source, text, error)) return false;
  MaterialDesc m;
  if (!loadMaterial(text, m, &error)) return false;
  if (!writeText(ctx.outDir / "material.kmat", saveMaterial(m), error)) return false;
  ctx.artifacts.push_back({"main", ctx.guid, ArtifactType::Material, stemOf(ctx.source), "material.kmat"});
  return true;
}

bool importSceneOrPrefab(ImportContext& ctx, std::string& error) {
  if (!ctx.registry) return fail(error, "validating scenes and prefabs needs a component registry");
  std::string text;
  if (!readBytes(ctx.source, text, error)) return false;
  const bool isScene = ctx.type == AssetType::Scene;
  ecs::World world;
  if (isScene) {
    if (!scene::loadScene(text, world, *ctx.registry, &error)) return false;
  } else if (scene::instantiatePrefab(text, world, *ctx.registry, ecs::kNullEntity, &error) ==
             ecs::kNullEntity) {
    return false;
  }
  const std::string file = isScene ? "scene.kscene" : "prefab.kprefab";
  if (!writeText(ctx.outDir / file, text, error)) return false;
  ctx.artifacts.push_back({"main", ctx.guid, isScene ? ArtifactType::Scene : ArtifactType::Prefab,
                           stemOf(ctx.source), file});
  return true;
}

}  // namespace

bool normalizeSettings(AssetType type, const Json& in, Json& out, std::string& error) {
  out = Json::object();
  switch (type) {
    case AssetType::Texture: {
      if (!checkKeys(in, {"color_space", "normal_map", "mipmaps", "max_size"}, error)) return false;
      std::string cs = "srgb";
      if (const auto it = in.find("color_space"); it != in.end()) {
        if (!it->is_string() || (*it != "srgb" && *it != "linear"))
          return fail(error, "settings.color_space must be \"srgb\" or \"linear\"");
        cs = it->get<std::string>();
      }
      bool normal = false, mips = true;
      if (!boolKey(in, "normal_map", normal, error) || !boolKey(in, "mipmaps", mips, error)) return false;
      u32 maxSize = 0;
      if (const auto it = in.find("max_size"); it != in.end()) {
        if (!it->is_number_integer() || it->get<i64>() < 0 || it->get<i64>() > 16384)
          return fail(error, "settings.max_size must be an integer in [0, 16384] (0 = no limit)");
        maxSize = it->get<u32>();
      }
      out["color_space"] = normal ? "linear" : cs;  // normal maps are data
      out["normal_map"] = normal;
      out["mipmaps"] = mips;
      out["max_size"] = maxSize;
      return true;
    }
    case AssetType::Model: {
      if (!checkKeys(in, {"split_translucent", "condition_layers", "double_sided", "compress_pages"},
                     error))
        return false;
      bool split = true, layers = true, sided = true, compress = true;
      if (!boolKey(in, "split_translucent", split, error) ||
          !boolKey(in, "condition_layers", layers, error) ||
          !boolKey(in, "double_sided", sided, error) ||
          !boolKey(in, "compress_pages", compress, error))
        return false;
      out["split_translucent"] = split;
      out["condition_layers"] = layers;
      out["double_sided"] = sided;
      out["compress_pages"] = compress;
      return true;
    }
    case AssetType::Material:
    case AssetType::Scene:
    case AssetType::Prefab:
      return checkKeys(in, {}, error);
  }
  return fail(error, "unknown asset type");
}

bool runImporter(ImportContext& ctx, std::string& error) {
  switch (ctx.type) {
    case AssetType::Texture: return importTexture(ctx, error);
    case AssetType::Model: return importModel(ctx, error);
    case AssetType::Material: return importMaterial(ctx, error);
    case AssetType::Scene:
    case AssetType::Prefab: return importSceneOrPrefab(ctx, error);
  }
  return fail(error, "unknown asset type");
}

}  // namespace assets::detail
KENSHIN_NS_END
