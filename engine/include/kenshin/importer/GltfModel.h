#pragma once

// ── glTF model import (roadmap 1.3b) ────────────────────────────
// A glTF file becomes a set of sub-assets plus a prefab:
//   * textures  - one per (image, colour role): base colour/emissive
//                 are sRGB colour, metallic-roughness/occlusion linear
//                 data, normal maps normal maps (the same image used in
//                 two roles yields two textures);
//   * materials - metallic-roughness parameters with every texture slot
//                 and its sampler;
//   * meshes    - one per glTF mesh, mesh-local, material slots,
//                 conditioned (MeshConditioning.h) with thresholds
//                 relative to the mesh's own size; BLEND triangles
//                 split into a translucent part;
//   * a prefab  - the node tree (1.2 format) with MeshRenderer entities.
// Every sub-asset has a stable KEY ("mesh/0", "material/2",
// "texture/1.srgb", "node/5"); the asset database (1.3c) derives Guids
// from them, so a re-import keeps every reference.

#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "kenshin/assets/GltfScene.h"
#include "kenshin/assets/TextureCooker.h"
#include "kenshin/core/Guid.h"
#include "kenshin/ecs/Reflection.h"
#include "kenshin/importer/MeshConditioning.h"
#include "kenshin/serialization/Json.h"

KENSHIN_NS_BEGIN
namespace importer {

struct ImportedTexture {
  std::string key;  // "texture/<image>.<srgb|linear|normal>"
  std::string name;
  int image = -1;   // into ImportedModel::scene.images
  assets::TextureSettings settings;
};

struct ImportedTextureSlot {
  std::string texture;  // texture key, empty = none
  backend::SamplerDesc sampler;
  bool valid() const { return !texture.empty(); }
};

struct ImportedMaterial {
  std::string key;  // "material/<index>"
  std::string name;
  assets::GltfSceneMaterial source;  // factors, alpha mode, flags
  ImportedTextureSlot baseColor, metallicRoughness, normal, occlusion, emissive;
};

struct ImportedMesh {
  std::string key;  // "mesh/<index>"
  std::string name;
  assets::MeshData mesh;  // opaque part, mesh-local
  /// Per slot: material key, empty = the default material.
  std::vector<std::string> slotMaterials;
  TranslucentMesh translucent;  // BLEND triangles (material id = slot)
};

struct ModelImportOptions {
  bool splitTranslucent = true;  // BLEND triangles leave the opaque mesh
  bool conditionLayers = true;   // BLEND sink + layer separation
  bool doubleSided = true;       // back-face copies for doubleSided materials
};

struct ImportedModel {
  std::string name;  // the file stem
  assets::GltfScene scene;  // nodes (hierarchy, local transforms), images
  std::vector<ImportedTexture> textures;
  std::vector<ImportedMaterial> materials;
  std::vector<ImportedMesh> meshes;  // index-aligned with scene.meshes
  std::vector<std::string> notes;    // the loader's plus the importer's

  const ImportedTexture* findTexture(std::string_view key) const;
};

bool importGltfModel(const std::string& path, const ModelImportOptions& options,
                     ImportedModel& out, std::string& log);

/// Sub-asset key -> Guid (the asset database's derivation; tests may
/// pass any injective function).
using GuidOf = std::function<core::Guid(std::string_view key)>;

/// The prefab document (1.2 "kenshin.prefab"): one root entity named
/// after the model (identity transform, key "root") holding the glTF
/// root nodes in order; each node keeps its name and local TRS (key
/// "node/<index>"), and nodes with a mesh get a MeshRenderer whose mesh
/// is guidOf(mesh key). nullopt-equivalent: an empty Json + `error` on
/// failure (a Guid collision from guidOf).
serialization::Json modelPrefab(const ImportedModel& model, const GuidOf& guidOf,
                                const ecs::ComponentRegistry& registry, std::string* error);

}  // namespace importer
KENSHIN_NS_END
