#include "kenshin/importer/GltfModel.h"

#include <algorithm>
#include <filesystem>

#include "kenshin/core/Log.h"
#include "kenshin/ecs/World.h"
#include "kenshin/scene/SceneSerializer.h"
#include "kenshin/util/File.h"

KENSHIN_NS_BEGIN
namespace importer {

using assets::GltfSceneMaterial;
using assets::TextureSettings;

const ImportedTexture* ImportedModel::findTexture(std::string_view key) const {
  for (const ImportedTexture& t : textures)
    if (t.key == key) return &t;
  return nullptr;
}

namespace {

enum class Role { Colour, Data, Normal };

std::string textureKey(int image, Role role) {
  static constexpr const char* kSuffix[] = {"srgb", "linear", "normal"};
  return "texture/" + std::to_string(image) + "." + kSuffix[static_cast<int>(role)];
}

/// The texture for (image, role), created on first use.
ImportedTextureSlot slot(ImportedModel& model, const assets::GltfTextureRef& ref, Role role) {
  ImportedTextureSlot s;
  if (!ref.valid()) return s;
  s.texture = textureKey(ref.image, role);
  s.sampler = ref.sampler;
  if (!model.findTexture(s.texture)) {
    ImportedTexture t;
    t.key = s.texture;
    t.image = ref.image;
    const std::string& base = model.scene.images[usize(ref.image)].name;
    switch (role) {
      case Role::Colour:
        t.name = base;
        break;
      case Role::Data:
        t.name = base + " (linear)";
        t.settings.colorSpace = TextureSettings::ColorSpace::Linear;
        break;
      case Role::Normal:
        t.name = base + " (normal)";
        t.settings.normalMap = true;
        break;
    }
    model.textures.push_back(std::move(t));
  }
  return s;
}

/// The conditioning passes' per-slot view of a mesh's materials.
std::vector<assets::GltfMaterial> slotView(const ImportedModel& model,
                                           const assets::GltfSceneMesh& mesh) {
  std::vector<assets::GltfMaterial> v;
  for (const int m : mesh.slotMaterials) {
    assets::GltfMaterial gm;
    if (m >= 0) {
      const GltfSceneMaterial& src = model.scene.materials[usize(m)];
      gm.alpha = src.alphaMode == GltfSceneMaterial::AlphaMode::Blend ? src.baseColor.a : 1.0f;
      gm.doubleSided = src.doubleSided;
      gm.kd = glm::vec3(src.baseColor);
      gm.roughness = src.roughness;
      gm.metallic = src.metallic;
    }
    v.push_back(gm);
  }
  return v;
}

f32 longestSide(const assets::MeshData& mesh) {
  glm::vec3 mn(1e30f), mx(-1e30f);
  for (usize i = 0; i + 2 < mesh.positions.size(); i += 3) {
    const glm::vec3 p(mesh.positions[i], mesh.positions[i + 1], mesh.positions[i + 2]);
    mn = glm::min(mn, p);
    mx = glm::max(mx, p);
  }
  const glm::vec3 e = mx - mn;
  return std::max({e.x, e.y, e.z, 0.0f});
}

}  // namespace

bool importGltfModel(const std::string& path, const ModelImportOptions& options,
                     ImportedModel& out, std::string& log) {
  ImportedModel model;
  if (!assets::loadGltfScene(path, model.scene, log)) return false;
  model.name = util::pathToUtf8(util::pathFromUtf8(path).stem());
  model.notes = model.scene.notes;

  for (usize i = 0; i < model.scene.materials.size(); ++i) {
    const GltfSceneMaterial& src = model.scene.materials[i];
    ImportedMaterial m;
    m.key = "material/" + std::to_string(i);
    m.name = src.name;
    m.source = src;
    m.baseColor = slot(model, src.baseColorTexture, Role::Colour);
    m.metallicRoughness = slot(model, src.metallicRoughnessTexture, Role::Data);
    m.normal = slot(model, src.normalTexture, Role::Normal);
    m.occlusion = slot(model, src.occlusionTexture, Role::Data);
    m.emissive = slot(model, src.emissiveTexture, Role::Colour);
    model.materials.push_back(std::move(m));
  }

  for (usize i = 0; i < model.scene.meshes.size(); ++i) {
    const assets::GltfSceneMesh& src = model.scene.meshes[i];
    ImportedMesh m;
    m.key = "mesh/" + std::to_string(i);
    m.name = src.name;
    m.mesh = src.mesh;
    for (const int mat : src.slotMaterials)
      m.slotMaterials.push_back(mat >= 0 ? model.materials[usize(mat)].key : std::string());
    // Conditioning thresholds are fractions of the mesh's own size (the
    // legacy builder's --fit semantics), so units do not matter.
    const std::vector<assets::GltfMaterial> slots = slotView(model, src);
    const f32 size = longestSide(m.mesh);
    if (options.splitTranslucent) splitTranslucent(m.mesh, slots, m.translucent);
    if (options.conditionLayers) {
      sinkHuggingBlend(m.mesh, slots, size);
      separateLayers(m.mesh, size);
    }
    if (options.doubleSided) appendBackFaces(m.mesh, slots, size);
    if (m.mesh.indices.empty())
      model.notes.push_back("mesh '" + m.name + "' is fully translucent (no opaque part)");
    model.meshes.push_back(std::move(m));
  }
  KENSHIN_INFO("[import] {}: {} nodes, {} meshes, {} materials, {} textures ({} images)",
               model.name, model.scene.nodes.size(), model.meshes.size(),
               model.materials.size(), model.textures.size(), model.scene.images.size());
  for (const std::string& n : model.notes) KENSHIN_INFO("[import] {}: note: {}", model.name, n);
  out = std::move(model);
  return true;
}

serialization::Json modelPrefab(const ImportedModel& model, const GuidOf& guidOf,
                                const ecs::ComponentRegistry& registry, std::string* error) {
  ecs::World world;
  auto fail = [&](const std::string& why) {
    if (error) *error = why;
    return serialization::Json();
  };
  const ecs::Entity root = world.createWithGuid(guidOf("root"), model.name);
  if (root == ecs::kNullEntity) return fail("guidOf(\"root\") returned a null guid");
  std::vector<ecs::Entity> entityOf(model.scene.nodes.size(), ecs::kNullEntity);
  // scene.nodes is pre-order: parents are created before children.
  for (usize i = 0; i < model.scene.nodes.size(); ++i) {
    const assets::GltfSceneNode& n = model.scene.nodes[i];
    const ecs::Entity parent = n.parent < 0 ? root : entityOf[usize(n.parent)];
    const ecs::Entity e =
        world.createWithGuid(guidOf("node/" + std::to_string(i)), n.name, parent);
    if (e == ecs::kNullEntity) return fail("guid collision for node/" + std::to_string(i));
    entityOf[i] = e;
    ecs::Transform t;
    if (!ecs::decomposeTransform(n.local, t)) {
      // Singular (a zero-scale node hides its subtree): keep the
      // translation and the zero axes rather than inventing a pose.
      t.position = glm::vec3(n.local[3]);
      t.scale = glm::vec3(glm::length(glm::vec3(n.local[0])), glm::length(glm::vec3(n.local[1])),
                          glm::length(glm::vec3(n.local[2])));
    }
    world.get<ecs::Transform>(e) = t;
    if (n.mesh >= 0) {
      ecs::MeshRenderer& mr = world.add<ecs::MeshRenderer>(e);
      mr.mesh = guidOf(model.meshes[usize(n.mesh)].key);
    }
  }
  return scene::prefabToJson(world, root, registry);
}

}  // namespace importer
KENSHIN_NS_END
