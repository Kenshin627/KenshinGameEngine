// glTF scene-graph import (roadmap step 1.3b): hierarchy, transforms
// (Y-up -> Z-up conjugation), mesh instancing, material slots, texture
// roles and samplers, notes for unsupported content, hostile input, and
// the prefab - checked against the flattening loader's world-space
// output on synthetic files and on the real test models.

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <vector>

#include "kenshin/assets/GltfLoader.h"
#include "kenshin/assets/GltfScene.h"
#include "kenshin/assets/Image.h"
#include "kenshin/ecs/Reflection.h"
#include "kenshin/ecs/World.h"
#include "kenshin/importer/GltfModel.h"
#include "kenshin/scene/SceneSerializer.h"
#include "kenshin/util/File.h"
#include "test_gltf.h"
#include "test_png.h"

using namespace kenshin;
namespace fs = std::filesystem;
using assets::GltfScene;
using assets::GltfSceneMaterial;

using testgltf::Bin;
using testgltf::ReferenceModel;
using testgltf::TempDir;

namespace {

bool approx(const glm::vec3& a, const glm::vec3& b, f32 eps = 1e-4f) {
  return glm::all(glm::lessThanEqual(glm::abs(a - b), glm::vec3(eps)));
}

bool hasNote(const std::vector<std::string>& notes, const std::string& part) {
  for (const auto& n : notes)
    if (n.find(part) != std::string::npos) return true;
  return false;
}

core::Guid keyGuid(std::string_view key) {
  // Injective for the test: FNV-1a of the key in both halves.
  u64 h = 1469598103934665603ull;
  for (const char c : key) h = (h ^ u8(c)) * 1099511628211ull;
  return core::Guid{h | 1, ~h};
}

/// World-space vertices of every mesh node (pre-order, each node's mesh
/// positions in order) - from the prefab instance's world matrices.
std::vector<glm::vec3> prefabWorldVertices(const importer::ImportedModel& model) {
  ecs::ComponentRegistry reg;
  ecs::registerBuiltinComponents(reg);
  std::string err;
  const serialization::Json doc = importer::modelPrefab(model, keyGuid, reg, &err);
  REQUIRE(err.empty());
  ecs::World w;
  const ecs::Entity root = scene::instantiatePrefabJson(doc, w, reg, ecs::kNullEntity, &err);
  INFO(err);
  REQUIRE(root != ecs::kNullEntity);
  std::vector<ecs::Entity> order = w.preOrder();  // [root, nodes...]
  REQUIRE(order.size() == model.scene.nodes.size() + 1);
  std::vector<glm::vec3> out;
  for (usize i = 0; i < model.scene.nodes.size(); ++i) {
    const int mesh = model.scene.nodes[i].mesh;
    if (mesh < 0) continue;
    const glm::mat4 world = w.computeWorldMatrix(order[i + 1]);
    const auto& pos = model.scene.meshes[usize(mesh)].mesh.positions;
    for (usize v = 0; v + 2 < pos.size(); v += 3)
      out.push_back(glm::vec3(world * glm::vec4(pos[v], pos[v + 1], pos[v + 2], 1.0f)));
  }
  return out;
}

std::vector<glm::vec3> flattenedVertices(const std::string& path) {
  assets::MeshData merged;
  std::string log;
  REQUIRE(assets::loadGltf(path, merged, nullptr, log));
  std::vector<glm::vec3> out;
  for (usize v = 0; v + 2 < merged.positions.size(); v += 3)
    out.emplace_back(merged.positions[v], merged.positions[v + 1], merged.positions[v + 2]);
  return out;
}

}  // namespace

TEST_CASE("gltf scene: hierarchy, names and transforms", "[gltf]") {
  ReferenceModel ref;
  GltfScene s;
  std::string log;
  REQUIRE(assets::loadGltfScene(ref.path, s, log));
  // The default scene only: the orphan node is not part of it.
  REQUIRE(s.nodes.size() == 4);
  CHECK(s.roots == std::vector<int>{0, 3});
  CHECK(s.nodes[0].name == "Root");
  CHECK(s.nodes[0].children == std::vector<int>{1, 2});
  CHECK(s.nodes[1].name == "Left");
  CHECK(s.nodes[1].parent == 0);
  CHECK(s.nodes[2].name == "Right");
  CHECK(s.nodes[3].name == "\xe4\xb8\xad\xe6\x96\x87 \xf0\x9f\x98\x80");  // \u escapes decoded
  // Engine frame: glTF (x, y, z) -> (x, -z, y).
  CHECK(approx(glm::vec3(s.nodes[0].local[3]), {1, -3, 2}));
  CHECK(approx(glm::vec3(s.nodes[2].local[3]), {5, 0, 0}));
  CHECK(approx(glm::vec3(s.nodes[1].local[3]), {0, 0, 1}));  // glTF +Y is engine +Z
  // Non-uniform scale on glTF Y lands on engine Z.
  CHECK(approx({glm::length(glm::vec3(s.nodes[3].local[0])),
                glm::length(glm::vec3(s.nodes[3].local[1])),
                glm::length(glm::vec3(s.nodes[3].local[2]))},
               {1, 1, 3}));
}

TEST_CASE("gltf scene: meshes, slots, instancing and attributes", "[gltf]") {
  ReferenceModel ref;
  GltfScene s;
  std::string log;
  REQUIRE(assets::loadGltfScene(ref.path, s, log));
  REQUIRE(s.meshes.size() == 2);  // Quad (instanced twice) + Tri
  CHECK(s.nodes[1].mesh == 0);
  CHECK(s.nodes[2].mesh == 0);
  CHECK(s.nodes[3].mesh == 1);
  const assets::GltfSceneMesh& quad = s.meshes[0];
  CHECK(quad.name == "Quad");
  CHECK(quad.slotMaterials == std::vector<int>{0, 1});
  CHECK(quad.mesh.triMaterial == std::vector<u32>{0, 1});
  CHECK(quad.mesh.vertexCount() == 6);
  CHECK(quad.mesh.uvs.size() == 12);
  // Mesh-local, engine frame: glTF vertex (0,1,0) -> (0,0,1); normal
  // (0,0,1) -> (0,-1,0).
  CHECK(approx({quad.mesh.positions[6], quad.mesh.positions[7], quad.mesh.positions[8]}, {0, 0, 1}));
  CHECK(approx({quad.mesh.normals[0], quad.mesh.normals[1], quad.mesh.normals[2]}, {0, -1, 0}));
  const assets::GltfSceneMesh& tri = s.meshes[1];
  CHECK(tri.slotMaterials == std::vector<int>{-1});  // no material: the default
  CHECK(tri.mesh.triMaterial.empty());
  CHECK(tri.mesh.uvs.empty());
  REQUIRE(tri.mesh.normals.size() == 9);  // computed, unit length
  CHECK(std::abs(glm::length(glm::vec3(tri.mesh.normals[0], tri.mesh.normals[1],
                                        tri.mesh.normals[2])) - 1.0f) < 1e-5f);
}

TEST_CASE("gltf scene: materials, texture roles and samplers", "[gltf]") {
  ReferenceModel ref;
  GltfScene s;
  std::string log;
  REQUIRE(assets::loadGltfScene(ref.path, s, log));
  REQUIRE(s.materials.size() == 2);
  const GltfSceneMaterial& red = s.materials[0];
  CHECK(red.name == "Red");
  CHECK(red.baseColor == glm::vec4(1, 0, 0, 1));
  CHECK(std::abs(red.metallic - 0.2f) < 1e-6f);
  CHECK(std::abs(red.roughness - 0.7f) < 1e-6f);
  CHECK(red.baseColorTexture.image == 0);
  CHECK(red.baseColorTexture.sampler.wrapU == backend::WrapMode::Clamp);
  CHECK(red.baseColorTexture.sampler.wrapV == backend::WrapMode::Mirror);
  CHECK(red.baseColorTexture.sampler.filter == backend::FilterMode::Nearest);
  CHECK(red.metallicRoughnessTexture.image == 0);
  CHECK(red.metallicRoughnessTexture.sampler == backend::SamplerDesc{});
  CHECK(red.normalTexture.image == 1);
  CHECK(red.normalTexture.texCoord == 1);
  CHECK(red.normalScale == 0.5f);
  CHECK(red.occlusionTexture.image == 0);
  CHECK(std::abs(red.occlusionStrength - 0.3f) < 1e-6f);
  const GltfSceneMaterial& glass = s.materials[1];
  CHECK(glass.alphaMode == GltfSceneMaterial::AlphaMode::Blend);
  CHECK(glass.baseColor.a == 0.25f);
  CHECK(glass.doubleSided);
  CHECK(glass.emissive == glm::vec3(2, 1, 0));  // factor x strength
  // Images keep their encoded bytes: nothing re-encoded or resampled.
  REQUIRE(s.images.size() == 2);
  CHECK(s.images[0].encoded == ref.pngA);  // via "tex%20one.png" -> "tex one.png"
  CHECK(s.images[0].name == "tex one");
  CHECK(s.images[1].encoded == ref.pngB);
  CHECK(s.images[1].mimeType == "image/png");
  assets::Image decoded;
  std::string err;
  CHECK(assets::decodeImage(s.images[0].encoded, decoded, err));
  // Unsupported content is reported, not silently dropped.
  CHECK(hasNote(s.notes, "TEXCOORD_1"));
  CHECK(hasNote(s.notes, "KHR_texture_transform"));
  CHECK(hasNote(s.notes, "COLOR_0"));
  CHECK(hasNote(s.notes, "animations"));
}

TEST_CASE("gltf model: sub-assets with stable keys", "[gltf]") {
  ReferenceModel ref;
  importer::ImportedModel m;
  std::string log;
  REQUIRE(importer::importGltfModel(ref.path, {}, m, log));
  CHECK(m.name == "model");
  // One texture per (image, role): image 0 is colour AND data.
  REQUIRE(m.textures.size() == 3);
  const importer::ImportedTexture* srgb = m.findTexture("texture/0.srgb");
  const importer::ImportedTexture* lin = m.findTexture("texture/0.linear");
  const importer::ImportedTexture* nrm = m.findTexture("texture/1.normal");
  REQUIRE(srgb);
  REQUIRE(lin);
  REQUIRE(nrm);
  CHECK(srgb->settings.colorSpace == assets::TextureSettings::ColorSpace::Srgb);
  CHECK(lin->settings.colorSpace == assets::TextureSettings::ColorSpace::Linear);
  CHECK(nrm->settings.normalMap);
  const importer::ImportedMaterial& red = m.materials[0];
  CHECK(red.key == "material/0");
  CHECK(red.baseColor.texture == "texture/0.srgb");
  CHECK(red.baseColor.sampler.filter == backend::FilterMode::Nearest);
  CHECK(red.metallicRoughness.texture == "texture/0.linear");
  CHECK(red.occlusion.texture == "texture/0.linear");
  CHECK(red.normal.texture == "texture/1.normal");
  CHECK_FALSE(red.emissive.valid());
  REQUIRE(m.meshes.size() == 2);
  const importer::ImportedMesh& quad = m.meshes[0];
  CHECK(quad.key == "mesh/0");
  CHECK(quad.slotMaterials == std::vector<std::string>{"material/0", "material/1"});
  // The BLEND (Glass) triangle left the opaque mesh.
  CHECK(quad.translucent.triangleCount() == 1);
  CHECK(quad.mesh.triangleCount() == 1);
  CHECK(m.meshes[1].slotMaterials == std::vector<std::string>{""});

  // Deterministic: a second import yields the same keys and data.
  importer::ImportedModel again;
  REQUIRE(importer::importGltfModel(ref.path, {}, again, log));
  CHECK(again.meshes[0].mesh.positions == quad.mesh.positions);
  CHECK(again.textures.size() == m.textures.size());
}

TEST_CASE("gltf model: prefab reproduces the flattened world space", "[gltf]") {
  ReferenceModel ref;
  importer::ImportedModel m;
  std::string log;
  REQUIRE(importer::importGltfModel(ref.path, {}, m, log));
  ecs::ComponentRegistry reg;
  ecs::registerBuiltinComponents(reg);
  std::string err;
  const serialization::Json doc = importer::modelPrefab(m, keyGuid, reg, &err);
  REQUIRE(err.empty());
  REQUIRE(doc["entities"].size() == 5);
  CHECK(doc["entities"][0]["name"] == "model");
  CHECK(doc["entities"][0]["guid"] == keyGuid("root").toString());
  CHECK(doc["entities"][2]["name"] == "Left");
  CHECK(doc["entities"][2]["components"]["mesh_renderer"]["mesh"] ==
        keyGuid("mesh/0").toString());
  CHECK(doc["entities"][3]["components"]["mesh_renderer"]["mesh"] ==
        keyGuid("mesh/0").toString());  // instanced
  CHECK(doc["entities"][1]["components"].contains("mesh_renderer") == false);

  // Every vertex of every instance lands where the flattening loader
  // puts it (hierarchy x conjugated locals x mesh-local vertices).
  const std::vector<glm::vec3> viaPrefab = prefabWorldVertices(m);
  const std::vector<glm::vec3> flat = flattenedVertices(ref.path);
  REQUIRE(viaPrefab.size() == flat.size());
  for (usize i = 0; i < flat.size(); ++i) {
    INFO("vertex " << i);
    CHECK(approx(viaPrefab[i], flat[i]));
  }
}

TEST_CASE("gltf scene: hostile and unsupported files", "[gltf]") {
  TempDir dir("kenshin_gltf_hostile_test");
  Bin b;
  const int pos = b.floats({0, 0, 0, 1, 0, 0, 0, 1, 0}, 3);
  dir.write("b.bin", b.data.data(), b.data.size());
  const std::string tail = R"(,
    "meshes": [{"primitives": [{"attributes": {"POSITION": )" + std::to_string(pos) + R"(}}]}],
    "buffers": [{"uri": "b.bin", "byteLength": )" + std::to_string(b.data.size()) + R"(}],
    "bufferViews": )" + Bin::join(b.views) + R"(, "accessors": )" + Bin::join(b.accessors) + "}";
  auto load = [&](const std::string& head, std::string& log) {
    dir.write("f.gltf", head + tail);
    GltfScene s;
    return assets::loadGltfScene(dir.str("f.gltf"), s, log) ? std::string("ok") : log;
  };
  std::string log;
  CHECK(load(R"({"extensionsRequired": ["KHR_draco_mesh_compression"], "nodes": [{"mesh": 0}])",
             log) == "requires unsupported extension KHR_draco_mesh_compression");
  CHECK(load(R"({"nodes": [{"mesh": 7}])", log) == "node mesh index out of range");
  CHECK(load(R"({"scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "children": [9]}])", log) ==
        "node child index out of range");
  // A cycle is a reachable-twice node: noted, not followed forever.
  {
    dir.write("f.gltf", R"({"scenes": [{"nodes": [0]}],
      "nodes": [{"mesh": 0, "children": [1]}, {"children": [0]}])" + tail);
    GltfScene s;
    REQUIRE(assets::loadGltfScene(dir.str("f.gltf"), s, log));
    CHECK(s.nodes.size() == 2);
    CHECK(hasNote(s.notes, "reachable twice"));
  }
  // 300 nested nodes.
  {
    std::string nodes = "[";
    for (int i = 0; i < 300; ++i)
      nodes += std::string(i ? "," : "") + "{\"children\": [" + std::to_string(i + 1) + "]}";
    nodes += ",{\"mesh\": 0}]";
    CHECK(load(R"({"scenes": [{"nodes": [0]}], "nodes": )" + nodes, log) ==
          "node hierarchy deeper than 256 levels");
  }
  // Image URIs may not leave the model's folder or name a scheme.
  for (const char* uri : {"../secret.png", "file:///etc/passwd", "http://x/y.png", "C:/x.png"}) {
    dir.write("f.gltf", std::string(R"({"nodes": [{"mesh": 0}],
      "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
      "textures": [{"source": 0}], "images": [{"uri": ")") + uri + R"("}])" +
                            tail.substr(0, tail.size() - 1) + "}");
    // (the mesh above has no material reference: patch one in)
    std::string text;
    REQUIRE(util::readFile(dir.path / "f.gltf", text));
    const auto at = text.find("{\"POSITION\"");
    text.insert(text.find('}', at) + 1, ", \"material\": 0");
    dir.write("f.gltf", text);
    GltfScene s;
    INFO(uri);
    REQUIRE(assets::loadGltfScene(dir.str("f.gltf"), s, log));
    CHECK(s.images.empty());
    CHECK(s.materials[0].baseColorTexture.image == -1);
    CHECK(hasNote(s.notes, "refused"));
  }
  CHECK(load(R"({"nodes": [])", log) == "no triangle meshes in the default scene");
  // Trailing content after the root value is corruption, not ignorable.
  dir.write("f.gltf", R"({"nodes": [{"mesh": 0}])" + tail + " {}");
  GltfScene trailing;
  CHECK_FALSE(assets::loadGltfScene(dir.str("f.gltf"), trailing, log));
  CHECK(log == "json: trailing data after the root value");
}

TEST_CASE("gltf model: real test models match the flattening loader", "[gltf][assets]") {
  const fs::path models = fs::path(KENSHIN_SOURCE_DIR) / "tests/assets/model";
  const char* files[] = {"vultures_helmet_mcu_version/scene.gltf",
                         "fighter_jet_pilot_helmet_f-35_-_free/scene.gltf"};
  int found = 0;
  for (const char* f : files) {
    const fs::path p = models / f;
    if (!fs::exists(p)) continue;
    ++found;
    INFO(f);
    importer::ImportedModel m;
    std::string log;
    REQUIRE(importer::importGltfModel(util::pathToUtf8(p), {}, m, log));
    CHECK(!m.meshes.empty());
    CHECK(!m.textures.empty());
    const std::vector<glm::vec3> viaPrefab = prefabWorldVertices(m);
    const std::vector<glm::vec3> flat = flattenedVertices(util::pathToUtf8(p));
    REQUIRE(viaPrefab.size() == flat.size());
    f32 worst = 0.0f, extent = 0.0f;
    for (usize i = 0; i < flat.size(); ++i) {
      worst = std::max(worst, glm::length(viaPrefab[i] - flat[i]));
      extent = std::max(extent, glm::length(flat[i]));
    }
    INFO("worst deviation " << worst << " of extent " << extent);
    CHECK(worst <= extent * 1e-5f);
  }
  if (found == 0) SKIP("tests/assets/model not present (untracked test assets)");
}
