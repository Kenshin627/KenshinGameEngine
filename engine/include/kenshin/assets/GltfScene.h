#pragma once

// ── glTF 2.0 as a scene graph (roadmap 1.3b) ────────────────────
// loadGltf (GltfLoader.h) flattens a file into ONE world-space mesh for
// the self-contained .kgeo path. loadGltfScene keeps the structure the
// asset system needs:
//   * the node hierarchy of the default scene, local transforms in the
//     engine frame (glTF's Y-up is conjugated into Z-up: every local
//     matrix becomes R * M * R^-1 and vertices R * v, so world space
//     matches loadGltf exactly);
//   * one mesh per glTF mesh, primitives merged, in MESH-LOCAL space,
//     with per-triangle material slots (several nodes may instance it);
//   * full metallic-roughness materials with every texture reference,
//     its sampler (wrap modes, filtering) and UV set;
//   * images as their ENCODED bytes (decoded and cooked by the texture
//     pipeline - nothing is re-encoded or resampled).
// Unsupported features (other UV sets, KHR_texture_transform, vertex
// colours, skins, animations, cameras, lights) are reported in `notes`.

#include <string>
#include <vector>

#include "kenshin/Common.h"
#include "kenshin/assets/MeshData.h"
#include "kenshin/backend/TextureFormat.h"

KENSHIN_NS_BEGIN
namespace assets {

struct GltfTextureRef {
  int image = -1;  // index into GltfScene::images, -1 = none
  backend::SamplerDesc sampler;
  u32 texCoord = 0;  // UV set the file asks for (the engine reads set 0)
  bool valid() const { return image >= 0; }
};

struct GltfSceneMaterial {
  enum class AlphaMode : u8 { Opaque, Mask, Blend };
  std::string name;
  glm::vec4 baseColor{1.0f};
  f32 metallic = 1.0f;
  f32 roughness = 1.0f;
  glm::vec3 emissive{0.0f};  // emissiveFactor x KHR_materials_emissive_strength
  AlphaMode alphaMode = AlphaMode::Opaque;
  f32 alphaCutoff = 0.5f;
  bool doubleSided = false;
  f32 normalScale = 1.0f;
  f32 occlusionStrength = 1.0f;
  GltfTextureRef baseColorTexture;          // sRGB colour
  GltfTextureRef metallicRoughnessTexture;  // linear: G roughness, B metallic
  GltfTextureRef normalTexture;             // tangent-space normal map
  GltfTextureRef occlusionTexture;          // linear: R occlusion
  GltfTextureRef emissiveTexture;           // sRGB colour
};

struct GltfSceneImage {
  std::string name;  // file-stem suggestion (image name / uri stem)
  std::string mimeType;
  std::vector<u8> encoded;  // PNG / JPEG / ... as stored in the file
};

struct GltfSceneMesh {
  std::string name;
  /// Mesh-local, engine frame. triMaterial = slot per triangle when the
  /// mesh has more than one slot (MeshData's "empty = slot 0" rule).
  MeshData mesh;
  /// Per slot: index into GltfScene::materials, -1 = the default
  /// material (a primitive without one).
  std::vector<int> slotMaterials;
};

struct GltfSceneNode {
  std::string name;
  glm::mat4 local{1.0f};  // relative to the parent, engine frame
  int parent = -1;
  int mesh = -1;  // index into GltfScene::meshes
  std::vector<int> children;
};

struct GltfScene {
  /// Pre-order (parents before children, siblings in file order).
  std::vector<GltfSceneNode> nodes;
  std::vector<int> roots;
  std::vector<GltfSceneMesh> meshes;  // first-use order
  std::vector<GltfSceneMaterial> materials;
  std::vector<GltfSceneImage> images;
  std::vector<std::string> notes;  // ignored/unsupported content
};

/// Never throws; false with a reason in `log` for unreadable or
/// malformed files (hostile-input hardened like loadGltf).
bool loadGltfScene(const std::string& path, GltfScene& out, std::string& log);

}  // namespace assets
KENSHIN_NS_END
