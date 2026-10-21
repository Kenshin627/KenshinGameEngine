#pragma once

// ── Mesh data (S2.1) ─────────────────────────────────────────────
// SOA layout: positions/normals/uvs in flat float arrays + u32 indices.
// Single material (v1). Conventions per staged-plan 0.5: RH, Y-up, meters,
// CCW front faces.

#include <glm/glm.hpp>
#include <string>
#include <vector>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace assets {

/// GPU-side mesh identifier: the currency of the draw stream. MeshPool
/// issues them (upload), DrawItem carries them, the renderer layer draws.
using MeshHandle = u32;
inline constexpr MeshHandle kInvalidMeshHandle = 0xFFFFFFFFu;

struct MeshData {
  std::vector<f32> positions;  // 3 per vertex
  std::vector<f32> normals;    // 3 per vertex
  std::vector<f32> uvs;        // 2 per vertex
  std::vector<u32> indices;
  /// Per-triangle material id (multi-material sources - the glTF
  /// loader tags every primitive's triangles; empty = single material
  /// 0, the OBJ/procedural case). The DAG assigns each cluster the
  /// majority material of its vertices.
  std::vector<u32> triMaterial;

  struct Material {
    glm::vec3 ka{0.2f, 0.2f, 0.2f};
    // White = the multiplicative identity: the builder folds this into
    // the v7 material record and the decode multiplies it onto the
    // global studio kd, so the default must be a no-op (0.8 would dim
    // every default-material asset 20%).
    glm::vec3 kd{1.0f, 1.0f, 1.0f};
    std::string diffuseTexture;  // path or empty
    f32 roughness = 0.5f;        // glTF pbr (GltfLoader fills it; the
    f32 metallic = 0.0f;         // builder folds it into the v7 table)
  };
  Material material;

  u32 vertexCount() const { return static_cast<u32>(positions.size() / 3); }
  u32 triangleCount() const { return static_cast<u32>(indices.size() / 3); }

  /// Size consistency check.
  bool valid() const {
    return positions.size() % 3 == 0 &&
           (normals.empty() || normals.size() == positions.size()) &&
           (uvs.empty() || uvs.size() / 2 == positions.size() / 3) &&
           indices.size() % 3 == 0;
  }

  void computeBounds(glm::vec3& outMin, glm::vec3& outMax) const {
    outMin = glm::vec3(FLT_MAX);
    outMax = glm::vec3(-FLT_MAX);
    for (usize i = 0; i + 2 < positions.size(); i += 3) {
      for (int c = 0; c < 3; ++c) {
        const f32 v = positions[i + c];
        outMin[c] = glm::min(outMin[c], v);
        outMax[c] = glm::max(outMax[c], v);
      }
    }
    if (positions.empty()) {
      outMin = outMax = glm::vec3(0.0f);
    }
  }
};

}  // namespace assets

KENSHIN_NS_END
