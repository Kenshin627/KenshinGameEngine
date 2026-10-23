#pragma once

// ── glTF mesh conditioning (roadmap 1.3b) ───────────────────────
// Build-time fixes that make real-world glTF content survive the
// cluster-LOD renderer, moved out of kgeo_builder so every importer
// (CLI, asset database, editor) applies the same rules. Each pass
// documents the artifact it prevents; all log what they changed.

#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "kenshin/assets/GltfLoader.h"
#include "kenshin/assets/MeshData.h"

KENSHIN_NS_BEGIN
namespace importer {

/// Display-asset normalization: xy to the bbox centre, the bottom to
/// z = 0; `fit` > 0 also scales the longest side to `fit` units.
/// (Sketchfab/FBX scans routinely sit far from the origin.)
void recenter(assets::MeshData& mesh, f32 fit);

/// BLEND triangles leave the opaque mesh (Nanite never rasterizes
/// translucency) into a vertex-pulled triangle list: per vertex 12 f32
/// (pos xyz, material id, normal xyz, 0, uv, 0, 0).
struct TranslucentMesh {
  std::vector<f32> vertices;
  usize triangleCount() const { return vertices.size() / 36; }
};
/// Moves BLEND triangles of `mesh` into `out`; returns false (mesh
/// untouched) when there are none.
bool splitTranslucent(assets::MeshData& mesh, std::span<const assets::GltfMaterial> materials,
                      TranslucentMesh& out);
/// "KTM1" | u32 version 2 | u32 materialCount | per material 8 f32
/// (kd rgb, alpha, roughness, metallic, 0, 0) | u32 vertexCount |
/// vertices. Material ids index the same table as the .kgeo's.
bool writeTranslucentMesh(const std::filesystem::path& path,
                          std::span<const assets::GltfMaterial> materials,
                          const TranslucentMesh& mesh, std::string& log);

/// Pushes BLEND geometry that hugs an opaque surface inward along its
/// normals, so LOD error cannot flip their depth order (decided per
/// material: >= 25% of its vertices within the hug distance).
void sinkHuggingBlend(assets::MeshData& mesh, std::span<const assets::GltfMaterial> materials,
                      f32 fit);

/// Moves geometry hidden a hair behind another material's surface back
/// to a safe gap (independent per-material simplification would poke it
/// through).
void separateLayers(assets::MeshData& mesh, f32 fit);

/// glTF doubleSided: appends a reversed-winding copy on its own,
/// slightly offset vertices with flipped normals (every raster path
/// culls back faces).
void appendBackFaces(assets::MeshData& mesh, std::span<const assets::GltfMaterial> materials,
                     f32 fit);

}  // namespace importer
KENSHIN_NS_END
