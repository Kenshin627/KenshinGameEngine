#pragma once

// ── Self-contained .kgeo material tables (roadmap 1.3b) ─────────
// A .kgeo built for direct loading (KENSHINDemo --kgeo) carries its own
// material table whose textures are bare file names of 24-bit BMP side
// files written next to it - the pair travels together. The asset
// database does not use this path: there, materials and textures are
// assets of their own (see GltfModel.h).

#include <filesystem>
#include <string>
#include <vector>

#include "kenshin/assets/GltfLoader.h"
#include "kenshin/assets/MeshData.h"
#include "kenshin/geometry/ClusterDag.h"

KENSHIN_NS_BEGIN
namespace importer {

/// Command-line material overrides (< 0 / empty = keep the asset's).
struct MaterialOverrides {
  f32 kd[3] = {-1.0f, -1.0f, -1.0f};
  f32 roughness = -1.0f;
  f32 metallic = -1.0f;
  std::string albedo;  // an albedo image to reference instead
};

/// Single-material glTF: writes its base-colour image as
/// <stem>_albedo.bmp in `outDir` and points the mesh material at it.
bool writeSingleAlbedo(const assets::GltfImage& albedo, const std::string& fallbackStem,
                       const std::filesystem::path& outDir, assets::MeshData& mesh,
                       std::string& log);

/// Fills dag.materials (and a zero clusterMaterial table when the DAG
/// carried none): one entry for single-material meshes, one per used
/// glTF material otherwise, each textured slot re-encoded as a BMP in
/// `outDir`. `albedoSource` receives the image to copy next to the
/// output afterwards (single-material path), or stays empty.
bool buildKgeoMaterials(const assets::MeshData& mesh, const MaterialOverrides& overrides,
                        std::vector<assets::GltfMaterial>& gltfMaterials,
                        std::vector<assets::GltfImage>& gltfImages,
                        const std::filesystem::path& outDir, geometry::ClusterDag& dag,
                        std::string& albedoSource, std::string& log);

/// Copies `albedoSource` next to `output` (no-op when it already is).
bool copyAlbedoBesideOutput(const std::string& albedoSource, const std::filesystem::path& output,
                            std::string& log);

}  // namespace importer
KENSHIN_NS_END
