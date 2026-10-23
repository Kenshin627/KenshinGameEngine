#pragma once

// ── Mesh cooking: MeshData -> cluster DAG -> .kgeo (roadmap 1.3b) ──
// The offline half of the virtual-geometry pipeline as a library:
// cluster LOD build + DAG assembly (per-cluster material ids from the
// mesh's triangle materials), the optional far-field imposter bake, and
// the paged (v8 codec / v7 raw) or unpaged (v2) file. kgeo_builder and
// the asset database both cook through here, so a .kgeo depends only
// on its inputs - never on which tool produced it.

#include <filesystem>
#include <string>

#include "kenshin/assets/MeshData.h"
#include "kenshin/geometry/ClusterDag.h"

KENSHIN_NS_BEGIN
namespace importer {

struct CookOptions {
  bool paged = true;          // false = v2 (pools) - tests and tools only
  bool compressPages = true;  // paged: v8 codec pages (false = v7 raw)
  bool imposter = false;      // bake the far-field atlas (v9)
};

/// Cluster LOD + DAG assembly. Fills `dag` (materials table empty; the
/// caller sets it - see KgeoMaterials.h).
bool buildMeshDag(const assets::MeshData& mesh, geometry::ClusterDag& dag, std::string& log,
                  geometry::DagBuildStats* stats = nullptr);

/// The imposter cut is the deepest level, which v1 requires to be a
/// single cluster; anything else fails with a reason.
bool bakeMeshImposter(geometry::ClusterDag& dag, std::string& log);

/// Pages (or not) and writes the file; bakes the imposter first when
/// options.imposter is set.
bool writeCookedMesh(const std::filesystem::path& path, geometry::ClusterDag& dag,
                     const CookOptions& options, std::string& log);

/// Diagnostics (logs only): DAG summary, per-level and per-material
/// cluster/triangle census.
void logDagCensus(const geometry::ClusterDag& dag, const geometry::DagBuildStats& stats);

}  // namespace importer
KENSHIN_NS_END
