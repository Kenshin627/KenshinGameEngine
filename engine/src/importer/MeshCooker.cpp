#include "kenshin/importer/MeshCooker.h"

#include <algorithm>
#include <cfloat>
#include <glm/glm.hpp>
#include <map>

#include "kenshin/core/Log.h"
#include "kenshin/geometry/ClusterLodBuilder.h"
#include "kenshin/geometry/ImposterBaker.h"
#include "kenshin/geometry/KGeoFile.h"
#include "kenshin/geometry/PagePacker.h"
#include "kenshin/util/File.h"

KENSHIN_NS_BEGIN
namespace importer {

bool buildMeshDag(const assets::MeshData& mesh, geometry::ClusterDag& dag, std::string& log,
                  geometry::DagBuildStats* statsOut) {
  std::vector<geometry::ClusterGroup> groups;
  geometry::BuildStats buildStats;
  if (!geometry::buildClusterLod(
          mesh, [&](const geometry::ClusterGroup& g) { groups.push_back(g); }, buildStats,
          log)) {
    log = "clusterlod failed: " + log;
    return false;
  }
  geometry::DagBuildStats stats;
  if (!geometry::assembleClusterDag(mesh, groups, dag, stats, log)) {
    log = "dag assembly failed: " + log;
    return false;
  }
  if (statsOut) *statsOut = stats;
  return true;
}

bool bakeMeshImposter(geometry::ClusterDag& dag, std::string& log) {
  // The imposter cut = the DEEPEST level's clusters (the coarsest
  // COVERING set; dag.roots can mix terminal fragments from finer
  // levels that overlap the coarse cut and blow the 7-bit budget).
  u32 maxLevel = 0;
  for (const auto& g : dag.groups) maxLevel = std::max(maxLevel, u32(g.lodLevel));
  std::vector<u32> cut;
  for (const auto& g : dag.groups)
    if (u32(g.lodLevel) == maxLevel)
      for (u32 c : g.clusters) cut.push_back(c);
  if (cut.size() != 1u) {
    log = "imposter cut has " + std::to_string(cut.size()) +
          " clusters (v1 needs exactly 1) - skipping the atlas";
    return false;
  }
  dag.imposterCluster = cut[0];  // serialized - the runtime's truth
  geometry::ImposterAtlas atlas;
  std::string ilog;
  if (!geometry::bakeImposterAtlas(dag, cut, atlas, ilog)) {
    log = "imposter bake failed: " + ilog;
    return false;
  }
  dag.imposterDepth = std::move(atlas.depth);
  dag.imposterTri = std::move(atlas.tri);
  KENSHIN_INFO("[import] imposter baked: cut cluster {} (of {} deepest-level clusters, {} roots)",
               dag.imposterCluster, cut.size(), dag.roots.size());
  return true;
}

bool writeCookedMesh(const std::filesystem::path& path, geometry::ClusterDag& dag,
                     const CookOptions& options, std::string& log) {
  const std::string file = util::pathToUtf8(path);
  if (!options.paged) {
    if (!geometry::writeKGeo(file, dag, log)) {
      log = "write failed: " + log;
      return false;
    }
    KENSHIN_INFO("[import] written (v2 unpaged): {}", file);
    return true;
  }
  if (options.imposter && !bakeMeshImposter(dag, log)) return false;
  geometry::PackedGeometry packed;
  if (!geometry::packPages(dag, packed, log)) {
    log = "page packing failed: " + log;
    return false;
  }
  if (!geometry::writeKGeoPaged(file, dag, packed, log, options.compressPages)) {
    log = "write failed: " + log;
    return false;
  }
  std::error_code ec;
  const auto bytes = std::filesystem::file_size(path, ec);
  KENSHIN_INFO("[import] {} ({}; {} bytes, {} materials)", file,
               options.imposter ? "v9 imposter+pages"
                                : (options.compressPages ? "v8 codec pages" : "v7 raw pages"),
               ec ? 0 : bytes, dag.materials.size());
  return true;
}

void logDagCensus(const geometry::ClusterDag& dag, const geometry::DagBuildStats& stats) {
  KENSHIN_INFO(
      "[import] dag: {} clusters, {} groups, {} levels, {} roots, {} monotonic fixes, "
      "hierarchy {} nodes (root {})",
      stats.clusterCount, stats.groupCount, stats.maxLevel, stats.rootCount,
      stats.monotonicFixes, dag.hierarchyNodes.size(), dag.hierarchyRoot);
  if (dag.clusterMaterial.empty()) return;
  // Per-level triangle census (which level is coarsest?) + roots.
  std::map<u32, u64> lvlTris, lvlClusters, lvlRoots;
  for (u32 c = 0; c < dag.clusters.size(); ++c) {
    const u32 lvl = dag.groups[dag.clusters[c].group].lodLevel;
    ++lvlClusters[lvl];
    lvlTris[lvl] += dag.clusters[c].indexCount / 3;
    if (dag.clusters[c].isRoot) ++lvlRoots[lvl];
  }
  for (const auto& [l, n] : lvlClusters)
    KENSHIN_INFO("[import] level {}: {} clusters, {} tris, {} roots", l, n, lvlTris[l],
                 lvlRoots[l]);
  // Per material: clusters, triangles, LOD histogram, union bounds.
  for (u32 m = 0; m < dag.materials.size(); ++m) {
    u32 cnt = 0, tris = 0;
    std::map<u32, u32> perLevel;
    glm::vec3 mn(FLT_MAX), mx(-FLT_MAX);
    for (u32 c = 0; c < dag.clusters.size(); ++c) {
      if (dag.clusterMaterial[c] != m) continue;
      ++cnt;
      tris += dag.clusters[c].indexCount / 3;
      ++perLevel[dag.groups[dag.clusters[c].group].lodLevel];
      for (usize i = 0; i < dag.clusters[c].indexCount; ++i) {
        const u32 li = dag.clusters[c].indexOffset + static_cast<u32>(i);
        if (li >= dag.indices.size()) continue;
        const u32 vi = dag.clusters[c].vertexOffset + dag.indices[li];
        if (usize(vi) * 3 + 2 >= dag.positions.size()) continue;
        for (int k = 0; k < 3; ++k) {
          const f32 p = dag.positions[usize(vi) * 3 + k];
          mn[k] = std::min(mn[k], p);
          mx[k] = std::max(mx[k], p);
        }
      }
    }
    std::string lv;
    for (const auto& [l, n] : perLevel) lv += " L" + std::to_string(l) + "=" + std::to_string(n);
    if (cnt > 0)
      KENSHIN_INFO(
          "[import] material {} '{}': {} clusters, {} tris, bounds "
          "({:.2f},{:.2f},{:.2f})..({:.2f},{:.2f},{:.2f}){}",
          m, dag.materials[m].texPath.empty() ? "?" : dag.materials[m].texPath, cnt, tris, mn.x,
          mn.y, mn.z, mx.x, mx.y, mx.z, lv);
  }
}

}  // namespace importer
KENSHIN_NS_END
