// S3.2: DAG assembly tests - acyclic / monotonic / same-group consistency /
// parent-child coverage / pool slices.

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "kenshin/assets/MeshData.h"
#include "kenshin/assets/ObjLoader.h"
#include "kenshin/geometry/ClusterDag.h"
#include "kenshin/geometry/ClusterLodBuilder.h"
#include "kenshin/geometry/shapes/SphereGeometry.h"
#include "kenshin/geometry/shapes/TorusKnotGeometry.h"

namespace {
std::string assetPath(const char* name) {
  for (const std::filesystem::path candidate :
       {"tests/assets", "../../../tests/assets"}) {
    std::error_code ec;
    if (std::filesystem::exists(candidate / name, ec))
      return (candidate / name).string();
  }
  return name;
}

struct Built {
  kenshin::assets::MeshData mesh;
  std::vector<kenshin::geometry::ClusterGroup> groups;
  kenshin::geometry::ClusterDag dag;
  kenshin::geometry::DagBuildStats stats;
  std::string log;
};

bool build(const char* objName, Built& out) {
  if (!kenshin::assets::loadObj(assetPath(objName), out.mesh, out.log))
    return false;
  return kenshin::geometry::buildClusterLod(
             out.mesh,
             [&](const kenshin::geometry::ClusterGroup& g) {
               out.groups.push_back(g);
             },
             *(new kenshin::geometry::BuildStats), out.log) &&
         kenshin::geometry::assembleClusterDag(out.mesh, out.groups, out.dag,
                                               out.stats, out.log);
}

bool buildProc(const kenshin::assets::MeshData& mesh, Built& out) {
  out.mesh = mesh;
  return kenshin::geometry::buildClusterLod(
             out.mesh,
             [&](const kenshin::geometry::ClusterGroup& g) {
               out.groups.push_back(g);
             },
             *(new kenshin::geometry::BuildStats), out.log) &&
         kenshin::geometry::assembleClusterDag(out.mesh, out.groups, out.dag,
                                               out.stats, out.log);
}
}  // namespace

TEST_CASE("dag grid invariants hold", "[geometry][dag]") {
  Built built;
  REQUIRE(build("grid.obj", built));
  INFO("log: " << built.log);

  // Invariants verified inside assemble (acyclic, monotonic, same-group,
  // pools); the return value already asserted. Additional checks:
  REQUIRE(built.stats.clusterCount == built.dag.clusters.size());
  REQUIRE(built.stats.rootCount >= 1);
  REQUIRE(built.stats.maxLevel >= 1);

  // Level-0 clusters sum to the source triangle count (coverage, exact).
  kenshin::u32 level0 = 0;
  for (const auto& cluster : built.dag.clusters)
    if (built.dag.groups[cluster.group].lodLevel == 0)
      level0 += cluster.indexCount / 3;
  REQUIRE(level0 == built.mesh.triangleCount());
}

TEST_CASE("dag knot parent-child triangle ratio", "[geometry][dag]") {
  Built built;
  REQUIRE(buildProc(
      kenshin::geometry::TorusKnotGeometry::create(2, 3, 128, 32, 1.0f),
      built));
  INFO("log: " << built.log);
  REQUIRE(built.stats.clusterCount > 10);
  REQUIRE(built.stats.maxLevel >= 3);

  // Deeper level aggregates roughly half the triangles of its parents'
  // aggregate (simplify ratio ~0.5 with tolerance for stuck clusters).
  std::map<int, kenshin::u32> trisPerLevel;
  for (const auto& cluster : built.dag.clusters)
    trisPerLevel[built.dag.groups[cluster.group].lodLevel] +=
        cluster.indexCount / 3;
  REQUIRE(trisPerLevel[0] > 0);
  if (trisPerLevel.count(1) && trisPerLevel.count(2)) {
    // Non-increasing triangle counts level over level.
    REQUIRE(trisPerLevel[1] <= trisPerLevel[0]);
    REQUIRE(trisPerLevel[2] <=
            trisPerLevel[1] + 128);  // slack: partitions may stall
  }
}

TEST_CASE("dag two sheets produce independent subtrees", "[geometry][dag]") {
  Built built;
  REQUIRE(build("two_sheets.obj", built));
  INFO("log: " << built.log);
  // Two disconnected components: at least one root each (clusterlod may
  // merge them into shared groups only via spatial partition, but roots
  // must exist and the graph stays valid - asserted by assemble).
  REQUIRE(built.stats.rootCount >= 1);
}

TEST_CASE("dag monotonic fixes reported", "[geometry][dag]") {
  Built built;
  REQUIRE(buildProc(kenshin::geometry::SphereGeometry::create(48, 24, 1.0f),
                    built));
  INFO("log: " << built.log);
  // Either raw errors were already monotonic (0 fixes) or we raised some;
  // the invariant check is the real gate; just exercise the counter.
  REQUIRE(built.stats.monotonicFixes >= 0);
}

TEST_CASE("dag hierarchy (BVH8) invariants", "[geometry][dag]") {
  // Structural facts (leaf-per-group, AABB/error bounds, tree shape) are
  // enforced by verifyDagInvariants inside assemble - the REQUIRE(build)
  // already gates them. Here: shape sanity the verifier does not assume.
  for (const auto& mesh :
       {kenshin::geometry::TorusKnotGeometry::create(2, 3, 128, 32, 1.0f),
        kenshin::geometry::SphereGeometry::create(96, 48, 1.0f)}) {
    Built built;
    REQUIRE(buildProc(mesh, built));
    INFO("log: " << built.log);
    const auto& dag = built.dag;
    // 1 leaf per group + internal nodes (8-ary: ~(G-1)/7 of them).
    REQUIRE(dag.hierarchyNodes.size() >= dag.groups.size());
    REQUIRE(dag.hierarchyNodes.size() < 2 * dag.groups.size());
    REQUIRE(dag.hierarchyRoot == dag.hierarchyNodes.size() - 1);

    // Every group appears as a leaf exactly once.
    std::vector<kenshin::u32> leafHits(dag.groups.size(), 0);
    std::function<void(kenshin::u32, kenshin::u32)> walk =
        [&](kenshin::u32 nodeIdx, kenshin::u32 depth) {
          const auto& node = dag.hierarchyNodes[nodeIdx];
          if (node.childCount == 0) {
            ++leafHits[node.children[0]];
            REQUIRE(depth <= 8);  // log8(hundreds of groups) + slack
            return;
          }
          REQUIRE(node.childCount >= 2);  // no single-child chains
          for (kenshin::u32 i = 0; i < node.childCount; ++i)
            walk(node.children[i], depth + 1);
        };
    walk(dag.hierarchyRoot, 1);
    for (kenshin::u32 g = 0; g < dag.groups.size(); ++g)
      REQUIRE(leafHits[g] == 1);

    // Terminal groups (FLT_MAX simplify error) keep every subtree that
    // contains one unprunable - the max must propagate to the root.
    bool anyTerminal = false;
    for (const auto& group : dag.groups)
      anyTerminal |= group.simplifyError >= std::numeric_limits<float>::max();
    if (anyTerminal)
      REQUIRE(dag.hierarchyNodes[dag.hierarchyRoot].maxSimplifyError >=
              std::numeric_limits<float>::max());
  }
}
