// S3.3: .kgeo roundtrip tests.

#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <string>
#include <vector>

#include "kenshin/assets/MeshData.h"
#include "kenshin/geometry/ClusterDag.h"
#include "kenshin/geometry/ClusterLodBuilder.h"
#include "kenshin/geometry/KGeoFile.h"
#include "kenshin/geometry/shapes/SphereGeometry.h"
#include "kenshin/geometry/shapes/TorusKnotGeometry.h"

namespace {
bool buildDag(const kenshin::assets::MeshData& mesh,
              kenshin::geometry::ClusterDag& dag, std::string& log) {
  std::vector<kenshin::geometry::ClusterGroup> groups;
  kenshin::geometry::BuildStats stats;
  kenshin::geometry::DagBuildStats dagStats;
  return kenshin::geometry::buildClusterLod(
             mesh,
             [&](const kenshin::geometry::ClusterGroup& g) {
               groups.push_back(g);
             },
             stats, log) &&
         kenshin::geometry::assembleClusterDag(mesh, groups, dag, dagStats,
                                               log);
}
}  // namespace

TEST_CASE("kgeo roundtrip sphere", "[geometry][kgeo]") {
  const auto mesh = kenshin::geometry::SphereGeometry::create(48, 24, 1.0f);
  kenshin::geometry::ClusterDag dag;
  std::string log;
  REQUIRE(buildDag(mesh, dag, log));
  INFO("log: " << log);

  REQUIRE(kenshin::geometry::writeKGeo("out/roundtrip.kgeo", dag, log));
  kenshin::geometry::ClusterDag loaded;
  REQUIRE(kenshin::geometry::readKGeo("out/roundtrip.kgeo", loaded, log));
  INFO("log: " << log);

  REQUIRE(loaded.clusters.size() == dag.clusters.size());
  REQUIRE(loaded.groups.size() == dag.groups.size());
  REQUIRE(loaded.roots.size() == dag.roots.size());
  REQUIRE(loaded.positions.size() == dag.positions.size());
  REQUIRE(loaded.normals.size() == dag.normals.size());
  REQUIRE(loaded.uvs.size() == dag.uvs.size());
  REQUIRE(loaded.indices.size() == dag.indices.size());

  for (kenshin::usize c = 0; c < dag.clusters.size(); ++c) {
    REQUIRE(loaded.clusters[c].children == dag.clusters[c].children);
    REQUIRE(loaded.clusters[c].group == dag.clusters[c].group);
    REQUIRE(loaded.clusters[c].isRoot == dag.clusters[c].isRoot);
    REQUIRE(loaded.clusters[c].vertexOffset == dag.clusters[c].vertexOffset);
    REQUIRE(loaded.clusters[c].vertexCount == dag.clusters[c].vertexCount);
  }
  REQUIRE(loaded.indices == dag.indices);
  REQUIRE(loaded.positions == dag.positions);

  // v2 (M5.1): the group BVH8 roundtrips exactly.
  REQUIRE(loaded.hierarchyNodes.size() == dag.hierarchyNodes.size());
  REQUIRE(loaded.hierarchyRoot == dag.hierarchyRoot);
  for (kenshin::usize n = 0; n < dag.hierarchyNodes.size(); ++n) {
    const auto& a = dag.hierarchyNodes[n];
    const auto& b = loaded.hierarchyNodes[n];
    REQUIRE(a.childCount == b.childCount);
    REQUIRE(a.maxSimplifyError == b.maxSimplifyError);
    for (int k = 0; k < 3; ++k) {
      REQUIRE(a.aabbMin[k] == b.aabbMin[k]);
      REQUIRE(a.aabbMax[k] == b.aabbMax[k]);
    }
    for (kenshin::u32 i = 0; i < 8; ++i)
      REQUIRE(a.children[i] == b.children[i]);
  }
  std::remove("out/roundtrip.kgeo");
}

TEST_CASE("kgeo rejects corrupt magic", "[geometry][kgeo]") {
  const auto mesh = kenshin::geometry::SphereGeometry::create(16, 8, 1.0f);
  kenshin::geometry::ClusterDag dag;
  std::string log;
  REQUIRE(buildDag(mesh, dag, log));
  REQUIRE(kenshin::geometry::writeKGeo("out/corrupt.kgeo", dag, log));

  // Stomp the first 4 bytes.
  std::FILE* file = nullptr;
  REQUIRE(fopen_s(&file, "out/corrupt.kgeo", "r+b") == 0);
  REQUIRE(file != nullptr);
  const unsigned char junk[4] = {0xDE, 0xAD, 0xBE, 0xEF};
  std::fwrite(junk, 1, 4, file);
  std::fclose(file);

  kenshin::geometry::ClusterDag loaded;
  REQUIRE_FALSE(kenshin::geometry::readKGeo("out/corrupt.kgeo", loaded, log));
  std::remove("out/corrupt.kgeo");
}
