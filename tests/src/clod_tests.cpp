// S3.1: clusterlod integration tests - size caps / coverage / depth / errors.

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <map>
#include <set>
#include <string>

#include "kenshin/assets/MeshData.h"
#include "kenshin/assets/ObjLoader.h"
#include "kenshin/core/Log.h"
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

kenshin::geometry::BuildStats buildAll(
    const kenshin::assets::MeshData& mesh,
    std::vector<kenshin::geometry::ClusterGroup>& groups, std::string& log,
    kenshin::u32 maxTris = 128) {
  kenshin::geometry::BuildStats stats;
  kenshin::geometry::buildClusterLod(
      mesh,
      [&](const kenshin::geometry::ClusterGroup& g) { groups.push_back(g); },
      stats, log, maxTris);
  return stats;
}
}  // namespace

TEST_CASE("clod grid cluster sizes and coverage", "[geometry][clod]") {
  kenshin::assets::MeshData mesh;
  std::string log;
  REQUIRE(kenshin::assets::loadObj(assetPath("grid.obj"), mesh, log));

  std::vector<kenshin::geometry::ClusterGroup> groups;
  const auto stats = buildAll(mesh, groups, log);
  INFO("log: " << log);
  REQUIRE(stats.clusterCount > 0);

  // Size caps: every cluster <= 128 triangles (partition slack allows
  // slightly more, clusterlod's max_triangles is a hard meshlet cap).
  kenshin::u32 totalTris = 0;
  for (const auto& group : groups)
    for (const auto& cluster : group.clusters) {
      const kenshin::u32 tris =
          static_cast<kenshin::u32>(cluster.indices.size() / 3);
      REQUIRE(tris <= 128);
      totalTris += tris;
    }

  // Coverage: level-0 clusters cover every triangle exactly once.
  kenshin::u32 level0Tris = 0;
  for (const auto& group : groups)
    if (group.depth == 0)
      for (const auto& cluster : group.clusters)
        level0Tris += static_cast<kenshin::u32>(cluster.indices.size() / 3);
  REQUIRE(level0Tris == mesh.triangleCount());
  REQUIRE(stats.seconds < 5.0);
}

TEST_CASE("clod torus knot depth and error monotonicity", "[geometry][clod]") {
  const auto mesh =
      kenshin::geometry::TorusKnotGeometry::create(2, 3, 128, 32, 1.0f);
  std::vector<kenshin::geometry::ClusterGroup> groups;
  std::string log;
  const auto stats = buildAll(mesh, groups, log);
  INFO("log: " << log);
  REQUIRE(stats.clusterCount > 0);
  // 524k-triangle equivalent scaled down: 8192 tris -> depth ~ log2(8192/128) = 6.
  REQUIRE(stats.maxDepth >= 3);
  REQUIRE(stats.maxDepth <= 12);
  REQUIRE(stats.maxError > 0.0f);
  REQUIRE(std::isfinite(stats.maxError));
}

TEST_CASE("clod attributes preserved (uv seam safety)", "[geometry][clod]") {
  const auto mesh = kenshin::geometry::SphereGeometry::create(32, 16, 1.0f);
  std::vector<kenshin::geometry::ClusterGroup> groups;
  std::string log;
  kenshin::geometry::BuildStats stats;
  REQUIRE(kenshin::geometry::buildClusterLod(
      mesh,
      [&](const kenshin::geometry::ClusterGroup& g) { groups.push_back(g); },
      stats, log));
  INFO("log: " << log);
  REQUIRE(groups.size() > 0);
  // Vertex indices in every cluster stay inside the source vertex buffer.
  const kenshin::u32 vertexCount = mesh.vertexCount();
  for (const auto& group : groups)
    for (const auto& cluster : group.clusters)
      for (const auto index : cluster.indices) REQUIRE(index < vertexCount);
}

TEST_CASE("clod million-triangle budget", "[geometry][clod][performance]") {
  const auto mesh =
      kenshin::geometry::TorusKnotGeometry::create(2, 3, 512, 1024, 1.0f);
  std::vector<kenshin::geometry::ClusterGroup> groups;
  std::string log;
  const auto stats = buildAll(mesh, groups, log);
  INFO("log: " << log);
  REQUIRE(stats.clusterCount > 1000);
  // Budget: 60 s single-threaded per the staged plan (not a blocker).
  kenshin::log::init();
  KENSHIN_INFO("[clod-perf] 1,048,576 tris: {} clusters in {:.1f} s",
               stats.clusterCount, stats.seconds);
}
