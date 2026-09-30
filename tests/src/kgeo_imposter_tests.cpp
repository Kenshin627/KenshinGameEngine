// M9.2: the v9 imposter-atlas file section roundtrip. The bake itself
// is covered by imposter_tests; this pins the SERIALIZATION: the
// atlas rides between the material section and the blobs, the
// version word flips to 9, a v9 reader recovers both u32 planes
// bit-exactly, and a v8 file (no atlas) still loads with empty
// planes (the runtime's "never imposter this mesh" contract).

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "kenshin/geometry/ClusterDag.h"
#include "kenshin/geometry/ClusterLodBuilder.h"
#include "kenshin/geometry/ImposterBaker.h"
#include "kenshin/geometry/KGeoFile.h"
#include "kenshin/geometry/PagePacker.h"
#include "kenshin/geometry/shapes/SphereGeometry.h"

using namespace kenshin::geometry;

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

// The deepest-level clusters (the coarsest COVERING cut; dag.roots
// can mix terminal fragments from finer levels).
std::vector<kenshin::u32> deepestCut(const ClusterDag& dag) {
  kenshin::u32 maxLevel = 0;
  for (const auto& g : dag.groups)
    maxLevel = std::max(maxLevel, kenshin::u32(g.lodLevel));
  std::vector<kenshin::u32> cut;
  for (const auto& g : dag.groups)
    if (kenshin::u32(g.lodLevel) == maxLevel)
      for (kenshin::u32 c : g.clusters) cut.push_back(c);
  return cut;
}

}  // namespace

TEST_CASE("kgeo v9 imposter atlas roundtrip", "[geometry][kgeo][imposter]") {
  const auto mesh = SphereGeometry::create(24, 12, 1.0f);
  ClusterDag dag;
  std::string log;
  REQUIRE(buildDag(mesh, dag, log));

  ImposterAtlas atlas;
  REQUIRE(bakeImposterAtlas(dag, deepestCut(dag), atlas, log));
  REQUIRE(atlas.depth.size() == kenshin::usize(12) * 12 * 12 * 12);
  REQUIRE(atlas.depth.size() == atlas.tri.size());
  // A sphere's silhouette is a disk filling ~pi/4 of the AABB tile;
  // the corners miss (only a cube fills the square). Coverage floor:
  // at least half the texels hit from every direction combined.
  const auto misses = std::count(
      atlas.tri.begin(), atlas.tri.end(), 0xFFFFFFFFu);
  REQUIRE(kenshin::usize(misses) * 4 < atlas.tri.size());

  PackedGeometry pg;
  REQUIRE(packPages(dag, pg, log));
  dag.imposterDepth = atlas.depth;
  dag.imposterTri = atlas.tri;
  REQUIRE(writeKGeoPaged("out/roundtrip_v9.kgeo", dag, pg, log, false));

  ClusterDag loaded;
  REQUIRE(readKGeo("out/roundtrip_v9.kgeo", loaded, log));
  REQUIRE(loaded.imposterDepth == dag.imposterDepth);
  REQUIRE(loaded.imposterTri == dag.imposterTri);

  // The v8 control (no atlas on the dag) still loads with empty
  // planes - the Stage B "never imposter" fallback.
  ClusterDag plain = dag;
  plain.imposterDepth.clear();
  plain.imposterTri.clear();
  REQUIRE(writeKGeoPaged("out/roundtrip_v8ctl.kgeo", plain, pg, log, true));
  ClusterDag loadedPlain;
  REQUIRE(readKGeo("out/roundtrip_v8ctl.kgeo", loadedPlain, log));
  REQUIRE(loadedPlain.imposterDepth.empty());
  REQUIRE(loadedPlain.imposterTri.empty());
}
