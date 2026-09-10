// S3.3: .kgeo roundtrip tests.

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "kenshin/assets/MeshData.h"
#include "kenshin/geometry/ClusterDag.h"
#include "kenshin/geometry/ClusterLodBuilder.h"
#include "kenshin/geometry/KGeoFile.h"
#include "kenshin/geometry/PagePacker.h"
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

// ── M6.1: paged v3 ────────────────────────────────────────────────

TEST_CASE("kgeo v3 paged roundtrip", "[geometry][kgeo][paged]") {
  // Multi-page asset: 128x32 knot packs into ~195 pages.
  const auto mesh =
      kenshin::geometry::TorusKnotGeometry::create(2, 3, 128, 32, 1.0f);
  kenshin::geometry::ClusterDag dag;
  std::string log;
  REQUIRE(buildDag(mesh, dag, log));
  INFO("log: " << log);

  kenshin::geometry::PackedGeometry pg;
  REQUIRE(kenshin::geometry::packPages(dag, pg, log));
  INFO("log: " << log);
  REQUIRE(pg.pageCount >= 2);  // actually multi-page

  // Conservation: page section totals equal the cluster sums, every
  // blob fits its slot (verifyPacking covers the slice equality).
  kenshin::usize totalVerts = 0, totalIdx = 0;
  for (const auto& cl : dag.clusters) {
    totalVerts += cl.vertexCount;
    totalIdx += cl.indexCount;
  }
  kenshin::usize metaVerts = 0, metaIdx = 0;
  for (kenshin::u32 p = 0; p < pg.pageCount; ++p) {
    metaVerts += pg.pageMeta[p].vertexCount;
    metaIdx += pg.pageMeta[p].indexCount;
    REQUIRE(pg.pages[p].size() <= kenshin::geometry::kPageSlotSize);
  }
  REQUIRE(metaVerts == totalVerts);
  REQUIRE(metaIdx == totalIdx);

  // Coarsest-first (D14.3 safety net): a cluster at a coarser lodLevel
  // never lands on a higher page index than a finer one.
  {
    std::map<kenshin::i32, std::pair<kenshin::u32, kenshin::u32>>
        levelPages;  // level -> [minPage, maxPage]
    for (kenshin::u32 c = 0; c < dag.clusters.size(); ++c) {
      const auto level = dag.groups[dag.clusters[c].group].lodLevel;
      const auto page = pg.clusterPage[c];
      auto& r = levelPages[level];
      r.first = std::min(r.first == 0 && r.second == 0 ? 0xFFFFFFFFu : r.first,
                         page);
      r.second = std::max(r.second, page);
    }
    kenshin::u32 prevMax = 0;  // max page of the level(s) processed so far
    bool first = true;
    for (auto it = levelPages.rbegin(); it != levelPages.rend(); ++it) {
      if (!first)
        REQUIRE(it->second.first >= prevMax);  // finer levels start later
      prevMax = std::max(prevMax, it->second.second);
      first = false;
    }
  }

  // v3 write + table-only read (readKGeo dispatches on the header).
  REQUIRE(kenshin::geometry::writeKGeoPaged("out/roundtrip_v3.kgeo", dag, pg,
                                            log));
  kenshin::geometry::ClusterDag loaded;
  REQUIRE(kenshin::geometry::readKGeo("out/roundtrip_v3.kgeo", loaded, log));
  INFO("log: " << log);
  REQUIRE(loaded.paged);
  REQUIRE(loaded.pageCount == pg.pageCount);
  REQUIRE(loaded.pageSlotSize == kenshin::geometry::kPageSlotSize);
  REQUIRE(loaded.pagesHaveNormals == pg.hasNormals);
  REQUIRE(loaded.pagesHaveUvs == pg.hasUvs);
  REQUIRE(loaded.clusterPage == pg.clusterPage);
  REQUIRE(loaded.pageMeta.size() == pg.pageMeta.size());
  for (kenshin::u32 p = 0; p < pg.pageCount; ++p) {
    REQUIRE(loaded.pageMeta[p].posBase == pg.pageMeta[p].posBase);
    REQUIRE(loaded.pageMeta[p].nrmBase == pg.pageMeta[p].nrmBase);
    REQUIRE(loaded.pageMeta[p].uvBase == pg.pageMeta[p].uvBase);
    REQUIRE(loaded.pageMeta[p].idxBase == pg.pageMeta[p].idxBase);
    REQUIRE(loaded.pageMeta[p].vertexCount == pg.pageMeta[p].vertexCount);
    REQUIRE(loaded.pageMeta[p].indexCount == pg.pageMeta[p].indexCount);
  }
  // Tables roundtrip exactly (v3 ClusterDisk offsets are page-local).
  REQUIRE(loaded.groups.size() == dag.groups.size());
  REQUIRE(loaded.clusters.size() == dag.clusters.size());
  REQUIRE(loaded.hierarchyNodes.size() == dag.hierarchyNodes.size());
  REQUIRE(loaded.hierarchyRoot == dag.hierarchyRoot);
  REQUIRE(loaded.roots == dag.roots);
  for (kenshin::usize c = 0; c < dag.clusters.size(); ++c) {
    REQUIRE(loaded.clusters[c].vertexCount == dag.clusters[c].vertexCount);
    REQUIRE(loaded.clusters[c].indexCount == dag.clusters[c].indexCount);
    REQUIRE(loaded.clusters[c].vertexOffset == pg.localVertexOffset[c]);
    REQUIRE(loaded.clusters[c].indexOffset == pg.localIndexOffset[c]);
    REQUIRE(loaded.clusters[c].children == dag.clusters[c].children);
  }

  // Page reads: every cluster's slice, re-extracted from its page blob
  // via the on-demand reader, is bit-identical to the v2 pools - the
  // "v3 all-resident == v2" anchor at builder level.
  kenshin::geometry::KGeoPageReader reader;
  REQUIRE(reader.open("out/roundtrip_v3.kgeo", log));
  REQUIRE(reader.pageCount() == pg.pageCount);
  std::vector<kenshin::u8> blob;
  for (kenshin::u32 p = 0; p < pg.pageCount; ++p)
    REQUIRE(reader.readPage(p, blob, log));
  for (kenshin::u32 c = 0; c < loaded.clusters.size(); ++c) {
    const auto& cl = loaded.clusters[c];
    const auto& m = loaded.pageMeta[loaded.clusterPage[c]];
    REQUIRE(reader.readPage(loaded.clusterPage[c], blob, log));
    const auto* pos = reinterpret_cast<const float*>(
        blob.data() + kenshin::usize(m.posBase) * 16);
    for (kenshin::u32 v = 0; v < cl.vertexCount; ++v) {
      const kenshin::usize s =
          kenshin::usize(dag.clusters[c].vertexOffset + v) * 3;
      REQUIRE(pos[(kenshin::usize(cl.vertexOffset) + v) * 4 + 0] ==
              dag.positions[s]);
      REQUIRE(pos[(kenshin::usize(cl.vertexOffset) + v) * 4 + 1] ==
              dag.positions[s + 1]);
      REQUIRE(pos[(kenshin::usize(cl.vertexOffset) + v) * 4 + 2] ==
              dag.positions[s + 2]);
      REQUIRE(pos[(kenshin::usize(cl.vertexOffset) + v) * 4 + 3] == 0.0f);
    }
    const auto* idx = reinterpret_cast<const kenshin::u32*>(
        blob.data() +
        (kenshin::usize(m.idxBase) + cl.indexOffset) * 4);
    for (kenshin::u32 i = 0; i < cl.indexCount; ++i)
      REQUIRE(idx[i] == dag.indices[dag.clusters[c].indexOffset + i]);
  }
  reader.close();
  std::remove("out/roundtrip_v3.kgeo");
}

TEST_CASE("kgeo page reader rejects v2", "[geometry][kgeo][paged]") {
  const auto mesh = kenshin::geometry::SphereGeometry::create(16, 8, 1.0f);
  kenshin::geometry::ClusterDag dag;
  std::string log;
  REQUIRE(buildDag(mesh, dag, log));
  REQUIRE(kenshin::geometry::writeKGeo("out/v2only.kgeo", dag, log));

  kenshin::geometry::KGeoPageReader reader;
  REQUIRE_FALSE(reader.open("out/v2only.kgeo", log));
  std::remove("out/v2only.kgeo");
}
