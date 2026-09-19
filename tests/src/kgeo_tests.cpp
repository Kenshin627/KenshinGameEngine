// S3.3: .kgeo roundtrip tests.

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "kenshin/assets/MeshData.h"
#include "kenshin/geometry/ClusterDag.h"
#include "kenshin/geometry/ClusterLodBuilder.h"
#include "kenshin/geometry/ClusterQuant.h"
#include "kenshin/geometry/KGeoFile.h"
#include "kenshin/geometry/PagePacker.h"
#include "kenshin/geometry/shapes/SphereGeometry.h"
#include "kenshin/geometry/shapes/TorusKnotGeometry.h"
#include "kenshin/renderer/VirtualGeometryScene.h"

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

TEST_CASE("kgeo v5 paged quantized roundtrip", "[geometry][kgeo][paged]") {
  // Multi-page asset: 256x64 knot packs into multiple pages (v6's
  // packed indices shrank the payload enough that the old 128x32 knot
  // now fits a single page).
  const auto mesh =
      kenshin::geometry::TorusKnotGeometry::create(2, 3, 256, 64, 1.0f);
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
  // v6: pageMeta.indexCount is the idx SECTION's byte size (packed bit
  // stream), so the index-total equality no longer holds - the vertex
  // total and the per-cluster decode checks (below + verifyPacking)
  // carry the conservation.
  kenshin::usize totalVerts = 0, totalIdx = 0;
  for (const auto& cl : dag.clusters) {
    totalVerts += cl.vertexCount;
    totalIdx += cl.indexCount;
  }
  kenshin::usize metaVerts = 0, idxSectionBytes = 0;
  for (kenshin::u32 p = 0; p < pg.pageCount; ++p) {
    metaVerts += pg.pageMeta[p].vertexCount;
    idxSectionBytes += pg.pageMeta[p].indexCount;
    REQUIRE(pg.pages[p].size() <= kenshin::geometry::kPageSlotSize);
  }
  REQUIRE(metaVerts == totalVerts);
  REQUIRE(idxSectionBytes > 0);
  REQUIRE(idxSectionBytes < totalIdx);  // packed < u8-flat, always

  // Coarsest-first (D14.3 safety net): a cluster at a coarser lodLevel
  // never lands on a higher page index than a finer one.
  {
    std::map<kenshin::i32, std::pair<kenshin::u32, kenshin::u32>>
        levelPages;  // level -> [minPage, maxPage]
    for (kenshin::u32 c = 0; c < dag.clusters.size(); ++c) {
      const auto level = dag.groups[dag.clusters[c].group].lodLevel;
      const auto page = pg.clusterPage[c];
      auto& r = levelPages[level];
      r.first =
          std::min(r.first == 0 && r.second == 0 ? 0xFFFFFFFFu : r.first, page);
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
  REQUIRE(
      kenshin::geometry::writeKGeoPaged("out/roundtrip_v3.kgeo", dag, pg, log));
  kenshin::geometry::ClusterDag loaded;
  REQUIRE(kenshin::geometry::readKGeo("out/roundtrip_v3.kgeo", loaded, log));
  INFO("log: " << log);
  REQUIRE(loaded.paged);
  REQUIRE(loaded.pagesQuantized);  // v5 gate
  REQUIRE(loaded.pageCount == pg.pageCount);
  REQUIRE(loaded.pageSlotSize == kenshin::geometry::kPageSlotSize);
  REQUIRE(loaded.pagesHaveNormals == pg.hasNormals);
  REQUIRE(loaded.pagesHaveUvs == pg.hasUvs);
  REQUIRE(loaded.clusterPage == pg.clusterPage);
  REQUIRE(loaded.clusterMetaIndex == pg.clusterMetaIndex);
  REQUIRE(loaded.pageMetaV5.size() == pg.pageMeta.size());
  for (kenshin::u32 p = 0; p < pg.pageCount; ++p) {
    REQUIRE(loaded.pageMetaV5[p].posBase == pg.pageMeta[p].posBase);
    REQUIRE(loaded.pageMetaV5[p].nrmBase == pg.pageMeta[p].nrmBase);
    REQUIRE(loaded.pageMetaV5[p].uvBase == pg.pageMeta[p].uvBase);
    REQUIRE(loaded.pageMetaV5[p].idxBase == pg.pageMeta[p].idxBase);
    REQUIRE(loaded.pageMetaV5[p].metaBase == pg.pageMeta[p].metaBase);
    REQUIRE(loaded.pageMetaV5[p].vertexCount == pg.pageMeta[p].vertexCount);
    REQUIRE(loaded.pageMetaV5[p].indexCount == pg.pageMeta[p].indexCount);
    REQUIRE(loaded.pageMetaV5[p].clusterCount == pg.pageMeta[p].clusterCount);
    REQUIRE(loaded.pageMetaV5[p].posStep == pg.pageMeta[p].posStep);
  }
  REQUIRE(loaded.clusterMaxEdge.size() == dag.clusters.size());
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
  const kenshin::f32 step = pg.grid.step;
  for (kenshin::u32 c = 0; c < loaded.clusters.size(); ++c) {
    const auto& cl = loaded.clusters[c];
    const auto& m = loaded.pageMetaV5[loaded.clusterPage[c]];
    REQUIRE(reader.readPage(loaded.clusterPage[c], blob, log));
    const auto* words = reinterpret_cast<const kenshin::u32*>(blob.data());
    kenshin::geometry::ClusterQuantMeta meta;
    std::memcpy(
        &meta,
        words + m.metaBase + kenshin::usize(loaded.clusterMetaIndex[c]) * 10,
        10 * sizeof(kenshin::u32));
    const kenshin::u32 b[3] = {meta.widths & 0xFu, (meta.widths >> 4) & 0xFu,
                               (meta.widths >> 8) & 0xFu};
    const kenshin::u32 nV = cl.vertexCount;
    const kenshin::u64 axisBase[3] = {
        kenshin::u64(meta.posWordOff) * 32,
        kenshin::u64(meta.posWordOff) * 32 + kenshin::u64(nV) * b[0],
        kenshin::u64(meta.posWordOff) * 32 + kenshin::u64(nV) * (b[0] + b[1])};
    const kenshin::u32* posSec = words + m.posBase;
    for (kenshin::u32 v = 0; v < nV; ++v) {
      kenshin::f32 dec[3];
      for (int a = 0; a < 3; ++a) {
        const kenshin::u32 bits = kenshin::geometry::readBits(
            posSec, axisBase[a] + kenshin::u64(v) * b[a], b[a]);
        dec[a] = kenshin::f32(kenshin::i32(bits) + meta.posStart[a]) * step;
      }
      const kenshin::f32* mirror =
          pg.decodedPositions.data() +
          kenshin::usize(dag.clusters[c].vertexOffset + v) * 3;
      REQUIRE(dec[0] == mirror[0]);
      REQUIRE(dec[1] == mirror[1]);
      REQUIRE(dec[2] == mirror[2]);
      const kenshin::f32* raw =
          dag.positions.data() +
          kenshin::usize(dag.clusters[c].vertexOffset + v) * 3;
      for (int a = 0; a < 3; ++a)
        REQUIRE(std::fabs(raw[a] - dec[a]) <= step * 0.5f + 1e-7f);
    }
    // Indices (M7.2 v6): decode the packed triple from the page blob
    // via the stored meta (bit base + widths) and compare CYCLICALLY -
    // the encode rotates each triangle min-first (winding-preserving).
    // readBits addresses ABSOLUTE bits (each subfield may live in the
    // word after the triangle's first); the sentinel word covers the
    // two-word combine.
    {
      const kenshin::u32 baseBits = meta.idxBits & 0xFu;
      const kenshin::u32 offsetBits = (meta.idxBits >> 4) & 0xFu;
      const kenshin::u32 triW = baseBits + 2 * offsetBits;
      const kenshin::u64 bitBase = kenshin::u64(meta.idxBits >> 8);
      const kenshin::u32* idxSec = words + m.idxBase;
      const kenshin::u32 triCount = cl.indexCount / 3;
      for (kenshin::u32 t = 0; t < triCount; ++t) {
        const kenshin::u64 bit = bitBase + kenshin::u64(t) * triW;
        const kenshin::u32 base =
            kenshin::geometry::readBits(idxSec, bit, baseBits);
        const kenshin::u32 o1 =
            kenshin::geometry::readBits(idxSec, bit + baseBits, offsetBits);
        const kenshin::u32 o2 = kenshin::geometry::readBits(
            idxSec, bit + baseBits + offsetBits, offsetBits);
        const kenshin::u32 got[3] = {base, base + o1, base + o2};
        const kenshin::u32 triBase = dag.clusters[c].indexOffset + t * 3;
        const kenshin::u32 ref[3] = {dag.indices[triBase],
                                     dag.indices[triBase + 1],
                                     dag.indices[triBase + 2]};
        const bool cyclic =
            (got[0] == ref[0] && got[1] == ref[1] && got[2] == ref[2]) ||
            (got[0] == ref[1] && got[1] == ref[2] && got[2] == ref[0]) ||
            (got[0] == ref[2] && got[1] == ref[0] && got[2] == ref[1]);
        REQUIRE(cyclic);
      }
    }
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

TEST_CASE("scene rejects pre-v5 paged dags", "[geometry][kgeo][paged]") {
  // The v5 gate (M7.1 follow-up): raster/decode shaders only speak the
  // quantized page format. A v3/v4 paged dag must be rejected loudly
  // at addMeshPaged - flowing through would build empty v5 tables, and
  // the backend silently drops every dispatch (black viewport, zero
  // diagnostics). The gate fires before any backend use.
  kenshin::renderer::VirtualGeometryScene scene;
  kenshin::geometry::ClusterDag dag;
  dag.paged = true;
  dag.pagesQuantized = false;
  dag.pageCount = 4;
  REQUIRE(scene.addMeshPaged(dag) == 0xFFFFFFFFu);

  // The v6 gate (M7.2): the shaders' index decode only speaks the
  // PACKED bit stream; a v5 page (quantized but u8-flat idx) would
  // decode garbage - the same silent-black-screen class.
  dag.pagesQuantized = true;
  dag.pageIdxPacked = false;
  REQUIRE(scene.addMeshPaged(dag) == 0xFFFFFFFFu);
}
