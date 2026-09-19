// M7.1a: ClusterQuant unit tests - bit-stream roundtrip (all widths,
// arbitrary offsets), octahedral encode/decode vs vendored meshopt,
// grid heuristic clamps, a full cluster quantize->decode roundtrip on
// a small built DAG (position error <= step/2, watertight by the
// I*step invariant).
#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/geometry/ClusterDag.h"
#include "kenshin/geometry/ClusterLodBuilder.h"
#include "kenshin/geometry/ClusterQuant.h"
#include "kenshin/geometry/KGeoFile.h"
#include "kenshin/geometry/PagePacker.h"
#include "kenshin/geometry/shapes/TorusKnotGeometry.h"
#include "kenshin/manager/ShaderManager.h"
#include "kenshin/renderer/VirtualGeometryScene.h"
#include "meshoptimizer/meshoptimizer.h"

using namespace kenshin;
using namespace kenshin::geometry;

namespace {
const char* kShaderRoot = KENSHIN_SOURCE_DIR "/engine/shaders";
bool buildDag(const assets::MeshData& mesh, ClusterDag& dag, std::string& log) {
  std::vector<ClusterGroup> groups;
  BuildStats stats;
  DagBuildStats dagStats;
  return buildClusterLod(
             mesh, [&](const ClusterGroup& g) { groups.push_back(g); }, stats,
             log) &&
         assembleClusterDag(mesh, groups, dag, dagStats, log);
}
}  // namespace

TEST_CASE("bit stream roundtrip across widths and offsets", "[quant]") {
  std::mt19937 rng(7);
  for (u32 bits = 0; bits <= 11; ++bits) {
    for (int trial = 0; trial < 40; ++trial) {
      std::vector<u32> words;
      BitWriter w(words);
      const u32 skip = u32(rng() % 97);  // arbitrary alignment prefix
      for (u32 i = 0; i < skip; ++i) w.write(i & 1u, 1);
      const u64 base = w.bit;
      std::vector<u32> values;
      const u32 mask = bits ? ((1u << bits) - 1) : 0;
      for (int i = 0; i < 64; ++i) {
        const u32 v = bits ? (rng() & mask) : 0;
        values.push_back(v);
        w.write(v, bits);
      }
      words.resize(usize((w.bit + 31) / 32) + 2, 0);
      u64 r = base;
      for (u32 v : values) {
        REQUIRE(readBits(words.data(), r, bits) == v);
        r += bits;
      }
    }
  }
}

TEST_CASE("octahedral encode/decode vs meshopt", "[quant]") {
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> dir(-1.0f, 1.0f);
  f32 maxAngle = 0.0f, sumAngle = 0.0f;
  u32 count = 0;
  for (int i = 0; i < 20000; ++i) {
    glm::vec3 n(dir(rng), dir(rng), dir(rng));
    if (glm::length(n) < 0.05f) continue;
    n = glm::normalize(n);
    u8 pair[2];
    octEncode(n.x, n.y, n.z, pair);
    const glm::vec3 d = octDecode(pair);
    REQUIRE(std::fabs(glm::length(d) - 1.0f) < 1e-5f);
    const f32 angle = std::acos(std::clamp(glm::dot(n, d), -1.0f, 1.0f)) *
                      180.0f / 3.14159265f;
    maxAngle = std::max(maxAngle, angle);
    sumAngle += angle;
    ++count;

    // Bit-level cross-check against vendored meshopt: same encoded
    // pair, same decoded snorm int triple (within 1 - rounding).
    float src[4] = {n.x, n.y, n.z, 1.0f};
    signed char enc[4];
    meshopt_encodeFilterOct(enc, 1, 4, 8, src);
    REQUIRE(enc[0] == i8(pair[0]));
    REQUIRE(enc[1] == i8(pair[1]));
    signed char dec[4] = {enc[0], enc[1], enc[2], 0};
    meshopt_decodeFilterOct(dec, 1, 4);
    const f32 mx = 127.0f;
    const i32 xq = i32(d.x * mx + (d.x >= 0 ? 0.5f : -0.5f));
    const i32 yq = i32(d.y * mx + (d.y >= 0 ? 0.5f : -0.5f));
    const i32 zq = i32(d.z * mx + (d.z >= 0 ? 0.5f : -0.5f));
    REQUIRE(std::abs(i32(dec[0]) - xq) <= 1);
    REQUIRE(std::abs(i32(dec[1]) - yq) <= 1);
    REQUIRE(std::abs(i32(dec[2]) - zq) <= 1);
  }
  printf("[quant] oct: max %.3f deg, mean %.3f deg over %u\n", maxAngle,
         sumAngle / std::max<u32>(count, 1), count);
  REQUIRE(maxAngle <= 1.5f);
  REQUIRE(sumAngle / std::max<u32>(count, 1) < 0.5f);
}

TEST_CASE("cluster quantize->decode roundtrip (small dag)", "[quant]") {
  const auto mesh = TorusKnotGeometry::create(2, 3, 128, 32, 1.0f);
  ClusterDag dag;
  std::string log;
  REQUIRE(buildDag(mesh, dag, log));
  const QuantMeshGrid grid = chooseQuantGrid(dag);
  printf("[quant] grid P=%d step=%.6f\n", grid.P, grid.step);
  REQUIRE(grid.P >= -16);
  REQUIRE(grid.P <= 20);
  const bool hasNormals = !dag.normals.empty();
  const bool hasUvs = dag.uvs.size() / 2 == dag.positions.size() / 3;

  u32 maxB = 0;
  f32 maxPosErr = 0.0f, maxUvErr = 0.0f, maxAngle = 0.0f;
  for (u32 c = 0; c < dag.clusters.size(); ++c) {
    const DagCluster& cl = dag.clusters[c];
    QuantCluster q;
    REQUIRE(quantizeCluster(dag, c, grid, hasNormals, hasUvs, q, log));
    decodeQuantCluster(q, grid.step);
    const u32 b[3] = {q.meta.widths & 0xFu, (q.meta.widths >> 4) & 0xFu,
                      (q.meta.widths >> 8) & 0xFu};
    maxB = std::max(maxB, b[0] + b[1] + b[2]);

    for (u32 v = 0; v < cl.vertexCount; ++v) {
      const f32* src = dag.positions.data() + usize(cl.vertexOffset + v) * 3;
      const f32* dst = q.decPos.data() + usize(v) * 3;
      for (int a = 0; a < 3; ++a) {
        const f32 err = std::fabs(src[a] - dst[a]);
        maxPosErr = std::max(maxPosErr, err);
        REQUIRE(err <= grid.step * 0.5f + 1e-7f);
      }
      if (hasNormals) {
        const f32* ns = dag.normals.data() + usize(cl.vertexOffset + v) * 3;
        const f32* nd = q.decNrm.data() + usize(v) * 3;
        const f32 angle = std::acos(
            std::clamp(glm::dot(glm::normalize(glm::vec3(ns[0], ns[1], ns[2])),
                                glm::vec3(nd[0], nd[1], nd[2])),
                       -1.0f, 1.0f));
        maxAngle = std::max(maxAngle, angle);
      }
      if (hasUvs) {
        const f32* ts = dag.uvs.data() + usize(cl.vertexOffset + v) * 2;
        const f32* td = q.decUv.data() + usize(v) * 2;
        for (int a = 0; a < 2; ++a) {
          const f32 err = std::fabs(ts[a] - td[a]);
          maxUvErr = std::max(maxUvErr, err);
          REQUIRE(err <= q.meta.uvRange[a] / 508.0f + 1e-7f);
        }
      }
    }
    // M7.2: decIdx holds the CYCLICALLY ROTATED triples (min index
    // first, winding-preserving) - compare per triangle under rotation.
    for (u32 t = 0; t < cl.indexCount / 3; ++t) {
      const u32 got[3] = {q.decIdx[t * 3], q.decIdx[t * 3 + 1],
                          q.decIdx[t * 3 + 2]};
      const u32 ref[3] = {dag.indices[usize(cl.indexOffset) + t * 3],
                          dag.indices[usize(cl.indexOffset) + t * 3 + 1],
                          dag.indices[usize(cl.indexOffset) + t * 3 + 2]};
      const bool cyclic =
          (got[0] == ref[0] && got[1] == ref[1] && got[2] == ref[2]) ||
          (got[0] == ref[1] && got[1] == ref[2] && got[2] == ref[0]) ||
          (got[0] == ref[2] && got[1] == ref[0] && got[2] == ref[1]);
      REQUIRE(cyclic);
    }
  }
  printf(
      "[quant] roundtrip: maxB=%u maxPosErr=%.7f maxUvErr=%.7f "
      "maxAngle=%.2f deg\n",
      maxB, maxPosErr, maxUvErr, maxAngle * 180.0f / 3.14159265f);
  REQUIRE(maxB <= 33);
  REQUIRE(maxAngle * 180.0f / 3.14159265f <= 1.5f);
}

TEST_CASE("watertight: shared vertices decode bit-identically", "[quant]") {
  const auto mesh = TorusKnotGeometry::create(2, 3, 128, 32, 1.0f);
  ClusterDag dag;
  std::string log;
  REQUIRE(buildDag(mesh, dag, log));
  const QuantMeshGrid grid = chooseQuantGrid(dag);

  // Map every dag pool vertex (by its exact float triple) to its
  // decoded floats; the same source triple in different clusters must
  // decode to the same bits (the I*step invariant - D9.2's anti-crack
  // guarantee, the reason positions live on ONE global grid).
  std::unordered_map<std::string, std::array<f32, 3>> decodedOf;
  u32 shared = 0, mismatches = 0;
  for (u32 c = 0; c < dag.clusters.size(); ++c) {
    QuantCluster q;
    REQUIRE(quantizeCluster(dag, c, grid, false, false, q, log));
    decodeQuantCluster(q, grid.step);
    for (u32 v = 0; v < q.vertexCount; ++v) {
      const f32* src =
          dag.positions.data() + usize(dag.clusters[c].vertexOffset + v) * 3;
      char key[64];
      std::snprintf(key, sizeof key, "%08x%08x%08x", *(const u32*)&src[0],
                    *(const u32*)&src[1], *(const u32*)&src[2]);
      const std::array<f32, 3> dst = {q.decPos[usize(v) * 3],
                                      q.decPos[usize(v) * 3 + 1],
                                      q.decPos[usize(v) * 3 + 2]};
      auto it = decodedOf.find(key);
      if (it == decodedOf.end()) {
        decodedOf.emplace(key, dst);
        continue;
      }
      ++shared;
      if (it->second != dst) ++mismatches;
    }
  }
  printf("[quant] watertight: %zu unique, %u shared, %u mismatches\n",
         decodedOf.size(), shared, mismatches);
  REQUIRE(mismatches == 0);
  REQUIRE(shared > 0);  // the small knot must actually share vertices
}

// M7.1 GPU probe: one dispatch per sampled (cluster, vertex) through
// the EXACT v5 chain the raster shaders use; compares against the CPU
// mirror. Any mismatch localizes the GPU-side defect.
TEST_CASE("gpu v5 decode probe", "[gpu][quantprobe]") {
  using namespace kenshin;
  const auto mesh = TorusKnotGeometry::create(2, 3, 128, 32, 1.0f);
  ClusterDag dag;
  std::string log;
  REQUIRE(buildDag(mesh, dag, log));
  PackedGeometry pg;
  REQUIRE(packPages(dag, pg, log));

  manager::ShaderManager shaders;
  REQUIRE(shaders.init(kShaderRoot));
  auto backend = backend::createVulkanBackend();
  backend::BackendInitParams bip;
  REQUIRE(backend->initialize(bip));

  std::vector<u32> vs;
  INFO("compile log: " << log);
  REQUIRE(shaders.compileFile("quant_probe.slang", vs, log, "main"));
  backend::IRenderBackend::ComputePipelineInfo ci;
  ci.csSpirv = vs.data();
  ci.csBytes = vs.size() * sizeof(u32);
  const u32 pipe = backend->createComputePipeline(ci);
  REQUIRE(pipe != backend::IRenderBackend::kInvalidPipelineHandle);

  // Slot pool upload + identity tables (same as the paged chain test).
  const u32 slotPool =
      backend->createPageSlotPool(pg.pageCount, geometry::kPageSlotSize);
  REQUIRE(slotPool != backend::IRenderBackend::kInvalidBufferHandle);
  for (u32 base = 0; base < pg.pageCount;) {
    const u32 n = std::min<u32>(64, pg.pageCount - base);
    REQUIRE(backend->beginPageUploads(slotPool));
    for (u32 i = 0; i < n; ++i)
      REQUIRE(backend->copyPageToSlot(pg.pages[base + i].data(), base + i,
                                      u32(pg.pages[base + i].size())));
    REQUIRE(backend->waitPageUploads(backend->submitPageUploads()));
    base += n;
  }
  auto makeStatic = [&](const void* data, usize bytes, const char* name) {
    backend::IRenderBackend::BufferDesc d;
    d.size = bytes;
    d.usage = backend::IRenderBackend::BufferStorage;
    d.initialData = data;
    d.debugName = name;
    return backend->createBuffer(d);
  };
  const u32 cpBuf =
      makeStatic(pg.clusterPage.data(), pg.clusterPage.size() * 4, "qp-cp");
  std::vector<u32> ptInit(pg.pageCount);
  for (u32 p = 0; p < pg.pageCount; ++p) ptInit[p] = p;
  backend::IRenderBackend::BufferDesc ptD;
  ptD.size = usize(pg.pageCount) * 4;
  ptD.usage = backend::IRenderBackend::BufferStorage |
              backend::IRenderBackend::BufferHostVisible;
  ptD.initialData = ptInit.data();
  ptD.debugName = "qp-pt";
  const u32 ptBuf = backend->createBuffer(ptD);
  std::vector<u32> phFlat;
  for (const auto& m : pg.pageMeta) {
    const u32 stepBits = *reinterpret_cast<const u32*>(&m.posStep);
    phFlat.insert(
        phFlat.end(),
        {m.posBase, m.nrmBase, m.uvBase, m.idxBase, m.metaBase, m.vertexCount,
         m.indexCount, m.clusterCount, stepBits, m.flags, m.pad0, m.pad1});
  }
  const u32 phBuf = makeStatic(phFlat.data(), phFlat.size() * 4, "qp-ph");
  const u32 cmBuf = makeStatic(pg.clusterMetaIndex.data(),
                               pg.clusterMetaIndex.size() * 4, "qp-cm");

  backend::IRenderBackend::BufferDesc pd;
  pd.size = 32;
  pd.usage = backend::IRenderBackend::BufferStorage |
             backend::IRenderBackend::BufferHostVisible;
  pd.debugName = "qp-params";
  const u32 paramBuf = backend->createBuffer(pd);
  backend::IRenderBackend::BufferDesc od;
  od.size = 32;
  od.usage = backend::IRenderBackend::BufferStorage |
             backend::IRenderBackend::BufferHostVisible |
             backend::IRenderBackend::BufferTransferSrc;
  od.debugName = "qp-out";
  const u32 outBuf = backend->createBuffer(od);

  backend::FrameBeginInfo fi;
  fi.width = 4;
  fi.height = 4;
  // (v6: the probe no longer reads the GpuCluster table - the packed
  // triple's bit base lives in the meta word - so no cluster-table
  // buffer is bound; the shader's unused binding 8 is dead-stripped
  // from the SPIR-V and the dispatch handle count must match.)

  u32 checked = 0, bad = 0, triChecked = 0, triBad = 0;
  std::mt19937 rng(3);
  for (int iter = 0; iter < 60; ++iter) {
    const u32 c = rng() % u32(dag.clusters.size());
    const bool doTri = iter % 2 == 0;
    if (doTri) {
      const u32 t = rng() % (dag.clusters[c].indexCount / 3);
      u32 pv[8] = {c, 0, dag.clusters[c].vertexCount, 0, t, 0, 0, 0};
      std::memcpy(backend->mappedPtr(paramBuf), pv, sizeof pv);
      std::memset(backend->mappedPtr(outBuf), 0, 32);
      REQUIRE(backend->beginFrame(fi));
      const u32 handles[8] = {paramBuf, slotPool, cmBuf,  cpBuf,
                              ptBuf,    phBuf,    outBuf, slotPool};
      backend->dispatchCompute(pipe, handles, 8, 1, 1, 1);
      backend->endFrame();
      std::vector<u8> ob;
      REQUIRE(backend->readbackBuffer(outBuf, 0, 32, ob));
      const u32* o = reinterpret_cast<const u32*>(ob.data());
      const u32 triBase = dag.clusters[c].indexOffset + t * 3;
      const u32 e0 = dag.indices[triBase], e1 = dag.indices[triBase + 1],
                e2 = dag.indices[triBase + 2];
      ++triChecked;
      // The GPU decodes the ROTATED triple (min-first, cyclic).
      const bool cyclic = (o[4] == e0 && o[5] == e1 && o[6] == e2) ||
                          (o[4] == e1 && o[5] == e2 && o[6] == e0) ||
                          (o[4] == e2 && o[5] == e0 && o[6] == e1);
      if (!cyclic) {
        if (triBad < 6)
          printf(
              "[probe] c%u t%u: gpu tri=(%u,%u,%u) expected=(%u,%u,%u) "
              "idxBits=%u%c",
              c, t, o[4], o[5], o[6], e0, e1, e2, o[3], 10);
        ++triBad;
      }
      continue;
    }
    const u32 v = rng() % dag.clusters[c].vertexCount;
    u32 pv[8] = {c, v, dag.clusters[c].vertexCount, 0, 0, 0, 0, 0};
    std::memcpy(backend->mappedPtr(paramBuf), pv, sizeof pv);
    std::memset(backend->mappedPtr(outBuf), 0, 32);
    REQUIRE(backend->beginFrame(fi));
    const u32 handles[8] = {paramBuf, slotPool, cmBuf,  cpBuf,
                            ptBuf,    phBuf,    outBuf, slotPool};
    backend->dispatchCompute(pipe, handles, 8, 1, 1, 1);
    backend->endFrame();
    std::vector<u8> ob;
    REQUIRE(backend->readbackBuffer(outBuf, 0, 32, ob));
    const u32* o = reinterpret_cast<const u32*>(ob.data());
    const f32 step = pg.grid.step;
    const f32* mir = pg.decodedPositions.data() +
                     usize(dag.clusters[c].vertexOffset + v) * 3;
    const i32 e0 = i32(mir[0] / step), e1 = i32(mir[1] / step),
              e2 = i32(mir[2] / step);
    ++checked;
    if (i32(o[0]) != e0 || i32(o[1]) != e1 || i32(o[2]) != e2) {
      if (bad < 6)
        printf("[probe] c%u v%u: gpu=(%d,%d,%d) expected=(%d,%d,%d)%c", c, v,
               i32(o[0]), i32(o[1]), i32(o[2]), e0, e1, e2, 10);
      ++bad;
    }
  }
  printf("[probe] verts checked %u bad %u; tris checked %u bad %u%c", checked,
         bad, triChecked, triBad, 10);
  fflush(stdout);
  REQUIRE(bad == 0);
  REQUIRE(triBad == 0);
  backend->shutdown();
}
