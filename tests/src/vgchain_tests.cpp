// B2 differential: the IN-FRAME VisBufferChain vs the one-shot chain
// (the S4.2-S4.4 reference path), windowless via the headless frame
// support. Both are GPU executions of the same shaders with the same
// inputs, driven differently (persistent pipelines + indirect
// dispatches + host-visible params vs per-call one-shot everything) -
// the VisBuffers must match EXACTLY, pixel for pixel.

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <map>
#include <vector>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/geometry/ClusterDag.h"
#include "kenshin/geometry/ClusterLodBuilder.h"
#include "kenshin/geometry/PagePacker.h"
#include "kenshin/geometry/shapes/TorusKnotGeometry.h"
#include "kenshin/manager/ShaderManager.h"
#include "kenshin/renderer/ClusterCullPass.h"
#include "kenshin/renderer/InstanceCullPass.h"
#include "kenshin/renderer/SoftRasterPass.h"
#include "kenshin/renderer/VirtualGeometryScene.h"
#include "kenshin/renderer/VisBufferChain.h"
#include "vis_compare.h"

namespace {

const char* kShaderRoot = KENSHIN_SOURCE_DIR "/engine/shaders";

struct Built {
  kenshin::geometry::ClusterDag dag;
  std::string log;
};
bool build(const kenshin::assets::MeshData& mesh, Built& out) {
  std::vector<kenshin::geometry::ClusterGroup> groups;
  kenshin::geometry::BuildStats stats;
  kenshin::geometry::DagBuildStats dagStats;
  return kenshin::geometry::buildClusterLod(
             mesh,
             [&](const kenshin::geometry::ClusterGroup& g) {
               groups.push_back(g);
             },
             stats, out.log) &&
         kenshin::geometry::assembleClusterDag(mesh, groups, out.dag, dagStats,
                                               out.log);
}

}  // namespace

TEST_CASE("in-frame VisBufferChain vs one-shot chain differential",
          "[gpu][vgchain]") {
  using namespace kenshin;
  Built built;
  REQUIRE(
      build(geometry::TorusKnotGeometry::create(2, 3, 128, 32, 1.0f), built));
  auto& dag = built.dag;  // packPages pads the cull bounds in place

  manager::ShaderManager shaders;
  REQUIRE(shaders.init(kShaderRoot));
  auto backend = backend::createVulkanBackend();
  backend::BackendInitParams params;  // windowless (headless frames)
  REQUIRE(backend->initialize(params));

  // Shared inputs: pools, FORTY instances (the wide-instance payload
  // proof - the old 5b instance field capped scenes at 32), one camera.
  std::vector<u8> clusterData, groupData;
  for (const auto& c : dag.clusters)
    renderer::packGpuCluster(clusterData, c.group, c.boundsSphere.x,
                             c.boundsSphere.y, c.boundsSphere.z,
                             c.boundsSphere.w, c.error, c.parentError,
                             c.vertexOffset, c.vertexCount, c.indexOffset,
                             c.indexCount, c.refinedGroup);
  for (const auto& g : dag.groups)
    renderer::packGpuGroup(groupData, g.boundsSphere.x, g.boundsSphere.y,
                           g.boundsSphere.z, g.boundsSphere.w, g.maxError,
                           g.simplifyError, static_cast<u32>(g.lodLevel));
  std::vector<glm::vec4> positions4(dag.positions.size() / 3);
  for (usize v = 0; v < positions4.size(); ++v)
    positions4[v] = glm::vec4(dag.positions[v * 3], dag.positions[v * 3 + 1],
                              dag.positions[v * 3 + 2], 0.0f);

  constexpr u32 kInst = 40;
  std::vector<renderer::CullGpuInstance> instances;
  for (u32 i = 0; i < kInst; ++i) {
    renderer::CullGpuInstance inst;
    // Spread along x; pairs overlap from this angle - every instance
    // wins some pixels (payload indirection exercised per instance).
    inst.model = glm::translate(glm::mat4(1.0f),
                                {1.1f * f32(i), 0.0f, 0.3f * f32(i % 4)});
    instances.push_back(inst);
  }
  const std::vector<renderer::GpuMeshRangePair> meshRanges = {
      {0u, static_cast<u32>(dag.clusters.size())}};

  const glm::vec3 cam(2.5f, 3.0f, 2.0f);
  const u32 W = 64, H = 48;
  const f32 fov = 60.0f, threshold = 1.0f;
  const f32 worldPerPixel =
      2.0f * std::tan(glm::radians(fov) * 0.5f) / static_cast<f32>(H);
  glm::mat4 view = glm::lookAtRH(cam, {0, 0, 0}, {0, 1, 0});
  glm::mat4 proj =
      glm::perspectiveRH_ZO(glm::radians(fov), 4.0f / 3.0f, 0.1f, 100.0f);
  proj[1][1] *= -1.0f;
  glm::vec4 planes[6];
  renderer::extractFrustumPlanes(proj * view, planes);

  // ---- Reference: the one-shot chain (S4.2-S4.4 path).
  std::vector<u64> visOneShot;
  std::vector<glm::uvec2> pairsOneShot;  // the payload's resolve table
  {
    std::vector<u32> visible;
    REQUIRE(renderer::cullInstancesOnGpu(*backend, shaders, instances, planes,
                                         visible));
    REQUIRE(visible.size() == kInst);
    std::vector<glm::uvec2> workItems;
    REQUIRE(renderer::expandWorkItemsOnGpu(*backend, shaders, visible,
                                           instances, meshRanges, workItems));
    REQUIRE(workItems.size() == usize(kInst) * dag.clusters.size());
    std::vector<glm::uvec2>& pairs = pairsOneShot;
    u32 cullStats[4] = {};
    REQUIRE(renderer::cullClustersOnGpu(
        *backend, shaders, cam, threshold, worldPerPixel, clusterData,
        groupData, workItems, instances, pairs, cullStats));
    REQUIRE(cullStats[0] == pairs.size());
    const renderer::SoftRasterFrame frame{view, proj, 0.1f,
                                          W,    H,    /*yFlip=*/true};
    u32 rasterStats[4] = {};
    REQUIRE(renderer::rasterizeOnGpu(*backend, shaders, frame, positions4,
                                     dag.indices, clusterData, instances, pairs,
                                     visOneShot, rasterStats));
    REQUIRE(rasterStats[1] > 0);  // something actually rasterized
  }

  // ---- Subject: the in-frame chain, on a headless frame - BOTH cull
  // modes (M5.2): linear (the expand+per-cluster reference shape) and
  // the level-wise BVH8 traversal. Each must reproduce the one-shot
  // bits exactly; the pair (linear, hierarchy) also pins the two
  // modes against each other transitively.
  const std::vector<u8> hierarchyData(
      reinterpret_cast<const u8*>(dag.hierarchyNodes.data()),
      reinterpret_cast<const u8*>(dag.hierarchyNodes.data()) +
          dag.hierarchyNodes.size() *
              sizeof(kenshin::geometry::DagHierarchyNode));
  std::vector<glm::uvec2> groupRanges;
  for (const auto& g : dag.groups)
    groupRanges.emplace_back(g.clusters.empty() ? 0 : g.clusters.front(),
                             static_cast<u32>(g.clusters.size()));
  const std::vector<u32> meshHierRoots{dag.hierarchyRoot};
  renderer::VisBufferChain chain;
  REQUIRE(chain.init(
      *backend, shaders, positions4, dag.indices, clusterData, groupData,
      hierarchyData, groupRanges, meshHierRoots, instances, meshRanges,
      static_cast<u32>(kInst * dag.clusters.size()), {}, nullptr, W, H));
  renderer::ChainInputs inputs;  // (planes cannot be aggregate-named)
  inputs.view = view;
  inputs.proj = proj;
  inputs.nearZ = 0.1f;
  inputs.camPos = cam;
  inputs.pixelThreshold = threshold;
  inputs.worldPerPixel = worldPerPixel;
  inputs.width = W;
  inputs.height = H;
  for (int p = 0; p < 6; ++p) inputs.planes[p] = planes[p];
  // Pure SW on both sides: the one-shot reference has no HW path, and
  // HW depth bits are perspective-correct (a documented divergence) -
  // this differential pins chain-vs-one-shot structural equivalence.
  inputs.hwMode = 0;
  // Per-cullMode probe histograms recorded by the mode-0 passes; the
  // mode-1 passes must reproduce them exactly (the pin lives inside
  // runChainMode, below the readRasterProbe read).
  std::map<u32, std::array<u32, 8>> probeHistByCull;

  const auto runChainMode = [&](u32 cullMode, u32 rasterMode) {
    inputs.cullMode = cullMode;
    // rasterMode 1 (the optimized scheduling: exact-int64 row-span
    // clamp + per-thread stat aggregation) must reproduce the one-shot
    // reference BITS exactly - only which pixels get tested (never
    // written) and how the counters accumulate change. The probe rides
    // along on every pass; its totals cross-check the same frame's
    // counters.
    inputs.rasterMode = rasterMode;
    inputs.rasterProbe = 1;
    backend::FrameBeginInfo frameInfo;
    frameInfo.width = W;
    frameInfo.height = H;
    // Two frames per mode (both slots): proves per-slot isolation (the
    // first frame's intermediates must not leak into or be clobbered
    // by this one).
    REQUIRE(backend->beginFrame(frameInfo));
    chain.execute(*backend, inputs);
    backend->endFrame();
    REQUIRE(backend->beginFrame(frameInfo));
    chain.execute(*backend, inputs);
    backend->endFrame();

    std::vector<u8> visBytes;
    REQUIRE(backend->readbackBuffer(chain.visBuffer(*backend), 0,
                                    usize(W) * H * sizeof(u64), visBytes));
    u32 stats[16] = {};
    REQUIRE(chain.readStats(*backend, stats));
    INFO("cullMode " << cullMode << " chain stats: visibleInst " << stats[0]
                     << ", workItems " << stats[1] << ", pairs " << stats[2]
                     << ", writes " << stats[5]);
    REQUIRE(stats[0] > 0);  // instances survived the cull
    REQUIRE(stats[2] > 0);  // clusters cut
    REQUIRE(stats[5] > 0);  // px writes
    // Resolved-attribute comparison (the payload's pair index is
    // enumeration-sensitive; the one-shot and chain paths order pairs
    // differently - depth + resolved instance/cluster/triangle equal
    // IS the decode-visible equality).
    viscmp::Lists chainLists;
    REQUIRE(chain.readVisiblePairs(*backend, chainLists.pairs));
    REQUIRE(chain.readHwPairs(*backend, chainLists.hw));
    REQUIRE(chain.readPostPairs(*backend, chainLists.post));
    const viscmp::Lists oneShotLists{pairsOneShot, {}, {}};
    const usize mismatches =
        viscmp::mismatches(visOneShot.data(), oneShotLists,
                           reinterpret_cast<const u64*>(visBytes.data()),
                           chainLists, visOneShot.size());
    INFO("cullMode " << cullMode << ": vis pixels " << visOneShot.size()
                     << ", mismatches " << mismatches);
    REQUIRE(mismatches == 0);
    // The probe's pixel sum must equal main + post px writes of the
    // SAME frame (both dispatches share the histogram buffer; hwMode 0
    // keeps raster_hw's pending[1] add out of rasterCnt[1]). Safe to
    // read mapped: readStats' blocking readbacks drained the queue.
    u32 ph[8] = {};
    REQUIRE(chain.readRasterProbe(*backend, ph, /*lastCompleted=*/true));
    INFO("cullMode " << cullMode << " rasterMode " << rasterMode
                     << " probe: pix " << ph[6] << " tris " << ph[7]
                     << " buckets " << ph[0] << "/" << ph[1] << "/" << ph[2]
                     << "/" << ph[3] << "/" << ph[4] << "/" << ph[5]);
    REQUIRE(ph[6] == stats[5] + stats[10]);
    REQUIRE(ph[7] > 0);
    // The buckets partition the counted triangles - a violation means
    // the probe itself wrote garbage.
    REQUIRE(ph[0] + ph[1] + ph[2] + ph[3] + ph[4] + ph[5] == ph[7]);
    // Mode invariance: the histogram is a pure function of each
    // triangle's covered pixels, so mode 1 must reproduce mode 0's
    // buckets per cull driver exactly - a strictly stronger pin than
    // the pixel-sum total above.
    auto& stored = probeHistByCull[cullMode];
    if (rasterMode == 0) {
      for (int k = 0; k < 8; ++k) stored[k] = ph[k];
    } else {
      REQUIRE(stored[7] != 0);  // the mode-0 pass recorded first
      for (int k = 0; k < 8; ++k) REQUIRE(stored[k] == ph[k]);
    }
  };
  runChainMode(0, 0);  // linear: the expand+per-cluster reference shape
  runChainMode(1, 0);  // M5.2: the level-wise BVH8 traversal
  runChainMode(2, 0);  // M5.3: the persistent-threads MPMC traversal
  // The optimized scheduling against every cull driver (each must
  // still match the mode-0 one-shot bits).
  runChainMode(1, 1);
  runChainMode(0, 1);
  runChainMode(2, 1);

  chain.onShutdown(*backend);
  backend->shutdown();
}

// ── M6.3: paged (slot-pool) addressing vs the v2 pools ───────────
TEST_CASE("paged chain differential + eviction skip + touched feedback",
          "[gpu][vgchain][paged]") {
  using namespace kenshin;
  Built built;
  REQUIRE(
      build(geometry::TorusKnotGeometry::create(2, 3, 128, 32, 1.0f), built));
  auto& dag = built.dag;  // packPages pads the cull bounds in place

  geometry::PackedGeometry pg;
  std::string packLog;
  REQUIRE(geometry::packPages(dag, pg, packLog));
  REQUIRE(pg.pageCount >= 2);

  manager::ShaderManager shaders;
  REQUIRE(shaders.init(kShaderRoot));
  auto backend = backend::createVulkanBackend();
  backend::BackendInitParams params;  // windowless
  REQUIRE(backend->initialize(params));

  std::vector<u8> clusterData, groupData;
  for (const auto& c : dag.clusters)
    renderer::packGpuCluster(clusterData, c.group, c.boundsSphere.x,
                             c.boundsSphere.y, c.boundsSphere.z,
                             c.boundsSphere.w, c.error, c.parentError,
                             c.vertexOffset, c.vertexCount, c.indexOffset,
                             c.indexCount, c.refinedGroup);
  for (const auto& g : dag.groups)
    renderer::packGpuGroup(groupData, g.boundsSphere.x, g.boundsSphere.y,
                           g.boundsSphere.z, g.boundsSphere.w, g.maxError,
                           g.simplifyError, static_cast<u32>(g.lodLevel));
  // Paged cluster offsets are PAGE-LOCAL - rebuild the GPU table with
  // pg.localVertexOffset/localIndexOffset (the v2 pools keep the
  // dag-global offsets for the reference chain).
  std::vector<u8> pagedClusterData;
  for (u32 c = 0; c < dag.clusters.size(); ++c) {
    const auto& cl = dag.clusters[c];
    renderer::packGpuCluster(
        pagedClusterData, cl.group, cl.boundsSphere.x, cl.boundsSphere.y,
        cl.boundsSphere.z, cl.boundsSphere.w, cl.error, cl.parentError,
        pg.localVertexOffset[c], cl.vertexCount, pg.localIndexOffset[c],
        cl.indexCount, cl.refinedGroup);
  }
  // v5: the v2 reference renders the DECODED pools (D5 bit-identity:
  // bit-identical to what the paged GPU decodes) over the PADDED
  // dag tables (clusterData packs the padded bounds).
  std::vector<glm::vec4> positions4(pg.decodedPositions.size() / 3);
  for (usize v = 0; v < positions4.size(); ++v)
    positions4[v] =
        glm::vec4(pg.decodedPositions[v * 3], pg.decodedPositions[v * 3 + 1],
                  pg.decodedPositions[v * 3 + 2], 0.0f);
  const std::vector<glm::vec4> aabbs =
      renderer::computeClusterAabbs(positions4, pg.decodedIndices, clusterData);

  std::vector<renderer::CullGpuInstance> instances;
  instances.push_back(renderer::CullGpuInstance{});
  const std::vector<renderer::GpuMeshRangePair> meshRanges = {
      {0u, static_cast<u32>(dag.clusters.size())}};

  const glm::vec3 cam(2.5f, 3.0f, 2.0f);
  const u32 W = 64, H = 48;
  const f32 fov = 60.0f, threshold = 1.0f;
  const f32 worldPerPixel =
      2.0f * std::tan(glm::radians(fov) * 0.5f) / static_cast<f32>(H);
  glm::mat4 view = glm::lookAtRH(cam, {0, 0, 0}, {0, 1, 0});
  glm::mat4 proj =
      glm::perspectiveRH_ZO(glm::radians(fov), 4.0f / 3.0f, 0.1f, 100.0f);
  proj[1][1] *= -1.0f;
  glm::vec4 planes[6];
  renderer::extractFrustumPlanes(proj * view, planes);

  const std::vector<u8> hierarchyData(
      reinterpret_cast<const u8*>(dag.hierarchyNodes.data()),
      reinterpret_cast<const u8*>(dag.hierarchyNodes.data()) +
          dag.hierarchyNodes.size() *
              sizeof(kenshin::geometry::DagHierarchyNode));
  std::vector<glm::uvec2> groupRanges;
  for (const auto& g : dag.groups)
    groupRanges.emplace_back(g.clusters.empty() ? 0 : g.clusters.front(),
                             static_cast<u32>(g.clusters.size()));
  const std::vector<u32> meshHierRoots{dag.hierarchyRoot};

  renderer::ChainInputs inputs;
  inputs.view = view;
  inputs.proj = proj;
  inputs.nearZ = 0.1f;
  inputs.camPos = cam;
  inputs.pixelThreshold = threshold;
  inputs.worldPerPixel = worldPerPixel;
  inputs.width = W;
  inputs.height = H;
  for (int p = 0; p < 6; ++p) inputs.planes[p] = planes[p];
  inputs.hwMode = 0;  // pure SW: the byte-level anchor
  inputs.cullMode = 1;
  backend::FrameBeginInfo frameInfo;
  frameInfo.width = W;
  frameInfo.height = H;

  const auto runFrames = [&](renderer::VisBufferChain& chain) {
    REQUIRE(backend->beginFrame(frameInfo));
    chain.execute(*backend, inputs);
    backend->endFrame();
    REQUIRE(backend->beginFrame(frameInfo));
    chain.execute(*backend, inputs);
    backend->endFrame();
  };

  // ---- v2 reference chain.
  std::vector<u8> visV2Bytes;
  u32 statsV2[16] = {};
  std::vector<glm::uvec2> pairsV2;
  viscmp::Lists listsV2;  // the v2 reference's resolve tables
  {
    renderer::VisBufferChain chain;
    REQUIRE(chain.init(
        *backend, shaders, positions4, pg.decodedIndices, clusterData,
        groupData, hierarchyData, groupRanges, meshHierRoots, instances,
        meshRanges, static_cast<u32>(dag.clusters.size()), {}, nullptr, W, H));
    runFrames(chain);
    REQUIRE(backend->readbackBuffer(chain.visBuffer(*backend), 0,
                                    usize(W) * H * sizeof(u64), visV2Bytes));
    REQUIRE(chain.readStats(*backend, statsV2));
    REQUIRE(statsV2[2] > 0);
    REQUIRE(statsV2[5] > 0);
    REQUIRE(statsV2[15] == 0);
    REQUIRE(chain.readVisiblePairs(*backend, pairsV2));
    listsV2.pairs = pairsV2;
    REQUIRE(chain.readHwPairs(*backend, listsV2.hw));
    REQUIRE(chain.readPostPairs(*backend, listsV2.post));
    chain.onShutdown(*backend);
  }

  // ---- Paged setup: slot pool + identity page table + static tables.
  const u32 slotPool =
      backend->createPageSlotPool(pg.pageCount, geometry::kPageSlotSize);
  REQUIRE(slotPool != backend::IRenderBackend::kInvalidBufferHandle);
  {
    // Upload every page through the M6.2 batch API (<= 64 pages/batch).
    for (u32 base = 0; base < pg.pageCount;) {
      const u32 n = std::min<u32>(64, pg.pageCount - base);
      REQUIRE(backend->beginPageUploads(slotPool));
      for (u32 i = 0; i < n; ++i)
        REQUIRE(backend->copyPageToSlot(
            pg.pages[base + i].data(), base + i,
            static_cast<u32>(pg.pages[base + i].size())));
      REQUIRE(backend->waitPageUploads(backend->submitPageUploads()));
      base += n;
    }
  }
  auto makeStatic = [&](const void* data, usize bytes, const char* name) {
    backend::IRenderBackend::BufferDesc d;
    d.size = bytes;
    d.usage = backend::IRenderBackend::BufferStorage;
    d.initialData = data;
    d.debugName = name;
    return backend->createBuffer(d);
  };
  const u32 clusterPageBuf = makeStatic(
      pg.clusterPage.data(), pg.clusterPage.size() * 4, "pg-clusterPage");
  std::vector<u32> pageTableInit(pg.pageCount);
  for (u32 p = 0; p < pg.pageCount; ++p) pageTableInit[p] = p;
  backend::IRenderBackend::BufferDesc ptDesc;
  ptDesc.size = usize(pg.pageCount) * 4;
  ptDesc.usage = backend::IRenderBackend::BufferStorage |
                 backend::IRenderBackend::BufferHostVisible;
  ptDesc.initialData = pageTableInit.data();
  ptDesc.debugName = "pg-pageTable";
  const u32 pageTableBuf = backend->createBuffer(ptDesc);
  auto* pageTable = static_cast<u32*>(backend->mappedPtr(pageTableBuf));
  std::vector<u32> pageHeaderFlat;
  pageHeaderFlat.reserve(usize(pg.pageCount) * 12);
  for (const auto& m : pg.pageMeta) {
    const u32 stepBits = *reinterpret_cast<const u32*>(&m.posStep);
    pageHeaderFlat.insert(
        pageHeaderFlat.end(),
        {m.posBase, m.nrmBase, m.uvBase, m.idxBase, m.metaBase, m.vertexCount,
         m.indexCount, m.clusterCount, stepBits, m.flags, m.pad0, m.pad1});
  }
  const u32 pageHeaderBuf = makeStatic(
      pageHeaderFlat.data(), pageHeaderFlat.size() * 4, "pg-pageHeader");
  const u32 clusterMetaBuf =
      makeStatic(pg.clusterMetaIndex.data(), pg.clusterMetaIndex.size() * 4,
                 "pg-clusterMeta");

  renderer::ChainPagedInputs pi;
  pi.slotPool = slotPool;
  pi.clusterPage = clusterPageBuf;
  pi.pageTable = pageTableBuf;
  pi.pageHeader = pageHeaderBuf;
  pi.pageCount = pg.pageCount;
  pi.slotU32Total = pg.pageCount * (geometry::kPageSlotSize / 4);
  pi.clusterMeta = clusterMetaBuf;

  // ---- Subject 1: all-resident paged chain - byte-identical to v2
  // across the cullMode x occlusion sweep. ONE v2 reference serves all
  // configs: occ on/off VisBuffer equality is the M3.3 differential's
  // proven strong form, and cullMode 0/1/2 equality is the M5.2/M5.3
  // chain-vs-one-shot result - here they re-verify through the PAGED
  // addressing path. Configs 6/7 add the rasterMode-1 scheduling
  // through the paged (v5) addressing - bit-identical to the same
  // reference.
  const u32 cfgTable[8][3] = {
      // cullMode, occ, rasterMode
      {0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {0, 0, 0},
      {1, 0, 0}, {2, 0, 0}, {1, 1, 1}, {0, 0, 1},
  };
  for (u32 config = 0; config < 8; ++config) {
    const u32 cullMode = cfgTable[config][0];
    const bool occ = cfgTable[config][1] != 0;
    inputs.cullMode = cullMode;
    inputs.occlusionEnabled = occ;
    inputs.rasterMode = cfgTable[config][2];
    renderer::VisBufferChain chain;
    REQUIRE(chain.init(*backend, shaders, {}, {}, pagedClusterData, groupData,
                       hierarchyData, groupRanges, meshHierRoots, instances,
                       meshRanges, static_cast<u32>(dag.clusters.size()), aabbs,
                       &pi, W, H));
    runFrames(chain);
    std::vector<u8> visBytes;
    REQUIRE(backend->readbackBuffer(chain.visBuffer(*backend), 0,
                                    usize(W) * H * sizeof(u64), visBytes));
    u32 stats[16] = {};
    REQUIRE(chain.readStats(*backend, stats));
    INFO("config " << config << " (cullMode " << cullMode << ", occ " << occ
                   << ")");
    REQUIRE(stats[2] == statsV2[2]);  // same cut
    REQUIRE(stats[5] == statsV2[5]);  // same px writes
    REQUIRE(stats[15] == 0);          // nothing skipped
    viscmp::Lists lists;
    REQUIRE(chain.readVisiblePairs(*backend, lists.pairs));
    REQUIRE(chain.readHwPairs(*backend, lists.hw));
    REQUIRE(chain.readPostPairs(*backend, lists.post));
    const usize mismatches = viscmp::mismatches(
        reinterpret_cast<const u64*>(visV2Bytes.data()), listsV2,
        reinterpret_cast<const u64*>(visBytes.data()), lists,
        visV2Bytes.size() / 8);
    INFO("paged all-resident config " << config << ": " << mismatches
                                      << " vis mismatches");
    REQUIRE(mismatches == 0);
    if (config != 0) continue;
    // Touched feedback (default config): every page of a visible
    // cluster is marked.
    std::vector<u32> touched;
    REQUIRE(chain.readTouchedPages(*backend, touched));
    std::vector<glm::uvec2> pairs;
    REQUIRE(chain.readVisiblePairs(*backend, pairs));
    const u32 expectWords = (pg.pageCount + 31) / 32;
    REQUIRE(touched.size() == expectWords);
    usize markedPages = 0;
    for (u32 w = 0; w < touched.size(); ++w)
      for (u32 b2 = 0; b2 < 32; ++b2) markedPages += (touched[w] >> b2) & 1u;
    REQUIRE(markedPages >= 1);
    REQUIRE(markedPages <= pg.pageCount);
    for (const auto& pr : pairs)
      REQUIRE(
          ((touched[pg.clusterPage[pr.y] >> 5] >> (pg.clusterPage[pr.y] & 31)) &
           1u) == 1u);
    chain.onShutdown(*backend);
  }

  // Restore the default inputs (Subject 1's sweep mutated them).
  inputs.cullMode = 1;
  inputs.occlusionEnabled = true;
  inputs.rasterMode = 0;
  // ---- Subject 1b: FORCE-HW paged chain (hwMode 2). Every cluster
  // routes through raster_hw's paged v5 decode + the 12-handle
  // drawNullIndirect - the exact app-default wiring where the a13ba83
  // handle-count bug lived invisible to every suite. The invariant is
  // COVERAGE exactness vs the v2 reference (the M4.2 force-HW result);
  // payload words are not comparable cross-scheme (HW depth is
  // perspective-correct, SW affine - see the note at the assert).
  {
    inputs.hwMode = 2;
    inputs.occlusionEnabled = false;
    renderer::VisBufferChain chain;
    REQUIRE(chain.init(*backend, shaders, {}, {}, pagedClusterData, groupData,
                       hierarchyData, groupRanges, meshHierRoots, instances,
                       meshRanges, static_cast<u32>(dag.clusters.size()), aabbs,
                       &pi, W, H));
    runFrames(chain);
    std::vector<u8> visBytes;
    REQUIRE(backend->readbackBuffer(chain.visBuffer(*backend), 0,
                                    usize(W) * H * sizeof(u64), visBytes));
    u32 stats[16] = {};
    REQUIRE(chain.readStats(*backend, stats));
    // hzb_tests' force-HW pattern: stats[2] is the SW-side pair count -
    // 0 BY DESIGN under force; stats[12] proves the HW path took them.
    REQUIRE(stats[12] > 0);   // hw-routed pairs
    REQUIRE(stats[2] == 0);   // ... ALL of them (force mode)
    REQUIRE(stats[15] == 0);  // nothing skipped (residency)
    // Coverage vs the v2 reference must be exact (the M4.2 result).
    // Payload mismatches get a budget: HW depth is perspective-correct
    // where SW is affine, so near-tie winners flip; the a13ba83 class
    // read ~16% of pixels wrong - 3 orders above the budget.
    const u64* v2w = reinterpret_cast<const u64*>(visV2Bytes.data());
    const u64* hwW = reinterpret_cast<const u64*>(visBytes.data());
    usize nzV2 = 0, nzHw = 0, holes = 0, extra = 0;
    for (usize i = 0; i < usize(W) * H; ++i) {
      const bool s = v2w[i] != 0;
      const bool h = hwW[i] != 0;
      nzV2 += s;
      nzHw += h;
      if (s && !h) ++holes;
      if (h && !s) ++extra;
    }
    INFO("paged FORCE-HW: nz v2 " << nzV2 << " hw " << nzHw << " holes "
                                  << holes << " extra " << extra);
    REQUIRE(holes == 0);
    REQUIRE(extra == 0);
    // No payload-word compare here, by design: HW depth is
    // perspective-correct where SW is affine, so the depth word (and
    // near-tie winners) differ on essentially every pixel - viscmp
    // resolves depth into its compare, making cross-scheme mismatch
    // counts by-construction ~100% (measured 812/824). The M4.2
    // force-HW precedent: COVERAGE exactness is the invariant that
    // catches decode garbage (degenerate triangles -> holes,
    // splinters -> extra); wrong-geometry-with-right-payload-id is
    // invisible to any VisBuffer compare and is the SW-path D5
    // differential's job.
    viscmp::Lists lists;
    REQUIRE(chain.readVisiblePairs(*backend, lists.pairs));
    REQUIRE(chain.readHwPairs(*backend, lists.hw));
    REQUIRE(chain.readPostPairs(*backend, lists.post));
    REQUIRE(lists.hw.size() > 0);  // actually routed HW
    chain.onShutdown(*backend);
    inputs.hwMode = 0;
    inputs.occlusionEnabled = true;
  }

  // ---- Subject 2: eviction - a page that provably carries CUT  // clusters (chosen from the v2 pair set) goes absent: skips counted,
  // the touched bitmap still marks it (miss detection), and restoring
  // the page returns the VisBuffer to byte-identity (pop-in recovery).
  u32 evicted = 0xFFFFFFFFu;
  for (auto it = pairsV2.rbegin(); it != pairsV2.rend(); ++it)
    if (pg.clusterPage[it->y] != 0) {
      evicted = pg.clusterPage[it->y];
      break;
    }
  if (evicted == 0xFFFFFFFFu && pg.pageCount > 1)
    evicted = pg.pageCount - 1;  // v5: this view's cut may sit entirely
                                 // in the coarse page - evict ANY other
  REQUIRE(evicted != 0xFFFFFFFFu);
  renderer::VisBufferChain chain;
  REQUIRE(chain.init(*backend, shaders, {}, {}, pagedClusterData, groupData,
                     hierarchyData, groupRanges, meshHierRoots, instances,
                     meshRanges, static_cast<u32>(dag.clusters.size()), aabbs,
                     &pi, W, H));
  pageTable[evicted] = 0xFFFFFFFFu;
  runFrames(chain);
  {
    u32 stats[16] = {};
    REQUIRE(chain.readStats(*backend, stats));
    REQUIRE(stats[2] == statsV2[2]);  // cull untouched (metadata)
    std::vector<glm::uvec2> pairs;
    REQUIRE(chain.readVisiblePairs(*backend, pairs));
    u32 pendingExpected = 0;
    for (const auto& pr : pairs)
      if (pg.clusterPage[pr.y] == evicted) ++pendingExpected;
    INFO("evicted page " << evicted << ": pending expected " << pendingExpected
                         << ", got " << stats[15] << " (pairs " << pairs.size()
                         << ")");
    REQUIRE(stats[15] == pendingExpected);
    REQUIRE(stats[5] <= statsV2[5]);  // strictly fewer px writes
    // Miss detection: an absent page CARRYING cut clusters is still
    // marked touched (a page this view never references legitimately
    // stays unmarked - v5's coarser pages can put the whole cut in
    // page 0).
    if (pendingExpected > 0) {
      std::vector<u32> touched;
      REQUIRE(chain.readTouchedPages(*backend, touched));
      REQUIRE(((touched[evicted >> 5] >> (evicted & 31)) & 1u) == 1u);
    }
  }
  pageTable[evicted] = evicted;  // stream it back in
  runFrames(chain);
  {
    std::vector<u8> visBytes;
    REQUIRE(backend->readbackBuffer(chain.visBuffer(*backend), 0,
                                    usize(W) * H * sizeof(u64), visBytes));
    u32 stats[16] = {};
    REQUIRE(chain.readStats(*backend, stats));
    REQUIRE(stats[15] == 0);
    viscmp::Lists lists;
    REQUIRE(chain.readVisiblePairs(*backend, lists.pairs));
    REQUIRE(chain.readHwPairs(*backend, lists.hw));
    REQUIRE(chain.readPostPairs(*backend, lists.post));
    const usize mismatches = viscmp::mismatches(
        reinterpret_cast<const u64*>(visV2Bytes.data()), listsV2,
        reinterpret_cast<const u64*>(visBytes.data()), lists,
        visV2Bytes.size() / 8);
    INFO("paged post-restore: " << mismatches << " vis mismatches");
    REQUIRE(mismatches == 0);
  }
  chain.onShutdown(*backend);

  backend->destroyBuffer(slotPool);
  backend->destroyBuffer(clusterPageBuf);
  backend->destroyBuffer(pageTableBuf);
  backend->destroyBuffer(pageHeaderBuf);
  backend->destroyBuffer(clusterMetaBuf);
  backend->shutdown();
}
// ── M6.4: the big-asset all-resident differential (byte-identity at
// scale): knot:1024:1024 (34193 clusters / ~1460 pages) - one v2 chain
// vs one paged chain, level-wise cull + occlusion, pure SW. The small
// [paged] test's sweep at scale would cost minutes of GPU time; here
// ONE config at the acceptance scale proves the addressing holds when
// the page table, cluster table and pair lists are production-sized.
TEST_CASE("paged big differential (knot:1024:1024)", "[gpu][vgchain][paged]") {
  using namespace kenshin;
  Built built;
  REQUIRE(build(geometry::TorusKnotGeometry::create(2, 3, 1024, 1024, 1.0f),
                built));
  auto& dag = built.dag;  // packPages pads the cull bounds in place

  geometry::PackedGeometry pg;
  std::string packLog;
  REQUIRE(geometry::packPages(dag, pg, packLog));
  INFO(packLog);
  REQUIRE(pg.pageCount >= 100);  // production-scale page table (v5:
  REQUIRE(pg.pageCount <= 400);  // ~5x smaller pages)

  manager::ShaderManager shaders;
  REQUIRE(shaders.init(kShaderRoot));
  auto backend = backend::createVulkanBackend();
  backend::BackendInitParams params;
  REQUIRE(backend->initialize(params));

  std::vector<u8> clusterData, groupData, pagedClusterData;
  for (const auto& c : dag.clusters)
    renderer::packGpuCluster(clusterData, c.group, c.boundsSphere.x,
                             c.boundsSphere.y, c.boundsSphere.z,
                             c.boundsSphere.w, c.error, c.parentError,
                             c.vertexOffset, c.vertexCount, c.indexOffset,
                             c.indexCount, c.refinedGroup);
  for (u32 c = 0; c < dag.clusters.size(); ++c) {
    const auto& cl = dag.clusters[c];
    renderer::packGpuCluster(
        pagedClusterData, cl.group, cl.boundsSphere.x, cl.boundsSphere.y,
        cl.boundsSphere.z, cl.boundsSphere.w, cl.error, cl.parentError,
        pg.localVertexOffset[c], cl.vertexCount, pg.localIndexOffset[c],
        cl.indexCount, cl.refinedGroup);
  }
  for (const auto& g : dag.groups)
    renderer::packGpuGroup(groupData, g.boundsSphere.x, g.boundsSphere.y,
                           g.boundsSphere.z, g.boundsSphere.w, g.maxError,
                           g.simplifyError, static_cast<u32>(g.lodLevel));
  // v5: the v2 reference renders the DECODED pools (D5 bit-identity:
  // bit-identical to what the paged GPU decodes) over the PADDED
  // dag tables (clusterData packs the padded bounds).
  std::vector<glm::vec4> positions4(pg.decodedPositions.size() / 3);
  for (usize v = 0; v < positions4.size(); ++v)
    positions4[v] =
        glm::vec4(pg.decodedPositions[v * 3], pg.decodedPositions[v * 3 + 1],
                  pg.decodedPositions[v * 3 + 2], 0.0f);
  const std::vector<glm::vec4> aabbs =
      renderer::computeClusterAabbs(positions4, pg.decodedIndices, clusterData);

  std::vector<renderer::CullGpuInstance> instances;
  instances.push_back(renderer::CullGpuInstance{});
  const std::vector<renderer::GpuMeshRangePair> meshRanges = {
      {0u, static_cast<u32>(dag.clusters.size())}};
  const glm::vec3 cam(2.5f, 3.0f, 2.0f);
  const u32 W = 64, H = 48;
  const f32 fov = 60.0f, threshold = 1.0f;
  const f32 worldPerPixel =
      2.0f * std::tan(glm::radians(fov) * 0.5f) / static_cast<f32>(H);
  glm::mat4 view = glm::lookAtRH(cam, {0, 0, 0}, {0, 1, 0});
  glm::mat4 proj =
      glm::perspectiveRH_ZO(glm::radians(fov), 4.0f / 3.0f, 0.1f, 100.0f);
  proj[1][1] *= -1.0f;
  glm::vec4 planes[6];
  renderer::extractFrustumPlanes(proj * view, planes);
  const std::vector<u8> hierarchyData(
      reinterpret_cast<const u8*>(dag.hierarchyNodes.data()),
      reinterpret_cast<const u8*>(dag.hierarchyNodes.data()) +
          dag.hierarchyNodes.size() *
              sizeof(kenshin::geometry::DagHierarchyNode));
  std::vector<glm::uvec2> groupRanges;
  for (const auto& g : dag.groups)
    groupRanges.emplace_back(g.clusters.empty() ? 0 : g.clusters.front(),
                             static_cast<u32>(g.clusters.size()));
  const std::vector<u32> meshHierRoots{dag.hierarchyRoot};

  renderer::ChainInputs inputs;
  inputs.view = view;
  inputs.proj = proj;
  inputs.nearZ = 0.1f;
  inputs.camPos = cam;
  inputs.pixelThreshold = threshold;
  inputs.worldPerPixel = worldPerPixel;
  inputs.width = W;
  inputs.height = H;
  for (int p = 0; p < 6; ++p) inputs.planes[p] = planes[p];
  inputs.hwMode = 0;
  inputs.cullMode = 1;
  backend::FrameBeginInfo frameInfo;
  frameInfo.width = W;
  frameInfo.height = H;
  const auto runFrames = [&](renderer::VisBufferChain& chain) {
    REQUIRE(backend->beginFrame(frameInfo));
    chain.execute(*backend, inputs);
    backend->endFrame();
    REQUIRE(backend->beginFrame(frameInfo));
    chain.execute(*backend, inputs);
    backend->endFrame();
  };

  std::vector<u8> visV2Bytes;
  u32 statsV2[16] = {};
  viscmp::Lists listsV2;
  {
    renderer::VisBufferChain chain;
    REQUIRE(chain.init(
        *backend, shaders, positions4, pg.decodedIndices, clusterData,
        groupData, hierarchyData, groupRanges, meshHierRoots, instances,
        meshRanges, static_cast<u32>(dag.clusters.size()), {}, nullptr, W, H));
    runFrames(chain);
    REQUIRE(backend->readbackBuffer(chain.visBuffer(*backend), 0,
                                    usize(W) * H * sizeof(u64), visV2Bytes));
    REQUIRE(chain.readStats(*backend, statsV2));
    REQUIRE(statsV2[2] > 0);
    REQUIRE(statsV2[5] > 0);
    REQUIRE(chain.readVisiblePairs(*backend, listsV2.pairs));
    REQUIRE(chain.readHwPairs(*backend, listsV2.hw));
    REQUIRE(chain.readPostPairs(*backend, listsV2.post));
    chain.onShutdown(*backend);
  }

  // Slot pool + identity page table + static tables; pages ride through
  // the M6.2 batch API (1460 pages / 64-per-batch = 23 batches).
  const u32 slotPool =
      backend->createPageSlotPool(pg.pageCount, geometry::kPageSlotSize);
  REQUIRE(slotPool != backend::IRenderBackend::kInvalidBufferHandle);
  for (u32 base = 0; base < pg.pageCount;) {
    const u32 n = std::min<u32>(64, pg.pageCount - base);
    REQUIRE(backend->beginPageUploads(slotPool));
    for (u32 i = 0; i < n; ++i)
      REQUIRE(
          backend->copyPageToSlot(pg.pages[base + i].data(), base + i,
                                  static_cast<u32>(pg.pages[base + i].size())));
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
  const u32 clusterPageBuf =
      makeStatic(pg.clusterPage.data(), pg.clusterPage.size() * 4, "pg-cp");
  std::vector<u32> pageTableInit(pg.pageCount);
  for (u32 p = 0; p < pg.pageCount; ++p) pageTableInit[p] = p;
  backend::IRenderBackend::BufferDesc ptDesc;
  ptDesc.size = usize(pg.pageCount) * 4;
  ptDesc.usage = backend::IRenderBackend::BufferStorage |
                 backend::IRenderBackend::BufferHostVisible;
  ptDesc.initialData = pageTableInit.data();
  ptDesc.debugName = "pg-pt";
  const u32 pageTableBuf = backend->createBuffer(ptDesc);
  std::vector<u32> pageHeaderFlat;
  pageHeaderFlat.reserve(usize(pg.pageCount) * 12);
  for (const auto& m : pg.pageMeta) {
    const u32 stepBits = *reinterpret_cast<const u32*>(&m.posStep);
    pageHeaderFlat.insert(
        pageHeaderFlat.end(),
        {m.posBase, m.nrmBase, m.uvBase, m.idxBase, m.metaBase, m.vertexCount,
         m.indexCount, m.clusterCount, stepBits, m.flags, m.pad0, m.pad1});
  }
  const u32 pageHeaderBuf =
      makeStatic(pageHeaderFlat.data(), pageHeaderFlat.size() * 4, "pg-ph");
  const u32 clusterMetaBuf = makeStatic(
      pg.clusterMetaIndex.data(), pg.clusterMetaIndex.size() * 4, "pg-cm");

  renderer::ChainPagedInputs pi;
  pi.slotPool = slotPool;
  pi.clusterPage = clusterPageBuf;
  pi.pageTable = pageTableBuf;
  pi.pageHeader = pageHeaderBuf;
  pi.pageCount = pg.pageCount;
  pi.slotU32Total = pg.pageCount * (geometry::kPageSlotSize / 4);
  pi.clusterMeta = clusterMetaBuf;

  {
    renderer::VisBufferChain chain;
    REQUIRE(chain.init(*backend, shaders, {}, {}, pagedClusterData, groupData,
                       hierarchyData, groupRanges, meshHierRoots, instances,
                       meshRanges, static_cast<u32>(dag.clusters.size()), aabbs,
                       &pi, W, H));
    runFrames(chain);
    std::vector<u8> visBytes;
    REQUIRE(backend->readbackBuffer(chain.visBuffer(*backend), 0,
                                    usize(W) * H * sizeof(u64), visBytes));
    u32 stats[16] = {};
    REQUIRE(chain.readStats(*backend, stats));
    REQUIRE(stats[2] == statsV2[2]);
    REQUIRE(stats[5] == statsV2[5]);
    REQUIRE(stats[15] == 0);  // every page resident: zero skips
    viscmp::Lists lists;
    REQUIRE(chain.readVisiblePairs(*backend, lists.pairs));
    REQUIRE(chain.readHwPairs(*backend, lists.hw));
    REQUIRE(chain.readPostPairs(*backend, lists.post));
    const usize mismatches = viscmp::mismatches(
        reinterpret_cast<const u64*>(visV2Bytes.data()), listsV2,
        reinterpret_cast<const u64*>(visBytes.data()), lists,
        visV2Bytes.size() / 8);
    INFO("paged big all-resident: " << mismatches << " vis mismatches over "
                                    << pg.pageCount << " pages");
    REQUIRE(mismatches == 0);
    chain.onShutdown(*backend);
  }

  backend->destroyBuffer(slotPool);
  backend->destroyBuffer(clusterPageBuf);
  backend->destroyBuffer(pageTableBuf);
  backend->destroyBuffer(pageHeaderBuf);
  backend->destroyBuffer(clusterMetaBuf);
  backend->shutdown();
}

// ── M6.4 saturation: pair lists capped at 600 against ~3000 visible
// pairs (far above any per-level width so the traversal does not
// starve, far below the leaf total so the appends overflow). The
// writers must saturate (drop, never write out of bounds) and every
// consumer dispatch clamps at the cap. The frame completes with the
// rasterized 600, nothing faults.
TEST_CASE("pair-list saturation at tiny capacity", "[gpu][vgchain]") {
  using namespace kenshin;
  Built built;
  REQUIRE(
      build(geometry::TorusKnotGeometry::create(2, 3, 128, 32, 1.0f), built));
  auto& dag = built.dag;  // packPages pads the cull bounds in place

  manager::ShaderManager shaders;
  REQUIRE(shaders.init(kShaderRoot));
  auto backend = backend::createVulkanBackend();
  backend::BackendInitParams params;
  REQUIRE(backend->initialize(params));

  std::vector<u8> clusterData, groupData;
  for (const auto& c : dag.clusters)
    renderer::packGpuCluster(clusterData, c.group, c.boundsSphere.x,
                             c.boundsSphere.y, c.boundsSphere.z,
                             c.boundsSphere.w, c.error, c.parentError,
                             c.vertexOffset, c.vertexCount, c.indexOffset,
                             c.indexCount, c.refinedGroup);
  for (const auto& g : dag.groups)
    renderer::packGpuGroup(groupData, g.boundsSphere.x, g.boundsSphere.y,
                           g.boundsSphere.z, g.boundsSphere.w, g.maxError,
                           g.simplifyError, static_cast<u32>(g.lodLevel));
  std::vector<glm::vec4> positions4(dag.positions.size() / 3);
  for (usize v = 0; v < positions4.size(); ++v)
    positions4[v] = glm::vec4(dag.positions[v * 3], dag.positions[v * 3 + 1],
                              dag.positions[v * 3 + 2], 0.0f);
  std::vector<renderer::CullGpuInstance> instances;
  for (u32 i = 0; i < 24; ++i) instances.push_back(renderer::CullGpuInstance{});
  const std::vector<renderer::GpuMeshRangePair> meshRanges = {
      {0u, static_cast<u32>(dag.clusters.size())}};

  const glm::vec3 cam(2.5f, 3.0f, 2.0f);
  const u32 W = 64, H = 48;
  const f32 fov = 60.0f;
  const f32 worldPerPixel =
      2.0f * std::tan(glm::radians(fov) * 0.5f) / static_cast<f32>(H);
  glm::mat4 view = glm::lookAtRH(cam, {0, 0, 0}, {0, 1, 0});
  glm::mat4 proj =
      glm::perspectiveRH_ZO(glm::radians(fov), 4.0f / 3.0f, 0.1f, 100.0f);
  proj[1][1] *= -1.0f;
  glm::vec4 planes[6];
  renderer::extractFrustumPlanes(proj * view, planes);
  const std::vector<u8> hierarchyData(
      reinterpret_cast<const u8*>(dag.hierarchyNodes.data()),
      reinterpret_cast<const u8*>(dag.hierarchyNodes.data()) +
          dag.hierarchyNodes.size() *
              sizeof(kenshin::geometry::DagHierarchyNode));
  std::vector<glm::uvec2> groupRanges;
  for (const auto& g : dag.groups)
    groupRanges.emplace_back(g.clusters.empty() ? 0 : g.clusters.front(),
                             static_cast<u32>(g.clusters.size()));

  for (const u32 cullMode : {0u, 1u, 2u}) {
    renderer::VisBufferChain chain;
    const std::vector<kenshin::u32> hierRoots{dag.hierarchyRoot};
    REQUIRE(chain.init(*backend, shaders, positions4, dag.indices, clusterData,
                       groupData, hierarchyData, groupRanges, hierRoots,
                       instances, meshRanges,
                       /*maxPairs=*/600, {}, nullptr, W, H));
    renderer::ChainInputs inputs;
    inputs.view = view;
    inputs.proj = proj;
    inputs.nearZ = 0.1f;
    inputs.camPos = cam;
    inputs.pixelThreshold = 1.0f;
    inputs.worldPerPixel = worldPerPixel;
    inputs.width = W;
    inputs.height = H;
    for (int p = 0; p < 6; ++p) inputs.planes[p] = planes[p];
    inputs.hwMode = 0;
    inputs.cullMode = cullMode;
    backend::FrameBeginInfo frameInfo;
    frameInfo.width = W;
    frameInfo.height = H;
    REQUIRE(backend->beginFrame(frameInfo));
    chain.execute(*backend, inputs);
    backend->endFrame();
    REQUIRE(backend->beginFrame(frameInfo));
    chain.execute(*backend, inputs);
    backend->endFrame();
    u32 stats[16] = {};
    REQUIRE(chain.readStats(*backend, stats));
    // Saturation PROOF: the raw count exceeded the 5-pair capacity
    // (reservations past it were dropped by the writers).
    INFO("cullMode " << cullMode << ": pairs " << stats[2] << " visible "
                     << stats[0] << " visited " << stats[13] << " pruned "
                     << stats[14] << " tooCoarse " << stats[3]);
    // Saturation PROOF (hierarchy modes process everything and cap the
    // WRITES: the raw count runs past the capacity; the linear mode
    // saturates UPSTREAM in the expand - only 5 items ever exist).
    if (cullMode != 0u) REQUIRE(stats[2] > 600);
    // The raster dispatched the clamped envelope: pixel writes only
    // provable for the hierarchy modes (the linear path's 5 surviving
    // work items may all be back-facing - completion is its proof).
    if (cullMode != 0u) REQUIRE(stats[5] > 0);
    chain.onShutdown(*backend);
  }
  backend->shutdown();
}
