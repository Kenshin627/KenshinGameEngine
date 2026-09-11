// B2 differential: the IN-FRAME VisBufferChain vs the one-shot chain
// (the S4.2-S4.4 reference path), windowless via the headless frame
// support. Both are GPU executions of the same shaders with the same
// inputs, driven differently (persistent pipelines + indirect
// dispatches + host-visible params vs per-call one-shot everything) -
// the VisBuffers must match EXACTLY, pixel for pixel.

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
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
  const auto& dag = built.dag;

  manager::ShaderManager shaders;
  REQUIRE(shaders.init(kShaderRoot));
  auto backend = backend::createVulkanBackend();
  backend::BackendInitParams params;  // windowless (headless frames)
  REQUIRE(backend->initialize(params));

  // Shared inputs: pools, two instances, one camera.
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
  {
    renderer::CullGpuInstance identity;
    instances.push_back(identity);
    renderer::CullGpuInstance moved;
    moved.model = glm::translate(glm::mat4(1.0f), {1.2f, 0.0f, 0.4f});
    instances.push_back(moved);
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
  {
    std::vector<u32> visible;
    REQUIRE(renderer::cullInstancesOnGpu(*backend, shaders, instances, planes,
                                         visible));
    REQUIRE(visible.size() == 2);
    std::vector<glm::uvec2> workItems;
    REQUIRE(renderer::expandWorkItemsOnGpu(*backend, shaders, visible,
                                           instances, meshRanges, workItems));
    REQUIRE(workItems.size() == 2 * dag.clusters.size());
    std::vector<glm::uvec2> pairs;
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
  REQUIRE(chain.init(*backend, shaders, positions4, dag.indices, clusterData,
                     groupData, hierarchyData, groupRanges, meshHierRoots,
                     instances, meshRanges,
                     static_cast<u32>(2 * dag.clusters.size()), {}, nullptr,
                     W, H));
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

  const auto runChainMode = [&](u32 cullMode) {
    inputs.cullMode = cullMode;
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
    const u64* visInFrame = reinterpret_cast<const u64*>(visBytes.data());
    u32 stats[16] = {};
    REQUIRE(chain.readStats(*backend, stats));
    INFO("cullMode " << cullMode << " chain stats: visibleInst " << stats[0]
                     << ", workItems " << stats[1] << ", pairs " << stats[2]
                     << ", writes " << stats[5]);
    REQUIRE(stats[0] > 0);  // instances survived the cull
    REQUIRE(stats[2] > 0);  // clusters cut
    REQUIRE(stats[5] > 0);  // px writes
    usize mismatches = 0;
    for (usize i = 0; i < visOneShot.size(); ++i)
      if (visOneShot[i] != visInFrame[i]) ++mismatches;
    INFO("cullMode " << cullMode << ": vis pixels " << visOneShot.size()
                     << ", mismatches " << mismatches);
    REQUIRE(mismatches == 0);
  };
  runChainMode(0);  // linear: the expand+per-cluster reference shape
  runChainMode(1);  // M5.2: the level-wise BVH8 traversal
  runChainMode(2);  // M5.3: the persistent-threads MPMC traversal

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
  const auto& dag = built.dag;

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
    renderer::packGpuCluster(pagedClusterData, cl.group, cl.boundsSphere.x,
                             cl.boundsSphere.y, cl.boundsSphere.z,
                             cl.boundsSphere.w, cl.error, cl.parentError,
                             pg.localVertexOffset[c], cl.vertexCount,
                             pg.localIndexOffset[c], cl.indexCount,
                             cl.refinedGroup);
  }
  std::vector<glm::vec4> positions4(dag.positions.size() / 3);
  for (usize v = 0; v < positions4.size(); ++v)
    positions4[v] = glm::vec4(dag.positions[v * 3], dag.positions[v * 3 + 1],
                              dag.positions[v * 3 + 2], 0.0f);
  const std::vector<glm::vec4> aabbs =
      renderer::computeClusterAabbs(positions4, dag.indices, clusterData);

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
  {
    renderer::VisBufferChain chain;
    REQUIRE(chain.init(*backend, shaders, positions4, dag.indices, clusterData,
                       groupData, hierarchyData, groupRanges, meshHierRoots,
                       instances, meshRanges,
                       static_cast<u32>(dag.clusters.size()), {}, nullptr, W,
                       H));
    runFrames(chain);
    REQUIRE(backend->readbackBuffer(chain.visBuffer(*backend), 0,
                                    usize(W) * H * sizeof(u64), visV2Bytes));
    REQUIRE(chain.readStats(*backend, statsV2));
    REQUIRE(statsV2[2] > 0);
    REQUIRE(statsV2[5] > 0);
    REQUIRE(statsV2[15] == 0);
    REQUIRE(chain.readVisiblePairs(*backend, pairsV2));
    chain.onShutdown(*backend);
  }

  // ---- Paged setup: slot pool + identity page table + static tables.
  const u32 slotPool = backend->createPageSlotPool(pg.pageCount,
                                                   geometry::kPageSlotSize);
  REQUIRE(slotPool != backend::IRenderBackend::kInvalidBufferHandle);
  {
    // Upload every page through the M6.2 batch API (<= 64 pages/batch).
    for (u32 base = 0; base < pg.pageCount;) {
      const u32 n =
          std::min<u32>(64, pg.pageCount - base);
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
  const u32 clusterPageBuf = makeStatic(pg.clusterPage.data(),
                                        pg.clusterPage.size() * 4, "pg-clusterPage");
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
  pageHeaderFlat.reserve(usize(pg.pageCount) * 6);
  for (const auto& m : pg.pageMeta)
    pageHeaderFlat.insert(pageHeaderFlat.end(),
                          {m.posBase, m.nrmBase, m.uvBase, m.idxBase,
                           m.vertexCount, m.indexCount});
  const u32 pageHeaderBuf = makeStatic(pageHeaderFlat.data(),
                                       pageHeaderFlat.size() * 4, "pg-pageHeader");

  renderer::ChainPagedInputs pi;
  pi.slotPool = slotPool;
  pi.clusterPage = clusterPageBuf;
  pi.pageTable = pageTableBuf;
  pi.pageHeader = pageHeaderBuf;
  pi.pageCount = pg.pageCount;
  pi.slotVertexTotal = pg.pageCount * (geometry::kPageSlotSize / 16);
  pi.slotIndexTotal = pg.pageCount * (geometry::kPageSlotSize / 4);

  // ---- Subject 1: all-resident paged chain - byte-identical to v2.
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
    REQUIRE(stats[2] == statsV2[2]);  // same cut
    REQUIRE(stats[5] == statsV2[5]);  // same px writes
    REQUIRE(stats[15] == 0);          // nothing skipped
    usize mismatches = 0;
    const u64* a = reinterpret_cast<const u64*>(visV2Bytes.data());
    const u64* b = reinterpret_cast<const u64*>(visBytes.data());
    for (usize i = 0; i < visV2Bytes.size() / 8; ++i)
      if (a[i] != b[i]) ++mismatches;
    INFO("paged all-resident: " << mismatches << " vis mismatches");
    REQUIRE(mismatches == 0);
    // Touched feedback: every page of a visible cluster is marked.
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
      REQUIRE(((touched[pg.clusterPage[pr.y] >> 5] >>
                (pg.clusterPage[pr.y] & 31)) &
               1u) == 1u);
    chain.onShutdown(*backend);
  }

  // ---- Subject 2: eviction - a page that provably carries CUT
  // clusters (chosen from the v2 pair set) goes absent: skips counted,
  // the touched bitmap still marks it (miss detection), and restoring
  // the page returns the VisBuffer to byte-identity (pop-in recovery).
  u32 evicted = 0xFFFFFFFFu;
  for (auto it = pairsV2.rbegin(); it != pairsV2.rend(); ++it)
    if (pg.clusterPage[it->y] != 0) {
      evicted = pg.clusterPage[it->y];
      break;
    }
  REQUIRE(evicted != 0xFFFFFFFFu);  // fine pages carry cut clusters here
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
    REQUIRE(stats[2] == statsV2[2]);            // cull untouched (metadata)
    std::vector<glm::uvec2> pairs;
    REQUIRE(chain.readVisiblePairs(*backend, pairs));
    u32 pendingExpected = 0;
    for (const auto& pr : pairs)
      if (pg.clusterPage[pr.y] == evicted) ++pendingExpected;
    INFO("evicted page " << evicted << ": pending expected "
                         << pendingExpected << ", got " << stats[15]
                         << " (pairs " << pairs.size() << ")");
    REQUIRE(pendingExpected > 0);        // the camera cuts fine clusters
    REQUIRE(stats[15] == pendingExpected);
    REQUIRE(stats[5] <= statsV2[5]);     // strictly fewer px writes
    // Miss detection: the absent page is still marked touched.
    std::vector<u32> touched;
    REQUIRE(chain.readTouchedPages(*backend, touched));
    REQUIRE(((touched[evicted >> 5] >> (evicted & 31)) & 1u) == 1u);
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
    usize mismatches = 0;
    const u64* a = reinterpret_cast<const u64*>(visV2Bytes.data());
    const u64* b = reinterpret_cast<const u64*>(visBytes.data());
    for (usize i = 0; i < visV2Bytes.size() / 8; ++i)
      if (a[i] != b[i]) ++mismatches;
    INFO("paged post-restore: " << mismatches << " vis mismatches");
    REQUIRE(mismatches == 0);
  }
  chain.onShutdown(*backend);

  backend->destroyBuffer(slotPool);
  backend->destroyBuffer(clusterPageBuf);
  backend->destroyBuffer(pageTableBuf);
  backend->destroyBuffer(pageHeaderBuf);
  backend->shutdown();
}
