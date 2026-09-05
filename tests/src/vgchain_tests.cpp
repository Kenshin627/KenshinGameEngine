// B2 differential: the IN-FRAME VisBufferChain vs the one-shot chain
// (the S4.2-S4.4 reference path), windowless via the headless frame
// support. Both are GPU executions of the same shaders with the same
// inputs, driven differently (persistent pipelines + indirect
// dispatches + host-visible params vs per-call one-shot everything) -
// the VisBuffers must match EXACTLY, pixel for pixel.

#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/geometry/ClusterDag.h"
#include "kenshin/geometry/ClusterLodBuilder.h"
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

  // ---- Subject: the in-frame chain, on a headless frame.
  renderer::VisBufferChain chain;
  REQUIRE(chain.init(*backend, shaders, positions4, dag.indices, clusterData,
                     groupData, instances, meshRanges,
                     static_cast<u32>(2 * dag.clusters.size()), W, H));
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

  backend::FrameBeginInfo frameInfo;
  frameInfo.width = W;
  frameInfo.height = H;
  REQUIRE(backend->beginFrame(frameInfo));
  chain.execute(*backend, inputs);
  backend->endFrame();

  // Second frame on the OTHER slot: proves per-slot isolation (the
  // first frame's intermediates must not leak into or be clobbered by
  // this one).
  REQUIRE(backend->beginFrame(frameInfo));
  chain.execute(*backend, inputs);
  backend->endFrame();

  std::vector<u8> visBytes;
  REQUIRE(backend->readbackBuffer(chain.visBuffer(*backend), 0,
                                  usize(W) * H * sizeof(u64), visBytes));
  const u64* visInFrame = reinterpret_cast<const u64*>(visBytes.data());
  u32 stats[13] = {};
  REQUIRE(chain.readStats(*backend, stats));
  u32 ax[3], cx[1], rx[1];
  REQUIRE(chain.readIndirectArgs(*backend, ax, cx, rx));
  INFO("chain stats: visibleInst "
       << stats[0] << ", workItems " << stats[1] << ", pairs " << stats[2]
       << ", writes " << stats[5] << "; indirect x: expand " << ax[0]
       << ", cull " << cx[0] << ", raster " << rx[0] << "; expCnt cursor "
       << ax[1] << ", fwd " << ax[2]);
  REQUIRE(stats[0] > 0);  // instances survived the cull
  REQUIRE(stats[2] > 0);  // clusters cut
  REQUIRE(stats[5] > 0);  // px writes

  usize mismatches = 0;
  for (usize i = 0; i < visOneShot.size(); ++i)
    if (visOneShot[i] != visInFrame[i]) ++mismatches;
  INFO("vis pixels " << visOneShot.size() << ", mismatches " << mismatches);
  REQUIRE(mismatches == 0);

  chain.onShutdown(*backend);
  backend->shutdown();
}
