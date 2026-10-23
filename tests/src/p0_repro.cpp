// P0 hole repro: the USER camera (inside the grid64) - the in-frame
// chain vs the one-shot reference path, resolved-attribute image
// differential. The one-shot path is the proven ground truth; any
// structural difference convicts the chain.
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <glm/gtc/type_ptr.hpp>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/geometry/ClusterDag.h"
#include "kenshin/geometry/KGeoFile.h"
#include "kenshin/manager/ShaderManager.h"
#include "kenshin/renderer/ClusterCullPass.h"
#include "kenshin/renderer/VirtualGeometryScene.h"
#include "kenshin/renderer/InstanceCullPass.h"
#include "kenshin/renderer/SoftRasterPass.h"
#include "kenshin/renderer/VisBufferChain.h"
#include "vis_compare.h"

namespace {
const char* kShaderRoot = KENSHIN_SOURCE_DIR "/engine/shaders";
}

TEST_CASE("P0 user-camera chain vs one-shot (grid64)", "[gpu][p0]") {
  using namespace kenshin;
  geometry::ClusterDag dag;
  std::string log;
  REQUIRE(geometry::readKGeo("out/bigknot.kgeo", dag, log));
  INFO(log);

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

  // The grid64 layout (depth-major 8 wide, step 2.25) + the user's
  // in-grid camera.
  constexpr u32 kInst = 64;
  std::vector<renderer::CullGpuInstance> instances;
  const glm::vec4 meshBounds = [&] {
    glm::vec4 b(0, 0, 0, -1);
    for (const auto& c : dag.clusters) {
      const glm::vec4 bs(c.boundsSphere);
      const glm::vec3 pa(b), pb(bs);
      const f32 d = glm::distance(pa, pb);
      if (b.w < 0)
        b = bs;
      else if (d + b.w < bs.w)
        b = bs;
      else if (d + bs.w > b.w) {
        const glm::vec3 dir = d > 0 ? (pb - pa) / d : glm::vec3(0);
        b = glm::vec4(pa + dir * (d + b.w - bs.w) * 0.5f,
                      (b.w + bs.w + d) * 0.5f);
      }
    }
    return b;
  }();
  for (u32 i = 0; i < kInst; ++i) {
    renderer::CullGpuInstance inst;
    const glm::vec3 o((i % 8) * 2.25f, 0.0f, (i / 8) * 2.25f);
    inst.model = glm::translate(glm::mat4(1.0f), o);
    inst.bounds = glm::vec4(glm::vec3(inst.model * glm::vec4(glm::vec3(meshBounds), 1)),
                            meshBounds.w);
    instances.push_back(inst);
  }
  const std::vector<renderer::GpuMeshRangePair> meshRanges = {
      {0u, static_cast<u32>(dag.clusters.size())}};

  const glm::vec3 cam(0.27f, 0.0f, 6.07f);   // the user's logged position
  const glm::vec3 target(8.0f, 0.0f, 4.0f);  // the orbit target
  const u32 W = 640, H = 461;
  const f32 fov = 60.0f, threshold = 1.0f;
  const f32 worldPerPixel =
      2.0f * std::tan(glm::radians(fov) * 0.5f) / static_cast<f32>(H);
  // The EXACT default-camera view matrix as logged by the app
  // (Z-up controller convention - a lookAt rebuild is NOT equivalent).
  glm::mat4 view(1.0f);
  {
    const float v[16] = {-0.0000f, +0.2588f, -0.9659f, +0.0000f,
                         -1.0000f, -0.0000f, +0.0000f, +0.0000f,
                         +0.0000f, +0.9659f, +0.2588f, +0.0000f,
                         +0.0000f, -5.9343f, -1.3079f, +1.0000f};
    // glm::value_ptr layout is column-major; logged vm[col][row] in
    // order c0,c1,c2,c3 - matches.
    std::memcpy(glm::value_ptr(view), v, sizeof(v));
  }
  glm::mat4 proj =
      glm::perspectiveRH_ZO(glm::radians(fov), f32(W) / f32(H), 0.1f, 100.0f);
  proj[1][1] *= -1.0f;
  glm::vec4 planes[6];
  renderer::extractFrustumPlanes(proj * view, planes);

  // ---- The one-shot reference.
  std::vector<u64> visRef;
  std::vector<glm::uvec2> pairsRef;
  {
    std::vector<u32> visible;
    REQUIRE(renderer::cullInstancesOnGpu(*backend, shaders, instances, planes,
                                         visible));
    INFO("visible " << visible.size());
    std::vector<glm::uvec2> workItems;
    REQUIRE(renderer::expandWorkItemsOnGpu(*backend, shaders, visible,
                                           instances, meshRanges, workItems));
    u32 cullStats[4] = {};
    REQUIRE(renderer::cullClustersOnGpu(
        *backend, shaders, cam, threshold, worldPerPixel, clusterData,
        groupData, workItems, instances, pairsRef, cullStats));
    INFO("reference pairs " << pairsRef.size());
    const renderer::SoftRasterFrame frame{view, proj, 0.1f, W, H, true};
    u32 rasterStats[4] = {};
    REQUIRE(renderer::rasterizeOnGpu(*backend, shaders, frame, positions4,
                                     dag.indices, clusterData, instances,
                                     pairsRef, visRef, rasterStats));
    INFO("reference px writes " << rasterStats[1]);
    REQUIRE(rasterStats[1] > 0);
  }

  // ---- The chain (level-wise, occ on and off variants).
  const std::vector<u8> hierarchyData(
      reinterpret_cast<const u8*>(dag.hierarchyNodes.data()),
      reinterpret_cast<const u8*>(dag.hierarchyNodes.data()) +
          dag.hierarchyNodes.size() * sizeof(geometry::DagHierarchyNode));
  std::vector<glm::uvec2> groupRanges;
  for (const auto& g : dag.groups)
    groupRanges.emplace_back(g.clusters.empty() ? 0 : g.clusters.front(),
                             static_cast<u32>(g.clusters.size()));
  for (const bool occ : {true, false}) {
    renderer::VisBufferChain chain;
    REQUIRE(chain.init(*backend, shaders, positions4, dag.indices, clusterData,
                       groupData, hierarchyData, groupRanges,
                       std::vector<u32>{dag.hierarchyRoot}, instances, meshRanges,
                       u32(kInst) * u32(dag.clusters.size()), {}, nullptr, W,
                       H));
    renderer::ChainInputs ci;
    ci.view = view;
    ci.proj = proj;
    ci.nearZ = 0.1f;
    ci.camPos = cam;
    ci.pixelThreshold = threshold;
    ci.worldPerPixel = worldPerPixel;
    ci.width = W;
    ci.height = H;
    for (int p = 0; p < 6; ++p) ci.planes[p] = planes[p];
    ci.hwMode = 0;
    ci.cullMode = 1;
    ci.occlusionEnabled = occ;
    backend::FrameBeginInfo fi;
    fi.width = W;
    fi.height = H;
    for (int f = 0; f < 4; ++f) {
      REQUIRE(backend->beginFrame(fi) == kenshin::backend::FrameStatus::Ready);
      chain.execute(*backend, ci);
      backend->endFrame();
    }
    std::vector<u8> visBytes;
    REQUIRE(backend->readbackBuffer(chain.visBuffer(*backend), 0,
                                    usize(W) * H * sizeof(u64), visBytes));
    u32 st[16] = {};
    REQUIRE(chain.readStats(*backend, st));
    viscmp::Lists lists;
    REQUIRE(chain.readVisiblePairs(*backend, lists.pairs));
    REQUIRE(chain.readHwPairs(*backend, lists.hw));
    REQUIRE(chain.readPostPairs(*backend, lists.post));
    const viscmp::Lists refLists{pairsRef, {}, {}};
    const usize mismatches = viscmp::mismatches(
        visRef.data(), refLists,
        reinterpret_cast<const u64*>(visBytes.data()), lists, visRef.size());
    INFO("occ " << occ << ": chain pairs " << st[2] << " (ref "
                << pairsRef.size() << "), resolved mismatches " << mismatches);
    REQUIRE(st[2] > 0);
    REQUIRE(mismatches == 0);
    chain.onShutdown(*backend);
  }

  // ---- P0: the HW raster path must actually write pixels. All-HW at
  // the same view; occupancy must match the SW reference's footprint.
  {
    renderer::VisBufferChain chain;
    REQUIRE(chain.init(*backend, shaders, positions4, dag.indices, clusterData,
                       groupData, hierarchyData, groupRanges,
                       std::vector<u32>{dag.hierarchyRoot}, instances, meshRanges,
                       u32(kInst) * u32(dag.clusters.size()), {}, nullptr, W,
                       H));
    renderer::ChainInputs ci;
    ci.view = view;
    ci.proj = proj;
    ci.nearZ = 0.1f;
    ci.camPos = cam;
    ci.pixelThreshold = threshold;
    ci.worldPerPixel = worldPerPixel;
    ci.width = W;
    ci.height = H;
    for (int p = 0; p < 6; ++p) ci.planes[p] = planes[p];
    ci.hwMode = 2;  // force all-HW: every pair through the null pass
    ci.cullMode = 1;
    ci.occlusionEnabled = false;
    backend::FrameBeginInfo fi;
    fi.width = W;
    fi.height = H;
    for (int f = 0; f < 4; ++f) {
      REQUIRE(backend->beginFrame(fi) == kenshin::backend::FrameStatus::Ready);
      chain.execute(*backend, ci);
      backend->endFrame();
    }
    std::vector<u8> visBytes;
    REQUIRE(backend->readbackBuffer(chain.visBuffer(*backend), 0,
                                    usize(W) * H * sizeof(u64), visBytes));
    u32 st[16] = {};
    REQUIRE(chain.readStats(*backend, st));
    const u64* hw =
        reinterpret_cast<const u64*>(visBytes.data());
    u64 nonzero = 0;
    for (usize i = 0; i < usize(W) * H; ++i) nonzero += hw[i] != 0 ? 1 : 0;
    u64 refNonzero = 0;
    for (const u64 v : visRef) refNonzero += v != 0 ? 1 : 0;
    INFO("all-HW: hw pairs " << st[12] << ", VisBuffer nonzero " << nonzero
                             << " (SW ref " << refNonzero << "), PS entries "
                             << st[15]);
    printf("[p0hw] hw=%u nz=%llu ref=%llu ps=%u | tri0=%u wle0=%u zeroA=%u "
           "st8=%u st9=%u st10=%u st11=%u st13=%u st14=%u\n",
           st[12], (unsigned long long)nonzero, (unsigned long long)refNonzero,
           st[15], st[7], st[6], st[5], st[8], st[9], st[10], st[11], st[13],
           st[14]);
    REQUIRE(st[12] > 0);
    REQUIRE(nonzero > refNonzero * 3 / 4);  // the notches must not recur
    chain.onShutdown(*backend);
  }
  backend->shutdown();
}
