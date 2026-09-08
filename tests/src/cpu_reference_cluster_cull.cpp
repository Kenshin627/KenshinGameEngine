// S4.3 differential: instance-aware cluster culling, GPU compute vs CPU
// reference running the identical Karis p.70 rule. Headless (windowless
// device + runtime Slang). Cases: identity / translated / 2x-scaled
// instances (all must survive the instance cull) + one instance-culled
// away (its clusters must not appear). The identity instance's GPU cut
// must ALSO match the engine's CPU selectLodCut cluster-for-cluster
// (the V3 "same count as S3.4" check, promoted to an assertion).

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdio>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <map>
#include <set>
#include <vector>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/geometry/ClusterDag.h"
#include "kenshin/geometry/ClusterLodBuilder.h"
#include "kenshin/geometry/LodCut.h"
#include "kenshin/geometry/shapes/TorusKnotGeometry.h"
#include "kenshin/manager/ShaderManager.h"
#include "kenshin/renderer/ClusterCullPass.h"
#include "kenshin/renderer/InstanceCullPass.h"
#include "kenshin/renderer/VirtualGeometryScene.h"
#include "kenshin/renderer/VisBufferChain.h"

namespace {

const char* kShaderRoot = KENSHIN_SOURCE_DIR "/engine/shaders";

struct Built {
  kenshin::assets::MeshData mesh;
  std::vector<kenshin::geometry::ClusterGroup> groups;
  kenshin::geometry::ClusterDag dag;
  std::string log;
};

bool build(const kenshin::assets::MeshData& mesh, Built& out) {
  out.mesh = mesh;
  kenshin::geometry::BuildStats stats;
  kenshin::geometry::DagBuildStats dagStats;
  return kenshin::geometry::buildClusterLod(
             out.mesh,
             [&](const kenshin::geometry::ClusterGroup& g) {
               out.groups.push_back(g);
             },
             stats, out.log) &&
         kenshin::geometry::assembleClusterDag(out.mesh, out.groups, out.dag,
                                               dagStats, out.log);
}

/// CPU reference - the exact shader math: fireable(group) via the
/// group's instance-transformed decision sphere, select per the
/// refined-source rule.
std::vector<glm::uvec2> cullCpu(
    const kenshin::geometry::ClusterDag& dag,
    const std::vector<kenshin::renderer::CullGpuInstance>& instances,
    const std::vector<kenshin::u32>& visible, const glm::vec3& cam,
    float pixelThreshold, float worldPerPixel) {
  std::vector<glm::uvec2> out;
  for (const kenshin::u32 idx : visible) {
    const glm::mat4& m = instances[idx].model;
    const float scale =
        std::max({glm::length(m[0]), glm::length(m[1]), glm::length(m[2])});
    const auto fireable = [&](kenshin::u32 g) {
      const auto& group = dag.groups[g];
      // w = 1 (a POINT): glm::vec4(boundsSphere) would smuggle the
      // RADIUS into w and scale the translation by it.
      const glm::vec3 c(m * glm::vec4(group.boundsSphere.x,
                                      group.boundsSphere.y,
                                      group.boundsSphere.z, 1.0f));
      const float dist =
          std::max(glm::distance(c, cam) - group.boundsSphere.w * scale, 0.1f);
      return group.simplifyError > pixelThreshold * worldPerPixel * dist;
    };
    for (const auto& cluster : dag.clusters) {
      if (!fireable(cluster.group)) continue;
      if (cluster.refinedGroup >= 0 && fireable(cluster.refinedGroup)) continue;
      out.emplace_back(
          idx, static_cast<kenshin::u32>(&cluster - dag.clusters.data()));
    }
  }
  return out;
}

using PairKey = std::uint64_t;
PairKey key(const glm::uvec2& p) {
  return (static_cast<std::uint64_t>(p.x) << 32) | p.y;
}

}  // namespace

TEST_CASE("cluster cull GPU/CPU differential", "[gpu][clustercull]") {
  using namespace kenshin;
  Built built;
  REQUIRE(
      build(geometry::TorusKnotGeometry::create(2, 3, 128, 32, 1.0f), built));
  const auto& dag = built.dag;

  manager::ShaderManager shaders;
  REQUIRE(shaders.init(kShaderRoot));
  auto backend = backend::createVulkanBackend();
  backend::BackendInitParams params;  // windowless
  REQUIRE(backend->initialize(params));

  // View: camera near the knot (the regime that mixes per-group scales).
  const glm::vec3 cam(2.5f, 3.0f, 2.0f);
  const float pixelThreshold = 1.0f;
  const float worldPerPixel =
      2.0f * std::tan(glm::radians(60.0f) * 0.5f) / 720.0f;

  // Instances: identity, translated, 2x scaled (visible) + far (culled
  // by the S4.2 instance pass; none of its clusters may appear).
  std::vector<renderer::CullGpuInstance> instances;
  const auto push = [&](glm::mat4 model, glm::vec3 c, float r) {
    renderer::CullGpuInstance in;
    in.model = model;
    in.bounds = glm::vec4(c, r);
    instances.push_back(in);
  };
  push(glm::mat4(1.0f), {0.0f, 0.0f, 0.0f}, 3.2f);           // 0 identity
  push(glm::translate(glm::mat4(1.0f), {5.0f, 0.0f, 0.0f}),  // 1 moved
       {5.0f, 0.0f, 0.0f}, 3.2f);
  push(glm::scale(glm::mat4(1.0f), {2.0f, 2.0f, 2.0f}),  // 2 scaled
       {0.0f, 0.0f, 0.0f}, 6.4f);
  push(glm::translate(glm::mat4(1.0f), {-40.0f, 0.0f, 0.0f}),  // 3 far
       {-40.0f, 0.0f, 0.0f}, 3.2f);

  // Pack the GPU tables with the shared single-point-of-truth helpers.
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
  const std::vector<renderer::GpuMeshRangePair> meshRanges = {
      {0u, static_cast<u32>(dag.clusters.size())}};

  // Instance cull (S4.2) -> visible.
  glm::vec4 planes[6];
  {
    const glm::mat4 view =
        glm::lookAtRH(cam, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
    glm::mat4 proj =
        glm::perspectiveRH_ZO(glm::radians(60.0f), 16.0f / 9.0f, 0.1f, 100.0f);
    proj[1][1] *= -1.0f;
    renderer::extractFrustumPlanes(proj * view, planes);
  }
  std::vector<u32> visible;
  REQUIRE(renderer::cullInstancesOnGpu(*backend, shaders, instances, planes,
                                       visible));
  REQUIRE(visible.size() == 3);  // the far instance must be gone

  // Expand -> cull on the GPU.
  std::vector<glm::uvec2> workItems;
  REQUIRE(renderer::expandWorkItemsOnGpu(*backend, shaders, visible, instances,
                                         meshRanges, workItems));
  REQUIRE(workItems.size() == 3 * dag.clusters.size());

  std::vector<glm::uvec2> gpuPairs;
  u32 stats[3] = {};
  REQUIRE(renderer::cullClustersOnGpu(*backend, shaders, cam, pixelThreshold,
                                      worldPerPixel, clusterData, groupData,
                                      workItems, instances, gpuPairs, stats));
  REQUIRE(stats[0] == gpuPairs.size());

  // CPU reference, same inputs.
  const std::vector<glm::uvec2> cpuPairs =
      cullCpu(dag, instances, visible, cam, pixelThreshold, worldPerPixel);

  std::multiset<PairKey> cpuSet, gpuSet;
  for (const auto& p : cpuPairs) cpuSet.insert(key(p));
  for (const auto& p : gpuPairs) gpuSet.insert(key(p));
  INFO("cpu visible " << cpuSet.size() << ", gpu visible " << gpuSet.size()
                      << " (culled too-coarse " << stats[1]
                      << ", source-too-coarse " << stats[2] << ")");
  {
    std::vector<PairKey> onlyCpu, onlyGpu;
    std::set_difference(cpuSet.begin(), cpuSet.end(), gpuSet.begin(),
                        gpuSet.end(), std::back_inserter(onlyCpu));
    std::set_difference(gpuSet.begin(), gpuSet.end(), cpuSet.begin(),
                        cpuSet.end(), std::back_inserter(onlyGpu));
    std::string diff;
    for (const PairKey k : onlyCpu)
      diff += "cpu-only i" + std::to_string(k >> 32) + "c" +
              std::to_string(k & 0xFFFFFFFF) + " ";
    for (const PairKey k : onlyGpu)
      diff += "gpu-only i" + std::to_string(k >> 32) + "c" +
              std::to_string(k & 0xFFFFFFFF) + " ";
  }
  REQUIRE(cpuSet == gpuSet);

  // Instance 3 (far) must contribute nothing even if it had survived.
  for (const auto& p : gpuPairs) REQUIRE(p.x != 3u);

  // V3 promoted to an assertion: the identity instance's GPU cut must
  // equal the engine's CPU selectLodCut cluster-for-cluster.
  {
    const float camArr[3] = {cam.x, cam.y, cam.z};
    std::vector<u32> cpuCut;
    geometry::selectLodCut(dag, pixelThreshold, 720, 60.0f, camArr, cpuCut);
    std::set<u32> expected;
    for (const u32 c : cpuCut) expected.insert(c);
    std::multiset<u32> identityClusters;
    for (const auto& p : gpuPairs)
      if (p.x == 0u) identityClusters.insert(p.y);
    REQUIRE(identityClusters.size() == expected.size());
    for (const u32 c : identityClusters) REQUIRE(expected.count(c) == 1);
    INFO("identity cut: gpu " << identityClusters.size() << " == selectLodCut "
                              << expected.size());
  }

  backend->shutdown();
}

// M5.2: the hierarchy chain's identity-instance cut must equal the CPU
// selectLodCut cluster-for-cluster (the permanent LOD semantic ground
// truth), across a threshold sweep. Occlusion off + pure SW: the cut
// is then purely the LOD rule, exactly what selectLodCut computes.
TEST_CASE("hierarchy chain identity cut == selectLodCut",
          "[gpu][vgchain][lodcut]") {
  using namespace kenshin;
  Built built;
  REQUIRE(
      build(geometry::TorusKnotGeometry::create(2, 3, 128, 32, 1.0f), built));
  const auto& dag = built.dag;

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
  std::vector<glm::vec4> positions4(dag.positions.size() / 3);
  for (usize v = 0; v < positions4.size(); ++v)
    positions4[v] = glm::vec4(dag.positions[v * 3], dag.positions[v * 3 + 1],
                              dag.positions[v * 3 + 2], 0.0f);
  const std::vector<u8> hierarchyData(
      reinterpret_cast<const u8*>(dag.hierarchyNodes.data()),
      reinterpret_cast<const u8*>(dag.hierarchyNodes.data()) +
          dag.hierarchyNodes.size() * sizeof(geometry::DagHierarchyNode));
  std::vector<glm::uvec2> groupRanges;
  for (const auto& g : dag.groups)
    groupRanges.emplace_back(g.clusters.empty() ? 0 : g.clusters.front(),
                             static_cast<u32>(g.clusters.size()));

  std::vector<renderer::CullGpuInstance> instances;
  {
    renderer::CullGpuInstance identity;
    identity.bounds = glm::vec4(0.0f, 0.0f, 0.0f, 3.2f);
    instances.push_back(identity);
  }
  const std::vector<renderer::GpuMeshRangePair> meshRanges = {
      {0u, static_cast<u32>(dag.clusters.size())}};

  const glm::vec3 cam(2.5f, 3.0f, 2.0f);
  const u32 W = 1280, H = 720;
  glm::mat4 view = glm::lookAtRH(cam, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
  glm::mat4 proj =
      glm::perspectiveRH_ZO(glm::radians(60.0f), 16.0f / 9.0f, 0.1f, 100.0f);
  proj[1][1] *= -1.0f;
  glm::vec4 planes[6];
  renderer::extractFrustumPlanes(proj * view, planes);
  const f32 worldPerPixel =
      2.0f * std::tan(glm::radians(60.0f) * 0.5f) / static_cast<f32>(H);

  renderer::VisBufferChain chain;
  const std::vector<u32> meshHierRoots{dag.hierarchyRoot};
  REQUIRE(chain.init(*backend, shaders, positions4, dag.indices, clusterData,
                     groupData, hierarchyData, groupRanges, meshHierRoots,
                     instances, meshRanges,
                     static_cast<u32>(dag.clusters.size()), W, H));
  renderer::ChainInputs inputs;
  inputs.view = view;
  inputs.proj = proj;
  inputs.nearZ = 0.1f;
  inputs.camPos = cam;
  inputs.worldPerPixel = worldPerPixel;
  inputs.width = W;
  inputs.height = H;
  for (int p = 0; p < 6; ++p) inputs.planes[p] = planes[p];
  inputs.occlusionEnabled = false;
  inputs.hwMode = 0;
  inputs.cullMode = 1;  // the hierarchy traversal under test

  backend::FrameBeginInfo frameInfo;
  frameInfo.width = W;
  frameInfo.height = H;
  const float camArr[3] = {cam.x, cam.y, cam.z};
  kenshin::usize lastCount = ~kenshin::usize(0);
  for (const f32 threshold : {0.25f, 1.0f, 4.0f}) {
    inputs.pixelThreshold = threshold;
    REQUIRE(backend->beginFrame(frameInfo));
    chain.execute(*backend, inputs);
    backend->endFrame();

    std::vector<glm::uvec2> pairs;
    REQUIRE(chain.readVisiblePairs(*backend, pairs));
    std::multiset<u32> got;
    for (const auto& p : pairs) got.insert(p.y);
    std::vector<u32> cpuCut;
    geometry::selectLodCut(dag, threshold, H, 60.0f, camArr, cpuCut);
    REQUIRE(got.size() == cpuCut.size());
    std::set<u32> expected(cpuCut.begin(), cpuCut.end());
    for (const u32 c : got) REQUIRE(expected.count(c) == 1);
    INFO("threshold " << threshold << ": hierarchy cut " << got.size()
                      << " == selectLodCut " << cpuCut.size());
    // Coarser thresholds can only shrink the cut (monotonicity, as the
    // lodcut tests assert for the CPU side).
    REQUIRE(got.size() <= lastCount);
    lastCount = got.size();
  }

  chain.onShutdown(*backend);
  backend->shutdown();
}
