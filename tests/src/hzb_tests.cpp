// M3.1 differential: the HZB pyramid (build_hzb.slang) vs a CPU
// reference reducing the same VisBuffer. Pure integer min-reduction -
// the comparison is EXACT, per mip per texel. The VisBuffer itself is
// produced by the proven one-shot rasterizer (S4.4 differential), so
// any divergence here is a pyramid bug.

#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <set>
#include <vector>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/geometry/ClusterDag.h"
#include "kenshin/geometry/ClusterLodBuilder.h"
#include "kenshin/geometry/shapes/SphereGeometry.h"
#include "kenshin/geometry/shapes/TorusKnotGeometry.h"
#include "kenshin/manager/ShaderManager.h"
#include "kenshin/renderer/ClusterCullPass.h"
#include "kenshin/renderer/HzbPass.h"
#include "kenshin/renderer/InstanceCullPass.h"
#include "kenshin/renderer/SoftRasterPass.h"
#include "kenshin/renderer/VirtualGeometryScene.h"
#include "kenshin/renderer/VisBufferChain.h"

namespace {

using namespace kenshin;

const char* kShaderRoot = KENSHIN_SOURCE_DIR "/engine/shaders";

// CPU mirror of build_hzb.slang (atomic climb): L0 = the RAW per-pixel
// depth words (one texel per pixel, no reduction), levels 1+ are
// clamped 2x2 min-reductions of the previous level. (The earlier
// per-level kernel's "L0" was a min-of-2x2 - a different, coarser
// convention both sides then shared.)
std::vector<u32> hzbCpu(const std::vector<u64>& vis, u32 w, u32 h) {
  const auto layout = renderer::hzbLayout(w, h);
  std::vector<u32> pyr(layout.totalTexels, 0);
  // L0: raw depth words.
  for (u32 i = 0; i < vis.size(); ++i) pyr[i] = u32(vis[i] >> 32);
  // Levels 1+: clamped 2x2 min of the previous level.
  for (u32 level = 1; level < layout.levelCount; ++level) {
    const u32 srcW = layout.levelW[level - 1];
    const u32 srcH = layout.levelH[level - 1];
    const u32 srcOffset = layout.levelOffset[level - 1];
    for (u32 t = 0; t < layout.levelW[level] * layout.levelH[level]; ++t) {
      const u32 dx = t % layout.levelW[level];
      const u32 dy = t / layout.levelW[level];
      u32 m = 0xFFFFFFFFu;
      for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i) {
          const u32 sx = std::min(dx * 2 + u32(i), srcW - 1);
          const u32 sy = std::min(dy * 2 + u32(j), srcH - 1);
          m = std::min(m, pyr[srcOffset + sy * srcW + sx]);
        }
      pyr[layout.levelOffset[level] + dy * layout.levelW[level] + dx] = m;
    }
  }
  return pyr;
}

}  // namespace

TEST_CASE("HZB pyramid GPU/CPU differential", "[gpu][hzb]") {
  using namespace kenshin;
  // Build a dag and rasterize ONE cluster set through the proven
  // one-shot path (two offset instances for overlap variety).
  geometry::ClusterDag dag;
  {
    const auto mesh = geometry::TorusKnotGeometry::create(2, 3, 128, 32, 1.0f);
    std::vector<geometry::ClusterGroup> groups;
    geometry::BuildStats stats;
    geometry::DagBuildStats dagStats;
    std::string log;
    const bool built = geometry::buildClusterLod(
        mesh, [&](const geometry::ClusterGroup& g) { groups.push_back(g); },
        stats, log);
    REQUIRE(built);
    REQUIRE(geometry::assembleClusterDag(mesh, groups, dag, dagStats, log));
  }

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
  instances.emplace_back();  // identity
  {
    renderer::CullGpuInstance moved;
    moved.model = glm::translate(glm::mat4(1.0f), {1.2f, 0.0f, 0.4f});
    instances.push_back(moved);
  }

  const glm::vec3 cam(2.5f, 3.0f, 2.0f);
  const u32 W = 96, H = 72;  // odd-ish sizes exercise edge clamping
  glm::mat4 view = glm::lookAtRH(cam, {0, 0, 0}, {0, 1, 0});
  glm::mat4 proj =
      glm::perspectiveRH_ZO(glm::radians(60.0f), 4.0f / 3.0f, 0.1f, 100.0f);
  proj[1][1] *= -1.0f;
  glm::vec4 planes[6];
  renderer::extractFrustumPlanes(proj * view, planes);

  std::vector<u32> visible;
  REQUIRE(renderer::cullInstancesOnGpu(*backend, shaders, instances, planes,
                                       visible));
  std::vector<glm::uvec2> workItems;
  const std::vector<renderer::GpuMeshRangePair> meshRanges = {
      {0u, static_cast<u32>(dag.clusters.size())}};
  REQUIRE(renderer::expandWorkItemsOnGpu(*backend, shaders, visible, instances,
                                         meshRanges, workItems));
  std::vector<glm::uvec2> pairs;
  u32 cullStats[4] = {};
  REQUIRE(renderer::cullClustersOnGpu(
      *backend, shaders, cam, 1.0f,
      2.0f * std::tan(glm::radians(60.0f) * 0.5f) / static_cast<f32>(H),
      clusterData, groupData, workItems, instances, pairs, cullStats));

  std::vector<u64> vis;
  u32 rasterStats[4] = {};
  const renderer::SoftRasterFrame frame{view,          proj, 0.1f, W, H,
                                        /*yFlip=*/true};
  REQUIRE(renderer::rasterizeOnGpu(*backend, shaders, frame, positions4,
                                   dag.indices, clusterData, instances, pairs,
                                   vis, rasterStats));
  REQUIRE(rasterStats[1] > 0);

  // The differential: GPU pyramid vs the CPU mirror.
  std::vector<u32> pyrGpu;
  REQUIRE(renderer::buildHzbOneshot(*backend, shaders, vis, W, H, pyrGpu));
  const std::vector<u32> pyrCpu = hzbCpu(vis, W, H);
  REQUIRE(pyrGpu.size() == pyrCpu.size());

  const auto layout = renderer::hzbLayout(W, H);
  usize mismatches = 0;
  std::string perLevel;
  for (u32 L = 0; L < layout.levelCount; ++L) {
    usize lm = 0;
    for (u32 t = 0; t < layout.levelW[L] * layout.levelH[L]; ++t)
      if (pyrGpu[layout.levelOffset[L] + t] !=
          pyrCpu[layout.levelOffset[L] + t])
        ++lm;
    mismatches += lm;
    perLevel += " L" + std::to_string(L) + "=" + std::to_string(lm);
  }
  INFO("levels " << layout.levelCount << ", texels " << pyrGpu.size()
                 << ", mismatches " << mismatches << " |" << perLevel);
  REQUIRE(mismatches == 0);

  backend->shutdown();
}

TEST_CASE("M3.2 occlusion cull is conservative", "[gpu][hzb][occ]") {
  using namespace kenshin;
  geometry::ClusterDag dag;
  {
    const auto mesh = geometry::SphereGeometry::create(96, 48, 1.0f);
    std::vector<geometry::ClusterGroup> groups;
    geometry::BuildStats stats;
    geometry::DagBuildStats dagStats;
    std::string log;
    const bool built = geometry::buildClusterLod(
        mesh, [&](const geometry::ClusterGroup& g) { groups.push_back(g); },
        stats, log);
    REQUIRE(built);
    REQUIRE(geometry::assembleClusterDag(mesh, groups, dag, dagStats, log));
  }
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

  const glm::vec3 cam(4.0f, 5.0f, 3.0f);
  const glm::vec3 away = glm::normalize(cam);
  std::vector<renderer::CullGpuInstance> instances;
  instances.emplace_back();
  {
    renderer::CullGpuInstance back;
    back.model = glm::translate(glm::mat4(1.0f), away * 3.0f);
    instances.push_back(back);
  }

  const u32 W = 96, H = 72;
  const f32 fov = 60.0f;
  const f32 worldPerPixel =
      2.0f * std::tan(glm::radians(fov) * 0.5f) / static_cast<f32>(H);
  glm::mat4 view = glm::lookAtRH(cam, {0, 0, 0}, {0, 1, 0});
  glm::mat4 proj =
      glm::perspectiveRH_ZO(glm::radians(fov), 4.0f / 3.0f, 0.1f, 100.0f);
  proj[1][1] *= -1.0f;
  const glm::mat4 viewProj = proj * view;
  glm::vec4 planes[6];
  renderer::extractFrustumPlanes(viewProj, planes);
  const renderer::SoftRasterFrame frame{view,          proj, 0.1f, W, H,
                                        /*yFlip=*/true};

  std::vector<u32> visible;
  REQUIRE(renderer::cullInstancesOnGpu(*backend, shaders, instances, planes,
                                       visible));
  std::vector<glm::uvec2> workItems;
  const std::vector<renderer::GpuMeshRangePair> meshRanges = {
      {0u, static_cast<u32>(dag.clusters.size())}};
  REQUIRE(renderer::expandWorkItemsOnGpu(*backend, shaders, visible, instances,
                                         meshRanges, workItems));

  std::vector<glm::uvec2> pairsFull;
  u32 statsFull[4] = {};
  REQUIRE(renderer::cullClustersOnGpu(
      *backend, shaders, cam, 0.05f, worldPerPixel, clusterData, groupData,
      workItems, instances, pairsFull, statsFull));
  std::vector<u64> visFull;
  u32 rasterFull[4] = {};
  REQUIRE(renderer::rasterizeOnGpu(*backend, shaders, frame, positions4,
                                   dag.indices, clusterData, instances,
                                   pairsFull, visFull, rasterFull));

  std::vector<u32> hzb;
  REQUIRE(renderer::buildHzbOneshot(*backend, shaders, visFull, W, H, hzb));

  renderer::OcclusionInputs occ;
  occ.enabled = true;
  occ.prevViewProj = viewProj;
  occ.prevCamPos = cam;
  occ.hzbTexels = hzb.data();
  occ.hzbCount = static_cast<u32>(hzb.size());
  occ.screenW = W;
  occ.screenH = H;
  occ.tanHalfFov = std::tan(glm::radians(fov) * 0.5f);
  const std::vector<glm::vec4> aabbs =
      renderer::computeClusterAabbs(positions4, clusterData);
  occ.clusterAabbs = aabbs.data();
  std::vector<glm::uvec2> pairsOcc;
  u32 statsOcc[4] = {};
  REQUIRE(renderer::cullClustersOnGpu(
      *backend, shaders, cam, 0.05f, worldPerPixel, clusterData, groupData,
      workItems, instances, pairsOcc, statsOcc, occ));

  INFO("full " << pairsFull.size() << " occ " << pairsOcc.size()
               << " occlusionCulled " << statsOcc[3]);
  REQUIRE(statsOcc[3] > 0);
  REQUIRE(pairsOcc.size() + statsOcc[3] == pairsFull.size());

  {
    std::multiset<u64> full;
    for (const auto& p : pairsFull) full.insert((u64(p.x) << 32) | p.y);
    for (const auto& p : pairsOcc) {
      const u64 k = (u64(p.x) << 32) | p.y;
      const auto it = full.find(k);
      REQUIRE(it != full.end());
      full.erase(it);
    }
  }

  std::vector<u64> visOcc;
  u32 rasterOcc[4] = {};
  REQUIRE(renderer::rasterizeOnGpu(*backend, shaders, frame, positions4,
                                   dag.indices, clusterData, instances,
                                   pairsOcc, visOcc, rasterOcc));
  usize mismatches = 0;
  for (usize i = 0; i < visFull.size(); ++i)
    if (visFull[i] != visOcc[i]) ++mismatches;
  INFO("vis mismatches " << mismatches);
  REQUIRE(mismatches == 0);

  backend->shutdown();
}

TEST_CASE("M3.2b in-frame chain occlusion - conservative, exact",
          "[gpu][vgchain][occ]") {
  using namespace kenshin;
  // The same sphere pair as the one-shot occlusion differential, now
  // through the IN-FRAME chain: frame 3 (slot 0 again) consumes the
  // HZB built from frame 1's VisBuffer. Occlusion must cull the back
  // sphere's covered clusters AND leave the VisBuffer bit-identical.
  geometry::ClusterDag dag;
  {
    const auto mesh = geometry::SphereGeometry::create(96, 48, 1.0f);
    std::vector<geometry::ClusterGroup> groups;
    geometry::BuildStats stats;
    geometry::DagBuildStats dagStats;
    std::string log;
    const bool built = geometry::buildClusterLod(
        mesh, [&](const geometry::ClusterGroup& g) { groups.push_back(g); },
        stats, log);
    REQUIRE(built);
    REQUIRE(geometry::assembleClusterDag(mesh, groups, dag, dagStats, log));
  }
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

  const glm::vec3 cam(4.0f, 5.0f, 3.0f);
  const glm::vec3 away = glm::normalize(cam);
  std::vector<renderer::CullGpuInstance> instances;
  instances.emplace_back();
  {
    renderer::CullGpuInstance back;
    back.model = glm::translate(glm::mat4(1.0f), away * 3.0f);
    instances.push_back(back);
  }
  const std::vector<renderer::GpuMeshRangePair> meshRanges = {
      {0u, static_cast<u32>(dag.clusters.size())}};

  const u32 W = 96, H = 72;
  const f32 fov = 60.0f;
  glm::mat4 view = glm::lookAtRH(cam, {0, 0, 0}, {0, 1, 0});
  glm::mat4 proj =
      glm::perspectiveRH_ZO(glm::radians(fov), 4.0f / 3.0f, 0.1f, 100.0f);
  proj[1][1] *= -1.0f;
  glm::vec4 planes[6];
  renderer::extractFrustumPlanes(proj * view, planes);

  renderer::VisBufferChain chain;
  REQUIRE(chain.init(*backend, shaders, positions4, dag.indices, clusterData,
                     groupData, instances, meshRanges,
                     static_cast<u32>(2 * dag.clusters.size()), W, H));
  renderer::ChainInputs inputs;
  inputs.view = view;
  inputs.proj = proj;
  inputs.nearZ = 0.1f;
  inputs.camPos = cam;
  inputs.pixelThreshold = 0.05f;
  inputs.worldPerPixel =
      2.0f * std::tan(glm::radians(fov) * 0.5f) / static_cast<f32>(H);
  inputs.width = W;
  inputs.height = H;
  for (int p = 0; p < 6; ++p) inputs.planes[p] = planes[p];

  auto frameInfo = backend::FrameBeginInfo{.width = W, .height = H};
  auto runFrame = [&](std::vector<u64>& visOut, u32 statsOut[12]) {
    REQUIRE(backend->beginFrame(frameInfo));
    chain.execute(*backend, inputs);
    backend->endFrame();
    std::vector<u8> vb;
    REQUIRE(backend->readbackBuffer(chain.visBuffer(*backend), 0,
                                    usize(W) * H * sizeof(u64), vb));
    visOut.assign(reinterpret_cast<const u64*>(vb.data()),
                  reinterpret_cast<const u64*>(vb.data()) + W * H);
    REQUIRE(chain.readStats(*backend, statsOut));
  };

  // M3.3 semantics: each frame ends by building the slot's FINAL HZB
  // (even warmup frames - the bootstrap), so the Main cull is active
  // from frame 2 on, testing against the PREVIOUS frame's complete
  // visibility - one frame stale, not two.
  std::vector<u64> vis1, vis2, vis3;
  u32 st1[12], st2[12], st3[12];
  runFrame(vis1, st1);  // slot 0, occ off (no HZB yet) - the reference
  runFrame(vis2, st2);  // slot 1, occ ON vs frame 1's final HZB
  runFrame(vis3, st3);  // slot 0, occ ON vs frame 2's final HZB

  INFO("stats: occCulled " << st1[8] << "/" << st2[8] << "/" << st3[8]
                           << " revived " << st1[9] << "/" << st2[9] << "/"
                           << st3[9] << ", pairs " << st1[2] << "/" << st2[2]
                           << "/" << st3[2]);
  REQUIRE(st1[8] == 0);  // warmup frame culled nothing
  REQUIRE(st2[8] > 0);   // occlusion active from frame 2
  REQUIRE(st2[9] == 0);  // static camera: nothing revived
  REQUIRE(st3[9] == 0);
  REQUIRE(st2[2] + st2[8] == st1[2]);  // only occlusion removed pairs
  REQUIRE(st3[2] + st3[8] == st1[2]);

  usize mismatch2 = 0, mismatch3 = 0;
  for (usize i = 0; i < vis1.size(); ++i) {
    if (vis1[i] != vis2[i]) ++mismatch2;
    if (vis1[i] != vis3[i]) ++mismatch3;
  }
  INFO("vis mismatches: frame2 " << mismatch2 << ", frame3 " << mismatch3);
  REQUIRE(mismatch2 == 0);  // conservative: identical VisBuffer
  REQUIRE(mismatch3 == 0);

  chain.onShutdown(*backend);
  backend->shutdown();
}

TEST_CASE("M3.3 post pass - disocclusion revive, exact",
          "[gpu][vgchain][occ]") {
  using namespace kenshin;
  // The M3.3 acceptance: with the camera MOVING, the prev-frame HZB
  // is stale and the Main cull wrongly kills newly-revealed clusters.
  // The post pass must revive them within the SAME frame. Ground
  // truth: a second chain with occlusion disabled produces the
  // occlusion-free VisBuffer - the occluded chain must match it
  // BIT-EXACTLY every frame of an orbiting-camera sequence.
  geometry::ClusterDag dag;
  {
    const auto mesh = geometry::SphereGeometry::create(96, 48, 1.0f);
    std::vector<geometry::ClusterGroup> groups;
    geometry::BuildStats stats;
    geometry::DagBuildStats dagStats;
    std::string log;
    const bool built = geometry::buildClusterLod(
        mesh, [&](const geometry::ClusterGroup& g) { groups.push_back(g); },
        stats, log);
    REQUIRE(built);
    REQUIRE(geometry::assembleClusterDag(mesh, groups, dag, dagStats, log));
  }
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

  // Same occluder/occluded pair as the M3.2 differential, but the
  // camera ORBITS: each frame's hidden set of back-sphere clusters
  // shifts, so the Main cull runs on stale occlusion every frame.
  const glm::vec3 camBase(4.0f, 5.0f, 3.0f);
  const glm::vec3 away = glm::normalize(camBase);
  std::vector<renderer::CullGpuInstance> instances;
  instances.emplace_back();
  {
    renderer::CullGpuInstance back;
    back.model = glm::translate(glm::mat4(1.0f), away * 3.0f);
    instances.push_back(back);
  }
  const std::vector<renderer::GpuMeshRangePair> meshRanges = {
      {0u, static_cast<u32>(dag.clusters.size())}};

  const u32 W = 96, H = 72;
  const f32 fov = 60.0f;
  glm::mat4 proj =
      glm::perspectiveRH_ZO(glm::radians(fov), 4.0f / 3.0f, 0.1f, 100.0f);
  proj[1][1] *= -1.0f;

  renderer::VisBufferChain chainOff, chainOn;
  REQUIRE(chainOff.init(*backend, shaders, positions4, dag.indices, clusterData,
                        groupData, instances, meshRanges,
                        static_cast<u32>(2 * dag.clusters.size()), W, H));
  REQUIRE(chainOn.init(*backend, shaders, positions4, dag.indices, clusterData,
                       groupData, instances, meshRanges,
                       static_cast<u32>(2 * dag.clusters.size()), W, H));

  auto frameInfo = backend::FrameBeginInfo{.width = W, .height = H};
  const int kFrames = 7;
  u32 totalRevived = 0, totalPending = 0;
  for (int f = 0; f < kFrames; ++f) {
    // Orbit the camera around the Y axis through the origin; the
    // step is large enough that occlusion relations change per frame.
    const f32 yaw = 0.45f * static_cast<f32>(f);
    const glm::vec3 cam =
        glm::rotate(glm::mat4(1.0f), yaw, {0, 1, 0}) * glm::vec4(camBase, 1.0f);
    glm::mat4 view = glm::lookAtRH(cam, {0, 0, 0}, {0, 1, 0});
    glm::vec4 planes[6];
    renderer::extractFrustumPlanes(proj * view, planes);

    renderer::ChainInputs ci;
    ci.view = view;
    ci.proj = proj;
    ci.nearZ = 0.1f;
    ci.camPos = cam;
    ci.pixelThreshold = 0.05f;
    ci.worldPerPixel =
        2.0f * std::tan(glm::radians(fov) * 0.5f) / static_cast<f32>(H);
    ci.width = W;
    ci.height = H;
    for (int p = 0; p < 6; ++p) ci.planes[p] = planes[p];
    ci.occlusionEnabled = false;
    renderer::ChainInputs ciOn = ci;
    ciOn.occlusionEnabled = true;

    REQUIRE(backend->beginFrame(frameInfo));
    chainOff.execute(*backend, ci);
    chainOn.execute(*backend, ciOn);
    backend->endFrame();

    std::vector<u8> vbOff, vbOn;
    REQUIRE(backend->readbackBuffer(chainOff.visBuffer(*backend), 0,
                                    usize(W) * H * sizeof(u64), vbOff));
    REQUIRE(backend->readbackBuffer(chainOn.visBuffer(*backend), 0,
                                    usize(W) * H * sizeof(u64), vbOn));
    usize mismatches = 0;
    for (usize i = 0; i < usize(W) * H; ++i)
      if (vbOff[i] != vbOn[i]) ++mismatches;
    u32 st[12];
    REQUIRE(chainOn.readStats(*backend, st));
    totalRevived += st[9];
    totalPending += st[8];
    INFO("frame " << f << " vis mismatches " << mismatches << " pending "
                  << st[8] << " revived " << st[9]);
    REQUIRE(mismatches == 0);  // the post pass closed every hole
  }
  INFO("sequence totals: pending " << totalPending << " revived "
                                   << totalRevived);
  REQUIRE(totalPending > 0);  // the Main cull actually ran occluded
  REQUIRE(totalRevived > 0);  // the post pass actually revived

  chainOff.onShutdown(*backend);
  chainOn.onShutdown(*backend);
  backend->shutdown();
}

TEST_CASE("M3.4 dense occluder stack - count drop, image exact",
          "[gpu][vgchain][occ]") {
  using namespace kenshin;
  // The M3.4 acceptance scene: three instances stacked along the view
  // axis (the app's --multi --dense layout). A static camera looks
  // dead-on, so the near mesh occludes most of the mid/far ones.
  // Occlusion ON must measurably shrink the cut (clusters AND tris)
  // while the VisBuffer stays bit-identical to occlusion OFF.
  geometry::ClusterDag dag;
  {
    const auto mesh = geometry::SphereGeometry::create(96, 48, 1.0f);
    std::vector<geometry::ClusterGroup> groups;
    geometry::BuildStats stats;
    geometry::DagBuildStats dagStats;
    std::string log;
    const bool built = geometry::buildClusterLod(
        mesh, [&](const geometry::ClusterGroup& g) { groups.push_back(g); },
        stats, log);
    REQUIRE(built);
    REQUIRE(geometry::assembleClusterDag(mesh, groups, dag, dagStats, log));
  }
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

  // The app's --dense offsets, along +x; the camera sits on -x.
  std::vector<renderer::CullGpuInstance> instances;
  for (const f32 x : {0.0f, 2.2f, 4.4f})
    instances.push_back(
        {glm::translate(glm::mat4(1.0f), glm::vec3(x, 0.0f, 0.0f))});
  const std::vector<renderer::GpuMeshRangePair> meshRanges = {
      {0u, static_cast<u32>(dag.clusters.size())}};

  const u32 W = 128, H = 96;
  const f32 fov = 60.0f;
  const glm::vec3 cam(-5.5f, 0.0f, 0.0f);
  glm::mat4 view = glm::lookAtRH(cam, {2.2f, 0.0f, 0.0f}, {0, 1, 0});
  glm::mat4 proj =
      glm::perspectiveRH_ZO(glm::radians(fov), 4.0f / 3.0f, 0.1f, 100.0f);
  proj[1][1] *= -1.0f;
  glm::vec4 planes[6];
  renderer::extractFrustumPlanes(proj * view, planes);

  renderer::VisBufferChain chainOff, chainOn;
  REQUIRE(chainOff.init(*backend, shaders, positions4, dag.indices, clusterData,
                        groupData, instances, meshRanges,
                        static_cast<u32>(3 * dag.clusters.size()), W, H));
  REQUIRE(chainOn.init(*backend, shaders, positions4, dag.indices, clusterData,
                       groupData, instances, meshRanges,
                       static_cast<u32>(3 * dag.clusters.size()), W, H));

  renderer::ChainInputs ci;
  ci.view = view;
  ci.proj = proj;
  ci.nearZ = 0.1f;
  ci.camPos = cam;
  ci.pixelThreshold = 0.05f;
  ci.worldPerPixel =
      2.0f * std::tan(glm::radians(fov) * 0.5f) / static_cast<f32>(H);
  ci.width = W;
  ci.height = H;
  for (int p = 0; p < 6; ++p) ci.planes[p] = planes[p];
  ci.occlusionEnabled = false;
  renderer::ChainInputs ciOn = ci;
  ciOn.occlusionEnabled = true;

  auto frameInfo = backend::FrameBeginInfo{.width = W, .height = H};
  std::vector<u8> vbOff, vbOn;
  u32 stOff[12] = {}, stOn[12] = {};
  for (int f = 0; f < 5; ++f) {  // steady state well past warmup
    REQUIRE(backend->beginFrame(frameInfo));
    chainOff.execute(*backend, ci);
    chainOn.execute(*backend, ciOn);
    backend->endFrame();
    REQUIRE(backend->readbackBuffer(chainOff.visBuffer(*backend), 0,
                                    usize(W) * H * sizeof(u64), vbOff));
    REQUIRE(backend->readbackBuffer(chainOn.visBuffer(*backend), 0,
                                    usize(W) * H * sizeof(u64), vbOn));
    REQUIRE(chainOff.readStats(*backend, stOff));
    REQUIRE(chainOn.readStats(*backend, stOn));
  }

  usize mismatches = 0;
  for (usize i = 0; i < vbOff.size(); ++i)
    if (vbOff[i] != vbOn[i]) ++mismatches;
  INFO("dense stack: off pairs "
       << stOff[2] << " tris " << stOff[11] << " px " << stOff[5]
       << " | on pairs " << stOn[2] << " tris " << stOn[11] << " px "
       << stOn[5] + stOn[10] << " occCulled " << stOn[8] << " revived "
       << stOn[9] << " | vis mismatches " << mismatches);
  REQUIRE(mismatches == 0);  // occlusion never changes the image
  REQUIRE(stOn[8] > 0);      // the stack actually occludes
  REQUIRE(stOn[9] == 0);     // static camera: nothing stays revived
  REQUIRE(stOn[2] + stOn[8] == stOff[2]);  // conservation vs occlusion off
  // The acceptance margin: the dense stack must cut a real fraction -
  // at least a quarter of the unoccluded clusters stay culled.
  REQUIRE(stOn[2] <= stOff[2] - stOff[2] / 4);
  REQUIRE(stOn[11] < stOff[11]);

  chainOff.onShutdown(*backend);
  chainOn.onShutdown(*backend);
  backend->shutdown();
}
