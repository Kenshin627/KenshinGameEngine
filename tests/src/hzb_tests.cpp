// M3.1 differential: the HZB pyramid (build_hzb.slang) vs a CPU
// reference reducing the same VisBuffer. Pure integer min-reduction -
// the comparison is EXACT, per mip per texel. The VisBuffer itself is
// produced by the proven one-shot rasterizer (S4.4 differential), so
// any divergence here is a pyramid bug.

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
#include "kenshin/renderer/HzbPass.h"
#include "kenshin/renderer/InstanceCullPass.h"
#include "kenshin/renderer/SoftRasterPass.h"
#include "kenshin/renderer/VirtualGeometryScene.h"

namespace {

using namespace kenshin;

const char* kShaderRoot = KENSHIN_SOURCE_DIR "/engine/shaders";

// CPU mirror of build_hzb.slang: identical clamped 2x2 min-reduce of
// the vis depth words (mode-0 extraction included).
std::vector<u32> hzbCpu(const std::vector<u64>& vis, u32 w, u32 h) {
  const auto layout = renderer::hzbLayout(w, h);
  std::vector<u32> pyr(layout.totalTexels, 0);
  std::vector<u32> src;
  for (u32 level = 0; level < layout.levelCount; ++level) {
    const bool extract = level == 0;
    if (extract) {
      src.resize(vis.size() * 2);
      std::memcpy(src.data(), vis.data(), vis.size() * 8);
    }
    const u32 srcW = extract ? w : layout.levelW[level - 1];
    const u32 srcH = extract ? h : layout.levelH[level - 1];
    const u32 srcOffset = extract ? 0 : layout.levelOffset[level - 1];
    for (u32 t = 0; t < layout.levelW[level] * layout.levelH[level]; ++t) {
      const u32 dx = t % layout.levelW[level];
      const u32 dy = t / layout.levelW[level];
      u32 m = 0xFFFFFFFFu;
      for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i) {
          const u32 sx = std::min(dx * 2 + u32(i), srcW - 1);
          const u32 sy = std::min(dy * 2 + u32(j), srcH - 1);
          const u32 word = extract ? src[2 * (sy * srcW + sx) + 1]
                                   : pyr[srcOffset + sy * srcW + sx];
          m = std::min(m, word);
        }
      pyr[layout.levelOffset[level] + dy * layout.levelW[level] + dx] = m;
    }
    // (the CPU chain works in-place on the accumulating pyramid)
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
  u32 cullStats[3] = {};
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
                 << ", mismatches " << mismatches << " |" << perLevel
                 << " | first L1 diff gpu=" << pyrGpu[layout.levelOffset[1]]
                 << " cpu=" << pyrCpu[layout.levelOffset[1]]);
  REQUIRE(mismatches == 0);

  backend->shutdown();
}
