// S4.2 differential: CPU reference frustum cull vs the GPU compute
// shader, instance by instance. Headless (windowless Vulkan device +
// runtime Slang compile). The staged plan's boundary cases: crossing
// the near plane, crossing the left plane, fully outside, fully
// inside, and an exact on-plane touch.

#include <catch2/catch_test_macros.hpp>

#include <cstdio>

#include <algorithm>
#include <set>
#include <vector>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/manager/ShaderManager.h"
#include "kenshin/renderer/InstanceCullPass.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace {

const char* kShaderRoot = KENSHIN_SOURCE_DIR "/engine/shaders";

/// CPU reference - the EXACT formula the shader runs: visible iff
/// dot(n, c) + d >= -r for every plane (touching counts as visible).
std::vector<kenshin::u32> cullCpu(
    const std::vector<kenshin::renderer::CullGpuInstance>& instances,
    const glm::vec4 planes[6]) {
  std::vector<kenshin::u32> visible;
  for (kenshin::u32 i = 0; i < instances.size(); ++i) {
    const glm::vec3 c(instances[i].bounds);
    const float r = instances[i].bounds.w;
    bool ok = true;
    for (int p = 0; p < 6 && ok; ++p)
      if (glm::dot(glm::vec3(planes[p]), c) + planes[p].w < -r) ok = false;
    if (ok) visible.push_back(i);
  }
  return visible;
}

}  // namespace

TEST_CASE("instance cull GPU/CPU differential", "[gpu][cull]") {
  kenshin::manager::ShaderManager shaders;
  REQUIRE(shaders.init(kShaderRoot));

  auto backend = kenshin::backend::createVulkanBackend();
  kenshin::backend::BackendInitParams params;  // windowless
  REQUIRE(backend->initialize(params));

  // Camera: looking down -Z from (0,0,8), Z-up world -> up +Z. NOTE the
  // engine camera is Z-up with a Y-flipped Vulkan projection; for THIS
  // test the view matrix uses up +Z so world +Z maps to screen up.
  const glm::mat4 view =
      glm::lookAtRH(glm::vec3(0.0f, 0.0f, 8.0f), glm::vec3(0.0f),
                    glm::vec3(0.0f, 1.0f, 0.0f));
  glm::mat4 proj =
      glm::perspectiveRH_ZO(glm::radians(60.0f), 16.0f / 9.0f, 0.1f, 100.0f);
  // glm's perspective gives OpenGL Y-up NDC; flip Y for Vulkan. Z is
  // already ZO. The flip does not change the frustum's shape, only its
  // handedness in NDC - the extracted planes follow the matrix either way.
  proj[1][1] *= -1.0f;
  const glm::mat4 viewProj = proj * view;

  glm::vec4 planes[6];
  kenshin::renderer::extractFrustumPlanes(viewProj, planes);

  // Boundary-case instance set (centers on the z=0 plane, the camera's
  // forward axis; radius 0.5 each unless noted). With 60deg fov at
  // distance 8 the left/right frustum edges sit around x = +-4.6 near
  // the z=0 plane.
  std::vector<kenshin::renderer::CullGpuInstance> instances;
  const auto add = [&](glm::vec3 c, float r = 0.5f) {
    kenshin::renderer::CullGpuInstance in;
    in.bounds = glm::vec4(c, r);
    instances.push_back(in);
  };
  add({0.0f, 0.0f, 0.0f});            // 0: dead center, fully inside
  add({4.0f, 0.0f, 0.0f});            // 1: near the right edge, inside
  add({40.0f, 0.0f, 0.0f});           // 2: fully outside (right)
  add({-40.0f, 0.0f, 0.0f});          // 3: fully outside (left)
  add({4.8f, 0.0f, 0.0f}, 1.0f);      // 4: center outside, sphere crosses
                                      //    the right plane
  add({0.0f, 0.0f, 8.3f}, 0.5f);      // 5: center between camera and
                                      //    the near plane (near plane at
                                      //    world z=7.9), sphere crosses it
  add({0.0f, 0.0f, 8.5f}, 0.2f);      // 6: fully in front of the near
                                      //    plane (outside)
  add({0.0f, 0.0f, -95.0f});          // 7: fully outside (103 units
                                      //    from the camera; the far
                                      //    plane sits at 100 => world
                                      //    z = -92)
  add({0.0f, 5.2f, 0.0f}, 1.0f);      // 8: center above the top edge
                                      //    (top edge ~y=4.62 at z=0:
                                      //    tan(30deg)*8), sphere crosses
  add({0.0f, -30.0f, 0.0f}, 29.0f);   // 9: bottom, huge sphere reaching in

  const std::vector<kenshin::u32> cpu = cullCpu(instances, planes);
  REQUIRE(cpu.size() >= 2);  // sanity: center + one more at least

  std::vector<kenshin::u32> gpu;
  REQUIRE(kenshin::renderer::cullInstancesOnGpu(
      *backend, shaders, instances, planes, gpu));

  // The GPU list order is nondeterministic - compare as sets.
  const std::multiset<kenshin::u32> cpuSet(cpu.begin(), cpu.end());
  const std::multiset<kenshin::u32> gpuSet(gpu.begin(), gpu.end());
  INFO("cpu visible " << cpuSet.size() << ", gpu visible " << gpuSet.size());
  REQUIRE(cpuSet == gpuSet);

  // Pin the semantics: dead-center MUST be visible on both.
  REQUIRE(cpuSet.count(0) == 1);
  REQUIRE(gpuSet.count(0) == 1);
  // Fully-outside cases must be culled on both.
  for (const kenshin::u32 i : {2u, 3u, 6u, 7u}) {
    INFO("outside case " << i);
    REQUIRE(cpuSet.count(i) == 0);
    REQUIRE(gpuSet.count(i) == 0);
  }
  // The crossing cases must survive on both (sphere reaches the
  // positive side -> visible).
  for (const kenshin::u32 i : {4u, 5u, 8u, 9u}) {
    REQUIRE(cpuSet.count(i) == 1);
    REQUIRE(gpuSet.count(i) == 1);
  }

  // Exact on-plane touch: sphere tangent to the near plane from the
  // inside is VISIBLE (>= -r includes touching). Construct values that
  // are exact in f32: near plane z = 7.9 with the camera at 8 and
  // near=0.1 is not representable; instead place a unit sphere whose
  // surface exactly touches x = 0 in world space through the LEFT plane
  // by using a sphere centered at x = -0.0 with r chosen so the test
  // reduces to comparing exactly representable numbers (dot of an axis
  // unit vector). The robust exact-touch check: center ON the plane
  // (distance 0, any r) -> 0 >= -r holds exactly on both sides.
  instances.clear();
  {
    kenshin::renderer::CullGpuInstance touch;
    // Center exactly on the near plane's world position along the view
    // axis: use the CPU-computed plane to place it - center satisfies
    // dot(n,c)+d == 0 by construction along the camera axis.
    const glm::vec3 n(planes[4]);  // near plane normal
    const float d = planes[4].w;
    const glm::vec3 eye(0.0f, 0.0f, 8.0f), fwd(0.0f, 0.0f, -1.0f);
    // Ray eye + t*fwd hits the plane: t = -(n.eye + d) / (n.fwd).
    const float t = -(glm::dot(n, eye) + d) / glm::dot(n, fwd);
    touch.bounds = glm::vec4(eye + fwd * t, 1.0f);
    instances.push_back(touch);
  }
  const std::vector<kenshin::u32> cpuTouch = cullCpu(instances, planes);
  std::vector<kenshin::u32> gpuTouch;
  REQUIRE(kenshin::renderer::cullInstancesOnGpu(
      *backend, shaders, instances, planes, gpuTouch));
  REQUIRE(cpuTouch.size() == gpuTouch.size());
  REQUIRE(gpuTouch.size() == 1);  // tangent == visible, both sides

  backend->shutdown();
}
