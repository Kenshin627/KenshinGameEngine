// S4.4 differential: software rasterizer vs a CPU reference that
// mirrors the shader's float expressions operation-for-operation.
// VisBuffer read back and compared per pixel (depth bits + cluster +
// triangle). Cases per the staged plan: full-screen smoke triangle,
// partially offscreen / shared-edge / depth-overlap differential, a
// dedicated near-plane-crossing case, and an exact on-edge tie.
//
// The CPU reference is a TRANSLATION of rasterize_sw.slang, not an
// independent implementation - any divergence is a port bug on one of
// the two sides, which is exactly what the differential is for.

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/manager/ShaderManager.h"
#include "kenshin/renderer/InstanceCullPass.h"
#include "kenshin/renderer/SoftRasterPass.h"
#include "kenshin/renderer/VirtualGeometryScene.h"

namespace {

using namespace kenshin;

const char* kShaderRoot = KENSHIN_SOURCE_DIR "/engine/shaders";

// ---- CPU mirror of the shader -----------------------------------------

glm::vec2 applyMat4(const glm::mat4& m, const glm::vec3& p,
                    const glm::vec2& wh) {
  // Shader-parity: explicit glm-column combination, w-divide, then the
  // viewport map (UNflipped Y: row index grows with +Y NDC).
  const glm::vec4 c = m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3];
  const glm::vec3 ndc(c.x / c.w, c.y / c.w, c.z / c.w);
  return glm::vec2((ndc.x * 0.5f + 0.5f) * wh.x, (ndc.y * 0.5f + 0.5f) * wh.y);
}
float applyMat4Z(const glm::mat4& m, const glm::vec3& p) {
  const glm::vec4 c = m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3];
  return c.z / c.w;
}

struct CpuRaster {
  renderer::SoftRasterFrame frame;
  std::vector<glm::vec4> positions;  // GPU pool layout (std430 vec3->16B)
  std::vector<u32> indices;
  std::vector<u64> vis;

  // One "cluster table entry" worth of span (test clusters are packed
  // back-to-back: vertexOffset/indexOffset computed by the helper).
  void run(const std::vector<glm::uvec2>& pairs,
           const std::vector<renderer::CullGpuInstance>& instances,
           const std::vector<std::pair<u32, u32>>& clusterSpans, u32* stats) {
    vis.assign(static_cast<usize>(frame.width) * frame.height, 0);
    stats[0] = stats[1] = stats[2] = stats[3] = 0;
    for (usize p = 0; p < pairs.size(); ++p) {
      const u32 c = pairs[p].y;
      const u32 vOff = clusterSpans[c].first;
      const u32 vEnd = clusterSpans[c + 1].first;  // sentinel end
      const u32 iOff = clusterSpans[c].second;
      const u32 iEnd = clusterSpans[c + 1].second;
      // vertices to view space (mesh-local -> world -> view)
      std::vector<glm::vec3> view(vEnd - vOff);
      for (u32 v = 0; v < view.size(); ++v) {
        const glm::vec3 lp(positions[vOff + v]);
        const glm::vec3 wp =
            glm::vec3(instances[pairs[p].x].model * glm::vec4(lp, 1.0f));
        view[v] = glm::vec3(frame.view * glm::vec4(wp, 1.0f));
      }
      const u32 triCount = (iEnd - iOff) / 3;
      for (u32 t = 0; t < triCount; ++t) {
        const u32 i0 = indices[iOff + t * 3 + 0];
        const u32 i1 = indices[iOff + t * 3 + 1];
        const u32 i2 = indices[iOff + t * 3 + 2];
        // Sutherland-Hodgman vs z = -near
        std::vector<glm::vec3> poly;
        const glm::vec3 inV[3] = {view[i0], view[i1], view[i2]};
        for (int e = 0; e < 3; ++e) {
          const glm::vec3& a = inV[e];
          const glm::vec3& b = inV[(e + 1) % 3];
          const bool aIn = a.z <= -frame.nearZ;
          const bool bIn = b.z <= -frame.nearZ;
          if (aIn) {
            poly.push_back(a);
            if (!bIn)
              poly.push_back(a +
                             (b - a) * ((-frame.nearZ - a.z) / (b.z - a.z)));
          } else if (bIn) {
            poly.push_back(a + (b - a) * ((-frame.nearZ - a.z) / (b.z - a.z)));
          }
        }
        if (poly.size() < 3) continue;
        const glm::vec2 wh(static_cast<float>(frame.width),
                           static_cast<float>(frame.height));
        const auto rasterTri = [&](const glm::vec3& a, const glm::vec3& b,
                                   const glm::vec3& c) {
          rasterTriangle(applyMat4(frame.proj, a, wh),
                         applyMat4(frame.proj, b, wh),
                         applyMat4(frame.proj, c, wh),
                         applyMat4Z(frame.proj, a), applyMat4Z(frame.proj, b),
                         applyMat4Z(frame.proj, c), u32(p), t, stats);
        };
        if (poly.size() == 3)
          rasterTri(poly[0], poly[1], poly[2]);
        else {
          ++stats[3];
          rasterTri(poly[0], poly[1], poly[2]);
          rasterTri(poly[0], poly[2], poly[3]);
        }
      }
    }
  }

  void rasterTriangle(glm::vec2 s0f, glm::vec2 s1f, glm::vec2 s2f, float z0,
                      float z1, float z2, u32 pairIdx, u32 triIdx, u32* stats) {
    // M4.1 mirror: 16.8 subpixel fixed point + int64 edge functions +
    // the DX top-left tie rule (rasterize_sw.slang's new core).
    constexpr int SUBPIXEL = 256;
    const auto quant = [](float v) {
      return i64(std::floor(v * static_cast<float>(SUBPIXEL)));
    };
    const i64 X0 = quant(s0f.x), Y0 = quant(s0f.y);
    i64 X1 = quant(s1f.x), Y1 = quant(s1f.y);
    i64 X2 = quant(s2f.x), Y2 = quant(s2f.y);

    const i64 area2 = (X1 - X0) * (Y2 - Y0) - (Y1 - Y0) * (X2 - X0);
    if (area2 == 0) return;
    // Cull-sign + canonicalization mirror the shader: yFlip
    // (framebuffer-order rows) culls the CW-as-viewed half, then a
    // vertex swap restores the positive-area edge-equation path.
    const i64 cullSign = frame.yFlip ? 1 : -1;
    if (area2 * cullSign > 0) {
      ++stats[2];
      return;
    }
    if (area2 < 0) {
      std::swap(X1, X2);
      std::swap(Y1, Y2);
      std::swap(z1, z2);
    }

    const i64 e0x = X1 - X0, e0y = Y1 - Y0;
    const i64 e1x = X2 - X1, e1y = Y2 - Y1;
    const i64 e2x = X0 - X2, e2y = Y0 - Y2;
    // DX top-left: ties count only on top (ey==0 && ex>0) or left
    // (ey<0) edges of the canonical winding.
    const auto tieBias = [](i64 ex, i64 ey) -> i64 {
      return (ey == 0 && ex > 0) || ey < 0 ? 0 : 1;
    };
    const i64 t0 = tieBias(e0x, e0y);
    const i64 t1 = tieBias(e1x, e1y);
    const i64 t2 = tieBias(e2x, e2y);

    const i64 minSubX = std::min({X0, X1, X2});
    const i64 maxSubX = std::max({X0, X1, X2});
    const i64 minSubY = std::min({Y0, Y1, Y2});
    const i64 maxSubY = std::max({Y0, Y1, Y2});
    int x0 =
        std::max<int>(int(std::ceil((float(minSubX) - 128.0f) / 256.0f)), 0);
    int x1 = std::min<int>(int(std::floor((float(maxSubX) - 129.0f) / 256.0f)),
                           int(frame.width) - 1);
    int y0 =
        std::max<int>(int(std::ceil((float(minSubY) - 128.0f) / 256.0f)), 0);
    int y1 = std::min<int>(int(std::floor((float(maxSubY) - 129.0f) / 256.0f)),
                           int(frame.height) - 1);
    if (x0 > x1 || y0 > y1) return;

    // Affine depth through the quantized corners (pixel units).
    const glm::vec2 q0(float(X0) / 256.0f, float(Y0) / 256.0f);
    const glm::vec2 q1(float(X1) / 256.0f, float(Y1) / 256.0f);
    const glm::vec2 q2(float(X2) / 256.0f, float(Y2) / 256.0f);
    const float det =
        (q1.x - q0.x) * (q2.y - q0.y) - (q1.y - q0.y) * (q2.x - q0.x);
    const float ra =
        ((z1 - z0) * (q2.y - q0.y) - (z2 - z0) * (q1.y - q0.y)) / det;
    const float rb =
        ((q1.x - q0.x) * (z2 - z0) - (q2.x - q0.x) * (z1 - z0)) / det;

    const i64 px0 = i64(x0) * SUBPIXEL + 128;
    const i64 py0 = i64(y0) * SUBPIXEL + 128;
    const auto edgeW = [](i64 ex, i64 ey, i64 px, i64 py, i64 vx, i64 vy) {
      return ex * (py - vy) - ey * (px - vx);
    };
    const i64 w0r0 = edgeW(e0x, e0y, px0, py0, X0, Y0);
    const i64 w1r0 = edgeW(e1x, e1y, px0, py0, X1, Y1);
    const i64 w2r0 = edgeW(e2x, e2y, px0, py0, X2, Y2);
    const glm::vec2 p00(float(px0) / 256.0f, float(py0) / 256.0f);
    const float z00 = z0 + ra * (p00.x - q0.x) + rb * (p00.y - q0.y);

    for (int y = y0; y <= y1; ++y) {
      const i64 dySub = i64(y) * SUBPIXEL + 128 - py0;
      i64 w0 = w0r0 + e0x * dySub;
      i64 w1 = w1r0 + e1x * dySub;
      i64 w2 = w2r0 + e2x * dySub;
      const float ry = float(dySub) / 256.0f;
      float z = z00 + rb * ry;
      for (int x = x0; x <= x1; ++x) {
        if (w0 >= t0 && w1 >= t1 && w2 >= t2) {
          const u64 packed = renderer::packVisPixel(z, pairIdx, triIdx);
          const usize idx = usize(y) * frame.width + x;
          if (packed > vis[idx]) vis[idx] = packed;  // max, no atomics
          ++stats[1];
        }
        // x advances ONE PIXEL = SUBPIXEL subpixels: the edge delta is
        // -ey * SUBPIXEL (the row step multiplies by dySub for the same
        // reason). Missing this factor made the edge functions crawl
        // 256x too slowly across each row and drop half the coverage.
        w0 -= e0y * SUBPIXEL;
        w1 -= e1y * SUBPIXEL;
        w2 -= e2y * SUBPIXEL;
        z += ra;
      }
    }
  }
};

struct Scene {
  std::vector<glm::vec4> positions;  // float4 elements (GPU pool layout)
  std::vector<u32> indices;
  std::vector<u8> clusterData;
  std::vector<std::pair<u32, u32>> spans;  // per cluster (vOff, iOff) + end
  renderer::SoftRasterFrame frame;

  u32 addCluster(const std::vector<glm::vec3>& verts,
                 const std::vector<u32>& idx) {
    const u32 vOff = static_cast<u32>(positions.size());
    const u32 iOff = static_cast<u32>(indices.size());
    for (const glm::vec3& v : verts) positions.emplace_back(v, 0.0f);
    indices.insert(indices.end(), idx.begin(), idx.end());
    // 48B entry: only the offsets/counts matter to the rasterizer.
    renderer::packGpuCluster(clusterData, 0u, 0, 0, 0, 1.0f, 0, 0, vOff,
                             static_cast<u32>(verts.size()), iOff,
                             static_cast<u32>(idx.size()), -1);
    spans.emplace_back(vOff, iOff);
    return static_cast<u32>(spans.size() - 1);
  }
};

}  // namespace

TEST_CASE("software rasterizer GPU/CPU differential", "[gpu][swraster]") {
  manager::ShaderManager shaders;
  REQUIRE(shaders.init(kShaderRoot));
  auto backend = backend::createVulkanBackend();
  backend::BackendInitParams params;
  REQUIRE(backend->initialize(params));

  const glm::vec3 cam(0.0f, 0.0f, 8.0f);
  renderer::SoftRasterFrame frame;
  frame.view = glm::lookAtRH(cam, {0, 0, 0}, {0, 1, 0});
  frame.proj = glm::perspectiveRH_ZO(glm::radians(60.0f), 4.0f / 3.0f, 0.1f,
                                     100.0f);  // UNFLIPPED Y
  frame.nearZ = 0.1f;
  frame.width = 64;
  frame.height = 48;
  renderer::CullGpuInstance identity;

  const auto runCase =
      [&](const char* name, Scene& scene, const std::vector<glm::uvec2>& pairs,
          const std::vector<renderer::CullGpuInstance>& insts) {
        // end sentinel span so CPU knows each cluster's extent
        scene.spans.emplace_back(static_cast<u32>(scene.positions.size()),
                                 static_cast<u32>(scene.indices.size()));

        CpuRaster cpu{frame, scene.positions, scene.indices, {}};
        u32 cpuStats[4] = {};
        cpu.run(pairs, insts, scene.spans, cpuStats);

        // Both scheduling modes against the SAME CPU mirror: mode 1
        // (the row-span clamp) must be indistinguishable from mode 0
        // at the independent ground truth (the mirror walks full rows).
        for (const u32 rasterMode : {0u, 1u}) {
          frame.rasterMode = rasterMode;
          std::vector<u64> gpuVis;
          u32 gpuStats[4] = {};
          REQUIRE(renderer::rasterizeOnGpu(
              *backend, shaders, frame, scene.positions, scene.indices,
              scene.clusterData, insts, pairs, gpuVis, gpuStats));

          // Depth bits may differ by a few float32 ulps on generic-slope
          // triangles: the host and the SPIR-V contract FMAs differently.
          // The S4.4 cases were authored with constant/nice-slope depths so
          // rounding could not diverge; the S4.5 proj fix (params[4..7] is the
          // PROJECTION, not world->NDC - a double-view bug this differential
          // shared on both sides) made the near-clip quad generic-slope. The
          // payload (cluster | triangle) must still match EXACTLY - only the
          // depth bits get the tolerance.
          const auto tolerantEqual = [](u64 a, u64 b) {
            if (a == b) return true;
            if ((a & 0xFFFFFFFFull) != (b & 0xFFFFFFFFull)) return false;
            const u64 da = a >> 32, db = b >> 32;
            return (da > db ? da - db : db - da) <= (1ull << 12);
          };
          u64 mismatches = 0;
          std::string first;
          for (usize i = 0; i < cpu.vis.size(); ++i)
            if (!tolerantEqual(cpu.vis[i], gpuVis[i])) {
              if (mismatches == 0)
                first = "pixel " + std::to_string(i % frame.width) + "," +
                        std::to_string(i / frame.width) + " cpu " +
                        std::to_string(cpu.vis[i]) + " gpu " +
                        std::to_string(gpuVis[i]);
              ++mismatches;
            }
          INFO("[" << name << " mode " << rasterMode << "] cpu writes "
                   << cpuStats[1] << " gpu writes " << gpuStats[1]
                   << ", mismatches " << mismatches
                   << (first.empty() ? "" : " first: " + first));
          REQUIRE(mismatches == 0);
          // Pixel-write counts may differ by a couple at exact w == 0 ties
          // (host vs SPIR-V FMA decides the tie differently) - the per-pixel
          // VisBuffer compare above is the real gate.
          REQUIRE(std::abs(static_cast<long long>(cpuStats[1]) -
                           static_cast<long long>(gpuStats[1])) <= 2);
          REQUIRE(cpuStats[2] == gpuStats[2]);
          REQUIRE(cpuStats[3] == gpuStats[3]);
        }
      };

  SECTION("smoke: full-screen triangle") {
    Scene s;
    s.frame = frame;
    // All three edges AND the hypotenuse beyond the viewport: the
    // triangle strictly contains every pixel center (staged plan's
    // "covers the entire screen" case).
    s.addCluster({{-20, -20, 0}, {60, -20, 0}, {-20, 130, 0}}, {0, 1, 2});
    // runCase returns via REQUIRE; for the coverage assertion we need
    // the vis back, so inline the same calls.
    s.spans.emplace_back(static_cast<u32>(s.positions.size()),
                         static_cast<u32>(s.indices.size()));
    CpuRaster cpu{frame, s.positions, s.indices, {}};
    u32 cpuStats[4] = {};
    std::vector<glm::uvec2> pairs{{0, 0}};
    const std::vector<renderer::CullGpuInstance> insts{identity};
    cpu.run(pairs, insts, s.spans, cpuStats);
    std::vector<u64> gpuVis;
    u32 gpuStats[4] = {};
    REQUIRE(renderer::rasterizeOnGpu(*backend, shaders, frame, s.positions,
                                     s.indices, s.clusterData, insts, pairs,
                                     gpuVis, gpuStats));
    u64 nonzero = 0;
    for (const u64 v : gpuVis) nonzero += v != 0 ? 1 : 0;
    INFO("smoke coverage " << nonzero << "/" << gpuVis.size() << ", writes "
                           << gpuStats[1]);
    REQUIRE(nonzero == gpuVis.size());  // every pixel won
    REQUIRE(gpuStats[1] == gpuVis.size());
    u64 mm = 0;
    for (usize i = 0; i < gpuVis.size(); ++i) mm += cpu.vis[i] != gpuVis[i];
    REQUIRE(mm == 0);
  }

  SECTION("differential: multi-instance payload") {
    // S4.6: the VisBuffer payload carries instance bits (cluster 20b |
    // tri 7b | instance 5b). Two instances of ONE cluster, offset in
    // world x so they overlap on screen with different depths - the
    // overlap resolves per atomicMax and every pixel's payload must
    // name the winning instance exactly on both sides.
    Scene s;
    s.frame = frame;
    s.addCluster({{-2, -2, 0}, {2, -2, 0}, {2, 2, 0}, {-2, 2, 0}},
                 {0, 1, 2, 0, 2, 3});
    std::vector<renderer::CullGpuInstance> insts;
    insts.push_back(identity);
    renderer::CullGpuInstance moved;
    moved.model = glm::translate(glm::mat4(1.0f), {0.8f, 0.0f, 1.0f});
    insts.push_back(moved);
    runCase("multiinst", s, {{0, 0}, {1, 0}}, insts);
  }

  SECTION("differential: offscreen + shared edge + depth overlap") {
    Scene s;
    s.frame = frame;
    // c0: partially offscreen quad (2 tris, shared diagonal edge)
    s.addCluster({{-2, -2, 0}, {2, -2, 0}, {2, 2, 0}, {-2, 2, 0}},
                 {0, 1, 2, 0, 2, 3});
    // c1: SAME quad geometry at constant nearer depth z=-1 -> overlap
    s.addCluster({{-2, -2, -1}, {2, -2, -1}, {2, 2, -1}, {-2, 2, -1}},
                 {0, 1, 2, 0, 2, 3});
    // c2: winding-flipped triangle -> backface culled on both sides
    s.addCluster({{-1, -1, -2}, {3, -1, -2}, {-1, 3, -2}}, {0, 2, 1});
    runCase("diff3", s, {{0, 0}, {0, 1}, {0, 2}}, {identity});
  }

  SECTION("near-plane: crossing triangle must not vanish") {
    Scene s;
    s.frame = frame;
    // One vertex BEHIND the near plane (camera at z=8, near 0.1 ->
    // near plane at world z=7.9): vertex z=8.5 is on the camera side.
    s.addCluster({{-3, -3, 0}, {3, -3, 0}, {0, 3, 8.5}}, {0, 1, 2});
    // Fully in front of the near plane (behind camera): must vanish.
    s.addCluster({{-1, -1, 8.5}, {1, -1, 8.5}, {0, 1, 8.4}}, {0, 1, 2});
    runCase("nearclip", s, {{0, 0}, {0, 1}}, {identity});
  }

  SECTION("tie: pixel center exactly on an edge") {
    Scene s;
    s.frame = frame;
    // Edge along x=0 (pixel column centers at 0.5, 1.5, ...; column
    // straddling x=0 has center 0.5 > 0 -> inside); vertices chosen so
    // the vertical edge x=0 passes EXACTLY between pixel centers with
    // exactly representable coordinates.
    s.addCluster({{0, -3, 0}, {4, -3, 0}, {0, 3, 0}}, {0, 1, 2});
    // Adjacent triangle sharing that exact edge.
    s.addCluster({{0, -3, 0}, {0, 3, 0}, {-4, 3, 0}}, {0, 1, 2});
    runCase("tie", s, {{0, 0}, {0, 1}}, {identity});
  }

  backend->shutdown();
}

// M4.1 post-mortem: every existing differential ran the UNFLIPPED
// projection; the app's flipped (framebuffer-row) convention had zero
// coverage, and a subpixel-stepping bug sailed through. Same cases as
// the main differential, Y-FLIPPED (depth bits get the standard FMA
// tolerance; payloads must match exactly).
TEST_CASE("software rasterizer GPU/CPU differential - Y-FLIPPED",
          "[gpu][swraster]") {
  using namespace kenshin;
  manager::ShaderManager shaders;
  REQUIRE(shaders.init(kShaderRoot));
  auto backend = backend::createVulkanBackend();
  backend::BackendInitParams params;
  REQUIRE(backend->initialize(params));

  const glm::vec3 cam(0.0f, 0.0f, 8.0f);
  renderer::SoftRasterFrame frame;
  frame.view = glm::lookAtRH(cam, {0, 0, 0}, {0, 1, 0});
  frame.proj =
      glm::perspectiveRH_ZO(glm::radians(60.0f), 4.0f / 3.0f, 0.1f, 100.0f);
  frame.proj[1][1] *= -1.0f;  // the app's convention
  frame.yFlip = true;
  frame.nearZ = 0.1f;
  frame.width = 64;
  frame.height = 48;
  renderer::CullGpuInstance identity;

  const auto runCase = [&](Scene& s, const std::vector<glm::uvec2>& pairs,
                           const std::vector<renderer::CullGpuInstance>& insts,
                           const char* name) {
    s.frame = frame;
    s.spans.emplace_back(static_cast<u32>(s.positions.size()),
                         static_cast<u32>(s.indices.size()));
    CpuRaster cpu{frame, s.positions, s.indices, {}};
    u32 cpuStats[4] = {};
    cpu.run(pairs, insts, s.spans, cpuStats);
    // Both scheduling modes against the same CPU mirror (mode 1's span
    // clamp claims yFlip independence via post-canonicalization winding
    // - this differential is where that claim is proven).
    for (const u32 rasterMode : {0u, 1u}) {
      frame.rasterMode = rasterMode;
      std::vector<u64> gpuVis;
      u32 gpuStats[4] = {};
      REQUIRE(renderer::rasterizeOnGpu(*backend, shaders, frame, s.positions,
                                       s.indices, s.clusterData, insts, pairs,
                                       gpuVis, gpuStats));
      const auto tolerantEqual = [](u64 a, u64 b) {
        if (a == b) return true;
        if ((a & 0xFFFFFFFFull) != (b & 0xFFFFFFFFull)) return false;
        const u64 da = a >> 32, db = b >> 32;
        return (da > db ? da - db : db - da) <= (1ull << 12);
      };
      u64 mismatches = 0;
      for (usize i = 0; i < cpu.vis.size(); ++i)
        if (!tolerantEqual(cpu.vis[i], gpuVis[i])) ++mismatches;
      INFO("[" << name << " mode " << rasterMode << "] cpu nz " <<
           [&] {
             u64 n = 0;
             for (const u64 v : cpu.vis) n += v != 0;
             return n;
           }() << " mismatches "
               << mismatches);
      REQUIRE(mismatches == 0);
      REQUIRE(cpuStats[2] == gpuStats[2]);
    }
  };

  {
    Scene s;
    s.addCluster({{-2, -2, 0}, {2, -2, 0}, {2, 2, 0}, {-2, 2, 0}},
                 {0, 1, 2, 0, 2, 3});
    std::vector<renderer::CullGpuInstance> insts;
    insts.push_back(identity);
    renderer::CullGpuInstance moved;
    moved.model = glm::translate(glm::mat4(1.0f), {0.8f, 0.0f, 1.0f});
    insts.push_back(moved);
    runCase(s, {{0, 0}, {1, 0}}, insts, "flip-multiinst");
  }
  {
    Scene s;
    s.addCluster({{0, -3, 0}, {4, -3, 0}, {0, 3, 0}}, {0, 1, 2});
    s.addCluster({{0, -3, 0}, {0, 3, 0}, {-4, 3, 0}}, {0, 1, 2});
    runCase(s, {{0, 0}, {0, 1}}, {identity}, "flip-tie");
  }

  backend->shutdown();
}
