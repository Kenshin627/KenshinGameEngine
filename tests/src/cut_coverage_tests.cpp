// P0 cut-coverage probe: is the LOD cut geometrically COMPLETE on
// bigknot? Pure CPU, no GPU. For each original-mesh cluster C0 the
// surface patch it covers must be represented in the cut by SOME
// coarser descendant (C0's transitive children). An uncovered C0 is a
// hole: no selected cluster draws that patch, at any level.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <queue>
#include <set>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "kenshin/Common.h"
#include "kenshin/geometry/ClusterDag.h"
#include "kenshin/geometry/KGeoFile.h"
#include "kenshin/geometry/LodCut.h"
#include "kenshin/util/File.h"

using namespace kenshin;

namespace {
// Transitive closure of a cluster's coarser derivatives.
void descendantsOf(u32 root, const kenshin::geometry::ClusterDag& dag,
                   std::set<u32>& out) {
  std::queue<u32> q;
  q.push(root);
  while (!q.empty()) {
    const u32 c = q.front();
    q.pop();
    for (u32 child : dag.clusters[c].children) {
      if (out.insert(child).second) q.push(child);
    }
  }
}
}  // namespace

TEST_CASE("selectLodCut coverage (bigknot, user camera)", "[p0]") {
  using namespace kenshin::geometry;
  ClusterDag dag;
  std::string log;
  REQUIRE(readKGeo("out/bigknot.kgeo", dag, log));
  INFO(log);

  // Pool consistency: every cluster's index/vertex slice must fit the
  // serialized pools exactly (sum-packed by the builder).
  {
    u64 sumIdx = 0, sumVert = 0;
    u32 over = 0, overV = 0;
    for (const auto& c : dag.clusters) {
      sumIdx += c.indexCount;
      sumVert += c.vertexCount;
      if (u64(c.indexOffset) + c.indexCount > dag.indices.size()) ++over;
      if (u64(c.vertexOffset) + c.vertexCount > dag.positions.size() / 3)
        ++overV;
    }
    REQUIRE(sumIdx == dag.indices.size());
    REQUIRE(sumVert == dag.positions.size() / 3);
    REQUIRE(over == 0);
    REQUIRE(overV == 0);
  }

  // Original (finest) clusters: refinedGroup < 0.
  std::vector<u32> originals;
  for (u32 c = 0; c < dag.clusters.size(); ++c)
    if (dag.clusters[c].refinedGroup < 0) originals.push_back(c);
  REQUIRE(!originals.empty());

  // The grid64 layout + the user's default camera.
  const float cam[3] = {0.27f, 0.0f, 6.07f};
  const u32 H = 461;
  const f32 fov = 60.0f;

  for (const f32 th : {0.25f, 1.0f, 4.0f}) {
    // Union of every instance's selected cluster set.
    std::set<u32> selectedAny;
    for (u32 i = 0; i < 64; ++i) {
      const float local[3] = {cam[0] - (i % 8) * 2.25f, cam[1],
                              cam[2] - (i / 8) * 2.25f};
      std::vector<u32> cut;
      selectLodCut(dag, th, H, fov, local, cut);
      selectedAny.insert(cut.begin(), cut.end());
    }

    // Coverage: every original cluster needs >= 1 selected cluster in
    // its own descendants (itself included - an unrefined original can
    // be selected directly).
    u32 covered = 0, orphanLinks = 0, blockedChain = 0;
    std::vector<u32> uncovered;
    for (u32 c0 : originals) {
      std::set<u32> desc;
      desc.insert(c0);
      descendantsOf(c0, dag, desc);
      bool hit = false;
      for (u32 d : desc)
        if (selectedAny.count(d)) {
          hit = true;
          break;
        }
      if (hit) {
        ++covered;
        continue;
      }
      uncovered.push_back(c0);
      // Diagnose: no coarser derivatives at all (tiling/link loss) vs
      // descendants exist but none selected (threshold/link mismatch).
      if (dag.clusters[c0].children.empty())
        ++orphanLinks;
      else
        ++blockedChain;
    }
    printf("[cutcov] th=%.2f selected(any inst)=%zu originals=%zu "
           "covered=%zu uncovered=%zu [no-children=%u chain-blocked=%u]\n",
           th, selectedAny.size(), originals.size(), size_t(covered) - 0,
           uncovered.size(), orphanLinks, blockedChain);

    // (chain printing omitted in release runs - see [cutcov] summary)
  }
}

// ── P0 raster probe: quantized (shader mirror) vs TRUE (float) ────────
// Same cut, same view: pixels covered by the float-exact raster but
// missed by the 16.8-snap raster are the subpixel-quantization loss.
TEST_CASE("quantized-vs-true raster occupancy (bigknot, user cam)",
          "[p0][p0raster]") {
  using namespace kenshin;
  using namespace kenshin::geometry;
  ClusterDag dag;
  std::string log;
  REQUIRE(readKGeo("out/bigknot.kgeo", dag, log));

  const float cam[3] = {0.27f, 0.0f, 6.07f};
  const u32 W = 640, H = 461;
  const f32 fov = 60.0f, nearZ = 0.1f;
  glm::mat4 view(1.0f);
  {
    const float v[16] = {-0.0000f, +0.2588f, -0.9659f, +0.0000f,
                         -1.0000f, -0.0000f, +0.0000f, +0.0000f,
                         +0.0000f, +0.9659f, +0.2588f, +0.0000f,
                         +0.0000f, -5.9343f, -1.3079f, +1.0000f};
    std::memcpy(glm::value_ptr(view), v, sizeof(v));
  }
  glm::mat4 proj = glm::perspectiveRH_ZO(glm::radians(fov), f32(W) / f32(H),
                                         nearZ, 100.0f);
  proj[1][1] *= -1.0f;  // y-flipped: cullSign = +1 (shader convention)

  // Pair list: per-instance selectLodCut (proven == GPU selection).
  std::vector<glm::uvec2> pairs;
  std::vector<glm::mat4> models;
  for (u32 i = 0; i < 64; ++i) {
    const float local[3] = {cam[0] - (i % 8) * 2.25f, cam[1],
                            cam[2] - (i / 8) * 2.25f};
    std::vector<u32> cut;
    selectLodCut(dag, 1.0f, H, fov, local, cut);
    for (u32 c : cut) pairs.emplace_back(i, c);
    models.push_back(glm::translate(glm::mat4(1.0f),
                                    glm::vec3((i % 8) * 2.25f, 0.0f,
                                              (i / 8) * 2.25f)));
  }
  printf("[p0ras] pairs=%zu\n", pairs.size());

  std::vector<u8> quant(W * H, 0);    // shader-mirror occupancy
  std::vector<u8> truth(W * H, 0);    // float-exact occupancy
  std::vector<u16> truthInst(W * H, 0xFFFF);

  u32 degenerateAfterSnap = 0, noCenterPixels = 0, trueWrites = 0;

  for (usize p = 0; p < pairs.size(); ++p) {
    const u32 c = pairs[p].y;
    const DagCluster& cl = dag.clusters[c];
    const glm::mat4& model = models[pairs[p].x];
    std::vector<glm::vec3> viewV(cl.vertexCount);
    for (u32 v = 0; v < cl.vertexCount; ++v) {
      const glm::vec3 lp(dag.positions[(cl.vertexOffset + v) * 3 + 0],
                         dag.positions[(cl.vertexOffset + v) * 3 + 1],
                         dag.positions[(cl.vertexOffset + v) * 3 + 2]);
      const glm::vec3 wp = glm::vec3(model * glm::vec4(lp, 1.0f));
      viewV[v] = glm::vec3(view * glm::vec4(wp, 1.0f));
    }
    const u32 triCount = cl.indexCount / 3;
    for (u32 t = 0; t < triCount; ++t) {
      const u32 i0 = dag.indices[cl.indexOffset + t * 3 + 0];
      const u32 i1 = dag.indices[cl.indexOffset + t * 3 + 1];
      const u32 i2 = dag.indices[cl.indexOffset + t * 3 + 2];
      const glm::vec3 v0 = viewV[i0], v1 = viewV[i1], v2 = viewV[i2];
      // Sutherland-Hodgman near clip (same as shader).
      std::vector<glm::vec3> poly;
      const glm::vec3 inV[3] = {v0, v1, v2};
      for (int e = 0; e < 3; ++e) {
        const glm::vec3& a = inV[e];
        const glm::vec3& b = inV[(e + 1) % 3];
        const bool aIn = a.z <= -nearZ, bIn = b.z <= -nearZ;
        if (aIn) {
          poly.push_back(a);
          if (!bIn)
            poly.push_back(a + (b - a) * ((-nearZ - a.z) / (b.z - a.z)));
        } else if (bIn) {
          poly.push_back(a + (b - a) * ((-nearZ - a.z) / (b.z - a.z)));
        }
      }
      if (poly.size() < 3) continue;
      for (usize tri = 0; tri + 3 <= poly.size(); ++tri) {
        // fan (0,1,2) (+ (0,2,3))
        const glm::vec3& pa = poly[0];
        const glm::vec3& pb = poly[tri + 1];
        const glm::vec3& pc = poly[tri + 2];
        const glm::vec2 wh{f32(W), f32(H)};
        const auto screen = [&](const glm::vec3& q) {
          const glm::vec4 clip =
              proj[0] * q.x + proj[1] * q.y + proj[2] * q.z + proj[3];
          const glm::vec3 ndc(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
          return glm::vec2((ndc.x * 0.5f + 0.5f) * wh.x,
                           (ndc.y * 0.5f + 0.5f) * wh.y);
        };
        const glm::vec2 sa = screen(pa), sb = screen(pb), sc2 = screen(pc);

        // ---- TRUE occupancy: float edges at pixel centers.
        {
          glm::vec2 A = sa, B = sb, C = sc2;
          float area2f = (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
          if (area2f == 0.0f) goto doQuant;
          if (area2f > 0.0f) goto doQuant;  // backface (yFlip culls +)
          std::swap(B, C);                  // canonical positive winding
          const f32 minx = std::min({A.x, B.x, C.x});
          const f32 maxx = std::max({A.x, B.x, C.x});
          const f32 miny = std::min({A.y, B.y, C.y});
          const f32 maxy = std::max({A.y, B.y, C.y});
          int x0 = std::max(int(std::floor(minx)), 0);
          int x1 = std::min(int(std::ceil(maxx)), int(W) - 1);
          int y0 = std::max(int(std::floor(miny)), 0);
          int y1 = std::min(int(std::ceil(maxy)), int(H) - 1);
          for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) {
              const float px = x + 0.5f, py = y + 0.5f;
              const float w0 = (B.x - A.x) * (py - A.y) -
                               (B.y - A.y) * (px - A.x);
              const float w1 = (C.x - B.x) * (py - B.y) -
                               (C.y - B.y) * (px - B.x);
              const float w2 = (A.x - C.x) * (py - C.y) -
                               (A.y - C.y) * (px - C.x);
              if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
                const usize idx = usize(y) * W + x;
                truth[idx] = 1;
                truthInst[idx] = u16(pairs[p].x);
                ++trueWrites;
              }
            }
        }
      doQuant:
        // ---- QUANTIZED occupancy: the shader's 16.8 snap mirror.
        {
          constexpr int SUBPIXEL = 256;
          const auto quantF = [](float v) {
            return (long long)(std::floor(v * float(SUBPIXEL)));
          };
          const long long X0 = quantF(sa.x), Y0 = quantF(sa.y);
          long long X1 = quantF(sb.x), Y1 = quantF(sb.y);
          long long X2 = quantF(sc2.x), Y2 = quantF(sc2.y);
          const long long area2 =
              (X1 - X0) * (Y2 - Y0) - (Y1 - Y0) * (X2 - X0);
          if (area2 == 0) {
            ++degenerateAfterSnap;
            continue;
          }
          const long long cullSign = 1;  // yFlip
          if (area2 * cullSign > 0) continue;  // backface
          long long aX1 = X1, aY1 = Y1, aX2 = X2, aY2 = Y2;
          if (area2 < 0) {
            std::swap(aX1, aX2);
            std::swap(aY1, aY2);
          }
          const long long e0x = aX1 - X0, e0y = aY1 - Y0;
          const long long e1x = aX2 - aX1, e1y = aY2 - aY1;
          const long long e2x = X0 - aX2, e2y = Y0 - aY2;
          const auto t0 = (e0y == 0 && e0x > 0) || e0y < 0 ? 0LL : 1LL;
          const auto t1 = (e1y == 0 && e1x > 0) || e1y < 0 ? 0LL : 1LL;
          const auto t2 = (e2y == 0 && e2x > 0) || e2y < 0 ? 0LL : 1LL;
          const long long minSubX = std::min({X0, aX1, aX2});
          const long long maxSubX = std::max({X0, aX1, aX2});
          const long long minSubY = std::min({Y0, aY1, aY2});
          const long long maxSubY = std::max({Y0, aY1, aY2});
          int x0 = std::max(
              int(std::ceil((float(minSubX) - 128.0f) / 256.0f)), 0);
          int x1 = std::min(
              int(std::floor((float(maxSubX) - 129.0f) / 256.0f)), int(W) - 1);
          int y0 = std::max(
              int(std::ceil((float(minSubY) - 128.0f) / 256.0f)), 0);
          int y1 = std::min(
              int(std::floor((float(maxSubY) - 129.0f) / 256.0f)), int(H) - 1);
          if (x0 > x1 || y0 > y1) continue;
          u32 hits = 0;
          for (int y = y0; y <= y1; ++y) {
            const long long dySub = (long long)(y) * SUBPIXEL + 128 -
                                    ((long long)(y0) * SUBPIXEL + 128);
            for (int x = x0; x <= x1; ++x) {
              const long long dxSub = static_cast<long long>(x) * SUBPIXEL + 128 -
                                      ((long long)(x0) * SUBPIXEL + 128);
              const long long px = (long long)(x0) * SUBPIXEL + 128 + dxSub;
              const long long py = (long long)(y0) * SUBPIXEL + 128 + dySub;
              const auto edgeW = [](long long ex, long long ey, long long qx,
                                    long long qy, long long vx, long long vy) {
                return ex * (qy - vy) - ey * (qx - vx);
              };
              const long long w0 = edgeW(e0x, e0y, px, py, X0, Y0);
              const long long w1 = edgeW(e1x, e1y, px, py, aX1, aY1);
              const long long w2 = edgeW(e2x, e2y, px, py, aX2, aY2);
              if (w0 >= t0 && w1 >= t1 && w2 >= t2) {
                quant[usize(y) * W + x] = 1;
                ++hits;
              }
            }
          }
          if (hits == 0) ++noCenterPixels;
        }
      }
    }
  }

  // Diff: true-covered but quantized-missing.
  u32 missing = 0;
  std::vector<u32> missPerInst(64, 0);
  for (usize i = 0; i < quant.size(); ++i)
    if (truth[i] && !quant[i]) {
      ++missing;
      if (truthInst[i] != 0xFFFF) ++missPerInst[truthInst[i]];
    }
  const u32 truthCount = (u32)std::count(truth.begin(), truth.end(), 1);
  const u32 quantCount = (u32)std::count(quant.begin(), quant.end(), 1);
  printf("[p0ras] truth=%u quant=%u | MISSING(truth&&!quant)=%u "
         "(%.2f%% of truth) degenerateAfterSnap=%u triNoCenter=%u\n",
         truthCount, quantCount, missing,
         100.0f * missing / std::max(truthCount, 1u), degenerateAfterSnap,
         noCenterPixels);
  u32 worst = 0, worstInst = 0;
  for (u32 i = 0; i < 64; ++i)
    if (missPerInst[i] > worst) {
      worst = missPerInst[i];
      worstInst = i;
    }
  printf("[p0ras] worst instance=%u missing=%u (offset %.2f,%.2f)\n",
         worstInst, worst, (worstInst % 8) * 2.25f, (worstInst / 8) * 2.25f);
  // Save occupancy bitmaps as PGMs for visual inspection.
  const auto savePGM = [](const char* path, const std::vector<u8>& img, u32 w,
                          u32 h) {
    FILE* f = kenshin::util::openFile(path, "wb");
    if (!f) return;
    fprintf(f, "P5\n%u %u\n255\n", w, h);
    fwrite(img.data(), 1, img.size(), f);
    fclose(f);
  };
  savePGM("C:/tmp/p0_truth.pgm", truth, W, H);
  savePGM("C:/tmp/p0_quant.pgm", quant, W, H);
  std::vector<u8> diff(W * H, 0);
  for (usize i = 0; i < quant.size(); ++i)
    diff[i] = truth[i] && !quant[i] ? 255 : (truth[i] ? 80 : 0);
  savePGM("C:/tmp/p0_diff.pgm", diff, W, H);
  printf("[p0ras] wrote C:/tmp/p0_truth.pgm p0_quant.pgm p0_diff.pgm\n");
}
