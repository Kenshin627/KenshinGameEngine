// kgeo_builder CLI (S3.3): input.obj | --proc <spec> -> output.kgeo

#include <algorithm>
#include <bit>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <glm/glm.hpp>
#include <string>
#include <vector>

#include "kenshin/Version.h"
#include "kenshin/assets/MeshData.h"
#include "kenshin/assets/ObjLoader.h"
#include "kenshin/core/Log.h"
#include "kenshin/geometry/ClusterDag.h"
#include "kenshin/geometry/ClusterLodBuilder.h"
#include "kenshin/geometry/KGeoFile.h"
#include "kenshin/geometry/shapes/SphereGeometry.h"
#include "kenshin/geometry/shapes/TorusKnotGeometry.h"

int main(int argc, char** argv) {
  kenshin::log::init();
  KENSHIN_INFO("{}", kenshin::getBanner());

  std::string input, output = "out/mesh.kgeo";
  std::string proc;      // "sphere" or "knot"
  bool unpaged = false;  // --unpaged: emit v2 (pools) instead of v3 (pages)
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc)
      output = argv[++i];
    else if (std::strcmp(argv[i], "--unpaged") == 0)
      unpaged = true;
    else if (std::strcmp(argv[i], "--proc") == 0 && i + 1 < argc)
      proc = argv[++i];
    else
      input = argv[i];
  }
  if (input.empty() && proc.empty()) {
    KENSHIN_ERROR("usage: kgeo_builder input.obj [-o out.kgeo]");
    KENSHIN_ERROR(
        "       kgeo_builder --proc knot|sphere[:segsA[:segsB]] [-o out]");
    kenshin::log::shutdown();
    return 1;
  }

  kenshin::assets::MeshData mesh;
  std::string log;
  if (!proc.empty()) {
    // --proc name[:segsA[:segsB]] - segment overrides for the M4.4
    // large-triangle acceptance (coarse meshes -> every cluster's max
    // edge far above the 18px HW routing threshold up close).
    std::string name = proc;
    int segA = 0, segB = 0;
    if (const auto colon = proc.find(':'); colon != std::string::npos) {
      name = proc.substr(0, colon);
      segA = std::atoi(proc.c_str() + colon + 1);
      if (const auto colon2 = proc.find(':', colon + 1);
          colon2 != std::string::npos)
        segB = std::atoi(proc.c_str() + colon2 + 1);
    }
    const int knotTube = segA > 0 ? segA : 512;
    const int knotRing = segB > 0 ? segB : 1024;
    const int sphereU = segA > 0 ? segA : 256;
    const int sphereV = segB > 0 ? segB : 128;
    if (name == "knot")
      mesh = kenshin::geometry::TorusKnotGeometry::create(2, 3, knotTube,
                                                          knotRing, 1.0f);
    else if (name == "sphere")
      mesh = kenshin::geometry::SphereGeometry::create(sphereU, sphereV, 1.0f);
    else {
      KENSHIN_ERROR("unknown --proc '{}'", proc);
      kenshin::log::shutdown();
      return 1;
    }
  } else if (!kenshin::assets::loadObj(input, mesh, log)) {
    KENSHIN_ERROR("load failed: {}", log);
    kenshin::log::shutdown();
    return 1;
  }
  KENSHIN_INFO("[builder] mesh: {} verts / {} tris", mesh.vertexCount(),
               mesh.triangleCount());

  std::vector<kenshin::geometry::ClusterGroup> groups;
  kenshin::geometry::BuildStats buildStats;
  if (!kenshin::geometry::buildClusterLod(
          mesh,
          [&](const kenshin::geometry::ClusterGroup& g) {
            groups.push_back(g);
          },
          buildStats, log)) {
    KENSHIN_ERROR("clusterlod failed: {}", log);
    kenshin::log::shutdown();
    return 1;
  }

  kenshin::geometry::ClusterDag dag;
  kenshin::geometry::DagBuildStats dagStats;
  if (!kenshin::geometry::assembleClusterDag(mesh, groups, dag, dagStats,
                                             log)) {
    KENSHIN_ERROR("dag assembly failed: {}", log);
    kenshin::log::shutdown();
    return 1;
  }
  KENSHIN_INFO(
      "[builder] dag: {} clusters, {} groups, {} levels, {} roots, "
      "{} monotonic fixes, hierarchy {} nodes (root {})",
      dagStats.clusterCount, dagStats.groupCount, dagStats.maxLevel,
      dagStats.rootCount, dagStats.monotonicFixes, dag.hierarchyNodes.size(),
      dag.hierarchyRoot);

  // M7.2 Step-0 probe: triangle index-span distribution under (a) the
  // existing first-use vertex ordering (the u8 locals ARE first-use)
  // and (b) a within-cluster Morton (Z-order) reordering by position -
  // the candidate locality scheme for the packed-index layout. A span
  // s needs offsetBits >= bit_width(s); s <= 31 fits the canonical
  // 7+5+5 (17-bit) layout.
  {
    using kenshin::f32;
    using kenshin::u32;
    using kenshin::usize;
    const auto spanStats = [&](const char* tag, std::vector<u32>& spans,
                               std::vector<u32>& clusterMaxSpan) {
      std::sort(spans.begin(), spans.end());
      const auto pct = [&](double p) {
        return spans.empty() ? 0
                             : spans[std::min<usize>(spans.size() - 1,
                                                     usize(p * spans.size()))];
      };
      usize over31 = 0;
      for (u32 s : spans) over31 += s > 31;
      // Per-cluster offsetBits histogram (bit_width(maxSpan)).
      u32 bitsHist[8] = {};
      for (u32 m : clusterMaxSpan) bitsHist[m == 0 ? 0 : std::bit_width(m)]++;
      std::string hist;
      for (u32 b = 0; b < 8; ++b)
        if (bitsHist[b])
          hist += " b" + std::to_string(b) + "=" + std::to_string(bitsHist[b]);
      KENSHIN_INFO(
          "[m72probe/{}] tri spans: p50={} p90={} p99={} max={} | >31: {} "
          "({:.3}%) | cluster maxSpan offsetBits: {}",
          tag, pct(0.50), pct(0.90), pct(0.99), spans.back(), over31,
          100.0 * double(over31) / double(std::max<usize>(spans.size(), 1)),
          hist);
    };

    std::vector<u32> spans, spansM, spansT, clusterMax, clusterMaxM,
        clusterMaxT;
    spans.reserve(dag.indices.size() / 3);
    spansM.reserve(dag.indices.size() / 3);
    for (const auto& cl : dag.clusters) {
      const usize n = cl.vertexCount;
      const usize triN = cl.indexCount / 3;
      std::vector<u32> morton(n), remap(n), inv(n);
      glm::vec3 mn(FLT_MAX), mx(-FLT_MAX);
      for (u32 v = 0; v < n; ++v) {
        const f32* p = dag.positions.data() + usize(cl.vertexOffset + v) * 3;
        mn = glm::min(mn, glm::vec3(p[0], p[1], p[2]));
        mx = glm::max(mx, glm::vec3(p[0], p[1], p[2]));
      }
      const glm::vec3 ext = glm::max(mx - mn, glm::vec3(1e-20f));
      const auto mortonOf = [&](const f32* p) {
        u32 code = 0;
        for (int a = 0; a < 3; ++a) {
          const u32 q =
              u32(std::min(
                  255.0,
                  std::max(0.0, std::floor(255.0 * (p[a] - mn[a]) / ext[a])))) &
              255u;
          for (int b = 0; b < 8; ++b) code |= ((q >> b) & 1u) << (b * 3 + a);
        }
        return code;
      };
      for (u32 v = 0; v < n; ++v) {
        morton[v] =
            mortonOf(dag.positions.data() + usize(cl.vertexOffset + v) * 3);
        remap[v] = v;
      }
      std::sort(remap.begin(), remap.end(),
                [&](u32 a, u32 b) { return morton[a] < morton[b]; });
      for (u32 i = 0; i < n; ++i) inv[remap[i]] = i;

      // (c) Triangle Morton order (by centroid), THEN first-use vertex
      // numbering - renumbers connectivity along a spatial walk.
      std::vector<u32> triOrder(triN), tMorton(triN), invT(n);
      for (u32 t = 0; t < triN; ++t) {
        const u32 i0 = dag.indices[usize(cl.indexOffset) + t * 3 + 0];
        const u32 i1 = dag.indices[usize(cl.indexOffset) + t * 3 + 1];
        const u32 i2 = dag.indices[usize(cl.indexOffset) + t * 3 + 2];
        glm::vec3 c(0.0f);
        for (u32 i : {i0, i1, i2}) {
          const f32* p = dag.positions.data() + usize(cl.vertexOffset + i) * 3;
          c += glm::vec3(p[0], p[1], p[2]);
        }
        const glm::vec3 ctr = c / 3.0f;
        tMorton[t] = mortonOf(&ctr[0]);
        triOrder[t] = t;
      }
      std::sort(triOrder.begin(), triOrder.end(),
                [&](u32 a, u32 b) { return tMorton[a] < tMorton[b]; });
      {
        u32 next = 0;
        for (u32 t : triOrder)
          for (int k = 0; k < 3; ++k) {
            const u32 i = dag.indices[usize(cl.indexOffset) + t * 3 + k];
            if (invT[i] == 0u) {
              // 0 is ambiguous as "unassigned"; use +1 sentinel.
              invT[i] = next + 1;
              ++next;
            }
          }
        for (u32 v = 0; v < n; ++v) invT[v] = invT[v] == 0 ? 0 : invT[v] - 1;
      }

      u32 worst = 0, worstM = 0, worstT = 0;
      for (u32 t = 0; t + 2 < cl.indexCount; t += 3) {
        const u32 i0 = dag.indices[usize(cl.indexOffset) + t + 0];
        const u32 i1 = dag.indices[usize(cl.indexOffset) + t + 1];
        const u32 i2 = dag.indices[usize(cl.indexOffset) + t + 2];
        const u32 s = std::max({i0, i1, i2}) - std::min({i0, i1, i2});
        spans.push_back(s);
        worst = std::max(worst, s);
        const u32 m0 = inv[i0], m1 = inv[i1], m2 = inv[i2];
        const u32 sM = std::max({m0, m1, m2}) - std::min({m0, m1, m2});
        spansM.push_back(sM);
        worstM = std::max(worstM, sM);
        const u32 t0 = invT[i0], t1 = invT[i1], t2 = invT[i2];
        const u32 sT = std::max({t0, t1, t2}) - std::min({t0, t1, t2});
        spansT.push_back(sT);
        worstT = std::max(worstT, sT);
      }
      clusterMax.push_back(worst);
      clusterMaxM.push_back(worstM);
      clusterMaxT.push_back(worstT);
    }
    spanStats("first-use", spans, clusterMax);
    spanStats("morton", spansM, clusterMaxM);
    spanStats("tri-morton+first-use", spansT, clusterMaxT);
  }

  if (unpaged) {
    if (!kenshin::geometry::writeKGeo(output, dag, log)) {
      KENSHIN_ERROR("write failed: {}", log);
      kenshin::log::shutdown();
      return 1;
    }
    KENSHIN_INFO("[builder] written (v2 unpaged): {}", output);
    kenshin::log::shutdown();
    return 0;
  }
  kenshin::geometry::PackedGeometry packed;
  if (!kenshin::geometry::packPages(dag, packed, log)) {
    KENSHIN_ERROR("page packing failed: {}", log);
    kenshin::log::shutdown();
    return 1;
  }
  if (!kenshin::geometry::writeKGeoPaged(output, dag, packed, log)) {
    KENSHIN_ERROR("write failed: {}", log);
    kenshin::log::shutdown();
    return 1;
  }
  KENSHIN_INFO("[builder] written (v5 quantized paged): {}", output);
  kenshin::log::shutdown();
  return 0;
}
