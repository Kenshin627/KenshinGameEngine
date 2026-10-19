// kgeo_builder CLI (S3.3): input.obj | --proc <spec> -> output.kgeo

#include <algorithm>
#include <bit>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <glm/glm.hpp>
#include <string>
#include <vector>

#include "kenshin/Version.h"
#include "kenshin/assets/GltfLoader.h"
#include "kenshin/assets/MeshData.h"
#include "kenshin/util/BmpWriter.h"

namespace fsg = std::filesystem;
#include "kenshin/assets/ObjLoader.h"
#include "kenshin/core/Log.h"
#include "kenshin/geometry/ClusterDag.h"
#include "kenshin/geometry/ClusterLodBuilder.h"
#include "kenshin/geometry/ImposterBaker.h"
#include "kenshin/geometry/KGeoFile.h"
#include "kenshin/geometry/shapes/SphereGeometry.h"
#include "kenshin/geometry/shapes/TorusKnotGeometry.h"

// Sketchfab/FBX-sourced glTF assets routinely sit FAR off the origin
// (a scan pivot 172 units up, say) - the raw scene is technically
// correct but unusable as a display asset: the default camera misses
// it, the LOD/shadow metrics degrade, and the v5 quant grid wastes
// precision. Recenter: xy to the bbox center, the bottom to z=0 (the
// ground-plane convention). `fit` additionally scales so the longest
// dimension = N units (cm-sourced scans).
static void recenterGltf(kenshin::assets::MeshData& mesh, float fit) {
  if (mesh.positions.empty()) return;
  float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
  for (kenshin::usize v = 0; v < mesh.vertexCount(); ++v)
    for (int k = 0; k < 3; ++k) {
      const float p = mesh.positions[v * 3 + k];
      mn[k] = std::min(mn[k], p);
      mx[k] = std::max(mx[k], p);
    }
  const float cx = (mn[0] + mx[0]) * 0.5f, cy = (mn[1] + mx[1]) * 0.5f;
  const float size = std::max({mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]});
  const float scale = fit > 0.0f && size > 1e-9f ? fit / size : 1.0f;
  if (std::abs(cx) > 1e-6f || std::abs(cy) > 1e-6f || std::abs(mn[2]) > 1e-6f ||
      scale != 1.0f) {
    for (kenshin::usize v = 0; v < mesh.vertexCount(); ++v) {
      mesh.positions[v * 3 + 0] = (mesh.positions[v * 3 + 0] - cx) * scale;
      mesh.positions[v * 3 + 1] = (mesh.positions[v * 3 + 1] - cy) * scale;
      mesh.positions[v * 3 + 2] = (mesh.positions[v * 3 + 2] - mn[2]) * scale;
    }
    KENSHIN_INFO(
        "[builder] gltf recentered (-{:.2f},-{:.2f},-{:.2f}){} bbox {:.2f} -> "
        "{:.2f} units",
        cx, cy, mn[2], scale != 1.0f ? " + scaled" : "", size, size * scale);
  }
}

int main(int argc, char** argv) {
  kenshin::log::init();
  KENSHIN_INFO("{}", kenshin::getBanner());

  std::string input, output = "out/mesh.kgeo";
  float fit = 0.0f;  // --fit N: uniform scale so the longest bbox
                     // dimension lands at N units (0 = as loaded)
  // Multi-material glTF set (filled only when the asset carries more
  // than one used material; single-material inputs take the legacy
  // single-table path unchanged).
  std::vector<kenshin::assets::GltfMaterial> gltfMaterials;
  std::vector<kenshin::assets::GltfImage> gltfImages;
  std::string proc;      // "sphere" or "knot"
  bool unpaged = false;  // --unpaged: emit v2 (pools) instead of v3+ (pages)
  // M8.2 material overrides (single material v1; OBJ inputs already
  // carry Kd/map_Kd from their MTL - the CLI wins where given).
  std::string albedo;  // --albedo <path>: 24-bit BMP, copied next to -o
  float kd[3] = {-1.0f, -1.0f, -1.0f};  // --kd r g b
  float rough = -1.0f, metal = -1.0f;   // --rough f / --metal f
  bool debugDump = false;               // --debug-dump: per-level DAG stats
  bool noCompress = false;              // --no-compress: emit v7 (raw pages)
  bool imposter = false;                // --imposter: bake the far-field
                                        // atlas (root cut) into the file
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc)
      output = argv[++i];
    else if (std::strcmp(argv[i], "--unpaged") == 0)
      unpaged = true;
    else if (std::strcmp(argv[i], "--proc") == 0 && i + 1 < argc)
      proc = argv[++i];
    else if (std::strcmp(argv[i], "--albedo") == 0 && i + 1 < argc)
      albedo = argv[++i];
    else if (std::strcmp(argv[i], "--kd") == 0 && i + 3 < argc) {
      kd[0] = std::atof(argv[++i]);
      kd[1] = std::atof(argv[++i]);
      kd[2] = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--rough") == 0 && i + 1 < argc)
      rough = std::atof(argv[++i]);
    else if (std::strcmp(argv[i], "--metal") == 0 && i + 1 < argc)
      metal = static_cast<float>(std::atof(argv[++i]));
    else if (std::strcmp(argv[i], "--fit") == 0 && i + 1 < argc)
      fit = static_cast<float>(std::atof(argv[++i]));
    else if (std::strcmp(argv[i], "--debug-dump") == 0)
      debugDump = true;
    else if (std::strcmp(argv[i], "--no-compress") == 0)
      noCompress = true;
    else if (std::strcmp(argv[i], "--imposter") == 0)
      imposter = true;
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
  } else {
    std::string inExt = fsg::path(input).extension().string();
    for (char& c : inExt) c = char(tolower(unsigned(c)));
    if (inExt == ".glb" || inExt == ".gltf") {
      kenshin::assets::GltfImage albedo;
      if (!kenshin::assets::loadGltf(input, mesh, &albedo, log, &gltfMaterials,
                                     &gltfImages)) {
        KENSHIN_ERROR("glTF load failed: {}", log);
        kenshin::log::shutdown();
        return 1;
      }
      if (!log.empty()) KENSHIN_WARN("[builder] gltf notes: {}", log);
      if (albedo.width > 0 && gltfMaterials.size() <= 1) {
        // Re-encode the embedded/decoded albedo as a BMP next to the
        // OUTPUT (the .kgeo's texture reference is a bare file name -
        // the pair travels together like an --albedo copy).
        const fsg::path outDir = fsg::path(output).parent_path();
        const std::string stem = albedo.name.empty()
                                     ? fsg::path(input).stem().string()
                                     : albedo.name;
        const fsg::path bmp = outDir / (stem + "_albedo.bmp");
        if (!kenshin::util::writeBmp24(bmp.string(), albedo.rgba,
                                       kenshin::u32(albedo.width),
                                       kenshin::u32(albedo.height))) {
          KENSHIN_ERROR("[builder] albedo re-encode failed: {}", bmp.string());
          kenshin::log::shutdown();
          return 1;
        }
        mesh.material.diffuseTexture = bmp.string();
        KENSHIN_INFO("[builder] gltf albedo {}x{} -> {}", albedo.width,
                     albedo.height, bmp.string());
      }
      recenterGltf(mesh, fit);
    } else if (!kenshin::assets::loadObj(input, mesh, log)) {
      KENSHIN_ERROR("load failed: {}", log);
      kenshin::log::shutdown();
      return 1;
    }
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
  if (debugDump) {
    // Per-level winding/size census: cross((v1-v0),(v2-v0)) sign tells
    // whether each triangle's geometric winding faces +Z or -Z; a flat
    // z=0 plane must be 100% one sign per consistently-wound input.
    const int maxLevel = 64;
    std::vector<kenshin::u32> cl(maxLevel, 0), tr(maxLevel, 0), up(maxLevel, 0),
        dn(maxLevel, 0), dg(maxLevel, 0);
    std::vector<double> maxEdge(maxLevel, 0.0), sumEdge(maxLevel, 0.0);
    for (kenshin::u32 ci = 0; ci < dag.clusters.size(); ++ci) {
      const auto& c = dag.clusters[ci];
      const int lvl = std::clamp(dag.groups[c.group].lodLevel, 0, maxLevel - 1);
      ++cl[lvl];
      for (kenshin::u32 t = 0; t + 2 < c.indexCount; t += 3) {
        const kenshin::u32 i0 = dag.indices[c.indexOffset + t],
                           i1 = dag.indices[c.indexOffset + t + 1],
                           i2 = dag.indices[c.indexOffset + t + 2];
        // indices are cluster-local; add the cluster's vertex slice base
        const glm::vec3 p0(dag.positions[size_t(c.vertexOffset + i0) * 3 + 0],
                           dag.positions[size_t(c.vertexOffset + i0) * 3 + 1],
                           dag.positions[size_t(c.vertexOffset + i0) * 3 + 2]);
        const glm::vec3 p1(dag.positions[size_t(c.vertexOffset + i1) * 3 + 0],
                           dag.positions[size_t(c.vertexOffset + i1) * 3 + 1],
                           dag.positions[size_t(c.vertexOffset + i1) * 3 + 2]);
        const glm::vec3 p2(dag.positions[size_t(c.vertexOffset + i2) * 3 + 0],
                           dag.positions[size_t(c.vertexOffset + i2) * 3 + 1],
                           dag.positions[size_t(c.vertexOffset + i2) * 3 + 2]);
        const glm::vec3 n = glm::cross(p1 - p0, p2 - p0);
        const float area2 = glm::length(n);
        ++tr[lvl];
        if (area2 < 1e-9f)
          ++dg[lvl];
        else if (n.z > 0)
          ++up[lvl];
        else
          ++dn[lvl];
        const float e01 = glm::length(p1 - p0), e12 = glm::length(p2 - p1),
                    e20 = glm::length(p0 - p2);
        const float em = std::max(e01, std::max(e12, e20));
        maxEdge[lvl] = std::max(maxEdge[lvl], double(em));
        sumEdge[lvl] += (e01 + e12 + e20) / 3.0;
      }
    }
    KENSHIN_INFO(
        "[dbg] level clusters tris upPct dnPct degPct maxEdge avgEdge");
    for (int l = 0; l < maxLevel; ++l)
      if (tr[l] > 0)
        KENSHIN_INFO(
            "[dbg] {:>5} {:>8} {:>7} {:>5.1f} {:>5.1f} {:>5.1f} "
            "{:>9.2f} {:>8.3f}",
            l, cl[l], tr[l], 100.0 * up[l] / tr[l], 100.0 * dn[l] / tr[l],
            100.0 * dg[l] / tr[l], maxEdge[l], sumEdge[l] / tr[l]);
  }
  // M8.2: the single-material table (v1). OBJ/MTL provides the
  // defaults; the CLI overrides win. The albedo reference is stored as
  // a bare FILE NAME - the runtime resolves it against the .kgeo's
  // directory (the mtllib precedent) and the builder copies the file
  // next to the output so the pair travels together.
  std::string albedoSrc;  // the path to copy FROM (CWD/MTL-relative)
  {
    kenshin::geometry::ClusterDag::DagMaterial mat;
    if (mesh.material.kd[0] >= 0.0f) {
      mat.kd[0] = mesh.material.kd[0];
      mat.kd[1] = mesh.material.kd[1];
      mat.kd[2] = mesh.material.kd[2];
    }
    if (kd[0] >= 0.0f) {
      mat.kd[0] = kd[0];
      mat.kd[1] = kd[1];
      mat.kd[2] = kd[2];
    }
    if (rough >= 0.0f)
      mat.roughness = rough;
    else if (mesh.material.roughness != 0.5f)
      mat.roughness = mesh.material.roughness;
    if (metal >= 0.0f)
      mat.metallic = metal;
    else if (mesh.material.metallic != 0.0f)
      mat.metallic = mesh.material.metallic;
    if (!albedo.empty())
      albedoSrc = albedo;
    else if (!mesh.material.diffuseTexture.empty())
      albedoSrc = mesh.material.diffuseTexture;
    if (!albedoSrc.empty())
      mat.texPath = std::filesystem::path(albedoSrc).filename().string();
    if (gltfMaterials.size() > 1) {
      // Multi-material table: one DagMaterial per used glTF material;
      // per-cluster ids came from the DAG's vertex-majority vote (do
      // NOT overwrite them with zeros). Explicit CLI overrides apply
      // to every entry (they stay global knobs). Each textured
      // material's albedo is re-encoded as its own BMP next to the
      // output - same travels-together contract as the single case.
      const fsg::path outDir = fsg::path(output).parent_path();
      std::vector<std::string> writtenFor(gltfImages.size());
      dag.materials.clear();
      for (const auto& gm : gltfMaterials) {
        kenshin::geometry::ClusterDag::DagMaterial dm = mat;
        dm.kd[0] = gm.kd[0];
        dm.kd[1] = gm.kd[1];
        dm.kd[2] = gm.kd[2];
        dm.roughness = gm.roughness;
        dm.metallic = gm.metallic;
        if (kd[0] >= 0.0f) {
          dm.kd[0] = kd[0];
          dm.kd[1] = kd[1];
          dm.kd[2] = kd[2];
        }
        if (rough >= 0.0f) dm.roughness = rough;
        if (metal >= 0.0f) dm.metallic = metal;
        if (gm.image >= 0 && gm.image < int(gltfImages.size())) {
          const auto& img = gltfImages[kenshin::usize(gm.image)];
          if (img.width > 0) {
            std::string& file = writtenFor[kenshin::usize(gm.image)];
            if (file.empty()) {
              const std::string stem =
                  img.name.empty()
                      ? "mat" + std::to_string(dag.materials.size())
                      : img.name;
              const fsg::path bmp = outDir / (stem + "_albedo.bmp");
              if (!kenshin::util::writeBmp24(bmp.string(), img.rgba,
                                             kenshin::u32(img.width),
                                             kenshin::u32(img.height))) {
                KENSHIN_ERROR("[builder] albedo re-encode failed: {}",
                              bmp.string());
                kenshin::log::shutdown();
                return 1;
              }
              file = bmp.filename().string();
              KENSHIN_INFO("[builder] gltf material {} albedo {}x{} -> {}",
                           dag.materials.size(), img.width, img.height, file);
            }
            dm.texPath = file;
          }
        }
        dag.materials.push_back(dm);
      }
    } else {
      dag.materials = {mat};
    }
    if (dag.clusterMaterial.size() != dag.clusters.size())
      dag.clusterMaterial.assign(dag.clusters.size(), 0);
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
                                                     size_t(p * spans.size()))];
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
        const f32* p = dag.positions.data() + size_t(cl.vertexOffset + v) * 3;
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
            mortonOf(dag.positions.data() + size_t(cl.vertexOffset + v) * 3);
        remap[v] = v;
      }
      std::sort(remap.begin(), remap.end(),
                [&](u32 a, u32 b) { return morton[a] < morton[b]; });
      for (u32 i = 0; i < n; ++i) inv[remap[i]] = i;

      // (c) Triangle Morton order (by centroid), THEN first-use vertex
      // numbering - renumbers connectivity along a spatial walk.
      std::vector<u32> triOrder(triN), tMorton(triN), invT(n);
      for (u32 t = 0; t < triN; ++t) {
        const u32 i0 = dag.indices[size_t(cl.indexOffset) + t * 3 + 0];
        const u32 i1 = dag.indices[size_t(cl.indexOffset) + t * 3 + 1];
        const u32 i2 = dag.indices[size_t(cl.indexOffset) + t * 3 + 2];
        glm::vec3 c(0.0f);
        for (u32 i : {i0, i1, i2}) {
          const f32* p = dag.positions.data() + size_t(cl.vertexOffset + i) * 3;
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
            const u32 i = dag.indices[size_t(cl.indexOffset) + t * 3 + k];
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
        const u32 i0 = dag.indices[size_t(cl.indexOffset) + t + 0];
        const u32 i1 = dag.indices[size_t(cl.indexOffset) + t + 1];
        const u32 i2 = dag.indices[size_t(cl.indexOffset) + t + 2];
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
  if (imposter) {
    // The imposter cut = the DEEPEST level's clusters (the coarsest
    // COVERING set; dag.roots can mix terminal fragments from finer
    // levels that overlap the coarse cut and blow the 7-bit budget).
    kenshin::u32 maxLevel = 0;
    for (const auto& g : dag.groups)
      maxLevel = std::max(maxLevel, kenshin::u32(g.lodLevel));
    std::vector<kenshin::u32> cut;
    for (const auto& g : dag.groups)
      if (kenshin::u32(g.lodLevel) == maxLevel)
        for (kenshin::u32 c : g.clusters) cut.push_back(c);
    kenshin::geometry::ImposterAtlas atlas;
    std::string ilog;
    if (cut.size() != 1u) {
      KENSHIN_INFO(
          "[builder] imposter cut has {} clusters (v1 needs exactly "
          "1) - skipping the atlas",
          cut.size());
      kenshin::log::shutdown();
      return 1;
    }
    dag.imposterCluster = cut[0];  // serialized - the runtime's truth
    if (!kenshin::geometry::bakeImposterAtlas(dag, cut, atlas, ilog)) {
      KENSHIN_ERROR("[builder] imposter bake failed: {}", ilog);
      kenshin::log::shutdown();
      return 1;
    }
    dag.imposterDepth = std::move(atlas.depth);
    dag.imposterTri = std::move(atlas.tri);
    KENSHIN_INFO(
        "[builder] imposter baked: cut cluster {} (of {} "
        "deepest-level clusters, {} roots)",
        dag.imposterCluster, cut.size(), dag.roots.size());
  }
  kenshin::geometry::PackedGeometry packed;
  if (!kenshin::geometry::packPages(dag, packed, log)) {
    KENSHIN_ERROR("page packing failed: {}", log);
    kenshin::log::shutdown();
    return 1;
  }
  if (!kenshin::geometry::writeKGeoPaged(output, dag, packed, log,
                                         !noCompress)) {
    KENSHIN_ERROR("write failed: {}", log);
    kenshin::log::shutdown();
    return 1;
  }
  {
    std::error_code ec;
    const auto raw = std::filesystem::file_size(fsg::path(output), ec);
    if (!ec)
      KENSHIN_INFO("[builder] {} ({}; on-disk {})", output,
                   imposter ? "v9 imposter+pages"
                            : (noCompress ? "v7 raw pages" : "v8 codec pages"),
                   raw);
  }
  // The albedo travels WITH the asset (bare-name reference; the
  // runtime resolves against the .kgeo's directory).
  if (!albedoSrc.empty()) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path src = fs::path(albedoSrc);
    if (!fs::exists(src, ec)) {
      KENSHIN_ERROR("[builder] albedo not found: {} (relative to CWD)",
                    albedoSrc);
      kenshin::log::shutdown();
      return 1;
    }
    const fs::path dst = fs::path(output).parent_path() / src.filename();
    // The glTF path may have written the BMP straight into the output
    // dir already - copying onto itself fails on Windows.
    if (fs::weakly_canonical(src, ec) != fs::weakly_canonical(dst, ec)) {
      if (!fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec)) {
        KENSHIN_ERROR("[builder] albedo copy failed: {} -> {} ({})",
                      src.string(), dst.string(), ec.message());
        kenshin::log::shutdown();
        return 1;
      }
      KENSHIN_INFO("[builder] albedo: {} -> {}", src.string(), dst.string());
    }
  }
  KENSHIN_INFO("[builder] written (v7 paged, materials: {}): {}",
               dag.materials.size(), output);
  kenshin::log::shutdown();
  return 0;
}
