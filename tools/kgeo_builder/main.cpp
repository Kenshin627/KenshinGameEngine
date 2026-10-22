// kgeo_builder CLI (S3.3): input.obj | --proc <spec> -> output.kgeo

#include <algorithm>
#include <bit>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <glm/glm.hpp>
#include <map>
#include <string>
#include <tuple>
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

/// Squared distance from p to triangle abc (Ericson, Real-Time
/// Collision Detection 5.1.5 - closest point by Voronoi region).
float pointTriDist2(const glm::vec3& p, const glm::vec3& a, const glm::vec3& b,
                    const glm::vec3& c) {
  const glm::vec3 ab = b - a, ac = c - a, ap = p - a;
  const float d1 = glm::dot(ab, ap), d2 = glm::dot(ac, ap);
  if (d1 <= 0.0f && d2 <= 0.0f) return glm::dot(ap, ap);
  const glm::vec3 bp = p - b;
  const float d3 = glm::dot(ab, bp), d4 = glm::dot(ac, bp);
  if (d3 >= 0.0f && d4 <= d3) return glm::dot(bp, bp);
  const float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
    const glm::vec3 q = a + ab * (d1 / (d1 - d3));
    return glm::dot(p - q, p - q);
  }
  const glm::vec3 cp = p - c;
  const float d5 = glm::dot(ab, cp), d6 = glm::dot(ac, cp);
  if (d6 >= 0.0f && d5 <= d6) return glm::dot(cp, cp);
  const float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
    const glm::vec3 q = a + ac * (d2 / (d2 - d6));
    return glm::dot(p - q, p - q);
  }
  const float va = d3 * d6 - d5 * d4;
  if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
    const glm::vec3 q = b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    return glm::dot(p - q, p - q);
  }
  const float denom = 1.0f / (va + vb + vc);
  const glm::vec3 q = a + ab * (vb * denom) + ac * (vc * denom);
  return glm::dot(p - q, p - q);
}

int main(int argc, char** argv) {
  kenshin::log::init();
  KENSHIN_INFO("{}", kenshin::getBanner());

  std::string input, output = "out/mesh.kgeo";
  bool opaqueBlend = false;  // --opaque-blend: keep BLEND materials in
                             // the opaque DAG (the pre-translucency
                             // path, with the hug sink) - A/B
  bool doubleSided = true;   // --no-double-sided: skip the glTF
                             // doubleSided back-face copies (A/B)
  float fit = 0.0f;          // --fit N: uniform scale so the longest bbox
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
    else if (std::strcmp(argv[i], "--opaque-blend") == 0)
      opaqueBlend = true;
    else if (std::strcmp(argv[i], "--no-double-sided") == 0)
      doubleSided = false;
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
      // Translucency (UE's split: Nanite never rasterizes translucent
      // materials - they render in a forward pass over plain meshes).
      // BLEND triangles leave the opaque DAG and go to <out>.tmesh:
      // "KTM1" | u32 version 2 | u32 materialCount | per material 8 f32
      // (kd rgb, alpha, roughness, metallic, 0, 0) | u32 vertexCount |
      // per vertex 12 f32 (pos xyz, material id, normal xyz, 0, uv, 0,
      // 0) - a NON-indexed triangle list the pass vertex-pulls. Material
      // ids index the SAME table as the kgeo's material section, so the
      // runtime samples the scene's texture layers for them.
      if (!opaqueBlend && !mesh.triMaterial.empty()) {
        using kenshin::u32;
        using kenshin::usize;
        std::vector<float> tverts;
        std::vector<u32> keepIdx, keepMat;
        keepIdx.reserve(mesh.indices.size());
        keepMat.reserve(mesh.triMaterial.size());
        const bool hasNrm = mesh.normals.size() == mesh.positions.size();
        const bool hasUv = mesh.uvs.size() == mesh.vertexCount() * 2;
        for (usize t = 0; t < mesh.triMaterial.size(); ++t) {
          const u32 m = mesh.triMaterial[t];
          const bool blend =
              m < gltfMaterials.size() && gltfMaterials[m].alpha < 0.999f;
          if (!blend) {
            for (int k = 0; k < 3; ++k)
              keepIdx.push_back(mesh.indices[t * 3 + k]);
            keepMat.push_back(m);
            continue;
          }
          for (int k = 0; k < 3; ++k) {
            const u32 v = mesh.indices[t * 3 + k];
            for (int c = 0; c < 3; ++c)
              tverts.push_back(mesh.positions[usize(v) * 3 + c]);
            tverts.push_back(float(m));
            for (int c = 0; c < 3; ++c)
              tverts.push_back(hasNrm ? mesh.normals[usize(v) * 3 + c]
                                      : (c == 2 ? 1.0f : 0.0f));
            tverts.push_back(0.0f);
            tverts.push_back(hasUv ? mesh.uvs[usize(v) * 2] : 0.0f);
            tverts.push_back(hasUv ? mesh.uvs[usize(v) * 2 + 1] : 0.0f);
            tverts.push_back(0.0f);
            tverts.push_back(0.0f);
          }
        }
        if (!tverts.empty()) {
          const std::filesystem::path tpath =
              std::filesystem::path(output).replace_extension(".tmesh");
          FILE* tf = std::fopen(tpath.string().c_str(), "wb");
          bool ok = tf != nullptr;
          if (ok) {
            const u32 head[3] = {0x314D544Bu /* "KTM1" */, 2u,
                                 static_cast<u32>(gltfMaterials.size())};
            ok = std::fwrite(head, sizeof(head), 1, tf) == 1;
            for (const auto& gm : gltfMaterials) {
              const float rec[8] = {gm.kd.r,  gm.kd.g,      gm.kd.b,
                                    gm.alpha, gm.roughness, gm.metallic,
                                    0.0f,     0.0f};
              ok = ok && std::fwrite(rec, sizeof(rec), 1, tf) == 1;
            }
            const u32 vcount = static_cast<u32>(tverts.size() / 12);
            ok = ok && std::fwrite(&vcount, sizeof(vcount), 1, tf) == 1 &&
                 std::fwrite(tverts.data(), sizeof(float), tverts.size(), tf) ==
                     tverts.size();
            std::fclose(tf);
          }
          if (!ok) {
            KENSHIN_ERROR("[builder] translucent mesh write failed: {}",
                          tpath.string());
            kenshin::log::shutdown();
            return 1;
          }
          KENSHIN_INFO("[builder] translucent: {} BLEND tris -> {}",
                       tverts.size() / 36, tpath.string());
          mesh.indices.swap(keepIdx);
          mesh.triMaterial.swap(keepMat);
        }
      }
      // BLEND-material sink: translucent plates that hug an opaque
      // surface (this asset's visor plate rides ~1e-3 behind the shell
      // face) lose that hair-thin depth order to LOD simplification
      // error at runtime - simplified shell wobble flips the winner
      // per cluster and paints opaque blocks over the camo. Push the
      // BLEND geometry inward along its normals by a margin larger
      // than any LOD error (1.5 per-mille of the fit box) so the opaque
      // surface wins every near-tie; plates that genuinely protrude
      // (the visor's wrap rim) stay visible - the offset is dwarfed by
      // their real clearance.
      {
        using kenshin::u32;
        using kenshin::usize;
        bool anyBlend = false;
        for (const auto& gm : gltfMaterials) anyBlend |= gm.alpha < 0.999f;
        if (anyBlend && !mesh.triMaterial.empty() &&
            mesh.normals.size() == mesh.positions.size()) {
          const float kSink = fit > 0.0f ? fit * 0.0015f : 0.003f;
          std::vector<bool> sink(mesh.vertexCount(), false);
          for (usize t = 0; t < mesh.triMaterial.size(); ++t) {
            const u32 m = mesh.triMaterial[t];
            if (m >= gltfMaterials.size() || gltfMaterials[m].alpha >= 0.999f)
              continue;
            for (int k = 0; k < 3; ++k) sink[mesh.indices[t * 3 + k]] = true;
          }
          // Only materials that HUG opaque geometry sink. Free-standing
          // glass (the F-35 visor stands well off its shell) sinking
          // 1.5 per-mille behind its own front drove the visor rim
          // under the shell's mounting parts - a ring of dark triangles
          // through the glass. Decided PER MATERIAL (>= 25% of its
          // vertices within kHug of an opaque triangle): a per-vertex
          // sink bends a plate at the hug boundary and lets the geometry
          // under it poke through in cluster-sized steps. Uniform grid of
          // opaque triangles, each inserted into every cell its
          // kHug-expanded bounds touch, so a vertex checks its own cell
          // only.
          const float kHug = fit > 0.0f ? fit * 0.005f : 0.01f;
          const float cell = kHug * 4.0f;
          std::map<std::tuple<int, int, int>, std::vector<u32>> grid;
          const auto P = [&](u32 v) {
            return glm::vec3(mesh.positions[usize(v) * 3],
                             mesh.positions[usize(v) * 3 + 1],
                             mesh.positions[usize(v) * 3 + 2]);
          };
          const auto key = [&](float x) {
            return static_cast<int>(std::floor(x / cell));
          };
          for (usize t = 0; t < mesh.triMaterial.size(); ++t) {
            const u32 m = mesh.triMaterial[t];
            if (m < gltfMaterials.size() && gltfMaterials[m].alpha < 0.999f)
              continue;
            const glm::vec3 a = P(mesh.indices[t * 3]),
                            b = P(mesh.indices[t * 3 + 1]),
                            c = P(mesh.indices[t * 3 + 2]);
            const glm::vec3 lo = glm::min(a, glm::min(b, c)) - kHug;
            const glm::vec3 hi = glm::max(a, glm::max(b, c)) + kHug;
            const int x0 = key(lo.x), x1 = key(hi.x), y0 = key(lo.y),
                      y1 = key(hi.y), z0 = key(lo.z), z1 = key(hi.z);
            if ((x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1) > 4096)
              continue;  // a huge opaque triangle cannot be "hugged" here
            for (int x = x0; x <= x1; ++x)
              for (int y = y0; y <= y1; ++y)
                for (int z = z0; z <= z1; ++z)
                  grid[{x, y, z}].push_back(static_cast<u32>(t));
          }
          // Per-material vote: a vertex counts toward every BLEND
          // material that references it.
          std::vector<u32> vertMat(mesh.vertexCount(), 0xFFFFFFFFu);
          for (usize t = 0; t < mesh.triMaterial.size(); ++t) {
            const u32 m = mesh.triMaterial[t];
            if (m >= gltfMaterials.size() || gltfMaterials[m].alpha >= 0.999f)
              continue;
            for (int k = 0; k < 3; ++k) vertMat[mesh.indices[t * 3 + k]] = m;
          }
          std::vector<usize> hugCount(gltfMaterials.size(), 0),
              blendCount(gltfMaterials.size(), 0);
          for (usize v = 0; v < sink.size(); ++v) {
            if (!sink[v] || vertMat[v] >= gltfMaterials.size()) continue;
            ++blendCount[vertMat[v]];
            const glm::vec3 p = P(static_cast<u32>(v));
            const auto it = grid.find({key(p.x), key(p.y), key(p.z)});
            if (it == grid.end()) continue;
            for (const u32 t : it->second)
              if (pointTriDist2(p, P(mesh.indices[t * 3]),
                                P(mesh.indices[t * 3 + 1]),
                                P(mesh.indices[t * 3 + 2])) <= kHug * kHug) {
                ++hugCount[vertMat[v]];
                break;
              }
          }
          for (usize m = 0; m < gltfMaterials.size(); ++m) {
            if (blendCount[m] == 0) continue;
            const bool hugs = hugCount[m] * 4 >= blendCount[m];
            KENSHIN_INFO(
                "[builder] BLEND material {}: {}/{} verts hug opaque "
                "geometry -> {}",
                m, hugCount[m], blendCount[m],
                hugs ? "sink" : "free-standing, kept in place");
            if (hugs) continue;
            for (usize v = 0; v < sink.size(); ++v)
              if (vertMat[v] == m) sink[v] = false;
          }
          usize sunk = 0;
          for (usize v = 0; v < sink.size(); ++v) {
            if (!sink[v]) continue;
            for (int k = 0; k < 3; ++k)
              mesh.positions[v * 3 + k] -= mesh.normals[v * 3 + k] * kSink;
            ++sunk;
          }
          if (sunk)
            KENSHIN_INFO("[builder] BLEND sink: {} verts -{} units (normal)",
                         sunk, kSink);
        }
      }
      // glTF doubleSided: every raster path (SW edge sign, HW cull
      // BACK) drops back faces, so single-sheet double-sided geometry
      // (visors, liners, cloth) shows holes wherever it curves away
      // from the camera. Build-time answer, zero runtime cost beyond
      // the triangles: append a reversed-winding copy on its OWN
      // vertices with flipped normals. A copy always faces away from
      // wherever its original is visible, so the pair never competes
      // for a pixel - even after independent LOD simplification.
      // The copy is pushed a hair along ITS normal (behind the
      // original's front side): bit-identical positions get welded by
      // clusterlod's position remap, every edge turns non-manifold
      // (two opposite triangles) and the simplifier locks it - the
      // first try left 5577 of 5742 clusters unsimplified LOD-0 roots.
      if (doubleSided && !mesh.triMaterial.empty()) {
        const float kSplit = fit > 0.0f ? fit * 1e-4f : 2e-4f;
        using kenshin::u32;
        using kenshin::usize;
        const usize triCount = mesh.triMaterial.size();
        const bool hasNrm = mesh.normals.size() == mesh.positions.size();
        const bool hasUv = mesh.uvs.size() == mesh.vertexCount() * 2;
        std::vector<u32> remap(mesh.vertexCount(), 0xFFFFFFFFu);
        usize added = 0;
        for (usize t = 0; t < triCount; ++t) {
          const u32 m = mesh.triMaterial[t];
          if (m >= gltfMaterials.size() || !gltfMaterials[m].doubleSided)
            continue;
          u32 tri[3];
          for (int k = 0; k < 3; ++k) {
            const u32 v = mesh.indices[t * 3 + k];
            if (remap[v] == 0xFFFFFFFFu) {
              remap[v] = static_cast<u32>(mesh.vertexCount());
              for (int c = 0; c < 3; ++c)
                mesh.positions.push_back(
                    mesh.positions[usize(v) * 3 + c] -
                    (hasNrm ? mesh.normals[usize(v) * 3 + c] * kSplit : 0.0f));
              if (hasNrm)
                for (int c = 0; c < 3; ++c)
                  mesh.normals.push_back(-mesh.normals[usize(v) * 3 + c]);
              if (hasUv)
                for (int c = 0; c < 2; ++c)
                  mesh.uvs.push_back(mesh.uvs[usize(v) * 2 + c]);
            }
            tri[k] = remap[v];
          }
          mesh.indices.push_back(tri[0]);  // reversed winding
          mesh.indices.push_back(tri[2]);
          mesh.indices.push_back(tri[1]);
          mesh.triMaterial.push_back(m);
          ++added;
        }
        if (added)
          KENSHIN_INFO("[builder] doubleSided: +{} back-face triangles", added);
      }
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
      int baseW = 0, baseH = 0;
      auto resampleTo = [](const kenshin::assets::GltfImage& im, int w, int h) {
        kenshin::assets::GltfImage out;
        out.name = im.name;
        out.width = w;
        out.height = h;
        out.rgba.assign(kenshin::usize(w) * h * 4, 0);
        for (int y = 0; y < h; ++y) {
          const int sy0 = int(kenshin::u64(y) * im.height / h);
          const int sy1 =
              std::max(int(kenshin::u64(y + 1) * im.height / h), sy0 + 1);
          for (int x = 0; x < w; ++x) {
            const int sx0 = int(kenshin::u64(x) * im.width / w);
            const int sx1 =
                std::max(int(kenshin::u64(x + 1) * im.width / w), sx0 + 1);
            kenshin::u32 r = 0, g = 0, b = 0, n = 0;
            for (int sy = sy0; sy < sy1 && sy < im.height; ++sy)
              for (int sx = sx0; sx < sx1 && sx < im.width; ++sx) {
                const kenshin::u8* p =
                    im.rgba.data() + (kenshin::usize(sy) * im.width + sx) * 4;
                r += p[0];
                g += p[1];
                b += p[2];
                ++n;
              }
            kenshin::u8* o = out.rgba.data() + (kenshin::usize(y) * w + x) * 4;
            o[0] = kenshin::u8(r / n);
            o[1] = kenshin::u8(g / n);
            o[2] = kenshin::u8(b / n);
            o[3] = 255;
          }
        }
        return out;
      };

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
        // Texture2DArray layers must share dimensions - the first
        // image sets the base; others box-resample onto it (fine for
        // both albedo and the linear MR data).
        auto imageFile = [&](int imageIdx, const char* suffix) -> std::string {
          if (imageIdx < 0 || imageIdx >= int(gltfImages.size())) return {};
          kenshin::assets::GltfImage imgStore;
          const kenshin::assets::GltfImage* imgPtr =
              &gltfImages[kenshin::usize(imageIdx)];
          if (baseW == 0) {
            baseW = imgPtr->width;
            baseH = imgPtr->height;
          } else if (imgPtr->width != baseW || imgPtr->height != baseH) {
            imgStore = resampleTo(*imgPtr, baseW, baseH);
            imgPtr = &imgStore;
          }
          const auto& img = *imgPtr;
          if (img.width <= 0) return {};
          std::string& file = writtenFor[kenshin::usize(imageIdx)];
          if (file.empty()) {
            const std::string stem =
                img.name.empty() ? "mat" + std::to_string(dag.materials.size())
                                 : img.name;
            const fsg::path bmp = outDir / (stem + suffix + ".bmp");
            if (!kenshin::util::writeBmp24(bmp.string(), img.rgba,
                                           kenshin::u32(img.width),
                                           kenshin::u32(img.height))) {
              KENSHIN_ERROR("[builder] texture re-encode failed: {}",
                            bmp.string());
              return {};
            }
            file = bmp.filename().string();
            KENSHIN_INFO("[builder] gltf material {} tex {}x{} -> {}",
                         dag.materials.size(), img.width, img.height, file);
          }
          return file;
        };
        const std::string albedoFile = imageFile(gm.image, "_albedo");
        if (!albedoFile.empty()) dm.texPath = albedoFile;
        const std::string mrFile = imageFile(gm.mrImage, "_mr");
        if (!mrFile.empty()) dm.mrPath = mrFile;
        const std::string nrmFile = imageFile(gm.nrmImage, "_nrm");
        if (!nrmFile.empty()) dm.nrmPath = nrmFile;
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

  // Per-material cluster census (diagnosing material-boundary
  // artifacts): counts + LOD histogram + the union bounds of each
  // material's clusters, plus how many triangles each material's
  // clusters actually carry (vertex-majority vs triangle reality).
  if (!dag.clusterMaterial.empty()) {
    using kenshin::u32;
    using kenshin::usize;
    // Per-LEVEL triangle census (which level is coarsest?) + roots per
    // level - settles level semantics from data.
    {
      std::map<u32, kenshin::u64> lvlTris, lvlClusters, lvlRoots;
      for (u32 c = 0; c < dag.clusters.size(); ++c) {
        const u32 lvl = dag.groups[dag.clusters[c].group].lodLevel;
        ++lvlClusters[lvl];
        lvlTris[lvl] += dag.clusters[c].indexCount / 3;
        if (dag.clusters[c].isRoot) ++lvlRoots[lvl];
      }
      for (const auto& [l, n] : lvlClusters)
        KENSHIN_INFO("[builder] level {}: {} clusters, {} tris, {} roots", l, n,
                     lvlTris[l], lvlRoots[l]);
    }
    std::map<u32, u32> perLevelAll;
    for (u32 m = 0; m < dag.materials.size(); ++m) {
      u32 cnt = 0, tris = 0;
      std::map<u32, u32> perLevel;
      glm::vec3 mn(FLT_MAX), mx(-FLT_MAX);
      for (u32 c = 0; c < dag.clusters.size(); ++c) {
        if (dag.clusterMaterial[c] != m) continue;
        ++cnt;
        tris += dag.clusters[c].indexCount / 3;
        ++perLevel[dag.groups[dag.clusters[c].group].lodLevel];
        for (usize i = 0; i < dag.clusters[c].indexCount; ++i) {
          const u32 li = dag.clusters[c].indexOffset + i;
          if (li >= dag.indices.size()) continue;
          const u32 vi = dag.clusters[c].vertexOffset + dag.indices[li];
          if (vi * 3 + 2 >= dag.positions.size()) continue;
          for (int k = 0; k < 3; ++k) {
            const float p = dag.positions[vi * 3 + k];
            mn[k] = std::min(mn[k], p);
            mx[k] = std::max(mx[k], p);
          }
        }
      }
      std::string lv;
      for (const auto& [l, n] : perLevel)
        lv += " L" + std::to_string(l) + "=" + std::to_string(n);
      if (cnt > 0)
        KENSHIN_INFO(
            "[builder] material {} '{}': {} clusters, {} tris, bounds "
            "({:.2f},{:.2f},{:.2f})..({:.2f},{:.2f},{:.2f}){}",
            m,
            dag.materials[m].texPath.empty() ? "?" : dag.materials[m].texPath,
            cnt, tris, mn.x, mn.y, mn.z, mx.x, mx.y, mx.z, lv);
    }
  }

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
