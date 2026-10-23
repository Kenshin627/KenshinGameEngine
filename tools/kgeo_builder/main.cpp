// kgeo_builder CLI: input.obj | input.gltf/.glb | --proc <spec> -> output.kgeo
//
// A thin front end over the engine's importer library (roadmap 1.3b):
// load -> condition (glTF) -> cluster DAG -> material table -> .kgeo.
// The asset database cooks through the same functions.

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <glm/glm.hpp>
#include <string>
#include <vector>

#include "kenshin/Version.h"
#include "kenshin/assets/GltfLoader.h"
#include "kenshin/assets/MeshData.h"
#include "kenshin/assets/ObjLoader.h"
#include "kenshin/core/Log.h"
#include "kenshin/geometry/ClusterDag.h"
#include "kenshin/geometry/shapes/SphereGeometry.h"
#include "kenshin/geometry/shapes/TorusKnotGeometry.h"
#include "kenshin/importer/KgeoMaterials.h"
#include "kenshin/importer/MeshConditioning.h"
#include "kenshin/importer/MeshCooker.h"

namespace fs = std::filesystem;
using kenshin::f32;
using kenshin::u32;
using kenshin::usize;

namespace {

int fail(const std::string& message) {
  KENSHIN_ERROR("[builder] {}", message);
  kenshin::log::shutdown();
  return 1;
}

/// --debug-dump: per-level winding/size census. cross((v1-v0),(v2-v0))
/// tells whether each triangle faces +Z or -Z; a flat z=0 plane must be
/// 100% one sign per consistently-wound input.
void debugDump(const kenshin::geometry::ClusterDag& dag) {
  const int maxLevel = 64;
  std::vector<u32> cl(maxLevel, 0), tr(maxLevel, 0), up(maxLevel, 0), dn(maxLevel, 0),
      dg(maxLevel, 0);
  std::vector<double> maxEdge(maxLevel, 0.0), sumEdge(maxLevel, 0.0);
  for (u32 ci = 0; ci < dag.clusters.size(); ++ci) {
    const auto& c = dag.clusters[ci];
    const int lvl = std::clamp(dag.groups[c.group].lodLevel, 0, maxLevel - 1);
    ++cl[lvl];
    const auto P = [&](u32 i) {
      const usize v = usize(c.vertexOffset + i) * 3;  // cluster-local index
      return glm::vec3(dag.positions[v], dag.positions[v + 1], dag.positions[v + 2]);
    };
    for (u32 t = 0; t + 2 < c.indexCount; t += 3) {
      const glm::vec3 p0 = P(dag.indices[c.indexOffset + t]),
                      p1 = P(dag.indices[c.indexOffset + t + 1]),
                      p2 = P(dag.indices[c.indexOffset + t + 2]);
      const glm::vec3 n = glm::cross(p1 - p0, p2 - p0);
      ++tr[lvl];
      if (glm::length(n) < 1e-9f)
        ++dg[lvl];
      else if (n.z > 0)
        ++up[lvl];
      else
        ++dn[lvl];
      const f32 e01 = glm::length(p1 - p0), e12 = glm::length(p2 - p1),
                e20 = glm::length(p0 - p2);
      maxEdge[lvl] = std::max(maxEdge[lvl], double(std::max({e01, e12, e20})));
      sumEdge[lvl] += (e01 + e12 + e20) / 3.0;
    }
  }
  KENSHIN_INFO("[dbg] level clusters tris upPct dnPct degPct maxEdge avgEdge");
  for (int l = 0; l < maxLevel; ++l)
    if (tr[l] > 0)
      KENSHIN_INFO("[dbg] {:>5} {:>8} {:>7} {:>5.1f} {:>5.1f} {:>5.1f} {:>9.2f} {:>8.3f}", l,
                   cl[l], tr[l], 100.0 * up[l] / tr[l], 100.0 * dn[l] / tr[l],
                   100.0 * dg[l] / tr[l], maxEdge[l], sumEdge[l] / tr[l]);
}

}  // namespace

int main(int argc, char** argv) {
  kenshin::log::init();
  KENSHIN_INFO("{}", kenshin::getBanner());

  std::string input, output = "out/mesh.kgeo";
  std::string proc;           // "sphere" or "knot"[:segsA[:segsB]]
  bool opaqueBlend = false;   // --opaque-blend: keep BLEND materials in the opaque DAG
  bool doubleSided = true;    // --no-double-sided: skip the back-face copies
  f32 fit = 0.0f;             // --fit N: longest bbox side -> N units
  bool debugDumpOn = false;   // --debug-dump: per-level DAG stats
  kenshin::importer::CookOptions cook;
  kenshin::importer::MaterialOverrides overrides;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc)
      output = argv[++i];
    else if (std::strcmp(argv[i], "--unpaged") == 0)
      cook.paged = false;
    else if (std::strcmp(argv[i], "--proc") == 0 && i + 1 < argc)
      proc = argv[++i];
    else if (std::strcmp(argv[i], "--albedo") == 0 && i + 1 < argc)
      overrides.albedo = argv[++i];
    else if (std::strcmp(argv[i], "--kd") == 0 && i + 3 < argc)
      for (int c = 0; c < 3; ++c) overrides.kd[c] = static_cast<f32>(std::atof(argv[++i]));
    else if (std::strcmp(argv[i], "--rough") == 0 && i + 1 < argc)
      overrides.roughness = static_cast<f32>(std::atof(argv[++i]));
    else if (std::strcmp(argv[i], "--metal") == 0 && i + 1 < argc)
      overrides.metallic = static_cast<f32>(std::atof(argv[++i]));
    else if (std::strcmp(argv[i], "--opaque-blend") == 0)
      opaqueBlend = true;
    else if (std::strcmp(argv[i], "--no-double-sided") == 0)
      doubleSided = false;
    else if (std::strcmp(argv[i], "--fit") == 0 && i + 1 < argc)
      fit = static_cast<f32>(std::atof(argv[++i]));
    else if (std::strcmp(argv[i], "--debug-dump") == 0)
      debugDumpOn = true;
    else if (std::strcmp(argv[i], "--no-compress") == 0)
      cook.compressPages = false;
    else if (std::strcmp(argv[i], "--imposter") == 0)
      cook.imposter = true;
    else
      input = argv[i];
  }
  if (input.empty() && proc.empty()) {
    KENSHIN_ERROR("usage: kgeo_builder input.obj|input.gltf|input.glb [-o out.kgeo]");
    KENSHIN_ERROR("       kgeo_builder --proc knot|sphere[:segsA[:segsB]] [-o out]");
    kenshin::log::shutdown();
    return 1;
  }
  const fs::path outPath(output);
  const fs::path outDir = outPath.parent_path();

  kenshin::assets::MeshData mesh;
  std::vector<kenshin::assets::GltfMaterial> gltfMaterials;
  std::vector<kenshin::assets::GltfImage> gltfImages;
  std::string log;
  if (!proc.empty()) {
    // --proc name[:segsA[:segsB]] - segment overrides (coarse meshes for
    // the large-triangle HW routing tests).
    std::string name = proc;
    int segA = 0, segB = 0;
    if (const auto colon = proc.find(':'); colon != std::string::npos) {
      name = proc.substr(0, colon);
      segA = std::atoi(proc.c_str() + colon + 1);
      if (const auto colon2 = proc.find(':', colon + 1); colon2 != std::string::npos)
        segB = std::atoi(proc.c_str() + colon2 + 1);
    }
    if (name == "knot")
      mesh = kenshin::geometry::TorusKnotGeometry::create(2, 3, segA > 0 ? segA : 512,
                                                          segB > 0 ? segB : 1024, 1.0f);
    else if (name == "sphere")
      mesh = kenshin::geometry::SphereGeometry::create(segA > 0 ? segA : 256,
                                                       segB > 0 ? segB : 128, 1.0f);
    else
      return fail("unknown --proc '" + proc + "'");
  } else {
    std::string ext = fs::path(input).extension().string();
    for (char& c : ext) c = char(tolower(unsigned(c)));
    if (ext == ".glb" || ext == ".gltf") {
      kenshin::assets::GltfImage gltfAlbedo;
      if (!kenshin::assets::loadGltf(input, mesh, &gltfAlbedo, log, &gltfMaterials, &gltfImages))
        return fail("glTF load failed: " + log);
      if (!log.empty()) KENSHIN_WARN("[builder] gltf notes: {}", log);
      if (gltfAlbedo.width > 0 && gltfMaterials.size() <= 1 &&
          !kenshin::importer::writeSingleAlbedo(gltfAlbedo, fs::path(input).stem().string(),
                                                outDir, mesh, log))
        return fail(log);
      kenshin::importer::recenter(mesh, fit);
      if (!opaqueBlend) {
        kenshin::importer::TranslucentMesh translucent;
        if (kenshin::importer::splitTranslucent(mesh, gltfMaterials, translucent) &&
            !kenshin::importer::writeTranslucentMesh(fs::path(outPath).replace_extension(".tmesh"),
                                                     gltfMaterials, translucent, log))
          return fail(log);
      }
      kenshin::importer::sinkHuggingBlend(mesh, gltfMaterials, fit);
      kenshin::importer::separateLayers(mesh, fit);
      if (doubleSided) kenshin::importer::appendBackFaces(mesh, gltfMaterials, fit);
    } else if (!kenshin::assets::loadObj(input, mesh, log)) {
      return fail("load failed: " + log);
    }
  }
  KENSHIN_INFO("[builder] mesh: {} verts / {} tris", mesh.vertexCount(), mesh.triangleCount());

  kenshin::geometry::ClusterDag dag;
  kenshin::geometry::DagBuildStats dagStats;
  if (!kenshin::importer::buildMeshDag(mesh, dag, log, &dagStats)) return fail(log);
  if (debugDumpOn) debugDump(dag);

  std::string albedoSource;
  if (!kenshin::importer::buildKgeoMaterials(mesh, overrides, gltfMaterials, gltfImages, outDir,
                                             dag, albedoSource, log))
    return fail(log);
  if (!log.empty()) KENSHIN_WARN("[builder] {}", log);
  kenshin::importer::logDagCensus(dag, dagStats);

  log.clear();
  if (!kenshin::importer::writeCookedMesh(outPath, dag, cook, log)) return fail(log);
  // The albedo travels WITH the asset (bare-name reference; the runtime
  // resolves against the .kgeo's directory). v2 files never carried it.
  if (cook.paged && !kenshin::importer::copyAlbedoBesideOutput(albedoSource, outPath, log))
    return fail(log);
  kenshin::log::shutdown();
  return 0;
}
