// kgeo_builder CLI (S3.3): input.obj | --proc <spec> -> output.kgeo

#include <cstdio>
#include <cstdlib>
#include <cstring>
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
  std::string proc;  // "sphere" or "knot"
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc)
      output = argv[++i];
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
      "{} monotonic fixes",
      dagStats.clusterCount, dagStats.groupCount, dagStats.maxLevel,
      dagStats.rootCount, dagStats.monotonicFixes);

  if (!kenshin::geometry::writeKGeo(output, dag, log)) {
    KENSHIN_ERROR("write failed: {}", log);
    kenshin::log::shutdown();
    return 1;
  }
  KENSHIN_INFO("[builder] written: {}", output);
  kenshin::log::shutdown();
  return 0;
}
