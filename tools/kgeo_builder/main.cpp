// kgeo_builder CLI (S3.3): input.obj | --proc <spec> -> output.kgeo

#include "kenshin/Version.h"
#include "kenshin/assets/MeshData.h"
#include "kenshin/assets/ObjLoader.h"
#include "kenshin/assets/ProceduralMesh.h"
#include "kenshin/core/Log.h"
#include "kenshin/geometry/ClusterDag.h"
#include "kenshin/geometry/ClusterLodBuilder.h"
#include "kenshin/geometry/KGeoFile.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    kenshin::log::init();
    KENSHIN_INFO("{}", kenshin::getBanner());

    std::string input, output = "out/mesh.kgeo";
    std::string proc; // "sphere" or "knot"
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc)
            output = argv[++i];
        else if (std::strcmp(argv[i], "--proc") == 0 && i + 1 < argc)
            proc = argv[++i];
        else
            input = argv[i];
    }
    if (input.empty() && proc.empty())
    {
        KENSHIN_ERROR("usage: kgeo_builder input.obj [-o out.kgeo]");
        KENSHIN_ERROR("       kgeo_builder --proc knot|sphere [-o out.kgeo]");
        kenshin::log::shutdown();
        return 1;
    }

    kenshin::assets::MeshData mesh;
    std::string log;
    if (!proc.empty())
    {
        if (proc == "knot")
            mesh = kenshin::assets::createTorusKnot(2, 3, 512, 1024, 1.0f, 0.35f);
        else if (proc == "sphere")
            mesh = kenshin::assets::createUvSphere(256, 128, 1.0f);
        else
        {
            KENSHIN_ERROR("unknown --proc '{}'", proc);
            kenshin::log::shutdown();
            return 1;
        }
    }
    else if (!kenshin::assets::loadObj(input, mesh, log))
    {
        KENSHIN_ERROR("load failed: {}", log);
        kenshin::log::shutdown();
        return 1;
    }
    KENSHIN_INFO("[builder] mesh: {} verts / {} tris", mesh.vertexCount(),
                 mesh.triangleCount());

    std::vector<kenshin::geometry::ClusterGroup> groups;
    kenshin::geometry::BuildStats buildStats;
    if (!kenshin::geometry::buildClusterLod(
            mesh, [&](const kenshin::geometry::ClusterGroup& g) { groups.push_back(g); },
            buildStats, log))
    {
        KENSHIN_ERROR("clusterlod failed: {}", log);
        kenshin::log::shutdown();
        return 1;
    }

    kenshin::geometry::ClusterDag dag;
    kenshin::geometry::DagBuildStats dagStats;
    if (!kenshin::geometry::assembleClusterDag(mesh, groups, dag, dagStats, log))
    {
        KENSHIN_ERROR("dag assembly failed: {}", log);
        kenshin::log::shutdown();
        return 1;
    }
    KENSHIN_INFO("[builder] dag: {} clusters, {} groups, {} levels, {} roots, "
                 "{} monotonic fixes",
                 dagStats.clusterCount, dagStats.groupCount, dagStats.maxLevel,
                 dagStats.rootCount, dagStats.monotonicFixes);

    if (!kenshin::geometry::writeKGeo(output, dag, log))
    {
        KENSHIN_ERROR("write failed: {}", log);
        kenshin::log::shutdown();
        return 1;
    }
    KENSHIN_INFO("[builder] written: {}", output);
    kenshin::log::shutdown();
    return 0;
}
