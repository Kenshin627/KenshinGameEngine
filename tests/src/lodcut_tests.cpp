// S3.4: LOD cut tests - threshold monotonicity + cut validity.

#include <catch2/catch_test_macros.hpp>

#include "kenshin/assets/MeshData.h"
#include "kenshin/assets/ProceduralMesh.h"
#include "kenshin/geometry/ClusterDag.h"
#include "kenshin/geometry/ClusterLodBuilder.h"
#include "kenshin/geometry/LodCut.h"

#include <string>
#include <vector>

namespace
{
    struct Built
    {
        kenshin::assets::MeshData mesh;
        std::vector<kenshin::geometry::ClusterGroup> groups;
        kenshin::geometry::ClusterDag dag;
        std::string log;
    };

    bool build(const kenshin::assets::MeshData& mesh, Built& out)
    {
        out.mesh = mesh;
        kenshin::geometry::BuildStats stats;
        kenshin::geometry::DagBuildStats dagStats;
        return kenshin::geometry::buildClusterLod(
                   out.mesh,
                   [&](const kenshin::geometry::ClusterGroup& g) {
                       out.groups.push_back(g);
                   }, stats, out.log) &&
               kenshin::geometry::assembleClusterDag(out.mesh, out.groups, out.dag,
                                                     dagStats, out.log);
    }
} // namespace

TEST_CASE("lod cut threshold monotonicity", "[geometry][lodcut]")
{
    Built built;
    REQUIRE(build(kenshin::assets::createTorusKnot(2, 3, 192, 48, 1.0f, 0.32f), built));
    // Far camera: all groups see ~the same threshold, restoring strict
    // monotonicity (near cameras mix per-group scales legitimately).
    const float camPos[3] = {0.0f, 0.0f, 1000.0f};

    // Thresholds scaled for the far camera (world error = px * scale *
    // 1000): the knot's first-level error ~0.17 lands around 0.1px.
    kenshin::u32 prevTris = 0xFFFFFFFF;
    for (const kenshin::f32 threshold : {0.004f, 0.016f, 0.064f, 0.25f, 1.0f})
    {
        std::vector<kenshin::u32> cut;
        kenshin::geometry::selectLodCut(built.dag, threshold, 720, 60.0f, camPos, cut);
        const auto stats = kenshin::geometry::summarizeCut(built.dag, cut);
        INFO("threshold " << threshold << ": " << stats.triangleCount << " tris");
        REQUIRE(stats.clusterCount > 0);
        REQUIRE(stats.triangleCount <= prevTris);
        prevTris = stats.triangleCount;
    }
    // The finest threshold should render close to the full mesh.
    std::vector<kenshin::u32> fine;
    kenshin::geometry::selectLodCut(built.dag, 0.0001f, 720, 60.0f, camPos, fine);
    const auto fineStats = kenshin::geometry::summarizeCut(built.dag, fine);
    REQUIRE(fineStats.triangleCount >= built.mesh.triangleCount() / 2);
    // A huge threshold degrades to the roots.
    std::vector<kenshin::u32> coarse;
    kenshin::geometry::selectLodCut(built.dag, 1e9f, 720, 60.0f, camPos, coarse);
    const auto coarseStats = kenshin::geometry::summarizeCut(built.dag, coarse);
    REQUIRE(coarseStats.triangleCount < fineStats.triangleCount);
}

TEST_CASE("lod cut single-level-per-path validity", "[geometry][lodcut]")
{
    Built built;
    REQUIRE(build(kenshin::assets::createTorusKnot(2, 3, 128, 32, 1.0f, 0.32f), built));
    const float camPos[3] = {0.0f, 0.0f, 4.0f};

    std::vector<kenshin::u32> cut;
    kenshin::geometry::selectLodCut(built.dag, 2.0f, 720, 60.0f, camPos, cut);

    // Validity: no selected cluster is an ancestor of another selected one
    // (a path renders exactly one level). Walk children transitively.
    std::vector<bool> selected(built.dag.clusters.size(), false);
    for (kenshin::u32 c : cut)
        selected[c] = true;
    for (kenshin::u32 c : cut)
    {
        // BFS descendants of c; none may be selected.
        std::vector<kenshin::u32> stack = built.dag.clusters[c].children;
        while (!stack.empty())
        {
            const kenshin::u32 d = stack.back();
            stack.pop_back();
            REQUIRE_FALSE(selected[d]);
            stack.insert(stack.end(), built.dag.clusters[d].children.begin(),
                         built.dag.clusters[d].children.end());
        }
    }
}

TEST_CASE("lod cut never empty", "[geometry][lodcut]")
{
    Built built;
    REQUIRE(build(kenshin::assets::createUvSphere(32, 16, 1.0f), built));
    const float camPos[3] = {0.0f, 0.0f, 4.0f};
    for (const kenshin::f32 threshold : {1e-6f, 1.0f, 1e12f})
    {
        std::vector<kenshin::u32> cut;
        kenshin::geometry::selectLodCut(built.dag, threshold, 720, 60.0f, camPos, cut);
        REQUIRE(!cut.empty());
    }
}
