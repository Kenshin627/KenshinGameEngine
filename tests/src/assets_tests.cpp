// S2.1 asset tests: OBJ known-answer assets + procedural generators.

#include <catch2/catch_test_macros.hpp>

#include "kenshin/assets/MeshData.h"
#include "kenshin/assets/ObjLoader.h"
#include "kenshin/assets/ProceduralMesh.h"
#include "kenshin/core/Log.h"
#include "kenshin/core/Timer.h"

#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp>
#include <filesystem>
#include <map>
#include <string>

namespace
{
    std::string assetPath(const char* name)
    {
        for (const std::filesystem::path candidate : {
                 "tests/assets",
                 "../../../tests/assets",
             })
        {
            const auto path = candidate / name;
            std::error_code ec;
            if (std::filesystem::exists(path, ec))
                return path.string();
        }
        return name;
    }

    const kenshin::assets::MeshData& loadOnce(const char* name,
                                              kenshin::assets::MeshData& out,
                                              std::string& log)
    {
        static std::map<std::string, kenshin::assets::MeshData> cache;
        const std::string path = assetPath(name);
        if (cache.find(path) == cache.end())
        {
            if (!kenshin::assets::loadObj(path, cache[path], log))
                FAIL("loadObj failed for " << path << ": " << log);
        }
        out = cache[path];
        return out;
    }
} // namespace

TEST_CASE("obj tetra known answer", "[assets][obj]")
{
    kenshin::assets::MeshData mesh;
    std::string log;
    loadOnce("tetra.obj", mesh, log);
    REQUIRE(mesh.vertexCount() == 4);
    REQUIRE(mesh.triangleCount() == 4);
    REQUIRE(mesh.valid());
    // Every edge shared by exactly 2 triangles (closed): each of 4 faces has
    // 3 edges -> 12 edge incidences -> 6 unique edges with degree 2.
    std::map<std::pair<kenshin::u32, kenshin::u32>, int> edges;
    for (kenshin::usize i = 0; i + 2 < mesh.indices.size(); i += 3)
    {
        const kenshin::u32 tri[3] = {mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]};
        for (int e = 0; e < 3; ++e)
        {
            const kenshin::u32 a = tri[e], b = tri[(e + 1) % 3];
            edges[{glm::min(a, b), glm::max(a, b)}]++;
        }
    }
    REQUIRE(edges.size() == 6);
    for (const auto& [edge, degree] : edges)
        REQUIRE(degree == 2);
}

TEST_CASE("obj cube known answer", "[assets][obj]")
{
    kenshin::assets::MeshData mesh;
    std::string log;
    loadOnce("cube.obj", mesh, log);
    // Quad faces are fan-triangulated; positions dedup to the 8 corners.
    REQUIRE(mesh.vertexCount() == 8);
    REQUIRE(mesh.triangleCount() == 12);
    glm::vec3 mn, mx;
    mesh.computeBounds(mn, mx);
    REQUIRE(glm::all(glm::epsilonEqual(mn, glm::vec3(-0.5f), 1e-6f)));
    REQUIRE(glm::all(glm::epsilonEqual(mx, glm::vec3(0.5f), 1e-6f)));
}

TEST_CASE("obj grid known answer", "[assets][obj]")
{
    kenshin::assets::MeshData mesh;
    std::string log;
    loadOnce("grid.obj", mesh, log);
    REQUIRE(mesh.vertexCount() == 81);
    REQUIRE(mesh.triangleCount() == 128);
    glm::vec3 mn, mx;
    mesh.computeBounds(mn, mx);
    REQUIRE(glm::all(glm::epsilonEqual(mn, glm::vec3(0.0f, 0.0f, 0.0f), 1e-6f)));
    REQUIRE(glm::all(glm::epsilonEqual(mx, glm::vec3(8.0f, 0.0f, 8.0f), 1e-6f)));
    // Planar: all y == 0.
    for (kenshin::usize i = 1; i < mesh.positions.size(); i += 3)
        REQUIRE(glm::abs(mesh.positions[i]) < 1e-6f);
}

TEST_CASE("obj two sheets load", "[assets][obj]")
{
    kenshin::assets::MeshData mesh;
    std::string log;
    loadOnce("two_sheets.obj", mesh, log);
    REQUIRE(mesh.vertexCount() == 18);
    REQUIRE(mesh.triangleCount() == 16); // 2 sheets x (2x2 quads x 2)
}

TEST_CASE("obj missing normals get defaults", "[assets][obj]")
{
    kenshin::assets::MeshData mesh;
    std::string log;
    loadOnce("cube.obj", mesh, log);
    REQUIRE(mesh.normals.size() == mesh.positions.size());
    // Default normal (0,1,0) since the file has no vn.
    for (kenshin::usize i = 0; i + 2 < mesh.normals.size(); i += 3)
    {
        REQUIRE(mesh.normals[i] == 0.0f);
        REQUIRE(mesh.normals[i + 1] == 1.0f);
        REQUIRE(mesh.normals[i + 2] == 0.0f);
    }
}

TEST_CASE("uv sphere exact counts", "[assets][procedural]")
{
    const auto mesh = kenshin::assets::createUvSphere(64, 32, 1.0f);
    REQUIRE(mesh.triangleCount() == 64u * (2u * (32u - 1u)));
    REQUIRE(mesh.valid());
    // Unit sphere bounds.
    glm::vec3 mn, mx;
    mesh.computeBounds(mn, mx);
    REQUIRE(glm::all(glm::epsilonEqual(mn, glm::vec3(-1.0f), 1e-4f)));
    REQUIRE(glm::all(glm::epsilonEqual(mx, glm::vec3(1.0f), 1e-4f)));
}

TEST_CASE("torus knot million triangles under 2s", "[assets][procedural]")
{
    kenshin::Timer timer;
    const auto mesh = kenshin::assets::createTorusKnot(2, 3, 512, 1024, 1.0f, 0.35f);
    const double seconds = timer.elapsedSeconds();
    REQUIRE(mesh.triangleCount() == 512u * 1024u * 2u);
    REQUIRE(mesh.valid());
    REQUIRE(seconds < 2.0);
    kenshin::log::init();
    KENSHIN_INFO("[test] torus knot 1,048,576 tris generated in {:.0f} ms",
                 seconds * 1000.0);
}
