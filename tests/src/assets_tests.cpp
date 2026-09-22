// S2.1 asset tests: OBJ known-answer assets + procedural generators.

#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp>
#include <map>
#include <string>
#include <tuple>

#include "kenshin/assets/MeshData.h"
#include "kenshin/assets/ObjLoader.h"
#include "kenshin/core/Log.h"
#include "kenshin/core/Timer.h"
#include "kenshin/geometry/shapes/SphereGeometry.h"
#include "kenshin/geometry/shapes/TorusKnotGeometry.h"
#include "kenshin/util/BmpReader.h"
#include "kenshin/util/BmpWriter.h"

namespace {
std::string assetPath(const char* name) {
  for (const std::filesystem::path candidate : {
           "tests/assets",
           "../../../tests/assets",
       }) {
    const auto path = candidate / name;
    std::error_code ec;
    if (std::filesystem::exists(path, ec)) return path.string();
  }
  return name;
}

const kenshin::assets::MeshData& loadOnce(const char* name,
                                          kenshin::assets::MeshData& out,
                                          std::string& log) {
  static std::map<std::string, kenshin::assets::MeshData> cache;
  const std::string path = assetPath(name);
  if (cache.find(path) == cache.end()) {
    if (!kenshin::assets::loadObj(path, cache[path], log))
      FAIL("loadObj failed for " << path << ": " << log);
  }
  out = cache[path];
  return out;
}
}  // namespace

TEST_CASE("obj tetra known answer", "[assets][obj]") {
  kenshin::assets::MeshData mesh;
  std::string log;
  loadOnce("tetra.obj", mesh, log);
  REQUIRE(mesh.vertexCount() == 4);
  REQUIRE(mesh.triangleCount() == 4);
  REQUIRE(mesh.valid());
  // Every edge shared by exactly 2 triangles (closed): each of 4 faces has
  // 3 edges -> 12 edge incidences -> 6 unique edges with degree 2.
  std::map<std::pair<kenshin::u32, kenshin::u32>, int> edges;
  for (kenshin::usize i = 0; i + 2 < mesh.indices.size(); i += 3) {
    const kenshin::u32 tri[3] = {mesh.indices[i], mesh.indices[i + 1],
                                 mesh.indices[i + 2]};
    for (int e = 0; e < 3; ++e) {
      const kenshin::u32 a = tri[e], b = tri[(e + 1) % 3];
      edges[{glm::min(a, b), glm::max(a, b)}]++;
    }
  }
  REQUIRE(edges.size() == 6);
  for (const auto& [edge, degree] : edges) REQUIRE(degree == 2);
}

TEST_CASE("obj cube known answer", "[assets][obj]") {
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

TEST_CASE("obj grid known answer", "[assets][obj]") {
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

TEST_CASE("obj two sheets load", "[assets][obj]") {
  kenshin::assets::MeshData mesh;
  std::string log;
  loadOnce("two_sheets.obj", mesh, log);
  REQUIRE(mesh.vertexCount() == 18);
  REQUIRE(mesh.triangleCount() == 16);  // 2 sheets x (2x2 quads x 2)
}

TEST_CASE("obj missing normals get defaults", "[assets][obj]") {
  kenshin::assets::MeshData mesh;
  std::string log;
  loadOnce("cube.obj", mesh, log);
  REQUIRE(mesh.normals.size() == mesh.positions.size());
  // Default normal (0,1,0) since the file has no vn.
  for (kenshin::usize i = 0; i + 2 < mesh.normals.size(); i += 3) {
    REQUIRE(mesh.normals[i] == 0.0f);
    REQUIRE(mesh.normals[i + 1] == 1.0f);
    REQUIRE(mesh.normals[i + 2] == 0.0f);
  }
}

TEST_CASE("uv sphere exact counts", "[assets][procedural]") {
  const auto mesh = kenshin::geometry::SphereGeometry::create(64, 32, 1.0f);
  REQUIRE(mesh.triangleCount() == 64u * (2u * (32u - 1u)));
  REQUIRE(mesh.valid());
  // Unit sphere bounds.
  glm::vec3 mn, mx;
  mesh.computeBounds(mn, mx);
  REQUIRE(glm::all(glm::epsilonEqual(mn, glm::vec3(-1.0f), 1e-4f)));
  REQUIRE(glm::all(glm::epsilonEqual(mx, glm::vec3(1.0f), 1e-4f)));
}

TEST_CASE("torus knot million triangles under 2s", "[assets][procedural]") {
  kenshin::Timer timer;
  const auto mesh =
      kenshin::geometry::TorusKnotGeometry::create(2, 3, 512, 1024, 1.0f);
  const double seconds = timer.elapsedSeconds();
  REQUIRE(mesh.triangleCount() == 512u * 1024u * 2u);
  REQUIRE(mesh.valid());
  REQUIRE(seconds < 2.0);
  kenshin::log::init();
  KENSHIN_INFO("[test] torus knot 1,048,576 tris generated in {:.0f} ms",
               seconds * 1000.0);
}

TEST_CASE("bmp reader roundtrips the writer and the committed checker",
          "[assets][bmp]") {
  // M8.2: the hand-written 24-bit BI_RGB reader. Roundtrip the writer's
  // exact output first (top-down RGBA in, top-down RGBA out).
  std::vector<kenshin::u8> rgba(8 * 5 * 4);
  for (kenshin::u32 i = 0; i < rgba.size() / 4; ++i) {
    rgba[i * 4 + 0] = kenshin::u8(i * 7);
    rgba[i * 4 + 1] = kenshin::u8(255 - i * 3);
    rgba[i * 4 + 2] = kenshin::u8(i);
    rgba[i * 4 + 3] = 255;
  }
  REQUIRE(kenshin::util::writeBmp24("out/bmp_roundtrip.bmp", rgba, 8, 5));
  kenshin::util::ImageRgba img;
  std::string log;
  REQUIRE(kenshin::util::readBmp24("out/bmp_roundtrip.bmp", img, log));
  INFO("log: " << log);
  REQUIRE(img.width == 8);
  REQUIRE(img.height == 5);
  REQUIRE(img.rgba == rgba);
  std::remove("out/bmp_roundtrip.bmp");

  // The committed test albedo parses with the expected structure: a
  // 64x64 two-tone checker with a colored border (the M8.2 regression
  // asset - catches silent format drift of the checked-in file).
  REQUIRE(kenshin::util::readBmp24(assetPath("checker_64.bmp"), img, log));
  REQUIRE(img.width == 64);
  REQUIRE(img.height == 64);
  REQUIRE(img.rgba.size() == 64 * 64 * 4);
  const auto px = [&](int x, int y) {
    const kenshin::u8* p = img.rgba.data() + (kenshin::usize(y) * 64 + x) * 4;
    return std::tuple<int, int, int>(p[0], p[1], p[2]);
  };
  // Border rows/cols carry the orientation colors; interior is the
  // two-tone 8px checker.
  auto [tr, tg, tb] = px(32, 1);     // top strip
  auto [br, bg, bb] = px(32, 62);    // bottom strip
  REQUIRE(tr > tb);                  // top reads red-ish
  REQUIRE(bb > br);                  // bottom reads blue-ish
  auto [c0r, c0g, c0b] = px(6, 8);   // first checker cell (light)
  auto [c1r, c1g, c1b] = px(10, 8);  // adjacent cell (dark)
  REQUIRE(c0r - c1r > 60);           // the two tones differ
}
