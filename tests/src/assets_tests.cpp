// S2.1 asset tests: OBJ known-answer assets + procedural generators.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp>
#include <map>
#include <string>
#include <tuple>

#include "kenshin/assets/GltfLoader.h"
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

namespace {
// GLB writer helpers: a tiny JSON + BIN with one triangle primitive
// (POSITION/TEXCOORD, ushort indices, NO normals -> compute path) and
// an embedded BMP image.
std::vector<kenshin::u8> pad4(const std::vector<kenshin::u8>& b) {
  std::vector<kenshin::u8> out = b;
  out.resize(out.size() + (4 - out.size() % 4) % 4, 0);
  return out;
}
void appendU32(std::vector<kenshin::u8>& out, kenshin::u32 v) {
  out.insert(out.end(), {kenshin::u8(v), kenshin::u8(v >> 8),
                         kenshin::u8(v >> 16), kenshin::u8(v >> 24)});
}
}  // namespace

TEST_CASE("gltf loader: glb with transform, computed normals, image",
          "[assets][gltf]") {
  // One triangle at x+0.5 (node translation), UVs, embedded 4x2 BMP.
  const float pos[9] = {0.5f, 0.0f, 0.0f, 1.5f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f};
  const float uv[6] = {0.0f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f};
  const kenshin::u16 idx[3] = {0, 1, 2};
  std::vector<kenshin::u8> posB((const kenshin::u8*)pos,
                                (const kenshin::u8*)pos + sizeof(pos));
  std::vector<kenshin::u8> uvB((const kenshin::u8*)uv,
                               (const kenshin::u8*)uv + sizeof(uv));
  std::vector<kenshin::u8> idxB((const kenshin::u8*)idx,
                                (const kenshin::u8*)idx + sizeof(idx));
  // 4x2 two-tone BMP (writeBmp24's exact shape).
  std::vector<kenshin::u8> rgba(4 * 2 * 4);
  for (int i = 0; i < 8; ++i) {
    const int tone = i % 2 ? 200 : 60;
    rgba[kenshin::usize(i) * 4 + 0] = kenshin::u8(tone);
    rgba[kenshin::usize(i) * 4 + 1] = kenshin::u8(tone);
    rgba[kenshin::usize(i) * 4 + 2] = kenshin::u8(tone);
    rgba[kenshin::usize(i) * 4 + 3] = 255;
  }
  std::vector<kenshin::u8> bmp;
  {
    std::vector<kenshin::u8> bgr;
    for (int y = 1; y >= 0; --y)
      for (int x = 0; x < 4; ++x) {
        bgr.push_back(rgba[(kenshin::usize(y) * 4 + x) * 4 + 2]);
        bgr.push_back(rgba[(kenshin::usize(y) * 4 + x) * 4 + 1]);
        bgr.push_back(rgba[(kenshin::usize(y) * 4 + x) * 4 + 0]);
      }
    const kenshin::u8 fh[14] = {0x42, 0x4D, 0, 0,  0, 0, 0,
                                0,    0,    0, 54, 0, 0, 0};
    bmp.insert(bmp.end(), fh, fh + 14);
    appendU32(bmp, 40);
    appendU32(bmp, 4);
    appendU32(bmp, 2);
    bmp.insert(bmp.end(), {1, 0, 24, 0, 0, 0, 0, 0});
    appendU32(bmp, kenshin::u32(bgr.size()));
    bmp.insert(bmp.end(), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    bmp.insert(bmp.end(), bgr.begin(), bgr.end());
  }
  const auto posP = pad4(posB), uvP = pad4(uvB), idxP = pad4(idxB);
  std::vector<kenshin::u8> bin = posP;
  bin.insert(bin.end(), uvP.begin(), uvP.end());
  bin.insert(bin.end(), idxP.begin(), idxP.end());
  bin.insert(bin.end(), bmp.begin(), bmp.end());
  const kenshin::u32 uvOff = kenshin::u32(posP.size());
  const kenshin::u32 idxOff = uvOff + kenshin::u32(uvP.size());
  const kenshin::u32 imgOff = idxOff + kenshin::u32(idxP.size());

  const std::string json =
      "{\"asset\":{\"version\":\"2.0\"},\"scene\":0,"
      "\"scenes\":[{\"nodes\":[0]}],"
      "\"nodes\":[{\"mesh\":0,\"translation\":[0.5,0,0]}],"
      "\"meshes\":[{\"primitives\":[{\"attributes\":{"
      "\"POSITION\":0,\"TEXCOORD_0\":1},\"indices\":2,"
      "\"material\":0}]}],"
      "\"materials\":[{\"pbrMetallicRoughness\":{"
      "\"baseColorFactor\":[0.9,0.8,0.7,1],\"metallicFactor\":0.1,"
      "\"roughnessFactor\":0.35,\"baseColorTexture\":{\"index\":0}}}],"
      "\"textures\":[{\"source\":0}],"
      "\"images\":[{\"bufferView\":3,\"mimeType\":\"image/bmp\"}],"
      "\"accessors\":["
      "{\"bufferView\":0,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\"},"
      "{\"bufferView\":1,\"componentType\":5126,\"count\":3,\"type\":\"VEC2\"},"
      "{\"bufferView\":2,\"componentType\":5123,\"count\":3,\"type\":"
      "\"SCALAR\"}],"
      "\"bufferViews\":["
      "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":" +
      std::to_string(posB.size()) +
      "},"
      "{\"buffer\":0,\"byteOffset\":" +
      std::to_string(uvOff) + ",\"byteLength\":" + std::to_string(uvB.size()) +
      "},"
      "{\"buffer\":0,\"byteOffset\":" +
      std::to_string(idxOff) +
      ",\"byteLength\":" + std::to_string(idxB.size()) +
      "},"
      "{\"buffer\":0,\"byteOffset\":" +
      std::to_string(imgOff) + ",\"byteLength\":" + std::to_string(bmp.size()) +
      "}],"
      "\"buffers\":[{\"byteLength\":" +
      std::to_string(bin.size()) + "}]}";
  const auto jsonP = pad4(std::vector<kenshin::u8>(json.begin(), json.end()));
  std::vector<kenshin::u8> glb;
  appendU32(glb, 0x46546C67);
  appendU32(glb, 2);
  appendU32(glb, kenshin::u32(12 + 8 + jsonP.size() + 8 + bin.size()));
  appendU32(glb, kenshin::u32(jsonP.size()));
  appendU32(glb, 0x4E4F534A);
  glb.insert(glb.end(), jsonP.begin(), jsonP.end());
  appendU32(glb, kenshin::u32(bin.size()));
  appendU32(glb, 0x004E4942);
  glb.insert(glb.end(), bin.begin(), bin.end());
  REQUIRE((std::filesystem::exists("out") ||
           std::filesystem::create_directories("out")));
  {
    FILE* f = nullptr;
    REQUIRE(fopen_s(&f, "out/tri.glb", "wb") == 0);
    std::fwrite(glb.data(), 1, glb.size(), f);
    std::fclose(f);
  }

  kenshin::assets::MeshData mesh;
  kenshin::assets::GltfImage albedo;
  std::string log;
  REQUIRE(kenshin::assets::loadGltf("out/tri.glb", mesh, &albedo, log));
  INFO("log: " << log);
  REQUIRE(mesh.vertexCount() == 3);
  REQUIRE(mesh.triangleCount() == 1);
  REQUIRE(mesh.positions.size() == 9);
  // Node translation applied: x + 0.5.
  REQUIRE(mesh.positions[0] == Catch::Approx(1.0f).margin(1e-5f));
  REQUIRE(mesh.positions[3] == Catch::Approx(2.0f).margin(1e-5f));
  REQUIRE(mesh.positions[7] == Catch::Approx(1.0f).margin(1e-5f));
  REQUIRE(mesh.positions[1] == Catch::Approx(0.0f).margin(1e-5f));
  REQUIRE(mesh.indices.size() == 3);
  REQUIRE(mesh.indices[0] == 0);
  REQUIRE(mesh.indices[2] == 2);
  REQUIRE(mesh.uvs.size() == 6);
  REQUIRE(mesh.uvs[2] == Catch::Approx(1.0f).margin(1e-5f));
  // Normals were COMPUTED (none in the file): +z face normal.
  REQUIRE(mesh.normals.size() == 9);
  REQUIRE(mesh.normals[2] == Catch::Approx(1.0f).margin(1e-4f));
  REQUIRE(mesh.normals[5] == Catch::Approx(1.0f).margin(1e-4f));
  // Material: factors + the decoded embedded image.
  REQUIRE(mesh.material.kd.r == Catch::Approx(0.9f).margin(1e-5f));
  REQUIRE(mesh.material.roughness == Catch::Approx(0.35f).margin(1e-5f));
  REQUIRE(mesh.material.metallic == Catch::Approx(0.1f).margin(1e-5f));
  REQUIRE(albedo.width == 4);
  REQUIRE(albedo.height == 2);
  REQUIRE(albedo.rgba.size() == 4 * 2 * 4);
  REQUIRE(albedo.rgba[0] == 60);   // first texel dark tone
  REQUIRE(albedo.rgba[4] == 200);  // second texel light tone
  std::remove("out/tri.glb");
}

TEST_CASE("gltf loader: rejects hostile bounds violations", "[assets][gltf]") {
  // The three P0 classes the adversarial review caught, now pinned:
  // index values beyond the vertex count, node mesh indices beyond the
  // meshes array, and the u32-wrapping image bufferView bound.
  const auto writeGlb = [](const std::string& json,
                           const std::vector<kenshin::u8>& bin,
                           const char* path) {
    std::vector<kenshin::u8> jsonP(json.begin(), json.end());
    jsonP.resize(jsonP.size() + (4 - jsonP.size() % 4) % 4, 0);
    std::vector<kenshin::u8> glb;
    auto u32 = [&glb](kenshin::u32 v) {
      glb.insert(glb.end(), {kenshin::u8(v), kenshin::u8(v >> 8),
                             kenshin::u8(v >> 16), kenshin::u8(v >> 24)});
    };
    u32(0x46546C67);
    u32(2);
    u32(kenshin::u32(12 + 8 + jsonP.size() + 8 + bin.size()));
    u32(kenshin::u32(jsonP.size()));
    u32(0x4E4F534A);
    glb.insert(glb.end(), jsonP.begin(), jsonP.end());
    u32(kenshin::u32(bin.size()));
    u32(0x004E4942);
    glb.insert(glb.end(), bin.begin(), bin.end());
    FILE* f = nullptr;
    REQUIRE(fopen_s(&f, path, "wb") == 0);
    std::fwrite(glb.data(), 1, glb.size(), f);
    std::fclose(f);
  };
  const float pos[9] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
  const kenshin::u16 badIdx[3] = {0, 60000, 2};  // 60000 >= 3 verts
  std::vector<kenshin::u8> bin((const kenshin::u8*)pos,
                               (const kenshin::u8*)pos + sizeof(pos));
  bin.insert(bin.end(), (const kenshin::u8*)badIdx,
             (const kenshin::u8*)badIdx + sizeof(badIdx));
  bin.resize(bin.size() + (4 - bin.size() % 4) % 4, 0);
  const std::string head =
      "{\"asset\":{\"version\":\"2.0\"},\"scene\":0,"
      "\"scenes\":[{\"nodes\":[0]}],"
      "\"nodes\":[{\"mesh\":0}],"
      "\"meshes\":[{\"primitives\":[{"
      "\"attributes\":{\"POSITION\":0},\"indices\":1}]}],"
      "\"accessors\":["
      "{\"bufferView\":0,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\"}"
      ",";
  const std::string tail =
      "],\"bufferViews\":["
      "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":36},"
      "{\"buffer\":0,\"byteOffset\":36,\"byteLength\":6}],"
      "\"buffers\":[{\"byteLength\":48}]}";
  REQUIRE((std::filesystem::exists("out") ||
           std::filesystem::create_directories("out")));

  kenshin::assets::MeshData mesh;
  kenshin::assets::GltfImage albedo;
  std::string log;
  writeGlb(head +
               "{\"bufferView\":1,\"componentType\":5123,\"count\":3,"
               "\"type\":\"SCALAR\"}" +
               tail,
           bin, "out/hostile_idx.glb");
  REQUIRE_FALSE(
      kenshin::assets::loadGltf("out/hostile_idx.glb", mesh, &albedo, log));
  REQUIRE(log.find("vertex range") != std::string::npos);
  std::remove("out/hostile_idx.glb");

  // Node mesh index 4294967295 against a 1-entry meshes array.
  const std::string badMeshJson =
      "{\"asset\":{\"version\":\"2.0\"},\"scene\":0,"
      "\"scenes\":[{\"nodes\":[0]}],"
      "\"nodes\":[{\"mesh\":4294967295}],"
      "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0}}]}],"
      "\"accessors\":[{\"bufferView\":0,\"componentType\":5126,"
      "\"count\":3,\"type\":\"VEC3\"}],"
      "\"bufferViews\":[{\"buffer\":0,\"byteOffset\":0,\"byteLength\":36}],"
      "\"buffers\":[{\"byteLength\":36}]}";
  writeGlb(badMeshJson,
           std::vector<kenshin::u8>((const kenshin::u8*)pos,
                                    (const kenshin::u8*)pos + sizeof(pos)),
           "out/hostile_mesh.glb");
  REQUIRE_FALSE(
      kenshin::assets::loadGltf("out/hostile_mesh.glb", mesh, &albedo, log));
  std::remove("out/hostile_mesh.glb");

  // Image bufferView with byteOffset 0xFFFFFFF0 + length 0x40: the
  // u32 add wraps to 48 and would pass a 32-bit bounds check.
  const std::string wrapJson =
      "{\"asset\":{\"version\":\"2.0\"},\"scene\":0,"
      "\"scenes\":[{\"nodes\":[0]}],"
      "\"nodes\":[{\"mesh\":0}],"
      "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0},"
      "\"material\":0}]}],"
      "\"materials\":[{\"pbrMetallicRoughness\":{\"baseColorTexture\":"
      "{\"index\":0}}}],"
      "\"textures\":[{\"source\":0}],"
      "\"images\":[{\"bufferView\":1,\"mimeType\":\"image/bmp\"}],"
      "\"accessors\":[{\"bufferView\":0,\"componentType\":5126,"
      "\"count\":3,\"type\":\"VEC3\"}],"
      "\"bufferViews\":["
      "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":36},"
      "{\"buffer\":0,\"byteOffset\":4294967280,\"byteLength\":64}],"
      "\"buffers\":[{\"byteLength\":48}]}";
  writeGlb(wrapJson,
           std::vector<kenshin::u8>((const kenshin::u8*)pos,
                                    (const kenshin::u8*)pos + sizeof(pos)),
           "out/hostile_wrap.glb");
  // Image failures are SOFT by design (an untextured model still
  // loads) - the assertion is that the wrap was CAUGHT, not that the
  // whole load died: no OOB read may happen either way.
  (void)kenshin::assets::loadGltf("out/hostile_wrap.glb", mesh, &albedo, log);
  REQUIRE(log.find("out of bounds") != std::string::npos);
  REQUIRE(albedo.width == 0);  // nothing decoded from the bogus view
  std::remove("out/hostile_wrap.glb");
}

TEST_CASE("gltf loader: rejects garbage and missing buffers",
          "[assets][gltf]") {
  kenshin::assets::MeshData mesh;
  kenshin::assets::GltfImage albedo;
  std::string log;
  REQUIRE((std::filesystem::exists("out") ||
           std::filesystem::create_directories("out")));
  {
    FILE* f = nullptr;
    REQUIRE(fopen_s(&f, "out/bad.glb", "wb") == 0);
    const char junk[] = "not a gltf at all";
    std::fwrite(junk, 1, sizeof(junk), f);
    std::fclose(f);
  }
  REQUIRE_FALSE(kenshin::assets::loadGltf("out/bad.glb", mesh, &albedo, log));
  INFO("log: " << log);
  REQUIRE_FALSE(log.empty());
  std::remove("out/bad.glb");

  // A .gltf whose buffer 0 is a data URI: loud v1 rejection.
  {
    FILE* f = nullptr;
    REQUIRE(fopen_s(&f, "out/datauri.gltf", "wb") == 0);
    const char json[] =
        "{\"asset\":{\"version\":\"2.0\"},\"scenes\":[{\"nodes\":[0]}],"
        "\"nodes\":[{\"mesh\":0}],\"meshes\":[{\"primitives\":[{\"attributes\":"
        "{"
        "\"POSITION\":0}}]}],"
        "\"accessors\":[{\"bufferView\":0,\"componentType\":5126,"
        "\"count\":3,\"type\":\"VEC3\"}],"
        "\"bufferViews\":[{\"buffer\":0,\"byteOffset\":0,\"byteLength\":36}],"
        "\"buffers\":[{\"uri\":\"data:application/"
        "octet-stream;base64,AAAA\"}]}";
    std::fwrite(json, 1, sizeof(json) - 1, f);
    std::fclose(f);
  }
  REQUIRE_FALSE(
      kenshin::assets::loadGltf("out/datauri.gltf", mesh, &albedo, log));
  INFO("log: " << log);
  REQUIRE(log.find("data-uri") != std::string::npos);
  std::remove("out/datauri.gltf");
}
