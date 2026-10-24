// Asset database (roadmap step 1.3c) on temporary projects: identities
// (.meta), repairs (moves, copies, corrupt metas), staleness (content,
// settings, importer version, dependencies, lost cache), model
// sub-assets with derived Guids, validation, garbage collection and
// byte-identical re-imports. Plus SHA-1 / RFC 4122 v5 test vectors.

#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <filesystem>
#include <map>
#include <string>

#include "kenshin/assets/AssetDatabase.h"
#include "kenshin/assets/Ktx2.h"
#include "kenshin/assets/Material.h"
#include "kenshin/core/Sha1.h"
#include "kenshin/scene/ProjectFile.h"
#include "kenshin/util/File.h"
#include "test_gltf.h"
#include "test_png.h"

using namespace kenshin;
namespace fs = std::filesystem;
using assets::AssetDatabase;
using assets::AssetType;
using assets::ArtifactType;
using serialization::Json;

namespace {

/// A throwaway project: project.kproject + assets/.
struct Project {
  testgltf::TempDir dir;
  explicit Project(const std::string& name) : dir("kenshin_assetdb_" + name) {
    REQUIRE(scene::saveProjectFile(dir.path / scene::kProjectFileName, {"Test", ""}, nullptr));
    fs::create_directories(dir.path / "assets");
  }
  fs::path at(const std::string& rel) const { return dir.path / util::pathFromUtf8(rel); }
  void write(const std::string& rel, const std::string& bytes) const {
    fs::create_directories(at(rel).parent_path());
    REQUIRE(util::writeFileAtomic(at(rel), bytes));
  }
  void write(const std::string& rel, const std::vector<u8>& bytes) const {
    write(rel, std::string(bytes.begin(), bytes.end()));
  }
  std::string read(const std::string& rel) const {
    std::string s;
    REQUIRE(util::readFile(at(rel), s));
    return s;
  }
};

std::vector<u8> png(u8 shade, u32 w = 4, u32 h = 4) {
  return testpng::makePng(w, h, 8, 6, std::vector<u32>(usize(w) * h * 4, shade));
}

const ecs::ComponentRegistry& registry() {
  static const ecs::ComponentRegistry r = [] {
    ecs::ComponentRegistry reg;
    ecs::registerBuiltinComponents(reg);
    return reg;
  }();
  return r;
}

AssetDatabase openDb(const Project& p, assets::ScanReport* report = nullptr, u32 salt = 0) {
  AssetDatabase db;
  assets::AssetDatabaseOptions o;
  o.registry = &registry();
  o.importerVersionSalt = salt;
  std::string err;
  INFO(err);
  REQUIRE(db.open(p.dir.path, o, &err, report));
  return db;
}

/// Every cache file (relative path -> SHA-1).
std::map<std::string, std::string> cacheHashes(const AssetDatabase& db) {
  std::map<std::string, std::string> out;
  for (const auto& e : fs::recursive_directory_iterator(db.cacheDir()))
    if (e.is_regular_file()) {
      std::string bytes;
      REQUIRE(util::readFile(e.path(), bytes));
      out[util::pathToUtf8(fs::relative(e.path(), db.cacheDir()).generic_u8string())] =
          core::Sha1::hex(core::Sha1::of(bytes));
    }
  return out;
}

bool contains(const std::vector<std::string>& v, const std::string& part) {
  for (const auto& s : v)
    if (s.find(part) != std::string::npos) return true;
  return false;
}

core::Guid metaGuid(const Project& p, const std::string& rel) {
  Json doc;
  REQUIRE(serialization::parseJson(p.read(rel + ".meta"), doc, nullptr));
  return *core::Guid::parse(doc["guid"].get<std::string>());
}

}  // namespace

// ── Hashing / identities ───────────────────────────────────────

TEST_CASE("sha1: FIPS 180 test vectors", "[assetdb]") {
  CHECK(core::Sha1::hex(core::Sha1::of("")) == "da39a3ee5e6b4b0d3255bfef95601890afd80709");
  CHECK(core::Sha1::hex(core::Sha1::of("abc")) == "a9993e364706816aba3e25717850c26c9cd0d89d");
  CHECK(core::Sha1::hex(core::Sha1::of(
            "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")) ==
        "84983e441c3bd26ebaae4aa1f95129e5e54670f1");
  // 1,000,000 x 'a', fed in uneven pieces (buffering across blocks).
  core::Sha1 sha;
  const std::string chunk(997, 'a');
  usize fed = 0;
  while (fed + chunk.size() <= 1000000) {
    sha.update(chunk);
    fed += chunk.size();
  }
  sha.update(std::string(1000000 - fed, 'a'));
  CHECK(core::Sha1::hex(sha.finish()) == "34aa973cd4c4daa4f61eeb2bdbad27316534016f");
}

TEST_CASE("guid: RFC 4122 version-5 name-based guids", "[assetdb]") {
  const core::Guid dns = *core::Guid::parse("6ba7b810-9dad-11d1-80b4-00c04fd430c8");
  CHECK(core::Guid::fromName(dns, "www.example.com").toString() ==
        "2ed6657d-e927-568b-95e1-2665a8aea6a2");
  const core::Guid a = core::Guid::generate();
  CHECK(AssetDatabase::subAssetGuid(a, "mesh/0") == AssetDatabase::subAssetGuid(a, "mesh/0"));
  CHECK(AssetDatabase::subAssetGuid(a, "mesh/0") != AssetDatabase::subAssetGuid(a, "mesh/1"));
  CHECK(AssetDatabase::subAssetGuid(a, "mesh/0") !=
        AssetDatabase::subAssetGuid(core::Guid::generate(), "mesh/0"));
  CHECK(AssetDatabase::subAssetGuid(a, "x").toString()[14] == '5');
}

// ── Scan + import ──────────────────────────────────────────────

TEST_CASE("asset db: new sources get metas, import, and stay current", "[assetdb]") {
  Project p("basic");
  p.write("assets/textures/brick.png", png(120));
  p.write("assets/textures/brick_normal.png", png(128));
  p.write("assets/textures/brick_rough.png", png(60));
  p.write("assets/notes.txt", "not an asset");
  assets::ScanReport scan;
  AssetDatabase db = openDb(p, &scan);
  CHECK(scan.createdMetas.size() == 3);
  CHECK_FALSE(fs::exists(p.at("assets/notes.txt.meta")));
  REQUIRE(db.assets().size() == 3);
  const assets::AssetRecord* brick = db.findByPath("assets/textures/brick.png");
  REQUIRE(brick);
  CHECK(brick->type == AssetType::Texture);
  CHECK(brick->guid == metaGuid(p, "assets/textures/brick.png"));
  CHECK(brick->settings["color_space"] == "srgb");
  CHECK(db.findByPath("assets/textures/brick_normal.png")->settings["normal_map"] == true);
  CHECK(db.findByPath("assets/textures/brick_rough.png")->settings["color_space"] == "linear");
  CHECK(db.isStale(brick->guid));

  const assets::ImportReport r = db.importAll();
  CHECK(r.imported.size() == 3);
  CHECK(r.failed.empty());
  CHECK_FALSE(db.isStale(brick->guid));
  REQUIRE(brick->artifacts.size() == 1);
  CHECK(brick->artifacts[0].guid == brick->guid);
  assets::CookedTexture t;
  std::string err;
  REQUIRE(assets::loadKtx2File(db.artifactPath(brick->guid), t, err));
  CHECK(t.format == backend::TextureFormat::RGBA8Srgb);
  CHECK(t.mips.size() == 3);
  REQUIRE(assets::loadKtx2File(
      db.artifactPath(db.findByPath("assets/textures/brick_normal.png")->guid), t, err));
  CHECK(t.format == backend::TextureFormat::RGBA8Unorm);

  // A new session: same identities, nothing to do.
  const core::Guid g = brick->guid;
  AssetDatabase again = openDb(p, &scan);
  CHECK(scan.createdMetas.empty());
  CHECK(again.findByPath("assets/textures/brick.png")->guid == g);
  const assets::ImportReport r2 = again.importAll();
  CHECK(r2.imported.empty());
  CHECK(r2.upToDate == 3);
}

TEST_CASE("asset db: staleness follows content, settings and importer version", "[assetdb]") {
  Project p("stale");
  p.write("assets/a.png", png(10));
  AssetDatabase db = openDb(p);
  db.importAll();
  const core::Guid g = db.findByPath("assets/a.png")->guid;
  REQUIRE_FALSE(db.isStale(g));

  // Rewritten with identical bytes: the mtime moved, the content did
  // not - still current.
  p.write("assets/a.png", png(10));
  CHECK_FALSE(db.isStale(g));
  // New content (same size): stale.
  p.write("assets/a.png", png(11));
  CHECK(db.isStale(g));
  CHECK(db.importAll().imported.size() == 1);
  CHECK_FALSE(db.isStale(g));

  // Settings: validated, written to the .meta, stale until re-imported.
  std::string err;
  Json s = db.find(g)->settings;
  s["color_space"] = "linear";
  REQUIRE(db.setSettings(g, s, &err));
  CHECK(db.isStale(g));
  CHECK(p.read("assets/a.png.meta").find("\"linear\"") != std::string::npos);
  db.importAll();
  assets::CookedTexture t;
  REQUIRE(assets::loadKtx2File(db.artifactPath(g), t, err));
  CHECK(t.format == backend::TextureFormat::RGBA8Unorm);
  Json bad = s;
  bad["colour_space"] = "srgb";
  CHECK_FALSE(db.setSettings(g, bad, &err));
  CHECK(err.find("unknown setting \"colour_space\"") != std::string::npos);
  bad = s;
  bad["max_size"] = -5;
  CHECK_FALSE(db.setSettings(g, bad, &err));
  CHECK(p.read("assets/a.png.meta").find("colour_space") == std::string::npos);

  // An engine update (importer version) re-imports everything.
  AssetDatabase upgraded = openDb(p, nullptr, 1);
  CHECK(upgraded.isStale(g));
  CHECK(upgraded.importAll().imported.size() == 1);
}

TEST_CASE("asset db: a deleted cache re-imports byte-identically", "[assetdb]") {
  Project p("determinism");
  p.write("assets/a.png", png(70, 9, 5));
  testgltf::ReferenceModel ref;
  for (const char* f : {"model.gltf", "model.bin", "tex one.png"})
    p.write(std::string("assets/models/") + f, [&] {
      std::string b;
      REQUIRE(util::readFile(ref.dir.path / util::pathFromUtf8(f), b));
      return b;
    }());
  AssetDatabase db = openDb(p);
  REQUIRE(db.importAll().failed.empty());
  const auto before = cacheHashes(db);
  CHECK(before.size() > 8);
  fs::remove_all(db.cacheDir());
  for (const auto* a : db.assets()) CHECK(db.isStale(a->guid));
  REQUIRE(db.importAll().failed.empty());
  CHECK(cacheHashes(db) == before);  // records included: same stamps, same bytes
}

// ── Repairs ────────────────────────────────────────────────────

TEST_CASE("asset db: moving with or without the .meta keeps the identity", "[assetdb]") {
  Project p("moves");
  p.write("assets/a.png", png(30));
  AssetDatabase db = openDb(p);
  db.importAll();
  const core::Guid g = db.findByPath("assets/a.png")->guid;

  // With its .meta: a plain move.
  fs::create_directories(p.at("assets/sub"));
  fs::rename(p.at("assets/a.png"), p.at("assets/sub/b.png"));
  fs::rename(p.at("assets/a.png.meta"), p.at("assets/sub/b.png.meta"));
  assets::ScanReport scan = db.refresh();
  CHECK(scan.createdMetas.empty());
  REQUIRE(db.findByPath("assets/sub/b.png"));
  CHECK(db.findByPath("assets/sub/b.png")->guid == g);
  CHECK_FALSE(db.isStale(g));  // same content: the artifacts still hold
  CHECK(fs::exists(db.artifactPath(g)));

  // Without it (the .meta left behind): adopted by content.
  fs::rename(p.at("assets/sub/b.png"), p.at("assets/c.png"));
  scan = db.refresh();
  CHECK(contains(scan.adoptedMoves, "assets/sub/b.png -> assets/c.png"));
  CHECK(scan.createdMetas.empty());
  CHECK(db.findByPath("assets/c.png")->guid == g);
  CHECK_FALSE(fs::exists(p.at("assets/sub/b.png.meta")));
  CHECK(fs::exists(p.at("assets/c.png.meta")));

  // A different file appearing where an orphan cannot match: new asset.
  fs::remove(p.at("assets/c.png"));
  p.write("assets/d.png", png(31));
  scan = db.refresh();
  CHECK(contains(scan.createdMetas, "assets/d.png"));
  CHECK(contains(scan.orphanedMetas, "assets/c.png.meta"));
  CHECK(db.findByPath("assets/d.png")->guid != g);
}

TEST_CASE("asset db: a copied .meta gets a fresh guid", "[assetdb]") {
  Project p("copies");
  p.write("assets/a.png", png(40));
  AssetDatabase db = openDb(p);
  db.importAll();
  const core::Guid g = db.findByPath("assets/a.png")->guid;
  fs::copy_file(p.at("assets/a.png"), p.at("assets/a_copy.png"));
  fs::copy_file(p.at("assets/a.png.meta"), p.at("assets/a_copy.png.meta"));
  const assets::ScanReport scan = db.refresh();
  CHECK(contains(scan.reissuedGuids, "assets/a_copy.png"));
  CHECK(db.findByPath("assets/a.png")->guid == g);  // the imported original keeps it
  const core::Guid copy = db.findByPath("assets/a_copy.png")->guid;
  CHECK(copy != g);
  CHECK(metaGuid(p, "assets/a_copy.png") == copy);
  CHECK(db.refresh().reissuedGuids.empty());  // repaired for good
}

TEST_CASE("asset db: corrupt metas are reported and never overwritten", "[assetdb]") {
  Project p("corrupt");
  p.write("assets/a.png", png(50));
  p.write("assets/a.png.meta", "{ this is not json");
  p.write("assets/b.png", png(51));
  p.write("assets/b.png.meta", R"({"format": "kenshin.meta", "version": 1,
    "guid": "00000000-0000-4000-8000-000000000001", "type": "model", "settings": {}})");
  p.write("assets/c.png", png(52));
  p.write("assets/c.png.meta", R"({"format": "kenshin.meta", "version": 1,
    "guid": "00000000-0000-4000-8000-000000000002", "type": "texture",
    "settings": {"color_space": "purple"}})");
  assets::ScanReport scan;
  AssetDatabase db = openDb(p, &scan);
  REQUIRE(scan.corruptMetas.size() == 3);
  CHECK(contains(scan.corruptMetas, "assets/a.png.meta: "));
  CHECK(contains(scan.corruptMetas, "parse error"));
  CHECK(contains(scan.corruptMetas, "does not match the file"));
  CHECK(contains(scan.corruptMetas, "color_space"));
  CHECK(db.assets().empty());
  CHECK(p.read("assets/a.png.meta") == "{ this is not json");  // left for the user to fix
}

TEST_CASE("asset db: orphans and garbage collection", "[assetdb]") {
  Project p("gc");
  p.write("assets/a.png", png(60));
  p.write("assets/b.png", png(61));
  AssetDatabase db = openDb(p);
  db.importAll();
  const core::Guid a = db.findByPath("assets/a.png")->guid;
  const core::Guid b = db.findByPath("assets/b.png")->guid;
  fs::remove(p.at("assets/a.png"));
  const assets::ScanReport scan = db.refresh();
  CHECK(contains(scan.orphanedMetas, "assets/a.png.meta"));
  CHECK(db.find(a) == nullptr);
  CHECK(fs::exists(p.at("assets/a.png.meta")));  // kept: the user may restore the file
  fs::create_directories(db.cacheDir() / "leftover.tmp");
  CHECK(db.collectGarbage() == 2);  // a's cache + the stale temp folder
  CHECK_FALSE(fs::exists(db.cacheDir() / a.toString()));
  CHECK(fs::exists(db.artifactPath(b)));
}

// ── Models ─────────────────────────────────────────────────────

TEST_CASE("asset db: model sub-assets with derived guids", "[assetdb]") {
  Project p("model");
  testgltf::ReferenceModel ref;
  for (const char* f : {"model.gltf", "model.bin", "tex one.png"}) {
    std::string b;
    REQUIRE(util::readFile(ref.dir.path / util::pathFromUtf8(f), b));
    p.write(std::string("assets/models/") + f, b);
  }
  AssetDatabase db = openDb(p);
  const assets::ImportReport r = db.importAll();
  INFO((r.failed.empty() ? std::string() : r.failed[0]));
  REQUIRE(r.failed.empty());
  const assets::AssetRecord* model = db.findByPath("assets/models/model.gltf");
  REQUIRE(model);
  CHECK(model->type == AssetType::Model);
  CHECK(db.findByPath("assets/models/tex one.png"));  // the image file is also its own asset
  std::map<std::string, ArtifactType> keys;
  for (const auto& a : model->artifacts) {
    keys[a.key] = a.type;
    CHECK(a.guid == AssetDatabase::subAssetGuid(model->guid, a.key));
    CHECK(fs::exists(db.artifactPath(a.guid)));
    CHECK(db.resolve(a.guid)->asset == model);
  }
  CHECK(keys == std::map<std::string, ArtifactType>{
                    {"material/0", ArtifactType::Material},
                    {"material/1", ArtifactType::Material},
                    {"mesh/0", ArtifactType::Mesh},
                    {"mesh/0#translucent", ArtifactType::TranslucentMesh},
                    {"mesh/1", ArtifactType::Mesh},
                    {"prefab", ArtifactType::Prefab},
                    {"texture/0.linear", ArtifactType::Texture},
                    {"texture/0.srgb", ArtifactType::Texture},
                    {"texture/1.normal", ArtifactType::Texture}});
  CHECK(contains(model->notes, "TEXCOORD_1"));

  // The prefab's mesh references and the materials' texture references
  // all resolve inside the database.
  Json prefab;
  REQUIRE(serialization::parseJson(
      [&] {
        std::string s;
        REQUIRE(util::readFile(db.artifactPath(AssetDatabase::subAssetGuid(model->guid, "prefab")), s));
        return s;
      }(),
      prefab, nullptr));
  int meshRefs = 0;
  for (const Json& e : prefab["entities"])
    if (e["components"].contains("mesh_renderer")) {
      const auto g = core::Guid::parse(e["components"]["mesh_renderer"]["mesh"].get<std::string>());
      REQUIRE(g);
      const auto res = db.resolve(*g);
      REQUIRE(res);
      CHECK(res->artifact->type == ArtifactType::Mesh);
      ++meshRefs;
    }
  CHECK(meshRefs == 3);
  assets::MaterialDesc red;
  std::string text, err;
  REQUIRE(util::readFile(db.artifactPath(AssetDatabase::subAssetGuid(model->guid, "material/0")), text));
  REQUIRE(assets::loadMaterial(text, red, &err));
  CHECK(db.resolve(red.baseColorTexture.texture)->artifact->key == "texture/0.srgb");
  CHECK(db.resolve(red.normalTexture.texture)->artifact->key == "texture/1.normal");
  CHECK(red.baseColorTexture.sampler.filter == backend::FilterMode::Nearest);

  // 1.3d: a mesh names its slot materials (the Quad's two primitives),
  // the translucent part the same list; every artifact carries the
  // SHA-1 of its file; both survive a new session.
  const auto artifact = [&](const AssetDatabase& d, const std::string& key) {
    return d.resolve(AssetDatabase::subAssetGuid(model->guid, key))->artifact;
  };
  const std::vector<core::Guid> quadSlots = {AssetDatabase::subAssetGuid(model->guid, "material/0"),
                                             AssetDatabase::subAssetGuid(model->guid, "material/1")};
  CHECK(artifact(db, "mesh/0")->materials == quadSlots);
  CHECK(artifact(db, "mesh/0#translucent")->materials == quadSlots);
  CHECK(artifact(db, "material/0")->materials.empty());
  for (const auto& a : model->artifacts) {
    std::string bytes;
    REQUIRE(util::readFile(db.artifactPath(a.guid), bytes));
    CHECK(a.sha1 == core::Sha1::hex(core::Sha1::of(bytes)));
  }
  {
    AssetDatabase again = openDb(p);
    CHECK(artifact(again, "mesh/0")->materials == quadSlots);
    CHECK(artifact(again, "texture/0.srgb")->sha1 == artifact(db, "texture/0.srgb")->sha1);
    CHECK_FALSE(artifact(again, "prefab")->sha1.empty());
  }

  // Dependencies: the .bin and the image file are part of the stamp.
  CHECK_FALSE(db.isStale(model->guid));
  std::string bin = p.read("assets/models/model.bin");
  bin.back() ^= 1;
  p.write("assets/models/model.bin", bin);
  CHECK(db.isStale(model->guid));
}

TEST_CASE("asset db: a re-import never half-replaces artifacts in use", "[assetdb]") {
  // The swap renames the old cache folder aside before the new one moves
  // in. Where an open file blocks that (Windows), the import fails with
  // every old artifact intact; elsewhere it succeeds and the open file
  // keeps reading the old bytes. Never a mix of the two.
  Project p("inuse");
  p.write("assets/a.png", png(10));
  AssetDatabase db = openDb(p);
  REQUIRE(db.importAll().failed.empty());
  const core::Guid g = db.findByPath("assets/a.png")->guid;
  const std::map<std::string, std::string> before = cacheHashes(db);
  p.write("assets/a.png", png(240));
  bool ok = false;
  std::string err;
  {
    std::FILE* held = std::fopen(util::pathToUtf8(db.artifactPath(g)).c_str(), "rb");
    REQUIRE(held);
    ok = db.reimport(g, &err);
    std::fclose(held);
  }
  INFO(err);
  if (!ok) {
    CHECK(err.find("in use") != std::string::npos);
    CHECK(cacheHashes(db) == before);  // the old import, complete
    CHECK(db.isStale(g));
    REQUIRE(db.reimport(g, &err));  // released: now it goes through
  }
  CHECK(cacheHashes(db) != before);
  CHECK_FALSE(db.isStale(g));
  for (const auto& e : fs::directory_iterator(db.cacheDir()))
    CHECK(e.path().filename().string().find('.') == std::string::npos);  // no .tmp / .old left
}

TEST_CASE("asset db: OBJ models track their MTL and texture", "[assetdb]") {
  Project p("obj");
  p.write("assets/box.obj",
          "mtllib box.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nusemtl m\nf 1/1 2/2 3/3\n");
  p.write("assets/box.mtl", "newmtl m\nKd 0.5 0.25 1\nmap_Kd box_albedo.png\n");
  p.write("assets/box_albedo.png", png(90));
  AssetDatabase db = openDb(p);
  REQUIRE(db.importAll().failed.empty());
  const assets::AssetRecord* box = db.findByPath("assets/box.obj");
  REQUIRE(box);
  std::map<std::string, ArtifactType> keys;
  for (const auto& a : box->artifacts) keys[a.key] = a.type;
  CHECK(keys.count("mesh/0"));
  CHECK(keys.count("material/0"));
  CHECK(keys.count("texture/0.srgb"));
  CHECK(keys.count("prefab"));
  assets::MaterialDesc m;
  std::string text, err;
  REQUIRE(util::readFile(db.artifactPath(AssetDatabase::subAssetGuid(box->guid, "material/0")), text));
  REQUIRE(assets::loadMaterial(text, m, &err));
  CHECK(m.baseColor == glm::vec4(0.5f, 0.25f, 1.0f, 1.0f));
  CHECK_FALSE(db.isStale(box->guid));
  p.write("assets/box.mtl", "newmtl m\nKd 1 1 1\nmap_Kd box_albedo.png\n");
  CHECK(db.isStale(box->guid));  // the MTL is a dependency
}

// ── Validation ─────────────────────────────────────────────────

TEST_CASE("asset db: materials, scenes and prefabs are validated", "[assetdb]") {
  Project p("validate");
  p.write("assets/good.kmat", assets::saveMaterial({}));
  p.write("assets/bad.kmat", R"({"format": "kenshin.material", "version": 1, "roughness": 2})");
  p.write("assets/level.kscene", R"({"format": "kenshin.scene", "version": 1, "entities": [
    {"name": "x", "components": {"light": {"intensity": 3}}}]})");
  p.write("assets/broken.kscene", R"({"format": "kenshin.scene", "version": 1, "entities": [
    {"name": "x", "components": {"lihgt": {}}}]})");
  AssetDatabase db = openDb(p);
  const assets::ImportReport r = db.importAll();
  CHECK(r.imported.size() == 2);
  REQUIRE(r.failed.size() == 2);
  CHECK(contains(r.failed, "assets/bad.kmat: roughness must be a number in [0, 1]"));
  CHECK(contains(r.failed, "unknown component \"lihgt\""));
  CHECK(db.findByPath("assets/bad.kmat")->error.find("roughness") != std::string::npos);
  CHECK_FALSE(db.findByPath("assets/bad.kmat")->imported);
  CHECK(db.findByPath("assets/level.kscene")->imported);

  // A failed re-import keeps the previous good artifacts.
  const core::Guid good = db.findByPath("assets/good.kmat")->guid;
  const std::string goodArtifact = [&] {
    std::string s;
    REQUIRE(util::readFile(db.artifactPath(good), s));
    return s;
  }();
  p.write("assets/good.kmat", "{ broken");
  std::string err;
  CHECK_FALSE(db.reimport(good, &err));
  std::string now;
  REQUIRE(util::readFile(db.artifactPath(good), now));
  CHECK(now == goodArtifact);

  // Without a component registry scenes cannot be validated.
  AssetDatabase noReg;
  REQUIRE(noReg.open(p.dir.path, {}, &err));
  noReg.importAll(true);
  CHECK(noReg.findByPath("assets/level.kscene")->error.find("component registry") !=
        std::string::npos);
}

TEST_CASE("material files: canonical round trip and strict loading", "[assetdb]") {
  assets::MaterialDesc m;
  m.baseColor = {0.5f, 0.25f, 1.0f, 0.75f};
  m.roughness = 0.3f;
  m.alphaMode = assets::MaterialDesc::AlphaMode::Mask;
  m.baseColorTexture.texture = core::Guid::generate();
  m.baseColorTexture.sampler = {backend::WrapMode::Clamp, backend::WrapMode::Mirror,
                                backend::FilterMode::Nearest};
  const std::string text = assets::saveMaterial(m);
  assets::MaterialDesc back;
  std::string err;
  REQUIRE(assets::loadMaterial(text, back, &err));
  CHECK(back == m);
  CHECK(assets::saveMaterial(back) == text);
  const std::pair<std::string, std::string> bad[] = {
      {R"({"format": "kenshin.material", "version": 1, "metalic": 1})", "unknown key \"metalic\""},
      {R"({"format": "kenshin.material", "version": 1, "alpha_mode": "glass"})", "alpha_mode"},
      {R"({"format": "kenshin.material", "version": 1, "textures": {"albedo": {}}})",
       "unknown texture slot \"albedo\""},
      {R"({"format": "kenshin.material", "version": 1, "textures": {"normal": {"filter": "linear"}}})",
       "needs a \"texture\" guid"},
      {R"({"format": "kenshin.material", "version": 1, "base_color": [1, 1, 1]})", "base_color"},
      {R"({"format": "kenshin.material", "version": 2})", "newer than"},
  };
  for (const auto& [t, expected] : bad) {
    INFO(t);
    assets::MaterialDesc keep;
    keep.roughness = 0.9f;
    CHECK_FALSE(assets::loadMaterial(t, keep, &err));
    INFO(err);
    CHECK(err.find(expected) != std::string::npos);
    CHECK(keep.roughness == 0.9f);
  }
}

TEST_CASE("asset db: a project without a project file is refused", "[assetdb]") {
  testgltf::TempDir dir("kenshin_assetdb_noproject");
  AssetDatabase db;
  std::string err;
  CHECK_FALSE(db.open(dir.path, {}, &err));
  CHECK(err.find("cannot read") != std::string::npos);
}

TEST_CASE("asset db: the real test model imports and resolves", "[assetdb][assets]") {
  const fs::path src = fs::path(KENSHIN_SOURCE_DIR) / "tests/assets/model/vultures_helmet_mcu_version";
  if (!fs::exists(src / "scene.gltf")) SKIP("tests/assets/model not present (untracked test assets)");
  Project p("vulture");
  fs::copy(src, p.at("assets/vulture"), fs::copy_options::recursive);
  AssetDatabase db = openDb(p);
  const assets::ImportReport r = db.importAll();
  INFO((r.failed.empty() ? std::string() : r.failed[0]));
  REQUIRE(r.failed.empty());
  const assets::AssetRecord* model = db.findByPath("assets/vulture/scene.gltf");
  REQUIRE(model);
  usize meshes = 0, textures = 0;
  for (const auto& a : model->artifacts) {
    meshes += a.type == ArtifactType::Mesh;
    textures += a.type == ArtifactType::Texture;
    CHECK(fs::file_size(db.artifactPath(a.guid)) > 0);
  }
  CHECK(meshes >= 1);
  CHECK(textures >= 3);
  // Every texture a material names is a sub-asset of the model.
  for (const auto& a : model->artifacts) {
    if (a.type != ArtifactType::Material) continue;
    std::string text, err;
    REQUIRE(util::readFile(db.artifactPath(a.guid), text));
    assets::MaterialDesc m;
    REQUIRE(assets::loadMaterial(text, m, &err));
    for (const auto* slot : {&m.baseColorTexture, &m.metallicRoughnessTexture, &m.normalTexture,
                             &m.occlusionTexture, &m.emissiveTexture})
      if (slot->valid()) CHECK(db.resolve(slot->texture)->artifact->type == ArtifactType::Texture);
  }
  CHECK(db.importAll().imported.empty());  // nothing stale right after
}
