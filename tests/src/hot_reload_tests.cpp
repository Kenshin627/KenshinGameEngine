// Runtime assets by Guid and hot reload (roadmap 1.3d): the polling
// file watcher, the AssetManager (reference counts, in-place texture
// replacement probed on the GPU, material and mesh reloads, failed and
// no-op re-imports) and the WorldRenderer's static build (instances,
// slot materials, overrides, record updates, reference hand-over) on a
// temporary copy of tests/data/project.

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <string>
#include <thread>

#include "kenshin/assets/AssetDatabase.h"
#include "kenshin/assets/TextureCooker.h"
#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/FileWatcher.h"
#include "kenshin/ecs/Components.h"
#include "kenshin/manager/AssetManager.h"
#include "kenshin/manager/ShaderManager.h"
#include "kenshin/renderer/VirtualGeometryScene.h"
#include "kenshin/renderer/WorldRenderer.h"
#include "kenshin/scene/SceneAssets.h"
#include "kenshin/util/File.h"
#include "test_bindless_probe.h"
#include "test_gltf.h"
#include "test_png.h"

using namespace kenshin;
namespace fs = std::filesystem;
using assets::AssetDatabase;
using core::FileWatcher;
using manager::AssetManager;

namespace {

using Kind = FileWatcher::ChangeKind;

void put(const fs::path& p, const std::string& bytes) {
  fs::create_directories(p.parent_path());
  REQUIRE(util::writeFileAtomic(p, bytes));
}

std::string changesOf(FileWatcher& w) {
  std::string s;
  for (const auto& c : w.poll())
    s += (c.kind == Kind::Added ? "+" : c.kind == Kind::Removed ? "-" : "~") + c.path + " ";
  return s;
}

/// tests/data/project in a temp folder, opened and imported.
struct Fixture {
  testgltf::TempDir dir;
  ecs::ComponentRegistry registry;
  AssetDatabase db;

  explicit Fixture(const std::string& name) : dir("kenshin_hotreload_" + name) {
    fs::copy(KENSHIN_SOURCE_DIR "/tests/data/project", dir.path, fs::copy_options::recursive);
    ecs::registerBuiltinComponents(registry);
    assets::AssetDatabaseOptions o;
    o.registry = &registry;
    std::string err;
    INFO(err);
    REQUIRE(db.open(dir.path, o, &err));
    REQUIRE(db.importAll().failed.empty());
  }
  fs::path at(const std::string& rel) const { return dir.path / util::pathFromUtf8(rel); }
  core::Guid guid(const std::string& rel) const {
    const assets::AssetRecord* r = db.findByPath(rel);
    REQUIRE(r);
    return r->guid;
  }
  core::Guid sub(const std::string& rel, const std::string& key) const {
    return AssetDatabase::subAssetGuid(guid(rel), key);
  }
  void edit(const std::string& rel, const std::string& bytes) const { put(at(rel), bytes); }
};

std::string png(u8 r, u8 g, u8 b) {
  std::vector<u32> px;
  for (int i = 0; i < 16 * 16; ++i) px.insert(px.end(), {r, g, b});
  const std::vector<u8> bytes = testpng::makePng(16, 16, 8, 2, px);
  return std::string(bytes.begin(), bytes.end());
}

glm::vec4 srgb(u8 r, u8 g, u8 b) {
  return glm::vec4(assets::srgbToLinear(r / 255.0f), assets::srgbToLinear(g / 255.0f),
                   assets::srgbToLinear(b / 255.0f), 1.0f);
}

bool sameColour(const glm::vec4& a, const glm::vec4& b) {
  return glm::all(glm::lessThanEqual(glm::abs(a - b), glm::vec4(2e-3f)));
}

const backend::SamplerDesc kLinear{};

/// One settled watcher cycle (pollInterval 0): the edit's poll, the
/// quiet poll, the plan and the reload.
manager::AssetReload cycle(AssetManager& am, renderer::WorldRenderer* wr = nullptr) {
  CHECK_FALSE(am.update(0.0f));  // sees the change, waits for quiet
  REQUIRE(am.update(0.0f));
  const manager::ReloadPlan plan = am.planReload();
  if (wr && !plan.residentMeshes.empty()) wr->closeFiles();
  return am.applyReload(plan);
}

bool has(const std::vector<core::Guid>& v, const core::Guid& g) {
  return std::find(v.begin(), v.end(), g) != v.end();
}

}  // namespace

// ── FileWatcher ────────────────────────────────────────────────

TEST_CASE("file watcher: adds, edits, removals, sorted and recursive", "[filewatcher]") {
  testgltf::TempDir dir("kenshin_filewatcher");
  put(dir.path / "a.txt", "one");
  put(dir.path / "sub/deep/b.txt", "two");
  FileWatcher w;
  w.watch(dir.path);
  CHECK(w.fileCount() == 2);
  CHECK(changesOf(w).empty());  // the baseline

  put(dir.path / "sub/c.txt", "three");
  put(dir.path / "a.txt", "one!");  // size changes
  CHECK(changesOf(w) == "~a.txt +sub/c.txt ");
  CHECK(changesOf(w).empty());  // reported once

  // Same size, new mtime (a same-length save).
  fs::last_write_time(dir.path / "a.txt",
                      fs::last_write_time(dir.path / "a.txt") + std::chrono::seconds(2));
  fs::remove(dir.path / "sub/deep/b.txt");
  CHECK(changesOf(w) == "~a.txt -sub/deep/b.txt ");
  CHECK(w.fileCount() == 2);
}

TEST_CASE("file watcher: a missing root watches nothing until it appears", "[filewatcher]") {
  testgltf::TempDir dir("kenshin_filewatcher_root");
  FileWatcher w;
  w.watch(dir.path / "later");
  CHECK(changesOf(w).empty());
  put(dir.path / "later/x.bin", "x");
  CHECK(changesOf(w) == "+x.bin ");
  fs::remove_all(dir.path / "later");
  CHECK(changesOf(w) == "-x.bin ");
}

// ── AssetManager ───────────────────────────────────────────────

TEST_CASE("asset manager: textures by guid, reference counts, in-place reload",
          "[gpu][hotreload]") {
  Fixture f("texture");
  auto be = backend::createVulkanBackend();
  REQUIRE(be->initialize({}));
  {
    testprobe::Prober prober(*be);
    AssetManager am;
    am.init(*be, f.db, {0.0f, true});
    const u32 baseline = be->bindlessUsed();
    const core::Guid swatch = f.guid("assets/textures/swatch.png");
    const u32 h = am.acquireTexture(swatch);
    REQUIRE(h != backend::IRenderBackend::kInvalidTextureHandle);
    CHECK(am.acquireTexture(swatch) == h);  // shared
    CHECK(am.references(swatch) == 2);
    CHECK(be->bindlessUsed() == baseline + 1);
    const u32 ref = am.textureRef(swatch, kLinear);
    const auto sample = [&] { return prober.run({{ref, 0.5f, 0.5f, testprobe::kMip0}})[0]; };
    CHECK(sameColour(sample(), srgb(200, 40, 30)));

    // Rewrite the source: re-imported, replaced in place - same handle,
    // same slot, so the reference sampled above sees the new texels.
    f.edit("assets/textures/swatch.png", png(40, 180, 60));
    const manager::AssetReload r = cycle(am);
    CHECK(r.reimported == std::vector<core::Guid>{swatch});
    CHECK(r.textures == std::vector<core::Guid>{swatch});
    CHECK(r.failed.empty());
    CHECK(am.textureRef(swatch, kLinear) == ref);
    CHECK(be->bindlessUsed() == baseline + 1);
    CHECK(sameColour(sample(), srgb(40, 180, 60)));

    // Same bytes again (a save without changes): the watcher fires on
    // the new mtime, the database's content hash says current - no
    // re-import. (A re-import whose cooked bytes come out identical
    // reloads nothing either: the model case in the world test.)
    f.edit("assets/textures/swatch.png", png(40, 180, 60));
    CHECK_FALSE(am.update(0.0f));
    REQUIRE(am.update(0.0f));
    CHECK(am.planReload().empty());

    // A broken file: the re-import fails, the loaded texture stays.
    f.edit("assets/textures/swatch.png", "not a png");
    const manager::AssetReload bad = cycle(am);
    CHECK(bad.failed.size() == 1);
    CHECK(bad.textures.empty());
    CHECK(sameColour(sample(), srgb(40, 180, 60)));

    // The last release frees the slot.
    am.releaseTexture(swatch);
    CHECK(am.residentTextures() == 1);
    am.releaseTexture(swatch);
    CHECK(am.residentTextures() == 0);
    CHECK(be->bindlessUsed() == baseline);
    CHECK(am.acquireTexture(f.guid("assets/materials/painted.kmat")) ==
          backend::IRenderBackend::kInvalidTextureHandle);  // not a texture
    am.shutdown();
  }
  be->shutdown();
}

TEST_CASE("asset manager: update() settles; untouched assets never reload", "[gpu][hotreload]") {
  Fixture f("settle");
  auto be = backend::createVulkanBackend();
  REQUIRE(be->initialize({}));
  {
    AssetManager am;
    am.init(*be, f.db, {0.5f, true});
    CHECK_FALSE(am.update(0.1f));  // nothing changed
    f.edit("assets/textures/swatch.png", png(1, 2, 3));
    CHECK_FALSE(am.update(0.1f));  // not yet time to poll (0.5 s)
    CHECK_FALSE(am.update(0.5f));  // polled: changed, waiting for quiet
    f.edit("assets/textures/swatch.png", png(4, 5, 6));
    CHECK_FALSE(am.update(0.5f));  // still changing
    CHECK(am.update(0.5f));        // quiet: ready
    CHECK_FALSE(am.update(0.5f));  // reported once
    const manager::ReloadPlan plan = am.planReload();
    CHECK(plan.assets == std::vector<core::Guid>{f.guid("assets/textures/swatch.png")});
    CHECK(plan.residentMeshes.empty());
    // Nothing resident: the re-import runs, nothing reloads.
    const manager::AssetReload r = am.applyReload(plan);
    CHECK(r.reimported.size() == 1);
    CHECK(r.textures.empty());
    am.shutdown();
  }
  be->shutdown();
}

TEST_CASE("asset manager: materials hold their textures; a reload swaps them", "[gpu][hotreload]") {
  Fixture f("material");
  auto be = backend::createVulkanBackend();
  REQUIRE(be->initialize({}));
  {
    AssetManager am;
    am.init(*be, f.db, {0.0f, true});
    const core::Guid painted = f.guid("assets/materials/painted.kmat");
    const core::Guid swatch = f.guid("assets/textures/swatch.png");
    const core::Guid checker = f.guid("assets/models/checker.png");
    const assets::MaterialDesc* m = am.acquireMaterial(painted);
    REQUIRE(m);
    CHECK(m->roughness == 0.35f);
    CHECK(am.references(swatch) == 1);
    CHECK(am.acquireMaterial(painted) == m);
    CHECK(am.references(painted) == 2);
    CHECK(am.references(swatch) == 1);  // per material, not per reference

    // Point the material at the checker: the new texture loads, the old
    // one is released.
    std::string text;
    REQUIRE(util::readFile(f.at("assets/materials/painted.kmat"), text));
    const auto pos = text.find(swatch.toString());
    REQUIRE(pos != std::string::npos);
    text.replace(pos, 36, checker.toString());
    f.edit("assets/materials/painted.kmat", text);
    const manager::AssetReload r = cycle(am);
    CHECK(r.materials == std::vector<core::Guid>{painted});
    const assets::MaterialDesc* m2 = am.acquireMaterial(painted);
    REQUIRE(m2);
    CHECK(m2->baseColorTexture.texture == checker);
    CHECK(am.references(swatch) == 0);
    CHECK(am.references(checker) == 1);
    for (int i = 0; i < 3; ++i) am.releaseMaterial(painted);
    CHECK(am.residentMaterials() == 0);
    CHECK(am.residentTextures() == 0);
    am.shutdown();
  }
  be->shutdown();
}

// ── WorldRenderer ──────────────────────────────────────────────

TEST_CASE("world renderer: a project scene's instances, slots and overrides", "[gpu][hotreload]") {
  Fixture f("world");
  auto be = backend::createVulkanBackend();
  REQUIRE(be->initialize({}));
  {
    manager::ShaderManager shaders;
    REQUIRE(shaders.init(testprobe::kShaderRoot));
    AssetManager am;
    am.init(*be, f.db, {0.0f, true});
    renderer::WorldRenderer wr;
    wr.init(*be, shaders, am);

    ecs::World world;
    std::string err;
    INFO(err);
    REQUIRE(scene::loadWorldAsset(f.db, f.guid("assets/scenes/main.kscene"), f.registry, world, &err));
    // Skipped: an invisible renderer, and one whose mesh does not exist.
    const ecs::Entity hidden = world.create("Hidden");
    world.add<ecs::MeshRenderer>(hidden).mesh = f.sub("assets/models/spheres.gltf", "mesh/0");
    world.get<ecs::MeshRenderer>(hidden).visible = false;
    world.add<ecs::MeshRenderer>(world.create("Missing")).mesh = core::Guid::generate();

    REQUIRE(wr.build(world, &err));
    CHECK(wr.instanceCount() == 3);
    CHECK(wr.meshCount() == 2);
    // Ground slot, sphere slot, and one override block (the sphere's one
    // slot) for Sphere B.
    REQUIRE(wr.materialRecordCount() == 3);
    const core::Guid sphere = f.sub("assets/models/spheres.gltf", "mesh/0");
    const core::Guid painted = f.guid("assets/materials/painted.kmat");
    const core::Guid swatch = f.guid("assets/textures/swatch.png");
    CHECK(am.references(sphere) == 1);
    CHECK(am.references(painted) == 1);
    CHECK(am.references(swatch) == 1);
    const auto record = [&](u32 i) {
      const auto d = wr.scene()->materialData();
      return std::vector<f32>(d.begin() + i * 12, d.begin() + (i + 1) * 12);
    };
    const auto bits = [](f32 v) { return std::bit_cast<u32>(v); };
    std::vector<std::vector<f32>> recs = {record(0), record(1), record(2)};
    // The override record: painted.kmat's factors and the swatch.
    const auto over = std::find_if(recs.begin(), recs.end(), [](const auto& r) { return r[4] == 0.35f; });
    REQUIRE(over != recs.end());
    CHECK((*over)[0] == 1.0f);
    CHECK(bits((*over)[3]) == am.textureRef(swatch, kLinear));
    // The sphere's own slot: the model's checker material and texture.
    const auto own = std::find_if(recs.begin(), recs.end(), [](const auto& r) { return r[4] == 0.45f; });
    REQUIRE(own != recs.end());
    CHECK(bits((*own)[3]) ==
          am.textureRef(f.sub("assets/models/spheres.gltf", "texture/0.srgb"), kLinear));
    const auto ground = std::find_if(recs.begin(), recs.end(), [](const auto& r) { return r[4] == 0.85f; });
    REQUIRE(ground != recs.end());
    CHECK((*ground)[0] == 0.55f);
    CHECK(bits((*ground)[3]) == backend::kNoTexture);
    CHECK(wr.bounds().w > 2.0f);  // the 5 m ground dominates

    // A rebuild hands the references over (no reload, no leak).
    const usize textures = am.residentTextures();
    const usize loads = am.loads();
    REQUIRE(wr.build(world));
    CHECK(am.loads() == loads);  // nothing was released and read again
    CHECK(am.references(sphere) == 1);
    CHECK(am.references(swatch) == 1);
    CHECK(am.residentTextures() == textures);

    // A material edit rewrites only its records.
    std::string text;
    REQUIRE(util::readFile(f.at("assets/materials/painted.kmat"), text));
    const auto pos = text.find("\"roughness\": 0.35");
    REQUIRE(pos != std::string::npos);
    text.replace(pos, 17, "\"roughness\": 0.6");
    f.edit("assets/materials/painted.kmat", text);
    const usize before = am.loads();
    const manager::AssetReload mr = cycle(am, &wr);
    CHECK(mr.materials == std::vector<core::Guid>{painted});
    CHECK(am.loads() == before + 1);  // the material only: its texture stayed resident
    CHECK(wr.updateMaterials(mr.materials) == 1);
    recs = {record(0), record(1), record(2)};
    CHECK(std::count_if(recs.begin(), recs.end(), [](const auto& r) { return r[4] == 0.6f; }) == 1);
    CHECK(std::count_if(recs.begin(), recs.end(), [](const auto& r) { return r[4] == 0.35f; }) == 0);

    // A model setting change: its meshes' files must close first (the
    // re-import replaces the folder they live in), the cooked meshes
    // change (uncompressed pages), the scene rebuilds.
    serialization::Json settings = f.db.find(f.guid("assets/models/spheres.gltf"))->settings;
    settings["compress_pages"] = false;
    REQUIRE(f.db.setSettings(f.guid("assets/models/spheres.gltf"), settings, &err));
    CHECK_FALSE(am.update(0.0f));  // the .meta was written
    REQUIRE(am.update(0.0f));
    const manager::ReloadPlan plan = am.planReload();
    CHECK(plan.residentMeshes.size() == 2);
    wr.closeFiles();
    CHECK_FALSE(wr.ready());
    const manager::AssetReload model = am.applyReload(plan);
    CHECK(model.failed.empty());
    CHECK(model.meshes.size() == 2);
    CHECK(model.textures.empty());  // the model's texture cooked identically
    REQUIRE(wr.build(world));
    CHECK(wr.ready());
    CHECK(wr.instanceCount() == 3);

    wr.clear();
    CHECK(am.residentMeshes() == 0);
    CHECK(am.residentMaterials() == 0);
    CHECK(am.residentTextures() == 0);
    wr.shutdown();
    am.shutdown();
  }
  be->shutdown();
}
