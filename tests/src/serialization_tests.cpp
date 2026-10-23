// Serialization (roadmap step 1.2): canonical JSON, schema migration,
// scene/prefab/project files. Golden files live in
// tests/data/serialization; after an intentional format change,
// regenerate them with KENSHIN_UPDATE_GOLDEN=1 and review the diff
// (Catch2 runs tests in random order: when a golden file is new, a
// reader may run before its writer - run "[golden]" twice).

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <random>

#include "kenshin/scene/ProjectFile.h"
#include "kenshin/scene/SceneSerializer.h"
#include "kenshin/serialization/Json.h"
#include "kenshin/util/File.h"

using kenshin::f32;
using kenshin::i32;
using kenshin::u32;
using kenshin::usize;
using kenshin::core::Guid;
using namespace kenshin::ecs;
using kenshin::serialization::Json;
using kenshin::serialization::SchemaMigrator;
namespace sc = kenshin::scene;
namespace ser = kenshin::serialization;
namespace fs = std::filesystem;

namespace {

const fs::path kData = fs::path(KENSHIN_SOURCE_DIR) / "tests" / "data" / "serialization";

std::string readText(const fs::path& p) {
  std::string s;
  REQUIRE(kenshin::util::readFile(p, s));
  return s;
}

/// Byte comparison against a committed file (or regeneration).
void checkGolden(const std::string& name, const std::string& actual) {
  const fs::path p = kData / name;
  if (std::getenv("KENSHIN_UPDATE_GOLDEN")) {
    REQUIRE(kenshin::util::writeFileAtomic(p, actual));
    WARN("golden updated: " << name);
    return;
  }
  const std::string expected = readText(p);
  if (actual != expected) {
    INFO("actual:\n" << actual);
    FAIL("differs from golden " << name);
  }
}

Guid guid(const char* text) {
  const auto g = Guid::parse(text);
  REQUIRE(g);
  return *g;
}

ComponentRegistry makeRegistry() {
  ComponentRegistry r;
  registerBuiltinComponents(r);
  return r;
}

/// Fixed guids, every component type, nesting, awkward names.
void buildSampleWorld(World& w) {
  const Entity root = w.createWithGuid(guid("00000000-0000-4000-8000-0000000000a1"), "World Root");
  Transform& rt = w.get<Transform>(root);
  rt.position = {1, 2, 3};
  rt.rotation = glm::angleAxis(glm::radians(30.0f), glm::vec3(0, 0, 1));

  const Entity sun = w.createWithGuid(guid("00000000-0000-4000-8000-0000000000a2"), "Sun", root);
  w.get<Transform>(sun).rotation = glm::angleAxis(glm::radians(-45.0f), glm::vec3(1, 0, 0));
  Light& sl = w.add<Light>(sun);
  sl.type = Light::Type::Directional;
  sl.color = {1.0f, 0.95f, 0.8f};
  sl.intensity = 3.5f;
  sl.castShadows = true;

  const Entity cam = w.createWithGuid(guid("00000000-0000-4000-8000-0000000000a3"), "Camera", root);
  w.get<Transform>(cam).position = {0, -10, 2};
  Camera& c = w.add<Camera>(cam);
  c.fovYDeg = 70.0f;
  c.nearPlane = 0.05f;
  c.farPlane = 500.0f;
  c.primary = true;

  const Entity crate = w.createWithGuid(guid("00000000-0000-4000-8000-0000000000a4"),
                                        "Crate \"A\" / \xe7\xae\xb1\xe5\xad\x90", root);
  w.get<Transform>(crate).scale = {2, 1, 0.5f};
  MeshRenderer& mr = w.add<MeshRenderer>(crate);
  mr.mesh = guid("11111111-2222-4333-8444-555555555555");
  mr.castShadows = false;

  const Entity lamp = w.createWithGuid(guid("00000000-0000-4000-8000-0000000000a5"), "Lamp", crate);
  w.get<Transform>(lamp).position = {0, 0, 1.25f};
  Light& ll = w.add<Light>(lamp);
  ll.type = Light::Type::Spot;
  ll.color = {0.2f, 0.4f, 1.0f};
  ll.range = 12.5f;
  ll.innerConeDeg = 20.0f;
  ll.outerConeDeg = 35.0f;

  w.createWithGuid(guid("00000000-0000-4000-8000-0000000000a6"), "Empty");
}

/// Structure and every reflected value, compared exactly.
void requireSameWorld(const World& a, const World& b, const ComponentRegistry& r) {
  const std::vector<Entity> oa = a.preOrder();
  const std::vector<Entity> ob = b.preOrder();
  REQUIRE(oa.size() == ob.size());
  for (usize i = 0; i < oa.size(); ++i) {
    INFO("entity " << i << " " << a.get<Name>(oa[i]).value);
    REQUIRE(a.guid(oa[i]) == b.guid(ob[i]));
    REQUIRE(a.get<Name>(oa[i]).value == b.get<Name>(ob[i]).value);
    const Entity pa = a.parent(oa[i]);
    const Entity pb = b.parent(ob[i]);
    REQUIRE((pa == kNullEntity) == (pb == kNullEntity));
    if (pa != kNullEntity) REQUIRE(a.guid(pa) == b.guid(pb));
    for (const auto& info : r.all()) {
      REQUIRE(info->has(a, oa[i]) == info->has(b, ob[i]));
      if (!info->has(a, oa[i])) continue;
      for (const FieldInfo& f : info->fields) {
        INFO(info->name << "." << f.name);
        REQUIRE(*getField(a, oa[i], *info, f) == *getField(b, ob[i], *info, f));
      }
    }
  }
}

std::string sampleScene() {
  World w;
  buildSampleWorld(w);
  return sc::saveScene(w, makeRegistry());
}

/// v0 -> v1 for the fictional legacy layout ("objects", "translation").
Json renamedKey(const Json& obj, const std::string& from, const std::string& to) {
  Json out = Json::object();
  for (auto it = obj.begin(); it != obj.end(); ++it)
    out[it.key() == from ? to : it.key()] = it.value();
  return out;
}

SchemaMigrator legacyMigrator() {
  SchemaMigrator m(std::string(sc::kSceneFormat), sc::kSceneVersion);
  m.add(0, "objects -> entities, translation -> position", [](Json& doc, std::string& err) {
    if (!doc.contains("objects")) {
      err = "no objects";
      return false;
    }
    doc = renamedKey(doc, "objects", "entities");
    for (Json& e : doc["entities"])
      if (e.contains("components") && e["components"].contains("transform"))
        e["components"]["transform"] =
            renamedKey(e["components"]["transform"], "translation", "position");
    return true;
  });
  return m;
}

}  // namespace

// ── Canonical JSON ─────────────────────────────────────────────

TEST_CASE("json: floats print short and read back bit-exact", "[serialization]") {
  CHECK(ser::writeJson(ser::floatToJson(0.1f)) == "0.1\n");
  CHECK(ser::writeJson(ser::floatToJson(1.0f)) == "1.0\n");
  CHECK(ser::writeJson(ser::floatToJson(-0.0f)) == "-0.0\n");
  CHECK(ser::writeJson(ser::floatToJson(0.70710677f)) == "0.70710677\n");
  CHECK(ser::writeJson(ser::floatToJson(1e-7f)) == "1e-07\n");

  std::mt19937 rng(42);
  int checked = 0;
  for (int i = 0; i < 300000; ++i) {
    const u32 bits = rng();
    f32 v;
    std::memcpy(&v, &bits, sizeof(v));
    if (!std::isfinite(v)) continue;
    Json back;
    REQUIRE(ser::parseJson(ser::writeJson(ser::floatToJson(v)), back, nullptr));
    const f32 r = static_cast<f32>(back.get<double>());
    u32 rbits;
    std::memcpy(&rbits, &r, sizeof(r));
    INFO("bits " << bits);
    REQUIRE(rbits == bits);
    ++checked;
  }
  CHECK(checked > 290000);
}

TEST_CASE("json: canonical layout", "[serialization]") {
  Json doc = Json::object();
  doc["z_first"] = 1;
  doc["vec"] = Json::array({ser::floatToJson(1.5f), 2, 3});
  doc["long"] = Json::array({1, 2, 3, 4, 5});
  doc["nested"] = Json::array({Json::object({{"a", true}}), Json::array()});
  doc["empty"] = Json::object();
  doc["text"] = "quote\" slash\\ \xe4\xb8\xad newline\n";
  CHECK(ser::writeJson(doc) ==
        "{\n"
        "  \"z_first\": 1,\n"
        "  \"vec\": [1.5, 2, 3],\n"
        "  \"long\": [\n    1,\n    2,\n    3,\n    4,\n    5\n  ],\n"
        "  \"nested\": [\n    {\n      \"a\": true\n    },\n    []\n  ],\n"
        "  \"empty\": {},\n"
        "  \"text\": \"quote\\\" slash\\\\ \xe4\xb8\xad newline\\n\"\n"
        "}\n");
}

TEST_CASE("json: strict parsing", "[serialization]") {
  Json j;
  std::string err;
  CHECK(ser::parseJson("{\"a\": [1, 2]}", j, &err));
  CHECK(j["a"][1] == 2);
  CHECK_FALSE(ser::parseJson("{\"a\": 1,}", j, &err));
  CHECK(err.find("line 1") != std::string::npos);
  CHECK_FALSE(ser::parseJson("{\"a\": 1, \"a\": 2}", j, &err));
  CHECK(err == "duplicate key \"a\"");
  CHECK(ser::parseJson("[{\"a\": 1}, {\"a\": 2}]", j, &err));  // same key, different objects
  CHECK_FALSE(ser::parseJson("{} trailing", j, &err));
  CHECK_FALSE(ser::parseJson("// comment\n{}", j, &err));
  CHECK_FALSE(ser::parseJson(std::string(200, '[') + std::string(200, ']'), j, &err));
  CHECK(err.find("nesting deeper") != std::string::npos);
  CHECK(ser::parseJson(std::string(100, '[') + std::string(100, ']'), j, &err));

  // Large objects switch to a hashed key set at 32 keys: duplicates are
  // still caught whether the first copy is before or after the switch.
  auto bigObject = [](int keys, int dupOf) {
    std::string t = "{";
    for (int i = 0; i < keys; ++i) t += std::format("\"k{}\": {}, ", i, i);
    t += dupOf >= 0 ? std::format("\"k{}\": 0}}", dupOf) : "\"last\": 0}";
    return t;
  };
  CHECK(ser::parseJson(bigObject(40, -1), j, &err));
  CHECK(j.size() == 41);
  CHECK_FALSE(ser::parseJson(bigObject(40, 3), j, &err));
  CHECK(err == "duplicate key \"k3\"");
  CHECK_FALSE(ser::parseJson(bigObject(40, 35), j, &err));
  CHECK(err == "duplicate key \"k35\"");
  CHECK_FALSE(ser::parseJson(bigObject(10, 3), j, &err));
}

TEST_CASE("json: schema migration chain", "[serialization][migration]") {
  SchemaMigrator m("demo", 3);
  m.add(1, "add b", [](Json& d, std::string&) {
    d["b"] = 1;
    return true;
  });
  m.add(2, "double b", [](Json& d, std::string&) {
    d["b"] = d["b"].get<int>() * 2;
    return true;
  });
  std::string err;
  Json doc = {{"format", "demo"}, {"version", 1}};
  REQUIRE(m.migrate(doc, err));
  CHECK(doc["version"] == 3);
  CHECK(doc["b"] == 2);

  Json current = {{"format", "demo"}, {"version", 3}, {"b", 7}};
  REQUIRE(m.migrate(current, err));
  CHECK(current["b"] == 7);  // nothing applied

  Json newer = {{"format", "demo"}, {"version", 4}};
  CHECK_FALSE(m.migrate(newer, err));
  CHECK(err == "version 4 is newer than this engine supports (3)");
  Json gap = {{"format", "demo"}, {"version", 0}};
  CHECK_FALSE(m.migrate(gap, err));
  CHECK(err == "no migration from version 0 (current 3)");
  Json wrong = {{"format", "other"}, {"version", 1}};
  CHECK_FALSE(m.migrate(wrong, err));
  CHECK(err == "format is \"other\", expected \"demo\"");
  Json noVersion = {{"format", "demo"}, {"version", "1"}};
  CHECK_FALSE(m.migrate(noVersion, err));
  CHECK(err == "missing integer \"version\"");

  SchemaMigrator failing("demo", 2);
  failing.add(1, "needs x", [](Json&, std::string& why) {
    why = "x missing";
    return false;
  });
  Json d1 = {{"format", "demo"}, {"version", 1}};
  CHECK_FALSE(failing.migrate(d1, err));
  CHECK(err == "migration 1 -> 2 (needs x) failed: x missing");

  SchemaMigrator throwing("demo", 2);
  throwing.add(1, "reads a number", [](Json& d, std::string&) {
    return d.at("missing").get<int>() > 0;  // throws out_of_range
  });
  Json d2 = {{"format", "demo"}, {"version", 1}};
  CHECK_FALSE(throwing.migrate(d2, err));
  CHECK(err.find("migration 1 -> 2 (reads a number) failed: ") == 0);
}

// ── Scenes ─────────────────────────────────────────────────────

TEST_CASE("scene: sample world matches the golden file", "[serialization][golden]") {
  checkGolden("sample.kscene", sampleScene());
}

TEST_CASE("scene: golden file reloads to the same world and re-saves byte-identically",
          "[serialization][golden]") {
  const ComponentRegistry r = makeRegistry();
  const std::string golden = readText(kData / "sample.kscene");
  World loaded;
  std::string err;
  REQUIRE(sc::loadScene(golden, loaded, r, &err));
  CHECK(sc::saveScene(loaded, r) == golden);
  World built;
  buildSampleWorld(built);
  requireSameWorld(built, loaded, r);
  // Loading leaves world transforms current.
  const Entity lamp = loaded.find(guid("00000000-0000-4000-8000-0000000000a5"));
  CHECK(loaded.get<WorldTransform>(lamp).matrix == loaded.computeWorldMatrix(lamp));
}

TEST_CASE("scene: random worlds round-trip exactly", "[serialization]") {
  const ComponentRegistry r = makeRegistry();
  std::mt19937 rng(99);
  auto unit = [&] { return std::uniform_real_distribution<f32>(0.0f, 1.0f)(rng); };
  const char* names[] = {"a", "", "tab\there", "quote\"", "back\\slash", "\xe6\xb5\x8b\xe8\xaf\x95",
                         "ctrl\x01", "new\nline"};
  for (int round = 0; round < 40; ++round) {
    World w;
    std::vector<Entity> all;
    for (int i = 0; i < 30; ++i) {
      const Entity parent = all.empty() || unit() < 0.3f ? kNullEntity : all[rng() % all.size()];
      const Entity e = w.create(names[rng() % std::size(names)], parent);
      all.push_back(e);
      for (const auto& info : r.all()) {
        if (info->optional) {
          if (unit() < 0.5f) continue;
          info->add(w, e);
        }
        for (const FieldInfo& f : info->fields) {
          const f32 lo = f.min ? *f.min : -1000.0f;
          const f32 hi = f.max ? *f.max : 1000.0f;
          auto num = [&] { return lo + (hi - lo) * unit(); };
          FieldValue v;
          switch (f.type) {
            case FieldType::Bool: v = unit() < 0.5f; break;
            case FieldType::Int: v = static_cast<i32>(num()); break;
            case FieldType::Float: v = num(); break;
            case FieldType::Vec2: v = glm::vec2(num(), num()); break;
            case FieldType::Vec3:
            case FieldType::Color3: v = glm::vec3(num(), num(), num()); break;
            case FieldType::Vec4: v = glm::vec4(num(), num(), num(), num()); break;
            case FieldType::Quat:
              v = glm::normalize(glm::quat(unit() - 0.5f, unit() - 0.5f, unit() - 0.5f, unit()));
              break;
            case FieldType::String: v = std::string(names[rng() % std::size(names)]); break;
            case FieldType::Guid: v = unit() < 0.3f ? Guid{} : Guid::generate(); break;
            case FieldType::Enum: v = f.enumEntries[rng() % f.enumEntries.size()].value; break;
          }
          std::string err;
          INFO(info->name << "." << f.name << " " << err);
          REQUIRE(setField(w, e, *info, f, v, &err));
        }
      }
    }
    const std::string first = sc::saveScene(w, r);
    World back;
    std::string err;
    INFO(err);
    REQUIRE(sc::loadScene(first, back, r, &err));
    requireSameWorld(w, back, r);
    REQUIRE(sc::saveScene(back, r) == first);
  }
}

TEST_CASE("scene: hand-written files may be loose", "[serialization]") {
  const ComponentRegistry r = makeRegistry();
  // Child before parent, omitted guid/fields/components, integer floats.
  const std::string text = R"({
    "format": "kenshin.scene", "version": 1,
    "entities": [
      {"name": "kid", "parent": "00000000-0000-4000-8000-000000000001",
       "components": {"light": {"intensity": 4, "type": "spot"}}},
      {"guid": "00000000-0000-4000-8000-000000000001", "name": "dad",
       "components": {"transform": {"position": [1, 2, 3]}}},
      {"guid": "00000000-0000-4000-8000-000000000002"}
    ]})";
  World w;
  std::string err;
  REQUIRE(sc::loadScene(text, w, r, &err));
  REQUIRE(w.size() == 3);
  const Entity dad = w.find(guid("00000000-0000-4000-8000-000000000001"));
  REQUIRE(w.childCount(dad) == 1);
  const Entity kid = w.firstChild(dad);
  CHECK(w.get<Name>(kid).value == "kid");
  CHECK_FALSE(w.guid(kid).isNull());
  CHECK(w.get<Light>(kid).intensity == 4.0f);
  CHECK(w.get<Light>(kid).type == Light::Type::Spot);
  CHECK(w.get<Light>(kid).range == Light{}.range);
  CHECK(w.get<Transform>(dad).position == glm::vec3(1, 2, 3));
  CHECK(w.get<Transform>(dad).scale == glm::vec3(1));
  CHECK(w.get<Name>(w.find(guid("00000000-0000-4000-8000-000000000002"))).value == "Entity");
  CHECK(w.roots() == std::vector<Entity>{dad, w.find(guid("00000000-0000-4000-8000-000000000002"))});
}

TEST_CASE("scene: invalid documents are rejected and the world is untouched",
          "[serialization]") {
  const ComponentRegistry r = makeRegistry();
  World w;
  buildSampleWorld(w);
  const std::string before = sc::saveScene(w, r);

  const std::string G1 = "\"00000000-0000-4000-8000-000000000001\"";
  const std::string G2 = "\"00000000-0000-4000-8000-000000000002\"";
  auto doc = [](const std::string& entities) {
    return "{\"format\": \"kenshin.scene\", \"version\": 1, \"entities\": [" + entities + "]}";
  };
  auto light = [&](const std::string& fields) {
    return doc("{\"name\": \"L\", \"components\": {\"light\": {" + fields + "}}}");
  };
  const std::pair<std::string, std::string> cases[] = {
      {"not json", "syntax error"},
      {"[]", "document is not a JSON object"},
      {"{\"version\": 1, \"entities\": []}", "missing \"format\""},
      {"{\"format\": \"kenshin.prefab\", \"version\": 1, \"entities\": []}",
       "format is \"kenshin.prefab\", expected \"kenshin.scene\""},
      {"{\"format\": \"kenshin.scene\", \"version\": 2, \"entities\": []}", "newer than"},
      {"{\"format\": \"kenshin.scene\", \"version\": 0, \"entities\": []}", "no migration from version 0"},
      {"{\"format\": \"kenshin.scene\", \"version\": 1, \"entities\": [], \"extra\": 1}",
       "unknown top-level key \"extra\""},
      {"{\"format\": \"kenshin.scene\", \"version\": 1, \"entities\": {}}", "\"entities\" must be an array"},
      {doc("3"), "entities[0] must be an object"},
      {doc("{\"nmae\": \"x\"}"), "entities[0]: unknown key \"nmae\""},
      {doc("{\"guid\": \"nope\"}"), "entities[0].guid must be a non-null guid string"},
      {doc("{\"guid\": \"00000000-0000-0000-0000-000000000000\"}"), "non-null guid"},
      {doc("{\"guid\": " + G1 + "}, {\"guid\": " + G1 + "}"), "entities[1]: guid 00000000-0000-4000-8000-000000000001 is used twice"},
      {doc("{\"name\": \"x\", \"parent\": " + G2 + "}"), "entities[0] (\"x\"): parent \"00000000-0000-4000-8000-000000000002\" is not an entity of this file"},
      {doc("{\"guid\": " + G1 + ", \"parent\": " + G1 + "}"), "is its own parent"},
      {doc("{\"guid\": " + G1 + ", \"name\": \"a\", \"parent\": " + G2 + "}, {\"guid\": " + G2 + ", \"parent\": " + G1 + "}"),
       "entities[0] (\"a\") is part of a parent cycle"},
      {doc("{\"name\": 5}"), "entities[0].name must be a string"},
      {doc("{\"components\": []}"), "entities[0].components must be an object"},
      {doc("{\"name\": \"x\", \"components\": {\"lihgt\": {}}}"),
       "entities[0] (\"x\"): unknown component \"lihgt\" (known: transform, mesh_renderer, light, camera)"},
      {doc("{\"components\": {\"light\": 1}}"), "component \"light\" must be an object"},
      {light("\"intensty\": 1"), "light has no field \"intensty\""},
      {light("\"intensity\": \"bright\""), "entities[0] (\"L\"): light.intensity expects a finite number, got a string"},
      {light("\"intensity\": -1"), "light.intensity: -1 is below the minimum 0"},
      {light("\"intensity\": 1e999"), "number overflow"},
      {"{\"format\": \"kenshin.scene\", \"version\": 99999999999, \"entities\": []}",
       "version 99999999999 is newer"},
      {"{\"format\": \"kenshin.scene\", \"version\": -1, \"entities\": []}",
       "version -1 is not valid"},
      {light("\"type\": \"laser\""), "light.type: \"laser\" is not one of [directional, point, spot]"},
      {light("\"type\": 1"), "light.type expects one of [directional, point, spot], got an integer"},
      {light("\"color\": [1, 1]"), "light.color expects an array of 3 numbers"},
      {light("\"cast_shadows\": 1"), "light.cast_shadows expects a boolean"},
      {doc("{\"components\": {\"transform\": {\"rotation\": [0, 0, 0, 0]}}}"), "transform.rotation: zero quaternion"},
      {doc("{\"components\": {\"mesh_renderer\": {\"mesh\": \"x\"}}}"), "mesh_renderer.mesh: \"x\" is not a guid"},
  };
  for (const auto& [text, expected] : cases) {
    INFO(text);
    std::string err;
    CHECK_FALSE(sc::loadScene(text, w, r, &err));
    INFO("error: " << err);
    CHECK(err.find(expected) != std::string::npos);
    REQUIRE(sc::saveScene(w, r) == before);
  }
}

TEST_CASE("scene: older versions load through the migration chain",
          "[serialization][migration][golden]") {
  const ComponentRegistry r = makeRegistry();
  const std::string legacy = readText(kData / "legacy_v0.kscene");
  World w;
  std::string err;
  CHECK_FALSE(sc::loadScene(legacy, w, r, &err));  // the engine has no v0
  CHECK(err == "no migration from version 0 (current 1)");

  const SchemaMigrator m = legacyMigrator();
  sc::LoadOptions opts;
  opts.migrator = &m;
  REQUIRE(sc::loadScene(legacy, w, r, &err, opts));
  const Entity lamp = w.find(guid("6a1f0c52-3b7e-4d2a-9c11-0f5e2b8d7a02"));
  REQUIRE(lamp != kNullEntity);
  CHECK(w.get<Transform>(lamp).position == glm::vec3(0, 0, 1.5f));
  CHECK(w.get<Light>(lamp).intensity == 2.0f);
  // Saved in the current format: the migrated golden.
  checkGolden("legacy_v0_migrated.kscene", sc::saveScene(w, r));
}

TEST_CASE("scene: files save atomically under unicode paths", "[serialization]") {
  const ComponentRegistry r = makeRegistry();
  const fs::path dir = fs::temp_directory_path() /
                       kenshin::util::pathFromUtf8("kenshin_\xe5\x9c\xba\xe6\x99\xaf_test");
  fs::remove_all(dir);
  fs::create_directories(dir);
  const fs::path file = dir / kenshin::util::pathFromUtf8("\xe5\x85\xb3\xe5\x8d\xa1.kscene");

  World w;
  buildSampleWorld(w);
  std::string err;
  REQUIRE(sc::saveSceneFile(file, w, r, &err));
  CHECK(readText(file) == sampleScene());
  w.destroy(w.firstRoot());
  REQUIRE(sc::saveSceneFile(file, w, r, &err));  // overwrite
  World back;
  REQUIRE(sc::loadSceneFile(file, back, r, &err));
  CHECK(back.size() == 1);
  usize entries = 0;
  for ([[maybe_unused]] const auto& e : fs::directory_iterator(dir)) ++entries;
  CHECK(entries == 1);  // no temp file left behind

  CHECK_FALSE(sc::loadSceneFile(dir / "missing.kscene", back, r, &err));
  CHECK(err.find("cannot read") != std::string::npos);
  CHECK(back.size() == 1);
  kenshin::util::writeFileAtomic(dir / "bad.kscene", "{");
  CHECK_FALSE(sc::loadSceneFile(dir / "bad.kscene", back, r, &err));
  CHECK(err.find("bad.kscene: ") != std::string::npos);
  fs::remove_all(dir);
}

// ── Prefabs ────────────────────────────────────────────────────

TEST_CASE("prefab: subtree saves to the golden file", "[serialization][golden]") {
  World w;
  buildSampleWorld(w);
  const Entity crate = w.find(guid("00000000-0000-4000-8000-0000000000a4"));
  checkGolden("crate.kprefab", sc::savePrefab(w, crate, makeRegistry()));
}

TEST_CASE("prefab: instances are fresh copies", "[serialization]") {
  const ComponentRegistry r = makeRegistry();
  const std::string prefab = readText(kData / "crate.kprefab");
  World w;
  const Entity holderA = w.create("A");
  const Entity holderB = w.create("B");
  w.create("existing", holderB);
  std::string err;
  const Entity a = sc::instantiatePrefab(prefab, w, r, holderA, &err);
  const Entity b = sc::instantiatePrefab(prefab, w, r, holderB, &err);
  const Entity c = sc::instantiatePrefab(prefab, w, r, kNullEntity, &err);
  REQUIRE(a != kNullEntity);
  REQUIRE(b != kNullEntity);
  REQUIRE(c != kNullEntity);
  CHECK(w.parent(a) == holderA);
  CHECK(w.children(holderB).back() == b);  // appended last
  CHECK(w.parent(c) == kNullEntity);
  CHECK(w.size() == 3 + 3 * 2);

  // Same content, distinct identities, none equal to the file's.
  CHECK(w.find(guid("00000000-0000-4000-8000-0000000000a4")) == kNullEntity);
  CHECK(w.guid(a) != w.guid(b));
  for (const Entity inst : {a, b, c}) {
    CHECK(w.get<Name>(inst).value == "Crate \"A\" / \xe7\xae\xb1\xe5\xad\x90");
    CHECK(w.get<Transform>(inst).scale == glm::vec3(2, 1, 0.5f));
    CHECK(w.get<MeshRenderer>(inst).mesh == guid("11111111-2222-4333-8444-555555555555"));
    REQUIRE(w.childCount(inst) == 1);
    const Entity lamp = w.firstChild(inst);
    CHECK(w.get<Light>(lamp).outerConeDeg == 35.0f);
  }
  // Re-saving an instance reproduces the prefab apart from the guids.
  World single;
  const Entity s = sc::instantiatePrefab(prefab, single, r, kNullEntity, &err);
  Json again = sc::prefabToJson(single, s, r);
  Json original;
  REQUIRE(ser::parseJson(prefab, original, nullptr));
  for (Json* d : {&again, &original})
    for (Json& e : (*d)["entities"]) {
      e.erase("guid");
      e.erase("parent");
    }
  CHECK(ser::writeJson(again) == ser::writeJson(original));
}

TEST_CASE("prefab: failures roll back completely", "[serialization]") {
  const ComponentRegistry r = makeRegistry();
  World w;
  const Entity holder = w.create("holder");
  std::string err;
  const std::string head = "{\"format\": \"kenshin.prefab\", \"version\": 1, \"entities\": [";
  // The second entity is bad: the first was already created.
  const std::string lateFailure = head +
      "{\"guid\": \"00000000-0000-4000-8000-000000000001\", \"name\": \"root\"},"
      "{\"name\": \"bad\", \"parent\": \"00000000-0000-4000-8000-000000000001\","
      " \"components\": {\"camera\": {\"fov_y_deg\": 500}}}]}";
  CHECK(sc::instantiatePrefab(lateFailure, w, r, holder, &err) == kNullEntity);
  CHECK(err.find("entities[1] (\"bad\"): camera.fov_y_deg: 500 is above the maximum 179") !=
        std::string::npos);
  CHECK(w.size() == 1);
  CHECK(w.childCount(holder) == 0);

  CHECK(sc::instantiatePrefab(head + "{}, {}]}", w, r, holder, &err) == kNullEntity);
  CHECK(err == "a prefab needs exactly one root entity, found 2");
  CHECK(sc::instantiatePrefab(head + "]}", w, r, holder, &err) == kNullEntity);
  CHECK(err == "a prefab needs exactly one root entity, found 0");
  CHECK(sc::instantiatePrefab(sampleScene(), w, r, holder, &err) == kNullEntity);
  CHECK(err == "format is \"kenshin.scene\", expected \"kenshin.prefab\"");
  CHECK(sc::instantiatePrefab(head + "{}]}", w, r, static_cast<Entity>(777), &err) == kNullEntity);
  CHECK(err == "invalid parent entity");
  CHECK(w.size() == 1);
}

// ── Project ────────────────────────────────────────────────────

TEST_CASE("project: golden round-trip and strict loading", "[serialization][golden]") {
  const sc::ProjectDesc p{"Tetris", "scenes/main.kscene"};
  checkGolden("tetris.kproject", sc::saveProject(p));
  sc::ProjectDesc back;
  std::string err;
  REQUIRE(sc::loadProject(readText(kData / "tetris.kproject"), back, &err));
  CHECK(back.name == "Tetris");
  CHECK(back.startupScene == "scenes/main.kscene");
  CHECK(sc::saveProject(back) == readText(kData / "tetris.kproject"));

  auto project = [](const std::string& body) {
    return "{\"format\": \"kenshin.project\", \"version\": 1" + body + "}";
  };
  const std::pair<std::string, std::string> cases[] = {
      {project(""), "missing \"name\""},
      {project(", \"name\": \"\""), "\"name\" must be a non-empty string"},
      {project(", \"name\": \"x\", \"startup\": \"a\""), "unknown key \"startup\""},
      {project(", \"name\": \"x\", \"startup_scene\": \"../out.kscene\""), "must be a relative path"},
      {"{\"format\": \"kenshin.scene\", \"version\": 1}", "expected \"kenshin.project\""},
  };
  for (const auto& [text, expected] : cases) {
    INFO(text);
    sc::ProjectDesc out{"keep", "keep"};
    CHECK_FALSE(sc::loadProject(text, out, &err));
    INFO(err);
    CHECK(err.find(expected) != std::string::npos);
    CHECK(out.name == "keep");
  }
  REQUIRE(sc::loadProject(project(", \"name\": \"x\""), back, &err));
  CHECK(back.startupScene.empty());
}

TEST_CASE("project: relative path rules", "[serialization]") {
  for (const char* ok : {"a", "scenes/main.kscene", "a/b/c.txt", "a.b/..c"})
    CHECK(sc::isProjectRelativePath(ok));
  for (const char* bad : {"", "/abs", "C:/x", "a\\b", "../x", "a/../b", "a/./b", "a//b", "a/",
                          ".", "file://x"})
    CHECK_FALSE(sc::isProjectRelativePath(bad));
}

// Hidden (run with "[bench]"): scene file cost for the roadmap log.
TEST_CASE("scene: 10k-entity save/load timing", "[.][bench]") {
  using Clock = std::chrono::steady_clock;
  auto ms = [](Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  const ComponentRegistry r = makeRegistry();
  World w;
  std::vector<Entity> all;
  for (int i = 0; i < 10000; ++i) {
    const Entity parent = i % 10 == 0 ? kNullEntity : all[static_cast<usize>(std::max(0, i - 1 - i % 3))];
    all.push_back(w.create("entity", parent));
    w.get<Transform>(all.back()).position = {0.1f * static_cast<f32>(i), 1.0f, 2.0f};
    if (i % 2) w.add<MeshRenderer>(all.back()).mesh = Guid::generate();
    if (i % 7 == 0) w.add<Light>(all.back());
  }
  const auto t0 = Clock::now();
  const std::string text = sc::saveScene(w, r);
  const auto t1 = Clock::now();
  World back;
  std::string err;
  REQUIRE(sc::loadScene(text, back, r, &err));
  const auto t2 = Clock::now();
  Json parsed;
  REQUIRE(ser::parseJson(text, parsed, nullptr));
  const auto t3 = Clock::now();
  const Json plain = Json::parse(text);
  const auto t4 = Clock::now();
  REQUIRE(sc::sceneFromJson(parsed, back, r, &err));
  const auto t5 = Clock::now();
  WARN("10k entities: " << text.size() / 1024 << " KiB, save " << ms(t0, t1) << " ms, load "
                        << ms(t1, t2) << " ms (strict parse " << ms(t2, t3) << ", plain parse "
                        << ms(t3, t4) << ", build " << ms(t4, t5) << ")");
}
