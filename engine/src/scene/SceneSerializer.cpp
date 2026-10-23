#include "kenshin/scene/SceneSerializer.h"

#include <cfloat>
#include <cmath>
#include <format>
#include <limits>
#include <unordered_map>

#include "kenshin/core/Log.h"
#include "kenshin/util/File.h"

KENSHIN_NS_BEGIN
namespace scene {

using ecs::ComponentInfo;
using ecs::ComponentRegistry;
using ecs::Entity;
using ecs::FieldInfo;
using ecs::FieldType;
using ecs::FieldValue;
using ecs::kNullEntity;
using ecs::World;
using serialization::Json;

const serialization::SchemaMigrator& sceneMigrator() {
  static const serialization::SchemaMigrator m(std::string(kSceneFormat), kSceneVersion);
  return m;
}

const serialization::SchemaMigrator& prefabMigrator() {
  static const serialization::SchemaMigrator m(std::string(kPrefabFormat), kSceneVersion);
  return m;
}

namespace {

// ── Field values <-> JSON ──────────────────────────────────────

Json floats(std::initializer_list<f32> values) {
  Json a = Json::array();
  for (const f32 v : values) a.push_back(serialization::floatToJson(v));
  return a;
}

Json fieldToJson(const FieldInfo& field, const FieldValue& value) {
  switch (field.type) {
    case FieldType::Bool: return std::get<bool>(value);
    case FieldType::Int: return std::get<i32>(value);
    case FieldType::Float: return serialization::floatToJson(std::get<f32>(value));
    case FieldType::Vec2: {
      const glm::vec2 v = std::get<glm::vec2>(value);
      return floats({v.x, v.y});
    }
    case FieldType::Vec3:
    case FieldType::Color3: {
      const glm::vec3 v = std::get<glm::vec3>(value);
      return floats({v.x, v.y, v.z});
    }
    case FieldType::Vec4: {
      const glm::vec4 v = std::get<glm::vec4>(value);
      return floats({v.x, v.y, v.z, v.w});
    }
    case FieldType::Quat: {
      const glm::quat q = std::get<glm::quat>(value);
      return floats({q.x, q.y, q.z, q.w});
    }
    case FieldType::String: return std::get<std::string>(value);
    case FieldType::Guid: {
      const core::Guid& g = std::get<core::Guid>(value);
      return g.isNull() ? Json(nullptr) : Json(g.toString());
    }
    case FieldType::Enum: {
      const i32 raw = std::get<i32>(value);
      if (const ecs::EnumEntry* e = field.enumByValue(raw)) return e->name;
      // Only reachable by writing a bad value from C++; keep the number
      // so the loader reports it instead of silently changing it.
      KENSHIN_WARN("scene save: {} holds {} (no such label)", field.name, raw);
      return raw;
    }
  }
  return nullptr;
}

const char* jsonKind(const Json& j) {
  if (j.is_null()) return "null";
  if (j.is_boolean()) return "a boolean";
  if (j.is_number_integer()) return "an integer";
  if (j.is_number()) return "a number";
  if (j.is_string()) return "a string";
  if (j.is_array()) return "an array";
  return "an object";
}

bool readFloat(const Json& j, f32& out) {
  if (!j.is_number()) return false;
  const double d = j.get<double>();
  if (!std::isfinite(d) || std::abs(d) > static_cast<double>(FLT_MAX)) return false;
  out = static_cast<f32>(d);
  return true;
}

bool readFloats(const Json& j, usize n, f32* out) {
  if (!j.is_array() || j.size() != n) return false;
  for (usize i = 0; i < n; ++i)
    if (!readFloat(j[i], out[i])) return false;
  return true;
}

/// JSON -> the field's storage type (range/enum/quat checks are
/// setField's job; this only decides the shape).
bool fieldFromJson(const ComponentInfo& component, const FieldInfo& field, const Json& j,
                   FieldValue& out, std::string& error) {
  const std::string where = std::format("{}.{}", component.name, field.name);
  auto expect = [&](const char* what) {
    error = std::format("{} expects {}, got {}", where, what, jsonKind(j));
    return false;
  };
  switch (field.type) {
    case FieldType::Bool:
      if (!j.is_boolean()) return expect("a boolean");
      out = j.get<bool>();
      return true;
    case FieldType::Int: {
      if (!j.is_number_integer()) return expect("an integer");
      const bool fits = j.is_number_unsigned()
                            ? j.get<u64>() <= static_cast<u64>(std::numeric_limits<i32>::max())
                            : j.get<i64>() >= std::numeric_limits<i32>::min() &&
                                  j.get<i64>() <= std::numeric_limits<i32>::max();
      if (!fits) {
        error = std::format("{}: {} does not fit a 32-bit integer", where, j.dump());
        return false;
      }
      const i64 v = j.is_number_unsigned() ? static_cast<i64>(j.get<u64>()) : j.get<i64>();
      out = static_cast<i32>(v);
      return true;
    }
    case FieldType::Float: {
      f32 v;
      if (!readFloat(j, v)) return expect("a finite number");
      out = v;
      return true;
    }
    case FieldType::Vec2: {
      glm::vec2 v;
      if (!readFloats(j, 2, &v.x)) return expect("an array of 2 numbers");
      out = v;
      return true;
    }
    case FieldType::Vec3:
    case FieldType::Color3: {
      glm::vec3 v;
      if (!readFloats(j, 3, &v.x)) return expect("an array of 3 numbers");
      out = v;
      return true;
    }
    case FieldType::Vec4: {
      glm::vec4 v;
      if (!readFloats(j, 4, &v.x)) return expect("an array of 4 numbers");
      out = v;
      return true;
    }
    case FieldType::Quat: {
      f32 q[4];
      if (!readFloats(j, 4, q)) return expect("an array of 4 numbers [x, y, z, w]");
      out = glm::quat(q[3], q[0], q[1], q[2]);
      return true;
    }
    case FieldType::String:
      if (!j.is_string()) return expect("a string");
      out = j.get<std::string>();
      return true;
    case FieldType::Guid: {
      if (j.is_null()) {
        out = core::Guid{};
        return true;
      }
      if (!j.is_string()) return expect("a guid string or null");
      const auto g = core::Guid::parse(j.get<std::string>());
      if (!g) {
        error = std::format("{}: \"{}\" is not a guid", where, j.get<std::string>());
        return false;
      }
      out = *g;
      return true;
    }
    case FieldType::Enum: {
      std::string labels;
      for (const ecs::EnumEntry& e : field.enumEntries)
        labels += (labels.empty() ? "" : ", ") + e.name;
      if (!j.is_string())
        return expect(std::format("one of [{}]", labels).c_str());
      const ecs::EnumEntry* e = field.enumByName(j.get<std::string>());
      if (!e) {
        error = std::format("{}: \"{}\" is not one of [{}]", where, j.get<std::string>(),
                            labels);
        return false;
      }
      out = e->value;
      return true;
    }
  }
  return false;
}

// ── Entities -> JSON ───────────────────────────────────────────

Json entityToJson(const World& world, Entity e, Entity parent,
                  const ComponentRegistry& registry) {
  Json j = Json::object();
  j["guid"] = world.guid(e).toString();
  j["name"] = world.get<ecs::Name>(e).value;
  j["parent"] = parent == kNullEntity ? Json(nullptr) : Json(world.guid(parent).toString());
  Json components = Json::object();
  for (const auto& info : registry.all()) {
    if (!info->has(world, e)) continue;
    const void* c = info->getConst(world, e);
    Json fields = Json::object();
    for (const FieldInfo& f : info->fields) fields[f.name] = fieldToJson(f, f.read(c));
    components[info->name] = std::move(fields);
  }
  j["components"] = std::move(components);
  return j;
}

/// `root`'s subtree in pre-order (iterative).
std::vector<Entity> subtreePreOrder(const World& world, Entity root) {
  std::vector<Entity> out;
  std::vector<Entity> stack{root};
  while (!stack.empty()) {
    const Entity e = stack.back();
    stack.pop_back();
    out.push_back(e);
    const std::vector<Entity> kids = world.children(e);
    for (auto it = kids.rbegin(); it != kids.rend(); ++it) stack.push_back(*it);
  }
  return out;
}

Json documentHeader(std::string_view format) {
  Json doc = Json::object();
  doc["format"] = std::string(format);
  doc["version"] = kSceneVersion;
  return doc;
}

// ── JSON -> entities ───────────────────────────────────────────

struct ParsedEntity {
  core::Guid guid;  // null = generate
  std::string name = "Entity";
  i64 parent = -1;  // index into the parsed list
  const Json* components = nullptr;
};

std::string entityLabel(const std::vector<ParsedEntity>& list, usize i) {
  return list[i].name.empty() ? std::format("entities[{}]", i)
                              : std::format("entities[{}] (\"{}\")", i, list[i].name);
}

bool parseDocument(const Json& doc, std::vector<ParsedEntity>& out, std::string& error) {
  for (auto it = doc.begin(); it != doc.end(); ++it)
    if (it.key() != "format" && it.key() != "version" && it.key() != "entities") {
      error = std::format("unknown top-level key \"{}\"", it.key());
      return false;
    }
  const auto entities = doc.find("entities");
  if (entities == doc.end() || !entities->is_array()) {
    error = "\"entities\" must be an array";
    return false;
  }
  out.assign(entities->size(), {});
  std::unordered_map<core::Guid, usize> byGuid;
  std::vector<std::string> parentText(out.size());
  for (usize i = 0; i < entities->size(); ++i) {
    const Json& j = (*entities)[i];
    const std::string at = std::format("entities[{}]", i);
    if (!j.is_object()) {
      error = at + " must be an object";
      return false;
    }
    for (auto it = j.begin(); it != j.end(); ++it) {
      const std::string& k = it.key();
      if (k == "guid") {
        if (!it->is_string() || !core::Guid::parse(it->get<std::string>()) ||
            core::Guid::parse(it->get<std::string>())->isNull()) {
          error = at + ".guid must be a non-null guid string";
          return false;
        }
        out[i].guid = *core::Guid::parse(it->get<std::string>());
        if (!byGuid.emplace(out[i].guid, i).second) {
          error = std::format("{}: guid {} is used twice", at, out[i].guid.toString());
          return false;
        }
      } else if (k == "name") {
        if (!it->is_string()) {
          error = at + ".name must be a string";
          return false;
        }
        out[i].name = it->get<std::string>();
      } else if (k == "parent") {
        if (!it->is_null() && !it->is_string()) {
          error = at + ".parent must be a guid string or null";
          return false;
        }
        if (it->is_string()) parentText[i] = it->get<std::string>();
      } else if (k == "components") {
        if (!it->is_object()) {
          error = at + ".components must be an object";
          return false;
        }
        out[i].components = &*it;
      } else {
        error = std::format("{}: unknown key \"{}\"", at, k);
        return false;
      }
    }
  }
  // Parents resolve only once every guid is known (any order allowed).
  for (usize i = 0; i < out.size(); ++i) {
    if (parentText[i].empty()) continue;
    const auto g = core::Guid::parse(parentText[i]);
    const auto it = g ? byGuid.find(*g) : byGuid.end();
    if (it == byGuid.end()) {
      error = std::format("{}: parent \"{}\" is not an entity of this file",
                          entityLabel(out, i), parentText[i]);
      return false;
    }
    if (it->second == i) {
      error = entityLabel(out, i) + " is its own parent";
      return false;
    }
    out[i].parent = static_cast<i64>(it->second);
  }
  return true;
}

/// Pre-order creation sequence: parents first, siblings in file order.
/// Entities never reached sit on a parent cycle.
bool creationOrder(const std::vector<ParsedEntity>& list, std::vector<usize>& order,
                   std::string& error) {
  std::vector<std::vector<usize>> kids(list.size());
  std::vector<usize> roots;
  for (usize i = 0; i < list.size(); ++i)
    (list[i].parent < 0 ? roots : kids[static_cast<usize>(list[i].parent)]).push_back(i);
  order.clear();
  std::vector<usize> stack(roots.rbegin(), roots.rend());
  while (!stack.empty()) {
    const usize i = stack.back();
    stack.pop_back();
    order.push_back(i);
    for (auto it = kids[i].rbegin(); it != kids[i].rend(); ++it) stack.push_back(*it);
  }
  if (order.size() != list.size()) {
    std::vector<bool> reached(list.size(), false);
    for (const usize i : order) reached[i] = true;
    for (usize i = 0; i < list.size(); ++i)
      if (!reached[i]) {
        error = entityLabel(list, i) + " is part of a parent cycle";
        return false;
      }
  }
  return true;
}

bool applyComponents(World& world, Entity e, const Json& components,
                     const ComponentRegistry& registry, std::string& error) {
  for (auto it = components.begin(); it != components.end(); ++it) {
    const ComponentInfo* info = registry.find(it.key());
    if (!info) {
      std::string known;
      for (const auto& c : registry.all()) known += (known.empty() ? "" : ", ") + c->name;
      error = std::format("unknown component \"{}\" (known: {})", it.key(), known);
      return false;
    }
    if (!it->is_object()) {
      error = std::format("component \"{}\" must be an object", info->name);
      return false;
    }
    if (!info->has(world, e)) info->add(world, e);
    for (auto f = it->begin(); f != it->end(); ++f) {
      const FieldInfo* field = info->findField(f.key());
      if (!field) {
        std::string known;
        for (const FieldInfo& k : info->fields) known += (known.empty() ? "" : ", ") + k.name;
        error = std::format("{} has no field \"{}\" (fields: {})", info->name, f.key(), known);
        return false;
      }
      FieldValue v;
      if (!fieldFromJson(*info, *field, *f, v, error)) return false;
      if (!ecs::setField(world, e, *info, *field, v, &error)) return false;
    }
  }
  return true;
}

/// Creates the entities under `attach` (their roots' parent). Every
/// created entity is appended to `created` (pre-order), so a failure
/// can be rolled back by destroying created[0..roots].
bool buildEntities(const std::vector<ParsedEntity>& list, World& world, Entity attach,
                   bool freshGuids, const ComponentRegistry& registry,
                   std::vector<Entity>& created, std::string& error) {
  std::vector<usize> order;
  if (!creationOrder(list, order, error)) return false;
  std::vector<Entity> entityOf(list.size(), kNullEntity);
  for (const usize i : order) {
    const ParsedEntity& p = list[i];
    const Entity parent = p.parent < 0 ? attach : entityOf[static_cast<usize>(p.parent)];
    const Entity e = freshGuids || p.guid.isNull() ? world.create(p.name, parent)
                                                   : world.createWithGuid(p.guid, p.name, parent);
    if (e == kNullEntity) {
      error = entityLabel(list, i) + ": cannot create (guid already in the world)";
      return false;
    }
    entityOf[i] = e;
    created.push_back(e);
    if (p.components && !applyComponents(world, e, *p.components, registry, error)) {
      error = entityLabel(list, i) + ": " + error;
      return false;
    }
  }
  return true;
}

bool migrate(Json& doc, const serialization::SchemaMigrator& builtin,
             const LoadOptions& options, std::string& error) {
  const serialization::SchemaMigrator& m = options.migrator ? *options.migrator : builtin;
  return m.migrate(doc, error);
}

bool setError(std::string* out, const std::string& message) {
  if (out) *out = message;
  return false;
}

}  // namespace

// ── Scenes ─────────────────────────────────────────────────────

Json sceneToJson(const World& world, const ComponentRegistry& registry) {
  Json doc = documentHeader(kSceneFormat);
  Json entities = Json::array();
  for (const Entity e : world.preOrder())
    entities.push_back(entityToJson(world, e, world.parent(e), registry));
  doc["entities"] = std::move(entities);
  return doc;
}

std::string saveScene(const World& world, const ComponentRegistry& registry) {
  return serialization::writeJson(sceneToJson(world, registry));
}

bool sceneFromJson(Json doc, World& world, const ComponentRegistry& registry,
                   std::string* error, const LoadOptions& options) {
  std::string why;
  std::vector<ParsedEntity> list;
  if (!migrate(doc, sceneMigrator(), options, why) || !parseDocument(doc, list, why))
    return setError(error, why);
  World staged;
  std::vector<Entity> created;
  if (!buildEntities(list, staged, kNullEntity, /*freshGuids=*/false, registry, created, why))
    return setError(error, why);
  staged.updateTransforms();
  world = std::move(staged);
  return true;
}

bool loadScene(std::string_view text, World& world, const ComponentRegistry& registry,
               std::string* error, const LoadOptions& options) {
  Json doc;
  std::string why;
  if (!serialization::parseJson(text, doc, &why)) return setError(error, why);
  return sceneFromJson(std::move(doc), world, registry, error, options);
}

bool saveSceneFile(const std::filesystem::path& path, const World& world,
                   const ComponentRegistry& registry, std::string* error) {
  return util::writeFileAtomic(path, saveScene(world, registry), error);
}

bool loadSceneFile(const std::filesystem::path& path, World& world,
                   const ComponentRegistry& registry, std::string* error,
                   const LoadOptions& options) {
  std::string text;
  if (!util::readFile(path, text))
    return setError(error, "cannot read " + util::pathToUtf8(path));
  std::string why;
  if (!loadScene(text, world, registry, &why, options))
    return setError(error, util::pathToUtf8(path) + ": " + why);
  return true;
}

// ── Prefabs ────────────────────────────────────────────────────

Json prefabToJson(const World& world, Entity root, const ComponentRegistry& registry) {
  Json doc = documentHeader(kPrefabFormat);
  Json entities = Json::array();
  for (const Entity e : subtreePreOrder(world, root))
    entities.push_back(entityToJson(world, e, e == root ? kNullEntity : world.parent(e), registry));
  doc["entities"] = std::move(entities);
  return doc;
}

std::string savePrefab(const World& world, Entity root, const ComponentRegistry& registry) {
  return serialization::writeJson(prefabToJson(world, root, registry));
}

Entity instantiatePrefabJson(Json doc, World& world, const ComponentRegistry& registry,
                             Entity parent, std::string* error, const LoadOptions& options) {
  std::string why;
  std::vector<ParsedEntity> list;
  if (!migrate(doc, prefabMigrator(), options, why) || !parseDocument(doc, list, why)) {
    setError(error, why);
    return kNullEntity;
  }
  usize roots = 0;
  for (const ParsedEntity& p : list) roots += p.parent < 0 ? 1 : 0;
  if (roots != 1) {
    setError(error, std::format("a prefab needs exactly one root entity, found {}", roots));
    return kNullEntity;
  }
  if (parent != kNullEntity && !world.valid(parent)) {
    setError(error, "invalid parent entity");
    return kNullEntity;
  }
  std::vector<Entity> created;
  if (!buildEntities(list, world, parent, /*freshGuids=*/true, registry, created, why)) {
    if (!created.empty()) world.destroy(created.front());  // the single root
    setError(error, why);
    return kNullEntity;
  }
  return created.front();
}

Entity instantiatePrefab(std::string_view text, World& world, const ComponentRegistry& registry,
                         Entity parent, std::string* error, const LoadOptions& options) {
  Json doc;
  std::string why;
  if (!serialization::parseJson(text, doc, &why)) {
    setError(error, why);
    return kNullEntity;
  }
  return instantiatePrefabJson(std::move(doc), world, registry, parent, error, options);
}

}  // namespace scene
KENSHIN_NS_END
