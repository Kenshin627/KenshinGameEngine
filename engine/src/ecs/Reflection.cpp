#include "kenshin/ecs/Reflection.h"

#include <cmath>
#include <cstdlib>
#include <format>

KENSHIN_NS_BEGIN
namespace ecs {

namespace {
bool isSnakeCase(std::string_view s) {
  if (s.empty() || !(s[0] >= 'a' && s[0] <= 'z')) return false;
  for (const char c : s)
    if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_'))
      return false;
  return true;
}

/// Index of the FieldValue alternative a field type stores as.
usize storageIndex(FieldType type) {
  switch (type) {
    case FieldType::Bool: return 0;
    case FieldType::Int:
    case FieldType::Enum: return 1;
    case FieldType::Float: return 2;
    case FieldType::Vec2: return 3;
    case FieldType::Vec3:
    case FieldType::Color3: return 4;
    case FieldType::Vec4: return 5;
    case FieldType::Quat: return 6;
    case FieldType::String: return 7;
    case FieldType::Guid: return 8;
  }
  return 0;
}

const char* storageName(usize index) {
  static constexpr const char* kNames[] = {"bool", "int", "float", "vec2", "vec3",
                                           "vec4", "quat", "string", "guid"};
  return index < std::size(kNames) ? kNames[index] : "?";
}

/// The scalar lanes of a numeric value (empty for non-numeric).
std::vector<f32> lanes(const FieldValue& v) {
  struct Visitor {
    std::vector<f32> operator()(bool) const { return {}; }
    std::vector<f32> operator()(i32 x) const { return {static_cast<f32>(x)}; }
    std::vector<f32> operator()(f32 x) const { return {x}; }
    std::vector<f32> operator()(const glm::vec2& x) const { return {x.x, x.y}; }
    std::vector<f32> operator()(const glm::vec3& x) const { return {x.x, x.y, x.z}; }
    std::vector<f32> operator()(const glm::vec4& x) const {
      return {x.x, x.y, x.z, x.w};
    }
    std::vector<f32> operator()(const glm::quat& x) const {
      return {x.x, x.y, x.z, x.w};
    }
    std::vector<f32> operator()(const std::string&) const { return {}; }
    std::vector<f32> operator()(const core::Guid&) const { return {}; }
  };
  return std::visit(Visitor{}, v);
}

bool fail(std::string* error, std::string message) {
  if (error) *error = std::move(message);
  return false;
}
}  // namespace

std::string_view fieldTypeName(FieldType type) {
  switch (type) {
    case FieldType::Bool: return "bool";
    case FieldType::Int: return "int";
    case FieldType::Float: return "float";
    case FieldType::Vec2: return "vec2";
    case FieldType::Vec3: return "vec3";
    case FieldType::Vec4: return "vec4";
    case FieldType::Quat: return "quat";
    case FieldType::Color3: return "color3";
    case FieldType::String: return "string";
    case FieldType::Guid: return "guid";
    case FieldType::Enum: return "enum";
  }
  return "?";
}

const EnumEntry* FieldInfo::enumByName(std::string_view label) const {
  for (const EnumEntry& e : enumEntries)
    if (e.name == label) return &e;
  return nullptr;
}

const EnumEntry* FieldInfo::enumByValue(i32 value) const {
  for (const EnumEntry& e : enumEntries)
    if (e.value == value) return &e;
  return nullptr;
}

const FieldInfo* ComponentInfo::findField(std::string_view fieldName) const {
  for (const FieldInfo& f : fields)
    if (f.name == fieldName) return &f;
  return nullptr;
}

// Registration mistakes are programming errors in engine or game code:
// they abort loudly in every build rather than produce a description
// the serializer or agent API would trust.
namespace detail {
[[noreturn]] void reflectionFatal(const std::string& message) {
  KENSHIN_ERROR("component reflection: {}", message);
  std::abort();
}

void addField(ComponentInfo& info, FieldInfo&& f, bool isVec3) {
  const std::string where = std::format("{}.{}", info.name, f.name);
  if (!isSnakeCase(f.name))
    reflectionFatal(std::format("field name '{}' is not snake_case", where));
  if (info.findField(f.name)) reflectionFatal(std::format("{} registered twice", where));
  if (f.type == FieldType::Color3 && !isVec3)
    reflectionFatal(std::format("{}: color on a non-vec3 member", where));
  if (f.type == FieldType::Enum) {
    if (f.enumEntries.empty())
      reflectionFatal(std::format("{}: enum member without entries", where));
    for (usize i = 0; i < f.enumEntries.size(); ++i)
      for (usize j = i + 1; j < f.enumEntries.size(); ++j)
        if (f.enumEntries[i].name == f.enumEntries[j].name ||
            f.enumEntries[i].value == f.enumEntries[j].value)
          reflectionFatal(std::format("{}: duplicate enum entry", where));
    for (const EnumEntry& entry : f.enumEntries)
      if (!isSnakeCase(entry.name))
        reflectionFatal(std::format("{}: enum label '{}' is not snake_case", where,
                                    entry.name));
  } else if (!f.enumEntries.empty()) {
    reflectionFatal(std::format("{}: enum entries on a non-enum member", where));
  }
  const bool numeric = f.type == FieldType::Int || f.type == FieldType::Float ||
                       f.type == FieldType::Vec2 || f.type == FieldType::Vec3 ||
                       f.type == FieldType::Vec4 || f.type == FieldType::Color3;
  if ((f.min || f.max) && !numeric)
    reflectionFatal(std::format("{}: range on a non-numeric member", where));
  if (f.min && f.max && *f.min > *f.max)
    reflectionFatal(std::format("{}: min > max", where));
  info.fields.push_back(std::move(f));
}
}  // namespace detail

ComponentInfo& ComponentRegistry::insert(std::unique_ptr<ComponentInfo> info) {
  if (!isSnakeCase(info->name))
    detail::reflectionFatal(std::format("component name '{}' is not snake_case",
                                        info->name));
  if (find(info->name) || m_byType.count(info->type))
    detail::reflectionFatal(
        std::format("component '{}' registered twice", info->name));
  m_byType.emplace(info->type, m_infos.size());
  m_infos.push_back(std::move(info));
  return *m_infos.back();
}

const ComponentInfo* ComponentRegistry::find(std::string_view name) const {
  for (const auto& info : m_infos)
    if (info->name == name) return info.get();
  return nullptr;
}

const ComponentInfo* ComponentRegistry::find(std::type_index type) const {
  const auto it = m_byType.find(type);
  return it == m_byType.end() ? nullptr : m_infos[it->second].get();
}

std::optional<FieldValue> getField(const World& world, Entity e,
                                   const ComponentInfo& component,
                                   const FieldInfo& field) {
  const void* c = component.getConst(world, e);
  if (!c) return std::nullopt;
  return field.read(c);
}

bool setField(World& world, Entity e, const ComponentInfo& component,
              const FieldInfo& field, const FieldValue& value,
              std::string* error) {
  const std::string where = std::format("{}.{}", component.name, field.name);
  if (!world.valid(e)) return fail(error, "invalid entity");
  void* c = component.get(world, e);
  if (!c) return fail(error, std::format("entity has no '{}' component", component.name));

  const usize want = storageIndex(field.type);
  if (value.index() != want)
    return fail(error, std::format("{} expects {}, got {}", where,
                                   fieldTypeName(field.type),
                                   storageName(value.index())));

  FieldValue v = value;
  for (const f32 lane : lanes(v))
    if (!std::isfinite(lane))
      return fail(error, std::format("{} must be finite", where));

  if (field.type == FieldType::Enum) {
    const i32 raw = std::get<i32>(v);
    if (!field.enumByValue(raw)) {
      std::string labels;
      for (const EnumEntry& entry : field.enumEntries)
        labels += (labels.empty() ? "" : ", ") + entry.name;
      return fail(error, std::format("{}: {} is not one of [{}]", where, raw, labels));
    }
  } else if (field.type == FieldType::Quat) {
    glm::quat& q = std::get<glm::quat>(v);
    const f32 len = glm::length(q);
    if (!(len > 1e-6f)) return fail(error, std::format("{}: zero quaternion", where));
    q /= len;  // stored as a unit rotation
  } else {
    for (const f32 lane : lanes(v)) {
      if (field.min && lane < *field.min)
        return fail(error, std::format("{}: {} is below the minimum {}", where,
                                       lane, *field.min));
      if (field.max && lane > *field.max)
        return fail(error, std::format("{}: {} is above the maximum {}", where,
                                       lane, *field.max));
    }
  }
  field.write(c, v);
  return true;
}

void registerBuiltinComponents(ComponentRegistry& registry) {
  registry.add<Transform>("transform", "Local position, rotation and scale")
      .field<&Transform::position>("position", {.description = "meters, parent space"})
      .field<&Transform::rotation>("rotation", {.description = "unit quaternion"})
      .field<&Transform::scale>("scale");

  registry.add<MeshRenderer>("mesh_renderer", "Draws a mesh asset")
      .field<&MeshRenderer::mesh>("mesh", {.description = "mesh asset guid"})
      .field<&MeshRenderer::material>(
          "material", {.description = "material asset guid; null = the mesh's own"})
      .field<&MeshRenderer::visible>("visible")
      .field<&MeshRenderer::castShadows>("cast_shadows");

  registry.add<Light>("light", "Directional, point or spot light")
      .field<&Light::type>("type", {.enumEntries = {{"directional", 0},
                                                    {"point", 1},
                                                    {"spot", 2}}})
      .field<&Light::color>("color", {.min = 0.0f, .color = true})
      .field<&Light::intensity>("intensity", {.min = 0.0f})
      .field<&Light::range>("range", {.min = 0.0f, .description = "meters"})
      .field<&Light::innerConeDeg>("inner_cone_deg", {.min = 0.0f, .max = 90.0f})
      .field<&Light::outerConeDeg>("outer_cone_deg", {.min = 0.0f, .max = 90.0f})
      .field<&Light::castShadows>("cast_shadows");

  registry.add<Camera>("camera", "View into the world along local +Y")
      .field<&Camera::projection>(
          "projection", {.enumEntries = {{"perspective", 0}, {"orthographic", 1}}})
      .field<&Camera::fovYDeg>("fov_y_deg", {.min = 1.0f, .max = 179.0f})
      .field<&Camera::orthoHeight>("ortho_height", {.min = 1e-4f})
      .field<&Camera::nearPlane>("near_plane", {.min = 1e-5f})
      .field<&Camera::farPlane>("far_plane", {.min = 1e-4f})
      .field<&Camera::primary>("primary");
}

}  // namespace ecs
KENSHIN_NS_END
