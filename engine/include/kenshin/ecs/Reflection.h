#pragma once

// ── Component reflection (roadmap 1.1) ─────────────────────────
// One description per component - stable names, field types, ranges,
// enum labels - that serialization (1.2), the inspector, the Lua
// bindings and the agent API all read, so they cannot drift apart.
//
//   registry.add<Light>("light", "Punctual or directional light")
//       .field<&Light::intensity>("intensity", {.min = 0.0f});
//
// Fields are reached through member pointers (no offsetof), and every
// write through setField() is validated: type, finiteness, range,
// enum membership, unit quaternions.

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <typeindex>
#include <unordered_map>
#include <variant>
#include <vector>

#include "kenshin/core/Guid.h"
#include "kenshin/ecs/World.h"

KENSHIN_NS_BEGIN
namespace ecs {

/// What a field means; the storage alternative of FieldValue follows
/// from it (Color3 -> vec3, Enum -> i32).
enum class FieldType : u8 {
  Bool, Int, Float, Vec2, Vec3, Vec4, Quat, Color3, String, Guid, Enum,
};
std::string_view fieldTypeName(FieldType type);

using FieldValue = std::variant<bool, i32, f32, glm::vec2, glm::vec3,
                                glm::vec4, glm::quat, std::string, core::Guid>;

struct EnumEntry {
  std::string name;  // stable label, snake_case
  i32 value;
};

struct FieldOptions {
  /// Inclusive bounds on every scalar lane (Int/Float/vectors/colors).
  std::optional<f32> min;
  std::optional<f32> max;
  /// vec3 fields: shown as a color picker, linear RGB.
  bool color = false;
  std::vector<EnumEntry> enumEntries;  // required for enum members
  std::string description;
};

struct FieldInfo {
  std::string name;  // stable key, snake_case
  FieldType type;
  std::optional<f32> min;
  std::optional<f32> max;
  std::vector<EnumEntry> enumEntries;
  std::string description;
  FieldValue (*read)(const void* component);
  void (*write)(void* component, const FieldValue& value);  // pre-validated

  /// nullptr if unknown.
  const EnumEntry* enumByName(std::string_view label) const;
  const EnumEntry* enumByValue(i32 value) const;
};

struct ComponentInfo {
  std::string name;  // stable key, snake_case
  std::string description;
  std::type_index type;
  /// False for core components (present on every entity).
  bool optional;
  std::vector<FieldInfo> fields;

  bool (*has)(const World&, Entity);
  void* (*get)(World&, Entity);              // nullptr if absent
  const void* (*getConst)(const World&, Entity);
  bool (*add)(World&, Entity);     // default value; false if present/core
  bool (*remove)(World&, Entity);  // false if absent/core

  const FieldInfo* findField(std::string_view fieldName) const;
};

namespace detail {
// Lazy: underlying_type of a non-enum is ill-formed, not false.
template <typename M, bool = std::is_enum_v<M>>
struct IsI32Enum : std::false_type {};
template <typename M>
struct IsI32Enum<M, true> : std::is_same<std::underlying_type_t<M>, i32> {};
template <typename M>
constexpr bool kReflectableEnum = IsI32Enum<M>::value;

template <typename M>
constexpr FieldType defaultFieldType() {
  if constexpr (std::is_same_v<M, bool>) return FieldType::Bool;
  else if constexpr (std::is_same_v<M, i32>) return FieldType::Int;
  else if constexpr (std::is_same_v<M, f32>) return FieldType::Float;
  else if constexpr (std::is_same_v<M, glm::vec2>) return FieldType::Vec2;
  else if constexpr (std::is_same_v<M, glm::vec3>) return FieldType::Vec3;
  else if constexpr (std::is_same_v<M, glm::vec4>) return FieldType::Vec4;
  else if constexpr (std::is_same_v<M, glm::quat>) return FieldType::Quat;
  else if constexpr (std::is_same_v<M, std::string>) return FieldType::String;
  else if constexpr (std::is_same_v<M, core::Guid>) return FieldType::Guid;
  else if constexpr (kReflectableEnum<M>) return FieldType::Enum;
  else static_assert(sizeof(M) == 0, "field type not reflectable");
}

/// Validates and appends a field (aborts on a registration error).
void addField(ComponentInfo& info, FieldInfo&& field, bool isVec3);

template <typename T>
struct MemberTraits;
template <typename C, typename M>
struct MemberTraits<M C::*> {
  using Component = C;
  using Member = M;
};
}  // namespace detail

/// Chained field registration for one component.
template <typename C>
class ComponentBuilder {
 public:
  explicit ComponentBuilder(ComponentInfo& info) : m_info(info) {}

  template <auto MemberPtr>
  ComponentBuilder& field(std::string name, FieldOptions options = {}) {
    using Traits = detail::MemberTraits<decltype(MemberPtr)>;
    using M = typename Traits::Member;
    static_assert(std::is_same_v<typename Traits::Component, C>,
                  "member of a different component");
    FieldInfo f;
    f.name = std::move(name);
    f.type = detail::defaultFieldType<M>();
    if (options.color) f.type = FieldType::Color3;  // vec3 only: addField checks
    f.min = options.min;
    f.max = options.max;
    f.enumEntries = std::move(options.enumEntries);
    f.description = std::move(options.description);
    f.read = [](const void* c) -> FieldValue {
      const M& v = static_cast<const C*>(c)->*MemberPtr;
      if constexpr (detail::kReflectableEnum<M>) return static_cast<i32>(v);
      else return v;
    };
    f.write = [](void* c, const FieldValue& v) {
      M& dst = static_cast<C*>(c)->*MemberPtr;
      if constexpr (detail::kReflectableEnum<M>) dst = static_cast<M>(std::get<i32>(v));
      else dst = std::get<M>(v);
    };
    detail::addField(m_info, std::move(f), std::is_same_v<M, glm::vec3>);
    return *this;
  }

 private:
  ComponentInfo& m_info;
};

class ComponentRegistry {
 public:
  template <typename C>
  ComponentBuilder<C> add(std::string name, std::string description = {}) {
    auto info = std::make_unique<ComponentInfo>(ComponentInfo{
        std::move(name), std::move(description), std::type_index(typeid(C)),
        !World::kCore<C>, {},
        [](const World& w, Entity e) { return w.has<C>(e); },
        [](World& w, [[maybe_unused]] Entity e) -> void* {
          if constexpr (World::kStructural<C>) return nullptr;
          else return w.tryGet<C>(e);
        },
        [](const World& w, Entity e) -> const void* { return w.tryGet<C>(e); },
        []([[maybe_unused]] World& w, [[maybe_unused]] Entity e) {
          if constexpr (World::kCore<C>) return false;
          else {
            if (!w.valid(e) || w.has<C>(e)) return false;
            w.add<C>(e);
            return true;
          }
        },
        []([[maybe_unused]] World& w, [[maybe_unused]] Entity e) {
          if constexpr (World::kCore<C>) return false;
          else return w.remove<C>(e);
        }});
    return ComponentBuilder<C>(insert(std::move(info)));
  }

  const ComponentInfo* find(std::string_view name) const;
  const ComponentInfo* find(std::type_index type) const;
  template <typename C>
  const ComponentInfo* find() const {
    return find(std::type_index(typeid(C)));
  }
  /// In registration order.
  std::span<const std::unique_ptr<ComponentInfo>> all() const { return m_infos; }

 private:
  ComponentInfo& insert(std::unique_ptr<ComponentInfo> info);
  std::vector<std::unique_ptr<ComponentInfo>> m_infos;
  std::unordered_map<std::type_index, usize> m_byType;
};

/// Transform, MeshRenderer, Light, Camera. (Name, Identity and the
/// structural components are entity-level data, not reflected fields.)
void registerBuiltinComponents(ComponentRegistry& registry);

/// nullopt when `e` lacks the component.
std::optional<FieldValue> getField(const World& world, Entity e,
                                   const ComponentInfo& component,
                                   const FieldInfo& field);
/// Validated write. On failure nothing changes and `error` (if given)
/// says why, in words fit for an agent or the inspector.
bool setField(World& world, Entity e, const ComponentInfo& component,
              const FieldInfo& field, const FieldValue& value,
              std::string* error = nullptr);

}  // namespace ecs
KENSHIN_NS_END
