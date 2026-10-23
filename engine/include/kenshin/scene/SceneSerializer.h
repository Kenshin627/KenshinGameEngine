#pragma once

// ── Scene and prefab files (roadmap 1.2) ────────────────────────
// A scene is a flat list of entities in hierarchy pre-order; each names
// its parent by Guid and carries its components, written field by field
// through the component reflection:
//
//   {
//     "format": "kenshin.scene",
//     "version": 1,
//     "entities": [
//       {
//         "guid": "4f1c...",
//         "name": "Sun",
//         "parent": null,
//         "components": {
//           "transform": {"position": [0, 0, 5], "rotation": [0, 0, 0, 1], "scale": [1, 1, 1]},
//           "light": {"type": "directional", ...}
//         }
//       }
//     ]
//   }
//
// Vectors are arrays, quaternions [x, y, z, w], enums their labels,
// Guids canonical strings (null = none). Loading is strict - unknown
// keys, components or fields, wrong types, out-of-range values, dangling
// or cyclic parents are errors that name the entity and field - and
// all-or-nothing. Hand-written files may list entities in any order and
// omit fields (defaults) or the guid (a fresh one); saving writes the
// canonical form, which reloads and re-saves byte-identically.
//
// A prefab is the same document with "format": "kenshin.prefab" and
// exactly one root; each instantiation gets fresh Guids.

#include <filesystem>
#include <string>
#include <string_view>

#include "kenshin/ecs/Reflection.h"
#include "kenshin/ecs/World.h"
#include "kenshin/serialization/Json.h"

KENSHIN_NS_BEGIN
namespace scene {

inline constexpr i32 kSceneVersion = 1;
inline constexpr std::string_view kSceneFormat = "kenshin.scene";
inline constexpr std::string_view kPrefabFormat = "kenshin.prefab";

/// The engine's migration chains (version 1 is the first format: empty).
const serialization::SchemaMigrator& sceneMigrator();
const serialization::SchemaMigrator& prefabMigrator();

struct LoadOptions {
  /// nullptr = the built-in chain for the document's format.
  const serialization::SchemaMigrator* migrator = nullptr;
};

// ── Scenes ─────────────────────────────────────────────────────
serialization::Json sceneToJson(const ecs::World& world,
                                const ecs::ComponentRegistry& registry);
std::string saveScene(const ecs::World& world, const ecs::ComponentRegistry& registry);
/// Replaces `world` with the document's entities. On failure `world`
/// is untouched and `error` names the problem. Success invalidates
/// every handle into the previous world.
bool sceneFromJson(serialization::Json doc, ecs::World& world,
                   const ecs::ComponentRegistry& registry, std::string* error,
                   const LoadOptions& options = {});
bool loadScene(std::string_view text, ecs::World& world,
               const ecs::ComponentRegistry& registry, std::string* error,
               const LoadOptions& options = {});

bool saveSceneFile(const std::filesystem::path& path, const ecs::World& world,
                   const ecs::ComponentRegistry& registry, std::string* error);
bool loadSceneFile(const std::filesystem::path& path, ecs::World& world,
                   const ecs::ComponentRegistry& registry, std::string* error,
                   const LoadOptions& options = {});

// ── Prefabs ────────────────────────────────────────────────────
/// `root` and its subtree; the root is written as a root (its own
/// parent is not part of the prefab).
serialization::Json prefabToJson(const ecs::World& world, ecs::Entity root,
                                 const ecs::ComponentRegistry& registry);
std::string savePrefab(const ecs::World& world, ecs::Entity root,
                       const ecs::ComponentRegistry& registry);
/// A new copy under `parent` (kNullEntity = a new root), appended last.
/// Returns its root, or kNullEntity with `error` set and `world`
/// unchanged.
ecs::Entity instantiatePrefabJson(serialization::Json doc, ecs::World& world,
                                  const ecs::ComponentRegistry& registry,
                                  ecs::Entity parent, std::string* error,
                                  const LoadOptions& options = {});
/// (Separate names: a std::string converts to both Json and string_view.)
ecs::Entity instantiatePrefab(std::string_view text, ecs::World& world,
                              const ecs::ComponentRegistry& registry,
                              ecs::Entity parent, std::string* error,
                              const LoadOptions& options = {});

}  // namespace scene
KENSHIN_NS_END
