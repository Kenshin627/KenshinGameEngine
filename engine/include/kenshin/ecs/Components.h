#pragma once

// ── Built-in components (roadmap 1.1) ───────────────────────────
// World frame: right-handed, Z-up, meters (the renderer's frame; glTF
// is rotated into it on import). An entity's local axes: +X right,
// +Y forward, +Z up - cameras look and lights shine along local +Y.
//
// Every entity owns Name, Identity, Transform, Hierarchy and
// WorldTransform from creation to destruction (World enforces this);
// MeshRenderer, Light and Camera are optional.

#include <entt/entity/entity.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <string>

#include "kenshin/Common.h"
#include "kenshin/core/Guid.h"

KENSHIN_NS_BEGIN
namespace ecs {

using Entity = entt::entity;
inline constexpr Entity kNullEntity = entt::null;

inline constexpr glm::vec3 kLocalRight{1.0f, 0.0f, 0.0f};
inline constexpr glm::vec3 kLocalForward{0.0f, 1.0f, 0.0f};
inline constexpr glm::vec3 kLocalUp{0.0f, 0.0f, 1.0f};

/// Display name; not unique (the Guid is the identity).
struct Name {
  std::string value;
};

/// Stable identity, fixed at creation (read-only through World).
struct Identity {
  core::Guid guid;
};

/// Local TRS relative to the parent (the world, for roots). Applied
/// scale first, then rotation, then translation.
struct Transform {
  glm::vec3 position{0.0f};
  glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};  // (w, x, y, z): identity
  glm::vec3 scale{1.0f};

  glm::mat4 matrix() const;
};

/// Split an affine matrix back into TRS. Exact for matrices built from
/// a Transform; shear (non-uniform scale under a rotated parent) has no
/// TRS form and is dropped. A mirrored matrix (det < 0) comes back with
/// a negative X scale. Returns false - leaving `out` untouched - when the
/// basis is singular (a zero or dependent column: no rotation exists).
bool decomposeTransform(const glm::mat4& m, Transform& out);

/// Parent-first world matrix as of the last World::updateTransforms()
/// (local edits since then are not reflected; computeWorldMatrix is the
/// always-exact query). A new entity starts on its parent's cached pose.
struct WorldTransform {
  glm::mat4 matrix{1.0f};
};

/// Intrusive, ordered tree links. Sibling order is meaningful (editor
/// outliner, serialization); roots form one more list owned by World.
/// Read-only outside World.
struct Hierarchy {
  Entity parent = kNullEntity;
  Entity firstChild = kNullEntity;
  Entity lastChild = kNullEntity;
  Entity prevSibling = kNullEntity;
  Entity nextSibling = kNullEntity;
  u32 childCount = 0;
};

/// Draws a mesh asset. `material` null = the mesh's own materials.
struct MeshRenderer {
  core::Guid mesh;
  core::Guid material;
  bool visible = true;
  bool castShadows = true;
};

struct Light {
  enum class Type : i32 { Directional = 0, Point = 1, Spot = 2 };

  Type type = Type::Point;
  glm::vec3 color{1.0f};  // linear RGB
  /// Directional: illuminance in lux-like units; point/spot: intensity
  /// scale (the renderer's current units, until physical units land).
  f32 intensity = 1.0f;
  f32 range = 10.0f;  // point/spot falloff distance, meters
  f32 innerConeDeg = 30.0f;  // spot: full intensity inside
  f32 outerConeDeg = 45.0f;  // spot: zero outside
  bool castShadows = false;
};

struct Camera {
  enum class Projection : i32 { Perspective = 0, Orthographic = 1 };

  Projection projection = Projection::Perspective;
  f32 fovYDeg = 60.0f;
  f32 orthoHeight = 10.0f;  // full view height, meters
  f32 nearPlane = 0.1f;
  f32 farPlane = 1000.0f;
  /// The runtime renders through the first primary camera in hierarchy
  /// order.
  bool primary = false;
};

}  // namespace ecs
KENSHIN_NS_END
