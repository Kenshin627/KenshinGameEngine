#pragma once

// ── World: the entity registry plus its invariants (roadmap 1.1) ──
// Wraps an entt::registry. Everything that must stay consistent - the
// core components every entity carries, the hierarchy links, the Guid
// index - is only mutable through World; systems iterate with view<>()
// and touch their own components freely.
//
// Not thread-safe: one World is owned by one thread (the simulation).

#include <entt/entity/registry.hpp>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "kenshin/core/Assert.h"
#include "kenshin/ecs/Components.h"

KENSHIN_NS_BEGIN
namespace ecs {

class World {
 public:
  /// Components World alone may write (links, identity, derived data).
  template <typename T>
  static constexpr bool kStructural =
      std::is_same_v<T, Hierarchy> || std::is_same_v<T, Identity> ||
      std::is_same_v<T, WorldTransform>;
  /// Present on every entity: never added or removed by callers.
  template <typename T>
  static constexpr bool kCore =
      kStructural<T> || std::is_same_v<T, Name> || std::is_same_v<T, Transform>;

  /// How a mutable World hands out T: const for structural components.
  template <typename T>
  using Access = std::conditional_t<kStructural<std::remove_const_t<T>>,
                                    const std::remove_const_t<T>, T>;

  enum class Reparent {
    KeepWorld,  // world pose preserved: the local transform is recomputed
    KeepLocal,  // local transform preserved: the entity moves with its parent
  };

  World() = default;
  World(const World&) = delete;
  World& operator=(const World&) = delete;

  // ── Lifetime ─────────────────────────────────────────────────
  /// New entity with a fresh Guid, appended as the last child of
  /// `parent` (or the last root). An invalid parent -> kNullEntity.
  Entity create(std::string_view name = "Entity", Entity parent = kNullEntity);
  /// Same, with a given identity (scene loading). A null or already
  /// used Guid -> kNullEntity.
  Entity createWithGuid(const core::Guid& guid, std::string_view name,
                        Entity parent = kNullEntity);
  /// Destroy `e` and its whole subtree now. Never call while iterating
  /// a view that may contain them - use queueDestroy instead. Invalid
  /// handles are ignored.
  void destroy(Entity e);
  /// Destroy at the next flushDestroyQueue() (scripts, systems mid-
  /// iteration). Queuing twice, or with an ancestor, is harmless.
  void queueDestroy(Entity e);
  /// Returns the number of entities destroyed (subtrees included).
  usize flushDestroyQueue();
  /// Destroy everything (the destroy queue too).
  void clear();

  /// True for a live entity of this World. Stale handles (destroyed,
  /// slot since reused) are false: entt versions every handle.
  bool valid(Entity e) const { return m_registry.valid(e); }
  usize size() const { return m_byGuid.size(); }

  // ── Identity / lookup ────────────────────────────────────────
  const core::Guid& guid(Entity e) const { return get<Identity>(e).guid; }
  Entity find(const core::Guid& guid) const;
  /// First entity with this exact name in hierarchy (pre-)order.
  Entity findByName(std::string_view name) const;

  // ── Hierarchy ────────────────────────────────────────────────
  /// Move `child` under `parent` (kNullEntity = make it a root),
  /// inserted before sibling `before` (kNullEntity = last). Fails,
  /// changing nothing, when an entity is invalid, `parent` is `child`
  /// or one of its descendants, or `before` is not a child of
  /// `parent` (a root, for a null parent) or is `child` itself.
  /// KeepWorld under a singular parent matrix keeps the local values.
  bool setParent(Entity child, Entity parent,
                 Reparent mode = Reparent::KeepWorld,
                 Entity before = kNullEntity);
  Entity parent(Entity e) const { return get<Hierarchy>(e).parent; }
  Entity firstChild(Entity e) const { return get<Hierarchy>(e).firstChild; }
  Entity nextSibling(Entity e) const { return get<Hierarchy>(e).nextSibling; }
  u32 childCount(Entity e) const { return get<Hierarchy>(e).childCount; }
  std::vector<Entity> children(Entity e) const;
  Entity firstRoot() const { return m_firstRoot; }
  u32 rootCount() const { return m_rootCount; }
  std::vector<Entity> roots() const;
  /// Strict: an entity is not its own descendant.
  bool isDescendantOf(Entity e, Entity ancestor) const;
  /// Every entity, parents before children, siblings in order.
  std::vector<Entity> preOrder() const;

  // ── Transforms ───────────────────────────────────────────────
  /// Recompute every WorldTransform (call after simulation, before
  /// rendering). Linear in entity count, no recursion.
  void updateTransforms();
  /// Exact world matrix from the current local transforms (walks the
  /// parent chain; independent of updateTransforms).
  glm::mat4 computeWorldMatrix(Entity e) const;
  /// Set the local transform so the world matrix becomes `world`
  /// (gizmos, physics write-back). False on a singular `world` or
  /// parent matrix (nothing changes).
  bool setWorldMatrix(Entity e, const glm::mat4& world);

  // ── Components ───────────────────────────────────────────────
  /// Add (or replace) an optional component.
  template <typename T, typename... Args>
  T& add(Entity e, Args&&... args) {
    static_assert(!kCore<T>, "core components exist on every entity");
    KENSHIN_ASSERT(valid(e));
    return m_registry.emplace_or_replace<T>(e, std::forward<Args>(args)...);
  }
  /// Returns false if `e` did not have it.
  template <typename T>
  bool remove(Entity e) {
    static_assert(!kCore<T>, "core components cannot be removed");
    return valid(e) && m_registry.remove<T>(e) > 0;
  }
  template <typename T>
  bool has(Entity e) const {
    return valid(e) && m_registry.all_of<T>(e);
  }
  /// Structural components come back const even from a mutable World.
  template <typename T>
  Access<T>& get(Entity e) {
    KENSHIN_ASSERT(has<T>(e));
    return m_registry.get<T>(e);
  }
  template <typename T>
  const T& get(Entity e) const {
    KENSHIN_ASSERT(has<T>(e));
    return m_registry.get<T>(e);
  }
  /// nullptr when `e` is invalid or lacks T.
  template <typename T>
  Access<T>* tryGet(Entity e) {
    return valid(e) ? m_registry.try_get<T>(e) : nullptr;
  }
  template <typename T>
  const T* tryGet(Entity e) const {
    return valid(e) ? m_registry.try_get<T>(e) : nullptr;
  }

  /// entt view (structural components iterate const). Adding/removing
  /// T while iterating a view of T is undefined (entt rules).
  template <typename... T>
  auto view() {
    return m_registry.view<Access<T>...>();
  }
  template <typename... T>
  auto view() const {
    return m_registry.view<const std::remove_const_t<T>...>();
  }

  const entt::registry& registry() const { return m_registry; }

 private:
  struct SiblingList {
    Entity* first;
    Entity* last;
    u32* count;
  };
  SiblingList listOf(Entity parent);
  void link(Entity e, Entity parent, Entity before);
  void unlink(Entity e);
  Entity createEntity(const core::Guid& guid, std::string_view name, Entity parent);

  entt::registry m_registry;
  std::unordered_map<core::Guid, Entity> m_byGuid;
  std::vector<Entity> m_destroyQueue;
  Entity m_firstRoot = kNullEntity;
  Entity m_lastRoot = kNullEntity;
  u32 m_rootCount = 0;
};

}  // namespace ecs
KENSHIN_NS_END
