#include "kenshin/ecs/World.h"

KENSHIN_NS_BEGIN
namespace ecs {

namespace {
// The same test decomposeTransform applies (per-column scale, then the
// unit basis), so tiny-but-valid scales are not mistaken for singular.
bool isSingular(const glm::mat4& m) {
  Transform unused;
  return !decomposeTransform(m, unused);
}
}  // namespace

// ── Lifetime ───────────────────────────────────────────────────

Entity World::create(std::string_view name, Entity parent) {
  // A random v4 collision is ~2^-122; the loop only guards a broken RNG.
  core::Guid guid = core::Guid::generate();
  while (m_byGuid.count(guid) != 0) guid = core::Guid::generate();
  return createEntity(guid, name, parent);
}

Entity World::createWithGuid(const core::Guid& guid, std::string_view name,
                             Entity parent) {
  if (guid.isNull()) {
    KENSHIN_ERROR("World::createWithGuid: null guid");
    return kNullEntity;
  }
  if (m_byGuid.count(guid) != 0) {
    KENSHIN_ERROR("World::createWithGuid: guid {} already in use",
                  guid.toString());
    return kNullEntity;
  }
  return createEntity(guid, name, parent);
}

Entity World::createEntity(const core::Guid& guid, std::string_view name,
                           Entity parent) {
  if (parent != kNullEntity && !valid(parent)) {
    KENSHIN_ERROR("World::create: invalid parent");
    return kNullEntity;
  }
  const Entity e = m_registry.create();
  m_registry.emplace<Name>(e, std::string(name));
  m_registry.emplace<Identity>(e, guid);
  m_registry.emplace<Transform>(e);
  m_registry.emplace<Hierarchy>(e);
  // An identity local sits exactly on its parent: seed from the
  // parent's cached pose (O(1); walking the chain would make building a
  // deep hierarchy quadratic). Like every WorldTransform it is "as of
  // the last updateTransforms()".
  m_registry.emplace<WorldTransform>(
      e, parent == kNullEntity ? glm::mat4(1.0f)
                               : m_registry.get<WorldTransform>(parent).matrix);
  m_byGuid.emplace(guid, e);
  link(e, parent, kNullEntity);
  return e;
}

void World::destroy(Entity e) {
  if (!valid(e)) return;
  // Gather the subtree first (pre-order inside it), then unhook only
  // its root: links among the doomed entities need no repair.
  std::vector<Entity> doomed;
  doomed.push_back(e);
  for (usize i = 0; i < doomed.size(); ++i) {
    for (Entity c = m_registry.get<Hierarchy>(doomed[i]).firstChild;
         c != kNullEntity; c = m_registry.get<Hierarchy>(c).nextSibling)
      doomed.push_back(c);
  }
  unlink(e);
  for (const Entity d : doomed) {
    m_byGuid.erase(m_registry.get<Identity>(d).guid);
    m_registry.destroy(d);
  }
}

void World::queueDestroy(Entity e) {
  if (valid(e)) m_destroyQueue.push_back(e);
}

usize World::flushDestroyQueue() {
  const usize before = size();
  // Swap out first: nothing a destroy triggers can grow the vector
  // being walked.
  std::vector<Entity> queue;
  queue.swap(m_destroyQueue);
  for (const Entity e : queue) destroy(e);  // dead descendants are skipped
  return before - size();
}

void World::clear() {
  m_registry.clear();
  m_byGuid.clear();
  m_destroyQueue.clear();
  m_firstRoot = m_lastRoot = kNullEntity;
  m_rootCount = 0;
}

// ── Identity / lookup ──────────────────────────────────────────

Entity World::find(const core::Guid& guid) const {
  const auto it = m_byGuid.find(guid);
  return it == m_byGuid.end() ? kNullEntity : it->second;
}

Entity World::findByName(std::string_view name) const {
  for (const Entity e : preOrder())
    if (m_registry.get<Name>(e).value == name) return e;
  return kNullEntity;
}

// ── Hierarchy ──────────────────────────────────────────────────

World::SiblingList World::listOf(Entity parent) {
  if (parent == kNullEntity) return {&m_firstRoot, &m_lastRoot, &m_rootCount};
  Hierarchy& h = m_registry.get<Hierarchy>(parent);
  return {&h.firstChild, &h.lastChild, &h.childCount};
}

void World::link(Entity e, Entity parent, Entity before) {
  Hierarchy& h = m_registry.get<Hierarchy>(e);
  const SiblingList list = listOf(parent);
  h.parent = parent;
  h.nextSibling = before;
  h.prevSibling = before == kNullEntity
                      ? *list.last
                      : m_registry.get<Hierarchy>(before).prevSibling;
  if (h.prevSibling == kNullEntity)
    *list.first = e;
  else
    m_registry.get<Hierarchy>(h.prevSibling).nextSibling = e;
  if (before == kNullEntity)
    *list.last = e;
  else
    m_registry.get<Hierarchy>(before).prevSibling = e;
  ++*list.count;
}

void World::unlink(Entity e) {
  Hierarchy& h = m_registry.get<Hierarchy>(e);
  const SiblingList list = listOf(h.parent);
  if (h.prevSibling == kNullEntity)
    *list.first = h.nextSibling;
  else
    m_registry.get<Hierarchy>(h.prevSibling).nextSibling = h.nextSibling;
  if (h.nextSibling == kNullEntity)
    *list.last = h.prevSibling;
  else
    m_registry.get<Hierarchy>(h.nextSibling).prevSibling = h.prevSibling;
  --*list.count;
  h.parent = h.prevSibling = h.nextSibling = kNullEntity;
}

bool World::setParent(Entity child, Entity parent, Reparent mode,
                      Entity before) {
  if (!valid(child)) return false;
  if (parent != kNullEntity &&
      (!valid(parent) || parent == child || isDescendantOf(parent, child)))
    return false;
  if (before != kNullEntity &&
      (!valid(before) || before == child ||
       m_registry.get<Hierarchy>(before).parent != parent))
    return false;

  if (mode == Reparent::KeepWorld) {
    const glm::mat4 parentWorld =
        parent == kNullEntity ? glm::mat4(1.0f) : computeWorldMatrix(parent);
    if (!isSingular(parentWorld)) {
      Transform local;
      if (decomposeTransform(glm::inverse(parentWorld) * computeWorldMatrix(child),
                             local))
        m_registry.get<Transform>(child) = local;
    }
  }
  unlink(child);
  link(child, parent, before);
  return true;
}

std::vector<Entity> World::children(Entity e) const {
  std::vector<Entity> out;
  if (!valid(e)) return out;
  const Hierarchy& h = m_registry.get<Hierarchy>(e);
  out.reserve(h.childCount);
  for (Entity c = h.firstChild; c != kNullEntity;
       c = m_registry.get<Hierarchy>(c).nextSibling)
    out.push_back(c);
  return out;
}

std::vector<Entity> World::roots() const {
  std::vector<Entity> out;
  out.reserve(m_rootCount);
  for (Entity r = m_firstRoot; r != kNullEntity;
       r = m_registry.get<Hierarchy>(r).nextSibling)
    out.push_back(r);
  return out;
}

bool World::isDescendantOf(Entity e, Entity ancestor) const {
  if (!valid(e) || !valid(ancestor)) return false;
  for (Entity p = m_registry.get<Hierarchy>(e).parent; p != kNullEntity;
       p = m_registry.get<Hierarchy>(p).parent)
    if (p == ancestor) return true;
  return false;
}

std::vector<Entity> World::preOrder() const {
  std::vector<Entity> out;
  out.reserve(size());
  // Iterative walk over the links: no recursion depth limit.
  Entity e = m_firstRoot;
  while (e != kNullEntity) {
    out.push_back(e);
    const Hierarchy* h = &m_registry.get<Hierarchy>(e);
    if (h->firstChild != kNullEntity) {
      e = h->firstChild;
      continue;
    }
    while (h->nextSibling == kNullEntity && h->parent != kNullEntity)
      h = &m_registry.get<Hierarchy>(h->parent);
    e = h->nextSibling;
  }
  return out;
}

// ── Transforms ─────────────────────────────────────────────────

void World::updateTransforms() {
  // Pre-order over the links (as preOrder(), without materializing it):
  // each parent's world matrix is fresh before its children read it.
  // Storages are resolved once - registry.get<T> looks the pool up by
  // type on every call.
  const auto& hierarchy = m_registry.storage<Hierarchy>();
  const auto& local = m_registry.storage<Transform>();
  auto& world = m_registry.storage<WorldTransform>();
  Entity e = m_firstRoot;
  while (e != kNullEntity) {
    const Hierarchy* h = &hierarchy.get(e);
    const glm::mat4 m = local.get(e).matrix();
    world.get(e).matrix =
        h->parent == kNullEntity ? m : world.get(h->parent).matrix * m;
    if (h->firstChild != kNullEntity) {
      e = h->firstChild;
      continue;
    }
    while (h->nextSibling == kNullEntity && h->parent != kNullEntity)
      h = &hierarchy.get(h->parent);
    e = h->nextSibling;
  }
}

glm::mat4 World::computeWorldMatrix(Entity e) const {
  KENSHIN_ASSERT(valid(e));
  glm::mat4 m = m_registry.get<Transform>(e).matrix();
  for (Entity p = m_registry.get<Hierarchy>(e).parent; p != kNullEntity;
       p = m_registry.get<Hierarchy>(p).parent)
    m = m_registry.get<Transform>(p).matrix() * m;
  return m;
}

bool World::setWorldMatrix(Entity e, const glm::mat4& world) {
  if (!valid(e)) return false;
  const Entity p = m_registry.get<Hierarchy>(e).parent;
  glm::mat4 local = world;
  if (p != kNullEntity) {
    const glm::mat4 parentWorld = computeWorldMatrix(p);
    if (isSingular(parentWorld)) return false;
    local = glm::inverse(parentWorld) * world;
  }
  Transform t;
  if (!decomposeTransform(local, t)) return false;
  m_registry.get<Transform>(e) = t;
  return true;
}

}  // namespace ecs
KENSHIN_NS_END
