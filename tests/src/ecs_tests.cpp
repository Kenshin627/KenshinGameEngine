// ECS (roadmap step 1.1): Guid, TRS math, the World hierarchy
// (create / reparent / destroy, checked link-by-link against a reference
// model under random operations) and component reflection.

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <glm/gtc/matrix_transform.hpp>
#include <map>
#include <random>
#include <set>
#include <unordered_set>
#include <utility>

#include "kenshin/core/Guid.h"
#include "kenshin/ecs/Components.h"
#include "kenshin/ecs/Reflection.h"
#include "kenshin/ecs/World.h"

using kenshin::f32;
using kenshin::i32;
using kenshin::u32;
using kenshin::usize;
using kenshin::core::Guid;
using namespace kenshin::ecs;

namespace {

bool approx(const glm::mat4& a, const glm::mat4& b, f32 eps = 1e-4f) {
  for (int c = 0; c < 4; ++c)
    for (int r = 0; r < 4; ++r)
      if (std::abs(a[c][r] - b[c][r]) > eps) return false;
  return true;
}

bool approx(const glm::vec3& a, const glm::vec3& b, f32 eps = 1e-4f) {
  return glm::all(glm::lessThanEqual(glm::abs(a - b), glm::vec3(eps)));
}

/// Same rotation (q and -q are equal rotations).
bool nearRotation(const glm::quat& a, const glm::quat& b, f32 eps = 1e-4f) {
  return std::abs(std::abs(glm::dot(a, b)) - 1.0f) < eps;
}

Transform trs(glm::vec3 p, f32 angleDeg, glm::vec3 axis, glm::vec3 s) {
  Transform t;
  t.position = p;
  t.rotation = glm::angleAxis(glm::radians(angleDeg), glm::normalize(axis));
  t.scale = s;
  return t;
}

/// Every hierarchy link agrees with its neighbours; counts, first/last
/// and the root list are exact; every live entity is reached exactly
/// once by preOrder; the Guid index maps both ways.
void checkInvariants(const World& w) {
  std::set<Entity> seen;
  const std::vector<Entity> order = w.preOrder();
  for (const Entity e : order) {
    REQUIRE(w.valid(e));
    REQUIRE(seen.insert(e).second);
  }
  REQUIRE(order.size() == w.size());
  REQUIRE(w.registry().view<const Identity>().size() == w.size());

  auto checkList = [&](Entity parent, Entity first, Entity last, u32 count) {
    u32 n = 0;
    Entity prev = kNullEntity;
    for (Entity c = first; c != kNullEntity;
         c = w.get<Hierarchy>(c).nextSibling) {
      const Hierarchy& h = w.get<Hierarchy>(c);
      REQUIRE(h.parent == parent);
      REQUIRE(h.prevSibling == prev);
      prev = c;
      ++n;
      REQUIRE(n <= w.size());  // no sibling cycle
    }
    REQUIRE(prev == last);
    REQUIRE(n == count);
  };
  checkList(kNullEntity, w.firstRoot(), w.roots().empty() ? kNullEntity : w.roots().back(),
            w.rootCount());
  for (const Entity e : order) {
    const Hierarchy& h = w.get<Hierarchy>(e);
    checkList(e, h.firstChild, h.lastChild, h.childCount);
    REQUIRE(w.find(w.guid(e)) == e);
    REQUIRE(w.has<Name>(e));
    REQUIRE(w.has<Transform>(e));
    REQUIRE(w.has<WorldTransform>(e));
  }
}

}  // namespace

// ── Guid ───────────────────────────────────────────────────────

TEST_CASE("guid: generate is unique, v4 and non-null", "[ecs][guid]") {
  std::unordered_set<Guid> seen;
  for (int i = 0; i < 20000; ++i) {
    const Guid g = Guid::generate();
    REQUIRE_FALSE(g.isNull());
    REQUIRE(seen.insert(g).second);
    const std::string s = g.toString();
    REQUIRE(s.size() == 36);
    CHECK(s[14] == '4');                              // version
    CHECK(std::string("89ab").find(s[19]) != std::string::npos);  // variant
  }
}

TEST_CASE("guid: text round-trip and strict parsing", "[ecs][guid]") {
  const Guid g = Guid::generate();
  REQUIRE(Guid::parse(g.toString()) == g);

  const auto known = Guid::parse("0123ABCD-4567-89ab-cdef-0123456789AB");
  REQUIRE(known);
  CHECK(known->hi == 0x0123abcd456789abull);
  CHECK(known->lo == 0xcdef0123456789abull);
  CHECK(known->toString() == "0123abcd-4567-89ab-cdef-0123456789ab");
  CHECK(Guid{}.toString() == "00000000-0000-0000-0000-000000000000");
  CHECK(Guid::parse("00000000-0000-0000-0000-000000000000")->isNull());

  CHECK_FALSE(Guid::parse(""));
  CHECK_FALSE(Guid::parse("0123abcd456789abcdef0123456789ab"));        // no dashes
  CHECK_FALSE(Guid::parse("0123abcd-4567-89ab-cdef-0123456789a"));     // short
  CHECK_FALSE(Guid::parse("0123abcd-4567-89ab-cdef-0123456789abc"));   // long
  CHECK_FALSE(Guid::parse("0123abcd-4567-89ab-cdef-0123456789ag"));    // bad hex
  CHECK_FALSE(Guid::parse("0123abc-d4567-89ab-cdef-0123456789ab"));    // dash moved
  CHECK_FALSE(Guid::parse("{0123abcd-4567-89ab-cdef-0123456789a}"));   // braces
}

// ── Transform math ─────────────────────────────────────────────

TEST_CASE("transform: matrix applies scale, then rotation, then translation",
          "[ecs][transform]") {
  const Transform t = trs({10, 0, 0}, 90.0f, {0, 0, 1}, {2, 3, 4});
  const glm::vec3 p = glm::vec3(t.matrix() * glm::vec4(1, 0, 0, 1));
  // (1,0,0) -> scale (2,0,0) -> rotate 90 about Z (0,2,0) -> +10x.
  CHECK(approx(p, {10, 2, 0}));
  CHECK(approx(glm::vec3(t.matrix() * glm::vec4(0, 1, 0, 1)), {7, 0, 0}));
  // Forward convention: local +Y.
  const Transform yaw = trs({}, 90.0f, {0, 0, 1}, {1, 1, 1});
  CHECK(approx(glm::mat3_cast(yaw.rotation) * kLocalForward, {-1, 0, 0}));
}

TEST_CASE("transform: decompose inverts matrix()", "[ecs][transform]") {
  std::mt19937 rng(7);
  std::uniform_real_distribution<f32> u(-1.0f, 1.0f);
  for (int i = 0; i < 500; ++i) {
    glm::vec3 scale(0.1f + 3.0f * std::abs(u(rng)), 0.1f + 3.0f * std::abs(u(rng)),
                    0.1f + 3.0f * std::abs(u(rng)));
    if (i % 3 == 0) scale.x = -scale.x;  // mirrored
    const Transform t = trs({u(rng) * 50, u(rng) * 50, u(rng) * 50}, u(rng) * 180.0f,
                            {u(rng) + 1.5f, u(rng), u(rng)}, scale);
    Transform back;
    REQUIRE(decomposeTransform(t.matrix(), back));
    INFO("case " << i);
    CHECK(approx(back.matrix(), t.matrix(), 1e-3f));
    CHECK(approx(back.position, t.position));
    CHECK(std::abs(glm::length(back.rotation) - 1.0f) < 1e-5f);
    // Mirror folds into X: the X scale sign comes back negative.
    CHECK((back.scale.x < 0) == (scale.x < 0));
  }
}

TEST_CASE("transform: decompose rejects singular matrices", "[ecs][transform]") {
  Transform keep = trs({1, 2, 3}, 10, {0, 0, 1}, {1, 1, 1});
  const Transform before = keep;
  glm::mat4 zeroScale = trs({}, 0, {0, 0, 1}, {1, 0, 1}).matrix();
  CHECK_FALSE(decomposeTransform(zeroScale, keep));
  glm::mat4 dependent(1.0f);
  dependent[1] = dependent[0];  // two equal basis columns
  CHECK_FALSE(decomposeTransform(dependent, keep));
  glm::mat4 nan(1.0f);
  nan[0][0] = std::numeric_limits<f32>::quiet_NaN();
  CHECK_FALSE(decomposeTransform(nan, keep));
  CHECK(keep.position == before.position);
  // Tiny but valid scales are not singular.
  CHECK(decomposeTransform(trs({}, 0, {0, 0, 1}, glm::vec3(1e-4f)).matrix(), keep));
}

// ── Hierarchy ──────────────────────────────────────────────────

TEST_CASE("world: create builds ordered children and roots", "[ecs][hierarchy]") {
  World w;
  const Entity a = w.create("a");
  const Entity b = w.create("b");
  const Entity a1 = w.create("a1", a);
  const Entity a2 = w.create("a2", a);
  const Entity a3 = w.create("a3", a);
  const Entity a21 = w.create("a21", a2);
  checkInvariants(w);

  CHECK(w.size() == 6);
  CHECK(w.roots() == std::vector<Entity>{a, b});
  CHECK(w.children(a) == std::vector<Entity>{a1, a2, a3});
  CHECK(w.childCount(a) == 3);
  CHECK(w.parent(a21) == a2);
  CHECK(w.preOrder() == std::vector<Entity>{a, a1, a2, a21, a3, b});
  CHECK(w.isDescendantOf(a21, a));
  CHECK_FALSE(w.isDescendantOf(a, a));
  CHECK_FALSE(w.isDescendantOf(b, a));
  CHECK(w.get<Name>(a21).value == "a21");
  CHECK(w.findByName("a3") == a3);
  CHECK(w.findByName("missing") == kNullEntity);

  CHECK(w.create("orphan", static_cast<Entity>(12345)) == kNullEntity);
  CHECK(w.size() == 6);
}

TEST_CASE("world: world matrices compose parent-first", "[ecs][hierarchy]") {
  World w;
  const Entity root = w.create("root");
  const Entity mid = w.create("mid", root);
  const Entity leaf = w.create("leaf", mid);
  w.get<Transform>(root) = trs({5, 0, 0}, 90, {0, 0, 1}, {2, 2, 2});
  w.get<Transform>(mid) = trs({1, 0, 0}, 0, {0, 0, 1}, {1, 1, 1});
  w.get<Transform>(leaf) = trs({0, 1, 0}, 45, {1, 0, 0}, {1, 0.5f, 1});
  w.updateTransforms();

  const glm::mat4 expect = w.get<Transform>(root).matrix() *
                           w.get<Transform>(mid).matrix() *
                           w.get<Transform>(leaf).matrix();
  CHECK(approx(w.get<WorldTransform>(leaf).matrix, expect));
  CHECK(approx(w.computeWorldMatrix(leaf), expect));
  // mid's origin: root scales by 2 and turns +X into +Y, then +5x.
  CHECK(approx(glm::vec3(w.get<WorldTransform>(mid).matrix[3]), {5, 2, 0}));
  // leaf origin: mid + (0,1,0) * 2, turned to -X.
  CHECK(approx(glm::vec3(w.get<WorldTransform>(leaf).matrix[3]), {3, 2, 0}));

  // A new child starts on its parent's (updated) pose.
  const Entity fresh = w.create("fresh", leaf);
  CHECK(approx(w.get<WorldTransform>(fresh).matrix, expect));
}

TEST_CASE("world: very deep chains need no recursion", "[ecs][hierarchy]") {
  World w;
  Entity e = w.create("0");
  const Entity first = e;
  for (int i = 1; i < 20000; ++i) {
    e = w.create("n", e);
    w.get<Transform>(e).position = {0.001f, 0, 0};
  }
  w.updateTransforms();
  CHECK(approx(glm::vec3(w.get<WorldTransform>(e).matrix[3]), {19.999f, 0, 0}, 1e-2f));
  CHECK(w.preOrder().size() == 20000);
  w.destroy(first);
  CHECK(w.size() == 0);
}

TEST_CASE("world: reparent keeps world or local pose", "[ecs][hierarchy]") {
  World w;
  const Entity p1 = w.create("p1");
  const Entity p2 = w.create("p2");
  const Entity c = w.create("c", p1);
  w.get<Transform>(p1) = trs({1, 2, 3}, 30, {0, 0, 1}, {2, 2, 2});
  w.get<Transform>(p2) = trs({-4, 0, 1}, -60, {1, 1, 0}, {0.5f, 0.5f, 0.5f});
  w.get<Transform>(c) = trs({1, 0, 0}, 15, {0, 1, 0}, {1, 1, 1});

  const glm::mat4 world = w.computeWorldMatrix(c);
  REQUIRE(w.setParent(c, p2));  // KeepWorld by default
  CHECK(approx(w.computeWorldMatrix(c), world));
  CHECK(w.parent(c) == p2);
  CHECK(w.childCount(p1) == 0);

  REQUIRE(w.setParent(c, kNullEntity));
  CHECK(approx(w.computeWorldMatrix(c), world));
  CHECK(approx(w.get<Transform>(c).matrix(), world));
  CHECK(w.roots() == std::vector<Entity>{p1, p2, c});

  const Transform local = w.get<Transform>(c);
  REQUIRE(w.setParent(c, p1, World::Reparent::KeepLocal));
  CHECK(w.get<Transform>(c).position == local.position);
  CHECK(approx(w.computeWorldMatrix(c), w.get<Transform>(p1).matrix() * local.matrix()));
  checkInvariants(w);
}

TEST_CASE("world: reparent under a singular parent keeps the local transform",
          "[ecs][hierarchy]") {
  World w;
  const Entity flat = w.create("flat");
  w.get<Transform>(flat).scale = {1, 0, 1};
  const Entity c = w.create("c");
  w.get<Transform>(c).position = {3, 4, 5};
  REQUIRE(w.setParent(c, flat));
  CHECK(w.get<Transform>(c).position == glm::vec3(3, 4, 5));
  CHECK_FALSE(w.setWorldMatrix(c, glm::mat4(1.0f)));
  checkInvariants(w);
}

TEST_CASE("world: setParent rejects cycles and bad siblings untouched",
          "[ecs][hierarchy]") {
  World w;
  const Entity a = w.create("a");
  const Entity b = w.create("b", a);
  const Entity c = w.create("c", b);
  const Entity other = w.create("other");
  const std::vector<Entity> order = w.preOrder();
  const Transform tb = w.get<Transform>(b);

  CHECK_FALSE(w.setParent(a, a));          // self
  CHECK_FALSE(w.setParent(a, c));          // own descendant
  CHECK_FALSE(w.setParent(b, c));
  CHECK_FALSE(w.setParent(b, other, World::Reparent::KeepWorld, c));  // before not a child of other
  CHECK_FALSE(w.setParent(b, a, World::Reparent::KeepWorld, b));      // before == child
  CHECK_FALSE(w.setParent(static_cast<Entity>(999), a));
  CHECK_FALSE(w.setParent(b, static_cast<Entity>(999)));
  CHECK(w.preOrder() == order);
  CHECK(w.get<Transform>(b).position == tb.position);
  checkInvariants(w);
}

TEST_CASE("world: sibling insertion and reordering", "[ecs][hierarchy]") {
  World w;
  const Entity p = w.create("p");
  const Entity x = w.create("x", p);
  const Entity y = w.create("y", p);
  const Entity z = w.create("z", p);

  REQUIRE(w.setParent(z, p, World::Reparent::KeepLocal, x));  // move to front
  CHECK(w.children(p) == std::vector<Entity>{z, x, y});
  REQUIRE(w.setParent(z, p, World::Reparent::KeepLocal));     // back to the end
  CHECK(w.children(p) == std::vector<Entity>{x, y, z});
  REQUIRE(w.setParent(x, p, World::Reparent::KeepLocal, z));  // middle
  CHECK(w.children(p) == std::vector<Entity>{y, x, z});
  const Entity r = w.create("r");
  REQUIRE(w.setParent(r, kNullEntity, World::Reparent::KeepLocal, p));  // root order
  CHECK(w.roots() == std::vector<Entity>{r, p});
  checkInvariants(w);
}

TEST_CASE("world: setWorldMatrix lands exactly under a parent", "[ecs][hierarchy]") {
  World w;
  const Entity p = w.create("p");
  const Entity c = w.create("c", p);
  w.get<Transform>(p) = trs({2, 0, 0}, 45, {0, 0, 1}, {3, 3, 3});
  const glm::mat4 target = trs({7, -1, 2}, 70, {1, 2, 3}, {1, 2, 1}).matrix();
  // Non-uniform child scale under a uniformly scaled parent: TRS-exact.
  REQUIRE(w.setWorldMatrix(c, target));
  CHECK(approx(w.computeWorldMatrix(c), target, 1e-3f));
  CHECK_FALSE(w.setWorldMatrix(c, glm::mat4(0.0f)));
}

// ── Destruction ────────────────────────────────────────────────

TEST_CASE("world: destroy removes the subtree and relinks siblings",
          "[ecs][destroy]") {
  World w;
  const Entity p = w.create("p");
  const Entity first = w.create("first", p);
  const Entity middle = w.create("middle", p);
  const Entity last = w.create("last", p);
  const Entity grandchild = w.create("gc", middle);
  const Guid gcGuid = w.guid(grandchild);
  w.add<Light>(grandchild);

  w.destroy(middle);
  CHECK_FALSE(w.valid(middle));
  CHECK_FALSE(w.valid(grandchild));
  CHECK(w.find(gcGuid) == kNullEntity);
  CHECK(w.children(p) == std::vector<Entity>{first, last});
  CHECK(w.registry().view<const Light>().size() == 0);
  checkInvariants(w);

  w.destroy(first);
  CHECK(w.children(p) == std::vector<Entity>{last});
  w.destroy(last);
  CHECK(w.childCount(p) == 0);
  CHECK(w.firstChild(p) == kNullEntity);
  checkInvariants(w);

  w.destroy(middle);  // stale handle: ignored
  CHECK(w.size() == 1);
}

TEST_CASE("world: stale handles stay invalid after slot reuse", "[ecs][destroy]") {
  World w;
  const Entity a = w.create("a");
  w.destroy(a);
  const Entity b = w.create("b");  // entt may reuse a's slot
  CHECK_FALSE(w.valid(a));
  CHECK(w.valid(b));
  CHECK(a != b);
  CHECK(w.tryGet<Name>(a) == nullptr);
  CHECK_FALSE(w.has<Name>(a));

  w.clear();
  CHECK(w.size() == 0);
  CHECK_FALSE(w.valid(b));
  CHECK(w.roots().empty());
  const Entity c = w.create("c");
  CHECK(w.roots() == std::vector<Entity>{c});
  checkInvariants(w);
}

TEST_CASE("world: queued destruction waits for the flush", "[ecs][destroy]") {
  World w;
  const Entity p = w.create("p");
  for (int i = 0; i < 10; ++i) {
    const Entity e = w.create("e", p);
    w.add<Light>(e).intensity = static_cast<f32>(i);
  }
  w.create("child_of_child", w.children(p)[0]);
  // Destroy from inside a view: queued, so the iteration is unaffected.
  int visited = 0;
  for (const Entity e : w.view<Light>()) {
    ++visited;
    if (w.get<Light>(e).intensity >= 5.0f) w.queueDestroy(e);
  }
  CHECK(visited == 10);
  CHECK(w.size() == 12);
  w.queueDestroy(w.children(p)[0]);  // has a child
  w.queueDestroy(p);                 // ancestor of everything queued
  w.queueDestroy(p);                 // twice
  CHECK(w.flushDestroyQueue() == 12);
  CHECK(w.size() == 0);
  CHECK(w.flushDestroyQueue() == 0);
  checkInvariants(w);
}

// ── Identity ───────────────────────────────────────────────────

TEST_CASE("world: guid identity and lookup", "[ecs][guid]") {
  World w;
  const Guid g = Guid::generate();
  const Entity e = w.createWithGuid(g, "loaded");
  REQUIRE(e != kNullEntity);
  CHECK(w.guid(e) == g);
  CHECK(w.find(g) == e);
  CHECK(w.createWithGuid(g, "dup") == kNullEntity);
  CHECK(w.createWithGuid(Guid{}, "null") == kNullEntity);
  CHECK(w.size() == 1);
  CHECK(w.find(Guid::generate()) == kNullEntity);
  const Entity other = w.create("other");
  CHECK(w.guid(other) != g);
}

// ── Randomized model check ─────────────────────────────────────

TEST_CASE("world: random create/reparent/destroy matches a reference model",
          "[ecs][hierarchy][stress]") {
  World w;
  // Reference: ordered child lists keyed by parent (kNullEntity = roots).
  std::map<Entity, std::vector<Entity>> kids;
  std::map<Entity, Entity> parentOf;
  std::vector<Entity> alive;
  std::mt19937 rng(1234);
  auto pick = [&](usize n) { return std::uniform_int_distribution<usize>(0, n - 1)(rng); };
  auto modelDescendant = [&](Entity e, Entity ancestor) {
    for (Entity p = parentOf[e]; p != kNullEntity; p = parentOf[p])
      if (p == ancestor) return true;
    return false;
  };

  for (int step = 0; step < 3000; ++step) {
    const usize op = alive.empty() ? 0 : pick(10);
    if (op < 4) {  // create
      const Entity parent = alive.empty() || pick(3) == 0 ? kNullEntity : alive[pick(alive.size())];
      const Entity e = w.create("n", parent);
      REQUIRE(e != kNullEntity);
      alive.push_back(e);
      parentOf[e] = parent;
      kids[parent].push_back(e);
    } else if (op < 8) {  // reparent, sometimes invalidly
      const Entity child = alive[pick(alive.size())];
      const Entity parent = pick(4) == 0 ? kNullEntity : alive[pick(alive.size())];
      std::vector<Entity>& target = kids[parent];
      Entity before = kNullEntity;
      if (!target.empty() && pick(2) == 0) before = target[pick(target.size())];
      if (pick(10) == 0) before = alive[pick(alive.size())];  // maybe not a sibling
      const bool expectOk = parent != child &&
                            (parent == kNullEntity || !modelDescendant(parent, child)) &&
                            before != child &&
                            (before == kNullEntity || parentOf[before] == parent);
      const World::Reparent mode =
          pick(2) ? World::Reparent::KeepWorld : World::Reparent::KeepLocal;
      REQUIRE(w.setParent(child, parent, mode, before) == expectOk);
      if (expectOk) {
        auto& old = kids[parentOf[child]];
        old.erase(std::find(old.begin(), old.end(), child));
        auto& dst = kids[parent];
        dst.insert(before == kNullEntity ? dst.end() : std::find(dst.begin(), dst.end(), before),
                   child);
        parentOf[child] = parent;
      }
    } else {  // destroy a subtree
      const Entity e = alive[pick(alive.size())];
      std::vector<Entity> doomed{e};
      for (usize i = 0; i < doomed.size(); ++i)
        for (const Entity c : kids[doomed[i]]) doomed.push_back(c);
      auto& siblings = kids[parentOf[e]];
      siblings.erase(std::find(siblings.begin(), siblings.end(), e));
      for (const Entity d : doomed) {
        kids.erase(d);
        parentOf.erase(d);
        alive.erase(std::find(alive.begin(), alive.end(), d));
      }
      w.destroy(e);
      for (const Entity d : doomed) REQUIRE_FALSE(w.valid(d));
    }

    if (step % 25 == 0 || step == 2999) {
      checkInvariants(w);
      REQUIRE(w.size() == alive.size());
      REQUIRE(w.roots() == kids[kNullEntity]);
      for (const Entity e : alive) {
        REQUIRE(w.parent(e) == parentOf[e]);
        REQUIRE(w.children(e) == kids[e]);
      }
    }
  }
}

// ── Reflection ─────────────────────────────────────────────────

namespace {
ComponentRegistry builtinRegistry() {
  ComponentRegistry r;
  registerBuiltinComponents(r);
  return r;
}
}  // namespace

TEST_CASE("reflection: built-in components are described", "[ecs][reflection]") {
  const ComponentRegistry r = builtinRegistry();
  REQUIRE(r.all().size() == 4);
  CHECK(r.all()[0]->name == "transform");
  CHECK(r.find("light") == r.find<Light>());
  CHECK(r.find<Camera>()->name == "camera");
  CHECK(r.find("nope") == nullptr);
  CHECK(r.find<Hierarchy>() == nullptr);
  CHECK_FALSE(r.find<Transform>()->optional);
  CHECK(r.find<MeshRenderer>()->optional);

  const ComponentInfo& light = *r.find<Light>();
  REQUIRE(light.findField("type"));
  CHECK(light.findField("type")->type == FieldType::Enum);
  CHECK(light.findField("type")->enumByName("spot")->value == 2);
  CHECK(light.findField("color")->type == FieldType::Color3);
  CHECK(*light.findField("intensity")->min == 0.0f);
  CHECK(r.find<Transform>()->findField("rotation")->type == FieldType::Quat);
  CHECK(r.find<MeshRenderer>()->findField("mesh")->type == FieldType::Guid);
}

TEST_CASE("reflection: field reads and writes reach the real members",
          "[ecs][reflection]") {
  const ComponentRegistry r = builtinRegistry();
  World w;
  const Entity e = w.create("e");
  w.add<Light>(e);
  w.add<Camera>(e);
  w.add<MeshRenderer>(e);

  // Write a distinct valid value to every field, then read all back:
  // catches fields wired to the wrong member.
  std::map<std::string, FieldValue> written;
  int n = 0;
  for (const auto& comp : r.all()) {
    for (const FieldInfo& f : comp->fields) {
      ++n;
      FieldValue v;
      const f32 nf = static_cast<f32>(n);
      switch (f.type) {
        case FieldType::Bool: v = (n % 2) == 0; break;
        case FieldType::Int: v = n; break;
        case FieldType::Enum: v = f.enumEntries.back().value; break;
        case FieldType::Float: v = (f.min ? *f.min : 0.0f) + 0.25f * static_cast<f32>(n % 7) + 0.5f; break;
        case FieldType::Vec2: v = glm::vec2(nf, nf + 1); break;
        case FieldType::Vec3:
        case FieldType::Color3: v = glm::vec3(nf, nf + 1, nf + 2) * 0.1f; break;
        case FieldType::Vec4: v = glm::vec4(nf); break;
        case FieldType::Quat: v = glm::angleAxis(0.1f * nf, glm::vec3(0, 0, 1)); break;
        case FieldType::String: v = std::string("s") + std::to_string(n); break;
        case FieldType::Guid: v = Guid::generate(); break;
      }
      if (f.max && v.index() == 2) v = std::min(std::get<f32>(v), *f.max);
      std::string err;
      INFO(comp->name << "." << f.name << ": " << err);
      REQUIRE(setField(w, e, *comp, f, v, &err));
      written[comp->name + "." + f.name] = v;
    }
  }
  for (const auto& comp : r.all())
    for (const FieldInfo& f : comp->fields) {
      INFO(comp->name << "." << f.name);
      const auto v = getField(w, e, *comp, f);
      REQUIRE(v);
      const FieldValue& want = written[comp->name + "." + f.name];
      if (f.type == FieldType::Quat)
        CHECK(nearRotation(std::get<glm::quat>(*v), std::get<glm::quat>(want)));
      else
        CHECK(*v == want);
    }
  // Spot checks through the typed API.
  CHECK(w.get<Light>(e).type == Light::Type::Spot);
  CHECK(w.get<Camera>(e).projection == Camera::Projection::Orthographic);
  CHECK(w.get<MeshRenderer>(e).mesh == std::get<Guid>(written["mesh_renderer.mesh"]));
  CHECK(w.get<Light>(e).color == std::get<glm::vec3>(written["light.color"]));
}

TEST_CASE("reflection: invalid writes are rejected and change nothing",
          "[ecs][reflection]") {
  const ComponentRegistry r = builtinRegistry();
  World w;
  const Entity e = w.create("e");
  w.add<Light>(e);
  const ComponentInfo& light = *r.find<Light>();
  const ComponentInfo& camera = *r.find<Camera>();
  const ComponentInfo& transform = *r.find<Transform>();
  const Light before = w.get<Light>(e);
  std::string err;

  CHECK_FALSE(setField(w, e, light, *light.findField("intensity"), FieldValue(1), &err));
  CHECK(err == "light.intensity expects float, got int");
  CHECK_FALSE(setField(w, e, light, *light.findField("intensity"), FieldValue(-1.0f), &err));
  CHECK(err.find("below the minimum") != std::string::npos);
  CHECK_FALSE(setField(w, e, light, *light.findField("intensity"),
                       FieldValue(std::numeric_limits<f32>::infinity()), &err));
  CHECK(err == "light.intensity must be finite");
  CHECK_FALSE(setField(w, e, light, *light.findField("color"),
                       FieldValue(glm::vec3(1, -0.1f, 1)), &err));
  CHECK_FALSE(setField(w, e, light, *light.findField("outer_cone_deg"), FieldValue(91.0f), &err));
  CHECK(err.find("above the maximum") != std::string::npos);
  CHECK_FALSE(setField(w, e, light, *light.findField("type"), FieldValue(7), &err));
  CHECK(err == "light.type: 7 is not one of [directional, point, spot]");
  CHECK(w.get<Light>(e).intensity == before.intensity);
  CHECK(w.get<Light>(e).type == before.type);
  CHECK(w.get<Light>(e).color == before.color);

  CHECK_FALSE(setField(w, e, camera, *camera.findField("primary"), FieldValue(true), &err));
  CHECK(err == "entity has no 'camera' component");
  CHECK_FALSE(setField(w, static_cast<Entity>(4242), light, *light.findField("range"),
                       FieldValue(1.0f), &err));
  CHECK(err == "invalid entity");

  const FieldInfo& rot = *transform.findField("rotation");
  CHECK_FALSE(setField(w, e, transform, rot, FieldValue(glm::quat(0, 0, 0, 0)), &err));
  REQUIRE(setField(w, e, transform, rot, FieldValue(glm::quat(2, 0, 0, 0)), &err));
  CHECK(w.get<Transform>(e).rotation == glm::quat(1, 0, 0, 0));  // normalized
}

TEST_CASE("reflection: add/remove only for optional components", "[ecs][reflection]") {
  const ComponentRegistry r = builtinRegistry();
  World w;
  const Entity e = w.create("e");
  const ComponentInfo& cam = *r.find<Camera>();
  const ComponentInfo& transform = *r.find<Transform>();

  CHECK_FALSE(cam.has(w, e));
  CHECK(cam.getConst(w, e) == nullptr);
  CHECK(cam.add(w, e));
  CHECK(cam.has(w, e));
  CHECK_FALSE(cam.add(w, e));  // already present
  CHECK(w.get<Camera>(e).fovYDeg == 60.0f);
  CHECK(cam.remove(w, e));
  CHECK_FALSE(cam.remove(w, e));
  CHECK_FALSE(w.has<Camera>(e));

  CHECK(transform.has(w, e));
  CHECK_FALSE(transform.add(w, e));
  CHECK_FALSE(transform.remove(w, e));
  CHECK(w.has<Transform>(e));
  CHECK_FALSE(cam.add(w, static_cast<Entity>(4242)));
}

TEST_CASE("reflection: custom components register through the same builder",
          "[ecs][reflection]") {
  struct Health {
    f32 current = 100.0f;
    i32 lives = 3;
    std::string tag;
  };
  ComponentRegistry r;
  registerBuiltinComponents(r);
  r.add<Health>("health", "game component")
      .field<&Health::current>("current", {.min = 0.0f})
      .field<&Health::lives>("lives", {.min = 0.0f, .max = 9.0f})
      .field<&Health::tag>("tag");
  const ComponentInfo& h = *r.find("health");
  CHECK(h.optional);
  CHECK(h.fields.size() == 3);
  CHECK(h.findField("lives")->type == FieldType::Int);

  World w;
  const Entity e = w.create("e");
  REQUIRE(h.add(w, e));
  CHECK(setField(w, e, h, *h.findField("tag"), FieldValue(std::string("boss"))));
  CHECK(w.get<Health>(e).tag == "boss");
  CHECK_FALSE(setField(w, e, h, *h.findField("lives"), FieldValue(10)));
  CHECK(w.get<Health>(e).lives == 3);
}

// Hidden (run with "[bench]"): scale numbers for the roadmap log.
TEST_CASE("world: 100k-entity build/update/destroy timing", "[.][bench]") {
  using Clock = std::chrono::steady_clock;
  auto ms = [](Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  World w;
  std::mt19937 rng(5);
  std::vector<Entity> all;
  all.reserve(100000);
  const auto t0 = Clock::now();
  for (int i = 0; i < 100000; ++i) {
    const Entity parent = all.empty() || i % 8 == 0
                              ? kNullEntity
                              : all[std::uniform_int_distribution<usize>(0, all.size() - 1)(rng)];
    all.push_back(w.create("e", parent));
    w.get<Transform>(all.back()).position = {0.01f, 0, 0};
  }
  const auto t1 = Clock::now();
  w.updateTransforms();
  const auto t2 = Clock::now();
  w.updateTransforms();
  const auto t3 = Clock::now();
  for (const Entity r : w.roots()) w.queueDestroy(r);
  w.flushDestroyQueue();
  const auto t4 = Clock::now();
  WARN("create " << ms(t0, t1) << " ms, updateTransforms " << ms(t1, t2) << " / "
                 << ms(t2, t3) << " ms, destroy " << ms(t3, t4) << " ms");
  CHECK(w.size() == 0);
}
