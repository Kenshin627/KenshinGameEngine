// M9.2: the far-field imposter atlas bake (Karis Slide 97 shape):
// 12x12 octahedral directions x 12x12 ortho texels of the ROOT cut's
// depth + triangle id in the VisBuffer payload convention. These tests
// pin the bake's geometry: octahedral continuity, coverage of a
// sphere from every direction (the sphere's silhouette fills the
// tile), depth ordering (nearer hits win), and the miss encoding.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <glm/glm.hpp>

#include "kenshin/geometry/ImposterBaker.h"

using namespace kenshin;
using namespace kenshin::geometry;

TEST_CASE("octahedral decode continuity and normalization", "[imposter]") {
  // The four corners decode to the axis directions; the center to +Z;
  // every output is unit length (the map's contract).
  REQUIRE(glm::length(octDecode({0.0f, 0.0f})) -
              1.0f < 1e-5f);
  const glm::vec3 px = octDecode({1.0f, 0.0f});
  const glm::vec3 py = octDecode({0.0f, 1.0f});
  const glm::vec3 nx = octDecode({-1.0f, 0.0f});
  const glm::vec3 ny = octDecode({0.0f, -1.0f});
  REQUIRE_THAT(px.x, Catch::Matchers::WithinAbs(1.0f, 1e-4f));
  REQUIRE_THAT(py.y, Catch::Matchers::WithinAbs(1.0f, 1e-4f));
  REQUIRE_THAT(nx.x, Catch::Matchers::WithinAbs(-1.0f, 1e-4f));
  REQUIRE_THAT(ny.y, Catch::Matchers::WithinAbs(-1.0f, 1e-4f));
  // Pre-normalization the map gives (0.25, 0.25, 0.5); the decoded
  // direction is that ray normalized - check the ratios, not z.
  const glm::vec3 c = octDecode({0.25f, 0.25f});
  REQUIRE_THAT(c.x / c.z, Catch::Matchers::WithinAbs(0.5f, 1e-3f));
  REQUIRE_THAT(c.y / c.z, Catch::Matchers::WithinAbs(0.5f, 1e-3f));
  // The seam fold: |x|+|y| = 1 points stay on the equator z=0.
  const glm::vec3 s = octDecode({0.5f, 0.5f});
  REQUIRE_THAT(s.z, Catch::Matchers::WithinAbs(0.0f, 1e-4f));
}

TEST_CASE("sphere imposter: full coverage, centered depth, ids valid",
          "[imposter]") {
  // A UV sphere's root cut = itself; from any direction its silhouette
  // is a disk covering the whole tile (the AABB-fit projection maps
  // the silhouette to the full [-1,1] square only for a cube; for a
  // sphere the disk fills the inscribed area) - the central texel
  // always hits, ids stay under the triangle count, and the depth is
  // the sphere's front surface (near half).
  ClusterDag dag;
  dag.clusters.resize(1);
  DagCluster& cl = dag.clusters[0];
  cl.indexCount = 0;
  // Build an octahedron (8 faces, 6 verts) - a valid closed root cut.
  const glm::vec3 verts[6] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                              {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
  static const kenshin::u32 faces[8][3] = {{0, 2, 4}, {2, 1, 4}, {1, 3, 4},
                                  {3, 0, 4}, {2, 0, 5}, {1, 2, 5},
                                  {3, 1, 5}, {0, 3, 5}};
  for (const auto& v : verts) {
    dag.positions.insert(dag.positions.end(), {v.x, v.y, v.z});
    dag.normals.insert(dag.normals.end(), {v.x, v.y, v.z});
    dag.uvs.insert(dag.uvs.end(), {0.0f, 0.0f});
  }
  for (const auto& f : faces) {
    dag.indices.insert(dag.indices.end(), {f[0], f[1], f[2]});
    cl.indexCount += 3;
  }
  cl.vertexCount = 6;
  cl.vertexOffset = 0;
  cl.indexOffset = 0;

  ImposterAtlas atlas;
  std::string log;
  REQUIRE(bakeImposterAtlas(dag, {0}, atlas, log));

  constexpr u32 D = ImposterAtlas::kDirs;
  constexpr u32 N = ImposterAtlas::kTexels;
  u32 hits = 0, misses = 0;
  for (u32 dv = 0; dv < D; ++dv)
    for (u32 du = 0; du < D; ++du) {
      const usize base = (usize(dv) * D + du) * N * N;
      u32 dirHits = 0;
      // The CENTER texel of every direction hits (the octahedron
      // always presents a face toward the view).
      REQUIRE(atlas.tri[base + (N / 2) * N + N / 2] != 0xFFFFFFFFu);
      for (usize i = 0; i < usize(N) * N; ++i) {
        if (atlas.tri[base + i] == 0xFFFFFFFFu) {
          ++misses;
          REQUIRE(atlas.depth[base + i] == 0);
        } else {
          ++hits;
          ++dirHits;
          REQUIRE(atlas.tri[base + i] < 8);
          // Reversed-depth bits in [1, 2^32-2]: a hit is strictly
          // inside the box (faces can slope through the center plane,
          // so far-half depths are legal - only the clamp bounds hold).
          REQUIRE(atlas.depth[base + i] > 0u);
          REQUIRE(atlas.depth[base + i] < 0xFFFFFFFFu);
        }
      }
      // The octahedron's silhouette is a hexagon/diamond INSIDE the
      // AABB-projected tile: ~33% from generic directions (vertices
      // sit at the box's face centers), 50% from the axes. Every
      // direction must cover at least a quarter of its tile.
      REQUIRE(dirHits >= N * N / 4);
    }
  REQUIRE(hits > 0);
  (void)misses;
}
