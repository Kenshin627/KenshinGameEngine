#pragma once

// ── ImposterBaker (M9.2) — the far-field Visibility Buffer imposter ─
// nanite.md §14 / Karis Slide 97: when a whole instance covers only a
// few pixels the DAG's cost floor (the 128-tri root cluster) stops
// scaling with distance. The imposter replaces those instances with a
// prebaked atlas: 12x12 view directions (octahedron-mapped), 12x12
// orthographic pixels each, storing the ROOT CLUSTER's depth +
// triangle id in the VisBuffer payload layout - so the runtime decode
// keeps material remapping and every VisBuffer property for free.
//
// The bake is CPU (the root cluster is <= 128 triangles; 144
// directions x 144 texels is trivial) and mirrors the GPU packing:
// depthBits = uint((1-z) * 2^32-1) in the high word, triangle index
// in the low 7 bits (the pair index is per-frame runtime state - the
// bake leaves it zero and the injection rewrites it).
//
// Storage: 12x12 atlas of 12x12 RG32_UINT texels:
//   .r = depth bits (reversed, matching packPixel)
//   .g = triangle index (7b) of the mesh's root-cluster slice;
//        0xFFFFFFFF = empty (missed)

#include <cstdint>
#include <vector>

#include "kenshin/Common.h"
#include "kenshin/geometry/ClusterDag.h"

KENSHIN_NS_BEGIN

namespace geometry {

struct ImposterAtlas {
  static constexpr u32 kDirs = 12;   // per octahedral axis
  static constexpr u32 kTexels = 12; // per direction tile
  std::vector<u32> depth;   // kDirs*kDirs * kTexels*kTexels (reversed bits)
  std::vector<u32> tri;     // same layout; 0xFFFFFFFF = miss
};

/// Octahedral map [-1,1]^2 -> unit direction (the standard
/// octEncode inverse; UE's 12x12 direction set).
glm::vec3 octDecode(const glm::vec2& uv);

/// Bake `dag`'s root clusters (the coarsest cut) into `out`.
/// `clusterIndices` selects the clusters forming the root cut (the
/// dag's `roots`, or a chosen level); triangles project ortho into the
/// mesh AABB fitted per direction. Returns false on an empty cut.
bool bakeImposterAtlas(const ClusterDag& dag,
                       const std::vector<u32>& clusterIndices,
                       ImposterAtlas& out, std::string& log);

}  // namespace geometry

KENSHIN_NS_END
