#pragma once

// ── ImposterPass (M9.2 Stage B) — the tiny-instance injection ──────
// A compute dispatch between the main chain's raster and the decode:
// every tiny instance (projected bounds under the panel threshold)
// covers its screen rect with one workgroup; each thread octahedral-
// maps the view direction into the baked atlas (loaded from the
// .kgeo's v9 section), reconstructs the surface's world position in
// the bake's ortho box, and atomicMaxes the SAME VisBuffer the
// cluster chains write - imposter and geometry compete for pixels
// through identical depth-keyed atomics, so occlusion between them
// is exact.
//
// The payload's listSel=3 routes the decode to the CPU-filled
// imposterPairs table ((instance, rootCluster) - the root cluster is
// a real cluster entry: paged addressing, material ids, UVs all work
// unchanged).

#include <glm/glm.hpp>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace backend {
class IRenderBackend;
}
namespace manager {
class ShaderManager;
}

namespace renderer {

struct ImposterQuad {
  glm::ivec4 rect{0};          // x0,y0,x1,y1 screen pixels
  glm::vec4 centerAtlas{};     // xyz world AABB center, w atlas word base
  glm::vec4 halfPair{};        // xyz AABB half extents, w pair index
  glm::vec4 pad{};
};

class ImposterPass {
 public:
  bool init(backend::IRenderBackend& backend, manager::ShaderManager& shaders);
  void onShutdown(backend::IRenderBackend& backend);

  /// `quads`/`params` are CPU-staged; this writes them into the
  /// pass's host-visible buffers and records the dispatch + the
  /// post-write VisBuffer barrier (the decode reads it next).
  void draw(backend::IRenderBackend& backend, u32 visBuffer,
            const glm::mat4& viewProj, const glm::vec3& camPos, u32 width,
            u32 height, const ImposterQuad* quads, u32 quadCount);

  u32 pairsBuffer() const { return m_pairs; }
  /// `pairs` maps the payload's pairIdx to (instance, rootCluster).
  void setPairs(backend::IRenderBackend& backend, const glm::uvec2* pairs,
                u32 count);
  /// Uploads a mesh's atlas (12*12*12*12 (depth,tri) words); returns
  /// the word base for quad.centerAtlas.w, or kNoAtlas.
  static constexpr u32 kNoAtlas = 0xFFFFFFFFu;
  u32 uploadAtlas(backend::IRenderBackend& backend, const u32* depth,
                  const u32* tri, u32 words);

 private:
  u32 m_pipeline = ~0u;
  u32 m_params = ~0u;
  u32 m_quads = ~0u;
  u32 m_pairs = ~0u;
  u32 m_atlas = ~0u;
  u32 m_atlasWords = 0;
  u32 m_atlasCapacity = 0;
};

}  // namespace renderer

KENSHIN_NS_END
