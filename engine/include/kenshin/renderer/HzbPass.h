#pragma once

// ── HzbPass (M3.1) — the occlusion-culling depth pyramid ─────────
// build_hzb.slang min-reduces the VisBuffer's reversed-depth bits
// (the winner's depth IS the per-pixel nearest-surface depth) into a
// 2x2 mip chain; a texel then bounds "the farthest any rendered
// geometry reaches in this tile", which the M3.2 occlusion test
// compares a cluster's nearest depth against.
//
// This header owns the LAYOUT (identical math for host and device)
// and the one-shot GPU driver used by the differential test; the
// in-frame variant joins VisBufferChain in M3.2.

#include <vector>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace backend {
class IRenderBackend;
}
namespace manager {
class ShaderManager;
}

namespace renderer {

/// Pyramid mip layout: level 0 = w x h (the depth image itself, the
/// word count of every level uses ceil division), levels 1.. halve.
struct HzbLayout {
  std::vector<u32> levelOffset;  // texel offset of each level
  std::vector<u32> levelW, levelH;
  u32 levelCount = 0;
  u32 totalTexels = 0;
};
inline HzbLayout hzbLayout(u32 w, u32 h) {
  HzbLayout l;
  u32 lw = w, lh = h;
  while (true) {
    l.levelOffset.push_back(l.totalTexels);
    l.levelW.push_back(lw);
    l.levelH.push_back(lh);
    l.totalTexels += lw * lh;
    ++l.levelCount;
    if (lw == 1 && lh == 1) break;
    lw = (lw + 1) / 2;
    lh = (lh + 1) / 2;
  }
  return l;
}

/// One-shot GPU pyramid build (blocking; test/debug path - the chain
// gets the in-frame variant in M3.2). `vis` is the rasterizer output
// (width*height u64s); `out` receives totalTexels u32s. Levels chain
/// through host readbacks (each one-shot dispatch is self-contained);
/// every step is integer-exact.
bool buildHzbOneshot(backend::IRenderBackend& backend,
                     manager::ShaderManager& shaders,
                     const std::vector<u64>& vis, u32 width, u32 height,
                     std::vector<u32>& out);

}  // namespace renderer

KENSHIN_NS_END
