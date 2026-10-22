#pragma once

// ── BloomPass — energy-conserving HDR bloom (bloom.slang) ───────────
// COD: Advanced Warfare's mip-chain bloom (Jimenez 2014): kLevels
// half-resolution 13-tap downsamples of the resolved HDR, then tent
// upsamples blended additively back up the chain. TonemapPass then
// composites scene * (1 - s) + bloom * s. One pipeline instance per pass
// (a descriptor set may only be written once per frame - the HZB
// per-level precedent); the backend owns the chain targets.

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace backend {
class IRenderBackend;
}
namespace manager {
class ShaderManager;
}

namespace renderer {

class BloomPass {
 public:
  static constexpr u32 kLevels = 6;

  bool init(backend::IRenderBackend& backend, manager::ShaderManager& shaders);
  /// Record the chain over this frame's resolved HDR (the TAA output
  /// when `fromTaa`). Returns false when nothing was recorded.
  bool draw(backend::IRenderBackend& backend, bool fromTaa, u32 width,
            u32 height);
  bool ready() const { return m_ready; }

 private:
  u32 m_down[kLevels] = {};
  u32 m_up[kLevels - 1] = {};
  bool m_ready = false;
};

}  // namespace renderer

KENSHIN_NS_END
