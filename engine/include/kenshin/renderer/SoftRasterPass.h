#pragma once

// ── SoftRasterPass (S4.4) ───────────────────────────────────────
// The VisBuffer producer: rasterize_sw.slang, one workgroup per
// visible (instance, cluster) pair, 128 threads (Karis p.85-86).
// Pixel = pack(reversed depth 32b | cluster 25b | triangle 7b),
// InterlockedMax on a u64 VisBuffer (Karis p.84).
//
// The CPU reference in tests/cpu_reference_raster.cpp mirrors the
// shader's float expressions operation-for-operation; both sides are
// pinned by the differential. Params packing lives here so the test
// shares the exact bytes.

#include <glm/glm.hpp>

#include <span>
#include <vector>

#include "kenshin/Common.h"
#include "kenshin/renderer/InstanceCullPass.h"

KENSHIN_NS_BEGIN

namespace backend {
class IRenderBackend;
}
namespace manager {
class ShaderManager;
}

namespace renderer {

/// Frame parameters for the rasterizer. `view` maps world -> view;
/// `viewProj` maps world -> NDC with an UNFLIPPED Y (the rasterizer's
/// row index grows with +Y NDC by convention; the CPU reference
/// shares it). `nearZ` > 0 is the near-plane distance.
struct SoftRasterFrame {
  glm::mat4 view = glm::mat4(1.0f);
  glm::mat4 viewProj = glm::mat4(1.0f);
  f32 nearZ = 0.1f;
  u32 width = 0;
  u32 height = 0;
};

/// Pack the params SSBO (float4[12]): [0..3] view columns,
/// [4..7] viewProj columns (glm raw), [8] = (w, h, pairCount, nearZ).
std::vector<u8> packRasterParams(const SoftRasterFrame& frame,
                                 u32 pairCount);

/// Run the rasterizer (one-shot; blocks). `positions` are mesh-local
/// vec3s, `clusterData` the 48B-packed table. `visOut` returns
/// width*height u64s (0 = nothing won the pixel); `stats[4]` =
/// (coverage is derived on CPU; writes, backfaceCulls, nearClipped,
/// spare).
bool rasterizeOnGpu(backend::IRenderBackend& backend,
                    manager::ShaderManager& shaders,
                    const SoftRasterFrame& frame,
                    std::span<const glm::vec3> positions,
                    std::span<const u32> indices,
                    const std::vector<u8>& clusterData,
                    std::span<const CullGpuInstance> instances,
                    std::span<const glm::uvec2> visiblePairs,
                    std::vector<u64>& visOut, u32 stats[4]);

/// The u64 pixel packing, shared by the CPU reference (identical ops).
inline u64 packVisPixel(f32 z, u32 clusterIdx, u32 triIdx) {
  z = glm::clamp(z, 0.0f, 1.0f);
  const u32 depthBits =
      static_cast<u32>((1.0f - z) * 4294967295.0f);
  return (static_cast<u64>(depthBits) << 32) |
         (static_cast<u64>(clusterIdx) << 7) |
         static_cast<u64>(triIdx & 0x7Fu);
}

}  // namespace renderer

KENSHIN_NS_END
