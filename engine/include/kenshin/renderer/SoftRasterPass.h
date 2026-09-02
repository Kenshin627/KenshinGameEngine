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
/// `proj` maps VIEW -> NDC (applied to the phase-1 VIEW-space vertices -
/// packing world->NDC here would run the view transform twice, a bug
/// the S4.4 GPU/CPU differential could not see because both sides
/// shared it; the S4.5 dual-path comparison vs the HW baseline caught
/// it). Y orientation is the CALLER's choice - rows grow with +Y NDC,
/// so the S4.4 differential feeds an unflipped proj while the app feeds
/// the camera's own flipped one (rows then arrive in framebuffer
/// order, which the decode pass indexes directly). `nearZ` > 0 is the
/// near-plane distance.
struct SoftRasterFrame {
  glm::mat4 view = glm::mat4(1.0f);
  glm::mat4 proj = glm::mat4(1.0f);
  f32 nearZ = 0.1f;
  u32 width = 0;
  u32 height = 0;
  /// proj is the camera's own Y-FLIPPED matrix: rows arrive in
  /// framebuffer order and the backface test culls the same half the
  /// HW path culls (CW-as-viewed). False (the S4.4 differential's
  /// convention) keeps the y-up cull sign.
  bool yFlip = false;
};

/// Pack the params SSBO (float4[12]): [0..3] view columns,
/// [4..7] PROJ columns (glm raw), [8] = (w, h, pairCount, nearZ),
/// [9].x = yFlip flag.
std::vector<u8> packRasterParams(const SoftRasterFrame& frame, u32 pairCount);

/// Run the rasterizer (one-shot; blocks). `positions` are the SHARED
/// pool's float4 elements (std430 vec3 arrays stride 16), `clusterData`
/// the 48B-packed table. `visOut` returns width*height u64s (0 =
/// nothing won the pixel); `stats[4]` mirrors the shader counters:
/// [0] unused (coverage derives on CPU), [1] pixel writes, [2] backface
/// culls, [3] near-clipped triangles.
bool rasterizeOnGpu(backend::IRenderBackend& backend,
                    manager::ShaderManager& shaders,
                    const SoftRasterFrame& frame,
                    std::span<const glm::vec4> positions,
                    std::span<const u32> indices,
                    const std::vector<u8>& clusterData,
                    std::span<const CullGpuInstance> instances,
                    std::span<const glm::uvec2> visiblePairs,
                    std::vector<u64>& visOut, u32 stats[4]);

/// The u64 pixel packing, shared by the CPU reference (identical ops).
inline u64 packVisPixel(f32 z, u32 clusterIdx, u32 triIdx) {
  z = glm::clamp(z, 0.0f, 1.0f);
  const u32 depthBits = static_cast<u32>((1.0f - z) * 4294967295.0f);
  return (static_cast<u64>(depthBits) << 32) |
         (static_cast<u64>(clusterIdx) << 7) | static_cast<u64>(triIdx & 0x7Fu);
}

}  // namespace renderer

KENSHIN_NS_END
