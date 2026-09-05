// M4.2 hardware raster path (UE HWRasterizeVS/PS, Karis Slide 87):
// large-triangle clusters (edgePx >= threshold, routed by
// cull_clusters) draw through the FIXED-FUNCTION rasterizer with ZERO
// attachments - the PS writes the same VisBuffer SSBO the software
// rasterizer atomically writes (uint64 max, depth in the high bits),
// so both paths share one output buffer with no merge pass and stay
// async-overlap friendly.
//
// Coverage agreement with the SW rasterizer is enforced IN the PS:
// the VS passes the vertex positions quantized to the same 1/256
// subpixel grid as rasterize_sw (flat varyings), and the PS re-runs
// the exact same edge functions + DX top-left rule before writing.
// That suppresses helper-lane writes (quad neighbors interpolate to
// out-of-triangle barycentrics -> negative edge values) AND removes
// any dependence on the hardware's internal snap semantics. Depth
// bits use SV_Position.z - the hardware's PERSPECTIVE-CORRECT NDC z
// while SW interpolates z affinely; a documented, UE-shared
// divergence (their MicropolyRasterizer is screen-affine too) -
// coverage is identical, near-tie winners can differ by it.
//
// Draw shape (UE's): ONE non-indexed indirect draw, VertexCount fixed
// at 3 * kMaxTrisPerCluster (VertexID/3 = triangle, %3 = corner - no
// index buffer), InstanceCount = routed clusters; the VS degenerates
// triangles beyond the cluster's indexCount.
//
// Bindings (SSBO set 0):
//   0: float4 params[8] - [0..3] viewProj columns (glm raw, the
//                         camera's Y-FLIPPED matrix - framebuffer rows
//                         match the VisBuffer's convention),
//                         [4] = (width, height, 0, 0)
//   1: uint2 hwPairs[]  - the routed (instanceIdx, clusterIdx) list
//   2: GpuCluster clusters[] - 48B, mirrors packGpuCluster
//   3: float4 positions[] - shared mesh-local vertex pool
//   4: uint indices[]   - shared index pool
//   5: GpuInstance instances[] - 96B (model transforms)
//   6: uint64 visBuffer[width*height] - RW, the PS's atomic max

static const uint kMaxTrisPerCluster = 128;  // matches the builder cap

struct GpuCluster {
  float4 bounds;
  float error;
  float parentError;
  uint vertexOffset;
  uint vertexCount;
  uint indexOffset;
  uint indexCount;
  uint group;
  int refinedGroup;
};

struct GpuInstance {
  float4x4 model;
  uint meshId;
  uint _pad0;
  uint _pad1;
  uint _pad2;
  float4 bounds;
};

[[vk::binding(0)]] StructuredBuffer<float4> params;
[[vk::binding(1)]] StructuredBuffer<uint2> hwPairs;
[[vk::binding(2)]] StructuredBuffer<GpuCluster> clusters;
[[vk::binding(3)]] StructuredBuffer<float4> positions;
[[vk::binding(4)]] StructuredBuffer<uint> indices;
[[vk::binding(5)]] StructuredBuffer<GpuInstance> instances;
[[vk::binding(6)]] RWStructuredBuffer<uint64_t> visBuffer;

struct HwVsOut {
  float4 pos : SV_Position;
  nointerpolation uint payload : PAYLOAD;
  // The three vertices in PIXEL space, floored to the 1/256 subpixel
  // grid - the same snap rasterize_sw applies (flat: each fragment
  // receives its provoking vertex's copy of the SAME corner values...
  // no: all three varyings are per-triangle constants because every
  // vertex of the triangle writes the same quantized corner set is
  // impossible - instead each varying interpolates the OTHER two
  // corners. See the VS: q1/q2 are the same at every vertex of the
  // triangle ONLY if flat-interpolated from the same slot - we cheat
  // by emitting them from the SAME corner: all three vertices emit
  // the full (q0, q1, q2) triple through separate varyings, so each
  // varying is constant across the triangle.
  nointerpolation float2 q0 : Q0;
  nointerpolation float2 q1 : Q1;
  nointerpolation float2 q2 : Q2;
};

[shader("vertex")]
HwVsOut vsMain(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID) {
  HwVsOut o;
  o.pos = float4(0.0, 0.0, 0.5, 1.0);
  o.payload = 0u;
  o.q0 = float2(0.0, 0.0);
  o.q1 = float2(0.0, 0.0);
  o.q2 = float2(0.0, 0.0);

  const uint2 item = hwPairs[instanceId];
  const GpuCluster cluster = clusters[item.y];
  const uint tri = vertexId / 3u;
  const uint corner = vertexId - tri * 3u;

  if (tri < cluster.indexCount / 3u) {
    // NOTE: glm stores column-major and Slang reads the buffer vec4s
    // as rows - every transform below is an explicit combination of
    // the stored columns (the project-wide convention).
    const float4x4 model = instances[item.x].model;
    const float4x4 vpCols = float4x4(params[0], params[1], params[2],
                                     params[3]);
    float2 q[3];
    for (uint k = 0; k < 3u; ++k) {
      const uint vi = indices[cluster.indexOffset + tri * 3u + k];
      const float3 lp = positions[cluster.vertexOffset + vi].xyz;
      const float3 wp = model[0].xyz * lp.x + model[1].xyz * lp.y +
                        model[2].xyz * lp.z + model[3].xyz;
      const float4 clip = params[0] * wp.x + params[1] * wp.y +
                          params[2] * wp.z + params[3];
      // All three corners: needed for the payload corner + the PS's
      // quantized triangle. SV_Position of THIS corner; the q's are
      // floored to the 1/256 grid (rasterize_sw's DX x.8 snap).
      const float2 px = float2((clip.x / clip.w * 0.5 + 0.5) * params[4].x,
                               (clip.y / clip.w * 0.5 + 0.5) * params[4].y);
      q[k] = floor(px * 256.0) * (1.0 / 256.0);
      if (k == corner) o.pos = clip;
    }
    // The SAME packPixel payload the SW rasterizer writes: cluster
    // 20b | triangle 7b | instance 5b.
    o.payload = ((item.y & 0xFFFFFu) << 12) | ((tri & 0x7Fu) << 5) |
                (item.x & 0x1Fu);
    // Every vertex emits the full quantized triangle -> each varying
    // is constant across the triangle (flat interpolation of equal
    // values), so the PS sees the exact geometry it needs.
    o.q0 = q[0];
    o.q1 = q[1];
    o.q2 = q[2];
  }
  // Out-of-range triangles keep the degenerate output (all corners at
  // the same NDC point): zero-area, no pixels.
  return o;
}

[shader("fragment")]
void psMain(HwVsOut i) {
  // rasterize_sw's coverage test, verbatim: the quantized triangle's
  // edge functions at the pixel center as EXACT int64 integers (the
  // varyings carry 1/256-exact multiples; x256 makes them integers),
  // ties resolved by the DX top-left rule (top: ey == 0 && ex > 0;
  // left: ey < 0 - winding-canonicalized to positive area below).
  int64_t X0 = int64_t(i.q0.x * 256.0), Y0 = int64_t(i.q0.y * 256.0);
  int64_t X1 = int64_t(i.q1.x * 256.0), Y1 = int64_t(i.q1.y * 256.0);
  int64_t X2 = int64_t(i.q2.x * 256.0), Y2 = int64_t(i.q2.y * 256.0);
  int64_t area2 = (X1 - X0) * (Y2 - Y0) - (Y1 - Y0) * (X2 - X0);
  if (area2 == 0) return;
  if (area2 < 0) {  // canonicalize to positive area (swap v1/v2)
    const int64_t tx = X1, ty = Y1;
    X1 = X2;
    Y1 = Y2;
    X2 = tx;
    Y2 = ty;
  }
  const int64_t e0x = X1 - X0, e0y = Y1 - Y0;
  const int64_t e1x = X2 - X1, e1y = Y2 - Y1;
  const int64_t e2x = X0 - X2, e2y = Y0 - Y2;
  // The pixel center at subpixel (x*256 + 128).
  const int64_t px = int64_t(i.pos.x) * 256 + 128;
  const int64_t py = int64_t(i.pos.y) * 256 + 128;
  const int64_t w0 = e0x * (py - Y0) - e0y * (px - X0);
  const int64_t w1 = e1x * (py - Y1) - e1y * (px - X1);
  const int64_t w2 = e2x * (py - Y2) - e2y * (px - X2);
  const int64_t t0 = (e0y == 0 && e0x > 0) || e0y < 0 ? 0 : 1;
  const int64_t t1 = (e1y == 0 && e1x > 0) || e1y < 0 ? 0 : 1;
  const int64_t t2 = (e2y == 0 && e2x > 0) || e2y < 0 ? 0 : 1;
  if (!(w0 >= t0 && w1 >= t1 && w2 >= t2))
    return;  // helper lanes / tie pixels owned by the other side

  const uint W = uint(params[4].x);
  const uint2 pixel = uint2(i.pos.xy);
  const float z = clamp(i.pos.z, 0.0, 1.0);
  const uint depthBits = uint((1.0 - z) * 4294967295.0);
  const uint64_t packed = (uint64_t(depthBits) << 32) | uint64_t(i.payload);
  InterlockedMax(visBuffer[uint64_t(pixel.y) * W + pixel.x], packed);
}
