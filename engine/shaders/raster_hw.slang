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
//                         [4] = (width, height, clusterCount,
//                                indexCount), [5].x = positionCount
//                         (every payload-driven index is CLAMPED into
//                         these bounds - the driver device-losts on
//                         multi-level payload-dependent addressing
//                         chains, the decode pass's documented
//                         workaround)
//   1: uint2 hwPairs[]  - the routed (instanceIdx, clusterIdx) list
//   2: GpuCluster clusters[] - 48B, mirrors packGpuCluster
//   3: float4 positions[] - shared mesh-local vertex pool (M6.3 paged:
//                          the PAGE SLOT POOL)
//   4: uint indices[]   - shared index pool (M6.3 paged: the slot pool)
//   5: GpuInstance instances[] - 96B (model transforms)
//   6: uint64 visBuffer[width*height] - RW, the PS's atomic max
//   7: uint clusterPage[] - M6.3 paged: cluster -> page (v2: dummy)
//   8: uint pageTable[]   - M6.3 paged: page -> slot, 0xFFFFFFFF =
//                          not resident (v2: dummy)
//   9: GpuPageHeader pageHeader[] - M6.3 paged section bases (v2: dummy)
//  10: uint pending[1]  - M6.3 RW: non-resident skip count (v2: dummy)
//
// M6.3 paged mode (params[5].y != 0): bindings 3/4 are the slot pool,
// cluster offsets are page-local, reads add the slot-strided section
// base - identical VALUES to the v2 pools when all pages are resident.
// A non-resident page keeps the VS's initialized degenerate output
// (zero-area triangle, no pixels) and counts into pending[0]. The
// existing clamps stay meaningful: params[4].zw / [5].x carry the SLOT
// POOL totals in paged mode (coarser but still in-bounds).

static const uint kMaxTrisPerCluster = 128;  // matches the builder cap

// kPageSlotSize (128 KB) in u32 elements - mirrors rasterize_sw / CPU
// (v5: all page sections are u32-addressed).
static const uint kSlotU32s = 32768;

struct GpuPageHeader  // M7.1 v5: 12 u32, uniform u32 units
{
  uint posBase;
  uint nrmBase;
  uint uvBase;
  uint idxBase;
  uint metaBase;
  uint vertexCount;
  uint indexCount;   // idx section BYTE total
  uint clusterCount; // meta array bound
  float posStep;
  uint flags;
  uint pad0;
  uint pad1;
};

struct GpuClusterQuant  // 10 u32, mirrors ClusterQuantMeta
{
  int posStartX;
  int posStartY;
  int posStartZ;
  uint widths;
  uint posWordOff;
  float uvMinX;
  float uvMinY;
  float uvRangeX;
  float uvRangeY;
  uint reserved;
};

// v5 decode helpers (CPU mirror = ClusterQuant.cpp, op-for-op - the D5
// bit-identity contract). VALUE path only - never feeds addresses.
uint loadBits2(uint w0, uint w1, uint bitIdx, uint bits) {
    const uint shift = bitIdx & 31u;
  const uint mask = (1u << bits) - 1u;  // bits <= 24
  if (shift + bits <= 32u) return (w0 >> shift) & mask;
  return ((w0 >> shift) | (w1 << (32u - shift))) & mask;
}

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
[[vk::binding(3)]] StructuredBuffer<uint> positions;  // v2: float4-as-4-u32; paged v5: the slot pool bit stream
[[vk::binding(4)]] StructuredBuffer<uint> indices;
[[vk::binding(5)]] StructuredBuffer<GpuInstance> instances;
[[vk::binding(6)]] RWStructuredBuffer<uint64_t> visBuffer;
[[vk::binding(7)]] StructuredBuffer<uint> clusterPage;
[[vk::binding(8)]] StructuredBuffer<uint> pageTable;
[[vk::binding(9)]] StructuredBuffer<GpuPageHeader> pageHeader;
[[vk::binding(10)]] RWStructuredBuffer<uint> pending;
[[vk::binding(11)]] StructuredBuffer<uint> clusterMeta;  // v5: per-cluster ordinal into its page's meta array

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

  // Clamp EVERY payload-driven index into its pool's proven range
  // (params[4].zw / [5].xz): this driver device-losts on the
  // multi-level addressing chain, the decode pass's workaround.
  const uint2 item = hwPairs[instanceId];
  const uint clusterCount = max(uint(params[4].z), 1u);
  const uint indexCount = max(uint(params[4].w), 1u);
  const uint positionCount = max(uint(params[5].x), 1u);
  const uint instanceCount = max(uint(params[5].z), 1u);
  const uint2 itemC =
      uint2(min(item.x, instanceCount - 1u), min(item.y, clusterCount - 1u));
  const GpuCluster cluster = clusters[itemC.y];
  const uint tri = vertexId / 3u;
  const uint corner = vertexId - tri * 3u;

  // v5 paged addressing: slot-strided u32 bases + the cluster's
  // quantization record, or the initialized degenerate output + a
  // pending count when the page is not resident.
  uint posBase = 0, idxBase = 0;
  GpuClusterQuant qm;
  float posStep = 1.0;
  uint nV = cluster.vertexCount;
  if (uint(params[5].y) != 0u) {
    const uint page = clusterPage[itemC.y];
    const uint slot = pageTable[page];
    if (slot == 0xFFFFFFFFu) {
      InterlockedAdd(pending[0], 1);
      return o;
    }
    const GpuPageHeader ph = pageHeader[page];
    const uint pool = slot * kSlotU32s;
    posBase = pool + ph.posBase;
    idxBase = pool + ph.idxBase;
    posStep = ph.posStep;
    const uint ord = min(clusterMeta[itemC.y], ph.clusterCount - 1u);
    // The 10-word meta record must also fit the pool's u32 capacity
    // (decode's metaEnd discipline - pageHeader fields are upload data,
    // not payload-derived, but the clamp costs nothing).
    const uint mb = min(pool + ph.metaBase + ord * 10u, positionCount - 10u);
    qm.posStartX = int(positions[mb]);
    qm.posStartY = int(positions[mb + 1u]);
    qm.posStartZ = int(positions[mb + 2u]);
    qm.widths = positions[mb + 3u];
    qm.posWordOff = positions[mb + 4u];
    qm.uvMinX = asfloat(positions[mb + 5u]);
    qm.uvMinY = asfloat(positions[mb + 6u]);
    qm.uvRangeX = asfloat(positions[mb + 7u]);
    qm.uvRangeY = asfloat(positions[mb + 8u]);
    qm.reserved = positions[mb + 9u];
  }

  if (tri < min(cluster.indexCount, 3u * kMaxTrisPerCluster) / 3u) {
    // NOTE: glm stores column-major and Slang reads the buffer vec4s
    // as rows - every transform below is an explicit combination of
    // the stored columns (the project-wide convention).
    const float4x4 model = instances[itemC.x].model;
    const float4x4 vpCols = float4x4(params[0], params[1], params[2],
                                     params[3]);
    float2 q[3];
    for (uint k = 0; k < 3u; ++k) {
      float3 lp;
      if (uint(params[5].y) != 0u) {
        // v5: u8 index lane at the cluster's index BYTE offset (u32
        // math only - this driver's Slang vertex-stage emit chokes on
        // u64 locals).
        const uint byteOff = cluster.indexOffset + tri * 3u + k;
        const uint lane = byteOff & 3u;
        const uint w0 = indices[min(idxBase + byteOff / 4u, indexCount - 1u)];
        uint vi8;
        if (lane == 0u) vi8 = w0 & 0xFFu;
        else if (lane == 1u) vi8 = (w0 >> 8u) & 0xFFu;
        else if (lane == 2u) vi8 = (w0 >> 16u) & 0xFFu;
        else vi8 = (w0 >> 24u) & 0xFFu;
        // v5: three SOA bit fields on the mesh's integer grid (page
        // offsets fit 32 bits: < 128 KB -> < 2^21 bits).
        const uint b0 = qm.widths & 0xFu;
        const uint b1 = (qm.widths >> 4) & 0xFu;
        const uint b2 = (qm.widths >> 8) & 0xFu;
        const uint base = qm.posWordOff * 32u;
        const uint bitA = base + vi8 * b0;
        const uint bitB = base + nV * b0 + vi8 * b1;
        const uint bitC = base + nV * (b0 + b1) + vi8 * b2;
        const uint qa = loadBits2(positions[min(posBase + bitA / 32u, positionCount - 1u)],
                                  positions[min(posBase + bitA / 32u + 1u, positionCount - 1u)],
                                  bitA & 31u, b0);
        const uint qb = loadBits2(positions[min(posBase + bitB / 32u, positionCount - 1u)],
                                  positions[min(posBase + bitB / 32u + 1u, positionCount - 1u)],
                                  bitB & 31u, b1);
        const uint qc = loadBits2(positions[min(posBase + bitC / 32u, positionCount - 1u)],
                                  positions[min(posBase + bitC / 32u + 1u, positionCount - 1u)],
                                  bitC & 31u, b2);
        const int3 iq = int3(int(qa), int(qb), int(qc)) +
                        int3(qm.posStartX, qm.posStartY, qm.posStartZ);
        lp = float3(iq) * posStep;
      } else {
        // v2: float4 elements as 4 u32 (w unused). Clamp in u32 WORD
        // units - the capacity is the pool's word count; the old form
        // clamped the vec4 ORDINAL against the 4x-larger word bound,
        // admitting word reads up to 16N in a 4N pool.
        const uint ii = min(idxBase + cluster.indexOffset + tri * 3u + k,
                            indexCount - 1u);
        const uint vw = min((indices[ii] + posBase + cluster.vertexOffset) * 4u,
                            positionCount - 3u);
        lp = float3(asfloat(positions[vw]), asfloat(positions[vw + 1u]),
                    asfloat(positions[vw + 2u]));
      }
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
    // The wide-instance payload (rasterize_sw's packing): this draw's
    // instanceId IS the hwPairs index; listSel 1 marks the HW table
    // for the decode indirection.
    o.payload = ((instanceId & 0x7FFFFFu) << 9) | (1u << 7) | (tri & 0x7Fu);
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
  // The pixel center at subpixel (x*256 + 128). SV_Position arrives
  // at the CENTER (x + 0.5): int-convert first to drop the half, then
  // add the half-pixel in subpixel units (P0 fix: the old form
  // evaluated at (x+0.5)*256+128 = the pixel CORNER - every edge test
  // off by half a pixel).
  const int64_t px = int64_t(i.pos.x - 0.5) * 256 + 128;
  const int64_t py = int64_t(i.pos.y - 0.5) * 256 + 128;
  const int64_t w0 = e0x * (py - Y0) - e0y * (px - X0);
  const int64_t w1 = e1x * (py - Y1) - e1y * (px - X1);
  const int64_t w2 = e2x * (py - Y2) - e2y * (px - X2);
  const int64_t t0 = (e0y == 0 && e0x > 0) || e0y < 0 ? 0 : 1;
  const int64_t t1 = (e1y == 0 && e1x > 0) || e1y < 0 ? 0 : 1;
  const int64_t t2 = (e2y == 0 && e2x > 0) || e2y < 0 ? 0 : 1;
  if (!(w0 >= t0 && w1 >= t1 && w2 >= t2))
    return;  // helper lanes / tie pixels owned by the other side

  const uint W = uint(params[4].x);
  const uint H = max(uint(params[4].y), 1u);
  const uint2 pixel =
      uint2(min(uint(i.pos.x), uint(params[4].x) - 1u),
            min(uint(i.pos.y), H - 1u));
  const float z = clamp(i.pos.z, 0.0, 1.0);
  const uint depthBits = uint((1.0 - z) * 4294967295.0);
  const uint64_t packed = (uint64_t(depthBits) << 32) | uint64_t(i.payload);
  InterlockedMax(visBuffer[uint64_t(pixel.y) * W + pixel.x], packed);
  // Count the passing pixel like rasterize_sw's counters[1] (the panel's
  // pxWrites = main + post + HW; without this an all-HW-routed cut
  // reports "writes 0" and reads as a dead draw - a diagnosis trap that
  // cost an hour on 2026-09-19).
  InterlockedAdd(pending[1], 1);
}
