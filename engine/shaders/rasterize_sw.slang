// S4.4: software rasterizer - the VisBuffer producer.
// Structure per Karis SIGGRAPH 2021 p.85-86: one workgroup per visible
// (instance, cluster) pair, 128 threads; phase 1 transforms vertices to
// VIEW space into groupshared (loop over >128 verts), phase 2 runs one
// thread per triangle - near-plane Sutherland-Hodgman clip (Nanite
// delegates this to the HW path; we have none in M2, so it is done
// in-thread per the staged plan), backface cull by doubled-area sign,
// screen-space edge equations + affine depth gradient, then the
// incremental half-space inner loop over the bounding rect.
// Pixel write = pack(reversed depth 32b + payload 32b) and
// InterlockedMax on the u64 VisBuffer (Karis p.84: depth in the high
// bits). Payload 32b = pairIndex 23b | listSel 2b | triangle 7b (the
// wide-instance layout: cluster/instance live in the frame's PAIR
// TABLES, resolved by one indirection at decode - the old 20b cluster
// | 7b tri | 5b instance packing capped the scene at 32 instances).
// listSel: 0 = the SW visiblePairs (params[9].z selects - the post
// raster runs the same shader over the REVIVED pair list with 2), the
// HW path writes 1. The payload is PAIR-ENUMERATION-SENSITIVE:
// different cull paths order pairs differently - differentials must
// compare RESOLVED attributes, not raw words.
//
// Bindings (SSBO set 0; buffers[i] -> binding i):
//   0: float4 params[12] - [0..3] VIEW matrix columns (glm raw),
//                          [4..7] PROJ columns (glm raw; applied to the
//                          VIEW-space vertices of phase 1; may be the
//                          camera's own Y-FLIPPED matrix - rows then
//                          arrive in framebuffer order),
//                          [8] = (width, height, pairCount, nearZ),
//                          [9].x = yFlip flag (cull-sign selector),
//                          [11].x = rasterMode (1 = the optimized
//                          scheduling: exact-int64 row-span clamp +
//                          per-thread pixel-count aggregation; the
//                          output is bit-identical to mode 0),
//                          [11].y = coverage-probe flag (binding 12)
//   1: float3 positions[] - mesh-local vertex pool (world = model * local)
//   2: uint   indices[]   - shared index pool
//   3: GpuCluster clusters[] - 48B, mirrors packGpuCluster
//   4: GpuInstance instances[] - 96B (model transform)
//   5: uint2  visiblePairs[] - (instanceIdx, clusterIdx), one per group
//   6: uint64_t visBuffer[width*height] - ZERO-INITIALIZED; max wins
//   7: uint   counters[6] - [0] pair count (in; GPU-written by the
//                            in-frame chain or pre-filled one-shot);
//                            [1] pixel writes, [2] backface culls,
//                            [3] near-clipped triangles (out);
//                            [4] pending-resident skips (M6.3, out)
//   8: uint   clusterPage[]  - M6.3 paged: cluster -> page (v2: dummy)
//   9: uint   pageTable[]    - M6.3 paged: page -> slot (0xFFFFFFFF
//                              = not resident; v2: dummy)
//  10: GpuPageHeader pageHeader[] - M6.3 paged: per-page section
//                              element bases (v2: dummy)
//  12: uint  probeStats[8] - coverage histogram (params[11].y gated;
//                              written by BOTH the main and post
//                              dispatches of a frame): [0..5] triangles
//                              whose covered-pixel count is 0 / 1 /
//                              2-4 / 5-16 / 17-64 / >64, [6] = sum of
//                              covered pixels, [7] = triangles that
//                              reached rasterization.
//
// M6.3 paged addressing: params[9].y != 0 switches bindings 1/2 to the
// PAGE SLOT POOL - cluster offsets are then PAGE-LOCAL and each read
// adds the page's slot-strided section base (kSlotFloat4s/kSlotU32s =
// kPageSlotSize/16 / kPageSlotSize/4, engine-wide constants mirrored
// CPU-side). With every page resident the VALUES read are bit-identical
// to the v2 pools (the M6.1 packer proves slice equality) - the
// VisBuffer output is byte-identical between modes. A non-resident page
// skips the whole workgroup and counts into counters[4]; the always-
// resident root pages guarantee a coarse image instead.

struct GpuCluster
{
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

struct GpuInstance
{
    float4x4 model;
    uint meshId;
    uint _pad0; uint _pad1; uint _pad2;
    float4 bounds;
};

struct GpuPageHeader  // M7.1 v5: 12 u32, uniform u32 units
{
    uint posBase;   // posBits section (u32 words)
    uint nrmBase;   // u32 units; byte start = nrmBase*4
    uint uvBase;    // u32 units
    uint idxBase;   // u32 units
    uint metaBase;  // clusterMeta array (u32 units)
    uint vertexCount;
    uint indexCount;   // idx section BYTE total
    uint clusterCount; // meta array bound
    float posStep;     // the mesh grid's 2^-P
    uint flags; uint pad0; uint pad1;
};

struct GpuClusterQuant  // 10 u32, mirrors ClusterQuantMeta
{
    int posStartX; int posStartY; int posStartZ;
    uint widths;      // bx | by<<4 | bz<<8
    uint posWordOff;  // u32 words into posBits
    float uvMinX; float uvMinY; float uvRangeX; float uvRangeY;
    uint idxBits;     // M7.2: baseBits | offsetBits<<4 | bitBase<<8
};

// v5 decode helpers (M7.1; CPU mirror = ClusterQuant.cpp, op-for-op -
// the D5 bit-identity contract behind the differential tests).
uint loadBits2(uint w0, uint w1, uint bitIdx, uint bits)
{
    // VALUE path only - never feeds address arithmetic (the driver's
    // documented DEVICE_LOST discipline).
      const uint shift = bitIdx & 31u;
  const uint mask = (1u << bits) - 1u;  // bits <= 24
  if (shift + bits <= 32u) return (w0 >> shift) & mask;
  return ((w0 >> shift) | (w1 << (32u - shift))) & mask;
}

// kPageSlotSize (128 KB) in u32 elements - the CPU-side
// createPageSlotPool asserts the pool was created with the same slot
// size before the paged mode is ever armed. (v5: all page sections are
// u32-addressed; the old float4/float2 element constants are gone.)
static const uint kSlotU32s = 32768;

// NOTE on matrices: glm stores column-major and Slang's mul(model, v)
// reads buffer vec4s as ROWS (probe-verified in S4.3) - every transform
// below is an explicit linear combination of the stored columns.

[[vk::binding(0)]] StructuredBuffer<float4> params;
[[vk::binding(1)]] StructuredBuffer<uint> positions;  // v2: float4-as-4-u32; paged v5: the slot pool bit stream
[[vk::binding(2)]] StructuredBuffer<uint>   indices;
[[vk::binding(3)]] StructuredBuffer<GpuCluster> clusters;
[[vk::binding(4)]] StructuredBuffer<GpuInstance> instances;
[[vk::binding(5)]] StructuredBuffer<uint2>  visiblePairs;
[[vk::binding(6)]] RWStructuredBuffer<uint64_t> visBuffer;
[[vk::binding(7)]] RWStructuredBuffer<uint>    counters;
[[vk::binding(8)]] StructuredBuffer<uint>          clusterPage;
[[vk::binding(9)]] StructuredBuffer<uint>          pageTable;
[[vk::binding(10)]] StructuredBuffer<GpuPageHeader> pageHeader;
[[vk::binding(11)]] StructuredBuffer<uint> clusterMeta;  // v5: per-cluster ordinal into its page's meta array
[[vk::binding(12)]] RWStructuredBuffer<uint> probeStats;  // params[11].y-gated coverage histogram

groupshared float3 sViewPos[256];

uint64_t packPixel(float z, uint pairIdx, uint triIdx)
{
    // Reversed depth in the high 32 bits: near (z=0) -> large, far
    // (z=1) -> small, so InterlockedMax keeps the NEAREST surface.
    // CPU reference performs the identical float ops in the same order.
    // Payload: pairIndex 23b | listSel (params[9].z) | triangle 7b.
    z = clamp(z, 0.0, 1.0);
    const uint depthBits = uint((1.0 - z) * 4294967295.0);
    const uint sel = uint(params[9].z) & 3u;
    return (uint64_t(depthBits) << 32) |
           (uint64_t(pairIdx & 0x7FFFFFu) << 9) |
           (uint64_t(sel) << 7) |
           (uint64_t(triIdx & 0x7Fu));
}

// M4.1: coverage runs in 16.8 SUBPIXEL FIXED POINT (256 units per
// pixel), mirroring DX/Vulkan hardware rasterization: vertex screen
// coords are floored to 1/256 and the edge functions are EXACT int64
// integers, so tie pixels (w == 0) are decided by the DX top-left rule
// - a tie counts only on a TOP edge (ey == 0 && ex > 0) or a LEFT edge
// (ey < 0), in the canonical positive-area winding (the rule is its
// own mirror image, so it holds for both yFlip conventions). This is
// what lets SW and HW clusters share a seam with zero cracks AND zero
// double coverage. Depth stays a float affine interpolation of NDC z
// at the QUANTIZED positions (UE's MicropolyRasterize does the same -
// its z is also screen-affine, not perspective-correct).
static const int SUBPIXEL = 256;

int64_t edgeW(int64_t ex, int64_t ey, int64_t px, int64_t py, int64_t vx, int64_t vy)
{
    return ex * (py - vy) - ey * (px - vx);
}

// Floor division for the exact row-span clamp (rasterMode 1),
// restricted to a POSITIVE divisor: both branches divide only
// NON-NEGATIVE operands, so the result is independent of whatever
// rounding/sign convention the compiler gives signed `/` (the first
// cut trusted C-style truncation + a remainder fixup and produced
// off-by-one bounds - 539/3072 differential mismatches).
// VALIDITY DOMAIN: the exactness argument (span never excludes a
// pixel the stepping loop would accept) holds while no intermediate
// edge value overflows int64 - guaranteed for on-screen geometry
// (|w| < 2^47), but a near-plane grazing mega-triangle projecting to
// ~+/-8M pixels could wrap mode 0's stepping mod 2^64 and "resurrect"
// pixels this clamp skips. Pre-existing UB-adjacent behavior in mode
// 0's own edgeW at that extreme; documented rather than guarded (the
// guard would cost every row).
int64_t floorDivPos(int64_t a, int64_t b)  // b > 0
{
    if (a >= 0) return a / b;
    return -((-a + b - 1) / b);
}

uint rasterizeTriangle(float2 s0f, float2 s1f, float2 s2f,
                       float z0, float z1, float z2,
                       uint2 widthHeight, int4 scissor, uint pairIdx,
                       uint triIdx)
{
    // Quantize to subpixel fixed point (floor - the DX spec's x.8 snap).
    const int64_t X0 = int64_t(int(floor(s0f.x * float(SUBPIXEL))));
    const int64_t Y0 = int64_t(int(floor(s0f.y * float(SUBPIXEL))));
    int64_t X1 = int64_t(int(floor(s1f.x * float(SUBPIXEL))));
    int64_t Y1 = int64_t(int(floor(s1f.y * float(SUBPIXEL))));
    int64_t X2 = int64_t(int(floor(s2f.x * float(SUBPIXEL))));
    int64_t Y2 = int64_t(int(floor(s2f.y * float(SUBPIXEL))));

    // Doubled signed area on the QUANTIZED corners; cull by sign. The
    // culled half depends on the projection's Y orientation: y-up rows
    // (unflipped proj, the S4.4 differential convention) cull area<0;
    // y-down rows (the camera's flipped proj, framebuffer order) cull
    // area>0 - the same CW-as-viewed half the HW path's CCW-front
    // culling drops.
    const int64_t area2 = (X1 - X0) * (Y2 - Y0) - (Y1 - Y0) * (X2 - X0);
    if (area2 == 0) return 0u;
    const int64_t cullSign = params[9].x > 0.5 ? 1 : -1;
    if (area2 * cullSign > 0)
    {
        InterlockedAdd(counters[2], 1);
        return 0u;
    }
    // Canonicalize to positive area (swap v1/v2; the payload names the
    // triangle by index, not by corner order).
    if (area2 < 0)
    {
        int64_t tx = X1; int64_t ty = Y1; X1 = X2; Y1 = Y2; X2 = tx; Y2 = ty;
        float tz = z1; z1 = z2; z2 = tz;
    }

    // Integer edge vectors of the canonical winding.
    const int64_t e0x = X1 - X0, e0y = Y1 - Y0;
    const int64_t e1x = X2 - X1, e1y = Y2 - Y1;
    const int64_t e2x = X0 - X2, e2y = Y0 - Y2;
    // DX top-left rule: ties (w == 0) count as inside ONLY on top
    // (horizontal, interior below) or left edges.
    const int64_t t0 = (e0y == 0 && e0x > 0) || e0y < 0 ? 0 : 1;
    const int64_t t1 = (e1y == 0 && e1x > 0) || e1y < 0 ? 0 : 1;
    const int64_t t2 = (e2y == 0 && e2x > 0) || e2y < 0 ? 0 : 1;

    // Bounding rect: pixel x's center is at subpixel 256*x + 128, so
    // the first candidate is ceil((minSub-128)/256) and the last is
    // floor((maxSub-128)/256) (inclusive). The float round-trips are
    // exact (values < 2^21, division by 256 is an exponent shift).
    const int64_t minSubX = min(min(X0, X1), X2);
    const int64_t maxSubX = max(max(X0, X1), X2);
    const int64_t minSubY = min(min(Y0, Y1), Y2);
    const int64_t maxSubY = max(max(Y0, Y1), Y2);
    // UE 2026-09-25 cross-check: MinPixel = ceil((min-128)/256) matches
    // UE exactly; MaxPixel = floor((max-129)/256) is UE's tighter bound
    // (floor((Max-129)/256)) - it drops the pixel whose center subpixel
    // equals MaxSubpixel exactly (a bbox-corner tie the fill rule
    // rejects on bottom/right edges; UE excludes it on all edges).
    int x0 = int(ceil((float(minSubX) - 128.0) / 256.0));
    int x1 = int(floor((float(maxSubX) - 129.0) / 256.0));
    int y0 = int(ceil((float(minSubY) - 128.0) / 256.0));
    int y1 = int(floor((float(maxSubY) - 129.0) / 256.0));
    x0 = max(x0, 0);
    y0 = max(y0, 0);
    x1 = min(x1, int(widthHeight.x) - 1);
    y1 = min(y1, int(widthHeight.y) - 1);
    // Stage C scissor ([11].zw, inclusive pixel bounds): the dirty-page
    // region; the host's full-rect default keeps writes bit-identical.
    x0 = max(x0, scissor.x);
    y0 = max(y0, scissor.y);
    x1 = min(x1, scissor.z);
    y1 = min(y1, scissor.w);
    if (x0 > x1 || y0 > y1) return 0u;

    // Screen-space affine depth through the QUANTIZED corners (floats
    // in pixel units; exactly representable, same solve on the CPU).
    const float2 q0 = float2(float(X0), float(Y0)) * (1.0 / 256.0);
    const float2 q1 = float2(float(X1), float(Y1)) * (1.0 / 256.0);
    const float2 q2 = float2(float(X2), float(Y2)) * (1.0 / 256.0);
    const float det = (q1.x - q0.x) * (q2.y - q0.y) -
                      (q1.y - q0.y) * (q2.x - q0.x);
    const float ra =
        ((z1 - z0) * (q2.y - q0.y) - (z2 - z0) * (q1.y - q0.y)) / det;
    const float rb = ((q1.x - q0.x) * (z2 - z0) -
                      (q2.x - q0.x) * (z1 - z0)) / det;

    // Edge values at the rect origin's pixel center; the loops only
    // ever ADD integer edge deltas to these (exact in int64 - no
    // drift, and the CPU reference mirrors the same increments).
    const int64_t px0 = int64_t(x0) * SUBPIXEL + 128;
    const int64_t py0 = int64_t(y0) * SUBPIXEL + 128;
    int64_t w0r0 = edgeW(e0x, e0y, px0, py0, X0, Y0);
    int64_t w1r0 = edgeW(e1x, e1y, px0, py0, X1, Y1);
    int64_t w2r0 = edgeW(e2x, e2y, px0, py0, X2, Y2);
    const float2 p00 = float2(float(px0), float(py0)) * (1.0 / 256.0);
    const float z00 = z0 + ra * (p00.x - q0.x) + rb * (p00.y - q0.y);

    // rasterMode 1 (params[11].x): the exact-int64 row-span clamp -
    // Karis' scanline variant (SIGGRAPH 2021 §11.7) in integer form.
    // Along a row the edge value is W(x) = w - ey*SUBPIXEL*(x - x0)
    // (the loop's own step), so each edge's inside test W(x) >= t
    // constrains x to a half-interval: ey > 0 gives an UPPER bound
    // x0 + floor((w-t)/(ey*256)), ey < 0 a LOWER bound
    // x0 + ceil((w-t)/(ey*256)), ey == 0 either no constraint or an
    // empty row. Intersecting the three intervals with [x0, x1] skips
    // only pixels that PROVABLY fail an edge test - the surviving
    // pixels still run the exact int64 tests, so the written words are
    // bit-identical to mode 0. Gated to rows wider than 4 pixels (the
    // per-row three int64 divisions only pay off on wide rows; Karis'
    // own threshold). The row-span skip also drops the dead z stepping
    // on the right side, but the LEFT side must still walk z += ra
    // from x0: z's depth bits depend on the exact SEQUENCE of float
    // additions, and a closed-form jump rounds differently.
    const bool optMode = params[11].x > 0.5;

    uint pix = 0;
    for (int y = y0; y <= y1; ++y)
    {
        const int64_t dySub = int64_t(y) * SUBPIXEL + 128 - py0;
        int64_t w0 = w0r0 + e0x * dySub;
        int64_t w1 = w1r0 + e1x * dySub;
        int64_t w2 = w2r0 + e2x * dySub;
        const float ry = float(dySub) * (1.0 / 256.0);
        float z = z00 + rb * ry;
        int xs = x0, xe = x1;
        if (optMode && (x1 - x0) > 4)
        {
            // Per edge (divisor kept positive; ceil(D/-B) == -floor(D/B)):
            //   ey > 0: upper bound x0 + floor((w-t) / (ey*256))
            //   ey < 0: lower bound x0 - floor((w-t) / (-ey*256))
            //   ey == 0: no constraint, or the whole row fails.
            // Explicit ternaries instead of min/max on int64 (belt and
            // braces - no reliance on int64 intrinsic overloads).
            int64_t lo = int64_t(x0), hi = int64_t(x1);
            bool empty = false;
            if (e0y == 0) { if (w0 < t0) empty = true; }
            else if (e0y > 0) {
                const int64_t c = int64_t(x0) + floorDivPos(w0 - t0, e0y * SUBPIXEL);
                hi = c < hi ? c : hi;
            } else {
                const int64_t c = int64_t(x0) - floorDivPos(w0 - t0, -e0y * SUBPIXEL);
                lo = c > lo ? c : lo;
            }
            if (e1y == 0) { if (w1 < t1) empty = true; }
            else if (e1y > 0) {
                const int64_t c = int64_t(x0) + floorDivPos(w1 - t1, e1y * SUBPIXEL);
                hi = c < hi ? c : hi;
            } else {
                const int64_t c = int64_t(x0) - floorDivPos(w1 - t1, -e1y * SUBPIXEL);
                lo = c > lo ? c : lo;
            }
            if (e2y == 0) { if (w2 < t2) empty = true; }
            else if (e2y > 0) {
                const int64_t c = int64_t(x0) + floorDivPos(w2 - t2, e2y * SUBPIXEL);
                hi = c < hi ? c : hi;
            } else {
                const int64_t c = int64_t(x0) - floorDivPos(w2 - t2, -e2y * SUBPIXEL);
                lo = c > lo ? c : lo;
            }
            if (empty || lo > hi) continue;
            xs = int(lo);
            xe = int(hi);
        }
        // The walk keeps mode 0's stepping sequence LITERALLY: one loop
        // from x0, the span only gates which pixels run the test/write.
        // A jump start (advance w by exact int64 math + run z += ra in
        // a dead pre-loop) is NOT bit-safe: Slang reassociates the pure
        // float induction loop into a multiply, and any closed form of
        // z(xs) rounds differently than the sequential adds (measured:
        // 483/3072 differential mismatches). Left-skipped pixels still
        // pay the cheap int64/f32 increments; right-skipped pixels walk
        // nothing; empty rows skip wholesale.
        for (int x = x0; x <= xe; ++x)
        {
            // Inside = strictly positive, or exactly zero on a
            // top/left edge (w >= t with t = 0/1 encodes both).
            if (x >= xs && w0 >= t0 && w1 >= t1 && w2 >= t2)
            {
                const uint64_t packed = packPixel(z, pairIdx, triIdx);
                InterlockedMax(visBuffer[uint(y) * widthHeight.x + uint(x)],
                               packed);
                // Mode 0 keeps the historical per-pixel stat atomic
                // (the differential baseline); mode 1 counts locally
                // and the caller adds the thread's total once.
                if (!optMode) InterlockedAdd(counters[1], 1);
                ++pix;
            }
            // x advances ONE PIXEL = SUBPIXEL subpixels: the edge delta is
            // -ey * SUBPIXEL (the row step multiplies by dySub for the same
            // reason). Missing this factor made the edge functions crawl
            // 256x too slowly across each row.
            w0 -= e0y * SUBPIXEL;
            w1 -= e1y * SUBPIXEL;
            w2 -= e2y * SUBPIXEL;
            z += ra;
        }
    }
    return pix;
}

uint clipNear(float3 v0, float3 v1, float3 v2, float nearZ,
              out float3 pa, out float3 pb, out float3 pc, out float3 pd)
{
    // Unrolled Sutherland-Hodgman over the 3 edges; outputs accumulate
    // into scalars (local arrays trip Slang's spirv emitter).
    uint n = 0;
    float3 a = v0; float3 b = v1;
    for (int e = 0; e < 3; ++e)
    {
        const bool aIn = a.z <= -nearZ;
        const bool bIn = b.z <= -nearZ;
        if (aIn)
        {
            if (n == 0) pa = a; else if (n == 1) pb = a; else if (n == 2) pc = a; else pd = a;
            n = n + 1;
            if (!bIn)
            {
                const float t = (-nearZ - a.z) / (b.z - a.z);
                const float3 x = a + (b - a) * t;
                if (n == 0) pa = x; else if (n == 1) pb = x; else if (n == 2) pc = x; else pd = x;
                n = n + 1;
            }
        }
        else if (bIn)
        {
            const float t = (-nearZ - a.z) / (b.z - a.z);
            const float3 x = a + (b - a) * t;
            if (n == 0) pa = x; else if (n == 1) pb = x; else if (n == 2) pc = x; else pd = x;
            n = n + 1;
        }
        a = b;
        b = (e == 0) ? v2 : v0;
    }
    return n;
}

float2 screenOf(float3 p, float4 c0, float4 c1, float4 c2, float4 c3,
                float2 wh)
{
    const float4 clip = c0 * p.x + c1 * p.y + c2 * p.z + c3;
    const float3 ndc = clip.xyz / clip.w;
    return float2((ndc.x * 0.5 + 0.5) * wh.x, (ndc.y * 0.5 + 0.5) * wh.y);
}

float ndcZOf(float3 p, float4 c0, float4 c1, float4 c2, float4 c3)
{
    const float4 clip = c0 * p.x + c1 * p.y + c2 * p.z + c3;
    return clip.z / clip.w;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint pairIdx = gid.x;
    // Ceil slack of the clamped envelope: guard at the capacity too
    // ([9].w, carried by the chain; the one-shot fills exact counts).
    if (pairIdx >= counters[0] || pairIdx >= uint(params[9].w)) return;
    const uint2 pair = visiblePairs[pairIdx];
    const GpuCluster cluster = clusters[pair.y];
    const GpuInstance inst = instances[pair.x];

    // ---- v5 paged addressing (see the header comment) ----------------
    // Pool u32 capacities ([10].xy) - the v5 clamp discipline bound for
    // the meta/bit/word reads below (both branches; v2 scales x4).
    const uint posTotal = max(uint(params[10].x), 1u);
    uint posBase = 0, idxBase = 0;
    GpuClusterQuant qm;          // valid only in paged mode
    float posStep = 1.0;
    if (uint(params[9].y) != 0u)
    {
        const uint page = clusterPage[pair.y];
        const uint slot = pageTable[page];
        if (slot == 0xFFFFFFFFu)
        {
            // Not resident: skip the whole group (one count - every
            // thread takes this path). The always-resident coarse
            // pages cover the area instead.
            if (gtid.x == 0u) InterlockedAdd(counters[4], 1);
            return;
        }
        const GpuPageHeader ph = pageHeader[page];
        const uint pool = slot * kSlotU32s;  // this slot's u32 window
        posBase = pool + ph.posBase;
        idxBase = pool + ph.idxBase;
        posStep = ph.posStep;
        const uint ord = min(clusterMeta[pair.y], ph.clusterCount - 1u);
        // The 10-word meta record must fit the pool's u32 capacity
        // (decode's metaEnd pattern).
        const uint mb = min(pool + ph.metaBase + ord * 10u, posTotal - 10u);
        qm.posStartX = int(positions[mb]); qm.posStartY = int(positions[mb + 1u]);
        qm.posStartZ = int(positions[mb + 2u]);
        qm.widths = positions[mb + 3u]; qm.posWordOff = positions[mb + 4u];
        qm.uvMinX = asfloat(positions[mb + 5u]); qm.uvMinY = asfloat(positions[mb + 6u]);
        qm.uvRangeX = asfloat(positions[mb + 7u]); qm.uvRangeY = asfloat(positions[mb + 8u]);
        qm.idxBits = positions[mb + 9u];
    }

    const float4 viewC0 = params[0];
    const float4 viewC1 = params[1];
    const float4 viewC2 = params[2];
    const float4 viewC3 = params[3];
    const float4 vpC0 = params[4];
    const float4 vpC1 = params[5];
    const float4 vpC2 = params[6];
    const float4 vpC3 = params[7];
    const float2 widthHeight = params[8].xy;
    const float nearZ = params[8].w;

    // ---- Phase 1: vertices -> VIEW space (groupshared) ---------------
    const uint vertCount = cluster.vertexCount;
    for (uint v = gtid.x; v < vertCount; v += 128)
    {
        float3 lp;
        if (uint(params[9].y) != 0u)
        {
            // v5: three SOA bit fields on the mesh's integer grid
            // (D9.2's anti-crack scheme - see ClusterQuant.cpp).
            const uint b0 = qm.widths & 0xFu;
            const uint b1 = (qm.widths >> 4) & 0xFu;
            const uint b2 = (qm.widths >> 8) & 0xFu;
            const uint64_t base = uint64_t(qm.posWordOff) * 32u;
            const uint64_t bitA = base + uint64_t(v) * b0;
            const uint64_t bitB = base + uint64_t(vertCount) * b0 + uint64_t(v) * b1;
            const uint64_t bitC = base + uint64_t(vertCount) * (b0 + b1) + uint64_t(v) * b2;
            const uint posEnd = posTotal - 1u;
            const uint q0 = loadBits2(positions[min(posBase + uint(bitA / 32u), posEnd)],
                                      positions[min(posBase + uint(bitA / 32u) + 1u, posEnd)],
                                      uint(bitA & 31u), b0);
            const uint q1 = loadBits2(positions[min(posBase + uint(bitB / 32u), posEnd)],
                                      positions[min(posBase + uint(bitB / 32u) + 1u, posEnd)],
                                      uint(bitB & 31u), b1);
            const uint q2 = loadBits2(positions[min(posBase + uint(bitC / 32u), posEnd)],
                                      positions[min(posBase + uint(bitC / 32u) + 1u, posEnd)],
                                      uint(bitC & 31u), b2);
            const int3 iq = int3(int(q0), int(q1), int(q2))
                          + int3(qm.posStartX, qm.posStartY, qm.posStartZ);
            lp = float3(iq) * posStep;
        }
        else
        {
            // v2: float4 elements as 4 u32 (w unused). Word-unit clamp
            // against the pool's u32 capacity ([10].x).
            const uint vw = min((posBase + cluster.vertexOffset + v) * 4u,
                                posTotal - 3u);
            lp = float3(asfloat(positions[vw]),
                       asfloat(positions[vw + 1u]),
                       asfloat(positions[vw + 2u]));
        }
        // world = inst.model * lp  (explicit glm-column combination)
        const float3 wp = inst.model[0].xyz * lp.x + inst.model[1].xyz * lp.y
                        + inst.model[2].xyz * lp.z + inst.model[3].xyz;
        // view = view * wp
        const float3 vp = viewC0.xyz * wp.x + viewC1.xyz * wp.y
                        + viewC2.xyz * wp.z + viewC3.xyz;
        sViewPos[v] = vp;
    }
    GroupMemoryBarrierWithGroupSync();

    // ---- Phase 2: one thread per triangle ------------------------------
    const uint triCount = cluster.indexCount / 3;
    if (gtid.x >= triCount) return;
    uint i0, i1, i2;
    if (uint(params[9].y) != 0u)
    {
        // v6: packed indices - the meta word carries
        // [baseBits | offsetBits<<4 | bitBase<<8]; triangle t lives at
        // bitBase + t*W (W = baseBits + 2*offsetBits) as the cyclic
        // min-first rotation [base | +off1 | +off2] (D5 mirror =
        // decodeQuantCluster). The whole <=22-bit field spans at most
        // the two loaded words; every word read clamps against the
        // pool capacity ([10].y) per the addressing discipline.
        const uint idxTotal = max(uint(params[10].y), 1u);
        const uint baseBits = qm.idxBits & 0xFu;
        const uint offsetBits = (qm.idxBits >> 4) & 0xFu;
        const uint bitIdx = (qm.idxBits >> 8) + gtid.x * (baseBits + 2u * offsetBits);
        // Per-subfield word loads: a subfield may cross into the word
        // after the triangle's first word, so each field addresses ITS
        // own (w0,w1) pair (masking the global bit into the first
        // word's window is the classic off-by-32 bug).
        const uint wi0 = idxBase + (bitIdx >> 5);
        const uint base = loadBits2(indices[min(wi0, idxTotal - 1u)],
                                    indices[min(wi0 + 1u, idxTotal - 1u)],
                                    bitIdx & 31u, baseBits);
        const uint at1 = bitIdx + baseBits;
        const uint wi1 = idxBase + (at1 >> 5);
        const uint o1 = loadBits2(indices[min(wi1, idxTotal - 1u)],
                                  indices[min(wi1 + 1u, idxTotal - 1u)],
                                  at1 & 31u, offsetBits);
        const uint at2 = at1 + offsetBits;
        const uint wi2 = idxBase + (at2 >> 5);
        const uint o2 = loadBits2(indices[min(wi2, idxTotal - 1u)],
                                  indices[min(wi2 + 1u, idxTotal - 1u)],
                                  at2 & 31u, offsetBits);
        i0 = base;
        i1 = base + o1;
        i2 = base + o2;
    }
    else
    {
        i0 = indices[idxBase + cluster.indexOffset + gtid.x * 3 + 0];
        i1 = indices[idxBase + cluster.indexOffset + gtid.x * 3 + 1];
        i2 = indices[idxBase + cluster.indexOffset + gtid.x * 3 + 2];
    }
    const float3 v0 = sViewPos[i0];
    const float3 v1 = sViewPos[i1];
    const float3 v2 = sViewPos[i2];

    // Near-plane clip (Sutherland-Hodgman in VIEW space against
    // z = -nearZ, keeping z <= -nearZ): a crossing triangle can become
    // a quad -> rasterize as two triangles, serially in this thread.
    // Fully on the camera side (nothing survives) skips silently - the
    // "triangle disappears entirely" bug this prevents is a staged-plan
    // dedicated test case. Scalar out-params: local ARRAY constructs
    // trip a Slang spirv-emit internal error on this toolchain.
    float3 pa, pb, pc, pd;
    const uint polyN = clipNear(v0, v1, v2, nearZ, pa, pb, pc, pd);
    if (polyN < 3) return;

    const uint2 widthHeightU = uint2(uint(widthHeight.x), uint(widthHeight.y));
    // Stage C scissor: 11-bit halves packed as (x<<11)|y into [11].zw
    // (22 bits = exactly representable in f32; the uint round-trip of
    // the exact integer float is exact).
    const uint2 scPacked = uint2(uint(params[11].z), uint(params[11].w));
    const int4 scissor = int4(int(scPacked.x >> 11u), int(scPacked.x & 2047u),
                              int(scPacked.y >> 11u), int(scPacked.y & 2047u));

    const float2 wh = widthHeight;
    const uint2 whU = widthHeightU;

    uint localPix = 0;
    if (polyN == 3)
    {
        // Project each view-space point inline (Slang has no capturing
        // lambdas; proj Y UNflipped - row index grows with +Y NDC).
        const float2 sa = screenOf(pa, vpC0, vpC1, vpC2, vpC3, wh);
        const float2 sb = screenOf(pb, vpC0, vpC1, vpC2, vpC3, wh);
        const float2 sc = screenOf(pc, vpC0, vpC1, vpC2, vpC3, wh);
        localPix = rasterizeTriangle(sa, sb, sc,
                          ndcZOf(pa, vpC0, vpC1, vpC2, vpC3),
                          ndcZOf(pb, vpC0, vpC1, vpC2, vpC3),
                          ndcZOf(pc, vpC0, vpC1, vpC2, vpC3),
                          whU, scissor, pairIdx, gtid.x);
    }
    else  // quad (near-plane crossing): fan (0,1,2) + (0,2,3)
    {
        InterlockedAdd(counters[3], 1);
        const float2 sa = screenOf(pa, vpC0, vpC1, vpC2, vpC3, wh);
        const float2 sb = screenOf(pb, vpC0, vpC1, vpC2, vpC3, wh);
        const float2 sc = screenOf(pc, vpC0, vpC1, vpC2, vpC3, wh);
        const float2 sd = screenOf(pd, vpC0, vpC1, vpC2, vpC3, wh);
        const float za = ndcZOf(pa, vpC0, vpC1, vpC2, vpC3);
        const float zb = ndcZOf(pb, vpC0, vpC1, vpC2, vpC3);
        const float zc = ndcZOf(pc, vpC0, vpC1, vpC2, vpC3);
        const float zd = ndcZOf(pd, vpC0, vpC1, vpC2, vpC3);
        localPix = rasterizeTriangle(sa, sb, sc, za, zb, zc, whU, scissor,
                                     pairIdx, gtid.x)
                 + rasterizeTriangle(sa, sc, sd, za, zc, zd, whU, scissor,
                                     pairIdx, gtid.x);
    }

    // Mode 1: one aggregated pixel-write counter add per thread instead
    // of one global atomic per covered pixel (the same total - integer
    // sums are order-free; the per-pixel form is the mode-0 baseline).
    if (params[11].x > 0.5 && localPix != 0u)
        InterlockedAdd(counters[1], localPix);

    // Coverage probe (params[11].y): classify this triangle by its
    // covered-pixel count. Threads that never reached rasterization
    // (behind the near plane) and phase-2 overflow lanes do not write;
    // backfaces land in bucket 0 (counters[2] separates them).
    if (params[11].y > 0.5)
    {
        uint b = 0u;
        if (localPix > 64u) b = 5u;
        else if (localPix > 16u) b = 4u;
        else if (localPix > 4u) b = 3u;
        else if (localPix > 1u) b = 2u;
        else if (localPix > 0u) b = 1u;
        InterlockedAdd(probeStats[b], 1u);
        InterlockedAdd(probeStats[6], localPix);
        InterlockedAdd(probeStats[7], 1u);
    }
}
