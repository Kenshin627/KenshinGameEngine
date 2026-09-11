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
// bits). Payload 32b = cluster 20b | triangle 7b | instance 5b
// (S4.6: instance bits for the multi-instance demo scene).
//
// Bindings (SSBO set 0; buffers[i] -> binding i):
//   0: float4 params[12] - [0..3] VIEW matrix columns (glm raw),
//                          [4..7] PROJ columns (glm raw; applied to the
//                          VIEW-space vertices of phase 1; may be the
//                          camera's own Y-FLIPPED matrix - rows then
//                          arrive in framebuffer order),
//                          [8] = (width, height, pairCount, nearZ),
//                          [9].x = yFlip flag (cull-sign selector)
//   1: float3 positions[] - mesh-local vertex pool (world = model * local)
//   2: uint   indices[]   - shared index pool
//   3: GpuCluster clusters[] - 48B, mirrors packGpuCluster
//   4: GpuInstance instances[] - 96B (model transform)
//   5: uint2  visiblePairs[] - (instanceIdx, clusterIdx), one per group
//   6: uint64 visBuffer[width*height] - ZERO-INITIALIZED; max wins
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

struct GpuPageHeader
{
    uint posBase;  // float4 elements within the page
    uint nrmBase;  // float4 elements
    uint uvBase;   // float2 elements
    uint idxBase;  // u32 elements
    uint vertexCount;
    uint indexCount;
};

// kPageSlotSize (128 KB) in float4 / u32 elements - the CPU-side
// createPageSlotPool asserts the pool was created with the same slot
// size before the paged mode is ever armed.
static const uint kSlotFloat4s = 8192;
static const uint kSlotU32s    = 32768;

// NOTE on matrices: glm stores column-major and Slang's mul(model, v)
// reads buffer vec4s as ROWS (probe-verified in S4.3) - every transform
// below is an explicit linear combination of the stored columns.

[[vk::binding(0)]] StructuredBuffer<float4> params;
[[vk::binding(1)]] StructuredBuffer<float4> positions;
[[vk::binding(2)]] StructuredBuffer<uint>   indices;
[[vk::binding(3)]] StructuredBuffer<GpuCluster> clusters;
[[vk::binding(4)]] StructuredBuffer<GpuInstance> instances;
[[vk::binding(5)]] StructuredBuffer<uint2>  visiblePairs;
[[vk::binding(6)]] RWStructuredBuffer<uint64_t> visBuffer;
[[vk::binding(7)]] RWStructuredBuffer<uint>    counters;
[[vk::binding(8)]] StructuredBuffer<uint>          clusterPage;
[[vk::binding(9)]] StructuredBuffer<uint>          pageTable;
[[vk::binding(10)]] StructuredBuffer<GpuPageHeader> pageHeader;

groupshared float3 sViewPos[256];

uint64_t packPixel(float z, uint clusterIdx, uint triIdx, uint instanceIdx)
{
    // Reversed depth in the high 32 bits: near (z=0) -> large, far
    // (z=1) -> small, so InterlockedMax keeps the NEAREST surface.
    // CPU reference performs the identical float ops in the same order.
    // Payload: cluster 20b | triangle 7b | instance 5b.
    z = clamp(z, 0.0, 1.0);
    const uint depthBits = uint((1.0 - z) * 4294967295.0);
    return (uint64_t(depthBits) << 32) |
           (uint64_t(clusterIdx & 0xFFFFFu) << 12) |
           (uint64_t(triIdx & 0x7Fu) << 5) | uint64_t(instanceIdx & 0x1Fu);
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

void rasterizeTriangle(float2 s0f, float2 s1f, float2 s2f,
                       float z0, float z1, float z2,
                       uint2 widthHeight, uint clusterIdx, uint triIdx,
                       uint instanceIdx)
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
    if (area2 == 0) return;
    const int64_t cullSign = params[9].x > 0.5 ? 1 : -1;
    if (area2 * cullSign > 0)
    {
        InterlockedAdd(counters[2], 1);
        return;
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
    int x0 = int(ceil((float(minSubX) - 128.0) / 256.0));
    int x1 = int(floor((float(maxSubX) - 128.0) / 256.0));
    int y0 = int(ceil((float(minSubY) - 128.0) / 256.0));
    int y1 = int(floor((float(maxSubY) - 128.0) / 256.0));
    x0 = max(x0, 0);
    y0 = max(y0, 0);
    x1 = min(x1, int(widthHeight.x) - 1);
    y1 = min(y1, int(widthHeight.y) - 1);
    if (x0 > x1 || y0 > y1) return;

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

    for (int y = y0; y <= y1; ++y)
    {
        const int64_t dySub = int64_t(y) * SUBPIXEL + 128 - py0;
        int64_t w0 = w0r0 + e0x * dySub;
        int64_t w1 = w1r0 + e1x * dySub;
        int64_t w2 = w2r0 + e2x * dySub;
        const float ry = float(dySub) * (1.0 / 256.0);
        float z = z00 + rb * ry;
        for (int x = x0; x <= x1; ++x)
        {
            // Inside = strictly positive, or exactly zero on a
            // top/left edge (w >= t with t = 0/1 encodes both).
            if (w0 >= t0 && w1 >= t1 && w2 >= t2)
            {
                const uint64_t packed =
                    packPixel(z, clusterIdx, triIdx, instanceIdx);
                InterlockedMax(visBuffer[uint(y) * widthHeight.x + uint(x)],
                               packed);
                InterlockedAdd(counters[1], 1);
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
    if (pairIdx >= counters[0]) return;
    const uint2 pair = visiblePairs[pairIdx];
    const GpuCluster cluster = clusters[pair.y];
    const GpuInstance inst = instances[pair.x];

    // ---- M6.3 paged addressing (see the header comment) -------------
    uint posBase = 0, idxBase = 0;
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
        posBase = ph.posBase + slot * kSlotFloat4s;
        idxBase = ph.idxBase + slot * kSlotU32s;
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
        // float4 elements: a vec3 ARRAY in std430 strides 16, so the
        // host packs xyz+pad (w unused).
        const float3 lp = positions[posBase + cluster.vertexOffset + v].xyz;
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
    const uint i0 = indices[idxBase + cluster.indexOffset + gtid.x * 3 + 0];
    const uint i1 = indices[idxBase + cluster.indexOffset + gtid.x * 3 + 1];
    const uint i2 = indices[idxBase + cluster.indexOffset + gtid.x * 3 + 2];
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

    const float2 wh = widthHeight;
    const uint2 whU = widthHeightU;

    if (polyN == 3)
    {
        // Project each view-space point inline (Slang has no capturing
        // lambdas; proj Y UNflipped - row index grows with +Y NDC).
        const float2 sa = screenOf(pa, vpC0, vpC1, vpC2, vpC3, wh);
        const float2 sb = screenOf(pb, vpC0, vpC1, vpC2, vpC3, wh);
        const float2 sc = screenOf(pc, vpC0, vpC1, vpC2, vpC3, wh);
        rasterizeTriangle(sa, sb, sc, ndcZOf(pa, vpC0, vpC1, vpC2, vpC3),
                          ndcZOf(pb, vpC0, vpC1, vpC2, vpC3),
                          ndcZOf(pc, vpC0, vpC1, vpC2, vpC3),
                          whU, pair.y, gtid.x, pair.x);
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
        rasterizeTriangle(sa, sb, sc, za, zb, zc, whU, pair.y, gtid.x,
                          pair.x);
        rasterizeTriangle(sa, sc, sd, za, zc, zd, whU, pair.y, gtid.x,
                          pair.x);
    }
}
