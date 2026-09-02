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
//   7: uint   counters[4] - [0] coverage, [1] pixel writes, [2] backface
//                            culls, [3] near-clipped triangles

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

void rasterizeTriangle(float2 s0, float2 s1, float2 s2,
                       float z0, float z1, float z2,
                       uint2 widthHeight, uint clusterIdx, uint triIdx,
                       uint instanceIdx)
{
    // Doubled signed area = cross(s1-s0, s2-s0); cull by sign. The
    // culled half depends on the projection's Y orientation: y-up rows
    // (unflipped proj, the S4.4 differential convention) cull area<0;
    // y-down rows (the camera's flipped proj, framebuffer order) cull
    // area>0 - the same CW-as-viewed half the HW path's CCW-front
    // culling drops. The CPU reference implements the identical rule.
    const float area2 = (s1.x - s0.x) * (s2.y - s0.y)
                      - (s1.y - s0.y) * (s2.x - s0.x);
    if (area2 == 0.0) return;
    const float cullSign = params[9].x > 0.5 ? 1.0 : -1.0;
    if (area2 * cullSign > 0.0)
    {
        InterlockedAdd(counters[2], 1);
        return;
    }
    // Canonicalize to positive area so the edge equations below keep
    // their w >= 0 inside test on both orientations: swap v1/v2 (the
    // payload names the triangle by index, not by corner order).
    if (area2 < 0.0)
    {
        float2 ts = s1; s1 = s2; s2 = ts;
        float tz = z1; z1 = z2; z2 = tz;
    }

    // Edge functions W_i(p) = cross(edge, p - v0): positive inside for a
    // consistently wound triangle; per-step deltas are the raw edge vecs.
    const float2 e0 = s1 - s0;
    const float2 e1 = s2 - s1;
    const float2 e2 = s0 - s2;

    // Screen-space affine depth: z = a*x + b*y + c through the three
    // vertices (NDC z is affine in screen space for a planar triangle).
    // Solve with the same formula on CPU; test scenes use constant-z or
    // nice-slope depths so rounding cannot diverge.
    const float det = e0.x * (s2.y - s0.y) - e0.y * (s2.x - s0.x);  // = area2
    const float ra = ((z1 - z0) * (s2.y - s0.y) - (z2 - z0) * (s1.y - s0.y)) / det;
    const float rb = (e0.x * (z2 - z0) - e0.x * 0.0 -
                      ((s2.x - s0.x) * (z1 - z0))) / det;

    // Bounding rect: pixel centers p = integer + 0.5 inside
    // [min(s), max(s)] -> first = ceil(min - 0.5), end = ceil(max - 0.5).
    const float minx = min(min(s0.x, s1.x), s2.x);
    const float maxx = max(max(s0.x, s1.x), s2.x);
    const float miny = min(min(s0.y, s1.y), s2.y);
    const float maxy = max(max(s0.y, s1.y), s2.y);
    int x0 = int(ceil(minx - 0.5));
    int x1 = int(ceil(maxx - 0.5));
    int y0 = int(ceil(miny - 0.5));
    int y1 = int(ceil(maxy - 0.5));
    x0 = max(x0, 0);
    y0 = max(y0, 0);
    x1 = min(x1, int(widthHeight.x));
    y1 = min(y1, int(widthHeight.y));
    if (x0 >= x1 || y0 >= y1) return;

    // Edge function values at the rect origin (first pixel center).
    const float2 p00 = float2(float(x0) + 0.5, float(y0) + 0.5);
    const float w0r0 = e0.x * (p00.y - s0.y) - e0.y * (p00.x - s0.x);
    const float w1r0 = e1.x * (p00.y - s1.y) - e1.y * (p00.x - s1.x);
    const float w2r0 = e2.x * (p00.y - s2.y) - e2.y * (p00.x - s2.x);
    const float z00 = z0 + ra * (p00.x - s0.x) + rb * (p00.y - s0.y);
    const float dzdx = ra;
    const float dzdy = rb;

    for (int y = y0; y < y1; ++y)
    {
        // Row-start values recomputed from the ORIGIN (p00) values with
        // a single multiply-add per edge/depth - no accumulation drift,
        // and the CPU reference mirrors the exact same expressions.
        // NOTE: + 0.5 - the ROW's pixel centers sit at y + 0.5, same as
        // p00 = (x0 + 0.5, y0 + 0.5); without it every row samples the
        // pixel's TOP EDGE and near-edge pixels flip in/out (the S4.5
        // crack bug - the differential missed it because the CPU mirror
        // shared the same half-pixel offset).
        const float ry = float(y) + 0.5 - p00.y;
        float w0 = w0r0 + e0.x * ry;
        float w1 = w1r0 + e1.x * ry;
        float w2 = w2r0 + e2.x * ry;
        float z = z00 + dzdy * ry;
        for (int x = x0; x < x1; ++x)
        {
            // Pixel-center inside test; EXACT ties (w == 0) count as
            // inside on every edge - the CPU reference implements the
            // same rule, so shared-edge pixels resolve identically.
            if (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0)
            {
                const uint64_t packed =
                    packPixel(z, clusterIdx, triIdx, instanceIdx);
                InterlockedMax(visBuffer[uint(y) * widthHeight.x + uint(x)],
                               packed);
                InterlockedAdd(counters[1], 1);
            }
            w0 += -e0.y;
            w1 += -e1.y;
            w2 += -e2.y;
            z += dzdx;
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
    if (pairIdx >= uint(params[8].z)) return;
    const uint2 pair = visiblePairs[pairIdx];
    const GpuCluster cluster = clusters[pair.y];
    const GpuInstance inst = instances[pair.x];

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
        const float3 lp = positions[cluster.vertexOffset + v].xyz;
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
    const uint i0 = indices[cluster.indexOffset + gtid.x * 3 + 0];
    const uint i1 = indices[cluster.indexOffset + gtid.x * 3 + 1];
    const uint i2 = indices[cluster.indexOffset + gtid.x * 3 + 2];
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
