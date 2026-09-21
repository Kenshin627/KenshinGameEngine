// S4.5: material decode pass - the VisBuffer consumer.
// Fullscreen triangle (VS) + per-pixel decode (FS) per Karis p.99:
// load VisBuffer -> cluster -> instance transform -> 3 vert indices ->
// 3 positions -> transform to screen -> analytic barycentrics (2x2) ->
// interpolate attributes. Analytic UV derivatives from the triangle's
// screen-space UV/position deltas (p.106-108: finite differences are
// nonsense across VisBuffer discontinuities). Blinn-Phong matches the
// baseline path formula so the dual-path screenshot comparison is
// apples-to-apples.
//
// Bindings (SSBO set 0; buffers[i] -> binding i):
//   0: uint64 vis[]           - S4.4's VisBuffer (0 = empty)
//   1: GpuCluster clusters[]  - 48B
//   2: GpuInstance instances[]- 96B
//   3: float4 positions[]     - xyz (std430 vec3 arrays stride 16)
//   4: float4 normals[]       - xyz
//   5: float2 uvs[]           - std430 float2 stride 8
//   6: uint indices[]
//
// Push constants (one 128-byte block, C++ packs identically):
//   viewProj (64B) | camPos+flags (16B) | lightDir (16B) |
//   kd+checkerScale (16B) | viewport w,h,1/w,1/h (16B)

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

struct GpuGroup
{
    float4 bounds;       // shared decision sphere
    float maxError;
    float simplifyError;
    uint lodLevel;       // S4.6 debug views
    float _pad4;
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

// v5 decode helpers (CPU mirror = ClusterQuant.cpp, op-for-op - the D5
// bit-identity contract). VALUE path only - the u64 combine never
// feeds address arithmetic (the documented DEVICE_LOST discipline).
uint loadBits2(uint w0, uint w1, uint bitIdx, uint bits)
{
    const uint shift = bitIdx & 31u;
    const uint mask = (1u << bits) - 1u;  // bits <= 24
    if (shift + bits <= 32u) return (w0 >> shift) & mask;
    return ((w0 >> shift) | (w1 << (32u - shift))) & mask;
}
float3 octDecodeN(uint q0, uint q1)
{
    // snorm-8 pair -> unit vector (meshopt decodeFilterOct's int stage
    // with fo = 127, then normalize).
    int x8 = int(q0) - (q0 >= 128u ? 256 : 0);
    int y8 = int(q1) - (q1 >= 128u ? 256 : 0);
    float x = float(x8);
    float y = float(y8);
    float z = 127.0 - abs(x) - abs(y);
    float t = z >= 0.0 ? 0.0 : z;
    x += x >= 0.0 ? t : -t;
    y += y >= 0.0 ? t : -t;
    return normalize(float3(x, y, z));
}

// v5 per-vertex decoders (file scope, explicit state - no captures).
// VALUE path only; the callers clamp the word addresses.
float3 v5Pos(uint vi, GpuClusterQuant qm, uint posBase, uint posTotal,
             uint vertexCount, float posStep)
{
    const uint b0 = qm.widths & 0xFu;
    const uint b1 = (qm.widths >> 4) & 0xFu;
    const uint b2 = (qm.widths >> 8) & 0xFu;
    const uint64_t base = uint64_t(qm.posWordOff) * 32u;
    const uint64_t bitA = base + uint64_t(vi) * b0;
    const uint64_t bitB = base + uint64_t(vertexCount) * b0 + uint64_t(vi) * b1;
    const uint64_t bitC = base + uint64_t(vertexCount) * (b0 + b1) + uint64_t(vi) * b2;
    const uint qa = loadBits2(positions[min(posBase + uint(bitA / 32u), posTotal - 1u)],
                              positions[min(posBase + uint(bitA / 32u) + 1u, posTotal - 1u)],
                              uint(bitA & 31u), b0);
    const uint qb = loadBits2(positions[min(posBase + uint(bitB / 32u), posTotal - 1u)],
                              positions[min(posBase + uint(bitB / 32u) + 1u, posTotal - 1u)],
                              uint(bitB & 31u), b1);
    const uint qc = loadBits2(positions[min(posBase + uint(bitC / 32u), posTotal - 1u)],
                              positions[min(posBase + uint(bitC / 32u) + 1u, posTotal - 1u)],
                              uint(bitC & 31u), b2);
    const int3 iq = int3(int(qa), int(qb), int(qc)) +
                    int3(qm.posStartX, qm.posStartY, qm.posStartZ);
    return float3(iq) * posStep;
}

float3 v5Nrm(uint vi, uint nrmBase, uint poolTotal, uint vertexOffset)
{
    // 2-byte oct pair at the cluster's vertex ordinal.
    const uint byteOff = vertexOffset * 2u + vi * 2u;
    const uint word = normals[min(nrmBase + byteOff / 4u, poolTotal - 1u)];
    const uint lane = byteOff & 3u;
    const uint q0 = (word >> (8u * lane)) & 0xFFu;
    const uint q1 = (word >> (8u * ((lane + 1u) & 3u))) & 0xFFu;
    // lane+1 crossing the word boundary only happens at lane == 3
    // (byte pairs are 2-aligned, so lane is 0 or 2 - never 3).
    return octDecodeN(q0, q1);
}

float2 v5Uv(uint vi, uint uvBase, uint poolTotal, uint vertexOffset,
            GpuClusterQuant qm)
{
    const uint byteOff = vertexOffset * 2u + vi * 2u;
    const uint word = uvs[min(uvBase + byteOff / 4u, poolTotal - 1u)];
    const uint lane = byteOff & 3u;  // 0 or 2 (pairs are 2-aligned)
    const uint qx = (word >> (8u * lane)) & 0xFFu;
    const uint qy = (word >> (8u * (lane + 1u))) & 0xFFu;
    return float2(qm.uvMinX + float(qx) * (qm.uvRangeX / 255.0),
                  qm.uvMinY + float(qy) * (qm.uvRangeY / 255.0));
}

struct DecodePush
{
    float4x4 viewProj;   // world -> NDC, the camera's own (Y-flipped) matrix
    float4 camFlags;     // xyz camPos, w view mode (0 final, 1 LOD, 2 err,
                         //             3 vis color, 4 checker)
    float4 lightDir;     // xyz TO light (world), w instanceCount
    float4 kdScale;      // rgb kd, w max LOD level (mode 1 ramp)
    float4 viewport;     // w, h, clusterCount, groupCount
};

[[vk::push_constant]] ConstantBuffer<DecodePush> push;

// The VisBuffer bytes read as u32 pairs (little-endian: [0]=payload,
// [1]=depth). The payload never touches u64 ALU in this shader: shifting
// a loaded u64 and feeding the result into SSBO address arithmetic
// hard-crashes this driver (DEVICE_LOST on zoom; the h-1-y row-index
// crash was the same pattern).
[[vk::binding(0)]] StructuredBuffer<uint> vis;
[[vk::binding(1)]] StructuredBuffer<GpuCluster> clusters;
[[vk::binding(2)]] StructuredBuffer<GpuInstance> instances;
[[vk::binding(3)]] StructuredBuffer<uint> positions;  // v2: float4-as-4-u32; paged v5: the slot pool
[[vk::binding(4)]] StructuredBuffer<uint> normals;     // v2: float4-as-4-u32; paged v5: the slot pool
[[vk::binding(5)]] StructuredBuffer<uint> uvs;         // v2: float2-as-2-u32; paged v5: the slot pool
[[vk::binding(6)]] StructuredBuffer<uint> indices;
[[vk::binding(7)]] StructuredBuffer<GpuGroup> groups;  // S4.6 debug views
// M6.3 paged addressing (decodeParams[0].x != 0): bindings 3..6 are the
// PAGE SLOT POOL, cluster offsets are page-local, and every read adds
// the slot-strided section base. No residency check is needed: pixels
// of non-resident clusters cannot exist - the rasterizers skipped
// those clusters and the always-resident root pages covered the area.
[[vk::binding(8)]] StructuredBuffer<uint> clusterPage;
[[vk::binding(9)]] StructuredBuffer<uint> pageTable;
[[vk::binding(10)]] StructuredBuffer<GpuPageHeader> pageHeader;
[[vk::binding(11)]] StructuredBuffer<float4> decodeParams;
// M6.4 wide-instance payload indirection: the VisBuffer payload is
// pairIndex 23b | listSel 2b | triangle 7b - these are the SAME
// frame's pair lists the rasterizers consumed (12 = the SW
// visiblePairs, 13 = the HW hwPairs, 14 = the post-revived pairs).
// decodeParams[1].x carries the shared capacity for the clamp.
[[vk::binding(12)]] StructuredBuffer<uint2> pairs;
[[vk::binding(13)]] StructuredBuffer<uint2> hwPairs;
[[vk::binding(14)]] StructuredBuffer<uint2> postPairs;
[[vk::binding(15)]] StructuredBuffer<uint> clusterMeta;  // v5: per-cluster ordinal into its page's meta array

// kPageSlotSize (128 KB) in u32 elements - mirrors rasterize_sw / CPU
// (v5: all page sections are u32-addressed).
static const uint kSlotU32s = 32768;

float2 worldToScreen(float3 w)
{
    // Explicit glm-column combination (Slang mul reads rows); raster
    // parity: the camera's own (Y-flipped) proj, row = framebuffer row.
    // The w-divide denominator is CLAMPED: near-plane-crossing triangles
    // have vertices at clip w ~ 0 (or <= 0 behind the camera), and a raw
    // division emits Inf/NaN - executing that pattern hard-crashes this
    // driver (DEVICE_LOST), and a discard guard does NOT help because
    // the compiler speculatively schedules the division anyway. Finite
    // garbage is safe garbage.
    const float4x4 m = push.viewProj;
    const float4 c = m[0] * w.x + m[1] * w.y + m[2] * w.z + m[3];
    const float2 ndc = c.xy / max(c.w, 0.001);
    return (ndc * 0.5 + 0.5) * push.viewport.xy;
}

float clipW(float3 w)
{
    const float4x4 m = push.viewProj;
    const float4 c = m[0] * w.x + m[1] * w.y + m[2] * w.z + m[3];
    return c.w;
}

float3 applyModel(float3 lp, uint instanceIdx)
{
    const float4x4 model = instances[instanceIdx].model;
    return model[0].xyz * lp.x + model[1].xyz * lp.y
         + model[2].xyz * lp.z + model[3].xyz;
}

struct VSOut
{
    float4 pos : SV_Position;
};

[shader("vertex")]
VSOut vsMain(uint id : SV_VertexID)
{
    VSOut o;
    o.pos = float4(float2((id << 1u) & 2u, id & 2u) * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

[shader("fragment")]
float4 fsMain(float4 frag : SV_Position) : SV_Target
{
    const float2 wh = push.viewport.xy;
    const uint2 pixel = uint2(frag.xy);
    // The chain feeds the rasterizer a Y-FLIPPED projection, so its row
    // order IS the framebuffer's (row 0 = top) - index directly.
    const uint visLo = vis[(pixel.y * uint(wh.x) + pixel.x) * 2u];
    if (visLo == 0u)
        discard;  // empty pixel: keep beginFrame's clear color (dual-path
                  // background parity with the baseline screenshot)

    // Wide-instance payload (see bindings 12..14): one indirection into
    // this frame's pair tables - instance counts are no longer capped
    // by payload bits. Every payload-driven index stays clamped (the
    // driver's payload-dependent addressing chain discipline).
    const uint triIdx = visLo & 0x7Fu;
    const uint listSel = (visLo >> 7) & 3u;
    const uint pairIdx = min((visLo >> 9) & 0x7FFFFFu,
                             max(uint(decodeParams[1].x), 1u) - 1u);
    const uint2 pair = listSel == 1u   ? hwPairs[pairIdx]
                       : listSel == 2u ? postPairs[pairIdx]
                                       : pairs[pairIdx];
    const uint instanceIdx = min(pair.x, uint(push.lightDir.w) - 1u);
    const uint clusterIdx = min(pair.y, uint(push.viewport.z) - 1u);
    const GpuCluster cluster = clusters[clusterIdx];
    const uint viewMode = uint(push.camFlags.w + 0.5);

    // v5 paged section bases + the cluster's quantization record (see
    // the binding comment). The derived bases clamp into the pool total
    // (decodeParams[0].y - one u32 capacity for every section, all
    // sections share the slot pool) on every use - the multi-level
    // payload-dependent addressing chain is this driver's documented
    // DEVICE_LOST trigger, and the value-level clamps are what dodge
    // it (the S4.5 discipline).
    uint posBase = 0, nrmBase = 0, uvBase = 0, idxBase = 0;
    uint posTotal = 0xFFFFFFFFu, idxTotal = 0xFFFFFFFFu;
    GpuClusterQuant qm;
    float posStep = 1.0;
    const bool paged = uint(decodeParams[0].x) != 0u;
    if (paged)
    {
        const uint page = clusterPage[clusterIdx];
        const GpuPageHeader ph = pageHeader[page];
        const uint slot = pageTable[page];
        const uint pool = slot * kSlotU32s;
        posTotal = max(uint(decodeParams[0].y), 1u);
        idxTotal = posTotal;
        posBase = min(pool + ph.posBase, posTotal - 1u);
        nrmBase = min(pool + ph.nrmBase, posTotal - 1u);
        uvBase  = min(pool + ph.uvBase, posTotal - 1u);
        idxBase = min(pool + ph.idxBase, posTotal - 1u);
        posStep = ph.posStep;
        // EVERY meta read is clamped against the pool capacity - the
        // multi-level payload->cluster->page->meta chain is this
        // driver's documented DEVICE_LOST trigger, and decode serves
        // ARBITRARY pixels (first-frame stale payloads included).
        const uint ord = min(clusterMeta[clusterIdx], ph.clusterCount - 1u);
        const uint metaEnd = posTotal - 10u;
        const uint mb = min(pool + ph.metaBase + ord * 10u, metaEnd);
        qm.posStartX = int(positions[min(mb, metaEnd)]);
        qm.posStartY = int(positions[min(mb + 1u, metaEnd)]);
        qm.posStartZ = int(positions[min(mb + 2u, metaEnd)]);
        qm.widths = positions[min(mb + 3u, metaEnd)];
        qm.posWordOff = positions[min(mb + 4u, metaEnd)];
        qm.uvMinX = asfloat(positions[min(mb + 5u, metaEnd)]);
        qm.uvMinY = asfloat(positions[min(mb + 6u, metaEnd)]);
        qm.uvRangeX = asfloat(positions[min(mb + 7u, metaEnd)]);
        qm.uvRangeY = asfloat(positions[min(mb + 8u, metaEnd)]);
        qm.idxBits = positions[min(mb + 9u, metaEnd)];
    }


    if (viewMode == 1u)
    {
        // LOD-level view: one hue PER LEVEL via a golden-ratio hash -
        // adjacent levels always land far apart on the color wheel, so
        // any visible level range contrasts strongly (a continuous
        // ramp left adjacent levels in the same color family).
        const uint groupIdx =
            min(cluster.group, uint(push.viewport.w) - 1u);
        const float level = float(groups[groupIdx].lodLevel) + 0.5;
        const float hue = frac(level * 0.618034) * 6.0;
        const int seg = int(min(hue, 5.999));
        const float f = hue - float(seg);
        const float3 wheel[7] = {
            float3(1.0, 0.0, 0.0), float3(1.0, 0.6, 0.0),
            float3(0.8, 1.0, 0.0), float3(0.0, 1.0, 0.3),
            float3(0.0, 1.0, 1.0), float3(0.3, 0.3, 1.0),
            float3(1.0, 0.0, 0.0)};
        return float4(lerp(wheel[seg], wheel[seg + 1], f), 1.0);
    }
    if (viewMode == 2u)
    {
        // Error heatmap: log-scaled cluster error, blue (tiny) -> red.
        const float e = clamp(log10(max(cluster.error, 1e-9)) / -6.0,
                              0.0, 1.0);
        return float4(1.0 - e, 0.2, 0.2 + 0.8 * e, 1.0);
    }
    if (viewMode == 3u)
    {
        // Triangle-density view: one hash color per triangle - the point
        // is seeing how close triangles sit to pixel size (the Nanite
        // goal), not telling them apart.
        uint h = (clusterIdx * 2654435761u) ^ (triIdx * 40503u);
        h ^= h >> 13u;
        return float4(float(h & 255u) / 255.0, float((h >> 8u) & 255u) / 255.0,
                      float((h >> 16u) & 255u) / 255.0, 1.0);
    }
    if (viewMode == 5u)
    {
        // Cluster view: one hash color per CLUSTER - cut boundaries and
        // cluster sizes at a glance.
        uint h = clusterIdx * 2654435761u;
        h ^= h >> 13u;
        return float4(0.25 + 0.75 * float(h & 255u) / 255.0,
                      0.25 + 0.75 * float((h >> 8u) & 255u) / 255.0,
                      0.25 + 0.75 * float((h >> 16u) & 255u) / 255.0, 1.0);
    }
    if (viewMode == 6u)
    {
        // Routing view (M4.2): which rasterizer path drew this pixel.
        // The payload's listSel IS the pair-table selector: 0 = SW main
        // (blue), 1 = HW fixed-function (red), 2 = post-pass revived
        // (yellow - occlusion-culled then re-tested). Reacts live to
        // the HW-route threshold slider.
        if (listSel == 1u) return float4(0.95, 0.25, 0.15, 1.0);
        if (listSel == 2u) return float4(0.95, 0.85, 0.15, 1.0);
        return float4(0.20, 0.45, 0.95, 1.0);
    }

    const uint tri3 = min(triIdx * 3u, cluster.indexCount - 3u);
    uint i0, i1, i2;
    if (paged)
    {
        // v6 packed triple (the cyclic min-first rotation; D5 mirror
        // = decodeQuantCluster) - per-pixel random access stays O(1):
        // the <=22-bit field spans at most the two loaded words.
        const uint tri = tri3 / 3u;
        const uint baseBits = qm.idxBits & 0xFu;
        const uint offsetBits = (qm.idxBits >> 4) & 0xFu;
        const uint bitIdx = (qm.idxBits >> 8) + tri * (baseBits + 2u * offsetBits);
        // Per-subfield word loads (a subfield may cross into the next
        // word - masking the global bit into the triangle window is
        // off by 32).
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
        i0 = min(base, cluster.vertexCount - 1u);
        i1 = min(base + o1, cluster.vertexCount - 1u);
        i2 = min(base + o2, cluster.vertexCount - 1u);
    }
    else
    {
        i0 = min(indices[min(idxBase + cluster.indexOffset + tri3 + 0u,
                             idxTotal - 1u)],
                 cluster.vertexCount - 1u);
        i1 = min(indices[min(idxBase + cluster.indexOffset + tri3 + 1u,
                             idxTotal - 1u)],
                 cluster.vertexCount - 1u);
        i2 = min(indices[min(idxBase + cluster.indexOffset + tri3 + 2u,
                             idxTotal - 1u)],
                 cluster.vertexCount - 1u);
    }
    float3 lp0, lp1, lp2;
    if (paged)
    {
        lp0 = v5Pos(i0, qm, posBase, posTotal, cluster.vertexCount, posStep);
        lp1 = v5Pos(i1, qm, posBase, posTotal, cluster.vertexCount, posStep);
        lp2 = v5Pos(i2, qm, posBase, posTotal, cluster.vertexCount, posStep);
    }
    else
    {
        const uint p0 = min((posBase + cluster.vertexOffset + i0) * 4u, posTotal - 2u);
        const uint p1 = min((posBase + cluster.vertexOffset + i1) * 4u, posTotal - 2u);
        const uint p2 = min((posBase + cluster.vertexOffset + i2) * 4u, posTotal - 2u);
        lp0 = float3(asfloat(positions[p0]), asfloat(positions[p0 + 1u]),
                     asfloat(positions[p0 + 2u]));
        lp1 = float3(asfloat(positions[p1]), asfloat(positions[p1 + 1u]),
                     asfloat(positions[p1 + 2u]));
        lp2 = float3(asfloat(positions[p2]), asfloat(positions[p2 + 1u]),
                     asfloat(positions[p2 + 2u]));
    }
    const float3 w0 = applyModel(lp0, instanceIdx);
    const float3 w1 = applyModel(lp1, instanceIdx);
    const float3 w2 = applyModel(lp2, instanceIdx);

    // NOTE: near-plane-crossing triangles (a vertex behind the camera)
    // used to be discarded outright - with the camera close to the mesh
    // that erased whole triangles and read as "clipped away". The
    // clamped denominators below already keep every division finite
    // (that, not the discard, is what keeps this driver alive), so
    // crossing pixels now SHADE approximately instead of vanishing;
    // proper crossing handling belongs to the M4 hardware path.

    // Raster-parity screen position of THIS pixel (framebuffer row
    // order, matching the flipped-proj rasterizer convention).
    const float2 pr = float2(pixel.x + 0.5, pixel.y + 0.5);

    const float2 s0 = worldToScreen(w0);
    const float2 s1 = worldToScreen(w1);
    const float2 s2 = worldToScreen(w2);

    // Analytic barycentrics: solve p = s0 + u*(s1-s0) + v*(s2-s0).
    const float2 e1 = s1 - s0;
    const float2 e2 = s2 - s0;
    const float det = e1.x * e2.y - e1.y * e2.x;
    if (abs(det) < 1e-12) return float4(1, 0, 1, 1);
    const float2 dp = pr - s0;
    const float u = (dp.x * e2.y - dp.y * e2.x) / det;
    const float v = (e1.x * dp.y - e1.y * dp.x) / det;
    const float b0 = 1.0 - u - v;

    // Perspective-correct weights (Karis p.99): screen-space barycentrics
    // interpolate AFFINELY, so attributes need the b/w rational form -
    // matching what the hardware interpolator does for the baseline.
    // Clamped denominators: same Inf/NaN-driver-crash story as
    // worldToScreen (crossing triangles reach here with the guard passed
    // but a vertex a hair in front of the plane).
    const float cw0 = max(clipW(w0), 0.001);
    const float cw1 = max(clipW(w1), 0.001);
    const float cw2 = max(clipW(w2), 0.001);
    const float p0 = b0 / cw0, p1 = u / cw1, p2 = v / cw2;
    // pSum crosses zero at reconstruction mismatches (barycentrics land
    // outside the triangle) - a raw reciprocal goes Inf and the Inf then
    // FLOWS through the shading math (the end-of-shader NaN scrub does
    // not prevent executing it). Clamp the magnitude, keep the sign.
    const float pSum = p0 + p1 + p2;
    const float pSumSafe = abs(pSum) < 1e-9f ? (pSum < 0.0 ? -1e-9f : 1e-9f) : pSum;
    const float pSumInv = 1.0 / pSumSafe;

    // Interpolated attributes.
    float3 n0, n1, n2;
    if (paged)
    {
        n0 = v5Nrm(i0, nrmBase, posTotal, cluster.vertexOffset);
        n1 = v5Nrm(i1, nrmBase, posTotal, cluster.vertexOffset);
        n2 = v5Nrm(i2, nrmBase, posTotal, cluster.vertexOffset);
    }
    else
    {
        const uint n0a = min((nrmBase + cluster.vertexOffset + i0) * 4u, posTotal - 2u);
        const uint n1a = min((nrmBase + cluster.vertexOffset + i1) * 4u, posTotal - 2u);
        const uint n2a = min((nrmBase + cluster.vertexOffset + i2) * 4u, posTotal - 2u);
        n0 = float3(asfloat(normals[n0a]), asfloat(normals[n0a + 1u]), asfloat(normals[n0a + 2u]));
        n1 = float3(asfloat(normals[n1a]), asfloat(normals[n1a + 1u]), asfloat(normals[n1a + 2u]));
        n2 = float3(asfloat(normals[n2a]), asfloat(normals[n2a + 1u]), asfloat(normals[n2a + 2u]));
    }
    const float3 nrm = normalize(n0 * p0 + n1 * p1 + n2 * p2);  // scale cancels
    float2 uv0, uv1, uv2;
    if (paged)
    {
        uv0 = v5Uv(i0, uvBase, posTotal, cluster.vertexOffset, qm);
        uv1 = v5Uv(i1, uvBase, posTotal, cluster.vertexOffset, qm);
        uv2 = v5Uv(i2, uvBase, posTotal, cluster.vertexOffset, qm);
    }
    else
    {
        const uint uvTotal = posTotal;  // no clamp in v2 (both ~0xFFFFFFFF)
        const uint a = min((uvBase + cluster.vertexOffset + i0) * 2u, uvTotal - 2u);
        const uint b = min((uvBase + cluster.vertexOffset + i1) * 2u, uvTotal - 2u);
        const uint c2 = min((uvBase + cluster.vertexOffset + i2) * 2u, uvTotal - 2u);
        uv0 = float2(asfloat(uvs[a]), asfloat(uvs[a + 1u]));
        uv1 = float2(asfloat(uvs[b]), asfloat(uvs[b + 1u]));
        uv2 = float2(asfloat(uvs[c2]), asfloat(uvs[c2 + 1u]));
    }
    const float2 uv = (uv0 * p0 + uv1 * p1 + uv2 * p2) * pSumInv;

    // Analytic UV derivatives of the rational interpolation (no
    // ddx/ddy - quads span triangles/seams in a VisBuffer). uv(x,y) =
    // A(x,y)/D(x,y) with p_i linear in the screen barycentrics:
    // d(uv)/d = (A' D - A D') / D^2. Barycentric gradients from the
    // inverse edge matrix.
    const float invDet = 1.0 / det;
    const float dudx = e2.y * invDet, dudy = -e2.x * invDet;
    const float dvdx = -e1.y * invDet, dvdy = e1.x * invDet;
    const float dp0x = -(dudx + dvdx) / cw0;
    const float dp0y = -(dudy + dvdy) / cw0;
    const float dp1x = dudx / cw1, dp1y = dudy / cw1;
    const float dp2x = dvdx / cw2, dp2y = dvdy / cw2;
    const float2 A = uv0 * p0 + uv1 * p1 + uv2 * p2;
    const float D = p0 + p1 + p2;
    const float2 Ax = float2(uv0.x * dp0x + uv1.x * dp1x + uv2.x * dp2x,
                             uv0.y * dp0x + uv1.y * dp1x + uv2.y * dp2x);
    const float2 Ay = float2(uv0.x * dp0y + uv1.x * dp1y + uv2.x * dp2y,
                             uv0.y * dp0y + uv1.y * dp1y + uv2.y * dp2y);
    const float Dx = dp0x + dp1x + dp2x;
    const float Dy = dp0y + dp1y + dp2y;
    const float dSafe = abs(D) < 1e-9f ? (D < 0.0 ? -1e-9f : 1e-9f) : D;
    const float2 dUVdx = (Ax * D - A * Dx) / (dSafe * dSafe);
    const float2 dUVdy = (Ay * D - A * Dy) / (dSafe * dSafe);

    float3 kd = push.kdScale.rgb;
    if (viewMode == 4u)
    {
        // Checkerboard debug: analytic derivatives pick the mip-parity
        // scale (2 px cells at base) - seam integrity is then visible
        // as any discontinuity mismatch.
        const float scale = push.kdScale.w;
        const float2 c = floor(uv * scale);
        const float checker = fmod(c.x + c.y, 2.0);
        const float2 duv = abs(dUVdx) + abs(dUVdy);
        const float2 w2f = clamp(duv * 0.5 * scale, float2(0.0), float2(1.0));
        const float fade = max(w2f.x, w2f.y);  // afar -> blend to gray
        kd = lerp(kd * (0.35 + 0.65 * checker), float3(0.5, 0.5, 0.5), fade);
    }

    // Blinn-Phong - the baseline path's formula verbatim.
    const float3 l = normalize(push.lightDir.xyz);
    const float3 worldPos =
        (w0 * p0 + w1 * p1 + w2 * p2) * pSumInv;  // perspective-correct
    const float3 viewDir = normalize(push.camFlags.xyz - worldPos);
    const float3 h = normalize(l + viewDir);
    const float ndl = max(0.0, dot(nrm, l));
    const float ndh = max(0.0, dot(nrm, h));
    const float spec = pow(max(ndh, 0.0), 48.0) * 0.35 * (ndl > 0.0 ? 1.0 : 0.0);
    const float3 color = kd * (0.15 + 0.85 * ndl) + spec;
    // NaN scrub for the crossing-triangle garbage paths (NaN compares
    // false, so !(x>=0) catches it) - keep every written color finite.
    if (!(color.x >= 0.0) || !(color.y >= 0.0) || !(color.z >= 0.0))
        return float4(0.5, 0.0, 0.5, 1.0);
    return float4(color, 1.0);
}

// ── M8.1: the GBuffer variant ──────────────────────────────────────
// Same decode chain as fsMain's mode-0 path (kept as a deliberate
// MIRROR - the repo's D5-style duplication discipline: the legacy
// entry is battle-tested against driver quirks; do not refactor one
// without the other). Differences: no UV (no textures yet), the
// normal is transformed to WORLD space (the legacy path's object-space
// normal was a latent bug only hidden by translation-only demos), and
// the outputs are the deferred GBuffer:
//   SV_Target0 (RGBA16F): rgb = world normal, a = euclidean distance
//                         to the camera (exact worldPos roundtrip:
//                         camPos + normalize(ray) * dist)
//   SV_Target1 (RGBA8):   rgb = base color (global kd until M8.2's
//                         material table), a = reserved (material id)
// Background: discarded -> the pass's zero clear; the lighting pass
// reads alpha==0 as background.
struct GBufferOut
{
    float4 a : SV_Target0;
    float4 b : SV_Target1;
};

[shader("fragment")]
GBufferOut fsGBuffer(float4 frag : SV_Position)
{
    GBufferOut o;
    o.a = float4(0, 0, 0, 0);
    o.b = float4(0, 0, 0, 0);

    const float2 wh = push.viewport.xy;
    const uint2 pixel = uint2(frag.xy);
    const uint visLo = vis[(pixel.y * uint(wh.x) + pixel.x) * 2u];
    if (visLo == 0u)
        discard;

    const uint triIdx = visLo & 0x7Fu;
    const uint listSel = (visLo >> 7) & 3u;
    const uint pairIdx = min((visLo >> 9) & 0x7FFFFFu,
                             max(uint(decodeParams[1].x), 1u) - 1u);
    const uint2 pair = listSel == 1u   ? hwPairs[pairIdx]
                       : listSel == 2u ? postPairs[pairIdx]
                                       : pairs[pairIdx];
    const uint instanceIdx = min(pair.x, uint(push.lightDir.w) - 1u);
    const uint clusterIdx = min(pair.y, uint(push.viewport.z) - 1u);
    const GpuCluster cluster = clusters[clusterIdx];

    uint posBase = 0, nrmBase = 0, uvBase = 0, idxBase = 0;
    uint posTotal = 0xFFFFFFFFu, idxTotal = 0xFFFFFFFFu;
    GpuClusterQuant qm;
    float posStep = 1.0;
    const bool paged = uint(decodeParams[0].x) != 0u;
    if (paged)
    {
        const uint page = clusterPage[clusterIdx];
        const GpuPageHeader ph = pageHeader[page];
        const uint slot = pageTable[page];
        const uint pool = slot * kSlotU32s;
        posTotal = max(uint(decodeParams[0].y), 1u);
        idxTotal = posTotal;
        posBase = min(pool + ph.posBase, posTotal - 1u);
        nrmBase = min(pool + ph.nrmBase, posTotal - 1u);
        idxBase = min(pool + ph.idxBase, posTotal - 1u);
        posStep = ph.posStep;
        const uint ord = min(clusterMeta[clusterIdx], ph.clusterCount - 1u);
        const uint metaEnd = posTotal - 10u;
        const uint mb = min(pool + ph.metaBase + ord * 10u, metaEnd);
        qm.posStartX = int(positions[min(mb, metaEnd)]);
        qm.posStartY = int(positions[min(mb + 1u, metaEnd)]);
        qm.posStartZ = int(positions[min(mb + 2u, metaEnd)]);
        qm.widths = positions[min(mb + 3u, metaEnd)];
        qm.posWordOff = positions[min(mb + 4u, metaEnd)];
        qm.idxBits = positions[min(mb + 9u, metaEnd)];
    }

    const uint tri3 = min(triIdx * 3u, cluster.indexCount - 3u);
    uint i0, i1, i2;
    if (paged)
    {
        const uint tri = tri3 / 3u;
        const uint baseBits = qm.idxBits & 0xFu;
        const uint offsetBits = (qm.idxBits >> 4) & 0xFu;
        const uint bitIdx = (qm.idxBits >> 8) + tri * (baseBits + 2u * offsetBits);
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
        i0 = min(base, cluster.vertexCount - 1u);
        i1 = min(base + o1, cluster.vertexCount - 1u);
        i2 = min(base + o2, cluster.vertexCount - 1u);
    }
    else
    {
        i0 = min(indices[min(idxBase + cluster.indexOffset + tri3 + 0u,
                             idxTotal - 1u)],
                 cluster.vertexCount - 1u);
        i1 = min(indices[min(idxBase + cluster.indexOffset + tri3 + 1u,
                             idxTotal - 1u)],
                 cluster.vertexCount - 1u);
        i2 = min(indices[min(idxBase + cluster.indexOffset + tri3 + 2u,
                             idxTotal - 1u)],
                 cluster.vertexCount - 1u);
    }
    float3 lp0, lp1, lp2;
    if (paged)
    {
        lp0 = v5Pos(i0, qm, posBase, posTotal, cluster.vertexCount, posStep);
        lp1 = v5Pos(i1, qm, posBase, posTotal, cluster.vertexCount, posStep);
        lp2 = v5Pos(i2, qm, posBase, posTotal, cluster.vertexCount, posStep);
    }
    else
    {
        const uint p0 = min((posBase + cluster.vertexOffset + i0) * 4u, posTotal - 2u);
        const uint p1 = min((posBase + cluster.vertexOffset + i1) * 4u, posTotal - 2u);
        const uint p2 = min((posBase + cluster.vertexOffset + i2) * 4u, posTotal - 2u);
        lp0 = float3(asfloat(positions[p0]), asfloat(positions[p0 + 1u]),
                     asfloat(positions[p0 + 2u]));
        lp1 = float3(asfloat(positions[p1]), asfloat(positions[p1 + 1u]),
                     asfloat(positions[p1 + 2u]));
        lp2 = float3(asfloat(positions[p2]), asfloat(positions[p2 + 1u]),
                     asfloat(positions[p2 + 2u]));
    }
    const float3 w0 = applyModel(lp0, instanceIdx);
    const float3 w1 = applyModel(lp1, instanceIdx);
    const float3 w2 = applyModel(lp2, instanceIdx);

    const float2 pr = float2(pixel.x + 0.5, pixel.y + 0.5);
    const float2 s0 = worldToScreen(w0);
    const float2 s1 = worldToScreen(w1);
    const float2 s2 = worldToScreen(w2);
    const float2 e1 = s1 - s0;
    const float2 e2 = s2 - s0;
    const float det = e1.x * e2.y - e1.y * e2.x;
    if (abs(det) < 1e-12) discard;
    const float2 dp = pr - s0;
    const float u = (dp.x * e2.y - dp.y * e2.x) / det;
    const float v = (e1.x * dp.y - e1.y * dp.x) / det;
    const float b0 = 1.0 - u - v;
    const float cw0 = max(clipW(w0), 0.001);
    const float cw1 = max(clipW(w1), 0.001);
    const float cw2 = max(clipW(w2), 0.001);
    const float p0 = b0 / cw0, p1 = u / cw1, p2 = v / cw2;
    const float pSum = p0 + p1 + p2;
    const float pSumSafe = abs(pSum) < 1e-9f ? (pSum < 0.0 ? -1e-9f : 1e-9f) : pSum;
    const float pSumInv = 1.0 / pSumSafe;

    float3 n0, n1, n2;
    if (paged)
    {
        n0 = v5Nrm(i0, nrmBase, posTotal, cluster.vertexOffset);
        n1 = v5Nrm(i1, nrmBase, posTotal, cluster.vertexOffset);
        n2 = v5Nrm(i2, nrmBase, posTotal, cluster.vertexOffset);
    }
    else
    {
        const uint n0a = min((nrmBase + cluster.vertexOffset + i0) * 4u, posTotal - 2u);
        const uint n1a = min((nrmBase + cluster.vertexOffset + i1) * 4u, posTotal - 2u);
        const uint n2a = min((nrmBase + cluster.vertexOffset + i2) * 4u, posTotal - 2u);
        n0 = float3(asfloat(normals[n0a]), asfloat(normals[n0a + 1u]), asfloat(normals[n0a + 2u]));
        n1 = float3(asfloat(normals[n1a]), asfloat(normals[n1a + 1u]), asfloat(normals[n1a + 2u]));
        n2 = float3(asfloat(normals[n2a]), asfloat(normals[n2a + 1u]), asfloat(normals[n2a + 2u]));
    }
    const float3 nrmObj = n0 * p0 + n1 * p1 + n2 * p2;
    // Object -> world: the instance model's inverse-transpose 3x3,
    // written as the standard cross-product form (Slang has no inverse()
    // intrinsic; for M with columns c0..c2, M^{-T}*n = (cross(c1,c2)*n.x
    // + cross(c2,c0)*n.y + cross(c0,c1)*n.z)/det - translation-only
    // models leave it identity, so the pure-translation demos are
    // bit-identical with the legacy object-space normal; rotated/scaled
    // instances now light correctly).
    const float4x4 model = instances[instanceIdx].model;
    const float3 c0 = model[0].xyz, c1 = model[1].xyz, c2 = model[2].xyz;
    const float3 cross0 = cross(c1, c2);
    const float detM = dot(c0, cross0);
    const float invDet =
        abs(detM) < 1e-20 ? (detM < 0.0 ? -1e20 : 1e20) : 1.0 / detM;
    const float3 nrm = normalize(
        (cross0 * nrmObj.x + cross(c2, c0) * nrmObj.y +
         cross(c0, c1) * nrmObj.z) * invDet);

    const float3 worldPos = (w0 * p0 + w1 * p1 + w2 * p2) * pSumInv;
    const float dist = length(worldPos - push.camFlags.xyz);

    o.a = float4(nrm, dist);
    o.b = float4(push.kdScale.rgb, 0.0);
    // NaN scrub (crossing-triangle garbage paths) - the lighting pass
    // must never read non-finite GBuffer data.
    if (!(nrm.x >= -1.0 && nrm.x <= 1.0) || !(dist >= 0.0) ||
        !(o.b.x >= 0.0))
    {
        o.a = float4(0, 0, 0, 0);  // reads as background
    }
    return o;
}
