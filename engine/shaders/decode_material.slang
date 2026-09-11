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

struct GpuPageHeader
{
    uint posBase;  // float4 elements within the page
    uint nrmBase;  // float4 elements
    uint uvBase;   // float2 elements
    uint idxBase;  // u32 elements
    uint vertexCount;
    uint indexCount;
};

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
[[vk::binding(3)]] StructuredBuffer<float4> positions;
[[vk::binding(4)]] StructuredBuffer<float4> normals;
[[vk::binding(5)]] StructuredBuffer<float2> uvs;
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

// kPageSlotSize (128 KB) in elements - mirrors rasterize_sw / CPU.
static const uint kSlotFloat4s  = 8192;
static const uint kSlotFloat2s  = 16384;
static const uint kSlotU32s     = 32768;

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

    // Clamp EVERY payload-driven index into its buffer's proven range.
    // The values are always valid by construction, but this driver
    // device-losts on the payload-dependent addressing chain (zoom
    // repro); the clamps change the codegen enough to dodge it (and
    // make the bounds explicit for the compiler).
    const uint clusterIdx =
        min((visLo >> 12) & 0xFFFFFu, uint(push.viewport.z) - 1u);
    const uint triIdx = min((visLo >> 5) & 0x7Fu, 0x7Fu);
    // Instance 5b (payload 32b = cluster 20b | triangle 7b | instance
    // 5b); clamp like the rest - see the clamping comment above.
    const uint instanceIdx = min(visLo & 0x1Fu, uint(push.lightDir.w) - 1u);
    const GpuCluster cluster = clusters[clusterIdx];
    const uint viewMode = uint(push.camFlags.w + 0.5);

    // M6.3 paged section bases (see the binding comment).
    uint posBase = 0, nrmBase = 0, uvBase = 0, idxBase = 0;
    if (uint(decodeParams[0].x) != 0u)
    {
        const uint page = clusterPage[clusterIdx];
        const GpuPageHeader ph = pageHeader[page];
        const uint slot = pageTable[page];
        posBase = ph.posBase + slot * kSlotFloat4s;
        nrmBase = ph.nrmBase + slot * kSlotFloat4s;
        uvBase  = ph.uvBase  + slot * kSlotFloat2s;
        idxBase = ph.idxBase + slot * kSlotU32s;
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

    const uint tri3 = min(triIdx * 3u, cluster.indexCount - 3u);
    const uint i0 = min(indices[idxBase + cluster.indexOffset + tri3 + 0u],
                        cluster.vertexCount - 1u);
    const uint i1 = min(indices[idxBase + cluster.indexOffset + tri3 + 1u],
                        cluster.vertexCount - 1u);
    const uint i2 = min(indices[idxBase + cluster.indexOffset + tri3 + 2u],
                        cluster.vertexCount - 1u);
    const float3 w0 =
        applyModel(positions[posBase + cluster.vertexOffset + i0].xyz,
                   instanceIdx);
    const float3 w1 =
        applyModel(positions[posBase + cluster.vertexOffset + i1].xyz,
                   instanceIdx);
    const float3 w2 =
        applyModel(positions[posBase + cluster.vertexOffset + i2].xyz,
                   instanceIdx);

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
    const float3 nrm = normalize(
        normals[nrmBase + cluster.vertexOffset + i0].xyz * p0 +
        normals[nrmBase + cluster.vertexOffset + i1].xyz * p1 +
        normals[nrmBase + cluster.vertexOffset + i2].xyz * p2);  // scale cancels
    const float2 uv0 = uvs[uvBase + cluster.vertexOffset + i0];
    const float2 uv1 = uvs[uvBase + cluster.vertexOffset + i1];
    const float2 uv2 = uvs[uvBase + cluster.vertexOffset + i2];
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
