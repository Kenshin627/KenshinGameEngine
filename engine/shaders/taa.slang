// M9 TAA: the temporal resolve between lighting and tonemap. The
// geometry passes render with a per-frame subpixel jitter (Halton
// (2,3), applied to the projection's NDC offsets in main.cpp); this
// pass reprojects the PREVIOUS resolved frame onto the current one
// and blends (UE's accumulate-and-clamp shape, v1.5: jitter-
// compensated current fetch + a motion-adaptive weight in place of a
// velocity buffer).
//
// Reprojection: GBuffer A's alpha is the EUCLIDEAN view-ray distance
// (decode_material.slang fsGBuffer's contract), so the world position
// is camPos + normalize(far - near) * dist with the two unprojected
// NDC-z corners of the pixel's ray. The matrices are the UNJITTERED
// pair on purpose: jitter must never leak into reprojection.
//
// Background (dist == 0) bypasses history: the regression net's
// byte-exact background contract rides the raw current sample.
//
// Bindings (set 0):
//   0..2: Sampler2D currentHdr / history / gbufferA (bound by the
//         backend in this order - slot HDR, the ping-pong history,
//         the slot's GBuffer A view)
//   3: SSBO taaParams - the two matrices + camera/blend/texel/jitter
//      (a push block would need 160B+; the guaranteed push limit is
//      128)

struct TaaParams
{
    // Matrices as glm-raw COLUMN arrays (the codebase idiom: Slang
    // chokes on operator* over StructuredBuffer matrix members, and
    // the manual column sum matches glm's column-major layout
    // bit-for-bit).
    float4 invVP[4];   // current UNJITTERED inverse view-projection
    float4 prevVP[4];  // previous UNJITTERED view-projection
    float4 camBlend;   // xyz camera position, w = history blend weight
                       // (0 = no history: first frames / resize / toggle)
    float4 texel;      // xy = 1/w, 1/h, z scratch, w = sharpen strength
    float4 jitter;     // xy = THIS frame's jitter in UV units (ndc/2):
                       // the render shifted content to +j, so the
                       // resolve reads it at uv - j
};

[[vk::binding(0)]] Sampler2D currentHdr;
[[vk::binding(1)]] Sampler2D history;
[[vk::binding(2)]] Sampler2D gbufferA;
[[vk::binding(3)]] StructuredBuffer<TaaParams> taaParams;

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

float3 unproj(float2 ndc, float z, TaaParams tp)
{
    const float4 v = float4(ndc, z, 1.0);
    const float4 p = tp.invVP[0] * v.x + tp.invVP[1] * v.y +
                     tp.invVP[2] * v.z + tp.invVP[3] * v.w;
    return p.xyz / p.w;
}

[shader("fragment")]
float4 fsMain(float4 frag : SV_Position) : SV_Target
{
    const TaaParams tp = taaParams[0];
    const float2 uv = frag.xy * tp.texel.xy;
    // Jitter-compensated current fetch: the render shifted content to
    // +j; sampling at uv - j reads it aligned (without this the raw
    // per-phase wobble leaks through the clamp - the perceptible
    // shimmer on high-frequency content).
    const float2 cuv = uv - tp.jitter.xy;
    const float3 cur = currentHdr.Sample(cuv).rgb;
    const float dist = gbufferA.Sample(uv).a;
    const float blend = tp.camBlend.w;
    if (blend <= 0.0)
        return float4(cur, 1.0);
    if (dist <= 0.0)
    {
        // Silhouette coverage: under jitter the edge pixels toggle
        // surface/background per phase - the raw early-out made the
        // background phases skip the history entirely (full-contrast
        // phase flipping = the edge shimmer). Background is at
        // infinity (clear color): sample the history AT THIS PIXEL
        // (parallax-free) and blend lightly - the edge resolves to a
        // temporal coverage average like the surface side.
        const float3 h = history.Sample(uv).rgb;
        return float4(lerp(cur, h, blend * 0.5), 1.0);
    }

    // World position of this pixel's surface, then last frame's UV.
    const float2 ndc = uv * 2.0 - 1.0;
    const float3 dir =
        normalize(unproj(ndc, 1.0, tp) - unproj(ndc, -1.0, tp));

    // ---- v2: closest-depth neighbor + variance clipping (Karis
    // "Temporal AA" 4.3) -------------------------------------------
    // The 3x3 pass collects colors (for the clip bounds) AND each
    // texel's euclidean view distance (GBuffer A's alpha) to pick the
    // CLOSEST surface - at a depth edge the nearest neighbor is the
    // surface that should own this pixel, and reprojection through IT
    // keeps the history sample on the correct surface (v1 reprojected
    // the center texel = wrong surface at edges = edge shimmer).
    float3 cMin = cur, cMax = cur, cSum = cur, cSum2 = cur * cur;
    float bestDist = dist;
    float2 bestNdc = ndc;
    // v2.1: radius 2 (5x5). A 3x3 at minification boundaries is
    // phase-dependent homogeneous (all-black one phase, all-white the
    // next) - the tight box clamps history to the SINGLE PHASE and
    // the output flickers at full contrast. 5x5 spans more phases ->
    // mixed box -> history keeps the temporal supersample.
    [unroll]
    for (int dy = -2; dy <= 2; ++dy)
        [unroll]
        for (int dx = -2; dx <= 2; ++dx)
        {
            if (dx == 0 && dy == 0) continue;
            const float2 suv = uv + float2(dx, dy) * tp.texel.xy;
            const float3 c = currentHdr.Sample(suv - tp.jitter.xy).rgb;
            cMin = min(cMin, c);
            cMax = max(cMax, c);
            cSum += c;
            cSum2 += c * c;
            const float nd = gbufferA.Sample(suv).a;
            if (nd > 0.0 && nd < bestDist)
            {
                bestDist = nd;
                bestNdc = suv * 2.0 - 1.0;
            }
        }
    // Reprojection point: the CENTER's own surface under a static or
    // near-static camera, the CLOSEST-depth neighbor's surface under
    // real motion. The center keeps each pixel on its OWN history at
    // rest (the closest-neighbor choice reads the NEIGHBOR's history -
    // per-phase flips of which neighbor is closest were a strong-flip
    // source, 53 -> 39 with the center); under motion the closest
    // surface is the one that should own this pixel (Karis 4.3 - the
    // velocity-buffer trick approximated through the world point).
    const float3 cworld = tp.camBlend.xyz + dir * dist;
    const float4 cw = float4(cworld, 1.0);
    const float4 cc = tp.prevVP[0] * cw.x + tp.prevVP[1] * cw.y +
                      tp.prevVP[2] * cw.z + tp.prevVP[3] * cw.w;
    if (cc.w <= 0.0)
        return float4(cur, 1.0);
    const float2 cpuv = (cc.xy / cc.w) * 0.5 + 0.5;
    const float2 cdpx = (cpuv - uv) / tp.texel.xy;
    float2 puv = cpuv;
    if (dot(cdpx, cdpx) > 1.0)
    {
        const float3 bdir =
            normalize(unproj(bestNdc, 1.0, tp) - unproj(bestNdc, -1.0, tp));
        const float3 world = tp.camBlend.xyz + bdir * bestDist;
        const float4 w4 = float4(world, 1.0);
        const float4 pc = tp.prevVP[0] * w4.x + tp.prevVP[1] * w4.y +
                          tp.prevVP[2] * w4.z + tp.prevVP[3] * w4.w;
        if (pc.w <= 0.0)
            return float4(cur, 1.0);
        puv = (pc.xy / pc.w) * 0.5 + 0.5;
    }
    if (puv.x < 0.0 || puv.y < 0.0 || puv.x > 1.0 || puv.y > 1.0)
        return float4(cur, 1.0);  // left the frame: disocclusion

    // ---- v2: variance clipping replaces the raw AABB ---------------
    // mean +- gamma*sigma intersected with the 3x3 AABB: tolerates
    // AA-energy inside the neighborhood while still rejecting ghosts
    // (Karis's VarClamp, gamma 1.25).
    const float3 mean = cSum / 25.0;
    const float3 sigma = sqrt(max(cSum2 / 25.0 - mean * mean, float3(0.0)));
    float3 mn = max(cMin, mean - sigma * 1.25);
    float3 mx = min(cMax, mean + sigma * 1.25);
    if (any(mn > mx)) { mn = cMin; mx = cMax; }  // empty: fall to AABB

    // Motion no longer kills the blend (v1's falloff): closest-depth
    // reprojection + variance clip keep the history valid UNDER
    // motion; only a huge displacement (>64px, a cut) takes current.
    const float2 dpx = (puv - uv) / tp.texel.xy;
    const float blendM = dot(dpx, dpx) > 4096.0 ? 0.0 : blend;
    float3 h = history.Sample(puv).rgb;
    h = clamp(h, mn, mx);
    // NOTE: NO sharpen here - the resolve output IS next frame's
    // history; baking an unsharp into it re-applies the sharpen every
    // frame (exponential ringing, the v1.6 "completely broken image").
    // The sharpen lives in the tonemap's sample of this resolve.
    return float4(lerp(cur, h, blendM), 1.0);
}
