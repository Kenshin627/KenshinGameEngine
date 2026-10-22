// M9 TAA (v3): the temporal resolve between lighting and tonemap. The
// geometry passes render with a per-frame subpixel jitter (Halton
// (2,3) on the projection's [2][0]/[2][1] in main.cpp); this pass
// reconstructs the unjittered current frame, reprojects the PREVIOUS
// resolve onto it and blends - the UE TemporalAA.usf shape (C:\code\ue5
// Engine/Shaders/Private/TemporalAA.usf, the AA_FILTERED + AA_TONE +
// AA_YCOCG + AA_BICUBIC + AA_CLIP permutation):
//
//   1. Filtered current: the 3x3 point neighborhood weighted by
//      exp(-2.29 d^2) (UE's Blackman-Harris fit, sigma 0.47) where d is
//      the distance from each texel's UNJITTERED sample position to the
//      output pixel center, times the HDR weight 1/(4 + Y*exposure)
//      (AA_TONE: a bright speck cannot dominate the filter).
//   2. Neighborhood box in YCoCg + coverage alpha: the 3x3 AABB
//      intersected with mean +- 1.25 sigma (Karis variance clip).
//   3. Velocity from the closest-depth neighbor (camera-only motion:
//      world point through the GBuffer distance, reprojected with the
//      UNJITTERED previous view-projection). The neighbor's VELOCITY is
//      applied to THIS pixel (velocity dilation) - reprojecting the
//      neighbor's own position offsets the history fetch by the
//      neighbor offset.
//   4. History: 5-tap Catmull-Rom (UE GetBicubic2DCatmullRomSamples) -
//      bilinear history accumulates blur under motion.
//   5. Clip the history TOWARD the filtered current (AA_CLIP), then a
//      luma-weighted lerp (HdrWeight on both sides), current weight
//      lerp(1 - blend, 0.2, saturate(velocityPx / 40)). At rest the
//      clip is partial (the motion-scaled clip below).
//
// Coverage (alpha): lighting writes alpha 1 on surfaces and (0,0,0,0)
// on background (the tonemap's clear-color sentinel). The resolve
// carries alpha as a PREMULTIPLIED coverage channel through the same
// filter/clip/blend, so silhouette pixels converge to a fractional
// coverage and tonemap composites them over the clear color - the edge
// against the background is antialiased like any other edge, and a
// pure background pixel stays exactly (0,0,0,0) (v2 wrote alpha 1
// everywhere: the background turned black and silhouettes darkened).
//
// Jitter convention (glm RH: clip.w = -z_view, so adding jn to
// proj[2][0] moves content by -jn in NDC): the texel at pixel center p
// shows the UNJITTERED scene position p + j (uv units, j = jn / 2).
//
// Bindings (set 0):
//   0..2: Sampler2D currentHdr / history / gbufferA (bound by the
//         backend in this order - slot HDR, the ping-pong history,
//         the slot's GBuffer A view; all on the linear/clamp sampler)
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
    float4 texel;      // xy = 1/w, 1/h, z = exposure (HDR weight),
                       // w = sharpen strength (tonemap-side, unused here)
    float4 jitter;     // xy = THIS frame's jitter in UV units: the texel
                       // at uv shows the unjittered position uv + j
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

float4 xform4(float4 c[4], float4 v)
{
    return c[0] * v.x + c[1] * v.y + c[2] * v.z + c[3] * v.w;
}

float3 unproj(float2 ndc, float z, TaaParams tp)
{
    const float4 p = xform4(tp.invVP, float4(ndc, z, 1.0));
    return p.xyz / p.w;
}

// UE RGBToYCoCg / YCoCgToRGB (TemporalAA.usf:602) - Y carries the x4
// scale (Y = R + 2G + B), which is what HdrWeightY expects.
float3 rgbToYCoCg(float3 c)
{
    return float3(dot(c, float3(1.0, 2.0, 1.0)), dot(c, float3(2.0, 0.0, -2.0)),
                  dot(c, float3(-1.0, 2.0, -1.0)));
}
float3 yCoCgToRgb(float3 c)
{
    const float y = c.x * 0.25, co = c.y * 0.25, cg = c.z * 0.25;
    return float3(y + co - cg, y + cg, y - co - cg);
}
float hdrWeightY(float y, float exposure)
{
    return 1.0 / (y * exposure + 4.0);
}

// Clip `h` toward `target` along the segment until it enters the box
// (Karis 2014 / UE AA_CLIP). 4D: YCoCg + coverage.
float4 clipToBox(float4 h, float4 target, float4 bmin, float4 bmax)
{
    target = clamp(target, bmin, bmax);
    const float4 d = h - target;
    float t = 1.0;
    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        const float room = d[i] > 0.0 ? bmax[i] - target[i] : target[i] - bmin[i];
        const float ad = abs(d[i]);
        if (ad > 1e-7) t = min(t, room / ad);
    }
    return target + d * saturate(t);
}

// UE GetBicubic2DCatmullRomSamples (TextureSampling.ush:120): 5
// bilinear taps (the corner taps dropped - their weights are tiny).
float4 sampleHistoryCatmullRom(float2 uv, float2 size, float2 invSize)
{
    const float2 pixel = uv * size;
    const float2 iuv = floor(pixel - 0.5) + 1.0;
    const float2 f = pixel - iuv;
    const float2 f2 = f * f;
    const float2 f_2 = f + f;
    float2 suv = (1.25 - f2) * f + 0.5;
    const float2 t0 = (0.25 * f2 - 0.0625) * (1.125 - 0.5 * f2.yx);
    const float2 w0 = t0 - f_2 * t0;
    const float2 w3 = t0 + f_2 * t0;
    const float2 cuv = (iuv - 0.5) * invSize;
    suv *= invSize;
    float4 acc = float4(0.0);
    acc += history.SampleLevel(cuv + float2(-invSize.x, suv.y), 0.0) * w0.x;
    acc += history.SampleLevel(cuv + float2(2.0 * invSize.x, suv.y), 0.0) * w3.x;
    acc += history.SampleLevel(cuv + suv, 0.0) *
           (1.0 - w0.x - w3.x - w0.y - w3.y);
    acc += history.SampleLevel(cuv + float2(suv.x, -invSize.y), 0.0) * w0.y;
    acc += history.SampleLevel(cuv + float2(suv.x, 2.0 * invSize.y), 0.0) * w3.y;
    return acc;
}

[shader("fragment")]
float4 fsMain(float4 frag : SV_Position) : SV_Target
{
    const TaaParams tp = taaParams[0];
    const float2 invSize = tp.texel.xy;
    const float2 size = 1.0 / invSize;
    const float2 uv = frag.xy * invSize;  // this pixel's center
    const float2 jPx = tp.jitter.xy * size;  // jitter in pixels
    const float exposure = max(tp.texel.z, 1e-4);

    // ---- 1+2: filtered current + neighborhood statistics -------------
    float4 filtered = float4(0.0);
    float filteredW = 0.0;
    float4 bMin = float4(1e30), bMax = float4(-1e30);
    float4 m1 = float4(0.0), m2 = float4(0.0);
    float bestDist = 1e30;
    float2 bestOff = float2(0.0);
    float4 center = float4(0.0);
    [unroll]
    for (int dy = -1; dy <= 1; ++dy)
        [unroll]
        for (int dx = -1; dx <= 1; ++dx)
        {
            const float2 off = float2(float(dx), float(dy));
            const float2 suv = uv + off * invSize;  // texel center
            const float4 raw = currentHdr.SampleLevel(suv, 0.0);
            // Premultiplied coverage: background is (0,0,0,0) already;
            // lit pixels are alpha 1. Scrub non-finite input.
            float4 c = float4(rgbToYCoCg(max(raw.rgb, float3(0.0))),
                              saturate(raw.a));
            if (!(c.x >= 0.0 && c.x < 1e6)) c = float4(0.0);
            if (dx == 0 && dy == 0) center = c;
            // The texel shows the unjittered position (its center + j):
            // distance to the output center in pixels.
            const float2 d = off + jPx;
            const float w = exp(-2.29 * dot(d, d)) * hdrWeightY(c.x, exposure);
            filtered += c * w;
            filteredW += w;
            bMin = min(bMin, c);
            bMax = max(bMax, c);
            m1 += c;
            m2 += c * c;
            const float nd = gbufferA.SampleLevel(suv, 0.0).a;
            if (nd > 0.0 && nd < bestDist)
            {
                bestDist = nd;
                bestOff = off;
            }
        }
    filtered /= max(filteredW, 1e-8);

    const float blend = tp.camBlend.w;
    if (blend <= 0.0)
    {
        // No usable history (first frames / resize / re-enable): seed
        // with the filtered current.
        const float4 o = float4(yCoCgToRgb(filtered.xyz), filtered.w);
        return float4(max(o.rgb, float3(0.0)), o.w > 0.995 ? 1.0 : o.w);
    }

    // ---- 3: velocity of the closest surface in the 3x3 ---------------
    // A texel's world point: the unjittered ray through (its center +
    // j) at the GBuffer's euclidean distance; background (no surface in
    // the whole 3x3) reprojects the far-plane point = rotation-only
    // parallax of a sky at infinity.
    float2 curUv;
    float3 world;
    if (bestDist < 1e29)
    {
        curUv = uv + bestOff * invSize + tp.jitter.xy;
        const float2 ndc = curUv * 2.0 - 1.0;
        const float3 dir = normalize(unproj(ndc, 1.0, tp) - unproj(ndc, 0.0, tp));
        world = tp.camBlend.xyz + dir * bestDist;
    }
    else
    {
        curUv = uv;
        world = unproj(uv * 2.0 - 1.0, 1.0, tp);
    }
    const float4 pc = xform4(tp.prevVP, float4(world, 1.0));
    float4 outC = filtered;
    if (pc.w > 0.0)
    {
        const float2 velocity = (pc.xy / pc.w * 0.5 + 0.5) - curUv;
        const float2 puv = uv + velocity;
        if (all(puv >= float2(0.0)) && all(puv <= float2(1.0)))
        {
            // ---- 4: history (Catmull-Rom), into the same space --------
            const float4 hRaw = sampleHistoryCatmullRom(puv, size, invSize);
            float4 h = float4(rgbToYCoCg(max(hRaw.rgb, float3(0.0))),
                              saturate(hRaw.a));
            if (!(h.x >= 0.0 && h.x < 1e6)) h = filtered;

            // ---- 5: variance-tightened box, clip, luma-weighted blend --
            const float4 mean = m1 / 9.0;
            const float4 sigma = sqrt(max(m2 / 9.0 - mean * mean, float4(0.0)));
            float4 lo = max(bMin, mean - sigma * 1.25);
            float4 hi = min(bMax, mean + sigma * 1.25);
            if (any(lo > hi))
            {
                lo = bMin;
                hi = bMax;
            }
            // Motion-scaled clip strength. At rest the reprojection is
            // exact and the history is geometrically valid - only
            // SHADING changes can invalidate it - while sub-pixel
            // features (1px rims, glints) are sampled by the
            // neighborhood in only SOME jitter phases: a full clip
            // throws the accumulated average away every time the
            // feature-less phase comes around (measured: a 1px rim
            // flipping 25 <-> 141 per phase). A 0.2 partial clip at rest
            // still decays stale shading at (0.8 * 0.95)^n (~1% after
            // 17 frames); any motion >= 0.5px restores the full clip.
            const float velPx = length(velocity * size);
            h = lerp(h, clipToBox(h, filtered, lo, hi),
                     lerp(0.2, 1.0, saturate(velPx * 2.0)));
            float curW = lerp(1.0 - blend, 0.2, saturate(velPx / 40.0));
            // (UE's minimum-contribution floor - take the current when it
            // is within 1% of the history - is deliberately NOT ported:
            // measured, it made 145k pixels follow the per-phase jitter
            // noise at +-1 level vs 40k without.)
            curW = min(curW, 1.0);
            const float wF = curW * hdrWeightY(filtered.x, exposure);
            const float wH = (1.0 - curW) * hdrWeightY(h.x, exposure);
            outC = (filtered * wF + h * wH) / max(wF + wH, 1e-8);
        }
    }
    float4 o = float4(max(yCoCgToRgb(outC.xyz), float3(0.0)), saturate(outC.w));
    // Close-to-opaque coverage snaps to opaque (UE's 0.995 rule) and
    // a vanishing coverage to exact background (keeps pure-background
    // pixels on the tonemap's clear-color path).
    if (o.w > 0.995) o.w = 1.0;
    if (o.w < 1e-3) o = float4(0.0);
    // NOTE: NO sharpen here - the resolve output IS next frame's
    // history; baking an unsharp into it re-applies the sharpen every
    // frame (exponential ringing, the v1.6 "completely broken image").
    // The sharpen lives in the tonemap's sample of this resolve.
    return o;
}
