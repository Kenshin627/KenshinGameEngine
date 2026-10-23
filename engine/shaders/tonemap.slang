// M8.3: the tonemap/present pass - the HDR chain's tail. lighting.slang
// writes LINEAR HDR into the per-slot RGBA16F target; this fullscreen
// pass samples it, applies exposure + the selected operator and writes
// the final LDR into the frame's color target (slot.offscreen - the
// imgui viewport and the screenshots only ever read THAT, the
// compositional red line).
//
// Background contract: lighting writes ALPHA 0 for background pixels
// (GBuffer alpha==0 and the scrubbed garbage paths); every lit pixel
// carries alpha 1 by construction - including rgb-zero ones (ambient=0,
// unlit backfaces, metallic black materials) and values the f16 store
// rounds to +0. Background passes the frame's clear color through
// UNTONEMAPPED - the regression net's "expected color" invariant and
// the byte-exact background survive the HDR switch unchanged.
//
// Operators (2026-09-25 UE cross-check, C:\code\ue5 @ 5.8.3):
//   0 Reinhard (extended, per-channel)
//   1 ACES (Narkowicz's cheap fit - NOT the full RRT+ODT; label says so)
//   2 Filmic (UE's FilmToneMap from TonemapCommon.ush, ported with the
//     Scene.cpp defaults Slope 0.88 / Toe 0.55 / Shoulder 0.26 /
//     BlackClip 0 / WhiteClip 0.04 - glow module, red modifier, log10
//     toe/straight/shoulder with the 0.18 matching, pre/post
//     desaturation). Fed like UE's CombineLUTs: linear sRGB -> AP1
//     (sRGB_2_AP1), FilmToneMap in AP1, back with AP1_2_sRGB (the v1
//     port skipped the gamut transforms and pushed sRGB values through
//     an AP1 curve - shifted hues/saturation).
//
// Coverage (M9 TAA v3): the TAA resolve carries a PREMULTIPLIED
// coverage alpha (antialiased silhouettes against the background);
// the tonemap un-premultiplies, maps the surface color and composites
// it over the clear color in display space. Non-TAA frames only ever
// hold alpha 0/1, so their output is unchanged.
//
// Bindings (set 0):
//   0: Sampler2D hdrTex - this frame slot's RGBA16F HDR target (the
//      backend binds it to the first reflected sampler binding)
//
// Push (48B):
//   float4 clearColorExp - rgb the frame's clear color, a exposure
//   float4 modeViewport  - x operator, yz = 1/w, 1/h, w = TAA sharpen
//   float4 bloomParams   - x strength (0 = off), y 1 / level count,
//                          z auto exposure (1 = exposureState[0].x
//                          replaces clearColorExp.a)

struct TonemapPush
{
    float4 clearColorExp;
    float4 modeViewport;
    float4 bloomParams;  // x bloom strength (0 = off), y 1 / level count,
                         // z auto exposure flag
    float4 skyParams;    // x env mip for the background (< 0 = clear
                         // color), y env intensity
    float4x4 invViewProj;  // unjittered; uv -> NDC directly (no y flip)
};

[[vk::push_constant]] ConstantBuffer<TonemapPush> push;

[[vk::binding(0)]] Sampler2D hdrTex;
// The bloom chain's top level (half resolution, bilinear upsampled);
// the backend binds the HDR view here when no bloom ran this frame.
[[vk::binding(1)]] Sampler2D bloomTex;
// exposure.slang's state ([0].x = this frame's adapted image scale);
// TonemapPass binds a 1-entry dummy when auto exposure is off.
[[vk::binding(2)]] StructuredBuffer<float4> exposureState;
// The IBL equirect (Z-up, lighting.slang's convention): drawn behind the
// model when skyParams.x >= 0; the backend's dummy array otherwise.
[[vk::binding(3)]] Sampler2DArray envMap;

float4 xform(float4x4 m, float4 v)
{
    return m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3] * v.w;
}

// Environment radiance behind pixel `uv` (lighting.slang's ray
// reconstruction + equirect lookup), linear HDR.
float3 skyRadiance(float2 uv)
{
    const float2 ndc = uv * 2.0 - 1.0;
    const float4 n4 = xform(push.invViewProj, float4(ndc, 0.0, 1.0));
    const float4 f4 = xform(push.invViewProj, float4(ndc, 1.0, 1.0));
    const float3 d = normalize(f4.xyz / f4.w - n4.xyz / n4.w);
    const float2 equi = float2(atan2(d.y, d.x) / 6.28318530718 + 0.5,
                               acos(clamp(d.z, -1.0, 1.0)) / 3.14159265359);
    const float3 rgb = envMap.SampleLevel(float3(equi, 0.0), push.skyParams.x).rgb;
    return max(rgb, float3(0.0)) * push.skyParams.y;
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

// Narkowicz ACES fitted curve (the standard cheap approximation;
// input expected in a ~[0,1]-ish exposure-mapped range).
float3 acesFitted(float3 x)
{
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e),
                 float3(0.0), float3(1.0));
}

// ---- UE FilmToneMap (TonemapCommon.ush + ACESCommon.ush, 5.8.3) ----
// Constant-permutation port: the uniforms ride compile-time defaults
// (Scene.cpp: Slope .88, Toe .55, Shoulder .26, BlackClip 0, WhiteClip
// .04 - UE's default "Filmic" tonemapping method).

static const float3x3 kAP0_2_AP1 = {
     1.4514393161, -0.2365107469, -0.2149285693,
    -0.0765537734,  1.1762296998, -0.0996759264,
     0.0083161484, -0.0060324498,  0.9977163014};
static const float3x3 kAP1_2_AP0 = {
     0.6954522414, 0.1406786965, 0.1638690622,
     0.0447945634, 0.8596711185, 0.0955343182,
    -0.0055258826, 0.0040252103, 1.0015006723};
static const float3 kAP1_RGB2Y = {0.2722287168, 0.6740817658,
                                  0.0536895174};
// UE TonemapCommon.ush: sRGB_2_AP1 = XYZ_2_AP1 * D65_2_D60 * sRGB_2_XYZ
// and its inverse, evaluated from ACESCommon.ush's constants.
static const float3x3 kSRGB_2_AP1 = {
     0.6130974024, 0.3395231461, 0.0473794514,
     0.0701937225, 0.9163538791, 0.0134523986,
     0.0206155929, 0.1095697729, 0.8698146341};
static const float3x3 kAP1_2_SRGB = {
     1.7050509926, -0.6217921205, -0.0832588722,
    -0.1302564175,  1.1408047365, -0.0105483190,
    -0.0240033568, -0.1289689761,  1.1529723328};

float ueRgb2Saturation(float3 rgb)
{
    const float minrgb = min(min(rgb.r, rgb.g), rgb.b);
    const float maxrgb = max(max(rgb.r, rgb.g), rgb.b);
    return (max(maxrgb, 1e-10) - max(minrgb, 1e-10)) / max(maxrgb, 1e-2);
}

float ueRgb2Hue(float3 rgb)
{
    float hue = 0.0;
    if (!(rgb[0] == rgb[1] && rgb[1] == rgb[2]))
        hue = (180.0 / 3.14159265) *
              atan2(sqrt(3.0) * (rgb[1] - rgb[2]),
                    2.0 * rgb[0] - rgb[1] - rgb[2]);
    if (hue < 0.0) hue = hue + 360.0;
    return clamp(hue, 0.0, 360.0);
}

float ueCenterHue(float hue, float centerH)
{
    float hueCentered = hue - centerH;
    if (hueCentered < -180.0) hueCentered += 360.0;
    else if (hueCentered > 180.0) hueCentered -= 360.0;
    return hueCentered;
}

float ueGlowFwd(float ycIn, float glowGainIn, float glowMid)
{
    if (ycIn <= 2.0 / 3.0 * glowMid) return glowGainIn;
    if (ycIn >= 2.0 * glowMid) return 0.0;
    return glowGainIn * (glowMid / ycIn - 0.5);
}

float ueRgb2Yc(float3 rgb)
{
    const float chroma =
        sqrt(rgb.b * (rgb.b - rgb.g) + rgb.g * (rgb.g - rgb.r) +
             rgb.r * (rgb.r - rgb.b));
    return (rgb.b + rgb.g + rgb.r + 1.75 * chroma) / 3.0;
}

float ueSigmoidShaper(float x)
{
    const float t = max(1.0 - abs(0.5 * x), 0.0);
    const float y = 1.0 + (x < 0.0 ? -1.0 : 1.0) * (1.0 - t * t);
    return 0.5 * y;
}

float3 filmToneMap(float3 linearColor)
{
    const float filmSlope = 0.88, filmToe = 0.55, filmShoulder = 0.26;
    const float filmBlackClip = 0.0, filmWhiteClip = 0.04;

    float3 colorAP0 = mul(kAP1_2_AP0, linearColor);

    // "Glow" module constants (RRT).
    const float kGlowGain = 0.05;
    const float kGlowMid = 0.08;
    const float saturation = ueRgb2Saturation(colorAP0);
    const float ycIn = ueRgb2Yc(colorAP0);
    const float s = ueSigmoidShaper((saturation - 0.4) / 0.2);
    colorAP0 *= 1.0 + ueGlowFwd(ycIn, kGlowGain * s, kGlowMid);

    // Red modifier.
    const float kRedScale = 0.82, kRedPivot = 0.03, kRedWidth = 135.0;
    const float hue = ueRgb2Hue(colorAP0);
    const float centeredHue = ueCenterHue(hue, 0.0);
    const float ax = saturate(abs(2.0 * centeredHue / kRedWidth));
    const float hueWeight = ax * ax * (3.0 - 2.0 * ax);  // smoothstep^2
    colorAP0.r += hueWeight * saturation * (kRedPivot - colorAP0.r) *
                  (1.0 - kRedScale);

    // ACEScg primaries as working space.
    float3 working = mul(kAP0_2_AP1, colorAP0);
    working = max(working, 0.0);
    working = lerp(dot(working, kAP1_RGB2Y), working, 0.96);  // pre-desat

    const float toeScale = 1.0 + filmBlackClip - filmToe;
    const float shoulderScale = 1.0 + filmWhiteClip - filmShoulder;
    const float inMatch = 0.18;
    const float outMatch = 0.18;

    float toeMatch;
    if (filmToe > 0.8)
    {
        toeMatch = (1.0 - filmToe - outMatch) / filmSlope +
                   log10(inMatch);
    }
    else
    {
        const float bt = (outMatch + filmBlackClip) / toeScale - 1.0;
        toeMatch = log10(inMatch) -
                   0.5 * log((1.0 + bt) / (1.0 - bt)) * (toeScale / filmSlope);
    }
    const float straightMatch = (1.0 - filmToe) / filmSlope - toeMatch;
    const float shoulderMatch = filmShoulder / filmSlope - straightMatch;

    const float3 logColor = log10(max(working, 1e-10));
    const float3 straightColor = filmSlope * (logColor + straightMatch);
    const float3 toeColor =
        -filmBlackClip +
        2.0 * toeScale /
            (1.0 + exp((-2.0 * filmSlope / toeScale) *
                       (logColor - toeMatch)));
    const float3 shoulderColor =
        1.0 + filmWhiteClip -
        2.0 * shoulderScale /
            (1.0 + exp((2.0 * filmSlope / shoulderScale) *
                       (logColor - shoulderMatch)));
    const float3 toeSel = logColor < toeMatch ? toeColor : straightColor;
    const float3 shSel = logColor > shoulderMatch ? shoulderColor
                                                  : straightColor;

    float3 t = saturate((logColor - toeMatch) /
                        (shoulderMatch - toeMatch));
    t = shoulderMatch < toeMatch ? 1.0 - t : t;
    t = (3.0 - 2.0 * t) * t * t;
    float3 toneColor = lerp(toeSel, shSel, t);

    toneColor = lerp(dot(toneColor, kAP1_RGB2Y), toneColor, 0.93);
    return max(toneColor, 0.0);
}

// The display OETF: the lighting chain is linear; the 8-bit target
// is interpreted as sRGB. Without the encode every tonemapped value
// lands in the crushed toe (mid-dark, flat contrast) - photographic
// textures lose their pattern.
float3 linearToSrgb(float3 c)
{
    const float3 lo = c * 12.92;
    const float3 hi = 1.055 * pow(max(c, float3(1e-10)), float3(1.0 / 2.4))
                      - 0.055;
    return float3(c.x <= 0.0031308 ? lo.x : hi.x,
                  c.y <= 0.0031308 ? lo.y : hi.y,
                  c.z <= 0.0031308 ? lo.z : hi.z);
}

float3 srgbToLinearExact(float3 c)
{
    const float3 lo = c / 12.92;
    const float3 hi = pow(max((c + 0.055) / 1.055, float3(0.0)), float3(2.4));
    return float3(c.x <= 0.04045 ? lo.x : hi.x, c.y <= 0.04045 ? lo.y : hi.y,
                  c.z <= 0.04045 ? lo.z : hi.z);
}

// The selected operator on an exposure-scaled linear color.
float3 toneMapOp(float3 mapped)
{
    if (push.modeViewport.x < 0.5)
        return mapped / (mapped + float3(1.0, 1.0, 1.0));  // ext. Reinhard
    if (push.modeViewport.x < 1.5)
        return acesFitted(mapped);
    return max(mul(kAP1_2_SRGB, filmToneMap(mul(kSRGB_2_AP1, mapped))),
               float3(0.0));
}

static const float2 kRing[8] = {
    float2(-1.0, 0.0), float2(1.0, 0.0),  float2(0.0, -1.0),
    float2(0.0, 1.0),  float2(-1.0, -1.0), float2(1.0, -1.0),
    float2(-1.0, 1.0), float2(1.0, 1.0)};

// Sharpen domain: rgb / (1 + max channel); alpha (coverage, already
// in [0, 1]) passes through. Monotonic, so the neighborhood clamp in
// compressed space bounds the decompressed result too.
float4 compressHdr(float4 v)
{
    const float3 rgb = max(v.rgb, float3(0.0));
    return float4(rgb / (1.0 + max(rgb.x, max(rgb.y, rgb.z))), v.a);
}
float4 decompressHdr(float4 v)
{
    const float3 rgb = max(v.rgb, float3(0.0));
    const float m = min(max(rgb.x, max(rgb.y, rgb.z)), 0.999);
    return float4(rgb / (1.0 - m), v.a);
}

[shader("fragment")]
float4 fsMain(float4 frag : SV_Position) : SV_Target
{
    const float2 uv = frag.xy * push.modeViewport.yz;
    float4 hdr4 = hdrTex.SampleLevel(uv, 0.0);
    // Bloom (bloom.slang's chain; bloomParams.x = strength s, .y = 1 /
    // level count - the chain SUMS its levels). Energy-conserving on
    // surfaces: scene * (1 - s) + bloom * s.
    const float s = push.bloomParams.x;
    float exposure = push.clearColorExp.a;
    if (push.bloomParams.z > 0.5)
    {
        const float autoScale = exposureState[0].x;
        if (autoScale > 0.0 && autoScale < 1e6) exposure = autoScale;
    }
    float3 bloom = float3(0.0);
    if (s > 0.0)
        bloom = max(bloomTex.SampleLevel(uv, 0.0).rgb, float3(0.0)) *
                push.bloomParams.y;
    // Background display color. Sky on: the environment behind the
    // model, exposed and tonemapped like the scene (bloom spilling past
    // silhouettes blends in the same energy-conserving way). Sky off:
    // the clear color plus the bloom halo (added in display-linear
    // space) - exact clear color wherever the halo is below half a code
    // value.
    float3 background = push.clearColorExp.rgb;
    if (push.skyParams.x >= 0.0)
    {
        float3 sky = skyRadiance(uv);
        if (s > 0.0) sky = sky * (1.0 - s) + bloom * s;
        const float3 skyMapped = toneMapOp(sky * exposure);
        if (skyMapped.x >= 0.0 && skyMapped.y >= 0.0 && skyMapped.z >= 0.0)
            background = linearToSrgb(saturate(skyMapped));
    }
    else if (s > 0.0)
    {
        const float3 halo = toneMapOp(bloom * s * exposure);
        if (max(halo.x, max(halo.y, halo.z)) > 0.5 / 255.0)
            background = linearToSrgb(
                saturate(srgbToLinearExact(push.clearColorExp.rgb) + halo));
    }
    // Background sentinel: ALPHA 0 passes the clear color through
    // (see the header - an rgb test would misclassify legitimately
    // black lit pixels).
    if (hdr4.a <= 0.0)
        return float4(background, 1.0);
    // TAA sharpen (modeViewport.w): the temporal accumulation costs
    // roughly half a pixel of softness; a small unsharp over the
    // RESOLVED frame claws it back. Applied HERE - never inside the
    // resolve, whose output is the next frame's history (a baked-in
    // unsharp re-applies itself every frame = exponential ringing).
    // Premultiplied: rgb AND coverage sharpen together, so an edge
    // against the background does not get a bright rim from the
    // zero-rgb background neighbors.
    if (push.modeViewport.w > 0.0)
    {
        // In a tone-compressed domain (x / (1 + max(x)), Karis's TAA
        // weighting) and clamped to the 3x3 neighborhood's range (CAS):
        // in linear HDR one specular neighbor at 100+ dragged the
        // unsharp of a merely bright pixel below zero - isolated BLACK
        // specks along hot highlights.
        const float2 t = push.modeViewport.yz;
        const float4 c = compressHdr(hdr4);
        float4 nb = float4(0.0);
        float4 lo = c, hi = c;
        [unroll] for (int k = 0; k < 8; ++k)
        {
            const float4 s =
                compressHdr(hdrTex.SampleLevel(uv + kRing[k] * t, 0.0));
            nb += s;
            lo = min(lo, s);
            hi = max(hi, s);
        }
        const float4 sharp =
            clamp(c + (c - nb * (1.0 / 8.0)) * push.modeViewport.w, lo, hi);
        hdr4 = decompressHdr(sharp);
        hdr4.a = saturate(hdr4.a);
        if (hdr4.a <= 0.0)
            return float4(background, 1.0);
    }
    const float coverage = hdr4.a;
    float3 hdr = hdr4.rgb / coverage;  // un-premultiply
    if (s > 0.0) hdr = hdr * (1.0 - s) + bloom * s;
    const float3 outColor = toneMapOp(hdr * exposure);
    if (!(outColor.x >= 0.0) || !(outColor.y >= 0.0) ||
        !(outColor.z >= 0.0))
        return float4(0.5, 0.0, 0.5, 1.0);  // magenta sentinel
    const float3 display = linearToSrgb(saturate(outColor));
    if (coverage >= 1.0)
        return float4(display, 1.0);
    // Fractional coverage (TAA silhouettes): composite over the
    // background (clear color + halo) in display space.
    return float4(lerp(background, display, coverage), 1.0);
}
