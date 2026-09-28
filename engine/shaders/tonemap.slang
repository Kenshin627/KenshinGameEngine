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
//     desaturation). One divergence: our scene color is linear-sRGB,
//     UE's is AP1 working space; the curve is fed our linear values
//     directly (gamut transform skipped, v1).
//
// Bindings (set 0):
//   0: Sampler2D hdrTex - this frame slot's RGBA16F HDR target (the
//      backend binds it to the first reflected sampler binding)
//
// Push (32B):
//   float4 clearColorExp - rgb the frame's clear color, a exposure
//   float4 modeViewport  - x operator, yz = 1/w, 1/h, w = TAA sharpen

struct TonemapPush
{
    float4 clearColorExp;
    float4 modeViewport;
};

[[vk::push_constant]] ConstantBuffer<TonemapPush> push;

[[vk::binding(0)]] Sampler2D hdrTex;

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

[shader("fragment")]
float4 fsMain(float4 frag : SV_Position) : SV_Target
{
    const float2 uv = frag.xy * push.modeViewport.yz;
    const float4 hdr4 = hdrTex.SampleLevel(uv, 0.0);
    // Background sentinel: ALPHA 0 passes the clear color through
    // (see the header - an rgb test would misclassify legitimately
    // black lit pixels).
    if (hdr4.a <= 0.0)
        return float4(push.clearColorExp.rgb, 1.0);
    // TAA sharpen (modeViewport.w): the temporal accumulation costs
    // roughly half a pixel of softness; a small unsharp over the
    // RESOLVED frame claws it back. Applied HERE - never inside the
    // resolve, whose output is the next frame's history (a baked-in
    // unsharp re-applies itself every frame = exponential ringing).
    float3 hdr = hdr4.rgb;
    if (push.modeViewport.w > 0.0)
    {
        const float2 t = push.modeViewport.yz;
        float3 nb = float3(0.0);
        nb += hdrTex.SampleLevel(uv + float2(-t.x, 0.0), 0.0).rgb;
        nb += hdrTex.SampleLevel(uv + float2(t.x, 0.0), 0.0).rgb;
        nb += hdrTex.SampleLevel(uv + float2(0.0, -t.y), 0.0).rgb;
        nb += hdrTex.SampleLevel(uv + float2(0.0, t.y), 0.0).rgb;
        nb += hdrTex.SampleLevel(uv + float2(-t.x, -t.y), 0.0).rgb;
        nb += hdrTex.SampleLevel(uv + float2(t.x, -t.y), 0.0).rgb;
        nb += hdrTex.SampleLevel(uv + float2(-t.x, t.y), 0.0).rgb;
        nb += hdrTex.SampleLevel(uv + float2(t.x, t.y), 0.0).rgb;
        hdr = max(hdr + (hdr - nb * (1.0 / 8.0)) * push.modeViewport.w,
                  float3(0.0));
    }
    const float3 mapped = hdr * push.clearColorExp.a;
    float3 outColor;
    if (push.modeViewport.x < 0.5)
    {
        // Extended Reinhard (per-channel; good-enough LDR target).
        outColor = mapped / (mapped + float3(1.0, 1.0, 1.0));
    }
    else if (push.modeViewport.x < 1.5)
    {
        outColor = acesFitted(mapped);
    }
    else
    {
        outColor = filmToneMap(mapped);
    }
    if (!(outColor.x >= 0.0) || !(outColor.y >= 0.0) ||
        !(outColor.z >= 0.0))
        return float4(0.5, 0.0, 0.5, 1.0);  // magenta sentinel
    return float4(outColor, 1.0);
}
