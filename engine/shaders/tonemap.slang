// M8.3: the tonemap/present pass - the HDR chain's tail. lighting.slang
// writes LINEAR HDR into the per-slot RGBA16F target; this fullscreen
// pass samples it, applies exposure + the selected operator (extended
// Reinhard or the Narkowicz ACES fit) and writes the final LDR into the
// frame's color target (slot.offscreen - the imgui viewport and the
// screenshots only ever read THAT, the compositional red line).
//
// Background contract: lighting writes ALPHA 0 for background pixels
// (GBuffer alpha==0 and the scrubbed garbage paths); every lit pixel
// carries alpha 1 by construction - including rgb-zero ones (ambient=0,
// unlit backfaces, metallic black materials) and values the f16 store
// rounds to +0. Background passes the frame's clear color through
// UNTONEMAPPED - the regression net's "expected color" invariant and
// the byte-exact background survive the HDR switch unchanged.
//
// Bindings (set 0):
//   0: Sampler2D hdrTex - this frame slot's RGBA16F HDR target (the
//      backend binds it to the first reflected sampler binding)
//
// Push (32B):
//   float4 clearColorExp - rgb the frame's clear color, a exposure
//   float4 modeViewport  - x operator (0 Reinhard, 1 ACES), yz = 1/w,
//                          1/h, w scratch

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

[shader("fragment")]
float4 fsMain(float4 frag : SV_Position) : SV_Target
{
    const float2 uv = frag.xy * push.modeViewport.yz;
    const float4 hdr4 = hdrTex.SampleLevel(uv, 0.0);
    const float3 hdr = hdr4.rgb;
    // Background sentinel: ALPHA 0 passes the clear color through
    // (see the header - an rgb test would misclassify legitimately
    // black lit pixels).
    if (hdr4.a <= 0.0)
        return float4(push.clearColorExp.rgb, 1.0);
    const float3 mapped = hdr * push.clearColorExp.a;
    float3 outColor;
    if (push.modeViewport.x < 0.5)
    {
        // Extended Reinhard (per-channel; good-enough LDR target).
        outColor = mapped / (mapped + float3(1.0, 1.0, 1.0));
    }
    else
    {
        outColor = acesFitted(mapped);
    }
    if (!(outColor.x >= 0.0) || !(outColor.y >= 0.0) ||
        !(outColor.z >= 0.0))
        return float4(0.5, 0.0, 0.5, 1.0);  // magenta sentinel
    return float4(outColor, 1.0);
}
