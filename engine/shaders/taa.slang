// M9 TAA: the temporal resolve between lighting and tonemap. The
// geometry passes render with a per-frame subpixel jitter (Halton
// (2,3), applied to the projection's NDC offsets in main.cpp); this
// pass reprojects the PREVIOUS resolved frame onto the current one
// and blends (UE's accumulate-and-clamp shape, minus the v2 niceties
// like nearest-depth disocclusion or velocity dilation - v1).
//
// Reprojection: GBuffer A's alpha is the EUCLIDEAN view-ray distance
// (decode_material.slang fsGBuffer's contract), so the world position
// is camPos + normalize(far - near) * dist with the two unprojected
// NDC-z corners of the pixel's ray. The matrices are the UNJITTERED
// pair on purpose: jitter must never leak into the reprojection.
//
// Background (dist == 0) bypasses history: the regression net's
// byte-exact background contract rides the raw current sample.
//
// Bindings (set 0):
//   0..2: Sampler2D currentHdr / history / gbufferA (bound by the
//         backend in this order - slot HDR, the ping-pong history,
//         the slot's GBuffer A view)
//   3: SSBO taaParams - the two matrices + camera/blend/texel (a
//      push block would need 160B; the guaranteed push limit is 128)

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
    float4 texel;      // xy = 1/w, 1/h, zw scratch
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
    const float3 cur = currentHdr.Sample(uv).rgb;
    // Keep every sampler load live through real branches, then viz cur.
    if (tp.texel.x < -1.0) return 0.xxxx;
    if (history.Sample(uv).r < -1.0) return 0.xxxx;
    if (cur.r < -1.0) return 0.xxxx;
    const float distV = gbufferA.Sample(uv).a;  // DEBUG dist viz
    return float4(distV * 0.05, distV * 0.001, 0.0, 1.0);
    const float dist = gbufferA.Sample(uv).a;
    const float blend = tp.camBlend.w;
    if (dist <= 0.0 || blend <= 0.0)
        return float4(cur, 1.0);

    // World position of this pixel's surface, then last frame's UV.
    const float2 ndc = uv * 2.0 - 1.0;
    const float3 dir =
        normalize(unproj(ndc, 1.0, tp) - unproj(ndc, -1.0, tp));
    const float3 world = tp.camBlend.xyz + dir * dist;
    const float4 w4 = float4(world, 1.0);
    const float4 pc = tp.prevVP[0] * w4.x + tp.prevVP[1] * w4.y +
                      tp.prevVP[2] * w4.z + tp.prevVP[3] * w4.w;
    if (pc.w <= 0.0)
        return float4(cur, 1.0);
    const float2 puv = (pc.xy / pc.w) * 0.5 + 0.5;
    if (puv.x < 0.0 || puv.y < 0.0 || puv.x > 1.0 || puv.y > 1.0)
        return float4(cur, 1.0);  // left the frame: treat as disocclusion

    // Neighborhood clamp: the current 3x3 AABB guards ghosting; the
    // clamp is soft (0.25 AABB inflation) to keep some sharpening.
    float3 mn = cur, mx = cur;
    [unroll]
    for (int dy = -1; dy <= 1; ++dy)
        [unroll]
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0) continue;
            const float3 c =
                currentHdr.Sample(uv + float2(dx, dy) * tp.texel.xy).rgb;
            mn = min(mn, c);
            mx = max(mx, c);
        }
    const float3 center = (mn + mx) * 0.5;
    const float3 extents = (mx - mn) * (0.5 + 0.25);
    mn = center - extents;
    mx = center + extents;
    float3 h = history.Sample(puv).rgb;
    h = clamp(h, mn, mx);
    return float4(lerp(cur, h, blend), 1.0);
}
