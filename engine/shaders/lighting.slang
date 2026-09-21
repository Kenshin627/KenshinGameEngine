// M8.1: deferred lighting pass - the GBuffer consumer (M8.2: the
// material table feeds per-pixel roughness/metallic via GBuffer B's
// alpha-carried material id).
// Fullscreen triangle; samples the decode pass's GBuffer (A = world
// normal + euclidean camera distance, B = base color), reconstructs the
// world position exactly (camPos + normalize(invViewProj ray) * dist -
// a euclidean distance with a normalized ray roundtrips the position),
// and accumulates minimal GGX (Cook-Torrance, constant roughness 0.5 /
// metallic 0 until M8.2's material table) over the light SSBO.
// Background (GBuffer A alpha == 0) passes the frame's clear color
// through, keeping the baseline-path background parity.
//
// Bindings (set 0):
//   0: Sampler2D gbufferA - RGBA16F (binding order matters: the
//      backend binds this frame slot's GBuffer views to the reflected
//      sampler bindings in ascending order)
//   1: Sampler2D gbufferB - RGBA8
//   2: StructuredBuffer<Lights> - 2 vec4s per light:
//      [i].a = xyz direction (directional, TO light) or position
//              (point), w = type (0 = directional, 1 = point)
//      [i].b = rgb color * intensity, w = range (point lights;
//              0 = unlimited)
//
// Push (packed identically in C++ - 112B of the 128B budget):
//   float4x4 invViewProj (64B) | camPos + lightCount (16B) |
//   clearColor rgba (16B) | 1/w, 1/h + ambient + exposure (16B)

struct LightPush
{
    float4x4 invViewProj;
    float4 camPosCount;   // xyz camPos, w light count
    float4 clearColor;    // rgb the frame's clear color (background),
                          // a the MATERIAL COUNT (M8.2 id clamp bound -
                          // the push block rides existing words; the
                          // alpha of a clear color is meaningless)
    float4 viewportAmb;   // xy = 1/w, 1/h, z ambient, w exposure
};

// The material table record (packed identically to decode_material's
// GpuMaterial and the scene's GPU table).
struct GpuMaterial
{
    float4 kdTex;    // rgb kd, a = texture-array layer (-1 = untextured)
    float4 params;   // x roughness, y metallic, zw pad
};

struct GpuLight
{
    float4 a;
    float4 b;
};

[[vk::push_constant]] ConstantBuffer<LightPush> push;

[[vk::binding(0)]] Sampler2D gbufferA;
[[vk::binding(1)]] Sampler2D gbufferB;
[[vk::binding(2)]] StructuredBuffer<GpuLight> lights;
[[vk::binding(3)]] StructuredBuffer<GpuMaterial> materials;

// NOTE on matrices: glm stores column-major and Slang's mul reads
// buffer vec4s as rows - every transform below is an explicit linear
// combination of the stored columns (the project-wide convention).
float4 xform(float4x4 m, float4 v)
{
    return m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3] * v.w;
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

// GGX terms (minimal PBR; roughness/metallic come from the material
// table via GBuffer B's alpha channel - the decode pass wrote the
// per-cluster id there).

float distributionGGX(float ndh, float rough)
{
    const float a = rough * rough;
    const float a2 = a * a;
    const float d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / max(3.14159265 * d * d, 1e-7);
}
float geometrySmith(float ndv, float ndl, float rough)
{
    const float k = (rough + 1.0) * (rough + 1.0) / 8.0;
    const float gv = ndv / (ndv * (1.0 - k) + k);
    const float gl = ndl / (ndl * (1.0 - k) + k);
    return gv * gl;
}
float3 fresnelSchlick(float ct, float3 f0)
{
    return f0 + (float3(1.0, 1.0, 1.0) - f0) * pow(1.0 - ct, 5.0);
}

[shader("fragment")]
float4 fsMain(float4 frag : SV_Position) : SV_Target
{
    const float2 uv = frag.xy * push.viewportAmb.xy;
    const float4 a = gbufferA.SampleLevel(uv, 0.0);
    if (a.w <= 0.0)
        return float4(push.clearColor.rgb, 1.0);

    const float4 bTexel = gbufferB.SampleLevel(uv, 0.0);
    const float3 albedo = bTexel.rgb;
    const uint matCount = max(uint(push.clearColor.w + 0.5), 1u);
    const uint matId =
        min(uint(round(bTexel.a * 255.0)) & 0xFFu, matCount - 1u);
    const float2 rm = clamp(materials[matId].params.xy,
                            float2(0.04, 0.0), float2(1.0, 1.0));
    const float3 nrm = normalize(a.xyz);
    const float3 camPos = push.camPosCount.xyz;

    // Exact world position: ray through this pixel's NDC, normalized,
    // scaled by the euclidean distance the decode pass stored. The
    // flipped-proj convention maps uv directly to NDC (no y flip) -
    // the same convention worldToScreen uses in reverse.
    const float2 ndc = uv * 2.0 - 1.0;
    const float4 nearP4 = xform(push.invViewProj, float4(ndc, 0.0, 1.0));
    const float4 farP4 = xform(push.invViewProj, float4(ndc, 1.0, 1.0));
    const float3 nearP = nearP4.xyz / nearP4.w;
    const float3 farP = farP4.xyz / farP4.w;
    const float3 ray = normalize(farP - nearP);
    const float3 worldPos = camPos + ray * a.w;

    const float3 viewDir = normalize(camPos - worldPos);
    const float ndv = max(dot(nrm, viewDir), 0.0);
    const float3 f0 = lerp(float3(0.04, 0.04, 0.04), albedo, rm.y);

    float3 lo = float3(0.0, 0.0, 0.0);
    const uint lightCount = uint(push.camPosCount.w + 0.5);
    for (uint i = 0; i < lightCount; ++i)
    {
        const float4 meta = lights[i].a;
        const float4 radiance = lights[i].b;
        float3 l;
        float3 li = radiance.rgb;
        if (meta.w < 0.5)
        {
            l = normalize(meta.xyz);  // directional: TO light
        }
        else
        {
            const float3 toLight = meta.xyz - worldPos;
            const float distL = length(toLight);
            l = toLight / max(distL, 1e-4);
            // Smooth inverse-square falloff with a range cutoff.
            const float range = radiance.w;
            const float att = range > 0.0
                ? clamp(1.0 - pow(distL / range, 4.0), 0.0, 1.0)
                : 1.0;
            li *= att / max(distL * distL, 1e-4);
        }
        const float3 h = normalize(viewDir + l);
        const float ndl = max(dot(nrm, l), 0.0);
        const float ndh = max(dot(nrm, h), 0.0);
        const float3 f = fresnelSchlick(max(dot(h, viewDir), 0.0), f0);
        const float ndf = distributionGGX(ndh, rm.x);
        const float g = geometrySmith(ndv, max(ndl, 1e-4), rm.x);
        const float3 spec =
            (ndf * g * f) / max(4.0 * ndv * max(ndl, 1e-4), 1e-4);
        const float3 kd = (float3(1.0, 1.0, 1.0) - f) * (1.0 - rm.y);
        lo += (kd * albedo / 3.14159265 + spec) * li * ndl;
    }

    const float3 ambient = albedo * push.viewportAmb.z;
    const float3 color = ambient + lo;
    // Reinhard tonemap with exposure - the offscreen is 8-bit UNORM;
    // raw GGX output would clip harshly.
    const float3 mapped =
        color * push.viewportAmb.w / (color * push.viewportAmb.w + 1.0);
    if (!(mapped.x >= 0.0) || !(mapped.y >= 0.0) || !(mapped.z >= 0.0))
        return float4(0.5, 0.0, 0.5, 1.0);
    return float4(mapped, 1.0);
}
