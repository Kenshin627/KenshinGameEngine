// Baseline forward path (S2.2/S2.3): PNU vertex input, push-constant MVP,
// Blinn-Phong (directional + ambient + specular). Vertex data arrives
// through three separate bindings (positions/normals/uvs, matching
// MeshData's SOA layout).

struct PushConstants
{
    float4x4 viewProj;
    float4   kd;       // material diffuse
    float4   lightDir; // xyz = direction TO light (world), w unused
    float4   camPos;   // xyz = world-space camera position, w unused
};

[[vk::push_constant]] ConstantBuffer<PushConstants> push;

struct VSInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal   : NORMAL;
    [[vk::location(2)]] float2 uv       : TEXCOORD0;
};

struct VSOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 normal   : NORMAL;
    [[vk::location(1)]] float2 uv       : TEXCOORD0;
    [[vk::location(2)]] float3 worldPos : WORLD_POS;
};

[shader("vertex")]
VSOutput vsMain(VSInput input)
{
    VSOutput output;
    output.position = mul(push.viewProj, float4(input.position, 1.0));
    output.normal = input.normal;
    output.uv = input.uv;
    output.worldPos = input.position;
    return output;
}

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target
{
    const float3 n = normalize(input.normal);
    const float3 l = normalize(push.lightDir.xyz);
    const float3 v = normalize(push.camPos.xyz - input.worldPos);
    const float3 h = normalize(l + v);

    const float ndl = max(0.0, dot(n, l));
    const float ndh = max(0.0, dot(n, h));
    const float specular = pow(ndh, 48.0) * 0.35;

    // Ambient + diffuse + Blinn-Phong specular.
    const float3 color = push.kd.rgb * (0.15 + 0.85 * ndl) + specular;
    return float4(1.0, 0.0, 0.0, 1.0);
}
