// Baseline forward path (S2.2): PNU vertex input, push-constant MVP,
// Lambert + ambient. Vertex data arrives through three separate bindings
// (positions/normals/uvs, matching MeshData's SOA layout).

struct PushConstants
{
    float4x4 viewProj;
    float4   kd;      // material diffuse
    float4   lightDir; // xyz = direction TO light (world), w unused
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
    [[vk::location(0)]] float3 normal : NORMAL;
    [[vk::location(1)]] float2 uv : TEXCOORD0;
};

[shader("vertex")]
VSOutput vsMain(VSInput input)
{
    VSOutput output;
    output.position = mul(push.viewProj, float4(input.position, 1.0));
    output.normal = input.normal;
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 fsMain(VSOutput input) : SV_Target
{
    const float3 n = normalize(input.normal);
    const float3 l = normalize(push.lightDir.xyz);
    const float ndl = max(0.0, dot(n, l));
    const float3 color = push.kd.rgb * (0.15 + 0.85 * ndl);
    return float4(color, 1.0);
}
