// M8.2 texture-array sampling probe: validates, headless on the GPU,
// the exact sampling form the albedo path uses - Texture2DArray +
// separate SamplerState + SampleLevel(uv3, lod) + GetDimensions - end
// to end (upload -> mip blits -> descriptor write -> sample -> read).
//
// Bindings (set 0):
//   0: Texture2DArray albedo (RGBA8, N layers, mip chain)
//   1: SamplerState albedoSamp (the backend's albedo sampler)
//   2: RWStructuredBuffer<float4> outColor - [0] = sampled rgb + mip
//      count (a in [0,1] encodes mips-1/15 so one word carries both)
//   3: StructuredBuffer<float4> params - [0] = (uv.x, uv.y, layer,
//      lod)

[[vk::binding(0)]] Texture2DArray<float4> albedo;
[[vk::binding(1)]] SamplerState albedoSamp;
[[vk::binding(2)]] RWStructuredBuffer<float4> outColor;
[[vk::binding(3)]] StructuredBuffer<float4> params;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
  const float4 p = params[0];
  uint w = 0, h = 0, elements = 0, mips = 0;
  albedo.GetDimensions(0, w, h, elements, mips);
  const float4 s =
      albedo.SampleLevel(albedoSamp, float3(p.xy, p.z), max(p.w, 0.0));
  outColor[0] = float4(s.rgb, min(float(mips - 1), 15.0) / 15.0);
}
