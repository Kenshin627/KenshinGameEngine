// bindless_probe.slang - test-only compute probe of the bindless
// material-texture table (tests/src/bindless_tests.cpp, roadmap 1.3a).
// Each probe samples one texture reference through sampleTextureGrad -
// the very function the material shaders call - with an isotropic UV
// gradient chosen on the CPU to land on an exact mip level.
//
//   0: float4 probes[]  - (asfloat(ref), u, v, gradient)
//   1: float4 results[] - the sampled RGBA

import bindless;

[[vk::binding(0)]] StructuredBuffer<float4> probes;
[[vk::binding(1)]] RWStructuredBuffer<float4> results;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  uint count = 0, stride = 0;
  probes.GetDimensions(count, stride);
  if (id.x >= count) return;
  const float4 p = probes[id.x];
  results[id.x] =
      sampleTextureGrad(asuint(p.x), p.yz, float2(p.w, 0.0), float2(0.0, p.w));
}
