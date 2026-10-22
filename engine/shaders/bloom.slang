// Bloom (COD: Advanced Warfare's "Next Generation Post Processing",
// Jimenez SIGGRAPH 2014 - the shape most engine bloom derives from):
// a half-resolution mip chain built with a 13-tap downsample, then
// walked back up with a 3x3 tent, each level ADDITIVELY blended into
// the next larger one (the pipeline's ONE/ONE blend). tonemap.slang
// composites the top level energy-conservingly: scene * (1 - s) +
// bloom * s - nothing is thresholded, every bright pixel spreads, and
// the frame does not brighten overall.
//
// The source is the frame's resolved HDR (the TAA output when TAA ran,
// otherwise the lighting target): premultiplied coverage rgb, i.e. the
// pixel's radiant energy - exactly what bloom should spread.
//
// Bindings: 0 = Sampler2D src (linear/clamp).
// Push (32B): srcTexel.xy = 1/srcW, 1/srcH, .z = 1 on the FIRST
//   downsample (Karis-weighted against fireflies: one sun glint
//   would otherwise flicker as a frame-sized blob), dstTexel.xy =
//   1/dstW, 1/dstH, dstTexel.z = the tent radius in source texels.

struct BloomPush {
  float4 srcTexel;
  float4 dstTexel;
};
[[vk::push_constant]] ConstantBuffer<BloomPush> push;
[[vk::binding(0)]] Sampler2D src;

struct VSOut {
  float4 pos : SV_Position;
};

[shader("vertex")]
VSOut vsMain(uint id : SV_VertexID) {
  VSOut o;
  o.pos = float4(float2((id << 1u) & 2u, id & 2u) * 2.0 - 1.0, 0.0, 1.0);
  return o;
}

float3 fetch(float2 uv) {
  const float3 c = src.SampleLevel(uv, 0.0).rgb;
  // NaN/inf/negative scrub + a ceiling (the f16 sun glint): bloom must
  // never turn one bad texel into a frame-wide smear.
  if (!(c.x >= 0.0 && c.y >= 0.0 && c.z >= 0.0)) return float3(0.0);
  return min(c, float3(1000.0));
}

float karisWeight(float3 c) {
  return 1.0 / (1.0 + dot(c, float3(0.2126, 0.7152, 0.0722)));
}

[shader("fragment")]
float4 fsDown(float4 frag : SV_Position) : SV_Target {
  const float2 uv = frag.xy * push.dstTexel.xy;
  const float2 t = push.srcTexel.xy;
  const float3 a = fetch(uv + t * float2(-2, -2));
  const float3 b = fetch(uv + t * float2(0, -2));
  const float3 c = fetch(uv + t * float2(2, -2));
  const float3 d = fetch(uv + t * float2(-1, -1));
  const float3 e = fetch(uv + t * float2(1, -1));
  const float3 f = fetch(uv + t * float2(-2, 0));
  const float3 g = fetch(uv);
  const float3 h = fetch(uv + t * float2(2, 0));
  const float3 i = fetch(uv + t * float2(-1, 1));
  const float3 j = fetch(uv + t * float2(1, 1));
  const float3 k = fetch(uv + t * float2(-2, 2));
  const float3 l = fetch(uv + t * float2(0, 2));
  const float3 m = fetch(uv + t * float2(2, 2));
  // Five overlapping 2x2 boxes: the center one at weight 0.5, the four
  // corner ones at 0.125 each.
  const float3 g0 = (d + e + i + j) * 0.25;
  const float3 g1 = (a + b + f + g) * 0.25;
  const float3 g2 = (b + c + g + h) * 0.25;
  const float3 g3 = (f + g + k + l) * 0.25;
  const float3 g4 = (g + h + l + m) * 0.25;
  float w0 = 0.5, w1 = 0.125, w2 = 0.125, w3 = 0.125, w4 = 0.125;
  if (push.srcTexel.z > 0.5) {
    // Karis average on the first level only (later levels are already
    // smooth).
    w0 *= karisWeight(g0);
    w1 *= karisWeight(g1);
    w2 *= karisWeight(g2);
    w3 *= karisWeight(g3);
    w4 *= karisWeight(g4);
  }
  const float3 sum =
      (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) /
      max(w0 + w1 + w2 + w3 + w4, 1e-6);
  return float4(sum, 1.0);
}

[shader("fragment")]
float4 fsUp(float4 frag : SV_Position) : SV_Target {
  const float2 uv = frag.xy * push.dstTexel.xy;
  const float2 t = push.srcTexel.xy * push.dstTexel.z;
  float3 s = fetch(uv) * 4.0;
  s += (fetch(uv + t * float2(-1, 0)) + fetch(uv + t * float2(1, 0)) +
        fetch(uv + t * float2(0, -1)) + fetch(uv + t * float2(0, 1))) *
       2.0;
  s += fetch(uv + t * float2(-1, -1)) + fetch(uv + t * float2(1, -1)) +
       fetch(uv + t * float2(-1, 1)) + fetch(uv + t * float2(1, 1));
  return float4(s * (1.0 / 16.0), 1.0);  // ADDITIVE into the larger level
}
