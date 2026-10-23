// ── exposure.slang — histogram auto exposure (UE eye adaptation port) ──
// UE 5.8 AEM_Histogram (PostProcessHistogram.usf / PostProcessHistogram
// Common.ush / PostProcessEyeAdaptation.usf), two compute entries:
//
//   csHistogram  64-bin log2-luminance histogram of the resolved HDR.
//                Each pixel splits its weight between the two nearest
//                buckets (UE's bilinear bucket weights). Background
//                pixels (alpha 0 = the lighting pass's sentinel) meter
//                the ENVIRONMENT radiance in their view direction: UE's
//                histogram sees the sky, and an object-only meter maps
//                every object to mid grey (a white knot turns grey, a
//                black helmet glows). The environment sets the exposure;
//                albedo then reads as albedo.
//   csAdapt      one group: the outlier-trimmed mean log luminance
//                (ComputeAverageLuminanceWithoutOutlier), the target
//                exposure and UE's linear-then-exponential adaptation
//                (ComputeEyeAdaptation), written to state[] for the
//                tonemap pass of the SAME frame.
//
// Bindings (one layout per entry; the backend fills by binding index):
//   0: Sampler2D sceneTex                - premultiplied HDR (rgb, a cov)
//   1: RWStructuredBuffer<uint> histogram - 64 fixed-point bucket sums
//   2: RWStructuredBuffer<float4> state   - [0] (scale, targetScale,
//      avgLum, valid), [1] (adapted white-point exposure E, 0, 0, 0)
//   3: Sampler2DArray envMap              - the IBL equirect (Z-up,
//      lighting.slang's convention), layer 0
//
// "Exposure" E follows UE: the white point (average luminance / 0.18);
// the image scale is compensation / E, i.e. the average luminance lands
// on 0.18 * compensation (compensation = 2^bias).

static const uint kBins = 64;
static const float kMiddleGrey = 0.18;

struct ExposurePush {
  float4 histo;  // x HistogramScale (1 / log2 range), y HistogramBias, z
                 // LuminanceMin (exp2(log min)), w env intensity
  float4 pct;    // x low fraction, y high fraction, z min average
                 // luminance, w max average luminance
  float4 adapt;  // x speed up, y speed down (f-stops / s), z frame dt
                 // (s), w force target (1 = jump: headless / first frame)
  float4 curve;  // x compensation 2^bias, y ExponentialUpM, z
                 // ExponentialDownM, w env mip (< 0 = no env: background
                 // pixels carry no weight)
  float4x4 invViewProj;  // unjittered; uv -> NDC directly (no y flip)
};
static const float kStartDistance = 1.5;  // r.EyeAdaptation.Exponential
                                          // TransitionDistance
[[vk::push_constant]] ConstantBuffer<ExposurePush> push;

[[vk::binding(0)]] Sampler2D sceneTex;
[[vk::binding(1)]] RWStructuredBuffer<uint> histogram;
[[vk::binding(2)]] RWStructuredBuffer<float4> state;
[[vk::binding(3)]] Sampler2DArray envMap;

float4 xform(float4x4 m, float4 v) {
  return m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3] * v.w;
}

float luminance(float3 rgb) { return dot(rgb, float3(0.2126, 0.7152, 0.0722)); }

// The environment radiance behind pixel `uv` (lighting.slang's ray
// reconstruction + equirect lookup).
float envLuminance(float2 uv) {
  const float2 ndc = uv * 2.0 - 1.0;
  const float4 n4 = xform(push.invViewProj, float4(ndc, 0.0, 1.0));
  const float4 f4 = xform(push.invViewProj, float4(ndc, 1.0, 1.0));
  const float3 d = normalize(f4.xyz / f4.w - n4.xyz / n4.w);
  const float2 equi = float2(atan2(d.y, d.x) / 6.28318530718 + 0.5,
                             acos(clamp(d.z, -1.0, 1.0)) / 3.14159265359);
  const float3 rgb =
      envMap.SampleLevel(float3(equi, 0.0), push.curve.w).rgb * push.histo.w;
  return luminance(max(rgb, float3(0.0)));
}

groupshared uint gHist[kBins];

// Weights are fixed point (x 256): order-independent integer atomics keep
// the histogram - and so every headless screenshot - bit-deterministic.
// 4K = 8.3M px x 256 = 2.1e9 < 2^32 per bucket.
static const float kWeightScale = 256.0;

[shader("compute")]
[numthreads(16, 16, 1)]
void csHistogram(uint3 tid: SV_DispatchThreadID, uint gi: SV_GroupIndex) {
  if (gi < kBins) gHist[gi] = 0u;
  GroupMemoryBarrierWithGroupSync();
  uint w = 1, h = 1;
  sceneTex.GetDimensions(w, h);
  if (tid.x < w && tid.y < h) {
    const float2 uv = (float2(tid.xy) + 0.5) / float2(w, h);
    const float4 c = sceneTex.SampleLevel(uv, 0.0);
    float lum = -1.0, cov = 0.0;
    if (c.a > 0.0 && c.r >= 0.0 && c.g >= 0.0 && c.b >= 0.0 &&
        c.r < 65504.0 && c.g < 65504.0 && c.b < 65504.0) {
      cov = saturate(c.a);
      lum = luminance(c.rgb / cov);  // un-premultiply (TAA silhouettes)
    } else if (c.a <= 0.0 && push.curve.w >= 0.0) {
      cov = 1.0;
      lum = envLuminance(uv);
    }
    if (lum >= 0.0 && lum < 65504.0) {
      lum = max(lum, push.histo.z);
      const float pos =
          saturate(log2(lum) * push.histo.x + push.histo.y) * float(kBins - 1);
      const uint b0 = min(uint(pos), kBins - 1u);
      const uint b1 = min(b0 + 1u, kBins - 1u);
      const float w1 = pos - float(b0);
      InterlockedAdd(gHist[b0], uint((1.0 - w1) * cov * kWeightScale + 0.5));
      InterlockedAdd(gHist[b1], uint(w1 * cov * kWeightScale + 0.5));
    }
  }
  GroupMemoryBarrierWithGroupSync();
  if (gi < kBins && gHist[gi] != 0u) InterlockedAdd(histogram[gi], gHist[gi]);
}

// UE ComputeEyeAdaptation (PostProcessHistogramCommon.ush): linear in
// log2 space while far from the target, exponential within StartDistance
// f-stops (M matches the two slopes at the switch point).
float adaptExposure(float oldE, float targetE, float dt) {
  const float logTarget = log2(targetE);
  const float logOld = log2(oldE);
  const float logDiff = logTarget - logOld;
  const float speed = logDiff > 0.0 ? push.adapt.x : push.adapt.y;
  const float m = logDiff > 0.0 ? push.curve.y : push.curve.z;
  const float expo = logOld + (logTarget - logOld) * (1.0 - exp2(-dt * speed)) * m;
  const float offset = dt * speed;
  const float lin = logOld < logTarget ? min(logTarget, logOld + offset)
                                       : max(logTarget, logOld - offset);
  return exp2(abs(logDiff) > kStartDistance ? lin : expo);
}

groupshared float gBin[kBins];

[shader("compute")]
[numthreads(64, 1, 1)]
void csAdapt(uint gi: SV_GroupIndex) {
  gBin[gi] = float(histogram[gi]);
  GroupMemoryBarrierWithGroupSync();
  if (gi != 0u) return;

  float sum = 0.0;
  for (uint i = 0; i < kBins; ++i) sum += gBin[i];
  // ComputeAverageLuminanceWithoutOutlier: drop the darkest low% and
  // everything above high% of the weight, average log2 luminance.
  float minFraction = sum * push.pct.x;
  float maxFraction = sum * push.pct.y;
  float2 acc = float2(0.0, 0.0);
  for (uint i = 0; i < kBins; ++i) {
    float v = gBin[i];
    const float sub = min(v, minFraction);
    v -= sub;
    minFraction -= sub;
    maxFraction -= sub;
    v = min(v, maxFraction);
    maxFraction -= v;
    const float logLum =
        (float(i) / float(kBins - 1) - push.histo.y) / push.histo.x;
    acc += float2(logLum, 1.0) * v;
  }
  // Empty histogram (all background): UE's fallback.
  const float avgLum =
      acc.y > 0.0 ? exp2(acc.x / acc.y) : push.pct.z / kMiddleGrey;

  const float targetE =
      clamp(avgLum, push.pct.z, push.pct.w) / kMiddleGrey;
  const float4 prev = state[0];
  const float oldE = state[1].x;
  const bool force = push.adapt.w > 0.5 || !(prev.w > 0.5) ||
                     !(oldE > 0.0) || !(oldE < 1e6);
  float e = force ? targetE : adaptExposure(oldE, targetE, push.adapt.z);
  e = clamp(e, push.pct.z / kMiddleGrey, push.pct.w / kMiddleGrey);
  const float comp = push.curve.x;
  state[0] = float4(comp / max(e, 1e-4), comp / max(targetE, 1e-4), avgLum,
                    1.0);
  state[1] = float4(e, 0.0, 0.0, 0.0);
}
