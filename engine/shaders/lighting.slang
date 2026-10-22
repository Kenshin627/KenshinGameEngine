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
// M8.3: output is LINEAR HDR into the per-slot RGBA16F target - the
// exposure/tonemap half moved to tonemap.slang (the present pass).
// Background pixels write ALPHA 0: tonemap.slang passes the clear
// color through for that sentinel. Lit pixels always carry alpha 1
// by construction, so an rgb-zero LIT pixel (ambient=0, unlit
// backfaces, metallic black) still tonemaps normally.
//
// Push (packed identically in C++ - 112B of the 128B budget):
//   float4x4 invViewProj (64B) | camPos + lightCount (16B) |
//   clearColor rgb + materialCount (16B) | 1/w, 1/h + ambient + pad (16B)

struct LightPush {
  float4x4 invViewProj;
  float4 camPosCount;  // xyz camPos, w light count
  float4 clearColor;   // rgb the frame's clear color (background),
                       // a the MATERIAL COUNT (M8.2 id clamp bound -
                       // the push block rides existing words; the
                       // alpha of a clear color is meaningless)
  float4
      viewportAmb;  // xy = 1/w, 1/h, z ambient, w pad (M8.3: exposure moved to the tonemap push)
};

// The material table record (packed identically to decode_material's
struct GpuLight {
  float4 a;
  float4 b;
};

[[vk::push_constant]] ConstantBuffer<LightPush> push;

[[vk::binding(0)]] Sampler2D gbufferA;
[[vk::binding(1)]] Sampler2D gbufferB;
[[vk::binding(2)]] StructuredBuffer<GpuLight> lights;
// The environment table: [0] = sky rgb + SH-valid flag, [1] = ground
// rgb + env intensity, [2..10] = the probe's irradiance as L2 SH
// (cosine-convolved, /pi - LightingPass::projectIrradianceSH). With a
// probe, diffuse AND specular ambient both come from it under one
// intensity (UE sky light); without one, the sky/ground hemisphere
// times `ambient` is the fallback.
[[vk::binding(3)]] StructuredBuffer<float4> env;

float3 shIrradiance(float3 n) {
  const float3 r =
      env[2].rgb * 0.282095 + env[3].rgb * (0.488603 * n.y) +
      env[4].rgb * (0.488603 * n.z) + env[5].rgb * (0.488603 * n.x) +
      env[6].rgb * (1.092548 * n.x * n.y) +
      env[7].rgb * (1.092548 * n.y * n.z) +
      env[8].rgb * (0.315392 * (3.0 * n.z * n.z - 1.0)) +
      env[9].rgb * (1.092548 * n.x * n.z) +
      env[10].rgb * (0.546274 * (n.x * n.x - n.y * n.y));
  return max(r, float3(0.0));
}
// M9.1: the directional light's shadow map (this frame's shadow-chain
// decode; the backend binds its view to the reflected sampler
// bindings after the GBuffer pair) + the shadow parameter block
// ([0..3] the light viewProj columns, [4] = mapSize, bias, texelWorld,
// enabled).
// M9.1 Stage B: the clipmap levels (0 = finest, extent0 radius;
// each coarser level doubles extent AND texel size - the same
// texel-density-per-distance calibration as UE's clipmaps). The
// sampling side picks the level by distance (log2) with a
// coarse-fallback when the pixel falls outside a level's box.
[[vk::binding(5)]] Sampler2D shadowMap0;
[[vk::binding(6)]] Sampler2D shadowMap1;
[[vk::binding(7)]] Sampler2D shadowMap2;
[[vk::binding(4)]] StructuredBuffer<float4> shadowParams;
// The IBL environment probe: an equirectangular HDR texture whose mip
// chain is the prefiltered radiance ladder (LOD = roughness).
[[vk::binding(8)]] Sampler2DArray envMap;  // layer 0 (arrayed views)

// NOTE on matrices: glm stores column-major and Slang's mul reads
// buffer vec4s as rows - every transform below is an explicit linear
// combination of the stored columns (the project-wide convention).
float4 xform(float4x4 m, float4 v) {
  return m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3] * v.w;
}

struct VSOut {
  float4 pos : SV_Position;
};

[shader("vertex")] VSOut vsMain(uint id : SV_VertexID) {
  VSOut o;
  o.pos = float4(float2((id << 1u) & 2u, id & 2u) * 2.0 - 1.0, 0.0, 1.0);
  return o;
}

/// PCF 3x3 over the shadow map in LIGHT clip space (2026-09-25 UE
/// cross-check vs ShadowFilteringCommon.ush): each tap compares through
/// a SOFT transition ramp saturate((stored - receiver)*scale + 1)
/// rather than a binary <=; the depth bias is constant + SLOPE-scaled
/// (slope = tan of the angle to the light, clamped). `sp` rows: the
/// light viewProj columns; params = (mapSize, bias, texelWorld,
/// enabled). Returns 1 = lit, 0 = shadowed.
// Level L block at base = 2 + 6*L: [base] = pad, [base+1] =
// (mapSize, bias, texelWorld, enabled), [base+2..base+5] = the
// level's light viewProj columns. Header: [0] = (log2Extent0,
// levelCount, globalEnable, pad); [1] = clipmap origin.xyz, pad.
// Level selection: UE's CalcAbsoluteClipmapLevel = log2(distance)
// biased by the extent calibration - ours is floor(log2(d) - log2(
// extent0)) clamped, the same shape with firstLevel folded in. A
// pixel outside its level's box falls back to COARSER levels (the
// single-level ancestor returned lit outside; the clipmap tries the
// bigger box - UE's LODOffset fallback in spirit). Returns -1.0 from
// a level = outside, try coarser.
float shadowFactorLevel(float3 worldPos, float3 nrm, float ndl, uint base,
                        Sampler2D map0, Sampler2D map1, Sampler2D map2,
                        uint level) {
  const float4 p4 = shadowParams[base + 1u];
  if (p4.w < 0.5) return -1.0;  // level disabled
  const float4x4 lvp =
      float4x4(shadowParams[base + 2u], shadowParams[base + 3u],
               shadowParams[base + 4u], shadowParams[base + 5u]);
  const float4 clip0 =
      lvp[0] * worldPos.x + lvp[1] * worldPos.y + lvp[2] * worldPos.z + lvp[3];
  if (abs(clip0.x) > 1.0 || abs(clip0.y) > 1.0 || clip0.z < 0.0 ||
      clip0.z > 1.0)
    return -1.0;  // outside this level's frustum: try coarser
  // Normal-offset bias: lift along the normal by two world texels
  // (texelWorld rides the level, so coarser levels lift more -
  // UE's per-level bias scaling). The uv SAMPLES the UN-lifted
  // position and only the COMPARE depth uses the lift: at coarse
  // levels the lift's horizontal projection exceeds a texel, and
  // sampling the shifted uv reads a NEIGHBOR texel's staircase
  // depth (the diagonal band artifact).
  const float4 clip =
      lvp[0] * worldPos.x + lvp[1] * worldPos.y + lvp[2] * worldPos.z + lvp[3];
  const float3 lifted = worldPos + nrm * (p4.z * 2.0);
  const float4 clipN =
      lvp[0] * lifted.x + lvp[1] * lifted.y + lvp[2] * lifted.z + lvp[3];
  const float2 uv = clip0.xy * 0.5 + 0.5;
  // UE ComputeDepthBiasDirectionalSpot: slope = tan(angle to light)
  // = sqrt(1-NoL^2)/NoL clamped; bias = constant + slopeScale*slope.
  const float slope =
      clamp(sqrt(saturate(1.0 - ndl * ndl)) / max(ndl, 1e-4), 0.0, 1.0);
  // UE ShadowRendering.cpp:2055: ShaderSlopeDepthBias = DepthBias *
  // SlopeScaleDepthBias (CSM slope scale 3) - the slope term scales
  // WITH the constant bias. The fixed 0.01 it replaced was 6.7x the
  // constant bias at max slope = ~0.4 m along the light ray in this
  // 40 m depth range: contact shadows detached from their casters
  // (peter-panning) on every low-sun receiver.
  const float bias = p4.y * (1.0 + 3.0 * slope);
  const float depth = clipN.z - bias;
  const float texel = 1.0 / p4.x;
  const float transitionScale = 400.0;
  float sv = 1.0;
  float lit = 0.0;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      const float2 suv = clamp(uv + float2(float(dx), float(dy)) * texel,
                               float2(0.0), float2(1.0));
      // Sampler bindings are not indexable - pick the map once
      // per tap through an if-chain (3-deep, uniform per pixel).
      if (level == 0u)
        sv = map0.SampleLevel(suv, 0.0).r;
      else if (level == 1u)
        sv = map1.SampleLevel(suv, 0.0).r;
      else
        sv = map2.SampleLevel(suv, 0.0).r;
      lit += saturate((sv - depth) * transitionScale + 1.0);
    }
  return lit / 9.0;
}

float shadowFactor(float3 worldPos, float3 nrm, float ndl) {
  const float4 header = shadowParams[0];
  if (header.z < 0.5) return 1.0;  // shadows disabled
  const uint count = uint(header.y);
  const float3 origin = shadowParams[1].xyz;
  const float d = max(length(worldPos - origin), 1e-4);
  int level = int(floor(log2(d) - header.x));  // header.x = log2(extent0)
  level = clamp(level, 0, int(count) - 1);
  for (uint lv = uint(level); lv < count; ++lv) {
    const float f = shadowFactorLevel(worldPos, nrm, ndl, 2u + lv * 6u,
                                      shadowMap0, shadowMap1, shadowMap2, lv);
    if (f >= 0.0) return f;
  }
  return 1.0;
}

// GGX terms (minimal PBR; roughness/metallic come from the material
// table via GBuffer B's alpha channel - the decode pass wrote the
// per-cluster id there).

float distributionGGX(float ndh, float rough) {
  const float a = rough * rough;
  const float a2 = a * a;
  const float d = ndh * ndh * (a2 - 1.0) + 1.0;
  return a2 / max(3.14159265 * d * d, 1e-7);
}
/// Heitz 2014 joint-Smith APPROXIMATION - UE's Vis_SmithJointApprox
/// (BRDF.ush:373). Includes the 1/(4 NoV NoL) denominator (the file
/// convention note below) - do NOT divide by it again at the use site.
/// `a2` = roughness^4 (the GGX alpha squared).
float visSmithJointApprox(float a2, float ndv, float ndl) {
  const float a = sqrt(a2);
  const float v = ndl * (ndv * (1.0 - a) + a);
  const float l = ndv * (ndl * (1.0 - a) + a);
  return 0.5 / max(v + l, 1e-7);
}
float3 fresnelSchlick(float ct, float3 f0) {
  // UE's micro-occlusion form (BRDF.ush:403): anything below ~2%
  // reflectance is treated as shadowing - the whole lobe collapses
  // (matters for metallic-dark albedos; identical at f0 = 0.04).
  const float fc = pow(1.0 - ct, 5.0);
  return saturate(50.0 * f0.g) * fc + (1.0 - fc) * f0;
}

[shader("fragment")] float4 fsMain(float4 frag : SV_Position) : SV_Target {
  const float2 uv = frag.xy * push.viewportAmb.xy;
  const float4 a = gbufferA.SampleLevel(uv, 0.0);
  // Background sentinel: alpha 0 (NOT an rgb test - lit pixels CAN
  // be exact-zero rgb with ambient=0 / unlit backfaces / metallic
  // black materials, and the f16 store rounds very dark values to
  // zero; alpha is written 1 for every lit pixel by construction).
  if (a.w <= 0.0) return float4(0.0, 0.0, 0.0, 0.0);

  const float4 bTexel = gbufferB.SampleLevel(uv, 0.0);
  const float3 albedo = bTexel.rgb;
  // Per-pixel roughness/metallic packed by the decode into B.a
  // (5 bits rough / 3 bits metal; the material-id payload it
  // replaced had no other consumer).
  const float rmPacked = round(bTexel.a * 255.0);
  const float roughQ = floor(rmPacked / 8.0);
  const float metalQ = rmPacked - roughQ * 8.0;
  const float2 rm = clamp(float2(roughQ / 31.0, metalQ / 7.0),
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
  // UE ShadingModels.ush:252 - abs + epsilon keeps saturate(dot)
  // finite at silhouette edges.
  const float ndv = saturate(abs(dot(nrm, viewDir)) + 1e-5);
  const float3 f0 = lerp(float3(0.04, 0.04, 0.04), albedo, rm.y);

  float3 lo = float3(0.0, 0.0, 0.0);
  const uint lightCount = uint(push.camPosCount.w + 0.5);
  for (uint i = 0; i < lightCount; ++i) {
    const float4 meta = lights[i].a;
    const float4 radiance = lights[i].b;
    // The light vector FIRST: the half vector and the shadow's slope
    // bias both depend on it (they used to read the still-unassigned
    // `l` - an undefined half vector put the GGX lobe around N.V
    // instead of N.H, and the slope bias read garbage).
    float3 l;
    float3 li = radiance.rgb;
    if (meta.w < 0.5) {
      l = normalize(meta.xyz);  // directional: TO light
    } else {
      const float3 toLight = meta.xyz - worldPos;
      const float distL = length(toLight);
      l = toLight / max(distL, 1e-4);
      // Smooth inverse-square falloff with a range cutoff.
      const float range = radiance.w;
      const float att =
          range > 0.0 ? clamp(1.0 - pow(distL / range, 4.0), 0.0, 1.0) : 1.0;
      li *= att / max(distL * distL, 1e-4);
    }
    const float ndl = max(dot(nrm, l), 0.0);
    if (ndl <= 0.0) continue;  // facing away: no diffuse, no specular
    const float3 h = normalize(viewDir + l);
    const float shadow =
        (meta.w < 0.5) ? shadowFactor(worldPos, nrm, ndl) : 1.0;
    const float ndh = max(dot(nrm, h), 0.0);
    const float3 f = fresnelSchlick(max(dot(h, viewDir), 0.0), f0);
    // UE DefaultLitBxDF (ShadingModels.ush:180): specular = D * Vis
    // * F with Vis carrying the 1/(4 NoV NoL); roughness clamped to
    // the view MinRoughness (0.02). Diffuse is UE-legacy WITHOUT
    // the (1-F) energy-conservation factor (USE_ENERGY_CONSERVATION
    // off unless Substrate) - albedo * (1-metallic) / PI.
    const float rough = max(rm.x, 0.02);
    const float a2 = rough * rough * rough * rough;
    const float ndf = distributionGGX(ndh, rough);
    const float vis = visSmithJointApprox(a2, ndv, max(ndl, 1e-4));
    const float3 spec = ndf * vis * f;
    const float3 kd = albedo * (1.0 - rm.y);
    lo += (kd / 3.14159265 + spec) * li * ndl * shadow;
  }

  // Hemisphere ambient (M8.3): sky from above, ground from below -
  // the cheap diffuse-IBL stand-in for a prefiltered probe. The engine
  // is Z-UP (the equirect probe below, the light rig and the glTF axis
  // fix all agree): the blend axis is n.z (it was n.y - every
  // horizontal surface got the half/half average and vertical faces
  // flipped sky/ground with their azimuth).
  const float3 hemi = lerp(env[1].rgb, env[0].rgb, nrm.z * 0.5 + 0.5);
  // Specular ambient (IBL): the environment probe sampled along the
  // REFLECTED view ray, LOD = roughness * maxLod (the mip chain is
  // the prefiltered radiance ladder), through the UE-mobile analytic
  // split-sum fit. Without it a metal surface is black between the
  // (few) direct highlights - the metal look IS the environment.
  const float3 rdir = reflect(-viewDir, nrm);
  const float2 equi = float2(atan2(rdir.y, rdir.x) / 6.28318530718 + 0.5,
                             acos(clamp(rdir.z, -1.0, 1.0)) / 3.14159265359);
  // Roughness -> mip: UE's ComputeReflectionCaptureMipFromRoughness
  // (ReflectionEnvironmentShared.ush: LevelFrom1x1 = 1 - 1.2 *
  // log2(roughness), mip = cubeMips - 1 - LevelFrom1x1). An equirect of
  // width W holds the same texel density as a W/4 cube face, i.e. a
  // cube with maxLod - 1 mips: mip = maxLod - 3 + 1.2 * log2(r). The
  // old linear r * maxLod under-blurred every mid roughness (r 0.25 ->
  // mip 2.3 vs 3.6) - mirror-sharp reflections on satin surfaces.
  const float maxLod = push.viewportAmb.w;
  const float envMip =
      clamp(maxLod - 3.0 + 1.2 * log2(max(rm.x, 0.001)), 0.0, maxLod);
  const float3 rHemi = envMap.SampleLevel(float3(equi, 0.0), envMip).rgb;
  // UE EnvBRDFApprox (BRDF.ush:622): the Lazarov fit, with F90 =
  // saturate(50 * f0.g) on the bias term (the same <2% "shadowing"
  // rule as the direct Fresnel - the bias term ignored it).
  const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
  const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
  const float4 rr = float4(rm.x) * c0 + c1;
  const float a004 = min(rr.x * rr.x, exp2(-9.28 * ndv)) * rr.x + rr.y;
  const float2 ab = float2(-1.04, 1.04) * a004 + rr.zw;
  const bool probe = env[0].w > 0.5;
  const float envIntensity = probe ? env[1].w : 1.0;
  const float3 envSpec =
      (f0 * ab.x + saturate(50.0 * f0.g) * ab.y) * rHemi * envIntensity;
  // Diffuse ambient carries the same (1 - metallic) as the direct
  // Lambert term - a metal has no diffuse lobe (it used to get the
  // full albedo * ambient on top of its specular reflection).
  const float3 diffuseColor = albedo * (1.0 - rm.y);
  const float3 ambientIrr =
      probe ? shIrradiance(nrm) * envIntensity : hemi * push.viewportAmb.z;
  const float3 color = diffuseColor * ambientIrr + envSpec + lo;
  // HDR out - scrub garbage (crossing-triangle paths) to the
  // BACKGROUND sentinel (alpha 0); magenta was too easy to miss
  // under an exposure curve.
  if (!(color.x >= 0.0) || !(color.y >= 0.0) || !(color.z >= 0.0) ||
      !(color.x < 65504.0) || !(color.y < 65504.0) || !(color.z < 65504.0))
    return float4(0.0, 0.0, 0.0, 0.0);
  return float4(color, 1.0);
}
