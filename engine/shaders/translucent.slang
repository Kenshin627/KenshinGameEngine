// Translucent forward pass - glTF alphaMode BLEND materials (UE's split:
// Nanite never rasterizes translucency; translucent geometry renders in
// a forward pass after the opaque lighting). The builder moves BLEND
// triangles out of the cluster DAG into <name>.tmesh, a non-indexed
// triangle list this pass vertex-pulls. Runs after lighting.slang into
// the same per-slot HDR target, before TAA (the resolve then
// antialiases the translucent surface with everything else).
//
// Shading: lighting.slang's model minus shadows (v1) - GGX key/point
// lights, IBL specular through the prefiltered probe + EnvBRDFApprox,
// SH irradiance diffuse. Blending: PREMULTIPLIED (the pipeline blends
// ONE / ONE_MINUS_SRC_ALPHA) with opacity on the DIFFUSE body only -
// Filament's TRANSPARENT mode, see the end of fsMain - and the alpha
// channel accumulates
// coverage (a = Opacity + a * (1 - Opacity)) so glass over the
// background composites through tonemap.slang's coverage path exactly
// like a TAA silhouette.
//
// Occlusion: no depth attachment exists - the opaque frame's depth IS
// the VisBuffer's high word ((1 - z) * 0xFFFFFFFF, near = large). Each
// fragment compares LINEAR depths with a small tolerance: a plate that
// hugs an opaque shell (the Vulture visor rides ~1e-3 off it) must not
// drop out where the shell's simplified LOD wobbles in front of it.
//
// Bindings (set 0) - VS-only buffers take the LOWEST numbers: the mixed
// layout reflects the FRAGMENT module, and binding gaps below its
// highest binding become STORAGE_BUFFER slots visible to the VS:
//   0: float4 verts[]     - 3 per vertex: (pos.xyz, materialId),
//                           (normal.xyz, 0), (uv, 0, 0)
//   1: float4 instances[] - 4 per instance: the model's glm columns
//   2: float4 materials[] - 2 per material (the .tmesh records):
//                           (kd.rgb, alpha), (roughness, metallic, 0, 0)
//   3: GpuLight lights[]  - lighting.slang's light table layout
//   4: float4 env[]       - lighting.slang's env table (SH at [2..10])
//   5: uint vis[]         - the VisBuffer as u32 pairs (lo payload, hi
//                           depth bits)
//   6: float4 sceneMaterials[] - the kgeo's RUNTIME material table
//                           (decode_material's GpuMaterial: kdTex = kd +
//                           albedo layer, params = rough, metal, MR
//                           layer) - the same ids; the texture layers the
//                           app resolved at load time live here
//   7: Sampler2DArray envMap - the IBL probe (layer 0)
//   8/9: Texture2DArray albedoTex + SamplerState albedoSampler - the
//                           scene's albedo/MR array (separate image +
//                           sampler: a second combined array sampler is
//                           the reflection-drop pattern noted in M8.2)
//
// Push (112B): viewProj columns | camPos + lightCount |
//   (width, height, near, far) | (envMaxLod, envIntensity,
//   sceneTableValid, 0)

struct TranslucentPush {
  float4 vp[4];
  float4 camPosCount;
  float4 viewport;  // w, h, near, far
  float4 envParams; // envMaxLod, envIntensity
};
[[vk::push_constant]] ConstantBuffer<TranslucentPush> push;

struct GpuLight {
  float4 a;
  float4 b;
};

[[vk::binding(0)]] StructuredBuffer<float4> verts;
[[vk::binding(1)]] StructuredBuffer<float4> instances;
[[vk::binding(2)]] StructuredBuffer<float4> materials;
[[vk::binding(3)]] StructuredBuffer<GpuLight> lights;
[[vk::binding(4)]] StructuredBuffer<float4> env;
[[vk::binding(5)]] StructuredBuffer<uint> vis;
[[vk::binding(6)]] StructuredBuffer<float4> sceneMaterials;
[[vk::binding(7)]] Sampler2DArray envMap;
[[vk::binding(8)]] Texture2DArray albedoTex;
[[vk::binding(9)]] SamplerState albedoSampler;

struct VSOut {
  float4 pos : SV_Position;
  float3 world : WORLDPOS;
  float3 nrm : NORMAL;
  float2 uv : TEXCOORD0;
  nointerpolation uint mat : MATERIAL;
};

[shader("vertex")]
VSOut vsMain(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
  const float4 p = verts[vid * 3u];
  const float4 n = verts[vid * 3u + 1u];
  const float4 t = verts[vid * 3u + 2u];
  const float4 c0 = instances[iid * 4u], c1 = instances[iid * 4u + 1u],
               c2 = instances[iid * 4u + 2u], c3 = instances[iid * 4u + 3u];
  VSOut o;
  o.world = (c0 * p.x + c1 * p.y + c2 * p.z + c3).xyz;
  // Uniform-scale instances (the demo's): the 3x3 itself, renormalized.
  o.nrm = normalize((c0 * n.x + c1 * n.y + c2 * n.z).xyz);
  o.pos = push.vp[0] * o.world.x + push.vp[1] * o.world.y +
          push.vp[2] * o.world.z + push.vp[3];
  o.mat = uint(p.w + 0.5);
  o.uv = t.xy;
  return o;
}

// ---- lighting.slang's BRDF (UE BRDF.ush), kept verbatim -------------
float distributionGGX(float ndh, float rough) {
  const float a = rough * rough;
  const float a2 = a * a;
  const float d = ndh * ndh * (a2 - 1.0) + 1.0;
  return a2 / max(3.14159265 * d * d, 1e-7);
}
float visSmithJointApprox(float a2, float ndv, float ndl) {
  const float a = sqrt(a2);
  const float v = ndl * (ndv * (1.0 - a) + a);
  const float l = ndv * (ndl * (1.0 - a) + a);
  return 0.5 / max(v + l, 1e-7);
}
float3 fresnelSchlick(float ct, float3 f0) {
  const float fc = pow(1.0 - ct, 5.0);
  return saturate(50.0 * f0.g) * fc + (1.0 - fc) * f0;
}
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

// Linear view depth from a ZO perspective NDC z (glm perspectiveRH_ZO:
// z = f/(f-n) * (1 - n/d)  ->  d = n f / (f - z (f - n))).
float linearDepth(float z) {
  const float n = push.viewport.z, f = push.viewport.w;
  return n * f / max(f - z * (f - n), 1e-6);
}

[shader("fragment")]
float4 fsMain(VSOut i) : SV_Target {
  // ---- occlusion against the opaque frame ------------------------------
  const uint2 pixel = uint2(i.pos.xy);
  const uint W = uint(push.viewport.x);
  const uint hi = vis[(pixel.y * W + pixel.x) * 2u + 1u];
  if (hi != 0u) {
    const float zOpaque = 1.0 - float(hi) / 4294967295.0;
    const float dOpaque = linearDepth(zOpaque);
    const float dSelf = linearDepth(i.pos.z);
    if (dSelf > dOpaque * 1.002 + 0.002) discard;
  }

  const float4 m0 = materials[i.mat * 2u];
  const float4 m1 = materials[i.mat * 2u + 1u];
  float3 albedo = m0.rgb;
  const float alpha = saturate(m0.a);
  float rough = m1.x;
  float metal = m1.y;
  if (push.envParams.z > 0.5) {
    // The opaque path's material chain (decode_material fsGBuffer):
    // factor x sRGB-decoded albedo texel; MR texel G = roughness, B =
    // metallic (linear data), factors multiply. Real derivatives here -
    // this is a forward pass, implicit LOD is valid.
    const float4 kdTex = sceneMaterials[i.mat * 2u];
    const float4 params = sceneMaterials[i.mat * 2u + 1u];
    albedo = kdTex.rgb;
    rough = params.x;
    metal = params.y;
    if (kdTex.a >= 0.0) {
      const float3 c =
          albedoTex.Sample(albedoSampler, float3(i.uv, kdTex.a)).rgb;
      albedo *= c * (c * (c * 0.305306011 + 0.682171111) + 0.012522878);
    }
    if (params.z >= 0.0) {
      const float3 mr =
          albedoTex.Sample(albedoSampler, float3(i.uv, params.z)).rgb;
      rough *= mr.g;
      metal *= mr.b;
    }
  }
  rough = clamp(rough, 0.04, 1.0);
  metal = saturate(metal);

  const float3 camPos = push.camPosCount.xyz;
  const float3 viewDir = normalize(camPos - i.world);
  float3 nrm = normalize(i.nrm);
  // Double-sided by construction (the pass culls nothing): shade the
  // side facing the viewer.
  if (dot(nrm, viewDir) < 0.0) nrm = -nrm;
  const float ndv = saturate(abs(dot(nrm, viewDir)) + 1e-5);
  const float3 f0 = lerp(float3(0.04, 0.04, 0.04), albedo, metal);
  const float3 diffuseColor = albedo * (1.0 - metal);

  float3 lo = float3(0.0);
  float3 specDirect = float3(0.0);  // the specular share of lo
  const uint lightCount = uint(push.camPosCount.w + 0.5);
  for (uint li = 0; li < lightCount; ++li) {
    const float4 meta = lights[li].a;
    float3 radiance = lights[li].b.rgb;
    float3 l;
    if (meta.w < 0.5) {
      l = normalize(meta.xyz);
    } else {
      const float3 toLight = meta.xyz - i.world;
      const float distL = length(toLight);
      l = toLight / max(distL, 1e-4);
      const float range = lights[li].b.w;
      const float att =
          range > 0.0 ? clamp(1.0 - pow(distL / range, 4.0), 0.0, 1.0) : 1.0;
      radiance *= att / max(distL * distL, 1e-4);
    }
    const float ndl = max(dot(nrm, l), 0.0);
    if (ndl <= 0.0) continue;
    const float3 h = normalize(viewDir + l);
    const float ndh = max(dot(nrm, h), 0.0);
    const float a2 = rough * rough * rough * rough;
    const float3 spec = distributionGGX(ndh, rough) *
                        visSmithJointApprox(a2, ndv, ndl) *
                        fresnelSchlick(max(dot(h, viewDir), 0.0), f0);
    lo += (diffuseColor / 3.14159265 + spec) * radiance * ndl;
    specDirect += spec * radiance * ndl;
  }

  // IBL: lighting.slang's probe sampling (mip from UE's roughness map).
  const float envIntensity = push.envParams.y;
  const float3 rdir = reflect(-viewDir, nrm);
  const float2 equi = float2(atan2(rdir.y, rdir.x) / 6.28318530718 + 0.5,
                             acos(clamp(rdir.z, -1.0, 1.0)) / 3.14159265359);
  const float maxLod = push.envParams.x;
  const float envMip = clamp(maxLod - 3.0 + 1.2 * log2(rough), 0.0, maxLod);
  const float3 rEnv = envMap.SampleLevel(float3(equi, 0.0), envMip).rgb;
  const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
  const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
  const float4 rr = float4(rough) * c0 + c1;
  const float a004 = min(rr.x * rr.x, exp2(-9.28 * ndv)) * rr.x + rr.y;
  const float2 ab = float2(-1.04, 1.04) * a004 + rr.zw;
  const float3 envSpec =
      (f0 * ab.x + saturate(50.0 * f0.g) * ab.y) * rEnv * envIntensity;
  const float3 ambient = diffuseColor * shIrradiance(nrm) * envIntensity;

  // Opacity scales the DIFFUSE body only; the specular reflection stays
  // whole (Filament's TRANSPARENT blending, what Sketchfab shows): a
  // clear visor at 53% opacity still mirrors its surroundings at full
  // Fresnel strength. UE's Translucent mode (everything x Opacity) made
  // this asset's roughness-0 gold-mirror visor a pale, flat film.
  const float3 diffuseLit = ambient + (lo - specDirect);
  const float3 specular = envSpec + specDirect;
  const float3 premul = diffuseLit * alpha + specular;
  if (!(premul.x >= 0.0 && premul.y >= 0.0 && premul.z >= 0.0 &&
        premul.x < 65504.0 && premul.y < 65504.0 && premul.z < 65504.0))
    discard;
  return float4(premul, alpha);
}
