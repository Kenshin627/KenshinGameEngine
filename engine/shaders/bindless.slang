// bindless.slang - the material texture table (roadmap 1.3a).
//
// Descriptor set 1 of every pipeline that samples material textures
// (backend/vulkan/BindlessTable): every texture in its real format
// (colour textures in *_SRGB formats, so samples come back LINEAR) and
// the fixed sampler table. A material slot holds a 32-bit reference:
// table slot (low 24 bits) | sampler index (high 8 bits) - the layout
// of kenshin/backend/TextureFormat.h's textureRef(); keep them in sync.
// 0xFFFFFFFF = no texture.

[[vk::binding(0, 1)]] Texture2D gTextures[];
[[vk::binding(1, 1)]] SamplerState gSamplers[18];

static const uint kNoTexture = 0xFFFFFFFFu;

bool hasTexture(uint ref) { return ref != kNoTexture; }

/// Explicit-gradient sample (deferred decode: no pixel quads, the UV
/// footprint comes from the triangle). Hardware picks the mip and the
/// anisotropic footprint per texture.
float4 sampleTextureGrad(uint ref, float2 uv, float2 ddx, float2 ddy) {
  const uint slot = ref & 0xFFFFFFu;
  const uint sampler = ref >> 24;
  return gTextures[NonUniformResourceIndex(slot)].SampleGrad(
      gSamplers[NonUniformResourceIndex(sampler)], uv, ddx, ddy);
}

/// Specular anti-aliasing from normal-map variance (Toksvig, in the
/// vMF form of Neubelt & Pettineo 2013). The texture cooker writes into
/// a normal map's ALPHA the length r of the average of the unit normals
/// each texel covers, encoded a = 1 - sqrt(1 - r) (resolution where it
/// matters, near r = 1; mip 0 is a = 1). A footprint that averages
/// disagreeing normals is rougher than any of them: GGX alpha^2 widens
/// by 2 / kappa, kappa = (3r - r^3) / (1 - r^2). Takes and returns
/// PERCEPTUAL roughness (alpha = rough^2).
float normalVarianceRoughness(float rough, float encodedLength) {
  const float s = 1.0 - saturate(encodedLength);
  const float r = 1.0 - s * s;
  if (r >= 0.9999) return rough;  // mip 0 / flat footprint: unchanged
  const float rc = max(r, 1e-3);
  const float kappa = (3.0 * rc - rc * rc * rc) / (1.0 - rc * rc);
  const float a = rough * rough;
  const float a2 = saturate(a * a + 2.0 / kappa);
  return sqrt(sqrt(a2));
}

/// Implicit-derivative sample (forward passes).
float4 sampleTexture(uint ref, float2 uv) {
  const uint slot = ref & 0xFFFFFFu;
  const uint sampler = ref >> 24;
  return gTextures[NonUniformResourceIndex(slot)].Sample(
      gSamplers[NonUniformResourceIndex(sampler)], uv);
}
