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

/// Implicit-derivative sample (forward passes).
float4 sampleTexture(uint ref, float2 uv) {
  const uint slot = ref & 0xFFFFFFu;
  const uint sampler = ref >> 24;
  return gTextures[NonUniformResourceIndex(slot)].Sample(
      gSamplers[NonUniformResourceIndex(sampler)], uv);
}
