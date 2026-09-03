// M3.1: HZB pyramid - one reduction step per dispatch.
// The VisBuffer's winner depth bits ARE the per-pixel nearest-surface
// depth (a standard depth buffer in reversed-bit form); the pyramid
// min-reduces them 2x2 (min bits = FARTHEST depth), so a texel is a
// conservative "nothing farther renders here" bound. Empty pixels
// (word 0) reduce a tile to 0 = "occludes nothing" - the safe side.
//
// Bindings (SSBO set 0):
//   0: float4 params[2] - [0] = (srcW, srcH, srcOffset, mode)
//                          [1] = (dstOffset, dstW, dstH, unused)
//                          mode 0: src = VisBuffer u64 words (depth is
//                                  the HIGH word = uint index 2*i+1)
//                          mode 1: src = pyramid u32 texels at srcOffset
//   1: uint src[]
//   2: uint pyr[]      - output pyramid texels

[[vk::binding(0)]] StructuredBuffer<float4> params;
[[vk::binding(1)]] StructuredBuffer<uint>   src;
[[vk::binding(2)]] RWStructuredBuffer<uint> pyr;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint tid : SV_DispatchThreadID)
{
    const float4 p0 = params[0];
    const float4 p1 = params[1];
    const uint srcW = uint(p0.x), srcH = uint(p0.y);
    const uint srcOffset = uint(p0.z);
    const uint mode = uint(p0.w);
    const uint dstOffset = uint(p1.x);
    const uint dstW = uint(p1.y);
    const uint dstH = uint(p1.z);
    if (tid >= dstW * dstH) return;

    const uint dx = tid % dstW;
    const uint dy = tid / dstW;

    uint m = 0xFFFFFFFFu;
    [unroll]
    for (int j = 0; j < 2; ++j)
    {
        [unroll]
        for (int i = 0; i < 2; ++i)
        {
            // Clamp reads at odd edges (replicate) - conservative.
            const uint sx = min(dx * 2u + uint(i), srcW - 1u);
            const uint sy = min(dy * 2u + uint(j), srcH - 1u);
            const uint word = mode == 0u
                ? src[2u * (sy * srcW + sx) + 1u]
                : src[srcOffset + sy * srcW + sx];
            m = min(m, word);
        }
    }
    pyr[dstOffset + dy * dstW + dx] = m;
}
