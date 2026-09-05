// M4 follow-up: level-by-level HZB reduction (replaces the atomic
// climb - at 640x461 EVERY one of the ~295k L0 threads climbed to
// the 1x1 apex with InterlockedMin on the SAME address: ~295k
// serialized same-address atomics = multiple MILLISECONDS, the real
// cost behind "occlusion halves the framerate"). One dispatch per
// level; every texel has EXACTLY ONE writer (plain store, no
// atomics, NO pyramid clear needed).
//
// params[0] = (childW, childH, childOffset, dstOffset); mode via
// childW == 0: L0 pass reads the VisBuffer's depth words instead.
//
// Bindings (SSBO set 0):
//   0: float4 params[1]
//   1: uint visWords[] - the u64 VisBuffer (L0 mode only)
//   2: uint pyr[]      - RW pyramid (read children, write dst)

[[vk::binding(0)]] StructuredBuffer<float4> params;
[[vk::binding(1)]] StructuredBuffer<uint> visWords;
[[vk::binding(2)]] RWStructuredBuffer<uint> pyr;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint tid : SV_DispatchThreadID)
{
    const uint childW = uint(params[0].x);
    const uint childH = uint(params[0].y);
    const uint childOff = uint(params[0].z);
    const uint dstOff = uint(params[0].w);
    const uint dstW = (childW + 1u) / 2u;

    if (childW == 0u)
    {
        // L0: the VisBuffer's high words, one texel per pixel.
        pyr[dstOff + tid] = visWords[2u * tid + 1u];
        return;
    }
    const uint dstH = (childH + 1u) / 2u;
    if (tid >= dstW * dstH) return;
    const uint dx = tid % dstW;
    const uint dy = tid / dstW;
    uint m = 0xFFFFFFFFu;
    for (uint j = 0u; j < 2u; ++j)
        for (uint i = 0u; i < 2u; ++i)
        {
            const uint sx = min(dx * 2u + i, childW - 1u);
            const uint sy = min(dy * 2u + j, childH - 1u);
            m = min(m, pyr[childOff + sy * childW + sx]);
        }
    pyr[dstOff + dy * dstW + dx] = m;
}
