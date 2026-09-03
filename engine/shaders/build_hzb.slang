// M3.2: HZB pyramid - single dispatch, atomic climb.
// Each thread owns one base-level texel: extract its depth word from
// the vis and InterlockedMin it into EVERY ancestor up the 2x2 mip
// chain (floor(x/2) mapping covers the same leaves a clamped-2x2
// chain reads) - texels converge with no inter-level barriers. The
// pyramid buffer must be PRE-FILLED with 0xFFFFFFFF.
//
// Bindings (SSBO set 0):
//   0: float4 params[1] - (width, height, 0, 0)
//   1: uint visWords[]  - the u64 VisBuffer (depth = HIGH word =
//                         uint index 2*i+1)
//   2: uint pyr[]       - RW pyramid (level offsets follow the same
//                         ceil-division chain as hzbLayout())

[[vk::binding(0)]] StructuredBuffer<float4> params;
[[vk::binding(1)]] StructuredBuffer<uint>   visWords;
[[vk::binding(2)]] RWStructuredBuffer<uint> pyr;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint tid : SV_DispatchThreadID)
{
    const uint w = uint(params[0].x);
    const uint h = uint(params[0].y);
    if (tid >= w * h) return;

    const uint m = visWords[2u * tid + 1u];

    uint off = 0, lw = w, lh = h, lx = tid % w, ly = tid / w;
    while (true)
    {
        if (off == 0u)
            pyr[ly * lw + lx] = m;         // L0: exactly one writer
        else
            InterlockedMin(pyr[off + ly * lw + lx], m);  // ancestors
        if (lw == 1u && lh == 1u) break;
        const uint nextOff = off + lw * lh;
        lx >>= 1u;
        ly >>= 1u;
        lw = (lw + 1u) / 2u;
        lh = (lh + 1u) / 2u;
        off = nextOff;
    }
}
