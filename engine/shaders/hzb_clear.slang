// M4 follow-up: compute-side pyramid clear (replaces the TRANSFER
// fillBuffer - transfer ops serialize the pipeline with cross-stage
// barriers and are disproportionately expensive under validation).
// ONE thread per pyramid texel writes the atomic climb's contract
// value (ancestors start at max so every write wins); dispatched
// INDIRECT with the pending-gated flag args (0 groups when nothing
// was occluded - the common case) or direct for the always-on final
// build. Extra groups guard out on tid < totalTexels.
//
// Bindings (SSBO set 0):
//   0: float4 params[1] - (width, height, 0, 0)
//   1: uint pyr[]       - RW pyramid to clear

[[vk::binding(0)]] StructuredBuffer<float4> params;
[[vk::binding(1)]] RWStructuredBuffer<uint> pyr;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint tid : SV_DispatchThreadID)
{
    const uint w = uint(params[0].x);
    const uint h = uint(params[0].y);
    // Total texels along the SAME ceil-division chain as hzbLayout().
    uint total = w * h;
    uint lw = w, lh = h;
    while (lw > 1u || lh > 1u)
    {
        lw = (lw + 1u) / 2u;
        lh = (lh + 1u) / 2u;
        total += lw * lh;
    }
    if (tid < total) pyr[tid] = 0xFFFFFFFFu;
}
