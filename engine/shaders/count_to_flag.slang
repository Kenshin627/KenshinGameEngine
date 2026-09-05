// M4 follow-up: convert a GPU-written count into a FLAGGED dispatch
// size - full size when the count is nonzero, ZERO when it is empty
// (the pending-gated post section: nothing occluded -> skip the
// preliminary HZB build, post cull and post raster entirely).
//
// Bindings (SSBO set 0):
//   0: uint src[1]  - the count (read at slot cfg[2])
//   1: uint cfg[4]  - [0] = dispatch size when nonzero, [2] = src slot
//   2: uint dst[4]  - VkDispatchIndirectCommand

[[vk::binding(0)]] StructuredBuffer<uint> src;
[[vk::binding(1)]] StructuredBuffer<uint> cfg;
[[vk::binding(2)]] RWStructuredBuffer<uint> dst;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    dst[0] = src[cfg[2]] > 0u ? cfg[0] : 0u;
    dst[1] = 1u;
    dst[2] = 1u;
}
