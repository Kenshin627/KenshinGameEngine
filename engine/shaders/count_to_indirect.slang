// 链入帧化: convert a GPU-written work count into a dispatch command.
// One thread: reads src[cfg[2]] (count, default slot 0), writes dst =
// {ceil(count/groups),1,1} (groups == 0 -> one group per item, for
// workgroup-per-item passes) and forwards the raw count into
// fwd[cfg[1]] (the next pass's input guard slot - its counters buffer
// is bound as fwd). cfg[2] (M3.3) lets one counters buffer feed
// several sites (e.g. the post pass reads the cull counters' [5]).
//
// Bindings (SSBO set 0):
//   0: uint src[1]    - GPU-written count (read at slot cfg[2])
//   1: uint cfg[4]    - [0] = group size (0 = one group per item)
//                       [1] = fwd write slot
//                       [2] = src read slot
//   2: uint dst[4]    - VkDispatchIndirectCommand {x,y,z} + pad
//   3: uint fwd[]     - the NEXT pass's counters buffer

[[vk::binding(0)]] StructuredBuffer<uint> src;
[[vk::binding(1)]] StructuredBuffer<uint> cfg;
[[vk::binding(2)]] RWStructuredBuffer<uint> dst;
[[vk::binding(3)]] RWStructuredBuffer<uint> fwd;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint count = src[cfg[2]];
    // cfg[3] > 0: the destination list's capacity (saturating writers
    // may have reserved past it - the dispatch envelope must clamp).
    if (cfg[3] > 0u) count = min(count, cfg[3]);
    const uint groups = cfg[0] == 0u ? count : (count + cfg[0] - 1u) / cfg[0];
    dst[0] = groups;
    dst[1] = 1u;
    dst[2] = 1u;
    fwd[cfg[1]] = count;
}
