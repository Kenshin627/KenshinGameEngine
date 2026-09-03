// 链入帧化: convert a GPU-written work count into a dispatch command.
// One thread: reads src[0] (count), writes dst = {ceil(count/groups),1,1}
// (groups == 0 -> x = count, for one-workgroup-per-item passes) and
// optionally forwards the raw count into fwd[0] for the next pass's
// bound check.
//
// Bindings (SSBO set 0):
//   0: uint src[1]    - GPU-written count
//   1: uint cfg[4]    - [0] = group size (0 = one group per item)
//   2: uint dst[4]    - VkDispatchIndirectCommand {x,y,z} + pad
//   3: uint fwd[1]    - raw count forwarded (next pass's input guard)

[[vk::binding(0)]] StructuredBuffer<uint> src;
[[vk::binding(1)]] StructuredBuffer<uint> cfg;
[[vk::binding(2)]] RWStructuredBuffer<uint> dst;
[[vk::binding(3)]] RWStructuredBuffer<uint> fwd;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const uint count = src[0];
    const uint groups = cfg[0] == 0u ? count : (count + cfg[0] - 1u) / cfg[0];
    dst[0] = groups;
    dst[1] = 1u;
    dst[2] = 1u;
    fwd[0] = count;
}
