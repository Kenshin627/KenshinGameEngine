// M4.2: convert a GPU-written cluster count into a draw command.
// count_to_indirect's draw twin: writes VkDrawIndirectCommand
// {vertexCount, instanceCount, firstVertex, firstInstance} - the HW
// pass draws a FIXED vertex count (3 * max tris per cluster; the VS
// degenerates out-of-range triangles) with one instance per routed
// cluster.
//
// Bindings (SSBO set 0):
//   0: uint src[1]   - GPU-written count (read at slot cfg[2])
//   1: uint cfg[4]   - [0] = vertex count, [2] = src read slot
//   2: uint dst[4]   - VkDrawIndirectCommand

[[vk::binding(0)]] StructuredBuffer<uint> src;
[[vk::binding(1)]] StructuredBuffer<uint> cfg;
[[vk::binding(2)]] RWStructuredBuffer<uint> dst;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    dst[0] = cfg[0];
    uint count = src[cfg[2]];
    if (cfg[3] > 0u) count = min(count, cfg[3]);
    dst[1] = count;
    dst[2] = 0u;
    dst[3] = 0u;
}
