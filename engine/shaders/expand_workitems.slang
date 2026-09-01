// S4.3 phase 1: expand visible instances x their meshes' cluster ranges
// into (instanceIdx, clusterIdx) work items. One thread per VISIBLE
// instance; each appends its whole contiguous cluster span with one
// atomic reservation. v1 scan/compaction shape - the persistent-traversal
// rewrite in a later milestone removes this pass.
//
// Bindings (SSBO set 0; buffers[i] -> binding i):
//   0: uint2  workItems[]   - output pairs (instanceIdx, clusterIdx)
//   1: uint   counters[2]   - [0] write cursor (zero-init, becomes total)
//                             [1] visible instance count (input)
//   2: uint   visibleList[] - indices into instances[]
//   3: GpuInstance instances[]   (96B, mirrors cull_instances.slang)
//   4: uint2  meshRanges[]   - per mesh: (clusterBegin, clusterEnd)

struct GpuInstance
{
    float4x4 model;
    uint meshId;
    uint _pad0; uint _pad1; uint _pad2;
    float4 bounds;
};

[[vk::binding(0)]] RWStructuredBuffer<uint2> workItems;
[[vk::binding(1)]] RWStructuredBuffer<uint>  counters;
[[vk::binding(2)]] StructuredBuffer<uint>    visibleList;
[[vk::binding(3)]] StructuredBuffer<GpuInstance> instances;
[[vk::binding(4)]] StructuredBuffer<uint2>   meshRanges;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint tid : SV_DispatchThreadID)
{
    if (tid >= counters[1]) return;
    const uint instanceIdx = visibleList[tid];
    const uint2 range = meshRanges[instances[instanceIdx].meshId];
    const uint count = range.y - range.x;

    // One atomic reservation per instance; InterlockedAdd returns the
    // ORIGINAL value = this instance's base slot.
    uint base = 0;
    InterlockedAdd(counters[0], count, base);
    for (uint j = 0; j < count; ++j)
        workItems[base + j] = uint2(instanceIdx, range.x + j);
}
