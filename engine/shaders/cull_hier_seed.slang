// M5.2: hierarchy seed - level 0 of the level-wise BVH8 traversal.
// One thread per VISIBLE instance (the argsExpand envelope from the
// instance cull's conversion): map it to its mesh's hierarchy ROOT and
// write the first active-node entry. Pure sequential writes, no
// atomics - the entry count equals the visible-instance count already
// forwarded into expCnt[1] by convert #0, which is exactly what the
// level-1 dispatch conversion reads.
//
// Bindings (SSBO set 0):
//   0: uint visibleList[]      - in: visible instance indices
//   1: GpuInstance instances[] - in: meshId per instance
//   2: uint meshHierRoots[]    - in: root node index per mesh
//   3: uint2 activeNodes[]     - out: (instanceIdx, nodeIdx) level-0 list
//   4: uint guardCnt[2]        - in: [1] = visible count (the guard)
//   5: uint cntOut[1]           - out (thread 0): the level-1 input
//                                 count mirrored from the guard - the
//                                 level shader reads its guard here

struct GpuInstance {
  float4x4 model;
  uint meshId;
  uint _pad0;
  uint _pad1;
  uint _pad2;
  float4 bounds;
};

[[vk::binding(0)]] StructuredBuffer<uint> visibleList;
[[vk::binding(1)]] StructuredBuffer<GpuInstance> instances;
[[vk::binding(2)]] StructuredBuffer<uint> meshHierRoots;
[[vk::binding(3)]] RWStructuredBuffer<uint2> activeNodes;
[[vk::binding(4)]] StructuredBuffer<uint> guardCnt;
[[vk::binding(5)]] RWStructuredBuffer<uint> cntOut;

[shader("compute")][numthreads(64, 1, 1)] void main(
    uint tid : SV_DispatchThreadID) {
  if (tid >= guardCnt[1]) return;
  // Indices come from trusted uploaded tables (the cull_clusters pattern
  // for workItems-derived indices).
  const uint inst = visibleList[tid];
  activeNodes[tid] = uint2(inst, meshHierRoots[instances[inst].meshId]);
  if (tid == 0u) cntOut[0] = guardCnt[1];
}
