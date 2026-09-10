// M5.3: persistent-mode seed - initializes the MPMC ticket queue from the
// visible instances. Plain writes (separate dispatch + bufferBarrier
// before the persistent dispatch orders them).
//
// Bindings (SSBO set 0):
//   0: uint visibleList[]      - in: visible instance indices
//   1: GpuInstance instances[] - in: meshId per instance
//   2: uint meshHierRoots[]    - in: root node index per mesh
//   3: uint queueWords[]       - out: entry i at [2i]=(inst, [2i+1]=node
//   4: uint state[8]           - out: {head=0, tail=count, outstanding=count,
//                                 done=(count==0), visited, pruned, pad, pad}
//   5: uint guardCnt[2]        - in: [1] = visible count (the guard)

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
[[vk::binding(3)]] RWStructuredBuffer<uint> queueWords;
[[vk::binding(4)]] RWStructuredBuffer<uint> state;
[[vk::binding(5)]] StructuredBuffer<uint> guardCnt;

[shader("compute")][numthreads(64, 1, 1)] void main(
    uint tid : SV_DispatchThreadID) {
  const uint count = guardCnt[1];
  // State rewrite EVERY frame (full init - no CPU memset needed). The
  // zero-visible-instance edge case sets done directly: otherwise the
  // persistent workers would wait forever on tickets that never arrive.
  if (tid == 0u) {
    state[0] = 0u;  // head: next ticket to hand out
    state[1] = count;  // tail: next push index
    state[2] = count;  // outstanding: queued + in-processing
    state[3] = count == 0u ? 1u : 0u;  // done flag
    state[4] = 0u;  // visited nodes (M5.5 panel curves)
    state[5] = 0u;  // pruned nodes
    state[6] = 0u;
    state[7] = 0u;
  }
  if (tid >= count) return;
  // Indices come from trusted uploaded tables (the cull_clusters pattern).
  const uint inst = visibleList[tid];
  const uint root = meshHierRoots[instances[inst].meshId];
  // ATOMIC writes (the Add-onto-EMPTY idiom): the persistent shader
  // reads these words with atomic RMWs, and plain stores on this
  // device-local buffer were never observed by them (post-mortem:
  // head advanced, the word spin never saw the value). Atomics on both
  // sides share one coherent path. 0xFFFFFFFF + v + 1 == v (mod 2^32).
  InterlockedAdd(queueWords[2u * tid + 0u], inst + 1u);
  InterlockedAdd(queueWords[2u * tid + 1u], root + 1u);
}
