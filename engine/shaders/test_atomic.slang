// S1.4 smoke: uint64 atomic max (the VisBuffer primitive) + atomic reduce.
// Expected with gInput = {1,2,3,4}:
//   gOutput[0] = 10   (sum via atomicAdd)
//   gAtomic[0] = 16   (max of v*v via atomicMax on a uint64 storage element)
[[vk::binding(0)]] RWStructuredBuffer<uint>     gInput;
[[vk::binding(1)]] RWStructuredBuffer<uint64_t> gAtomic;
[[vk::binding(2)]] RWStructuredBuffer<uint>     gOutput;

[shader("compute")]
[numthreads(4, 1, 1)]
void main(uint tid : SV_DispatchThreadID)
{
    const uint v = gInput[tid];
    InterlockedAdd(gOutput[0], v);
    InterlockedMax(gAtomic[0], (uint64_t)v * v);
}
