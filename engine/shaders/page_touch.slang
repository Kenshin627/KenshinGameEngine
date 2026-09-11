// M6.3: page residency feedback. One thread per (instance, cluster)
// pair of a rasterized list (the SW visiblePairs and the HW hwPairs -
// dispatched once per list per frame by the chain): mark the pair's
// page in the touched bitmap. The streaming manager reads the bitmap
// (mapped, non-blocking, the readStatsMapped pacing contract) to
// refresh LRU timestamps AND to detect misses (touched page with no
// resident slot -> request it). Dispatched ONLY by paged chains - the
// clusterPage read is unconditional and v2's dummy table would read
// out of bounds.
//
// Bindings (SSBO set 0):
//   0: uint2 pairs[]        - (instanceIdx, clusterIdx); one per group
//   1: uint  clusterPage[]  - cluster -> page
//   2: uint  touchedBitmap[] (RW) - one bit per page, word per 32

[[vk::binding(0)]] StructuredBuffer<uint2> pairs;
[[vk::binding(1)]] StructuredBuffer<uint> clusterPage;
[[vk::binding(2)]] RWStructuredBuffer<uint> touchedBitmap;

[numthreads(128, 1, 1)]
void main(uint3 gid : SV_GroupThreadID, uint3 group : SV_GroupID) {
  // The indirect dispatch x = the pair COUNT: 128 threads per group,
  // one thread per pair - the group index alone selects the pair.
  const uint2 pair = pairs[group.x];
  const uint page = clusterPage[pair.y];
  InterlockedOr(touchedBitmap[page >> 5], 1u << (page & 31));
}
