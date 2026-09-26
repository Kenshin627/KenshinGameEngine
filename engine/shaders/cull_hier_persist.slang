// M5.3: the PERSISTENT-THREADS BVH8 traversal (Nanite's driver shape -
// one dispatch, no per-level drain). Every THREAD is a worker looping
// "take a node -> test -> push surviving children / emit the leaf's
// clusters" until the whole traversal completes. The per-node tests and
// the leaf's per-cluster body are the level shader's VERBATIM (M5.2
// differential already pinned them) - only the driver changed.
//
// THE QUEUE: an Add-only ticket MPMC. Tickets are handed out densely by
// InterlockedAdd on state.head; ticket i's entry lives at queueWords
// [2*(i & mask)] and has EXACTLY one producer and one consumer, so no
// CAS is ever needed - every operation is the proven 3-arg
// InterlockedAdd form:
//   read   : InterlockedAdd(word, 0, orig)
//   write  : InterlockedAdd(word, v + 1)     (word was EMPTY=0xFFFFFFFF;
//                                             0xFFFFFFFF + v + 1 == v)
//   consume: InterlockedAdd(word, EMPTY - v) (sole consumer re-empties)
// A worker holding a dead ticket (no producer will ever claim it) spins
// on the EMPTY slot until state.done flips, then exits. done is set by
// the worker whose outstanding decrement reaches zero: outstanding
// counts queued + in-processing nodes, and pushes increment it BEFORE
// their slot writes, so zero means no producer is left - the standard
// relaxed-atomics argument (same-location totality + per-location
// eventual visibility; Nanite's D12.4 "scheduler goodwill" caveat
// applies, as it does to UE's own implementation).
//
// Bindings (SSBO set 0):
//   0: float4 params[8]  - cull_clusters layout PLUS [1].y=nodeCount,
//      [1].w=queueMask (capacity-1), [7].z=clusterCount, [7].w=groupCount
//   1: GpuCluster clusters[]           (48B)
//   2: GpuGroup groups[]               (32B)
//   3: GpuInstance instances[]         (96B)
//   4: uint2 visiblePairs[]  - out (cull_clusters semantics)
//   5: uint counters[8]      - out, cull_clusters semantics
//   6: uint hzb[]             - PREV frame pyramid (occlusion)
//   7: float4 aabbs[]         - 3 per cluster (min, max, maxEdgeLen)
//   8: uint2 pendingPairs[]   - out (occlusion)
//   9: uint2 hwPairs[]        - out (M4.2 routing)
//  10: float4 frustumPlanes[6]
//  11: GpuHierNode hierNodes[]     - the M5.1 BVH8 (GPU repack, 80B)
//  12: uint2 groupRanges[]         - (clusterBegin, clusterCount)/group
//  13: uint queueWords[]   - RW: the ticket ring (2 words/entry)
//  14: uint state[8]       - RW: {head, tail, outstanding, done,
//                                 visited, pruned, pad, pad}

struct GpuCluster {
  float4 bounds;
  float error;
  float parentError;
  uint vertexOffset;
  uint vertexCount;
  uint indexOffset;
  uint indexCount;
  uint group;
  int refinedGroup;
};

struct GpuGroup {
  float4 bounds;
  float maxError;
  float simplifyError;
  uint lodLevel;
  float _pad4;
};

struct GpuInstance {
  float4x4 model;
  uint meshId;
  uint _pad0;
  uint _pad1;
  uint _pad2;
  float4 bounds;
};

struct GpuHierNode {
  float4 aabbMin;
  float4 aabbMax;
  float maxSimplifyError;
  uint childCount;
  uint children[8];
  uint _pad[2];
};

[[vk::binding(0)]] StructuredBuffer<float4> params;
[[vk::binding(1)]] StructuredBuffer<GpuCluster> clusters;
[[vk::binding(2)]] StructuredBuffer<GpuGroup> groups;
[[vk::binding(3)]] StructuredBuffer<GpuInstance> instances;
[[vk::binding(4)]] RWStructuredBuffer<uint2> visiblePairs;
[[vk::binding(5)]] RWStructuredBuffer<uint> counters;
[[vk::binding(6)]] StructuredBuffer<uint> hzb;
[[vk::binding(7)]] StructuredBuffer<float4> aabbs;
[[vk::binding(8)]] RWStructuredBuffer<uint2> pendingPairs;
[[vk::binding(9)]] RWStructuredBuffer<uint2> hwPairs;
[[vk::binding(10)]] StructuredBuffer<float4> frustumPlanes;
[[vk::binding(11)]] StructuredBuffer<GpuHierNode> hierNodes;
[[vk::binding(12)]] StructuredBuffer<uint2> groupRanges;
[[vk::binding(13)]] RWStructuredBuffer<uint> queueWords;
[[vk::binding(14)]] RWStructuredBuffer<uint> state;

static const uint kEmpty = 0xFFFFFFFFu;

bool fireable(uint groupIdx, float4x4 model) {
  const GpuGroup g = groups[groupIdx];
  // Instance-transformed decision sphere - explicit glm-column linear
  // combination (layout-proof; see cull_clusters.slang).
  const float3 c = model[0].xyz * g.bounds.x + model[1].xyz * g.bounds.y +
                   model[2].xyz * g.bounds.z + model[3].xyz;
  const float scale = max(length(model[0].xyz),
                          max(length(model[1].xyz), length(model[2].xyz)));
  const float dist = max(distance(c, params[0].xyz) - g.bounds.w * scale, 0.1);
  const float t = params[0].w * params[1].x * dist;
  return g.simplifyError > t;
}

bool occluded(uint2 item, float4x4 model) {
  const uint W = uint(params[6].y), H = uint(params[6].z);
  const float3 mn = aabbs[3u * item.y + 0u].xyz;
  const float3 mx = aabbs[3u * item.y + 1u].xyz;

  float4 clip[8];
  for (uint k = 0; k < 8u; ++k) {
    const float3 lc =
        float3((k & 1u) != 0u ? mx.x : mn.x, (k & 2u) != 0u ? mx.y : mn.y,
               (k & 4u) != 0u ? mx.z : mn.z);
    const float3 wc = model[0].xyz * lc.x + model[1].xyz * lc.y +
                      model[2].xyz * lc.z + model[3].xyz;
    clip[k] =
        params[2] * wc.x + params[3] * wc.y + params[4] * wc.z + params[5];
    if (clip[k].w <= 0.0) return false;
  }

  float zMin = clip[0].z / clip[0].w;
  float rx0 = 1e30f, rx1 = -1e30f, ry0 = 1e30f, ry1 = -1e30f;
  for (uint k = 0; k < 8u; ++k) {
    const float2 ndc = clip[k].xy / clip[k].w;
    const float sx = (ndc.x * 0.5 + 0.5) * float(W);
    const float sy = (ndc.y * 0.5 + 0.5) * float(H);
    rx0 = min(rx0, sx);
    rx1 = max(rx1, sx);
    ry0 = min(ry0, sy);
    ry1 = max(ry1, sy);
    zMin = min(zMin, clip[k].z / clip[k].w);
  }
  zMin = clamp(zMin, 0.0, 1.0);
  const uint nearBits = uint((1.0 - zMin) * 4294967295.0);

  const int x0 = int(floor(rx0)), x1 = int(ceil(rx1));
  const int y0 = int(floor(ry0)), y1 = int(ceil(ry1));
  if (x0 < 0 || y0 < 0 || x1 >= int(W) || y1 >= int(H)) return false;

  uint m = 0, lw = W, lh = H;
  while (((uint(x1) >> m) - (uint(x0) >> m) >= 4u ||
          (uint(y1) >> m) - (uint(y0) >> m) >= 4u) &&
         lw > 1u && lh > 1u) {
    ++m;
    lw = (lw + 1u) / 2u;
    lh = (lh + 1u) / 2u;
  }
  uint off = 0, ww = W, hh = H;
  for (uint k = 0; k < m; ++k) {
    off += ww * hh;
    ww = (ww + 1u) / 2u;
    hh = (hh + 1u) / 2u;
  }

  const uint tx0 = uint(x0) >> m, tx1 = uint(x1) >> m;
  const uint ty0 = uint(y0) >> m, ty1 = uint(y1) >> m;
  uint tileMin = 0xFFFFFFFFu;
  for (uint ty = ty0; ty <= ty1; ++ty)
    for (uint tx = tx0; tx <= tx1; ++tx)
      tileMin =
          min(tileMin, hzb[off + min(ty, lh - 1u) * lw + min(tx, lw - 1u)]);

  return nearBits < tileMin;
}

/// The cull_clusters.slang per-cluster body, VERBATIM (identical
/// counters/pairs semantics - the three-mode differential pins this).
void emitCluster(uint2 item, float4x4 model) {
  const GpuCluster cluster = clusters[item.y];

  if (!fireable(cluster.group, model)) {
    InterlockedAdd(counters[1], 1);
    return;
  }
  if (cluster.refinedGroup >= 0 && fireable(uint(cluster.refinedGroup), model)) {
    InterlockedAdd(counters[2], 1);
    return;
  }

  if (params[6].x > 0.5 && occluded(item, model)) {
    uint pend = 0;
    InterlockedAdd(counters[4], 1, pend);
    pendingPairs[pend] = item;
    return;
  }

  if (params[7].x > 0.5) {
    bool hw = params[7].x > 1.5;
    if (!hw) {
      const float maxScale = max(length(model[0].xyz),
                                 max(length(model[1].xyz),
                                     length(model[2].xyz)));
      const float3 c = model[0].xyz * cluster.bounds.x +
                       model[1].xyz * cluster.bounds.y +
                       model[2].xyz * cluster.bounds.z + model[3].xyz;
      const float sphereR = cluster.bounds.w * maxScale;
      const float dist = max(distance(c, params[0].xyz) - sphereR, 0.1);
      if (dist < sphereR + params[1].z) hw = false;
      // [6].w > 0 = perspective tanHalf; < 0 = -extentY for ortho
      // chains (shadow clipmaps) - px/unit is distance-free there
      // (dividing by dist never routed: the M9.1 shadow hw=0 case).
      const float pxPerUnit =
          params[6].w > 0.0
              ? (params[6].z * 0.5) / (max(params[6].w, 1e-6) * dist)
              : (params[6].z * 0.5) / max(-params[6].w, 1e-6);
      const float edgePx = aabbs[3u * item.y + 2u].x * maxScale * pxPerUnit;
      hw = hw && edgePx >= params[7].y;
    }
    if (hw) {
      uint hs = 0;
      InterlockedAdd(counters[7], 1, hs);
      hwPairs[hs] = item;
      InterlockedAdd(counters[6], cluster.indexCount / 3u);
      return;
    }
  }

  uint slot = 0;
  InterlockedAdd(counters[0], 1, slot);
  visiblePairs[slot] = item;
  InterlockedAdd(counters[6], cluster.indexCount / 3u);
}

/// Node-AABB occlusion test (M5.4a) against the PREV frame's HZB - the
/// same conservative rect walk as occluded(), fed already-computed
/// world-space corners. Dominance: a box tested occluded implies every
/// contained box tests occluded (child zMin >= parent zMin -> nearBits
/// only smaller; child rect subset -> tileMin only larger), so the bit
/// can be INHERITED down the subtree without retesting.
bool boxOccluded(float3 corner[8]) {
  const uint W = uint(params[6].y), H = uint(params[6].z);
  float4 clip[8];
  for (uint k = 0; k < 8u; ++k) {
    clip[k] = params[2] * corner[k].x + params[3] * corner[k].y +
              params[4] * corner[k].z + params[5];
    if (clip[k].w <= 0.0) return false;
  }
  float zMin = clip[0].z / clip[0].w;
  float rx0 = 1e30f, rx1 = -1e30f, ry0 = 1e30f, ry1 = -1e30f;
  for (uint k = 0; k < 8u; ++k) {
    const float2 ndc = clip[k].xy / clip[k].w;
    rx0 = min(rx0, (ndc.x * 0.5 + 0.5) * float(W));
    rx1 = max(rx1, (ndc.x * 0.5 + 0.5) * float(W));
    ry0 = min(ry0, (ndc.y * 0.5 + 0.5) * float(H));
    ry1 = max(ry1, (ndc.y * 0.5 + 0.5) * float(H));
    zMin = min(zMin, clip[k].z / clip[k].w);
  }
  zMin = clamp(zMin, 0.0, 1.0);
  const uint nearBits = uint((1.0 - zMin) * 4294967295.0);
  const int x0 = int(floor(rx0)), x1 = int(ceil(rx1));
  const int y0 = int(floor(ry0)), y1 = int(ceil(ry1));
  if (x0 < 0 || y0 < 0 || x1 >= int(W) || y1 >= int(H)) return false;
  uint m = 0, lw = W, lh = H;
  while (((uint(x1) >> m) - (uint(x0) >> m) >= 4u ||
          (uint(y1) >> m) - (uint(y0) >> m) >= 4u) &&
         lw > 1u && lh > 1u) {
    ++m;
    lw = (lw + 1u) / 2u;
    lh = (lh + 1u) / 2u;
  }
  uint off = 0, ww = W, hh = H;
  for (uint k = 0; k < m; ++k) {
    off += ww * hh;
    ww = (ww + 1u) / 2u;
    hh = (hh + 1u) / 2u;
  }
  const uint tx0 = uint(x0) >> m, tx1 = uint(x1) >> m;
  const uint ty0 = uint(y0) >> m, ty1 = uint(y1) >> m;
  uint tileMin = 0xFFFFFFFFu;
  for (uint ty = ty0; ty <= ty1; ++ty)
    for (uint tx = tx0; tx <= tx1; ++tx)
      tileMin =
          min(tileMin, hzb[off + min(ty, lh - 1u) * lw + min(tx, lw - 1u)]);
  return nearBits < tileMin;
}

/// One worker's node visit. Returns the number of children to push;
/// childOcc tells the caller whether they INHERIT the occluded bit
/// (either inherited from above, or set by this node's own AABB test -
/// the ROOT node's test plays the instance-level pre-pass).
uint visitNode(uint inst, uint nodeIdx, float4x4 model, uint nodeCount,
               uint clusterCount, uint groupCount, bool inherited,
               out bool childOcc) {
  childOcc = inherited;
  InterlockedAdd(state[4], 1);  // visited (M5.5 curves)
  const GpuHierNode node = hierNodes[min(nodeIdx, nodeCount - 1u)];

  // World-space corners of the node AABB under the instance transform.
  float3 corner[8];
  for (uint k = 0; k < 8u; ++k) {
    const float3 lc =
        float3((k & 1u) != 0u ? node.aabbMax.x : node.aabbMin.x,
               (k & 2u) != 0u ? node.aabbMax.y : node.aabbMin.y,
               (k & 4u) != 0u ? node.aabbMax.z : node.aabbMin.z);
    corner[k] = model[0].xyz * lc.x + model[1].xyz * lc.y +
                model[2].xyz * lc.z + model[3].xyz;
  }

  // Conservative frustum prune: one plane with ALL corners outside.
  for (uint p = 0; p < 6u; ++p) {
    bool allOutside = true;
    for (uint k = 0; k < 8u; ++k) {
      if (dot(frustumPlanes[p].xyz, corner[k]) + frustumPlanes[p].w >= 0.0) {
        allOutside = false;
        break;
      }
    }
    if (allOutside) {
      InterlockedAdd(state[5], 1);  // pruned
      return 0u;
    }
  }

  // LOD prune: the subtree is uniformly "too fine" for this view.
  float3 wmn = corner[0], wmx = corner[0];
  for (uint k = 1u; k < 8u; ++k) {
    wmn = min(wmn, corner[k]);
    wmx = max(wmx, corner[k]);
  }
  const float3 nearest = clamp(params[0].xyz, wmn, wmx);
  const float dist = max(distance(nearest, params[0].xyz), 0.1);
  const float tNode = params[0].w * params[1].x * dist;
  if (node.maxSimplifyError <= tNode) {
    InterlockedAdd(state[5], 1);  // pruned
    return 0u;
  }

  if (node.childCount == 0u) {
    // Leaf: one group; its clusters are a contiguous range.
    const uint g = min(node.children[0], groupCount - 1u);
    const uint2 range = groupRanges[g];
    const uint begin = min(range.x, clusterCount);
    const uint end = min(range.x + range.y, clusterCount);
    if (inherited) {
      // M5.4a: dominance proved every cluster here tests occluded - the
      // per-cluster rect walk is skipped and the LOD-selected clusters
      // go straight to pending (the exact set the per-cluster tests
      // would have pended; counters[6]/HW routing intentionally NOT
      // touched, matching the occluded branch of emitCluster).
      for (uint c = begin; c < end; ++c) {
        if (!fireable(clusters[c].group, model)) continue;
        if (clusters[c].refinedGroup >= 0 &&
            fireable(uint(clusters[c].refinedGroup), model))
          continue;
        uint p = 0;
        InterlockedAdd(counters[4], 1, p);
        pendingPairs[p] = uint2(inst, c);
      }
      return 0u;
    }
    for (uint c = begin; c < end; ++c) emitCluster(uint2(inst, c), model);
    return 0u;
  }

  // Internal survivor: the node's own AABB occlusion test sets the
  // inherited bit for the whole subtree.
  if (!inherited && params[6].x > 0.5 && boxOccluded(corner))
    childOcc = true;
  return min(node.childCount, 8u);
}

/// Push one entry: outstanding++ FIRST, then reserve the tail index,
/// then make the words visible (Add onto the EMPTY sentinel).
void pushEntry(uint inst, uint node, uint mask) {
  InterlockedAdd(state[2], 1);
  uint t = 0;
  InterlockedAdd(state[1], 1, t);
  const uint w = 2u * (t & mask);
  InterlockedAdd(queueWords[w + 0u], inst + 1u);
  InterlockedAdd(queueWords[w + 1u], node + 1u);
}

[shader("compute")][numthreads(64, 1, 1)] void main(
    uint tid : SV_DispatchThreadID, uint gtid : SV_GroupThreadID) {
  // ONE WORKER PER GROUP (lane 0). A divergent spin pair inside one
  // warp deadlocks on this hardware: a lane waiting for a producer
  // while warp-mates spin on tickets only THAT lane can fulfill - the
  // SIMT scheduler starves the producing branch (post-mortem: head
  // fully advanced, zero items consumed). Exited lanes never block
  // their warp, so lane 0's spins are safe. Group-cooperative work
  // (the Nanite shape: groupshared staging + barriers) is a later
  // optimization; correctness first.
  if (gtid != 0u) return;
  const uint nodeCount = uint(params[1].y);
  const uint mask = uint(params[1].w);
  const uint clusterCount = uint(params[7].z);
  const uint groupCount = uint(params[7].w);

  for (;;) {
    // Take a ticket. Tickets are dense; ticket i's entry is produced by
    // exactly one push (seed or a visitor) - commit and wait for it.
    uint my = 0;
    InterlockedAdd(state[0], 1, my);

    // Wait until a producer has claimed this ticket's index (my < tail)
    // BEFORE touching the ring: idle threads eagerly hold tickets beyond
    // the current tail, and my & mask would wrap onto REAL slots if we
    // read the words unconditionally (total tickets handed out can far
    // exceed the ring capacity; only tickets < tail ever have an item).
    // tail is monotonic, and pushes reserve their tail index BEFORE the
    // outstanding that keeps the frame alive - so tail passing my means
    // the item IS coming (unless done: no producer is left).
    for (;;) {
      uint t = 0;
      InterlockedAdd(state[1], 0u, t);
      if (my < t) break;
      uint dn = 0;
      InterlockedAdd(state[3], 0u, dn);
      if (dn != 0u) return;
    }

    // Wait for the push's words to become visible (producer reserved the
    // index, the two Adds may still be in flight).
    const uint w = 2u * (my & mask);
    uint inst = kEmpty, nodeIdx = kEmpty;
    for (;;) {
      uint lo = 0;
      InterlockedAdd(queueWords[w + 0u], 0u, lo);
      if (lo != kEmpty) {
        uint hi = 0;
        InterlockedAdd(queueWords[w + 1u], 0u, hi);
        if (hi != kEmpty) {
          inst = lo;
          nodeIdx = hi;
          break;
        }
      }
      uint dn = 0;
      InterlockedAdd(state[3], 0u, dn);
      if (dn != 0u) return;
    }

    // Consume: re-empty the words (sole consumer of this ticket's slot).
    InterlockedAdd(queueWords[w + 0u], kEmpty - inst);
    InterlockedAdd(queueWords[w + 1u], kEmpty - nodeIdx);

    // M5.4a: the node word packs the INHERITED-OCCLUDED bit in 31.
    const bool inherited = (nodeIdx & 0x80000000u) != 0u;
    const uint nodeOnly = min(nodeIdx & 0x7FFFFFFFu, nodeCount - 1u);

    const float4x4 model = instances[inst].model;
    const GpuHierNode node = hierNodes[nodeOnly];
    bool childOcc = false;
    const uint nChildren =
        visitNode(inst, nodeOnly, model, nodeCount, clusterCount, groupCount,
                  inherited, childOcc);
    for (uint i = 0; i < nChildren; ++i)
      pushEntry(inst, node.children[i] | (childOcc ? 0x80000000u : 0u), mask);

    // retire: the LAST retiree (outstanding was exactly 1) sets done.
    uint last = 0;
    InterlockedAdd(state[2], 0xFFFFFFFFu, last);
    if (last == 1u) InterlockedAdd(state[3], 1u);
  }
}
