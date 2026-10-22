// M5.2: one LEVEL of the level-wise BVH8 traversal - the correctness
// stepping stone for the persistent-threads version (M5.3; Nanite's
// "three sins" of per-level dispatch get solved there). One thread per
// ACTIVE NODE (instanceIdx, nodeIdx); survivors' children are appended
// to the next level's list, leaves emit through the EXACT per-cluster
// body of cull_clusters.slang (copied verbatim - the LOD/occlusion/HW
// semantics and counters[0..7] meanings are identical, which is what
// the dual-mode byte-exact differential pins).
//
// Node tests, in order (both CONSERVATIVE - they may only drop nodes
// whose clusters could never write a pixel nor differ from the linear
// path's counters-feeding output):
//   1. Frustum: all 8 world AABB corners behind ONE plane -> drop.
//      (The linear path has no cluster-level frustum test, but a fully
//      outside cluster rasterizes an empty screen rect, so the final
//      VisBuffer is unchanged. The set-equality test frames the mesh.)
//   2. LOD prune (ShouldVisitChild, inverted): the subtree's
//      maxSimplifyError <= T_node means EVERY group in it is "too
//      fine" - the whole subtree is dropped in one test. Terminal
//      groups carry FLT_MAX, so subtrees containing them never prune
//      (the cut must be able to reach the roots).
//   3. Internal survivor: children appended to the next level.
//   4. Leaf (one group): the group's contiguous cluster range runs the
//      cull_clusters body per cluster.
//
// Bindings (SSBO set 0):
//   0: float4 params[8]  - cull_clusters layout PLUS the counts the
//      clamps need (Slang StructuredBuffers carry no length):
//      [1].y = nodeCount, [7].z = clusterCount, [7].w = groupCount
//      (stored as f32; exact up to 2^24)
//   1: GpuCluster clusters[]           (48B)
//   2: GpuGroup groups[]               (32B)
//   3: GpuInstance instances[]         (96B)
//   4: uint2 visiblePairs[]  - out (identical semantics to cull_clusters)
//   5: uint counters[8]      - out, cull_clusters semantics
//   6: uint hzb[]             - PREV frame pyramid (occlusion)
//   7: float4 aabbs[]         - 3 per cluster (min, max, maxEdgeLen)
//   8: uint2 pendingPairs[]   - out (occlusion)
//   9: uint2 hwPairs[]        - out (M4.2 routing)
//  10: float4 frustumPlanes[6]
//  11: GpuHierNode hierNodes[]     - the M5.1 BVH8 (GPU repack, 80B)
//  12: uint2 groupRanges[]         - (clusterBegin, clusterCount)/group
//  13: uint2 activeIn[]            - this level's (instance, node) list
//  14: uint2 activeOut[]   - RW: next level's list (atomic appends)
//  15: uint cntIn[1]        - this level's entry count
//  16: uint cntOut[1]       - RW: next level's append cursor
//
// TWO pipeline instances exist (parity): parity 0 binds listA in /
// listB out, parity 1 the reverse - one descriptor-set update per site
// per frame (the M3.3 discipline).

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
  uint _pad0;        // 68 (materialBase lives in the SCENE's decode
                    // instance struct; the chain buffer keeps it pad)
  uint castShadows;  // 72: receiver-only instances (the ground) never
                     // enter the shadow maps
  uint _pad2;
  float4 bounds;
};

struct GpuHierNode {
  float4 aabbMin;          // xyz; w spare
  float4 aabbMax;          // xyz; w spare
  float maxSimplifyError;  // conservative max over the subtree's groups
  uint childCount;         // 0 = leaf, children[0] = group index
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
[[vk::binding(13)]] StructuredBuffer<uint2> activeIn;
[[vk::binding(14)]] RWStructuredBuffer<uint2> activeOut;
[[vk::binding(15)]] StructuredBuffer<uint> cntIn;
[[vk::binding(16)]] RWStructuredBuffer<uint> cntOut;
// M6.4 saturation: the pair/active-list capacity (params[1].w is the
// persistent queue's ring mask - the shared params buffer is full).
[[vk::binding(17)]] StructuredBuffer<uint> pairCaps;
[[vk::binding(18)]] RWStructuredBuffer<uint> travStats;  // [4] visited, [5] pruned

bool fireable(uint groupIdx, float4x4 model) {
  const GpuGroup g = groups[groupIdx];
  // Instance-transformed decision sphere - explicit glm-column linear
  // combination (layout-proof; see cull_clusters.slang).
  const float3 c = model[0].xyz * g.bounds.x + model[1].xyz * g.bounds.y +
                   model[2].xyz * g.bounds.z + model[3].xyz;
  const float scale = max(length(model[0].xyz),
                          max(length(model[1].xyz), length(model[2].xyz)));
  const float dist = max(distance(c, params[0].xyz) - g.bounds.w * scale, 0.1);
  // Perspective chains scale the tolerance by distance (a world-space
  // error projects smaller with range); ORTHO chains (the shadow
  // clipmaps, params[6].w < 0 - the routing sign encoding) must NOT:
  // projected error in an ortho map is error/texel regardless of
  // range. Carrying the x dist over-relaxed the shadow cut ~eyeRange
  // fold (700 tris / 0.47m tolerance at a 5.9mm texel - the visibly
  // simplified silhouette).
  const float t = params[6].w > 0.0
                     ? params[0].w * params[1].x * dist
                     : params[0].w * params[1].x;
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
/// counters/pairs semantics - the dual-mode differential pins this).
void emitCluster(uint2 item, float4x4 model) {
  // params[7].z (shadow chains): receiver-only instances never emit -
  // a ground plane in its own map self-receives at minified sampling
  // (moire bands no bias can clear; it occludes nothing above it in a
  // directional light anyway).
  if (params[1].z > 0.5 && instances[item.x].castShadows == 0u) return;
  const GpuCluster cluster = clusters[item.y];

  if (!fireable(cluster.group, model)) {
    InterlockedAdd(counters[1], 1);
    return;
  }
  if (cluster.refinedGroup >= 0 &&
      fireable(uint(cluster.refinedGroup), model)) {
    InterlockedAdd(counters[2], 1);
    return;
  }

  if (params[6].x > 0.5 && occluded(item, model)) {
    uint pend = 0;
    InterlockedAdd(counters[4], 1, pend);
    if (pend < pairCaps[0]) pendingPairs[pend] = item;
    return;
  }

  if (params[7].x > 0.5) {
    bool hw = params[7].x > 1.5;
    if (!hw) {
      // Route mode: START routable - the guards below are VETOES only.
      // (hw initialized false from the mode test made `hw && edge >=
      // threshold` permanently false - routing dead since the M4.2
      // near-guard refactor, masked by force-hw bypassing it.)
      hw = true;
      const float maxScale =
          max(length(model[0].xyz),
              max(length(model[1].xyz), length(model[2].xyz)));
      const float3 c = model[0].xyz * cluster.bounds.x +
                       model[1].xyz * cluster.bounds.y +
                       model[2].xyz * cluster.bounds.z + model[3].xyz;
      const float sphereR = cluster.bounds.w * maxScale;
      const float dist = max(distance(c, params[0].xyz) - sphereR, 0.1);
      // Near-plane clusters route to HW (UE: bUseHWRaster |=
      // Cull.bNeedsClipping): the fixed-function clipper handles crossing
      // triangles (raster_hw's clipped mode), while SW would Sutherland-
      // Hodgman-clip and scan the largest, closest triangles of the frame -
      // its worst case. (The guard used to force these to SW: the old HW
      // VS divided by w for its snap.)
      const bool nearCross = dist < sphereR + params[1].z;
      // [6].w > 0 = perspective tanHalf; < 0 = -extentY for ortho
      // chains (shadow clipmaps) - px/unit is distance-free there
      // (dividing by dist never routed: the M9.1 shadow hw=0 case).
      const float pxPerUnit =
          params[6].w > 0.0
              ? (params[6].z * 0.5) / (max(params[6].w, 1e-6) * dist)
              : (params[6].z * 0.5) / max(-params[6].w, 1e-6);
      const float edgePx = aabbs[3u * item.y + 2u].x * maxScale * pxPerUnit;
      hw = nearCross || edgePx >= params[7].y;
    }
    if (hw) {
      uint hs = 0;
      InterlockedAdd(counters[7], 1, hs);
      if (hs < pairCaps[0]) hwPairs[hs] = item;
      InterlockedAdd(counters[6], cluster.indexCount / 3u);
      return;
    }
  }

  uint slot = 0;
  InterlockedAdd(counters[0], 1, slot);
  if (slot < pairCaps[0]) visiblePairs[slot] = item;
  InterlockedAdd(counters[6], cluster.indexCount / 3u);
}

/// Node-AABB occlusion test (M5.4a) against the PREV frame's HZB - the
/// same conservative rect walk as occluded(), fed the already-computed
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

[shader("compute")][numthreads(64, 1, 1)] void main(
    uint tid : SV_DispatchThreadID) {
  // The clamped dispatch envelope's ceil slack admits threads past
  // the capacity - the in-guard needs BOTH bounds (raw count and cap).
  if (tid >= cntIn[0] || tid >= pairCaps[0]) return;
  InterlockedAdd(travStats[4], 1);  // visited (M5.5 curves)
  const uint nodeCount = uint(params[1].y);
  const uint clusterCount = uint(params[7].z);
  const uint groupCount = uint(params[7].w);
  const uint2 entry = activeIn[tid];
  const uint inst = entry.x;  // written by our own passes (trusted)
  // M5.4a: entry.y packs the INHERITED-OCCLUDED bit in 31; the node
  // index in the low 31 bits (node counts are far below 2^31).
  const bool inherited = (entry.y & 0x80000000u) != 0u;
  const float4x4 model = instances[inst].model;
  const uint nodeIdx = min(entry.y & 0x7FFFFFFFu, nodeCount - 1u);
  const GpuHierNode node = hierNodes[nodeIdx];

  // World-space corners of the node AABB under the instance transform
  // (explicit glm-column combination).
  float3 corner[8];
  for (uint k = 0; k < 8u; ++k) {
    const float3 lc = float3((k & 1u) != 0u ? node.aabbMax.x : node.aabbMin.x,
                             (k & 2u) != 0u ? node.aabbMax.y : node.aabbMin.y,
                             (k & 4u) != 0u ? node.aabbMax.z : node.aabbMin.z);
    corner[k] = model[0].xyz * lc.x + model[1].xyz * lc.y +
                model[2].xyz * lc.z + model[3].xyz;
  }

  // 1) Conservative frustum prune: one plane with ALL corners outside.
  for (uint p = 0; p < 6u; ++p) {
    bool allOutside = true;
    for (uint k = 0; k < 8u; ++k) {
      if (dot(frustumPlanes[p].xyz, corner[k]) + frustumPlanes[p].w >= 0.0) {
        allOutside = false;
        break;
      }
    }
    if (allOutside) {
      InterlockedAdd(travStats[5], 1);  // pruned (frustum)
      return;
    }
  }

  // 2) LOD prune: the subtree is uniformly "too fine" for this view.
  float3 wmn = corner[0], wmx = corner[0];
  for (uint k = 1u; k < 8u; ++k) {
    wmn = min(wmn, corner[k]);
    wmx = max(wmx, corner[k]);
  }
  const float3 nearest = clamp(params[0].xyz, wmn, wmx);
  const float dist = max(distance(nearest, params[0].xyz), 0.1);
  // Ortho chains (params[6].w < 0): no distance term - see fireable.
  const float tNode = params[6].w > 0.0
                          ? params[0].w * params[1].x * dist
                          : params[0].w * params[1].x;
  if (node.maxSimplifyError <= tNode) {
    InterlockedAdd(travStats[5], 1);  // pruned (LOD)
    return;
  }

  if (node.childCount == 0u) {
    // 4) Leaf: one group; its clusters are a contiguous range.
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
        // LOD rule still applies (bulk-pend only replaces the
        // occlusion test, not the cut).
        if (!fireable(clusters[c].group, model)) continue;
        if (clusters[c].refinedGroup >= 0 &&
            fireable(uint(clusters[c].refinedGroup), model))
          continue;
        uint p = 0;
        InterlockedAdd(counters[4], 1, p);
        pendingPairs[p] = uint2(inst, c);
      }
      return;
    }
    for (uint c = begin; c < end; ++c) emitCluster(uint2(inst, c), model);
    return;
  }

  // 3) Internal survivor: push the children to the next level. The
  // node's own AABB occlusion test (the ROOT node's test plays the
  // instance-level pre-pass: its transformed box wraps the instance)
  // sets the inherited bit for the whole subtree.
  bool childOcc = inherited;
  if (!inherited && params[6].x > 0.5 && boxOccluded(corner)) childOcc = true;
  const uint k = min(node.childCount, 8u);
  for (uint i = 0; i < k; ++i) {
    uint s = 0;
    InterlockedAdd(cntOut[0], 1, s);
    if (s < pairCaps[0])
    activeOut[s] =
        uint2(inst, node.children[i] | (childOcc ? 0x80000000u : 0u));
  }
}
