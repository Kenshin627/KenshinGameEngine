// S4.3 phase 2: cluster culling - the LOD cut on the GPU.
// One thread per work item (visibleInstance, cluster). The decision is
// Karis SIGGRAPH 2021 p.70 verbatim, in clusterlod's data vocabulary:
//   Render: ParentError > T && ClusterError <= T
//   -> fireable(g)  = groups[g].simplifyError > T_g
//      (dropping to the coarser level would exceed the threshold)
//   -> selected     = fireable(cluster.group) &&
//                     (refinedGroup < 0 || !fireable(refinedGroup))
//      (the source's committed error was acceptable; original geometry
//       always passes)
// T_g scales the pixel threshold by the GROUP's bounds distance to the
// camera UNDER THE INSTANCE TRANSFORM - identical math to the CPU
// selectLodCut (engine/src/geometry/LodCut.cpp); the differential test
// pins them together.
//
// Bindings (SSBO set 0; buffers[i] -> binding i):
//   0: float4 params[8]  - [0]=(camX, camY, camZ, pixelThreshold)
//                          [1]=(worldPerPixel, workItemCount, 0, 0)
//                          [2..5] PREV frame viewProj columns (glm raw,
//                          Y-flipped - the space the HZB lives in)
//                          [6]=(occEnabled, screenW, screenH, tanHalfFov)
//                          [7]=(hwMode, hwThresholdPx, 0, 0): 0 = all
//                          SW, 1 = route by threshold, 2 = force all
//                          HW (the differential switch, UE's
//                          RENDER_FLAG_FORCE_HW_RASTER)
//   1: GpuCluster clusters[]  - 48B, mirrors packGpuCluster
//   2: GpuGroup groups[]      - 32B, mirrors packGpuGroup
//   3: uint2 workItems[]      - (instanceIdx, clusterIdx)
//   4: GpuInstance instances[]- 96B, mirrors cull_instances.slang
//   5: uint2 visiblePairs[]   - out: surviving (instanceIdx, clusterIdx)
//   7: uint hzb[]             - the PREV frame's HZB pyramid (M3.2;
//                              only read when occEnabled)
//   8: float4 aabbs[]         - 3 per cluster: (min.xyz,0), (max.xyz,0),
//                              (maxEdgeLen,0,0,0) - object-space, the
//                              occlusion rect projects min/max and the
//                              M4.2 routing reads maxEdgeLen (derived
//                              at runtime, not serialized)
//   9: uint2 pendingPairs[]   - out (M3.3): occluded items, kept for
//                              the post pass's retest instead of being
//                              dropped
//   10: uint2 hwPairs[]       - out (M4.2): selected items routed to
//                              the hardware rasterizer (UE
//                              SmallEnoughToDraw, 18px default)
//   6: uint counters[8]       - [4] occlusion-culled count (out),
//                              [5] post-revived count (written by the
//                              POST cull, zero here),
//                              [6] cut-triangle accumulation (out,
//                              M3.4: surviving clusters' indexCount/3,
//                              the post cull adds its revived share),
//                              [7] HW-routed count (out, M4.2);
//                              [3] work item count (in; GPU-written by
//                              the in-frame chain or pre-filled one-shot),
//                              [0] visible count (zero-init out);
//                               [1..2] zero-init debug stats

struct GpuCluster {
  float4 bounds;  // xyz center, w radius
  float error;
  float parentError;
  uint vertexOffset;
  uint vertexCount;
  uint indexOffset;
  uint indexCount;
  uint group;
  int refinedGroup;  // clodCluster::refined, -1 = original
};

struct GpuGroup {
  float4 bounds;  // shared decision sphere (monotone-dominated)
  float maxError;
  float simplifyError;  // FLT_MAX on terminal groups
  uint lodLevel;        // S4.6 debug views (packed into the old pad)
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

[[vk::binding(0)]] StructuredBuffer<float4> params;
[[vk::binding(1)]] StructuredBuffer<GpuCluster> clusters;
[[vk::binding(2)]] StructuredBuffer<GpuGroup> groups;
[[vk::binding(3)]] StructuredBuffer<uint2> workItems;
[[vk::binding(4)]] StructuredBuffer<GpuInstance> instances;
[[vk::binding(5)]] RWStructuredBuffer<uint2> visiblePairs;
[[vk::binding(6)]] RWStructuredBuffer<uint> counters;
[[vk::binding(7)]] StructuredBuffer<uint> hzb;
[[vk::binding(8)]] StructuredBuffer<float4> aabbs;
[[vk::binding(9)]] RWStructuredBuffer<uint2> pendingPairs;
[[vk::binding(10)]] RWStructuredBuffer<uint2> hwPairs;

bool fireable(uint groupIdx, float4x4 model) {
  const GpuGroup g = groups[groupIdx];
  // Instance-transformed decision sphere - the same convention as
  // VirtualGeometryScene::setInstances and the CPU reference.
  // NOTE: glm stores column-major and Slang's mul(model, v) reads the
  // buffer vec4s as ROWS (probe-verified: a glm translate matrix came
  // out identity in xyz) - so apply the transform as an explicit
  // linear combination of the stored columns, which is layout-proof.
  // model[i] returns the raw i-th vec4 = glm's i-th column.
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

bool occluded(uint2 item, float4x4 model)
{
    const uint W = uint(params[6].y), H = uint(params[6].z);
    const float3 mn = aabbs[3u * item.y + 0u].xyz;
    const float3 mx = aabbs[3u * item.y + 1u].xyz;

    // 8 cluster-local corners -> world (explicit glm-column
    // combination per corner), projected with the PREV viewProj
    // (Y-flipped: row = framebuffer row, the HZB's space).
    float4 clip[8];
    for (uint k = 0; k < 8u; ++k)
    {
        const float3 lc = float3((k & 1u) != 0u ? mx.x : mn.x,
                                 (k & 2u) != 0u ? mx.y : mn.y,
                                 (k & 4u) != 0u ? mx.z : mn.z);
        const float3 wc = model[0].xyz * lc.x + model[1].xyz * lc.y
                        + model[2].xyz * lc.z + model[3].xyz;
        clip[k] = params[2] * wc.x + params[3] * wc.y
                + params[4] * wc.z + params[5];
        if (clip[k].w <= 0.0) return false;  // a corner behind the
                                             // prev camera: keep
    }

    // The cluster's nearest possible depth = the corner with the
    // smallest NDC z (largest reversed bits); pack with packPixel's
    // formula so both sides share the bit layout. AABB corners bound
    // the box exactly (perspective maps the box to a convex polygon
    // whose extremes are corner projections).
    float zMin = clip[0].z / clip[0].w;
    float rx0 = 1e30f, rx1 = -1e30f, ry0 = 1e30f, ry1 = -1e30f;
    for (uint k = 0; k < 8u; ++k)
    {
        const float2 ndc = clip[k].xy / clip[k].w;
        const float sx = (ndc.x * 0.5 + 0.5) * float(W);
        const float sy = (ndc.y * 0.5 + 0.5) * float(H);
        rx0 = min(rx0, sx); rx1 = max(rx1, sx);
        ry0 = min(ry0, sy); ry1 = max(ry1, sy);
        zMin = min(zMin, clip[k].z / clip[k].w);
    }
    zMin = clamp(zMin, 0.0, 1.0);
    const uint nearBits = uint((1.0 - zMin) * 4294967295.0);

    // Fully-on-screen only; partially-off rects keep (the M3.3 post
    // pass revisits). The rect is outer-inclusive.
    const int x0 = int(floor(rx0)), x1 = int(ceil(rx1));
    const int y0 = int(floor(ry0)), y1 = int(ceil(ry1));
    if (x0 < 0 || y0 < 0 || x1 >= int(W) || y1 >= int(H)) return false;

    // Pick the lowest mip whose coverage of the rect is <= 4x4 texels,
    // walking the SAME ceil-division dims as hzbLayout.
    uint m = 0, lw = W, lh = H;
    while (((uint(x1) >> m) - (uint(x0) >> m) >= 4u ||
            (uint(y1) >> m) - (uint(y0) >> m) >= 4u) &&
           lw > 1u && lh > 1u)
    {
        ++m;
        lw = (lw + 1u) / 2u;
        lh = (lh + 1u) / 2u;
    }
    uint off = 0, ww = W, hh = H;
    for (uint k = 0; k < m; ++k)
    {
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
                min(tileMin,
                    hzb[off + min(ty, lh - 1u) * lw + min(tx, lw - 1u)]);

    // Reversed bits: near = large. Occluded iff even the cluster's
    // NEAREST point is farther (smaller bits) than the tile's
    // farthest rendered geometry (min bits).
    return nearBits < tileMin;
}

[shader("compute")][numthreads(64, 1, 1)] void main(
    uint tid : SV_DispatchThreadID) {
  // Ceil slack of the clamped envelope: guard at the capacity too.
  if (tid >= counters[3] || tid >= uint(params[1].w)) return;

  const uint2 item = workItems[tid];
  // M9.1: shadow chains (params[7].z) skip receiver-only instances -
  // a ground plane in its own map self-receives at minified sampling
  // (moire bands no bias clears; it occludes nothing above it in a
  // directional light anyway).
  if (params[1].z > 0.5 && instances[item.x].castShadows == 0u) return;
  const GpuCluster cluster = clusters[item.y];
  const float4x4 model = instances[item.x].model;

  // Karis p.70: Render iff ParentError > T && ClusterError <= T.
  if (!fireable(cluster.group, model)) {
    InterlockedAdd(counters[1], 1);  // too coarse to stop here
    return;
  }
  if (cluster.refinedGroup >= 0 && fireable(cluster.refinedGroup, model)) {
    InterlockedAdd(counters[2], 1);  // source already too coarse
    return;
  }

  // M3.2: occlusion test against the PREV frame's HZB (Karis
  // Slide 17 / Hill-Collin): project the cluster's bounds sphere
  // with the PREV viewProj, take its nearest point's depth bits, and
  // cull when even the nearest is farther than everything rendered
  // in the covered tile(s) last frame. Conservative by construction;
  // any doubt (off-screen rect, camera inside the sphere, behind the
  // prev camera, test disabled) keeps the cluster.
  // M3.3: the occluded item lands in pendingPairs for the post
  // pass's same-frame retest (Karis Slide 75) instead of being
  // dropped - disoccluded geometry gets revived within the frame.
  // One atomic add yields both the counter increment and the write
  // slot (counters[4] doubles as the pending count).
  if (params[6].x > 0.5 && occluded(item, model)) {
    uint pend = 0;
    InterlockedAdd(counters[4], 1, pend);
    // Saturating append: reservations past the list capacity drop
    // (the consumers' converts clamp the count - never an OOB write).
    if (pend < uint(params[1].w)) pendingPairs[pend] = item;
    return;
  }

  // M4.2 routing (UE SmallEnoughToDraw / D6.4): project the cluster's
  // max object-space edge through the instance's max scale at the
  // bounds sphere's nearest point (axis-distance approximation of
  // UE's Min projected-edge scale; routing is a soft perf decision,
  // both paths are correct). edgePx >= threshold (18px, UE default)
  // -> hardware rasterizer.
  if (params[7].x > 0.5) {
    bool hw = params[7].x > 1.5;
    if (!hw) {
      // Route mode: START routable - the guards below are VETOES only.
      // (This block's hw initialized false from the mode test made
      // `hw && edge >= threshold` permanently false - routing dead since
      // the M4.2 near-guard refactor, masked by force-hw bypassing it.)
      hw = true;
      const float maxScale = max(length(model[0].xyz),
                                 max(length(model[1].xyz),
                                     length(model[2].xyz)));
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
      // [6].w > 0 = perspective tanHalf (px/unit falls with distance);
      // < 0 = -extentY for ortho chains (shadow clipmaps) - px/unit is
      // distance-free there; dividing by dist under-estimated edgePx
      // ~dist-fold and never routed (the M9.1 shadow hw=0 case).
      const float pxPerUnit =
          params[6].w > 0.0
              ? (params[6].z * 0.5) / (max(params[6].w, 1e-6) * dist)
              : (params[6].z * 0.5) / max(-params[6].w, 1e-6);
      const float edgePx =
          aabbs[3u * item.y + 2u].x * maxScale * pxPerUnit;
      hw = nearCross || edgePx >= params[7].y;
    }
    if (hw) {
      uint hs = 0;
      InterlockedAdd(counters[7], 1, hs);
      if (hs < uint(params[1].w)) hwPairs[hs] = item;
      InterlockedAdd(counters[6], cluster.indexCount / 3u);
      return;
    }
  }

  uint slot = 0;
  InterlockedAdd(counters[0], 1, slot);
  if (slot < uint(params[1].w)) visiblePairs[slot] = item;
  InterlockedAdd(counters[6], cluster.indexCount / 3u);  // cut tris
}
