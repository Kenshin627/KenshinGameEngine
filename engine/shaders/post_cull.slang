// M3.3 post cull (Karis Slide 75, right half): re-test the Main
// pass's occluded (pending) clusters against the CURRENT frame's
// preliminary HZB - built right after the Main raster from the very
// VisBuffer it just produced. Whatever the current camera can see
// gets revived and re-rasterized within the SAME frame; whatever is
// still behind the current occluders stays culled. This is what
// closes the disocclusion holes a single-pass prev-HZB test leaves
// when the camera moves.
//
// LOD/frustum results carry over from the Main cull (same frame, so
// the view position is identical - only occlusion is retested; see
// nanite.md 10.4).
//
// The occluded() body is a deliberate duplicate of cull_clusters'
// (Slang module sharing would drag its bindings in; revisit when a
// shared no-binding helper module lands).
//
// Bindings (SSBO set 0):
//   0: float4 params[8]   - [0].xyz camPos, [1].z the near-guard
//                           margin (the Main cull's word), [1].w pair
//                           capacity, [2..5] CURRENT viewProj columns
//                           (glm raw, Y-flipped), [6]=(occEnabled,
//                           screenW, screenH, tanHalfFov), [7] = the
//                           Main cull's (hwMode, hwThresholdPx)
//   1: uint2 pendingPairs[] - the Main cull's occluded items (in)
//   2: uint2 postPairs[]    - out: revived (instanceIdx, clusterIdx)
//   3: GpuInstance instances[] - model transforms (AABB corners)
//   4: uint hzb[]           - the CURRENT frame's preliminary pyramid
//   5: float4 aabbs[]       - 2 per cluster (min.xyz, max.xyz)
//   6: uint counters[7]     - [4] pending count (in, Main cull's),
//                             [5] revived count (out),
//                             [6] cut-triangle accumulation (out; this
//                             pass adds the revived clusters' tris)
//   7: GpuCluster clusters[] - 48B, mirrors packGpuCluster (the
//                             revived items' indexCount for [6])
//   8: uint2 hwPairs[]      - out: revived items that ROUTE to the
//                             hardware rasterizer, appended after the
//                             Main cull's via the same counters[7]
//                             cursor. The HW draw runs after the whole
//                             chain (count_to_draw reads the final
//                             [7]), so one draw covers both - UE's post
//                             pass routes exactly like its Main pass;
//                             v1 sent every revived cluster to SW,
//                             force-HW included.

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

struct GpuInstance {
  float4x4 model;
  uint meshId;
  uint _pad0;
  uint _pad1;
  uint _pad2;
  float4 bounds;
};

[[vk::binding(0)]] StructuredBuffer<float4> params;
[[vk::binding(1)]] StructuredBuffer<uint2> pendingPairs;
[[vk::binding(2)]] RWStructuredBuffer<uint2> postPairs;
[[vk::binding(3)]] StructuredBuffer<GpuInstance> instances;
[[vk::binding(4)]] StructuredBuffer<uint> hzb;
[[vk::binding(5)]] StructuredBuffer<float4> aabbs;
[[vk::binding(6)]] RWStructuredBuffer<uint> counters;
[[vk::binding(7)]] StructuredBuffer<GpuCluster> clusters;
[[vk::binding(8)]] RWStructuredBuffer<uint2> hwPairs;

/// The Main cull's routing test, VERBATIM (cull_clusters /
/// cull_hier_level emitCluster - keep the four copies together).
bool routeHw(uint2 item, float4x4 model)
{
    if (params[7].x < 0.5) return false;
    if (params[7].x > 1.5) return true;
    const GpuCluster cluster = clusters[item.y];
    const float maxScale = max(length(model[0].xyz),
                               max(length(model[1].xyz), length(model[2].xyz)));
    const float3 c = model[0].xyz * cluster.bounds.x +
                     model[1].xyz * cluster.bounds.y +
                     model[2].xyz * cluster.bounds.z + model[3].xyz;
    const float sphereR = cluster.bounds.w * maxScale;
    const float dist = max(distance(c, params[0].xyz) - sphereR, 0.1);
    if (dist < sphereR + params[1].z) return true;  // near plane -> HW clips
    const float pxPerUnit =
        (params[6].z * 0.5) / (max(params[6].w, 1e-6) * dist);
    const float edgePx = aabbs[3u * item.y + 2u].x * maxScale * pxPerUnit;
    return edgePx >= params[7].y;
}

bool occluded(uint2 item, float4x4 model)
{
    const uint W = uint(params[6].y), H = uint(params[6].z);
    const float3 mn = aabbs[3u * item.y + 0u].xyz;
    const float3 mx = aabbs[3u * item.y + 1u].xyz;

    // 8 cluster-local corners -> world (explicit glm-column
    // combination per corner), projected with the CURRENT viewProj
    // (Y-flipped: row = framebuffer row, this HZB's space).
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
                                             // camera: keep
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

    // Fully-on-screen only; partially-off rects keep (conservative).
    // The rect is outer-inclusive.
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

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint tid : SV_DispatchThreadID)
{
    if (tid >= counters[4] || tid >= uint(params[1].w)) return;
    const uint2 item = pendingPairs[tid];
    if (occluded(item, instances[item.x].model)) return;  // still hidden

    if (routeHw(item, instances[item.x].model))
    {
        uint hs = 0;
        InterlockedAdd(counters[7], 1, hs);
        if (hs < uint(params[1].w)) hwPairs[hs] = item;
        InterlockedAdd(counters[6], clusters[item.y].indexCount / 3u);
        return;
    }
    uint slot = 0;
    InterlockedAdd(counters[5], 1, slot);
    if (slot < uint(params[1].w)) postPairs[slot] = item;
    // The revived cluster's triangles join this frame's cut (M3.4:
    // the panel/test readout counts everything actually rasterized).
    InterlockedAdd(counters[6], clusters[item.y].indexCount / 3u);
}
