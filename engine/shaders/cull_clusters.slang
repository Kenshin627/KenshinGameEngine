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
//                          [7]=prevCamPos.xyz (nearest-point ray)
//   1: GpuCluster clusters[]  - 48B, mirrors packGpuCluster
//   2: GpuGroup groups[]      - 32B, mirrors packGpuGroup
//   3: uint2 workItems[]      - (instanceIdx, clusterIdx)
//   4: GpuInstance instances[]- 96B, mirrors cull_instances.slang
//   5: uint2 visiblePairs[]   - out: surviving (instanceIdx, clusterIdx)
//   7: uint hzb[]             - the PREV frame's HZB pyramid (M3.2;
//                              only read when occEnabled)
//   6: uint counters[5]       - [4] occlusion-culled count (out);
//                              [3] work item count (in; GPU-written by
//                              the in-frame chain or pre-filled one-shot),
//                              [0] visible count (zero-init out);
//                               [1..3] zero-init debug stats

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
  uint _pad0;
  uint _pad1;
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
  const float t = params[0].w * params[1].x * dist;
  return g.simplifyError > t;
}

bool occluded(uint2 item, float4x4 model) {
  const GpuCluster cluster = clusters[item.y];
  const uint W = uint(params[6].y), H = uint(params[6].z);

  // Instance-transformed bounds sphere (fireable's convention).
  const float3 c = model[0].xyz * cluster.bounds.x +
                   model[1].xyz * cluster.bounds.y +
                   model[2].xyz * cluster.bounds.z + model[3].xyz;
  const float scale = max(length(model[0].xyz),
                          max(length(model[1].xyz), length(model[2].xyz)));
  const float r = cluster.bounds.w * scale;

  // Nearest sphere point toward the PREV camera.
  const float3 toCam = params[7].xyz - c;
  const float d = length(toCam);
  if (d <= r) return false;  // prev cam inside: keep
  const float3 q = c + toCam * (r / d);

  // Project with the PREV viewProj (Y-flipped: row = framebuffer
  // row, the HZB's space). Explicit glm-column combination.
  const float4 clip =
      params[2] * q.x + params[3] * q.y + params[4] * q.z + params[5];
  if (clip.w <= 0.0) return false;  // behind prev camera: keep

  const float2 ndc = clip.xy / clip.w;
  const float sx = (ndc.x * 0.5 + 0.5) * float(W);
  const float sy = (ndc.y * 0.5 + 0.5) * float(H);
  const float z = clamp(clip.z / clip.w, 0.0, 1.0);
  const uint nearBits = uint((1.0 - z) * 4294967295.0);

  // Screen bbox: the sphere's pixel radius (vertical == horizontal
  // under tanHalfFov: W/(aspect) == H).
  const float tanHalf = params[6].w;
  const float R = (r / max(d - r, 0.1)) * (float(H) / (2.0 * tanHalf));
  const int x0 = int(floor(sx - R)), x1 = int(ceil(sx + R));
  const int y0 = int(floor(sy - R)), y1 = int(ceil(sy + R));
  // Fully-on-screen only; partially-off rects keep (the post pass
  // in M3.3 revisits).
  if (x0 < 0 || y0 < 0 || x1 >= int(W) || y1 >= int(H)) return false;

  // Pick the lowest mip whose coverage of the rect is <= 4x4 texels,
  // walking the SAME ceil-division dims as hzbLayout.
  uint m = 0, lw = W, lh = H;
  while (((uint(x1) >> m) - (uint(x0) >> m) >= 4u ||
          (uint(y1) >> m) - (uint(y0) >> m) >= 4u) &&
         lw > 1u && lh > 1u) {
    ++m;
    lw = (lw + 1u) / 2u;
    lh = (lh + 1u) / 2u;
  }
  // Texel offset of level m (walk again summing sizes).
  uint off = 0, ww = W, hh = H;
  for (uint k = 0; k < m; ++k) {
    const uint sz = ww * hh;
    off += sz;
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

  // Reversed bits: near = large. Occluded iff even the cluster's
  // NEAREST point is farther (smaller bits) than the tile's
  // farthest rendered geometry (min bits).
  return nearBits < tileMin;
}

[shader("compute")][numthreads(64, 1, 1)] void main(
    uint tid : SV_DispatchThreadID) {
  if (tid >= counters[3]) return;

  const uint2 item = workItems[tid];
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
  if (params[6].x > 0.5 && occluded(item, model)) {
    InterlockedAdd(counters[4], 1);
    return;
  }

  uint slot = 0;
  InterlockedAdd(counters[0], 1, slot);
  visiblePairs[slot] = item;
}
