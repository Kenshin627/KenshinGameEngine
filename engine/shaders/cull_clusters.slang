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
//   0: float4 params[2]  - [0]=(camX, camY, camZ, pixelThreshold)
//                          [1]=(worldPerPixel, workItemCount, 0, 0)
//   1: GpuCluster clusters[]  - 48B, mirrors packGpuCluster
//   2: GpuGroup groups[]      - 32B, mirrors packGpuGroup
//   3: uint2 workItems[]      - (instanceIdx, clusterIdx)
//   4: GpuInstance instances[]- 96B, mirrors cull_instances.slang
//   5: uint2 visiblePairs[]   - out: surviving (instanceIdx, clusterIdx)
//   6: uint counters[4]       - [0] visible count (zero-init out);
//                               [1..3] zero-init debug stats

struct GpuCluster
{
    float4 bounds;       // xyz center, w radius
    float error;
    float parentError;
    uint vertexOffset;
    uint vertexCount;
    uint indexOffset;
    uint indexCount;
    uint group;
    int refinedGroup;    // clodCluster::refined, -1 = original
};

struct GpuGroup
{
    float4 bounds;       // shared decision sphere (monotone-dominated)
    float maxError;
    float simplifyError; // FLT_MAX on terminal groups
    uint lodLevel;       // S4.6 debug views (packed into the old pad)
    float _pad4;
};

struct GpuInstance
{
    float4x4 model;
    uint meshId;
    uint _pad0; uint _pad1; uint _pad2;
    float4 bounds;
};

[[vk::binding(0)]] StructuredBuffer<float4> params;
[[vk::binding(1)]] StructuredBuffer<GpuCluster> clusters;
[[vk::binding(2)]] StructuredBuffer<GpuGroup> groups;
[[vk::binding(3)]] StructuredBuffer<uint2> workItems;
[[vk::binding(4)]] StructuredBuffer<GpuInstance> instances;
[[vk::binding(5)]] RWStructuredBuffer<uint2> visiblePairs;
[[vk::binding(6)]] RWStructuredBuffer<uint> counters;

bool fireable(uint groupIdx, float4x4 model)
{
    const GpuGroup g = groups[groupIdx];
    // Instance-transformed decision sphere - the same convention as
    // VirtualGeometryScene::setInstances and the CPU reference.
    // NOTE: glm stores column-major and Slang's mul(model, v) reads the
    // buffer vec4s as ROWS (probe-verified: a glm translate matrix came
    // out identity in xyz) - so apply the transform as an explicit
    // linear combination of the stored columns, which is layout-proof.
    // model[i] returns the raw i-th vec4 = glm's i-th column.
    const float3 c = model[0].xyz * g.bounds.x + model[1].xyz * g.bounds.y
                   + model[2].xyz * g.bounds.z + model[3].xyz;
    const float scale = max(length(model[0].xyz),
                        max(length(model[1].xyz), length(model[2].xyz)));
    const float dist = max(distance(c, params[0].xyz) - g.bounds.w * scale,
                           0.1);
    const float t = params[0].w * params[1].x * dist;
    return g.simplifyError > t;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint tid : SV_DispatchThreadID)
{
    if (tid >= uint(params[1].y)) return;

    const uint2 item = workItems[tid];
    const GpuCluster cluster = clusters[item.y];
    const float4x4 model = instances[item.x].model;

    // Karis p.70: Render iff ParentError > T && ClusterError <= T.
    if (!fireable(cluster.group, model))
    {
        InterlockedAdd(counters[1], 1);  // too coarse to stop here
        return;
    }
    if (cluster.refinedGroup >= 0 && fireable(cluster.refinedGroup, model))
    {
        InterlockedAdd(counters[2], 1);  // source already too coarse
        return;
    }

    uint slot = 0;
    InterlockedAdd(counters[0], 1, slot);
    visiblePairs[slot] = item;
}
