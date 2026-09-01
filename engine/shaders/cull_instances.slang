// S4.2: instance culling compute shader.
// One thread per instance: frustum 6-plane test against the instance's
// WORLD-space bounds sphere (packed in the instance buffer by
// VirtualGeometryScene), atomic-append survivors to the visible list.
//
// Bindings (SSBO set 0; the Job convention maps buffers[i] -> binding i):
//   0: float4 frustumPlanes[6]   - xyz normal (normalized), w distance;
//                                  positive side = inside
//   1: GpuInstance instances[]   - mirrors VirtualGeometryScene's 96B
//                                  std430 layout EXACTLY
//   2: uint visibleList[]        - output: indices of visible instances
//   3: uint counter[2]           - [0] visible count (zero-init output),
//                                  [1] instance count (input; Slang
//                                  StructuredBuffer exposes no length)

struct GpuInstance
{
    float4x4 model;     // 64B
    uint meshId;        // +4
    uint _pad0;         // +4 each: three separate uints pad 12B to the
    uint _pad1;         // vec4 at 80 (a float3/uint3 would align to 16
    uint _pad2;         // and push bounds to 96 - stride must be 96B)
    float4 bounds;      // xyz world-space center, w radius
};

[[vk::binding(0)]] StructuredBuffer<float4>  frustumPlanes;
[[vk::binding(1)]] StructuredBuffer<GpuInstance> instances;
[[vk::binding(2)]] RWStructuredBuffer<uint>  visibleList;
[[vk::binding(3)]] RWStructuredBuffer<uint>  counter;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint tid : SV_DispatchThreadID)
{
    // Out-of-range threads still bump the counter guard-free: the list is
    // sized to the instance count, so only in-range threads may append.
    if (tid >= counter[1]) return;

    const float3 c = instances[tid].bounds.xyz;
    const float r = instances[tid].bounds.w;

    // Visible = the sphere reaches the positive side of every plane:
    // dot(n, c) + d >= -r  (sphere touching the plane counts as visible).
    bool visible = true;
    [unroll]
    for (int p = 0; p < 6; ++p)
    {
        if (dot(frustumPlanes[p].xyz, c) + frustumPlanes[p].w < -r)
        {
            visible = false;
            break;
        }
    }

    if (visible)
    {
        // Atomic append: InterlockedAdd returns the ORIGINAL value into
        // `slot` - plain read-modify-write races and loses appends.
        uint slot = 0;
        InterlockedAdd(counter[0], 1, slot);
        visibleList[slot] = tid;
    }
}
