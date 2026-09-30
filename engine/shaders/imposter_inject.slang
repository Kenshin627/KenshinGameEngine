// M9.2 Stage B: the far-field imposter injection (nanite.md §14 /
// Slide 97). One workgroup per tiny instance, one thread per pixel of
// its screen rect: octahedral-map the view direction into the baked
// atlas (SSBO u32 planes - triangles cannot bilinear), reconstruct
// the world position inside the instance's AABB ortho box, reproject
// through the MAIN viewProj, and atomicMax the SAME VisBuffer the
// cluster chains write - imposter and geometry compete for pixels
// through the identical depth-keyed atomics.
//
// Bindings (SSBO set 0):
//   0: RWStructuredBuffer<uint64_t> vis - the main chain's VisBuffer
//   1: StructuredBuffer<float4> params[8]
//      [0..3] viewProj columns (glm raw)
//      [4]   camPos.xyz, viewportW
//      [5]   1/viewport.xy, viewportH, quadCount
//   2: StructuredBuffer<float4> quads - 4 words per instance:
//      [0] rect (x0,y0,x1,y1) pixels, inclusive
//      [1] worldCenter.xyz, atlasBase (u32 word of tile 0)
//      [2] halfExtents.xyz, pairIdx
//      [3] unused pad
//   3: StructuredBuffer<uint2> atlas - (depthBits, tri) per texel,
//      12x12 per direction, 12x12 directions per mesh (concatenated)

[[vk::binding(0)]] RWStructuredBuffer<uint64_t> vis;
[[vk::binding(1)]] StructuredBuffer<float4> params;
[[vk::binding(2)]] StructuredBuffer<float4> quads;
[[vk::binding(3)]] StructuredBuffer<uint2> atlas;

static const uint kDirs = 12u;
static const uint kTexels = 12u;

float3 octDecode(float2 p)
{
    float z = 1.0 - abs(p.x) - abs(p.y);
    if (z < 0.0)
    {
        const float ox = p.x;
        p.x = (1.0 - abs(p.y)) * (ox < 0.0 ? -1.0 : 1.0);
        p.y = (1.0 - abs(ox)) * (p.y < 0.0 ? -1.0 : 1.0);
    }
    return normalize(float3(p.x, p.y, z));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID)
{
    const uint q = gid.x;
    if (q >= uint(params[5].w)) return;
    const float4 e0 = quads[q * 4u + 0u];
    const float4 e1 = quads[q * 4u + 1u];
    const float4 e2 = quads[q * 4u + 2u];
    const int x0 = int(e0.x), y0 = int(e0.y);
    const int x1 = int(e0.z), y1 = int(e0.w);
    const uint W = uint(params[4].w);
    // 64 threads stride the FLAT rect (columns AND rows).
    const int cols = max(x1 - x0 + 1, 1);
    const int rows = max(y1 - y0 + 1, 1);
    for (int k = int(tid.x); k < cols * rows; k += 64)
    {
        const int px = x0 + k % cols;
        const int py = y0 + k / cols;
        if (px < 0 || py < 0 || uint(px) >= W || uint(py) >= uint(params[5].z))
            continue;

        // The view direction through this instance (the bake's ortho
        // approximation: the direction to the AABB center).
        const float3 dir = normalize(e1.xyz - params[4].xyz);
        // Octahedral tile: uv in [-1,1] -> dir indices.
        float2 uv = dir.xz / (abs(dir.x) + abs(dir.y) + abs(dir.z));
        if (1.0 - abs(uv.x) - abs(uv.y) < 0.0)
        {
            const float ox = uv.x;
            uv.x = (1.0 - abs(uv.y)) * (ox < 0.0 ? -1.0 : 1.0);
            uv.y = (1.0 - abs(ox)) * (uv.y < 0.0 ? -1.0 : 1.0);
        }
        const int du = clamp(int((uv.x + 1.0) * 0.5 * float(kDirs)), 0, int(kDirs) - 1);
        const int dv = clamp(int((uv.y + 1.0) * 0.5 * float(kDirs)), 0, int(kDirs) - 1);

        // The pixel's tile coordinates: the rect linearly spans the
        // AABB's screen projection (UE ray-marches parallax here; v1
        // approximates - tiny instances are a few pixels).
        const float tx = (float(px - x0) / max(float(x1 - x0), 1.0)) * 2.0 - 1.0;
        const float ty = (float(py - y0) / max(float(y1 - y0), 1.0)) * 2.0 - 1.0;
        const int tu = clamp(int((tx + 1.0) * 0.5 * float(kTexels)), 0, int(kTexels) - 1);
        const int tv = clamp(int((ty + 1.0) * 0.5 * float(kTexels)), 0, int(kTexels) - 1);

        const uint2 hit = atlas[uint(e1.w) + (uint(dv) * kDirs + uint(du)) * kTexels * kTexels
                                + uint(tv) * kTexels + uint(tu)];
        if (hit.y == 0xFFFFFFFFu) continue;  // miss: nothing to write

        // Reconstruct the world position inside the bake's ortho box.
        const float z = 1.0 - float(hit.x) / 4294967295.0;
        const float3 fwd = -dir;
        const float3 upRef = abs(fwd.z) > 0.95 ? float3(1, 0, 0) : float3(0, 0, 1);
        const float3 right = normalize(cross(upRef, fwd));
        const float3 up = cross(fwd, right);
        const float3 he = e2.xyz;
        const float3 world = e1.xyz + right * (tx * he.x) + up * (ty * he.y)
                           + fwd * ((z * 2.0 - 1.0) * he.z);

        // The main camera's depth bits + the imposter payload.
        const float4 clip = params[0] * world.x + params[1] * world.y
                          + params[2] * world.z + params[3];
        const float ndcZ = clamp(clip.z / clip.w, 0.0, 1.0);
        const uint depthBits = uint((1.0 - ndcZ) * 4294967295.0);
        const uint payload = (uint(e2.w) << 9) | (3u << 7) | (hit.y & 0x7Fu);
        InterlockedMax(vis[uint(py) * W + uint(px)],
                       (uint64_t(depthBits) << 32) | uint64_t(payload));
    }
}
