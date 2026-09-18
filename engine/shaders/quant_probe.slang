// M7.1 debug probe: decode ONE vertex of ONE cluster through the same
// v5 chain (clusterMeta -> page -> slot -> header -> bits) the raster
// shaders use, and write the decoded int3 + word audit to `out`. The
// test compares against the CPU mirror - any divergence localizes the
// GPU-side defect WITHOUT the full pipeline in the way.
//
// Bindings:
//   0: uint4 params - x clusterIdx, y vertexIdx, z pad, w pad
//   1: uint positions[]    - the page slot pool
//   2: uint clusterMeta[]  - per-cluster ordinal
//   3: uint clusterPage[]
//   4: uint pageTable[]
//   5: GpuPageHeader pageHeader[]  (v5 12 u32)
//   6: RWuint out[8] - [0..2] int3 grid coords, [3] posWordOff,
//      [4] widths, [5] wordA, [6] meta word0 (posStartX), [7] posStep bits

struct GpuPageHeader
{
    uint posBase;
    uint nrmBase;
    uint uvBase;
    uint idxBase;
    uint metaBase;
    uint vertexCount;
    uint indexCount;
    uint clusterCount;
    float posStep;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::binding(7)]] StructuredBuffer<uint> indices;
[[vk::binding(8)]] StructuredBuffer<uint4> clusterTable;  // raw u32 view of GpuCluster (48B = 12 u32)

[[vk::binding(0)]] StructuredBuffer<uint4> params;
[[vk::binding(1)]] StructuredBuffer<uint> positions;
[[vk::binding(2)]] StructuredBuffer<uint> clusterMeta;
[[vk::binding(3)]] StructuredBuffer<uint> clusterPage;
[[vk::binding(4)]] StructuredBuffer<uint> pageTable;
[[vk::binding(5)]] StructuredBuffer<GpuPageHeader> pageHeader;
[[vk::binding(6)]] RWStructuredBuffer<uint> out;

uint loadBits2(uint w0, uint w1, uint bitIdx, uint bits)
{
    const uint shift = bitIdx & 31u;
    const uint mask = (1u << bits) - 1u;
    if (shift + bits <= 32u) return (w0 >> shift) & mask;
    return ((w0 >> shift) | (w1 << (32u - shift))) & mask;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    const uint clusterIdx = params[0].x;
    const uint v = params[0].y;
    const uint page = clusterPage[clusterIdx];
    const uint slot = pageTable[page];
    const GpuPageHeader ph = pageHeader[page];
    const uint pool = slot * 32768u;
    const uint ord = clusterMeta[clusterIdx];
    const uint mb = pool + ph.metaBase + ord * 10u;
    const uint posWordOff = positions[mb + 4u];
    const uint widths = positions[mb + 3u];
    const uint b0 = widths & 0xFu;
    const uint b1 = (widths >> 4) & 0xFu;
    const uint b2 = (widths >> 8) & 0xFu;
    // The axis bases need the CLUSTER's vertex count - passed via
    // params[0].z (the page header only carries the page total).
    const uint nVc = params[0].z;
    const uint base = posWordOff * 32u;
    const uint bitA = base + v * b0;
    const uint bitB = base + nVc * b0 + v * b1;
    const uint bitC = base + nVc * (b0 + b1) + v * b2;
    const uint posBase = pool + ph.posBase;
    const uint q0 = loadBits2(positions[posBase + bitA / 32u],
                              positions[posBase + bitA / 32u + 1u],
                              bitA & 31u, b0);
    const uint q1 = loadBits2(positions[posBase + bitB / 32u],
                              positions[posBase + bitB / 32u + 1u],
                              bitB & 31u, b1);
    const uint q2 = loadBits2(positions[posBase + bitC / 32u],
                              positions[posBase + bitC / 32u + 1u],
                              bitC & 31u, b2);
    out[0] = q0 + uint(positions[mb]);
    out[1] = q1 + uint(positions[mb + 1u]);
    out[2] = q2 + uint(positions[mb + 2u]);
    out[3] = posWordOff;
    out[4] = widths;
    out[5] = positions[posBase + bitA / 32u];
    out[6] = positions[mb];
    out[7] = uint(ph.posStep);
    // Triangle audit: params[1] = triIdx; decode the 3 indices via the
    // raster's byteAt path and store them in out[4..6].
    // GpuCluster layout (u32): bounds(4) error(1) parentError(1)
    // vertexOffset(6) vertexCount(7) indexOffset(8) indexCount(9).
    const uint triIdx2 = params[1].x;
    // GpuCluster 12 u32: bounds(0..3) err(4,5) vertexOffset(6)
    // vertexCount(7) indexOffset(8) indexCount(9) group/refined(10,11).
    // slot 8 lives in the THIRD vec4's .x.
    const uint4 cc2 = clusterTable[clusterIdx * 3u + 2u];
    const uint indexOffset = cc2.x;
    const uint idxBase2 = pool + ph.idxBase;
    const uint byteOff = indexOffset + triIdx2 * 3u;
    const uint lane = byteOff & 3u;
    const uint w0 = indices[idxBase2 + byteOff / 4u];
    const uint w1 = indices[idxBase2 + byteOff / 4u + 1u];
    uint ii0, ii1, ii2;
    ii0 = (w0 >> (8u * lane)) & 0xFFu;
    if (lane == 0u) { ii1 = (w0 >> 8u) & 0xFFu; ii2 = (w0 >> 16u) & 0xFFu; }
    else if (lane == 1u) { ii1 = (w0 >> 16u) & 0xFFu; ii2 = (w0 >> 24u) & 0xFFu; }
    else if (lane == 2u) { ii1 = (w0 >> 24u) & 0xFFu; ii2 = w1 & 0xFFu; }
    else { ii1 = w1 & 0xFFu; ii2 = (w1 >> 8u) & 0xFFu; }
    out[4] = ii0; out[5] = ii1; out[6] = ii2;
    out[3] = indexOffset;  // reuse: audit
}
