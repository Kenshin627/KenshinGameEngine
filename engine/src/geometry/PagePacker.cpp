#include "kenshin/geometry/PagePacker.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "kenshin/core/Log.h"

KENSHIN_NS_BEGIN

namespace geometry {

namespace {

/// Running per-page section builders; flattened to the GPU-layout blob
/// when the page closes. Sections grow independently, so the capacity
/// check accounts the combined byte total.
struct PageBuilder {
  std::vector<float> pos;   // 4/vertex (xyz, w=0)
  std::vector<float> nrm;   // 4/vertex
  std::vector<float> uv;    // 2/vertex
  std::vector<u32> idx;

  usize bytes() const {
    return (pos.size() + nrm.size()) * sizeof(float) +
           uv.size() * sizeof(float) + idx.size() * sizeof(u32);
  }

  ClusterDag::DagPageMeta flatten(std::vector<u8>& blob) const {
    ClusterDag::DagPageMeta m;
    m.vertexCount = static_cast<u32>(pos.size() / 4);
    m.indexCount = static_cast<u32>(idx.size());
    m.posBase = 0;
    m.nrmBase = static_cast<u32>(pos.size() / 4);  // float4 elements
    m.uvBase =
        static_cast<u32>((pos.size() + nrm.size()) / 2);  // float2 elems
    m.idxBase =
        static_cast<u32>((pos.size() + nrm.size()) * sizeof(float) / 4 +
                         uv.size() * sizeof(float) / 4);
    blob.resize(bytes());
    usize o = 0;
    if (!pos.empty()) {
      std::memcpy(blob.data() + o, pos.data(), pos.size() * sizeof(float));
      o += pos.size() * sizeof(float);
    }
    if (!nrm.empty()) {
      std::memcpy(blob.data() + o, nrm.data(), nrm.size() * sizeof(float));
      o += nrm.size() * sizeof(float);
    }
    if (!uv.empty()) {
      std::memcpy(blob.data() + o, uv.data(), uv.size() * sizeof(float));
      o += uv.size() * sizeof(float);
    }
    if (!idx.empty())
      std::memcpy(blob.data() + o, idx.data(), idx.size() * sizeof(u32));
    return m;
  }
};

}  // namespace

bool packPages(const ClusterDag& dag, PackedGeometry& out, std::string& log) {
  const usize clusterCount = dag.clusters.size();
  if (clusterCount == 0) {
    log = "no clusters to pack";
    return false;
  }
  const bool hasNormals = !dag.normals.empty();
  const bool hasUvs = dag.uvs.size() / 2 == dag.positions.size() / 3;

  out = {};
  out.hasNormals = hasNormals;
  out.hasUvs = hasUvs;
  out.clusterPage.resize(clusterCount);
  out.localVertexOffset.resize(clusterCount);
  out.localIndexOffset.resize(clusterCount);

  // Coarsest-first assignment (D14.3): clusters ordered by their
  // group's lodLevel descending, so the leading pages are the root
  // pages. The CLUSTER TABLE ORDER is untouched (the group-contiguity
  // invariant lives in cluster-index space).
  std::vector<u32> order(clusterCount);
  for (u32 c = 0; c < clusterCount; ++c) order[c] = c;
  std::stable_sort(order.begin(), order.end(), [&](u32 a, u32 b) {
    return dag.groups[dag.clusters[a].group].lodLevel >
           dag.groups[dag.clusters[b].group].lodLevel;
  });

  PageBuilder cur;
  u32 curVerts = 0;  // section element cursors for local offsets
  auto closePage = [&]() {
    if (cur.pos.empty() && cur.idx.empty()) return;
    out.pageMeta.push_back(cur.flatten(out.pages.emplace_back()));
  };

  for (const u32 c : order) {
    const DagCluster& cl = dag.clusters[c];
    // Byte delta if this cluster lands in the current page.
    const usize addBytes =
        usize(cl.vertexCount) * 16 * (1 + (hasNormals ? 1 : 0)) +
        usize(cl.vertexCount) * (hasUvs ? 8 : 0) + cl.indexCount * 4;
    if (!cur.idx.empty() && cur.bytes() + addBytes > kPageSlotSize)
      closePage(), cur = {}, curVerts = 0;
    if (addBytes > kPageSlotSize) {
      log = "cluster " + std::to_string(c) + " exceeds the page size";
      return false;
    }

    out.clusterPage[c] = static_cast<u32>(out.pageMeta.size());
    out.localVertexOffset[c] = curVerts;
    out.localIndexOffset[c] = static_cast<u32>(cur.idx.size());
    for (u32 v = 0; v < cl.vertexCount; ++v) {
      const usize p = usize(cl.vertexOffset + v) * 3;
      cur.pos.push_back(dag.positions[p]);
      cur.pos.push_back(dag.positions[p + 1]);
      cur.pos.push_back(dag.positions[p + 2]);
      cur.pos.push_back(0.0f);
      if (hasNormals) {
        cur.nrm.push_back(dag.normals[p]);
        cur.nrm.push_back(dag.normals[p + 1]);
        cur.nrm.push_back(dag.normals[p + 2]);
        cur.nrm.push_back(0.0f);
      }
      if (hasUvs) {
        const usize q = usize(cl.vertexOffset + v) * 2;
        cur.uv.push_back(dag.uvs[q]);
        cur.uv.push_back(dag.uvs[q + 1]);
      }
    }
    for (u32 i = 0; i < cl.indexCount; ++i)
      cur.idx.push_back(dag.indices[cl.indexOffset + i]);
    curVerts += cl.vertexCount;
  }
  closePage();
  out.pageCount = static_cast<u32>(out.pageMeta.size());

  {
    usize used = 0;
    for (const auto& p : out.pages) used += p.size();
    KENSHIN_INFO(
        "[pages] {} clusters -> {} pages ({} B slots, {:.1f}% fill, {:.2f} MB)",
        clusterCount, out.pageCount, kPageSlotSize,
        100.0 * double(used) / (double(out.pageCount) * kPageSlotSize),
        double(used) / (1024.0 * 1024.0));
  }
  return verifyPacking(dag, out, log);
}

bool verifyPacking(const ClusterDag& dag, const PackedGeometry& pg,
                   std::string& log) {
  const usize clusterCount = dag.clusters.size();
  if (pg.clusterPage.size() != clusterCount ||
      pg.localVertexOffset.size() != clusterCount ||
      pg.localIndexOffset.size() != clusterCount) {
    log = "packing: per-cluster table size mismatch";
    return false;
  }
  usize totalVerts = 0, totalIdx = 0;
  for (u32 c = 0; c < clusterCount; ++c) {
    const DagCluster& cl = dag.clusters[c];
    const u32 p = pg.clusterPage[c];
    if (p >= pg.pageCount) {
      log = "packing: cluster " + std::to_string(c) + " page out of range";
      return false;
    }
    const auto& m = pg.pageMeta[p];
    const std::vector<u8>& blob = pg.pages[p];
    if (usize(pg.localVertexOffset[c]) + cl.vertexCount > m.vertexCount ||
        usize(pg.localIndexOffset[c]) + cl.indexCount > m.indexCount) {
      log = "packing: cluster " + std::to_string(c) +
            " slice exceeds page section";
      return false;
    }
    // Bit-exact slice comparison against the dag pools.
    const float* pos = reinterpret_cast<const float*>(blob.data());
    for (u32 v = 0; v < cl.vertexCount; ++v) {
      const glm::vec4 got = *reinterpret_cast<const glm::vec4*>(
          pos + (usize(m.posBase) + pg.localVertexOffset[c] + v) * 4);
      const usize s = usize(cl.vertexOffset + v) * 3;
      if (got.x != dag.positions[s] || got.y != dag.positions[s + 1] ||
          got.z != dag.positions[s + 2] || got.w != 0.0f) {
        log = "packing: cluster " + std::to_string(c) + " position mismatch";
        return false;
      }
    }
    const u32* idx = reinterpret_cast<const u32*>(
        blob.data() + (usize(m.idxBase) + pg.localIndexOffset[c]) * 4);
    for (u32 i = 0; i < cl.indexCount; ++i)
      if (idx[i] != dag.indices[cl.indexOffset + i]) {
        log = "packing: cluster " + std::to_string(c) + " index mismatch";
        return false;
      }
    totalVerts += cl.vertexCount;
    totalIdx += cl.indexCount;
  }
  usize metaVerts = 0, metaIdx = 0, blobBytes = 0;
  for (u32 p = 0; p < pg.pageCount; ++p) {
    metaVerts += pg.pageMeta[p].vertexCount;
    metaIdx += pg.pageMeta[p].indexCount;
    blobBytes += pg.pages[p].size();
    if (pg.pages[p].size() > kPageSlotSize) {
      log = "packing: page " + std::to_string(p) + " exceeds the slot size";
      return false;
    }
  }
  if (metaVerts != totalVerts || metaIdx != totalIdx) {
    log = "packing: page totals disagree with the cluster sum";
    return false;
  }
  return true;
}

}  // namespace geometry

KENSHIN_NS_END
