#include "kenshin/geometry/PagePacker.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_map>

#include "kenshin/core/Log.h"

KENSHIN_NS_BEGIN

namespace geometry {

namespace {

constexpr u32 kMetaU32 = 10;  // ClusterQuantMeta in u32 words

void storeMeta(std::vector<u32>& words, const ClusterQuantMeta& m) {
  const u32* raw = reinterpret_cast<const u32*>(&m);
  words.insert(words.end(), raw, raw + kMetaU32);
}

ClusterQuantMeta loadMeta(const u32* words) {
  ClusterQuantMeta m;
  std::memcpy(&m, words, kMetaU32 * sizeof(u32));
  return m;
}

/// Append a byte block with 4-byte padding, returning the u32-aligned
/// size consumed.
usize appendPadded(std::vector<u8>& blob, const void* data, usize bytes) {
  blob.insert(blob.end(), static_cast<const u8*>(data),
              static_cast<const u8*>(data) + bytes);
  const usize pad = (4 - (bytes % 4)) % 4;
  blob.insert(blob.end(), usize(pad), 0);
  return bytes + pad;
}

}  // namespace

bool packPages(ClusterDag& dag, PackedGeometry& out, std::string& log) {
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
  out.grid = chooseQuantGrid(dag);
  out.clusterPage.resize(clusterCount);
  out.localVertexOffset.resize(clusterCount);
  out.localIndexOffset.resize(clusterCount);
  out.clusterMetaIndex.resize(clusterCount);

  // 1) Quantize every cluster once (exact sizes for page assignment).
  std::vector<QuantCluster> quant(clusterCount);
  for (u32 c = 0; c < clusterCount; ++c) {
    if (!quantizeCluster(dag, c, out.grid, hasNormals, hasUvs, quant[c],
                         log))
      return false;
    decodeQuantCluster(quant[c], out.grid.step);
  }

  // 2) Coarsest-first assignment (D14.3): the leading pages are the
  // root pages. Cluster-table order untouched.
  std::vector<u32> order(clusterCount);
  for (u32 c = 0; c < clusterCount; ++c) order[c] = c;
  std::stable_sort(order.begin(), order.end(), [&](u32 a, u32 b) {
    return dag.groups[dag.clusters[a].group].lodLevel >
           dag.groups[dag.clusters[b].group].lodLevel;
  });

  // Running page sections.
  std::vector<u32> metaWords;
  std::vector<u32> posWords;
  std::vector<u8> nrmBytes, uvBytes, idxBytes;
  u32 pageVerts = 0;
  auto sectionBytes = [&]() {
    return (metaWords.size() + posWords.size()) * sizeof(u32) +
           nrmBytes.size() + uvBytes.size() + idxBytes.size();
  };
  bool pageFail = false;
  auto closePage = [&]() {
    if (metaWords.empty()) return;
    // Layout: [meta][posBits][nrm][uv][idx], all bases in u32 units.
    ClusterDag::DagPageMetaV5 m;
    m.metaBase = 0;
    m.posBase = u32(metaWords.size());
    const u32 posWordsTotal = u32(posWords.size());
    m.nrmBase = m.posBase + posWordsTotal;
    const u32 nrmU32 = u32((nrmBytes.size() + 3) / 4);
    m.uvBase = m.nrmBase + nrmU32;
    const u32 uvU32 = u32((uvBytes.size() + 3) / 4);
    m.idxBase = m.uvBase + uvU32;
    const u32 idxU32 = u32((idxBytes.size() + 3) / 4);
    m.vertexCount = pageVerts;
    m.indexCount = u32(idxBytes.size());
    m.clusterCount = u32(metaWords.size() / kMetaU32);
    m.posStep = out.grid.step;

    std::vector<u8>& blob = out.pages.emplace_back();
    blob.reserve((m.idxBase + idxU32) * sizeof(u32));
    blob.insert(blob.end(), reinterpret_cast<const u8*>(metaWords.data()),
                reinterpret_cast<const u8*>(metaWords.data()) +
                    metaWords.size() * sizeof(u32));
    blob.insert(blob.end(), reinterpret_cast<const u8*>(posWords.data()),
                reinterpret_cast<const u8*>(posWords.data()) +
                    posWords.size() * sizeof(u32));
    usize o = blob.size();
    o += appendPadded(blob, nrmBytes.data(), nrmBytes.size());
    o += appendPadded(blob, uvBytes.data(), uvBytes.size());
    o += appendPadded(blob, idxBytes.data(), idxBytes.size());
    if (blob.size() != usize(m.idxBase + idxU32) * sizeof(u32) ||
        blob.size() > kPageSlotSize) {
      log = "page assembly size mismatch";
      pageFail = true;
      return;
    }
    (void)o;
    out.pageMeta.push_back(m);
  };

  for (const u32 c : order) {
    QuantCluster& q = quant[c];
    const usize addBytes = usize(kMetaU32) * 4 + q.posWords.size() * 4 +
                           q.nrmOct.size() + q.uvQ.size() + q.idx.size();
    if (!metaWords.empty() &&
        sectionBytes() + addBytes + 16 > kPageSlotSize) {
      closePage();
      metaWords.clear(), posWords.clear(), nrmBytes.clear(),
          uvBytes.clear(), idxBytes.clear(), pageVerts = 0;
    }
    if (addBytes > kPageSlotSize) {
      log = "cluster " + std::to_string(c) +
            " exceeds the page size even quantized";
      return false;
    }

    out.clusterPage[c] = u32(out.pageMeta.size());
    out.localVertexOffset[c] = pageVerts;
    out.localIndexOffset[c] = u32(idxBytes.size());
    out.clusterMetaIndex[c] = u32(metaWords.size() / kMetaU32);

    q.meta.posWordOff = u32(posWords.size());
    storeMeta(metaWords, q.meta);
    posWords.insert(posWords.end(), q.posWords.begin(), q.posWords.end());
    nrmBytes.insert(nrmBytes.end(), q.nrmOct.begin(), q.nrmOct.end());
    uvBytes.insert(uvBytes.end(), q.uvQ.begin(), q.uvQ.end());
    idxBytes.insert(idxBytes.end(), q.idx.begin(), q.idx.end());
    pageVerts += q.vertexCount;
  }
  closePage();
  out.pageCount = u32(out.pageMeta.size());
  if (pageFail) return false;

  // 3) Cull-bound padding: the decoded geometry can sit up to one grid
  // step outside the raw-float cull volumes.
  const f32 step = out.grid.step;
  for (DagCluster& cl : dag.clusters) cl.boundsSphere.w += step;
  for (DagGroup& g : dag.groups) g.boundsSphere.w += step;
  for (DagHierarchyNode& n : dag.hierarchyNodes)
    for (int k = 0; k < 3; ++k) {
      n.aabbMin[k] -= step;
      n.aabbMax[k] += step;
    }

  // 4) Decoded mirror pools in DAG order (the v2-shaped reference).
  for (u32 c = 0; c < clusterCount; ++c) {
    const QuantCluster& q = quant[c];
    out.decodedPositions.insert(out.decodedPositions.end(), q.decPos.begin(),
                                q.decPos.end());
    out.decodedNormals.insert(out.decodedNormals.end(), q.decNrm.begin(),
                              q.decNrm.end());
    out.decodedUvs.insert(out.decodedUvs.end(), q.decUv.begin(), q.decUv.end());
    out.decodedIndices.insert(out.decodedIndices.end(), q.decIdx.begin(),
                              q.decIdx.end());
  }

  {
    usize used = 0;
    for (const auto& p : out.pages) used += p.size();
    KENSHIN_INFO(
        "[pages] {} clusters -> {} pages ({} B slots, {:.1f}% fill, {:.2f} "
        "MB; quant grid 2^-{}={:.6f})",
        clusterCount, out.pageCount, kPageSlotSize,
        100.0 * double(used) / (double(out.pageCount) * kPageSlotSize),
        double(used) / (1024.0 * 1024.0), out.grid.P, out.grid.step);
  }
  return verifyPacking(dag, out, log);
}

bool verifyPacking(const ClusterDag& dag, const PackedGeometry& pg,
                   std::string& log) {
  const usize clusterCount = dag.clusters.size();
  if (pg.clusterPage.size() != clusterCount ||
      pg.localVertexOffset.size() != clusterCount ||
      pg.localIndexOffset.size() != clusterCount ||
      pg.clusterMetaIndex.size() != clusterCount) {
    log = "packing: per-cluster table size mismatch";
    return false;
  }
  const bool hasNormals = !dag.normals.empty();
  const bool hasUvs = dag.uvs.size() / 2 == dag.positions.size() / 3;
  const f32 step = pg.grid.step;

  usize totalVerts = 0, totalIdx = 0, blobBytes = 0;
  f32 maxPosErr = 0.0f, maxUvErr = 0.0f, maxAngle = 0.0f;
  // Watertight: same source float triple -> same decoded bits.
  std::unordered_map<std::string, std::array<f32, 3>> decodedOf;

  for (u32 c = 0; c < clusterCount; ++c) {
    const DagCluster& cl = dag.clusters[c];
    const u32 p = pg.clusterPage[c];
    if (p >= pg.pageCount) {
      log = "packing: cluster " + std::to_string(c) + " page out of range";
      return false;
    }
    const ClusterDag::DagPageMetaV5& m = pg.pageMeta[p];
    const std::vector<u8>& blob = pg.pages[p];
    const u32* words = reinterpret_cast<const u32*>(blob.data());
    const usize wordTotal = blob.size() / 4;
    if (usize(m.metaBase) + usize(m.clusterCount) * kMetaU32 > wordTotal ||
        pg.clusterMetaIndex[c] >= m.clusterCount) {
      log = "packing: page " + std::to_string(p) + " meta out of range";
      return false;
    }
    const ClusterQuantMeta meta =
        loadMeta(words + m.metaBase + usize(pg.clusterMetaIndex[c]) * kMetaU32);
    const u32 b[3] = {meta.widths & 0xFu, (meta.widths >> 4) & 0xFu,
                      (meta.widths >> 8) & 0xFu};
    const u32 nV = cl.vertexCount;
    const u64 axisBase[3] = {u64(meta.posWordOff) * 32,
                             u64(meta.posWordOff) * 32 + u64(nV) * b[0],
                             u64(meta.posWordOff) * 32 +
                                 u64(nV) * (b[0] + b[1])};
    const u32* posSec = words + m.posBase;

    for (u32 v = 0; v < nV; ++v) {
      // Re-decode from the PAGE BLOB bits.
      f32 dec[3];
      for (int a = 0; a < 3; ++a) {
        const u32 bits = readBits(posSec, axisBase[a] + u64(v) * b[a], b[a]);
        dec[a] = f32(i32(bits) + meta.posStart[a]) * m.posStep;
      }
      // vs the decoded mirror: bit-identical (D5).
      const f32* mirror =
          pg.decodedPositions.data() + usize(cl.vertexOffset + v) * 3;
      if (dec[0] != mirror[0] || dec[1] != mirror[1] || dec[2] != mirror[2]) {
        log = "packing: cluster " + std::to_string(c) +
              " page/mirror decode mismatch";
        return false;
      }
      // vs the dag raw: within the quantizer's bound.
      const f32* src = dag.positions.data() + usize(cl.vertexOffset + v) * 3;
      for (int a = 0; a < 3; ++a) {
        const f32 err = std::fabs(src[a] - dec[a]);
        maxPosErr = std::max(maxPosErr, err);
        if (err > step * 0.5f + 1e-7f) {
          log = "packing: cluster " + std::to_string(c) +
                " position error over step/2";
          return false;
        }
      }
      // Watertight bookkeeping.
      char key[64];
      std::snprintf(key, sizeof key, "%08x%08x%08x",
                    *(const u32*)&src[0], *(const u32*)&src[1],
                    *(const u32*)&src[2]);
      const std::array<f32, 3> dst = {dec[0], dec[1], dec[2]};
      auto it = decodedOf.find(key);
      if (it == decodedOf.end())
        decodedOf.emplace(key, dst);
      else if (it->second != dst) {
        log = "packing: shared vertex decodes differently (crack)";
        return false;
      }

      if (hasNormals) {
        // Oct decode from the page bytes; angle bound vs the raw.
        const u8* pair =
            blob.data() + usize(m.nrmBase) * 4 + usize(pg.localVertexOffset[c] + v) * 2;
        u8 q2[2] = {pair[0], pair[1]};
        const glm::vec3 n = octDecode(q2);
        const f32* raw =
            dag.normals.data() + usize(cl.vertexOffset + v) * 3;
        const f32 angle =
            std::acos(std::clamp(glm::dot(
                                       glm::normalize(glm::vec3(
                                           raw[0], raw[1], raw[2])),
                                       n),
                                   -1.0f, 1.0f));
        maxAngle = std::max(maxAngle, angle);
      }
      if (hasUvs) {
        const u8* qp =
            blob.data() + usize(m.uvBase) * 4 + usize(pg.localVertexOffset[c] + v) * 2;
        const f32* src =
            dag.uvs.data() + usize(cl.vertexOffset + v) * 2;
        for (int a = 0; a < 2; ++a) {
          const f32 got = meta.uvMin[a] + f32(qp[a]) * (meta.uvRange[a] / 255.0f);
          const f32 err = std::fabs(src[a] - got);
          maxUvErr = std::max(maxUvErr, err);
          if (err > meta.uvRange[a] / 508.0f + 1e-7f) {
            log = "packing: cluster " + std::to_string(c) + " uv error";
            return false;
          }
        }
      }
    }
    // Indices: exact u8 roundtrip from the page bytes.
    const u8* idxSec = blob.data() + usize(m.idxBase) * 4;
    for (u32 i = 0; i < cl.indexCount; ++i) {
      const u8 got = idxSec[pg.localIndexOffset[c] + i];
      if (u32(got) != dag.indices[usize(cl.indexOffset) + i]) {
        log = "packing: cluster " + std::to_string(c) + " index mismatch";
        return false;
      }
    }
    totalVerts += nV;
    totalIdx += cl.indexCount;
  }
  usize metaVerts = 0, metaIdx = 0;
  for (u32 p = 0; p < pg.pageCount; ++p) {
    metaVerts += pg.pageMeta[p].vertexCount;
    metaIdx += pg.pageMeta[p].indexCount;
    blobBytes += pg.pages[p].size();
    if (pg.pages[p].size() > kPageSlotSize) {
      log = "packing: page " + std::to_string(p) + " exceeds the slot size";
      return false;
    }
    if (pg.pageMeta[p].posStep != pg.grid.step) {
      log = "packing: page " + std::to_string(p) + " posStep mismatch";
      return false;
    }
  }
  if (metaVerts != totalVerts || metaIdx != totalIdx) {
    log = "packing: page totals disagree with the cluster sum";
    return false;
  }
  KENSHIN_INFO(
      "[pages] verify: maxPosErr={:.7f} maxUvErr={:.7f} maxAngle={:.2f} deg "
      "({} blobs, {:.2f} MB)",
      maxPosErr, maxUvErr, maxAngle * 180.0f / 3.14159265f, pg.pageCount,
      double(blobBytes) / (1024.0 * 1024.0));
  return true;
}

}  // namespace geometry

KENSHIN_NS_END
