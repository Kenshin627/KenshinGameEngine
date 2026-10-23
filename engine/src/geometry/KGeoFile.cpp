#include "kenshin/geometry/KGeoFile.h"

#include <cstdio>
#include <cstring>
#include <glm/glm.hpp>
#include <vector>

#include "kenshin/core/Log.h"
#include "kenshin/util/File.h"
#include "meshoptimizer/meshoptimizer.h"  // M7.2b page codec

KENSHIN_NS_BEGIN

namespace geometry {

namespace {
constexpr u32 kMagic = 0x324F474B;  // "KGO2"
constexpr u32 kVersion = 2;
constexpr u32 kMagicV3 = 0x334F474B;  // "KGO3" (M6.1 paged)
constexpr u32 kVersion3 = 3;          // paged, no maxEdge table
constexpr u32 kVersion4 = 4;          // paged + per-cluster maxEdge table
constexpr u32 kVersion5 =
    5;  // M7.1: quantized pages + clusterMetaIndex + 12-u32 page meta
constexpr u32 kVersion6 =
    6;  // M7.2: packed triangle indices (same table layout as v5; the idx section becomes a bit stream and the meta's 10th word carries the widths/bit base). Old assets are loadable for TABLE reads but the RUNTIME paged path requires v6 (pageIdxPacked).
constexpr u32 kVersion7 =
    7;  // M8.2: material table. A 17th header u32 (materialCount)
        // follows the fixed 16-u32 header, and a material section
        // sits between the maxEdge table and the page blobs: a string
        // blob (albedo paths) + per-material path offsets + 8-f32
        // records. The per-cluster material id rides the clusterMeta
        // widths word's high bits (12..19). The RUNTIME paged path
        // requires v7 (hasMaterialTable).
constexpr u32 kVersion8 = 8;
constexpr u32 kVersion9 = 9;    // M9.2: v8 + the imposter atlas section
constexpr u32 kVersion10 = 10;  // v9 + per-material metallicRoughness map
// v13: v11 + per-material EMISSIVE - material records grow to 12 f32
// (v11's 8 + emissive rgb + pad). Skips 12, which the reverted
// per-triangle-material experiment (a3813c8) used, so no v12 file can be
// misread as this layout. v11 files keep their 8-f32 records.
constexpr u32 kVersion13 = 13;
constexpr u32 kVersion11 = 11;  // v10 + per-material NORMAL map (a third
                                // path-offset array after mrOff; the
                                // record's 8th float flags presence)
// M7.2b: v7 with meshopt-CODEC page blobs on disk (transcoded to the
// v7 byte layout on every read; the in-memory/runtime format and all
// tables are identical to v7). The tail directory entry grows to 12
// bytes: {fileOffset, storedLen, rawLen}. storedLen == rawLen marks a
// page stored raw (codec inflation fallback).

constexpr bool isV7Plus(u32 v) {
  return v == kVersion7 || v == kVersion8 || v == kVersion9 ||
         v == kVersion10 || v == kVersion11 || v == kVersion13;
}
constexpr bool isV9Plus(u32 v) {
  return v == kVersion9 || v == kVersion10 || v == kVersion11 ||
         v == kVersion13;
}
constexpr bool isV10Plus(u32 v) {
  return v == kVersion10 || v == kVersion11 || v == kVersion13;
}
// Carries the nrmOff[] array (v11 and its v13 extension).
constexpr bool hasNrmTable(u32 v) { return v == kVersion11 || v == kVersion13; }

#pragma pack(push, 4)
struct FileHeader {
  u32 magic;
  u32 version;
  u32 groupCount;
  u32 clusterCount;
  u32 childEdgeCount;
  u32 rootCount;
  u32 poolVertexCount;
  u32 poolIndexCount;
  u32 hasNormals;
  u32 hasUvs;
  u32 sourceVertexCount;
  u32 sourceTriangleCount;
  u32 hierarchyNodeCount;  // v2 (M5.1): the group BVH8
  u32 hierarchyRoot;
};

// v3 (M6.1): the pool counts carry the PAGE-VERTEX/INDEX totals (the
// conservation numbers), and the page machinery follows.
struct FileHeaderV3 {
  u32 magic;
  u32 version;
  u32 groupCount;
  u32 clusterCount;
  u32 childEdgeCount;
  u32 rootCount;
  u32 pageVertexTotal;
  u32 pageIndexTotal;
  u32 hasNormals;
  u32 hasUvs;
  u32 sourceVertexCount;
  u32 sourceTriangleCount;
  u32 hierarchyNodeCount;
  u32 hierarchyRoot;
  u32 pageCount;
  u32 pageSlotSize;
};

struct PageDirEntry {
  u32 fileOffset;
  u32 byteLen;
};

// M7.2b v8: compressed page blobs need the RAW length for the decode;
// the v7 entry stays untouched (older files parse as today).
struct PageDirEntryV8 {
  u32 fileOffset;
  u32 storedLen;  // bytes in the file (compressed, or raw on fallback)
  u32 rawLen;     // decoded bytes (== storedLen when stored raw)
};

struct GroupDisk {
  i32 lodLevel;
  f32 maxError;       // monotonic subtree error (GPU rule data)
  f32 simplifyError;  // clusterlod semantics: error of coarsening FROM
                      // this group; FLT_MAX on terminal groups
  f32 bounds[4];      // xyz + radius
  u32 memberOffset;
  u32 memberCount;
  u32 parentOffset;
  u32 parentCount;
};

struct ClusterDisk {
  f32 bounds[4];
  f32 error;
  f32 parentError;
  u32 vertexOffset;
  u32 vertexCount;
  u32 indexOffset;
  u32 indexCount;
  u32 group;
  u32 childOffset;
  u32 childCount;
  u32 isRoot;
  u32 refinedGroup;  // clodCluster::refined, 0xFFFFFFFF = -1
};

// v2 (M5.1): the group BVH8 nodes, flat on disk exactly as the engine
// struct (leaf = one group; see ClusterDag.h).
static_assert(sizeof(DagHierarchyNode) == 64, "hierarchy node layout");
#pragma pack(pop)

/// Span arrays are packed end-inclusively: offsets[i] = start of
/// span i, count in the parallel field; offset[0] = 0.
template <typename T>
bool writeVector(std::FILE* file, const std::vector<T>& data) {
  return data.empty() ||
         std::fwrite(data.data(), sizeof(T), data.size(), file) == data.size();
}

template <typename T>
bool readVector(std::FILE* file, std::vector<T>& data, usize count) {
  data.resize(count);
  return count == 0 || std::fread(data.data(), sizeof(T), count, file) == count;
}
}  // namespace

// v3 table-only load + the shared table rehydration (defined below,
// after the writers; FileHeaderV3/PageDir are anon-namespace types so
// these stay internal by ODR-friendliness of identical TU scope).
bool readKGeoPagedTables(const std::string& path, const FileHeaderV3& h3,
                         u32 materialCount, ClusterDag& dag, std::string& log);
bool rehydrateTables(std::FILE* file, u32 groupCount, u32 clusterCount,
                     u32 childEdgeCount, u32 rootCount, u32 hierarchyCount,
                     ClusterDag& dag, std::string& log);

bool writeKGeo(const std::string& path, const ClusterDag& dag,
               std::string& log) {
  std::FILE* file = nullptr;
  if ((file = kenshin::util::openFile(path.c_str(), "wb")) == nullptr) {
    log = "cannot open for write: " + path;
    return false;
  }

  // Pack member/parent/child spans.
  std::vector<u32> memberIndices, parentIndices, childIndices;
  std::vector<GroupDisk> groups(dag.groups.size());
  for (u32 g = 0; g < dag.groups.size(); ++g) {
    const DagGroup& src = dag.groups[g];
    groups[g] = GroupDisk{
        .lodLevel = src.lodLevel,
        .maxError = src.maxError,
        .simplifyError = src.simplifyError,
        .bounds = {src.boundsSphere.x, src.boundsSphere.y, src.boundsSphere.z,
                   src.boundsSphere.w},
        .memberOffset = static_cast<u32>(memberIndices.size()),
        .memberCount = static_cast<u32>(src.clusters.size()),
        .parentOffset = static_cast<u32>(parentIndices.size()),
        .parentCount = static_cast<u32>(src.parents.size()),
    };
    memberIndices.insert(memberIndices.end(), src.clusters.begin(),
                         src.clusters.end());
    parentIndices.insert(parentIndices.end(), src.parents.begin(),
                         src.parents.end());
  }

  std::vector<ClusterDisk> clusters(dag.clusters.size());
  for (u32 c = 0; c < dag.clusters.size(); ++c) {
    const DagCluster& src = dag.clusters[c];
    clusters[c] = ClusterDisk{
        .bounds = {src.boundsSphere.x, src.boundsSphere.y, src.boundsSphere.z,
                   src.boundsSphere.w},
        .error = src.error,
        .parentError = src.parentError,
        .vertexOffset = src.vertexOffset,
        .vertexCount = src.vertexCount,
        .indexOffset = src.indexOffset,
        .indexCount = src.indexCount,
        .group = src.group,
        .childOffset = static_cast<u32>(childIndices.size()),
        .childCount = static_cast<u32>(src.children.size()),
        .isRoot = src.isRoot ? 1u : 0u,
        .refinedGroup = src.refinedGroup < 0
                            ? 0xFFFFFFFFu
                            : static_cast<u32>(src.refinedGroup),
    };
    childIndices.insert(childIndices.end(), src.children.begin(),
                        src.children.end());
  }

  const FileHeader header{
      .magic = kMagic,
      .version = kVersion,
      .groupCount = static_cast<u32>(dag.groups.size()),
      .clusterCount = static_cast<u32>(dag.clusters.size()),
      .childEdgeCount = static_cast<u32>(childIndices.size()),
      .rootCount = static_cast<u32>(dag.roots.size()),
      .poolVertexCount = static_cast<u32>(dag.positions.size() / 3),
      .poolIndexCount = static_cast<u32>(dag.indices.size()),
      .hasNormals = dag.normals.empty() ? 0u : 1u,
      .hasUvs = dag.uvs.size() / 2 == dag.positions.size() / 3 ? 1u : 0u,
      .sourceVertexCount = dag.sourceVertexCount,
      .sourceTriangleCount = dag.sourceTriangleCount,
      .hierarchyNodeCount = static_cast<u32>(dag.hierarchyNodes.size()),
      .hierarchyRoot = dag.hierarchyRoot,
  };

  const bool ok =
      std::fwrite(&header, sizeof(header), 1, file) == 1 &&
      writeVector(file, groups) && writeVector(file, clusters) &&
      writeVector(file, childIndices) && writeVector(file, memberIndices) &&
      writeVector(file, parentIndices) &&
      writeVector(file, dag.hierarchyNodes) && writeVector(file, dag.roots) &&
      writeVector(file, dag.positions) &&
      (header.hasNormals == 0 || writeVector(file, dag.normals)) &&
      (header.hasUvs == 0 || writeVector(file, dag.uvs)) &&
      writeVector(file, dag.indices);
  std::fclose(file);
  if (!ok) log = "write failed (disk full?)";
  return ok;
}

bool writeKGeoPaged(const std::string& path, const ClusterDag& dag,
                    const PackedGeometry& pg, std::string& log, bool compress) {
  if (pg.pageCount != pg.pageMeta.size() || pg.pageCount != pg.pages.size() ||
      pg.clusterPage.size() != dag.clusters.size()) {
    log = "paged write: packing table size mismatch";
    return false;
  }
  std::FILE* file = nullptr;
  if ((file = kenshin::util::openFile(path.c_str(), "wb")) == nullptr) {
    log = "cannot open for write: " + path;
    return false;
  }

  std::vector<u32> memberIndices, parentIndices, childIndices;
  std::vector<GroupDisk> groups(dag.groups.size());
  for (u32 g = 0; g < dag.groups.size(); ++g) {
    const DagGroup& src = dag.groups[g];
    groups[g] = GroupDisk{
        .lodLevel = src.lodLevel,
        .maxError = src.maxError,
        .simplifyError = src.simplifyError,
        .bounds = {src.boundsSphere.x, src.boundsSphere.y, src.boundsSphere.z,
                   src.boundsSphere.w},
        .memberOffset = static_cast<u32>(memberIndices.size()),
        .memberCount = static_cast<u32>(src.clusters.size()),
        .parentOffset = static_cast<u32>(parentIndices.size()),
        .parentCount = static_cast<u32>(src.parents.size()),
    };
    memberIndices.insert(memberIndices.end(), src.clusters.begin(),
                         src.clusters.end());
    parentIndices.insert(parentIndices.end(), src.parents.begin(),
                         src.parents.end());
  }
  // ClusterDisk offsets are PAGE-LOCAL in v3.
  std::vector<ClusterDisk> clusters(dag.clusters.size());
  for (u32 c = 0; c < dag.clusters.size(); ++c) {
    const DagCluster& src = dag.clusters[c];
    clusters[c] = ClusterDisk{
        .bounds = {src.boundsSphere.x, src.boundsSphere.y, src.boundsSphere.z,
                   src.boundsSphere.w},
        .error = src.error,
        .parentError = src.parentError,
        .vertexOffset = pg.localVertexOffset[c],
        .vertexCount = src.vertexCount,
        .indexOffset = pg.localIndexOffset[c],
        .indexCount = src.indexCount,
        .group = src.group,
        .childOffset = static_cast<u32>(childIndices.size()),
        .childCount = static_cast<u32>(src.children.size()),
        .isRoot = src.isRoot ? 1u : 0u,
        .refinedGroup = src.refinedGroup < 0
                            ? 0xFFFFFFFFu
                            : static_cast<u32>(src.refinedGroup),
    };
    childIndices.insert(childIndices.end(), src.children.begin(),
                        src.children.end());
  }

  usize pageVerts = 0, pageIdx = 0;
  std::vector<u32> pageMetaFlat;
  pageMetaFlat.reserve(usize(pg.pageCount) * 12);
  for (const auto& m : pg.pageMeta) {
    pageVerts += m.vertexCount;
    pageIdx += m.indexCount;
    const u32 stepBits = *reinterpret_cast<const u32*>(&m.posStep);
    pageMetaFlat.insert(
        pageMetaFlat.end(),
        {m.posBase, m.nrmBase, m.uvBase, m.idxBase, m.metaBase, m.vertexCount,
         m.indexCount, m.clusterCount, stepBits, m.flags, m.pad0, m.pad1});
  }

  // v7 always carries a material table; synthesize the single
  // default entry when the dag has none (the builder sets it from
  // MeshData::Material / CLI overrides).
  std::vector<ClusterDag::DagMaterial> mats = dag.materials;
  if (mats.empty()) mats.emplace_back();

  const bool hasImposter = !dag.imposterDepth.empty() &&
                           dag.imposterDepth.size() == dag.imposterTri.size();
  bool hasMr = false, hasNrm = false, hasEmissive = false;
  for (const auto& m : mats) {
    if (!m.mrPath.empty()) hasMr = true;
    if (!m.nrmPath.empty()) hasNrm = true;
    if (m.emissive[0] > 0.0f || m.emissive[1] > 0.0f || m.emissive[2] > 0.0f ||
        (m.occlusion > 0.0f && !m.mrPath.empty()))
      hasEmissive = true;  // the 12-f32 record (emissive + occlusion)
  }
  // v11 carries the v10 layout plus the normal-map paths (a normal map
  // without an MR map still writes the v10 arrays - mrOff all-absent).
  // v13 (emissive) records are 12 f32; v11 keeps 8. Both carry nrmOff.
  const bool isV13 = hasEmissive;
  const bool isV11 = hasNrm || isV13;
  const bool isV10 = hasMr || isV11;
  const u32 version = isV13         ? kVersion13
                      : isV11       ? kVersion11
                      : isV10       ? kVersion10
                      : hasImposter ? kVersion9
                                    : (compress ? kVersion8 : kVersion7);
  const FileHeaderV3 header{
      .magic = kMagicV3,
      .version = version,
      .groupCount = static_cast<u32>(dag.groups.size()),
      .clusterCount = static_cast<u32>(dag.clusters.size()),
      .childEdgeCount = static_cast<u32>(childIndices.size()),
      .rootCount = static_cast<u32>(dag.roots.size()),
      .pageVertexTotal = static_cast<u32>(pageVerts),
      .pageIndexTotal = static_cast<u32>(pageIdx),
      .hasNormals = pg.hasNormals ? 1u : 0u,
      .hasUvs = pg.hasUvs ? 1u : 0u,
      .sourceVertexCount = dag.sourceVertexCount,
      .sourceTriangleCount = dag.sourceTriangleCount,
      .hierarchyNodeCount = static_cast<u32>(dag.hierarchyNodes.size()),
      .hierarchyRoot = dag.hierarchyRoot,
      .pageCount = pg.pageCount,
      .pageSlotSize = kPageSlotSize,
  };

  // Per-cluster max triangle edge (the routing's EdgeLength; the same
  // formula as renderer::computeClusterAabbs). v5 computes it from the
  // DECODED mirror pools - routing sees what actually rasterizes; v4
  // read the raw dag pools.
  std::vector<f32> maxEdge(dag.clusters.size(), 0.0f);
  for (u32 c = 0; c < dag.clusters.size(); ++c) {
    const DagCluster& cl = dag.clusters[c];
    const f32* pos = pg.decodedPositions.data();
    const u32* idx = pg.decodedIndices.data();
    f32 e = 0.0f;
    for (u32 t = 0; t + 2 < cl.indexCount; t += 3) {
      const glm::vec3 a(
          pos[usize(cl.vertexOffset + idx[usize(cl.indexOffset + t)]) * 3],
          pos[usize(cl.vertexOffset + idx[usize(cl.indexOffset + t)]) * 3 + 1],
          pos[usize(cl.vertexOffset + idx[usize(cl.indexOffset + t)]) * 3 + 2]);
      const glm::vec3 b(
          pos[usize(cl.vertexOffset + idx[usize(cl.indexOffset + t + 1)]) * 3],
          pos[usize(cl.vertexOffset + idx[usize(cl.indexOffset + t + 1)]) * 3 +
              1],
          pos[usize(cl.vertexOffset + idx[usize(cl.indexOffset + t + 1)]) * 3 +
              2]);
      const glm::vec3 c3(
          pos[usize(cl.vertexOffset + idx[usize(cl.indexOffset + t + 2)]) * 3],
          pos[usize(cl.vertexOffset + idx[usize(cl.indexOffset + t + 2)]) * 3 +
              1],
          pos[usize(cl.vertexOffset + idx[usize(cl.indexOffset + t + 2)]) * 3 +
              2]);
      e = std::max(
          {e, glm::length(b - a), glm::length(c3 - b), glm::length(a - c3)});
    }
    maxEdge[c] = e;
  }

  // v7: the 17th header word (materialCount) - the v2->v3 append
  // precedent; v7 readers seek past it, older readers never get here.
  const u32 materialCount = static_cast<u32>(mats.size());
  bool ok = std::fwrite(&header, sizeof(header), 1, file) == 1 &&
            std::fwrite(&materialCount, sizeof(materialCount), 1, file) == 1 &&
            writeVector(file, groups) && writeVector(file, clusters) &&
            writeVector(file, childIndices) &&
            writeVector(file, memberIndices) &&
            writeVector(file, parentIndices) &&
            writeVector(file, dag.hierarchyNodes) &&
            writeVector(file, dag.roots) && writeVector(file, pg.clusterPage) &&
            writeVector(file, pg.clusterMetaIndex) &&
            writeVector(file, pageMetaFlat) && writeVector(file, maxEdge);

  // v7 material section (before the blobs - the tail-directory math
  // is offset-relative and stays untouched): string blob + per-
  // material path offsets + 8-f32 records (kd, texFlag, rough, metal).
  std::vector<u32> texOff(mats.size(), 0xFFFFFFFFu);
  std::vector<u32> mrOff(mats.size(), 0xFFFFFFFFu);
  std::vector<u32> nrmOff(mats.size(), 0xFFFFFFFFu);
  std::string strings;
  for (usize m = 0; m < mats.size(); ++m) {
    if (mats[m].texPath.empty()) continue;
    texOff[m] = static_cast<u32>(strings.size());
    strings += mats[m].texPath;
    strings.push_back('\0');
  }
  for (usize m = 0; m < mats.size(); ++m) {
    if (mats[m].mrPath.empty()) continue;
    mrOff[m] = static_cast<u32>(strings.size());
    strings += mats[m].mrPath;
    strings.push_back('\0');
  }
  for (usize m = 0; m < mats.size(); ++m) {
    if (mats[m].nrmPath.empty()) continue;
    nrmOff[m] = static_cast<u32>(strings.size());
    strings += mats[m].nrmPath;
    strings.push_back('\0');
  }
  while (strings.size() % 4 != 0) strings.push_back('\0');
  if (ok) {
    const usize recF = isV13 ? 12 : 8;
    std::vector<f32> records(mats.size() * recF);
    for (usize m = 0; m < mats.size(); ++m) {
      f32* r = records.data() + m * recF;
      r[0] = mats[m].kd[0];
      r[1] = mats[m].kd[1];
      r[2] = mats[m].kd[2];
      r[3] = texOff[m] == 0xFFFFFFFFu ? -1.0f : 0.0f;  // has-texture flag
      r[4] = mats[m].roughness;
      r[5] = mats[m].metallic;
      r[6] = (isV10 && mrOff[m] == 0xFFFFFFFFu) ? -1.0f : 0.0f;
      // v11: normal-map presence (-1 absent); pre-v11 writes 0 = pad.
      r[7] = (isV11 && nrmOff[m] == 0xFFFFFFFFu) ? -1.0f : 0.0f;
      if (isV13)
      {
        for (int c = 0; c < 3; ++c) r[8 + c] = mats[m].emissive[c];
        r[11] = mats[m].occlusion;  // was a zero pad: old v13 = no AO
      }
    }
    const u32 stringBytes = static_cast<u32>(strings.size());
    ok = std::fwrite(&stringBytes, sizeof(stringBytes), 1, file) == 1 &&
         (stringBytes == 0 ||
          std::fwrite(strings.data(), 1, stringBytes, file) == stringBytes) &&
         writeVector(file, texOff) && (!isV10 || writeVector(file, mrOff)) &&
         (!isV11 || writeVector(file, nrmOff)) && writeVector(file, records);
    if (ok && isV10) {
      const u32 imposterFlag = hasImposter ? 1u : 0u;
      ok = std::fwrite(&imposterFlag, sizeof(imposterFlag), 1, file) == 1;
    }
  }

  // M9.2 v9: the imposter atlas (12x12 dirs x 12x12 texels of
  // depth-bits + tri-id u32s; raw - 166 KB worst case, the codec gain
  // on mostly-0xFFFFFFFF tri words is poor and the section is
  // latency-critical at injection time). A u32 header word first
  // (0xFFFFFFFF = absent, for pre-v9 readers that land here via
  // version confusion).
  if (hasImposter) {
    const u32 atlasWords = static_cast<u32>(dag.imposterDepth.size());
    ok = ok && std::fwrite(&atlasWords, sizeof(atlasWords), 1, file) == 1 &&
         std::fwrite(&dag.imposterCluster, sizeof(u32), 1, file) == 1 &&
         writeVector(file, dag.imposterDepth) &&
         writeVector(file, dag.imposterTri);
    if (ok)
      KENSHIN_INFO("[builder] imposter atlas: {} dirs, {} words",
                   atlasWords / (12u * 12u * 12u * 12u) * 0u + 12u * 12u,
                   atlasWords);
  }

  // Blobs (16 B aligned, offsets recorded); the DIRECTORY goes to the
  // file tail so the writer stays single-pass - the reader lands on it
  // via fileSize - pageCount * dirEntryBytes(version).
  // M7.2b v8: each blob rides the meshopt codec (the u32 stream,
  // stride 4; padded to 4 so the codec sees whole words). A page that
  // would INFLATE is stored raw instead (storedLen == rawLen flags
  // it) - decode stays lossless either way.
  std::vector<PageDirEntryV8> dir(pg.pageCount);
  std::vector<u8> codec;
  for (u32 p = 0; p < pg.pageCount && ok; ++p) {
    // 64-bit: .kgeo assets cross the 2GB long boundary at scale.
    const long long rawPos = util::tell64(file);
    ok = rawPos >= 0;
    if (!ok) break;
    const usize pos = static_cast<usize>(rawPos);
    const usize pad = (16 - (pos % 16)) % 16;
    for (usize b = 0; b < pad && ok; ++b) ok = std::fputc(0, file) != EOF;
    const long long dataPos = ok ? util::tell64(file) : -1;
    ok = ok && dataPos >= 0;
    if (!ok) break;
    const std::vector<u8>& raw = pg.pages[p];
    u32 storedLen = static_cast<u32>(raw.size());
    u32 rawLen = storedLen;
    if (compress && !raw.empty()) {
      const usize words = (raw.size() + 3u) / 4u;
      std::vector<u8> padded(raw);
      padded.resize(words * 4u, 0);
      codec.resize(meshopt_encodeVertexBufferBound(words, 4u));
      const usize enc = meshopt_encodeVertexBuffer(codec.data(), codec.size(),
                                                   padded.data(), words, 4u);
      if (enc > 0 && enc < raw.size()) {
        storedLen = static_cast<u32>(enc);
        ok = std::fwrite(codec.data(), 1, enc, file) == enc;
        dir[p] = PageDirEntryV8{static_cast<u32>(dataPos), storedLen, rawLen};
        continue;
      }
      // Inflation (or codec refusal): fall through, store raw.
    }
    dir[p] = PageDirEntryV8{static_cast<u32>(dataPos), storedLen, rawLen};
    ok = raw.empty() ||
         std::fwrite(raw.data(), 1, raw.size(), file) == raw.size();
  }
  if (ok) ok = writeVector(file, dir);
  std::fclose(file);
  if (!ok) log = "paged write failed (disk full?)";
  return ok;
}

bool readKGeo(const std::string& path, ClusterDag& dag, std::string& log) {
  std::FILE* file = nullptr;
  if ((file = kenshin::util::openFile(path.c_str(), "rb")) == nullptr) {
    log = "cannot open for read: " + path;
    return false;
  }

  FileHeader header{};
  if (std::fread(&header, sizeof(header), 1, file) != 1 ||
      (header.magic != kMagic || header.version != kVersion)) {
    // v3? Read the v3 header over the same leading 14 u32 + 2 more.
    FileHeaderV3 h3{};
    std::rewind(file);
    if (std::fread(&h3, sizeof(h3), 1, file) != 1 || h3.magic != kMagicV3 ||
        (h3.version != kVersion3 && h3.version != kVersion4 &&
         h3.version != kVersion5 && h3.version != kVersion6 &&
         !isV7Plus(h3.version))) {
      log = "bad magic/version";
      std::fclose(file);
      return false;
    }
    // v7: the 17th header word (materialCount) follows the fixed
    // header; hand it to the table reader alongside.
    u32 materialCount = 0;
    if (isV7Plus(h3.version) &&
        std::fread(&materialCount, sizeof(materialCount), 1, file) != 1) {
      log = "truncated (materialCount)";
      std::fclose(file);
      return false;
    }
    std::fclose(file);
    return readKGeoPagedTables(path, h3, materialCount, dag, log);
  }

  std::vector<GroupDisk> groups;
  std::vector<ClusterDisk> clusters;
  std::vector<u32> childIndices, memberIndices, parentIndices;
  if (!readVector(file, groups, header.groupCount) ||
      !readVector(file, clusters, header.clusterCount) ||
      !readVector(file, childIndices, header.childEdgeCount)) {
    log = "truncated (groups/clusters/children)";
    std::fclose(file);
    return false;
  }

  usize memberTotal = 0, parentTotal = 0;
  for (const auto& g : groups) {
    memberTotal += g.memberCount;
    parentTotal += g.parentCount;
  }
  if (!readVector(file, memberIndices, memberTotal) ||
      !readVector(file, parentIndices, parentTotal) ||
      !readVector(file, dag.hierarchyNodes, header.hierarchyNodeCount) ||
      !readVector(file, dag.roots, header.rootCount) ||
      !readVector(file, dag.positions, usize(header.poolVertexCount) * 3)) {
    log = "truncated (members/parents/roots/positions)";
    std::fclose(file);
    return false;
  }
  if (header.hasNormals &&
      !readVector(file, dag.normals, usize(header.poolVertexCount) * 3)) {
    log = "truncated (normals)";
    std::fclose(file);
    return false;
  }
  if (header.hasUvs &&
      !readVector(file, dag.uvs, usize(header.poolVertexCount) * 2)) {
    log = "truncated (uvs)";
    std::fclose(file);
    return false;
  }
  if (!readVector(file, dag.indices, header.poolIndexCount)) {
    log = "truncated (indices)";
    std::fclose(file);
    return false;
  }
  std::fclose(file);

  // Rehydrate engine structures.
  dag.groups.resize(groups.size());
  for (u32 g = 0; g < groups.size(); ++g) {
    DagGroup& dst = dag.groups[g];
    dst.lodLevel = groups[g].lodLevel;
    dst.maxError = groups[g].maxError;
    dst.simplifyError = groups[g].simplifyError;
    dst.boundsSphere = glm::vec4(groups[g].bounds[0], groups[g].bounds[1],
                                 groups[g].bounds[2], groups[g].bounds[3]);
    dst.clusters.assign(
        memberIndices.begin() + groups[g].memberOffset,
        memberIndices.begin() + groups[g].memberOffset + groups[g].memberCount);
    dst.parents.assign(
        parentIndices.begin() + groups[g].parentOffset,
        parentIndices.begin() + groups[g].parentOffset + groups[g].parentCount);
  }
  dag.clusters.resize(clusters.size());
  for (u32 c = 0; c < clusters.size(); ++c) {
    DagCluster& dst = dag.clusters[c];
    const ClusterDisk& src = clusters[c];
    dst.boundsSphere =
        glm::vec4(src.bounds[0], src.bounds[1], src.bounds[2], src.bounds[3]);
    dst.error = src.error;
    dst.parentError = src.parentError;
    dst.vertexOffset = src.vertexOffset;
    dst.vertexCount = src.vertexCount;
    dst.indexOffset = src.indexOffset;
    dst.indexCount = src.indexCount;
    dst.group = src.group;
    dst.children.assign(
        childIndices.begin() + src.childOffset,
        childIndices.begin() + src.childOffset + src.childCount);
    dst.isRoot = src.isRoot != 0;
    dst.refinedGroup = src.refinedGroup == 0xFFFFFFFFu
                           ? -1
                           : static_cast<i32>(src.refinedGroup);
  }
  dag.sourceVertexCount = header.sourceVertexCount;
  dag.sourceTriangleCount = header.sourceTriangleCount;
  dag.hierarchyRoot = header.hierarchyRoot;

  return verifyDagInvariants(dag, log);
}

// Shared v2/v3 table rehydration (everything except the pools). The
// file must be positioned right after the header.
bool rehydrateTables(std::FILE* file, u32 groupCount, u32 clusterCount,
                     u32 childEdgeCount, u32 rootCount, u32 hierarchyCount,
                     ClusterDag& dag, std::string& log) {
  std::vector<GroupDisk> groups;
  std::vector<ClusterDisk> clusters;
  std::vector<u32> childIndices, memberIndices, parentIndices;
  if (!readVector(file, groups, groupCount) ||
      !readVector(file, clusters, clusterCount) ||
      !readVector(file, childIndices, childEdgeCount)) {
    log = "truncated (groups/clusters/children)";
    return false;
  }
  usize memberTotal = 0, parentTotal = 0;
  for (const auto& g : groups) {
    memberTotal += g.memberCount;
    parentTotal += g.parentCount;
  }
  if (!readVector(file, memberIndices, memberTotal) ||
      !readVector(file, parentIndices, parentTotal) ||
      !readVector(file, dag.hierarchyNodes, hierarchyCount) ||
      !readVector(file, dag.roots, rootCount)) {
    log = "truncated (members/parents/hierarchy/roots)";
    return false;
  }
  dag.groups.resize(groups.size());
  for (u32 g = 0; g < groups.size(); ++g) {
    DagGroup& dst = dag.groups[g];
    dst.lodLevel = groups[g].lodLevel;
    dst.maxError = groups[g].maxError;
    dst.simplifyError = groups[g].simplifyError;
    dst.boundsSphere = glm::vec4(groups[g].bounds[0], groups[g].bounds[1],
                                 groups[g].bounds[2], groups[g].bounds[3]);
    dst.clusters.assign(
        memberIndices.begin() + groups[g].memberOffset,
        memberIndices.begin() + groups[g].memberOffset + groups[g].memberCount);
    dst.parents.assign(
        parentIndices.begin() + groups[g].parentOffset,
        parentIndices.begin() + groups[g].parentOffset + groups[g].parentCount);
  }
  dag.clusters.resize(clusters.size());
  for (u32 c = 0; c < clusters.size(); ++c) {
    DagCluster& dst = dag.clusters[c];
    const ClusterDisk& src = clusters[c];
    dst.boundsSphere =
        glm::vec4(src.bounds[0], src.bounds[1], src.bounds[2], src.bounds[3]);
    dst.error = src.error;
    dst.parentError = src.parentError;
    dst.vertexOffset = src.vertexOffset;
    dst.vertexCount = src.vertexCount;
    dst.indexOffset = src.indexOffset;
    dst.indexCount = src.indexCount;
    dst.group = src.group;
    dst.children.assign(
        childIndices.begin() + src.childOffset,
        childIndices.begin() + src.childOffset + src.childCount);
    dst.isRoot = src.isRoot != 0;
    dst.refinedGroup = src.refinedGroup == 0xFFFFFFFFu
                           ? -1
                           : static_cast<i32>(src.refinedGroup);
  }
  return true;
}

bool readKGeoPagedTables(const std::string& path, const FileHeaderV3& h3,
                         u32 materialCount, ClusterDag& dag, std::string& log) {
  std::FILE* file = nullptr;
  if ((file = kenshin::util::openFile(path.c_str(), "rb")) == nullptr) {
    log = "cannot open for read: " + path;
    return false;
  }
  dag = {};
  // v7's 17th header word sits between the fixed header and the
  // sections; skip past it (the caller already consumed its value).
  const u32 headerWords = 16 + (h3.version >= kVersion7 ? 1 : 0);
  if (std::fseek(file, long(headerWords) * 4, SEEK_SET) != 0 ||
      !rehydrateTables(file, h3.groupCount, h3.clusterCount, h3.childEdgeCount,
                       h3.rootCount, h3.hierarchyNodeCount, dag, log) ||
      !readVector(file, dag.clusterPage, h3.clusterCount)) {
    std::fclose(file);
    return false;
  }
  if ((h3.version == kVersion5 || h3.version == kVersion6 ||
       isV7Plus(h3.version)) &&
      !readVector(file, dag.clusterMetaIndex, h3.clusterCount)) {
    log = "truncated (clusterMetaIndex)";
    std::fclose(file);
    return false;
  }
  const u32 metaWordsPerPage = (h3.version == kVersion5 ||
                                h3.version == kVersion6 || isV7Plus(h3.version))
                                   ? 12u
                                   : 6u;
  std::vector<u32> metaFlat;
  if (!readVector(file, metaFlat, usize(h3.pageCount) * metaWordsPerPage)) {
    log = "truncated (page metadata)";
    std::fclose(file);
    return false;
  }
  if ((h3.version == kVersion4 || h3.version == kVersion5 ||
       h3.version == kVersion6 || isV7Plus(h3.version)) &&
      !readVector(file, dag.clusterMaxEdge, h3.clusterCount)) {
    log = "truncated (maxEdge table)";
    std::fclose(file);
    return false;
  }
  if (isV7Plus(h3.version)) {
    // The material section: string blob + per-material path offsets +
    // 8-f32 records. texPath carries the disk reference; the texture
    // refs stay kNoTexture until the app resolves them (TextureManager).
    u32 stringBytes = 0;
    std::string strings;
    std::vector<u32> texOff;
    std::vector<f32> records;
    // Bounds: material ids ride the meta widths word's 8 high bits, so
    // more than 256 materials could never be addressed; stringBytes is
    // bounded by the file's own remaining size (a corrupt word must
    // not turn into a multi-GB allocation - reject loudly instead).
    const long long curPos = util::tell64(file);
    util::seek64(file, 0, SEEK_END);
    const long long endPos = util::tell64(file);
    util::seek64(file, curPos, SEEK_SET);
    if (std::fread(&stringBytes, sizeof(stringBytes), 1, file) != 1 ||
        materialCount == 0 || materialCount > 256 ||
        endPos - curPos - 4 < static_cast<long long>(stringBytes)) {
      log = "truncated (material section)";
      std::fclose(file);
      return false;
    }
    strings.resize(stringBytes);
    if (stringBytes > 0 &&
        std::fread(strings.data(), 1, stringBytes, file) != stringBytes) {
      log = "truncated (material strings)";
      std::fclose(file);
      return false;
    }
    // File order (v10): texOff[] then mrOff[] then records; v11 adds
    // nrmOff[] after mrOff[].
    std::vector<u32> mrOff, nrmOff;
    if (!readVector(file, texOff, materialCount) ||
        (isV10Plus(h3.version) && !readVector(file, mrOff, materialCount)) ||
        (hasNrmTable(h3.version) && !readVector(file, nrmOff, materialCount)) ||
        !readVector(
            file, records,
            usize(materialCount) * (h3.version == kVersion13 ? 12u : 8u))) {
      log = "truncated (material table)";
      std::fclose(file);
      return false;
    }
    dag.materials.resize(materialCount);
    for (u32 m = 0; m < materialCount; ++m) {
      const usize recF = h3.version == kVersion13 ? 12 : 8;
      const f32* r = records.data() + usize(m) * recF;
      auto& mat = dag.materials[m];
      if (recF == 12) {
        for (int c = 0; c < 3; ++c)
          mat.emissive[c] =
              (r[8 + c] >= 0.0f && r[8 + c] < 1e4f) ? r[8 + c] : 0.0f;
        mat.occlusion = (r[11] >= 0.0f && r[11] <= 1.0f) ? r[11] : 0.0f;
      }
      mat.kd[0] = r[0];
      mat.kd[1] = r[1];
      mat.kd[2] = r[2];
      mat.roughness = r[4];
      mat.metallic = r[5];
      if (hasNrmTable(h3.version) && r[7] >= 0.0f && nrmOff[m] != 0xFFFFFFFFu &&
          nrmOff[m] < strings.size())
        mat.nrmPath = std::string(strings.data() + nrmOff[m]);
      if (isV10Plus(h3.version) && r[6] >= 0.0f && mrOff[m] != 0xFFFFFFFFu &&
          mrOff[m] < strings.size()) {
        mat.mrPath = std::string(strings.data() + mrOff[m]);
      }
      if (r[3] >= 0.0f && texOff[m] != 0xFFFFFFFFu &&
          texOff[m] < strings.size()) {
        mat.texPath = std::string(strings.data() + texOff[m]);
      }
    }
    dag.hasMaterialTable = true;
  }
  // M9.2 v9: the imposter atlas sits between the material section and
  // the blobs (u32 word count, then depth[] and tri[]). v10 carries an
  // explicit presence flag (v10 does not imply the imposter feature).
  bool wantImposter = h3.version == kVersion9;
  if (isV10Plus(h3.version)) {
    u32 imposterFlag = 0;
    if (std::fread(&imposterFlag, sizeof(imposterFlag), 1, file) != 1) {
      log = "truncated (imposter flag)";
      std::fclose(file);
      return false;
    }
    wantImposter = imposterFlag != 0;
  }
  if (wantImposter) {
    u32 atlasWords = 0;
    if (std::fread(&atlasWords, sizeof(atlasWords), 1, file) != 1 ||
        atlasWords == 0xFFFFFFFFu || atlasWords == 0 ||
        std::fread(&dag.imposterCluster, sizeof(u32), 1, file) != 1 ||
        !readVector(file, dag.imposterDepth, atlasWords) ||
        !readVector(file, dag.imposterTri, atlasWords)) {
      log = "truncated (imposter atlas)";
      std::fclose(file);
      return false;
    }
  }
  dag.pageMeta.resize(h3.pageCount);
  dag.pageMetaV5.clear();
  if (h3.version == kVersion5 || h3.version == kVersion6 ||
      isV7Plus(h3.version)) {
    dag.pageMetaV5.resize(h3.pageCount);
    dag.pagesQuantized = true;
    // M7.2: v6+ pages carry the PACKED index bit stream - the runtime
    // paged gate requires this (v5 pages' idx is u8-flat).
    dag.pageIdxPacked = h3.version >= kVersion6;
    for (u32 p = 0; p < h3.pageCount; ++p) {
      const u32* m = metaFlat.data() + usize(p) * 12;
      dag.pageMetaV5[p] = ClusterDag::DagPageMetaV5{
          m[0], m[1],  m[2],
          m[3], m[4],  m[5],
          m[6], m[7],  *reinterpret_cast<const f32*>(&m[8]),
          m[9], m[10], m[11]};
    }
  }
  for (u32 p = 0; p < h3.pageCount; ++p) {
    if (h3.version == kVersion5 || h3.version == kVersion6 ||
        isV7Plus(h3.version)) {
      // v5+: the old 6-field view keeps only the CONSERVATION fields
      // (vertex/index totals; the base fields' semantics changed to u32
      // units and live in pageMetaV5).
      const u32* m = metaFlat.data() + usize(p) * 12;
      dag.pageMeta[p] = ClusterDag::DagPageMeta{0u, 0u, 0u, 0u, m[5], m[6]};
    } else {
      const u32* m = metaFlat.data() + usize(p) * 6;
      dag.pageMeta[p] =
          ClusterDag::DagPageMeta{m[0], m[1], m[2], m[3], m[4], m[5]};
    }
  }
  dag.sourceVertexCount = h3.sourceVertexCount;
  dag.sourceTriangleCount = h3.sourceTriangleCount;
  dag.hierarchyRoot = h3.hierarchyRoot;
  dag.paged = true;
  dag.pageCount = h3.pageCount;
  dag.pageSlotSize = h3.pageSlotSize;
  dag.pagesHaveNormals = h3.hasNormals != 0;
  dag.pagesHaveUvs = h3.hasUvs != 0;

  // Paged verification (the pool checks of verifyDagInvariants do not
  // apply - pools are empty by design): tables + page conservation.
  for (u32 c = 0; c < dag.clusters.size(); ++c) {
    const DagCluster& cl = dag.clusters[c];
    if (dag.clusterPage[c] >= dag.pageCount) {
      log = "paged: cluster " + std::to_string(c) + " page out of range";
      std::fclose(file);
      return false;
    }
    const auto& m = dag.pageMeta[dag.clusterPage[c]];
    // v5: indexOffset/indexCount are BYTE counts against the page's
    // idx-section byte total (raw - the 4B section padding is extra).
    // v6: indexOffset is the cluster's BIT base into the packed idx
    // bit stream and m.indexCount the section's padded byte size -
    // bound the bit base by the section's bits (exact per-cluster idx
    // conservation lives in verifyPacking + the shader pool clamps).
    const bool idxOk =
        dag.pageIdxPacked
            ? usize(cl.indexOffset) <= usize(m.indexCount) * 8u
            : usize(cl.indexOffset) + cl.indexCount <= m.indexCount;
    if (usize(cl.vertexOffset) + cl.vertexCount > m.vertexCount || !idxOk) {
      log = "paged: cluster " + std::to_string(c) + " slice exceeds page";
      std::fclose(file);
      return false;
    }
  }
  std::fclose(file);
  return true;
}

bool KGeoPageReader::open(const std::string& path, std::string& log) {
  close();
  std::FILE* file = nullptr;
  if ((file = kenshin::util::openFile(path.c_str(), "rb")) == nullptr) {
    log = "cannot open for read: " + path;
    return false;
  }
  FileHeaderV3 h3{};
  if (std::fread(&h3, sizeof(h3), 1, file) != 1 || h3.magic != kMagicV3 ||
      (h3.version != kVersion3 && h3.version != kVersion4 &&
       h3.version != kVersion5 && h3.version != kVersion6 &&
       !isV7Plus(h3.version))) {
    log = "not a v3 paged file: " + path;
    std::fclose(file);
    return false;
  }
  // The page directory is the file's tail: pageCount * 8 (v7-) or 12
  // (v8+) bytes - the entry width rides the version word.
  const bool v8 = h3.version >= kVersion8;
  const usize entryBytes = v8 ? sizeof(PageDirEntryV8) : sizeof(PageDirEntry);
  if (!util::seek64(file, -static_cast<i64>(usize(h3.pageCount) * entryBytes),
                    SEEK_END)) {
    log = "cannot seek to the page directory";
    std::fclose(file);
    return false;
  }
  std::vector<u8> dirBytes(usize(h3.pageCount) * entryBytes);
  if (std::fread(dirBytes.data(), 1, dirBytes.size(), file) !=
      dirBytes.size()) {
    log = "truncated page directory";
    std::fclose(file);
    return false;
  }
  std::fclose(file);

  m_pageCount = h3.pageCount;
  m_slotSize = h3.pageSlotSize;
  m_hasNormals = h3.hasNormals != 0;
  m_hasUvs = h3.hasUvs != 0;
  m_compressed = v8;
  m_offsets.resize(usize(h3.pageCount) * 2);
  m_rawLen.resize(h3.pageCount);
  for (u32 p = 0; p < h3.pageCount; ++p) {
    if (v8) {
      const PageDirEntryV8& e = *reinterpret_cast<const PageDirEntryV8*>(
          dirBytes.data() + usize(p) * entryBytes);
      m_offsets[usize(p) * 2] = e.fileOffset;
      m_offsets[usize(p) * 2 + 1] = e.storedLen;
      m_rawLen[p] = e.rawLen;
    } else {
      const PageDirEntry& e = *reinterpret_cast<const PageDirEntry*>(
          dirBytes.data() + usize(p) * entryBytes);
      m_offsets[usize(p) * 2] = e.fileOffset;
      m_offsets[usize(p) * 2 + 1] = e.byteLen;
      m_rawLen[p] = e.byteLen;
    }
  }
  m_path = path;
  return true;
}

bool KGeoPageReader::readPage(u32 page, std::vector<u8>& out,
                              std::string& log) {
  if (page >= m_pageCount) {
    log = "page out of range";
    return false;
  }
  std::FILE* file = nullptr;
  if ((file = kenshin::util::openFile(m_path.c_str(), "rb")) == nullptr) {
    log = "cannot open for read: " + m_path;
    return false;
  }
  const u64 off = m_offsets[usize(page) * 2];
  const u64 len = m_offsets[usize(page) * 2 + 1];
  const u64 rawLen = m_rawLen[page];
  bool ok = util::seek64(file, static_cast<i64>(off), SEEK_SET);
  std::vector<u8> stored;
  stored.resize(len);
  ok = ok && (len == 0 || std::fread(stored.data(), 1, len, file) == len);
  std::fclose(file);
  if (!ok) {
    log = "page read failed";
    return false;
  }
  if (!m_compressed || len == rawLen) {
    // v7, or a v8 page the codec could not shrink (stored raw).
    out = std::move(stored);
    return true;
  }
  // M7.2b: inflate to the exact v7 byte layout (the codec saw a
  // 4-byte-padded word stream; truncate back to rawLen).
  const usize words = usize((rawLen + 3) / 4);
  out.resize(words * 4);  // the codec writes whole words; truncate after
  const int rc = meshopt_decodeVertexBuffer(out.data(), words, 4, stored.data(),
                                            stored.size());
  if (rc != 0) {
    log = "page decode failed (corrupt blob?)";
    return false;
  }
  out.resize(rawLen);
  return true;
}

void KGeoPageReader::close() {
  m_pageCount = 0;
  m_offsets.clear();
  m_path.clear();
}

}  // namespace geometry

KENSHIN_NS_END
