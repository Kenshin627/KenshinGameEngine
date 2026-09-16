#include "kenshin/geometry/KGeoFile.h"

#include <cstdio>
#include <cstring>
#include <vector>

#include <glm/glm.hpp>

#include "kenshin/core/Log.h"

KENSHIN_NS_BEGIN

namespace geometry {

namespace {
constexpr u32 kMagic = 0x324F474B;  // "KGO2"
constexpr u32 kVersion = 2;
constexpr u32 kMagicV3 = 0x334F474B;  // "KGO3" (M6.1 paged)
constexpr u32 kVersion3 = 3;   // paged, no maxEdge table
constexpr u32 kVersion4 = 4;   // paged + per-cluster maxEdge table

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
                         ClusterDag& dag, std::string& log);
bool rehydrateTables(std::FILE* file, u32 groupCount, u32 clusterCount,
                    u32 childEdgeCount, u32 rootCount, u32 hierarchyCount,
                    ClusterDag& dag, std::string& log);

bool writeKGeo(const std::string& path, const ClusterDag& dag,
               std::string& log) {
  std::FILE* file = nullptr;
  if (fopen_s(&file, path.c_str(), "wb") != 0 || !file) {
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
                     const PackedGeometry& pg, std::string& log) {
  if (pg.pageCount != pg.pageMeta.size() ||
      pg.pageCount != pg.pages.size() ||
      pg.clusterPage.size() != dag.clusters.size()) {
    log = "paged write: packing table size mismatch";
    return false;
  }
  std::FILE* file = nullptr;
  if (fopen_s(&file, path.c_str(), "wb") != 0 || !file) {
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
  pageMetaFlat.reserve(usize(pg.pageCount) * 6);
  for (const auto& m : pg.pageMeta) {
    pageVerts += m.vertexCount;
    pageIdx += m.indexCount;
    pageMetaFlat.insert(pageMetaFlat.end(),
                        {m.posBase, m.nrmBase, m.uvBase, m.idxBase,
                         m.vertexCount, m.indexCount});
  }

  const FileHeaderV3 header{
      .magic = kMagicV3,
      .version = kVersion3,
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

  // v4: per-cluster max triangle edge (the routing's EdgeLength; the
  // same formula as renderer::computeClusterAabbs - v2/v3 routing
  // parity), written between the tables and the page blobs.
  std::vector<f32> maxEdge(dag.clusters.size(), 0.0f);
  for (u32 c = 0; c < dag.clusters.size(); ++c) {
    const DagCluster& cl = dag.clusters[c];
    f32 e = 0.0f;
    for (u32 t = 0; t + 2 < cl.indexCount; t += 3) {
      const glm::vec3& a = *reinterpret_cast<const glm::vec3*>(
          dag.positions.data() + usize(cl.vertexOffset + dag.indices[usize(cl.indexOffset + t)]) * 3);
      const glm::vec3& b = *reinterpret_cast<const glm::vec3*>(
          dag.positions.data() + usize(cl.vertexOffset + dag.indices[usize(cl.indexOffset + t + 1)]) * 3);
      const glm::vec3& c3 = *reinterpret_cast<const glm::vec3*>(
          dag.positions.data() + usize(cl.vertexOffset + dag.indices[usize(cl.indexOffset + t + 2)]) * 3);
      e = std::max({e, glm::length(b - a), glm::length(c3 - b),
                    glm::length(a - c3)});
    }
    maxEdge[c] = e;
  }

  bool ok = std::fwrite(&header, sizeof(header), 1, file) == 1 &&
            writeVector(file, groups) && writeVector(file, clusters) &&
            writeVector(file, childIndices) &&
            writeVector(file, memberIndices) && writeVector(file, parentIndices) &&
            writeVector(file, dag.hierarchyNodes) &&
            writeVector(file, dag.roots) && writeVector(file, pg.clusterPage) &&
            writeVector(file, pageMetaFlat) && writeVector(file, maxEdge);

  // Blobs (16 B aligned, offsets recorded); the DIRECTORY goes to the
  // file tail so the writer stays single-pass - the reader lands on it
  // via fileSize - pageCount * sizeof(PageDirEntry).
  std::vector<PageDirEntry> dir(pg.pageCount);
  for (u32 p = 0; p < pg.pageCount && ok; ++p) {
    // _ftelli64: .kgeo assets cross the 2GB long boundary at scale.
    const long long rawPos = _ftelli64(file);
    ok = rawPos >= 0;
    if (!ok) break;
    const usize pos = static_cast<usize>(rawPos);
    const usize pad = (16 - (pos % 16)) % 16;
    for (usize b = 0; b < pad && ok; ++b)
      ok = std::fputc(0, file) != EOF;
    const long long dataPos = ok ? _ftelli64(file) : -1;
    ok = ok && dataPos >= 0;
    if (!ok) break;
    dir[p] = PageDirEntry{static_cast<u32>(dataPos),
                          static_cast<u32>(pg.pages[p].size())};
    ok = pg.pages[p].empty() ||
         std::fwrite(pg.pages[p].data(), 1, pg.pages[p].size(), file) ==
             pg.pages[p].size();
  }
  if (ok) ok = writeVector(file, dir);
  std::fclose(file);
  if (!ok) log = "paged write failed (disk full?)";
  return ok;
}

bool readKGeo(const std::string& path, ClusterDag& dag, std::string& log) {
  std::FILE* file = nullptr;
  if (fopen_s(&file, path.c_str(), "rb") != 0 || !file) {
    log = "cannot open for read: " + path;
    return false;
  }

  FileHeader header{};
  if (std::fread(&header, sizeof(header), 1, file) != 1 ||
      (header.magic != kMagic || header.version != kVersion)) {
    // v3? Read the v3 header over the same leading 14 u32 + 2 more.
    FileHeaderV3 h3{};
    std::rewind(file);
    if (std::fread(&h3, sizeof(h3), 1, file) != 1 ||
        h3.magic != kMagicV3 ||
        (h3.version != kVersion3 && h3.version != kVersion4)) {
      log = "bad magic/version";
      std::fclose(file);
      return false;
    }
    std::fclose(file);
    return readKGeoPagedTables(path, h3, dag, log);
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
    dst.refinedGroup =
        src.refinedGroup == 0xFFFFFFFFu ? -1 : static_cast<i32>(src.refinedGroup);
  }
  return true;
}

bool readKGeoPagedTables(const std::string& path, const FileHeaderV3& h3,
                          ClusterDag& dag, std::string& log) {
  std::FILE* file = nullptr;
  if (fopen_s(&file, path.c_str(), "rb") != 0 || !file) {
    log = "cannot open for read: " + path;
    return false;
  }
  dag = {};
  if (std::fseek(file, long(sizeof(FileHeaderV3)), SEEK_SET) != 0 ||
      !rehydrateTables(file, h3.groupCount, h3.clusterCount, h3.childEdgeCount,
                       h3.rootCount, h3.hierarchyNodeCount, dag, log) ||
      !readVector(file, dag.clusterPage, h3.clusterCount)) {
    std::fclose(file);
    return false;
  }
  std::vector<u32> metaFlat;
  if (!readVector(file, metaFlat, usize(h3.pageCount) * 6)) {
    log = "truncated (page metadata)";
    std::fclose(file);
    return false;
  }
  if (h3.version == kVersion4 &&
      !readVector(file, dag.clusterMaxEdge, h3.clusterCount)) {
    log = "truncated (maxEdge table)";
    std::fclose(file);
    return false;
  }
  dag.pageMeta.resize(h3.pageCount);
  for (u32 p = 0; p < h3.pageCount; ++p) {
    const u32* m = metaFlat.data() + usize(p) * 6;
    dag.pageMeta[p] = ClusterDag::DagPageMeta{m[0], m[1], m[2], m[3], m[4], m[5]};
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
    if (usize(cl.vertexOffset) + cl.vertexCount > m.vertexCount ||
        usize(cl.indexOffset) + cl.indexCount > m.indexCount) {
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
  if (fopen_s(&file, path.c_str(), "rb") != 0 || !file) {
    log = "cannot open for read: " + path;
    return false;
  }
  FileHeaderV3 h3{};
  if (std::fread(&h3, sizeof(h3), 1, file) != 1 || h3.magic != kMagicV3 ||
      h3.version != kVersion3 && h3.version != kVersion4) {
    log = "not a v3 paged file: " + path;
    std::fclose(file);
    return false;
  }
  // The page directory is the file's tail: pageCount * 8 bytes.
  if (_fseeki64(file, -static_cast<long long>(usize(h3.pageCount) *
                                                sizeof(PageDirEntry)),
                SEEK_END) != 0) {
    log = "cannot seek to the page directory";
    std::fclose(file);
    return false;
  }
  std::vector<PageDirEntry> dir(h3.pageCount);
  if (std::fread(dir.data(), sizeof(PageDirEntry), h3.pageCount, file) !=
      h3.pageCount) {
    log = "truncated page directory";
    std::fclose(file);
    return false;
  }
  std::fclose(file);

  m_pageCount = h3.pageCount;
  m_slotSize = h3.pageSlotSize;
  m_hasNormals = h3.hasNormals != 0;
  m_hasUvs = h3.hasUvs != 0;
  m_offsets.resize(usize(h3.pageCount) * 2);
  for (u32 p = 0; p < h3.pageCount; ++p) {
    m_offsets[usize(p) * 2] = dir[p].fileOffset;
    m_offsets[usize(p) * 2 + 1] = dir[p].byteLen;
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
  if (fopen_s(&file, m_path.c_str(), "rb") != 0 || !file) {
    log = "cannot open for read: " + m_path;
    return false;
  }
  const u64 off = m_offsets[usize(page) * 2];
  const u64 len = m_offsets[usize(page) * 2 + 1];
  bool ok = _fseeki64(file, static_cast<long long>(off), SEEK_SET) == 0;
  out.resize(len);
  ok = ok && (len == 0 || std::fread(out.data(), 1, len, file) == len);
  std::fclose(file);
  if (!ok) log = "page read failed";
  return ok;
}

void KGeoPageReader::close() {
  m_pageCount = 0;
  m_offsets.clear();
  m_path.clear();
}

}  // namespace geometry

KENSHIN_NS_END
