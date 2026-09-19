#include "kenshin/renderer/VirtualGeometryScene.h"

#include <algorithm>
#include <chrono>
#include <cstring>

#include "kenshin/Engine.h"
#include "kenshin/core/Log.h"
#include "kenshin/geometry/ClusterDag.h"
#include "kenshin/geometry/PagePacker.h"
#include "kenshin/util/Crc32.h"

KENSHIN_NS_BEGIN

namespace renderer {

namespace {

// ---- std430 accumulators ------------------------------------------------
// appendGpuCluster writes 48B: vec4 bounds + error/parentError +
// vertexOffset/vertexCount/indexOffset/indexCount/group. GpuGroup writes
// 32B: vec4 bounds + maxError/simplifyError + 8B zero pad. GpuInstance
// (in setInstances) writes 80B: mat4 + meshId + 12B pad. The S4.2+
// shaders mirror these layouts - single declaration point is the header.

void appendGpuCluster(std::vector<u8>& out, const geometry::DagCluster& c) {
  packGpuCluster(out, c.group, c.boundsSphere.x, c.boundsSphere.y,
                 c.boundsSphere.z, c.boundsSphere.w, c.error, c.parentError,
                 c.vertexOffset, c.vertexCount, c.indexOffset, c.indexCount,
                 c.refinedGroup);
}

void appendGpuGroup(std::vector<u8>& out, const geometry::DagGroup& g,
                    u32& maxLodLevel) {
  maxLodLevel = std::max(maxLodLevel, static_cast<u32>(g.lodLevel));
  packGpuGroup(out, g.boundsSphere.x, g.boundsSphere.y, g.boundsSphere.z,
               g.boundsSphere.w, g.maxError, g.simplifyError,
               static_cast<u32>(g.lodLevel));
}

/// Upload (or replace) one pool buffer; verify with a CRC round trip.
/// Returns the handle, logs crc ok/broken. `name` feeds the log + the
/// debug name.
u32 uploadVerified(backend::IRenderBackend* backend, const void* data,
                   usize bytes, const char* name, f64* uploadMs = nullptr,
                   f64* verifyMs = nullptr) {
  const auto t0 = std::chrono::steady_clock::now();
  backend::IRenderBackend::BufferDesc desc;
  desc.size = bytes;
  // Storage is the S4.2+ currency; TransferSrc is for the CRC readback
  // verification below (copy source).
  desc.usage = backend::IRenderBackend::BufferStorage |
               backend::IRenderBackend::BufferTransferSrc;
  desc.initialData = data;
  desc.debugName = name;
  const u32 handle = backend->createBuffer(desc);
  const auto t1 = std::chrono::steady_clock::now();
  if (uploadMs)
    *uploadMs = std::chrono::duration<f64, std::milli>(t1 - t0).count();
  if (handle == backend::IRenderBackend::kInvalidBufferHandle) return handle;

  std::vector<u8> back;
  // (verification readback below is measured by the caller's verifyMs)
  const auto v0 = std::chrono::steady_clock::now();
  const bool read = backend->readbackBuffer(handle, 0, bytes, back);
  const auto v1 = std::chrono::steady_clock::now();
  if (verifyMs)
    *verifyMs = std::chrono::duration<f64, std::milli>(v1 - v0).count();
  if (!read) {
    KENSHIN_ERROR("[vgscene] {} readback failed ({} bytes)", name, bytes);
    return handle;
  }
  const u32 cpuCrc = util::crc32({static_cast<const u8*>(data), bytes});
  const u32 gpuCrc = util::crc32(back);
  if (cpuCrc == gpuCrc)
    KENSHIN_INFO("[vgscene] {} crc ok {:08x} ({} bytes)", name, cpuCrc, bytes);
  else
    KENSHIN_ERROR("[vgscene] {} CRC MISMATCH cpu={:08x} gpu={:08x}", name,
                  cpuCrc, gpuCrc);
  return handle;
}

}  // namespace

bool VirtualGeometryScene::onInit(Engine& engine) {
  m_backend = &engine.get<backend::IRenderBackend>();
  return true;
}

u32 VirtualGeometryScene::addMesh(const geometry::ClusterDag& dag) {
  VgMeshRange range;
  range.clusterBegin = m_clusterCount;
  range.groupBegin = m_groupCount;

  // Groups referenced by this mesh's clusters span [0, groups) of the
  // dag; the dag is self-contained, so the whole group table is copied.
  for (const auto& group : dag.groups)
    appendGpuGroup(m_groupData, group, m_maxLodLevel);
  m_groupCount += static_cast<u32>(dag.groups.size());
  range.groupEnd = m_groupCount;

  // Clusters: offsets rebased into the SHARED vertex/index pools.
  const u32 vertexBase = static_cast<u32>(m_pos4.size());
  const u32 indexBase = static_cast<u32>(m_indices.size());
  for (const auto& cluster : dag.clusters) {
    geometry::DagCluster rebased = cluster;
    rebased.vertexOffset += vertexBase;
    rebased.indexOffset += indexBase;
    appendGpuCluster(m_clusterData, rebased);
  }
  m_clusterCount += static_cast<u32>(dag.clusters.size());
  range.clusterEnd = m_clusterCount;

  // M5.2: the group BVH8 (built in M5.1) - flat 64B nodes appended to
  // the shared hierarchy pool; the mesh's root is rebased by the node
  // base. Group -> cluster spans come from the contiguity invariant
  // (verifyDagInvariants enforces it; the traversal never needs the
  // member indices). Cluster indices rebased into the shared pool.
  if (!dag.hierarchyNodes.empty()) {
    const u32 nodeBase = static_cast<u32>(m_hierarchyData.size() /
                                          sizeof(geometry::DagHierarchyNode));
    m_hierarchyData.insert(
        m_hierarchyData.end(),
        reinterpret_cast<const u8*>(dag.hierarchyNodes.data()),
        reinterpret_cast<const u8*>(dag.hierarchyNodes.data()) +
            dag.hierarchyNodes.size() * sizeof(geometry::DagHierarchyNode));
    range.hierarchyRoot = dag.hierarchyRoot == 0xFFFFFFFFu
                              ? 0xFFFFFFFFu
                              : dag.hierarchyRoot + nodeBase;
  }
  for (const auto& group : dag.groups) {
    const u32 begin = group.clusters.empty()
                          ? range.clusterBegin
                          : group.clusters.front() + range.clusterBegin;
    m_groupRanges.emplace_back(begin, static_cast<u32>(group.clusters.size()));
  }

  // vec4/vec2 pools (std430 vec3 arrays stride 16; shaders read float4).
  for (usize v = 0; v < dag.positions.size() / 3; ++v)
    m_pos4.emplace_back(dag.positions[v * 3], dag.positions[v * 3 + 1],
                        dag.positions[v * 3 + 2], 0.0f);
  for (usize v = 0; v < dag.normals.size() / 3; ++v)
    m_norm4.emplace_back(dag.normals[v * 3], dag.normals[v * 3 + 1],
                         dag.normals[v * 3 + 2], 0.0f);
  for (usize v = 0; v < dag.uvs.size() / 2; ++v)
    m_uv2.emplace_back(dag.uvs[v * 2], dag.uvs[v * 2 + 1]);
  m_indices.insert(m_indices.end(), dag.indices.begin(), dag.indices.end());

  // Mesh cull sphere: union of the level-0 (finest) clusters' bounds -
  // the instance cull input (setInstances transforms it per instance).
  {
    glm::vec4 bounds(0.0f, 0.0f, 0.0f, -1.0f);
    for (u32 c = 0; c < dag.clusters.size(); ++c) {
      const glm::vec4 bs(dag.clusters[c].boundsSphere);
      const glm::vec3 pa(bounds), pb(bs);
      const f32 d = glm::distance(pa, pb);
      if (bounds.w < 0.0f)
        bounds = bs;
      else if (d + bounds.w < bs.w)
        bounds = bs;
      else if (d + bs.w > bounds.w) {
        const glm::vec3 dir = d > 0.0f ? (pb - pa) / d : glm::vec3(0.0f);
        bounds = glm::vec4(pa + dir * (d + bounds.w - bs.w) * 0.5f,
                           (bounds.w + bs.w + d) * 0.5f);
      }
    }
    range.boundsSphere = bounds;
  }

  m_meshRanges.push_back(range);
  m_poolsDirty = true;
  return static_cast<u32>(m_meshRanges.size() - 1);
}

u32 VirtualGeometryScene::addMeshPaged(const geometry::ClusterDag& dag) {
  // v3 (tables-only) dag: pools stay empty, cluster offsets are already
  // PAGE-LOCAL (nothing to rebase - pages are the pool), and the page
  // machinery appends into the shared page tables.
  // v5 gate (M7.1): the raster/decode shaders only speak the quantized
  // v5 page format. A v3/v4 paged asset would flow through with EMPTY
  // pageMetaV5/clusterMetaIndex tables -> 0-byte buffers -> silently
  // dropped dispatches (a black viewport with zero diagnostics).
  if (!dag.pagesQuantized) {
    KENSHIN_ERROR(
        "[vgscene] paged asset is pre-v5 (no quantized pages); rebuild it "
        "with the current kgeo_builder - the shaders only decode v5");
    return 0xFFFFFFFFu;  // the app checks and exits loudly
  }
  m_paged = true;
  VgMeshRange range;
  range.clusterBegin = m_clusterCount;
  range.groupBegin = m_groupCount;
  range.pageBegin = m_pageCount;

  for (const auto& group : dag.groups)
    appendGpuGroup(m_groupData, group, m_maxLodLevel);
  m_groupCount += static_cast<u32>(dag.groups.size());
  range.groupEnd = m_groupCount;

  // Cluster table: offsets stay page-local (the shaders add the page's
  // slot-strided section base). clusterPage rebases by the page base.
  for (const auto& cluster : dag.clusters)
    appendGpuCluster(m_clusterData, cluster);
  m_clusterCount += static_cast<u32>(dag.clusters.size());
  range.clusterEnd = m_clusterCount;
  const u32 pageBase = m_pageCount;
  for (const u32 page : dag.clusterPage)
    m_clusterPage.push_back(page + pageBase);
  for (const auto& m : dag.pageMetaV5) {
    const u32 stepBits = *reinterpret_cast<const u32*>(&m.posStep);
    m_pageMetaFlat.insert(
        m_pageMetaFlat.end(),
        {m.posBase, m.nrmBase, m.uvBase, m.idxBase, m.metaBase, m.vertexCount,
         m.indexCount, m.clusterCount, stepBits, m.flags, m.pad0, m.pad1});
  }
  m_clusterMetaFlat.insert(m_clusterMetaFlat.end(),
                           dag.clusterMetaIndex.begin(),
                           dag.clusterMetaIndex.end());
  m_pageCount += dag.pageCount;
  range.pageEnd = m_pageCount;

  if (!dag.hierarchyNodes.empty()) {
    const u32 nodeBase = static_cast<u32>(m_hierarchyData.size() /
                                          sizeof(geometry::DagHierarchyNode));
    m_hierarchyData.insert(
        m_hierarchyData.end(),
        reinterpret_cast<const u8*>(dag.hierarchyNodes.data()),
        reinterpret_cast<const u8*>(dag.hierarchyNodes.data()) +
            dag.hierarchyNodes.size() * sizeof(geometry::DagHierarchyNode));
    range.hierarchyRoot = dag.hierarchyRoot == 0xFFFFFFFFu
                              ? 0xFFFFFFFFu
                              : dag.hierarchyRoot + nodeBase;
  }
  for (const auto& group : dag.groups) {
    const u32 begin = group.clusters.empty()
                          ? range.clusterBegin
                          : group.clusters.front() + range.clusterBegin;
    m_groupRanges.emplace_back(begin, static_cast<u32>(group.clusters.size()));
  }

  // Conservative per-cluster AABB from the bounds SPHERE (occlusion
  // rects only widen - correctness preserved); maxEdge uses the v4
  // per-cluster table when present (the routing's true EdgeLength,
  // same formula as the v2 path) and falls back to the sphere's 2r
  // upper bound on old v3 assets.
  const bool exactMaxEdge = dag.clusterMaxEdge.size() == dag.clusters.size();
  for (u32 ci = 0; ci < dag.clusters.size(); ++ci) {
    const auto& cluster = dag.clusters[ci];
    const glm::vec3 c(cluster.boundsSphere);
    const f32 r = cluster.boundsSphere.w;
    m_aabbs.emplace_back(c - r, 0.0f);
    m_aabbs.emplace_back(c + r, 0.0f);
    m_aabbs.emplace_back(exactMaxEdge ? dag.clusterMaxEdge[ci] : 2.0f * r, 0.0f,
                         0.0f, 0.0f);
  }

  // Mesh cull sphere: same union-of-bounds walk as addMesh.
  {
    glm::vec4 bounds(0.0f, 0.0f, 0.0f, -1.0f);
    for (u32 c = 0; c < dag.clusters.size(); ++c) {
      const glm::vec4 bs(dag.clusters[c].boundsSphere);
      const glm::vec3 pa(bounds), pb(bs);
      const f32 d = glm::distance(pa, pb);
      if (bounds.w < 0.0f)
        bounds = bs;
      else if (d + bounds.w < bs.w)
        bounds = bs;
      else if (d + bs.w > bounds.w) {
        const glm::vec3 dir = d > 0.0f ? (pb - pa) / d : glm::vec3(0.0f);
        bounds = glm::vec4(pa + dir * (d + bounds.w - bs.w) * 0.5f,
                           (bounds.w + bs.w + d) * 0.5f);
      }
    }
    range.boundsSphere = bounds;
  }

  m_meshRanges.push_back(range);
  m_poolsDirty = true;
  return static_cast<u32>(m_meshRanges.size() - 1);
}

void VirtualGeometryScene::setInstances(
    const std::vector<VgInstance>& instances) {
  if (m_backend == nullptr) return;
  m_instanceCount = static_cast<u32>(instances.size());

  // std430 GpuInstance (96B): mat4 (64B) + meshId (4B) + 12B pad +
  // world bounds vec4 (16B, offset 80). When the caller passes a
  // negative-radius sphere the scene derives it from the mesh bounds
  // and the transform (center transformed, radius scaled by the max
  // column length - conservative for non-uniform scales).
  std::vector<u8> data(instances.size() * 96);
  for (usize i = 0; i < instances.size(); ++i) {
    u8* ptr = data.data() + i * 96;
    std::memcpy(ptr + 0, &instances[i].model, 64);
    std::memcpy(ptr + 64, &instances[i].meshId, 4);
    std::memset(ptr + 68, 0, 12);
    glm::vec4 world = instances[i].boundsSphere;
    if (world.w < 0.0f) {
      const VgMeshRange& mesh = m_meshRanges[instances[i].meshId];
      const glm::mat4& m = instances[i].model;
      const glm::vec3 c = glm::vec3(mesh.boundsSphere);
      const f32 scale =
          std::max({glm::length(m[0]), glm::length(m[1]), glm::length(m[2])});
      world = glm::vec4(glm::vec3(m * glm::vec4(c, 1.0f)),
                        mesh.boundsSphere.w * scale);
    }
    std::memcpy(ptr + 80, &world, 16);
  }

  if (m_instanceBuffer != backend::IRenderBackend::kInvalidBufferHandle)
    m_backend->destroyBuffer(m_instanceBuffer);
  m_instanceBuffer =
      uploadVerified(m_backend, data.data(), data.size(), "vg-instances");
}

void VirtualGeometryScene::onShutdown() {
  if (m_backend == nullptr) return;
  for (const u32 handle :
       {m_clusterBuffer, m_groupBuffer, m_positionBuffer, m_normalBuffer,
        m_uvBuffer, m_indexBuffer, m_instanceBuffer, m_clusterPageBuffer,
        m_clusterMetaBuffer, m_pageHeaderBuffer, m_pageTableBuffer,
        m_pageSlotPool})
    if (handle != backend::IRenderBackend::kInvalidBufferHandle)
      m_backend->destroyBuffer(handle);
  m_pageTable = false;
  m_backend = nullptr;
}

// ---- pool flush ----------------------------------------------------------
// Called by the app after all addMesh calls (declared out-of-line here to
// keep the header lean): uploads the shared pools once and verifies each
// with a CRC round trip. Registered as a member via a small wrapper.

u32 VirtualGeometryScene::flushPools(u32 slotCap) {
  if (m_backend == nullptr || !m_poolsDirty) return m_clusterBuffer;

  f64 uploadMs = 0.0, verifyMs = 0.0;
  auto upload = [&](const void* data, usize bytes, const char* name) {
    f64 u = 0.0, v = 0.0;
    const u32 handle = uploadVerified(m_backend, data, bytes, name, &u, &v);
    uploadMs += u;
    verifyMs += v;
    return handle;
  };
  if (m_paged) {
    // Tables only; then the slot pool + the three page tables. The
    // pageTable starts ALL-ABSENT (the streaming manager fills it);
    // page 0 of each asset gets pinned-resident by the manager (the
    // coarse-level safety net).
    m_clusterBuffer =
        upload(m_clusterData.data(), m_clusterData.size(), "vg-clusters");
    m_groupBuffer = upload(m_groupData.data(), m_groupData.size(), "vg-groups");
    m_clusterPageBuffer = upload(m_clusterPage.data(), m_clusterPage.size() * 4,
                                 "vg-clusterPage");
    m_pageHeaderBuffer = upload(m_pageMetaFlat.data(),
                                m_pageMetaFlat.size() * 4, "vg-pageHeader");
    m_clusterMetaBuffer =
        upload(m_clusterMetaFlat.data(), m_clusterMetaFlat.size() * 4,
               "vg-clusterMeta");
    m_pageSlotCount = std::min(m_pageCount, slotCap);
    m_pageSlotPool =
        m_backend->createPageSlotPool(m_pageSlotCount, geometry::kPageSlotSize);
    backend::IRenderBackend::BufferDesc ptDesc;
    ptDesc.size = usize(m_pageCount) * 4;
    ptDesc.usage = backend::IRenderBackend::BufferStorage |
                   backend::IRenderBackend::BufferHostVisible;
    ptDesc.debugName = "vg-pageTable";
    m_pageTableBuffer = m_backend->createBuffer(ptDesc);
    if (m_pageTableBuffer != backend::IRenderBackend::kInvalidBufferHandle) {
      m_pageTable = true;
      u32* pt = pageTableMapped();
      for (u32 p = 0; p < m_pageCount; ++p) pt[p] = 0xFFFFFFFFu;
    }
    KENSHIN_INFO(
        "[vgscene] paged tables uploaded in {:.0f}ms: {} clusters / {} "
        "groups / {} pages ({} MB slot pool = {}/{} slots)",
        uploadMs + verifyMs, m_clusterCount, m_groupCount, m_pageCount,
        double(usize(m_pageSlotCount) * geometry::kPageSlotSize) /
            (1024.0 * 1024.0),
        m_pageSlotCount, m_pageCount);
    m_poolsDirty = false;
    return m_clusterBuffer;
  }

  m_clusterBuffer =
      upload(m_clusterData.data(), m_clusterData.size(), "vg-clusters");
  m_groupBuffer = upload(m_groupData.data(), m_groupData.size(), "vg-groups");
  m_positionBuffer =
      upload(m_pos4.data(), m_pos4.size() * sizeof(glm::vec4), "vg-positions");
  m_normalBuffer =
      upload(m_norm4.data(), m_norm4.size() * sizeof(glm::vec4), "vg-normals");
  m_uvBuffer = upload(m_uv2.data(), m_uv2.size() * sizeof(glm::vec2), "vg-uvs");
  m_indexBuffer =
      upload(m_indices.data(), m_indices.size() * sizeof(u32), "vg-indices");
  KENSHIN_INFO(
      "[vgscene] pools uploaded in {:.0f}ms (verify readback {:.0f}ms): "
      "{} clusters / {} groups / {} verts / {} indices",
      uploadMs, verifyMs, m_clusterCount, m_groupCount, m_pos4.size(),
      m_indices.size() / 3);
  m_poolsDirty = false;
  return m_clusterBuffer;
}

}  // namespace renderer

KENSHIN_NS_END
