#include "kenshin/renderer/VirtualGeometryScene.h"

#include <algorithm>
#include <chrono>
#include <cstring>

#include "kenshin/Engine.h"
#include "kenshin/core/Log.h"
#include "kenshin/geometry/ClusterDag.h"
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
  const usize base = out.size();  // resize FIRST: data() may reallocate
  out.resize(base + 48);
  u8* ptr = out.data() + base;
  std::memcpy(ptr + 0, &c.boundsSphere, 16);
  std::memcpy(ptr + 16, &c.error, 4);
  std::memcpy(ptr + 20, &c.parentError, 4);
  std::memcpy(ptr + 24, &c.vertexOffset, 4);
  std::memcpy(ptr + 28, &c.vertexCount, 4);
  std::memcpy(ptr + 32, &c.indexOffset, 4);
  std::memcpy(ptr + 36, &c.indexCount, 4);
  std::memcpy(ptr + 40, &c.group, 4);
  std::memcpy(ptr + 44, &c.refinedGroup, 4);  // i32, -1 = original
}

void appendGpuGroup(std::vector<u8>& out, const geometry::DagGroup& g) {
  const usize base = out.size();
  out.resize(base + 32);
  u8* ptr = out.data() + base;
  std::memcpy(ptr + 0, &g.boundsSphere, 16);
  std::memcpy(ptr + 16, &g.maxError, 4);
  std::memcpy(ptr + 20, &g.simplifyError, 4);
  std::memset(ptr + 24, 0, 8);  // pad to 32
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
  const u32 cpuCrc = util::crc32(
      {static_cast<const u8*>(data), bytes});
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
  for (const auto& group : dag.groups) appendGpuGroup(m_groupData, group);
  m_groupCount += static_cast<u32>(dag.groups.size());
  range.groupEnd = m_groupCount;

  // Clusters: offsets rebased into the SHARED vertex/index pools.
  const u32 vertexBase = static_cast<u32>(m_positions.size() / 3);
  const u32 indexBase = static_cast<u32>(m_indices.size());
  for (const auto& cluster : dag.clusters) {
    geometry::DagCluster rebased = cluster;
    rebased.vertexOffset += vertexBase;
    rebased.indexOffset += indexBase;
    appendGpuCluster(m_clusterData, rebased);
  }
  m_clusterCount += static_cast<u32>(dag.clusters.size());
  range.clusterEnd = m_clusterCount;

  m_positions.insert(m_positions.end(), dag.positions.begin(),
                     dag.positions.end());
  m_normals.insert(m_normals.end(), dag.normals.begin(), dag.normals.end());
  m_uvs.insert(m_uvs.end(), dag.uvs.begin(), dag.uvs.end());
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

void VirtualGeometryScene::setInstances(
    const std::vector<VgInstance>& instances) {
  if (m_backend == nullptr) return;

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
  for (const u32 handle : {m_clusterBuffer, m_groupBuffer, m_positionBuffer,
                           m_normalBuffer, m_uvBuffer, m_indexBuffer,
                           m_instanceBuffer})
    if (handle != backend::IRenderBackend::kInvalidBufferHandle)
      m_backend->destroyBuffer(handle);
  m_backend = nullptr;
}

// ---- pool flush ----------------------------------------------------------
// Called by the app after all addMesh calls (declared out-of-line here to
// keep the header lean): uploads the shared pools once and verifies each
// with a CRC round trip. Registered as a member via a small wrapper.

u32 VirtualGeometryScene::flushPools() {
  if (m_backend == nullptr || !m_poolsDirty) return m_clusterBuffer;

  f64 uploadMs = 0.0, verifyMs = 0.0;
  auto upload = [&](const void* data, usize bytes, const char* name) {
    f64 u = 0.0, v = 0.0;
    const u32 handle =
        uploadVerified(m_backend, data, bytes, name, &u, &v);
    uploadMs += u;
    verifyMs += v;
    return handle;
  };
  m_clusterBuffer = upload(m_clusterData.data(), m_clusterData.size(),
                           "vg-clusters");
  m_groupBuffer = upload(m_groupData.data(), m_groupData.size(), "vg-groups");
  m_positionBuffer = upload(m_positions.data(),
                            m_positions.size() * sizeof(f32), "vg-positions");
  m_normalBuffer = upload(m_normals.data(), m_normals.size() * sizeof(f32),
                          "vg-normals");
  m_uvBuffer =
      upload(m_uvs.data(), m_uvs.size() * sizeof(f32), "vg-uvs");
  m_indexBuffer = upload(m_indices.data(), m_indices.size() * sizeof(u32),
                         "vg-indices");
  KENSHIN_INFO(
      "[vgscene] pools uploaded in {:.0f}ms (verify readback {:.0f}ms): "
      "{} clusters / {} groups / {} verts / {} indices",
      uploadMs, verifyMs, m_clusterCount, m_groupCount,
      m_positions.size() / 3, m_indices.size() / 3);
  m_poolsDirty = false;
  return m_clusterBuffer;
}

}  // namespace renderer

KENSHIN_NS_END
