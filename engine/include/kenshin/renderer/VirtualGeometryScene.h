#pragma once

// ── VirtualGeometryScene — GPU-resident scene for the virtual path ──
// S4.1: .kgeo ClusterDags are appended into SHARED device pools
// (cluster table / group table / vertex pools / index pool), a mesh
// registry remembers each mesh's cluster range, and an instance table
// carries per-draw transforms. The culling/rasterizing passes of S4.2+
// consume these buffers through the handles this scene exposes.
//
// GPU struct layouts are std430: GpuCluster = 48B exact (vec4 + 8
// scalars, struct stride 16-aligned), GpuGroup = 32B (vec4 + 2 floats +
// 8B pad), GpuInstance = 96B (mat4 + meshId + 12B pad + world bounds
// vec4). The S4.x shaders must mirror these layouts exactly - they are
// declared once here on purpose.

#include <cstring>
#include <glm/glm.hpp>
#include <vector>

#include "kenshin/Common.h"
#include "kenshin/backend/RenderBackend.h"

KENSHIN_NS_BEGIN

class Engine;

namespace geometry {
struct ClusterDag;
}

namespace renderer {

/// ---- GPU packing helpers (single declaration point) -------------------
/// The std430 layouts the S4.x shaders mirror. packGpuCluster writes 48B,
/// packGpuGroup 32B; the instance 96B packing lives in the .cpp (used by
/// setInstances; the cull test reuses GpuInstance structs directly).
inline void packGpuCluster(std::vector<u8>& out, u32 group, f32 boundsX,
                           f32 boundsY, f32 boundsZ, f32 boundsW, f32 error,
                           f32 parentError, u32 vertexOffset, u32 vertexCount,
                           u32 indexOffset, u32 indexCount, i32 refinedGroup) {
  const usize base = out.size();  // resize FIRST: data() may reallocate
  out.resize(base + 48);
  u8* p = out.data() + base;
  const f32 bounds[4] = {boundsX, boundsY, boundsZ, boundsW};
  std::memcpy(p + 0, bounds, 16);
  std::memcpy(p + 16, &error, 4);
  std::memcpy(p + 20, &parentError, 4);
  std::memcpy(p + 24, &vertexOffset, 4);
  std::memcpy(p + 28, &vertexCount, 4);
  std::memcpy(p + 32, &indexOffset, 4);
  std::memcpy(p + 36, &indexCount, 4);
  std::memcpy(p + 40, &group, 4);
  std::memcpy(p + 44, &refinedGroup, 4);
}

inline void packGpuGroup(std::vector<u8>& out, f32 boundsX, f32 boundsY,
                         f32 boundsZ, f32 boundsW, f32 maxError,
                         f32 simplifyError, u32 lodLevel) {
  const usize base = out.size();
  out.resize(base + 32);
  u8* p = out.data() + base;
  const f32 bounds[4] = {boundsX, boundsY, boundsZ, boundsW};
  std::memcpy(p + 0, bounds, 16);
  std::memcpy(p + 16, &maxError, 4);
  std::memcpy(p + 20, &simplifyError, 4);
  std::memcpy(p + 24, &lodLevel, 4);
  std::memset(p + 28, 0, 4);
}

/// Mesh registry entry: where this mesh's clusters live in the shared
/// pools (clusters [clusterBegin, clusterEnd) of the cluster table; the
/// group range covers every group its clusters belong to).
struct VgMeshRange {
  u32 clusterBegin = 0;
  u32 clusterEnd = 0;
  u32 groupBegin = 0;
  u32 groupEnd = 0;
  glm::vec4 boundsSphere{0.0f, 0.0f, 0.0f, -1.0f};  // union of cluster
                                                    // bounds (instance
                                                    // cull input)
};

/// One drawn instance: which mesh + its world transform. `boundsSphere`
/// is the WORLD-space cull sphere (scene computes it from the mesh
/// bounds and the transform at setInstances time).
struct VgInstance {
  glm::mat4 model{1.0f};
  u32 meshId = 0;
  glm::vec4 boundsSphere{0.0f, 0.0f, 0.0f, -1.0f};
};

class VirtualGeometryScene {
 public:
  /// Subsystem hook: resolve the render backend from the registry.
  bool onInit(Engine& engine);

  /// Append a dag's clusters/vertices/indices into the shared pools.
  /// Returns the mesh id (index into the registry). Must be called
  /// before setInstances' first use of the id; dags are copied - the
  /// caller may release its ClusterDag afterwards.
  u32 addMesh(const geometry::ClusterDag& dag);

  /// Upload the shared pools (once, after all addMesh calls) with a CRC
  /// round-trip per buffer. Idempotent until the next addMesh.
  u32 flushPools();

  /// Replace the instance table (one-shot upload; per-frame updates
  /// move to the staging ring in a later stage).
  void setInstances(const std::vector<VgInstance>& instances);

  const std::vector<VgMeshRange>& meshes() const { return m_meshRanges; }
  u32 clusterCount() const { return m_clusterCount; }
  u32 groupCount() const { return m_groupCount; }
  u32 instanceCount() const { return m_instanceCount; }
  u32 maxLodLevel() const { return m_maxLodLevel; }

  // Buffer handles for the S4.2+ passes (bind as SSBOs).
  u32 clusterBuffer() const { return m_clusterBuffer; }
  u32 groupBuffer() const { return m_groupBuffer; }
  u32 positionBuffer() const { return m_positionBuffer; }
  u32 normalBuffer() const { return m_normalBuffer; }
  u32 uvBuffer() const { return m_uvBuffer; }
  u32 indexBuffer() const { return m_indexBuffer; }
  u32 instanceBuffer() const { return m_instanceBuffer; }

  // Vertex/index counts (host-side pool sizes).
  u32 vertexCount() const { return static_cast<u32>(m_pos4.size()); }
  u32 indexCount() const { return static_cast<u32>(m_indices.size()); }

  // Host mirrors of the GPU pools - inputs for the one-shot chain the
  // app drives until the passes move in-frame (cluster cull wants the
  // packed tables, the rasterizer the float4 positions + rebased
  // indices).
  const std::vector<u8>& clusterData() const { return m_clusterData; }
  const std::vector<u8>& groupData() const { return m_groupData; }
  const std::vector<glm::vec4>& positions4() const { return m_pos4; }
  const std::vector<u32>& indices() const { return m_indices; }

  void onShutdown();

 private:
  backend::IRenderBackend* m_backend = nullptr;
  std::vector<VgMeshRange> m_meshRanges;
  u32 m_clusterCount = 0;
  u32 m_groupCount = 0;
  u32 m_instanceCount = 0;
  u32 m_maxLodLevel = 0;

  // Packed std430 staging accumulators (grown by addMesh, uploaded once
  // in flushPools - see the .cpp). Positions/normals are vec4 elements
  // (std430 vec3 arrays stride 16 - the shaders read float4).
  std::vector<u8> m_clusterData;
  std::vector<u8> m_groupData;
  std::vector<glm::vec4> m_pos4;
  std::vector<glm::vec4> m_norm4;
  std::vector<glm::vec2> m_uv2;
  std::vector<u32> m_indices;
  bool m_poolsDirty = false;

  u32 m_clusterBuffer = backend::IRenderBackend::kInvalidBufferHandle;
  u32 m_groupBuffer = backend::IRenderBackend::kInvalidBufferHandle;
  u32 m_positionBuffer = backend::IRenderBackend::kInvalidBufferHandle;
  u32 m_normalBuffer = backend::IRenderBackend::kInvalidBufferHandle;
  u32 m_uvBuffer = backend::IRenderBackend::kInvalidBufferHandle;
  u32 m_indexBuffer = backend::IRenderBackend::kInvalidBufferHandle;
  u32 m_instanceBuffer = backend::IRenderBackend::kInvalidBufferHandle;
};

}  // namespace renderer

KENSHIN_NS_END
