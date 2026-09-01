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

  // Buffer handles for the S4.2+ passes (bind as SSBOs).
  u32 clusterBuffer() const { return m_clusterBuffer; }
  u32 groupBuffer() const { return m_groupBuffer; }
  u32 positionBuffer() const { return m_positionBuffer; }
  u32 normalBuffer() const { return m_normalBuffer; }
  u32 uvBuffer() const { return m_uvBuffer; }
  u32 indexBuffer() const { return m_indexBuffer; }
  u32 instanceBuffer() const { return m_instanceBuffer; }

  void onShutdown();

 private:
  backend::IRenderBackend* m_backend = nullptr;
  std::vector<VgMeshRange> m_meshRanges;
  u32 m_clusterCount = 0;
  u32 m_groupCount = 0;

  // Packed std430 staging accumulators (grown by addMesh, uploaded once
  // in flushPools - see the .cpp).
  std::vector<u8> m_clusterData;
  std::vector<u8> m_groupData;
  std::vector<f32> m_positions;
  std::vector<f32> m_normals;
  std::vector<f32> m_uvs;
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
