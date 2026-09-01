#pragma once

// ── MeshStore — uploaded mesh table ─────────────────────────────
// SOA vertex buffers (positions/normals/uvs + indices, device-local)
// behind u32 handles (index + 1). Creation uploads through the
// one-shot path and names the buffers for debugging tools; teardown
// goes through the frame slots' deletion queues (deferred) or
// destroyAll at shutdown.

#include <volk.h>

#include <vma/vk_mem_alloc.h>

#include <vector>

#include "kenshin/Common.h"
#include "kenshin/backend/RenderBackend.h"

KENSHIN_NS_BEGIN

namespace backend {

struct VulkanDevice;

class MeshStore {
 public:
  struct MeshResource {
    VkBuffer positions = VK_NULL_HANDLE;
    VmaAllocation positionsAlloc = VK_NULL_HANDLE;
    VkBuffer normals = VK_NULL_HANDLE;
    VmaAllocation normalsAlloc = VK_NULL_HANDLE;
    VkBuffer uvs = VK_NULL_HANDLE;
    VmaAllocation uvsAlloc = VK_NULL_HANDLE;
    VkBuffer indices = VK_NULL_HANDLE;
    VmaAllocation indicesAlloc = VK_NULL_HANDLE;
    u32 vertexCount = 0;
    u32 indexCount = 0;
  };

  /// One-shot upload of an SOA PNU mesh; names the buffers. Returns the
  /// handle or kInvalidMeshHandle on failure. `pool` must not be
  /// concurrently recorded (frame slot 0's is the norm).
  u32 create(const VulkanDevice& device, VkCommandPool pool,
             const IRenderBackend::MeshUpload& upload);

  bool valid(u32 handle) const {
    return handle != IRenderBackend::kInvalidMeshHandle && handle != 0 &&
           handle <= m_slots.size();
  }
  const MeshResource& mesh(u32 handle) const { return m_slots[handle - 1]; }

  /// Zero the slot and return its raw resources (for a deletion-queue
  /// lambda that destroys them when the frame completes).
  MeshResource detach(u32 handle);

  /// Immediate destruction of detached resources.
  static void destroyRaw(VmaAllocator allocator, const MeshResource& raw);

  /// Destroy everything (shutdown, after device idle).
  void destroyAll(VmaAllocator allocator);

 private:
  std::vector<MeshResource> m_slots;
};

}  // namespace backend

KENSHIN_NS_END
