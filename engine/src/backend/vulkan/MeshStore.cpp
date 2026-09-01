#include "MeshStore.h"

#include "UploadEngine.h"
#include "Vk.h"
#include "VulkanDevice.h"
#include "kenshin/core/Log.h"

#include <cstdio>

KENSHIN_NS_BEGIN

namespace backend {

MeshStore::MeshResource MeshStore::detach(u32 handle) {
  if (!valid(handle)) return {};
  MeshResource raw = m_slots[handle - 1];
  m_slots[handle - 1] = {};
  return raw;
}

void MeshStore::destroyRaw(VmaAllocator allocator, const MeshResource& raw) {
  if (raw.positions != VK_NULL_HANDLE)
    vmaDestroyBuffer(allocator, raw.positions, raw.positionsAlloc);
  if (raw.normals != VK_NULL_HANDLE)
    vmaDestroyBuffer(allocator, raw.normals, raw.normalsAlloc);
  if (raw.uvs != VK_NULL_HANDLE)
    vmaDestroyBuffer(allocator, raw.uvs, raw.uvsAlloc);
  if (raw.indices != VK_NULL_HANDLE)
    vmaDestroyBuffer(allocator, raw.indices, raw.indicesAlloc);
}

void MeshStore::destroyAll(VmaAllocator allocator) {
  for (const MeshResource& m : m_slots)
    if (m.positions != VK_NULL_HANDLE || m.indices != VK_NULL_HANDLE)
      destroyRaw(allocator, m);
  m_slots.clear();
}

u32 MeshStore::create(const VulkanDevice& device, VkCommandPool pool,
                      const IRenderBackend::MeshUpload& upload) {
  if (!upload.positions || !upload.indices || upload.vertexCount == 0 ||
      upload.indexCount == 0)
    return IRenderBackend::kInvalidMeshHandle;

  MeshResource mesh;
  const auto uploadF32 = [&](const f32* data, usize floatCount,
                             VkBufferUsageFlags usage, VkBuffer& buffer,
                             VmaAllocation& alloc) {
    return uploadDeviceBuffer(
        device.allocator, device.device, device.graphicsQueue,
        pool, data,
        VkDeviceSize(floatCount) * sizeof(f32), usage, buffer, alloc);
  };
  const bool ok =
      uploadF32(upload.positions, usize(upload.vertexCount) * 3,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.positions,
                mesh.positionsAlloc) &&
      (!upload.normals ||
       uploadF32(upload.normals, usize(upload.vertexCount) * 3,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.normals,
                 mesh.normalsAlloc)) &&
      (!upload.uvs ||
       uploadF32(upload.uvs, usize(upload.vertexCount) * 2,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.uvs, mesh.uvsAlloc)) &&
      uploadDeviceBuffer(
          device.allocator, device.device, device.graphicsQueue,
          pool, upload.indices,
          VkDeviceSize(upload.indexCount) * sizeof(u32),
          VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh.indices, mesh.indicesAlloc);
  if (!ok) {
    destroyRaw(device.allocator, mesh);
    KENSHIN_ERROR("[baseline] mesh buffer upload failed");
    return IRenderBackend::kInvalidMeshHandle;
  }
  mesh.vertexCount = upload.vertexCount;
  mesh.indexCount = upload.indexCount;
  m_slots.push_back(mesh);
  const u32 handle = static_cast<u32>(m_slots.size());
  char name[40];
  snprintf(name, sizeof(name), "mesh-%u-positions", handle);
  vkutil::setName(device.device, VK_OBJECT_TYPE_BUFFER,
                  (uint64_t)m_slots[handle - 1].positions, name);
  snprintf(name, sizeof(name), "mesh-%u-indices", handle);
  vkutil::setName(device.device, VK_OBJECT_TYPE_BUFFER,
                  (uint64_t)m_slots[handle - 1].indices, name);
  return handle;
}

}  // namespace backend

KENSHIN_NS_END
