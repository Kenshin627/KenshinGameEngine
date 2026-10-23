#pragma once

// ── Bindless material texture table (internal to the backend) ────
// Descriptor set 1 of every pipeline that samples material textures:
//   binding 0: Texture2D[capacity]  (partially bound, update-after-bind)
//   binding 1: SamplerState[kSamplerCount]  (immutable; TextureFormat.h)
// One global set, written when a texture is created or destroyed and
// never per frame. Slot 0 permanently holds a 1x1 white texture, so a
// stale or default index samples white rather than garbage.

#include <volk.h>

#include <vector>

#include "kenshin/backend/TextureFormat.h"

KENSHIN_NS_BEGIN
namespace backend {

class BindlessTable {
 public:
  static constexpr u32 kInvalidSlot = 0xFFFFFFFFu;

  /// `capacity` slots (clamped to the device limit by the caller);
  /// `maxAnisotropy` <= 1 disables anisotropic filtering.
  bool init(VkDevice device, u32 capacity, f32 maxAnisotropy, VkImageView whiteView);
  void shutdown(VkDevice device);

  /// Writes `view` (SHADER_READ_ONLY_OPTIMAL, 2D) into a free slot.
  u32 allocate(VkDevice device, VkImageView view);
  /// Points the slot back at white and frees it. The caller guarantees
  /// no in-flight work samples it (destroyTexture drains the device).
  void release(VkDevice device, u32 slot);

  VkDescriptorSetLayout layout() const { return m_layout; }
  VkDescriptorSet set() const { return m_set; }
  u32 capacity() const { return m_capacity; }
  u32 used() const { return m_used; }

 private:
  void write(VkDevice device, u32 slot, VkImageView view);

  VkSampler m_samplers[kSamplerCount] = {};
  VkDescriptorSetLayout m_layout = VK_NULL_HANDLE;
  VkDescriptorPool m_pool = VK_NULL_HANDLE;
  VkDescriptorSet m_set = VK_NULL_HANDLE;
  VkImageView m_white = VK_NULL_HANDLE;
  u32 m_capacity = 0;
  u32 m_next = 1;  // slot 0 = white
  u32 m_used = 0;
  std::vector<u32> m_free;
};

}  // namespace backend
KENSHIN_NS_END
