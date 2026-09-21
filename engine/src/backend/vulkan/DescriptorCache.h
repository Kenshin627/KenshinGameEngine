#pragma once

// ── DescriptorCache — descriptor pool ownership ──────────────────
// P0-1: owns the ImGui pool and hosts the transient one-shot pool
// helper. P0-3 grows the layered layout cache (per-frame / material /
// instance), shader-reflection-driven layouts, and the per-frame flush.

#include <volk.h>

#include <vector>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace backend {

class DescriptorCache {
 public:
  /// Pool backing the ImGui Vulkan backend (sizes match what its texture
  /// and font descriptors allocate).
  bool initImGuiPool(VkDevice device);
  void shutdown(VkDevice device);

  VkDescriptorPool imguiPool() const { return m_imguiPool; }

  /// Transient pool for a self-contained job (one set, one type); destroy
  /// with destroyTransientPool when the job is done.
  static VkDescriptorPool createTransientPool(VkDevice device,
                                              VkDescriptorType type,
                                              u32 descriptorCount, u32 maxSets);
  static void destroyTransientPool(VkDevice device, VkDescriptorPool pool);

  /// ---- reflection-driven SSBO set (shared by one-shot compute and the
  /// fullscreen pass) -----------------------------------------------
  /// A one-set-per-slot layout covering every STORAGE_BUFFER binding the
  /// SPIR-V declares (binding slots from reflection, one descriptor
  /// each - the ComputeOneShot-proven pattern). `bindingCountOut`
  /// (optional) returns the binding count for sizing the pool.
  static VkDescriptorSetLayout createStorageLayout(
      VkDevice device, const u32* spirv, usize spirvWords,
      u32* bindingCountOut = nullptr);
  /// Pool for `setCount` sets over that layout; destroy via
  /// destroyTransientPool.
  static VkDescriptorPool createStoragePool(VkDevice device,
                                            VkDescriptorSetLayout layout,
                                            u32 bindingCount, u32 setCount);
  /// M8.1 lighting pass: a MIXED layout - reflected StorageBuffer
  /// bindings become STORAGE_BUFFER, reflected SampledImage bindings
  /// (Slang Sampler2D reflects as combined) become
  /// COMBINED_IMAGE_SAMPLER. `typesOut` (capacity >= bindingCount)
  /// receives the per-binding type so the draw path knows which writes
  /// take image infos. Dense 0..maxBinding like createStorageLayout.
  static constexpr u32 kMaxMixedBindings = 16;
  static VkDescriptorSetLayout createMixedLayout(VkDevice device,
                                                 const u32* spirv,
                                                 usize spirvWords,
                                                 VkDescriptorType* typesOut,
                                                 u32* bindingCountOut);
  /// Pool sized for the mixed type mix; destroy via
  /// destroyTransientPool.
  static VkDescriptorPool createMixedPool(VkDevice device,
                                          const VkDescriptorType* types,
                                          u32 bindingCount, u32 setCount);
  /// Allocate `count` sets (per-frame-slot rotation: a set is re-written
  /// only when its slot's prior frame has provably completed).
  static bool allocateSets(VkDevice device, VkDescriptorPool pool,
                           VkDescriptorSetLayout layout, VkDescriptorSet* out,
                           u32 count);

 private:
  VkDescriptorPool m_imguiPool = VK_NULL_HANDLE;
};

}  // namespace backend

KENSHIN_NS_END
