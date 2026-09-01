#pragma once

// ── DescriptorCache — descriptor pool ownership ──────────────────
// P0-1: owns the ImGui pool and hosts the transient one-shot pool
// helper. P0-3 grows the layered layout cache (per-frame / material /
// instance), shader-reflection-driven layouts, and the per-frame flush.

#include <volk.h>

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

 private:
  VkDescriptorPool m_imguiPool = VK_NULL_HANDLE;
};

}  // namespace backend

KENSHIN_NS_END
