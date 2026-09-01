#include "DescriptorCache.h"

#include "Vk.h"
#include "kenshin/core/Log.h"

KENSHIN_NS_BEGIN

namespace backend {

bool DescriptorCache::initImGuiPool(VkDevice device) {
  // All three sizes counted: the ImGui Vulkan backend allocates
  // VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE sets for its texture descriptors, and
  // undercounting pool sizes is a validation error (this exact pool shipped
  // with poolSizeCount=2 for three array entries - the long-standing
  // SAMPLED_IMAGE warning, fixed in P0-3).
  const VkDescriptorPoolSize poolSizes[] = {
      {VK_DESCRIPTOR_TYPE_SAMPLER, 16},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 64},
  };
  const VkDescriptorPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
      .maxSets = 64,
      .poolSizeCount = 3,
      .pPoolSizes = poolSizes,
  };
  if (!KE_VK(vkCreateDescriptorPool(device, &poolInfo, nullptr,
                                    &m_imguiPool))) {
    KENSHIN_ERROR("[imgui] descriptor pool creation failed");
    return false;
  }
  return true;
}

void DescriptorCache::shutdown(VkDevice device) {
  if (m_imguiPool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(device, m_imguiPool, nullptr);
    m_imguiPool = VK_NULL_HANDLE;
  }
}

VkDescriptorPool DescriptorCache::createTransientPool(
    VkDevice device, VkDescriptorType type, u32 descriptorCount, u32 maxSets) {
  const VkDescriptorPoolSize poolSize{
      .type = type,
      .descriptorCount = descriptorCount,
  };
  const VkDescriptorPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = maxSets,
      .poolSizeCount = 1,
      .pPoolSizes = &poolSize,
  };
  VkDescriptorPool pool = VK_NULL_HANDLE;
  KE_VK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool));
  return pool;
}

void DescriptorCache::destroyTransientPool(VkDevice device,
                                           VkDescriptorPool pool) {
  if (pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, pool, nullptr);
}

}  // namespace backend

KENSHIN_NS_END
