#include "DescriptorCache.h"

#include "Vk.h"
#include "kenshin/core/Log.h"
#include "kenshin/manager/ShaderManager.h"

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
  if (!KE_VK(
          vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_imguiPool))) {
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

VkDescriptorPool DescriptorCache::createTransientPool(VkDevice device,
                                                      VkDescriptorType type,
                                                      u32 descriptorCount,
                                                      u32 maxSets) {
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

VkDescriptorSetLayout DescriptorCache::createStorageLayout(
    VkDevice device, const u32* spirv, usize spirvWords, u32* bindingCountOut) {
  std::vector<manager::ReflectedBinding> reflected;
  if (!manager::reflectDescriptorBindings(
          std::vector<u32>(spirv, spirv + spirvWords), reflected)) {
    KENSHIN_ERROR("[descriptors] reflection failed");
    return VK_NULL_HANDLE;
  }
  u32 maxBinding = 0;
  u32 count = 0;
  for (const auto& r : reflected)
    if (r.set == 0 && r.type == manager::ReflectedBinding::Type::StorageBuffer) {
      maxBinding = std::max(maxBinding, r.binding);
      ++count;
    }
  if (count == 0) return VK_NULL_HANDLE;
  if (bindingCountOut) *bindingCountOut = maxBinding + 1;

  // One descriptor per reflected binding (dense 0..maxBinding) - the
  // same shape ComputeOneShot's reflection-driven layouts use.
  std::vector<VkDescriptorSetLayoutBinding> bindings(maxBinding + 1);
  for (u32 b = 0; b <= maxBinding; ++b)
    bindings[b] = VkDescriptorSetLayoutBinding{
        .binding = b,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT |
                      VK_SHADER_STAGE_FRAGMENT_BIT,
    };
  const VkDescriptorSetLayoutCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = static_cast<u32>(bindings.size()),
      .pBindings = bindings.data(),
  };
  VkDescriptorSetLayout layout = VK_NULL_HANDLE;
  if (!KE_VK(vkCreateDescriptorSetLayout(device, &info, nullptr, &layout)))
    return VK_NULL_HANDLE;
  return layout;
}

VkDescriptorPool DescriptorCache::createStoragePool(
    VkDevice device, VkDescriptorSetLayout layout, u32 bindingCount,
    u32 setCount) {
  if (layout == VK_NULL_HANDLE || bindingCount == 0 || setCount == 0)
    return VK_NULL_HANDLE;
  const VkDescriptorPoolSize size{
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = bindingCount * setCount,
  };
  const VkDescriptorPoolCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = setCount,
      .poolSizeCount = 1,
      .pPoolSizes = &size,
  };
  VkDescriptorPool pool = VK_NULL_HANDLE;
  if (!KE_VK(vkCreateDescriptorPool(device, &info, nullptr, &pool)))
    return VK_NULL_HANDLE;
  return pool;
}

VkDescriptorSetLayout DescriptorCache::createMixedLayout(
    VkDevice device, const u32* spirv, usize spirvWords,
    VkDescriptorType* typesOut, u32* bindingCountOut) {
  std::vector<manager::ReflectedBinding> reflected;
  if (!manager::reflectDescriptorBindings(
          std::vector<u32>(spirv, spirv + spirvWords), reflected)) {
    KENSHIN_ERROR("[descriptors] reflection failed");
    return VK_NULL_HANDLE;
  }
  u32 maxBinding = 0;
  u32 count = 0;
  for (const auto& r : reflected) {
    const bool usable =
        r.set == 0 && (r.type == manager::ReflectedBinding::Type::StorageBuffer ||
        r.type == manager::ReflectedBinding::Type::CombinedImageSampler ||
        r.type == manager::ReflectedBinding::Type::SampledTexture ||
        r.type == manager::ReflectedBinding::Type::Sampler);
    if (usable) {
      maxBinding = std::max(maxBinding, r.binding);
      ++count;
    }
  }
  if (count == 0 || maxBinding >= kMaxMixedBindings) return VK_NULL_HANDLE;

  // Dense 0..maxBinding; gaps stay the SSBO type (never written by the
  // draw path, harmless). Reflected image/sampler types override their
  // slot (see the header contract).
  std::vector<VkDescriptorSetLayoutBinding> bindings(maxBinding + 1);
  for (u32 b = 0; b <= maxBinding; ++b) {
    typesOut[b] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[b] = VkDescriptorSetLayoutBinding{
        .binding = b,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT |
                      VK_SHADER_STAGE_FRAGMENT_BIT,
    };
  }
  for (const auto& r : reflected) {
    if (r.set != 0 || r.binding > maxBinding) continue;
    VkDescriptorType t = VK_DESCRIPTOR_TYPE_MAX_ENUM;
    if (r.type == manager::ReflectedBinding::Type::CombinedImageSampler)
      t = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    else if (r.type == manager::ReflectedBinding::Type::SampledTexture)
      t = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    else if (r.type == manager::ReflectedBinding::Type::Sampler)
      t = VK_DESCRIPTOR_TYPE_SAMPLER;
    if (t != VK_DESCRIPTOR_TYPE_MAX_ENUM) {
      typesOut[r.binding] = t;
      bindings[r.binding].descriptorType = t;
    }
  }
  if (bindingCountOut) *bindingCountOut = maxBinding + 1;

  const VkDescriptorSetLayoutCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = static_cast<u32>(bindings.size()),
      .pBindings = bindings.data(),
  };
  VkDescriptorSetLayout layout = VK_NULL_HANDLE;
  if (!KE_VK(vkCreateDescriptorSetLayout(device, &info, nullptr, &layout)))
    return VK_NULL_HANDLE;
  return layout;
}

bool DescriptorCache::reflectUsesSet(const u32* spirv, usize spirvWords, u32 set) {
  std::vector<manager::ReflectedBinding> reflected;
  if (!manager::reflectDescriptorBindings(std::vector<u32>(spirv, spirv + spirvWords),
                                          reflected))
    return false;
  for (const auto& r : reflected)
    if (r.set == set) return true;
  return false;
}

VkDescriptorPool DescriptorCache::createMixedPool(VkDevice device,
                                                  const VkDescriptorType* types,
                                                  u32 bindingCount,
                                                  u32 setCount) {
  u32 ssbo = 0, cis = 0, sampled = 0, samp = 0;
  for (u32 b = 0; b < bindingCount; ++b) {
    if (types[b] == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
      ++cis;
    else if (types[b] == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE)
      ++sampled;
    else if (types[b] == VK_DESCRIPTOR_TYPE_SAMPLER)
      ++samp;
    else
      ++ssbo;
  }
  VkDescriptorPoolSize sizes[4] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ssbo * setCount},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, cis * setCount},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, sampled * setCount},
      {VK_DESCRIPTOR_TYPE_SAMPLER, samp * setCount},
  };
  u32 sizeCount = 0;
  for (u32 i = 0; i < 4; ++i)
    if (sizes[i].descriptorCount > 0) sizes[sizeCount++] = sizes[i];
  const VkDescriptorPoolCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = setCount,
      .poolSizeCount = sizeCount,
      .pPoolSizes = sizes,
  };
  VkDescriptorPool pool = VK_NULL_HANDLE;
  if (!KE_VK(vkCreateDescriptorPool(device, &info, nullptr, &pool)))
    return VK_NULL_HANDLE;
  return pool;
}

bool DescriptorCache::allocateSets(VkDevice device, VkDescriptorPool pool,
                                   VkDescriptorSetLayout layout,
                                   VkDescriptorSet* out, u32 count) {
  if (pool == VK_NULL_HANDLE || out == nullptr) return false;
  std::vector<VkDescriptorSetLayout> layouts(count, layout);
  const VkDescriptorSetAllocateInfo alloc{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool,
      .descriptorSetCount = count,
      .pSetLayouts = layouts.data(),
  };
  return KE_VK(vkAllocateDescriptorSets(device, &alloc, out));
}

}  // namespace backend

KENSHIN_NS_END
