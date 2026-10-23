#include "BindlessTable.h"

#include "Vk.h"
#include "kenshin/core/Log.h"

KENSHIN_NS_BEGIN
namespace backend {

namespace {
VkSamplerAddressMode addressMode(WrapMode m) {
  switch (m) {
    case WrapMode::Repeat: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case WrapMode::Clamp: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case WrapMode::Mirror: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
  }
  return VK_SAMPLER_ADDRESS_MODE_REPEAT;
}
}  // namespace

bool BindlessTable::init(VkDevice device, u32 capacity, f32 maxAnisotropy,
                         VkImageView whiteView) {
  m_capacity = capacity;
  m_white = whiteView;
  for (u32 i = 0; i < kSamplerCount; ++i) {
    const SamplerDesc d = samplerFromIndex(i);
    const bool linear = d.filter == FilterMode::Linear;
    const VkSamplerCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .minFilter = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .mipmapMode = linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = addressMode(d.wrapU),
        .addressModeV = addressMode(d.wrapV),
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .anisotropyEnable = linear && maxAnisotropy > 1.0f ? VK_TRUE : VK_FALSE,
        .maxAnisotropy = linear && maxAnisotropy > 1.0f ? maxAnisotropy : 1.0f,
        .maxLod = VK_LOD_CLAMP_NONE,
    };
    if (!KE_VK(vkCreateSampler(device, &info, nullptr, &m_samplers[i]))) return false;
  }

  const VkDescriptorSetLayoutBinding bindings[2] = {
      {
          .binding = 0,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          .descriptorCount = capacity,
          .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT |
                        VK_SHADER_STAGE_COMPUTE_BIT,
      },
      {
          .binding = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
          .descriptorCount = kSamplerCount,
          .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT |
                        VK_SHADER_STAGE_COMPUTE_BIT,
          .pImmutableSamplers = m_samplers,
      },
  };
  const VkDescriptorBindingFlags flags[2] = {
      VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
          VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT,
      0,
  };
  const VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
      .bindingCount = 2,
      .pBindingFlags = flags,
  };
  const VkDescriptorSetLayoutCreateInfo layoutInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = &flagsInfo,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
      .bindingCount = 2,
      .pBindings = bindings,
  };
  if (!KE_VK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_layout))) return false;

  const VkDescriptorPoolSize sizes[2] = {
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, capacity},
      {VK_DESCRIPTOR_TYPE_SAMPLER, kSamplerCount},
  };
  const VkDescriptorPoolCreateInfo poolInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
      .maxSets = 1,
      .poolSizeCount = 2,
      .pPoolSizes = sizes,
  };
  if (!KE_VK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_pool))) return false;
  const VkDescriptorSetAllocateInfo allocInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = m_pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &m_layout,
  };
  if (!KE_VK(vkAllocateDescriptorSets(device, &allocInfo, &m_set))) return false;
  write(device, 0, m_white);
  KENSHIN_INFO("[vulkan] bindless textures: {} slots, anisotropy {}x", capacity,
               maxAnisotropy > 1.0f ? maxAnisotropy : 1.0f);
  return true;
}

void BindlessTable::shutdown(VkDevice device) {
  if (m_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, m_pool, nullptr);
  if (m_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, m_layout, nullptr);
  for (VkSampler& s : m_samplers)
    if (s != VK_NULL_HANDLE) {
      vkDestroySampler(device, s, nullptr);
      s = VK_NULL_HANDLE;
    }
  m_pool = VK_NULL_HANDLE;
  m_layout = VK_NULL_HANDLE;
  m_set = VK_NULL_HANDLE;
  m_free.clear();
  m_next = 1;
  m_used = 0;
}

void BindlessTable::write(VkDevice device, u32 slot, VkImageView view) {
  const VkDescriptorImageInfo image{
      .imageView = view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
  };
  const VkWriteDescriptorSet w{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = m_set,
      .dstBinding = 0,
      .dstArrayElement = slot,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
      .pImageInfo = &image,
  };
  vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
}

u32 BindlessTable::allocate(VkDevice device, VkImageView view) {
  u32 slot;
  if (!m_free.empty()) {
    slot = m_free.back();
    m_free.pop_back();
  } else if (m_next < m_capacity) {
    slot = m_next++;
  } else {
    KENSHIN_ERROR("[vulkan] bindless texture table full ({} slots)", m_capacity);
    return kInvalidSlot;
  }
  write(device, slot, view);
  ++m_used;
  return slot;
}

void BindlessTable::release(VkDevice device, u32 slot) {
  if (slot == 0 || slot >= m_next) return;
  write(device, slot, m_white);
  m_free.push_back(slot);
  --m_used;
}

}  // namespace backend
KENSHIN_NS_END
