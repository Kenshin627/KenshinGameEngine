#include "ComputeOneShot.h"

#include "Vk.h"
#include "VulkanDevice.h"
#include "kenshin/core/Log.h"
#include "kenshin/manager/ShaderManager.h"

#include <cstring>

KENSHIN_NS_BEGIN

namespace backend {

bool runComputeOneshotJob(const VulkanDevice& device_,
                          VkCommandPool commandPool,
                          const IRenderBackend::Job& job,
                          std::vector<u8> outputs[IRenderBackend::IRenderBackend::Job::kMaxBuffers]) {
  VkDevice device = device_.device;
  VmaAllocator allocator = device_.allocator;
  if (!device || !job.spirv || job.spirvBytes == 0) return false;

  // Count used buffers.
  u32 bufferCount = 0;
  for (u32 i = 0; i < IRenderBackend::Job::kMaxBuffers; ++i)
    if (job.buffers[i].size > 0) bufferCount = i + 1;

  // Shader module + descriptor set layout + pipeline layout + pipeline.
  const VkShaderModuleCreateInfo moduleInfo{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = job.spirvBytes,
      .pCode = job.spirv,
  };
  VkShaderModule module = VK_NULL_HANDLE;
  if (!KE_VK(vkCreateShaderModule(device, &moduleInfo, nullptr, &module))) {
    KENSHIN_ERROR("[compute] shader module creation failed");
    return false;
  }

  // Descriptor layout from shader reflection: the compiled SPIR-V is the
  // authority on bindings (spirv-reflect via ShaderManager), replacing a
  // hand-maintained copy. Job convention: buffers[i] feeds binding i; if
  // reflection is unavailable or disagrees with that, fall back to it.
  std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
  std::vector<manager::ReflectedBinding> reflected;
  const std::vector<u32> spirvWords(job.spirv,
                                    job.spirv + job.spirvBytes / sizeof(u32));
  bool reflectedOk = false;
  if (manager::reflectDescriptorBindings(spirvWords, reflected)) {
    u32 storageCount = 0;
    for (const auto& r : reflected)
      if (r.type == manager::ReflectedBinding::Type::StorageBuffer &&
          r.set == 0)
        ++storageCount;
    reflectedOk = storageCount == bufferCount;
  }
  if (reflectedOk) {
    for (const auto& r : reflected) {
      if (r.type != manager::ReflectedBinding::Type::StorageBuffer) continue;
      layoutBindings.push_back(VkDescriptorSetLayoutBinding{
          .binding = r.binding,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      });
    }
  } else {
    for (u32 i = 0; i < bufferCount; ++i) {
      layoutBindings.push_back(VkDescriptorSetLayoutBinding{
          .binding = i,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      });
    }
  }
  const VkDescriptorSetLayoutCreateInfo setLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = static_cast<u32>(layoutBindings.size()),
      .pBindings = layoutBindings.data(),
  };
  VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
  KE_VK(vkCreateDescriptorSetLayout(device, &setLayoutInfo, nullptr, &setLayout));

  const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = bufferCount ? 1u : 0u,
      .pSetLayouts = &setLayout,
  };
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  KE_VK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout));

  const VkComputePipelineCreateInfo pipelineInfo{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage =
          VkPipelineShaderStageCreateInfo{
              .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
              .stage = VK_SHADER_STAGE_COMPUTE_BIT,
              .module = module,
              .pName = job.entryPoint,
          },
      .layout = pipelineLayout,
  };
  VkPipeline pipeline = VK_NULL_HANDLE;
  if (!KE_VK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                               nullptr, &pipeline))) {
    KENSHIN_ERROR("[compute] pipeline creation failed");
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
    vkDestroyShaderModule(device, module, nullptr);
    return false;
  }

  // Host-visible, host-coherent buffers (smoke-test scale: simple over fast).
  VkBuffer buffers[IRenderBackend::Job::kMaxBuffers] = {};
  VmaAllocation allocations[IRenderBackend::Job::kMaxBuffers] = {};
  void* mapped[IRenderBackend::Job::kMaxBuffers] = {};
  bool ok = true;
  for (u32 i = 0; i < bufferCount && ok; ++i) {
    const VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = static_cast<VkDeviceSize>(job.buffers[i].size),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo allocInfo{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    };
    VmaAllocationInfo allocInfoOut{};
    if (vmaCreateBuffer(allocator, &bufferInfo, &allocInfo,
                        &buffers[i], &allocations[i],
                        &allocInfoOut) != VK_SUCCESS) {
      KENSHIN_ERROR("[compute] buffer creation failed (slot {})", i);
      ok = false;
      break;
    }
    mapped[i] = allocInfoOut.pMappedData;
    if (job.buffers[i].initData)
      std::memcpy(mapped[i], job.buffers[i].initData, job.buffers[i].size);
  }

  // Descriptor set from a transient pool.
  VkDescriptorPool pool = VK_NULL_HANDLE;
  VkDescriptorSet set = VK_NULL_HANDLE;
  if (ok && bufferCount > 0) {
    const VkDescriptorPoolSize poolSize{
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = IRenderBackend::Job::kMaxBuffers,
    };
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };
    KE_VK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool));

    const VkDescriptorSetAllocateInfo setInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &setLayout,
    };
    if (!KE_VK(vkAllocateDescriptorSets(device, &setInfo, &set))) {
      KENSHIN_ERROR("[compute] descriptor set allocation failed");
      ok = false;
    }
  }
  if (ok && bufferCount > 0) {
    VkDescriptorBufferInfo bufferInfos[IRenderBackend::Job::kMaxBuffers] = {};
    VkWriteDescriptorSet writes[IRenderBackend::Job::kMaxBuffers] = {};
    for (u32 i = 0; i < bufferCount; ++i) {
      bufferInfos[i] = VkDescriptorBufferInfo{
          .buffer = buffers[i],
          .range = VK_WHOLE_SIZE,
      };
      writes[i] = VkWriteDescriptorSet{
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = set,
          .dstBinding = layoutBindings[i].binding,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &bufferInfos[i],
      };
    }
    vkUpdateDescriptorSets(device, bufferCount, writes, 0, nullptr);
  }

  // Record + submit the dispatch, then wait (host-coherent: submit and
  // queueWaitIdle provide the necessary host sync boundaries).
  if (ok) {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    const VkCommandBufferAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    KE_VK(vkAllocateCommandBuffers(device, &allocInfo, &cmd));
    const VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    KE_VK(vkBeginCommandBuffer(cmd, &beginInfo));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    if (bufferCount > 0)
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdDispatch(cmd, job.groupCountX, 1, 1);
    KE_VK(vkEndCommandBuffer(cmd));

    const VkCommandBufferSubmitInfo cmdSubmit{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = cmd,
    };
    const VkSubmitInfo2 submit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &cmdSubmit,
    };
    KE_VK(vkQueueSubmit2(device_.graphicsQueue, 1, &submit, VK_NULL_HANDLE));
    KE_VK(vkQueueWaitIdle(device_.graphicsQueue));
    vkFreeCommandBuffers(device, commandPool, 1, &cmd);
  }

  // Readback.
  if (ok) {
    for (u32 i = 0; i < bufferCount; ++i) {
      if (!job.buffers[i].readBack || !mapped[i]) continue;
      outputs[i].resize(job.buffers[i].size);
      std::memcpy(outputs[i].data(), mapped[i], job.buffers[i].size);
    }
  }

  // Teardown.
  if (pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, pool, nullptr);
  for (u32 i = 0; i < bufferCount; ++i)
    if (buffers[i] != VK_NULL_HANDLE)
      vmaDestroyBuffer(allocator, buffers[i], allocations[i]);
  vkDestroyPipeline(device, pipeline, nullptr);
  vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
  vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
  vkDestroyShaderModule(device, module, nullptr);
  return ok;
}

}  // namespace backend

KENSHIN_NS_END
