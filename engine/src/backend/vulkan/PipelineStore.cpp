#include "PipelineStore.h"

#include <cstdio>
#include <filesystem>
#include <fstream>

#include "DescriptorCache.h"
#include "Vk.h"
#include "kenshin/core/Log.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

KENSHIN_NS_BEGIN

namespace backend {

namespace {
VkCompareOp toVk(IRenderBackend::CompareOp op) {
  using Op = IRenderBackend::CompareOp;
  switch (op) {
    case Op::Never:
      return VK_COMPARE_OP_NEVER;
    case Op::Less:
      return VK_COMPARE_OP_LESS;
    case Op::Equal:
      return VK_COMPARE_OP_EQUAL;
    case Op::LessOrEqual:
      return VK_COMPARE_OP_LESS_OR_EQUAL;
    case Op::Greater:
      return VK_COMPARE_OP_GREATER;
    case Op::NotEqual:
      return VK_COMPARE_OP_NOT_EQUAL;
    case Op::GreaterOrEqual:
      return VK_COMPARE_OP_GREATER_OR_EQUAL;
    default:
      return VK_COMPARE_OP_ALWAYS;
  }
}
VkCullModeFlags toVk(IRenderBackend::CullMode mode) {
  using M = IRenderBackend::CullMode;
  switch (mode) {
    case M::None:
      return VK_CULL_MODE_NONE;
    case M::Front:
      return VK_CULL_MODE_FRONT_BIT;
    default:
      return VK_CULL_MODE_BACK_BIT;
  }
}
VkFrontFace toVk(IRenderBackend::FrontFace face) {
  return face == IRenderBackend::FrontFace::Clockwise
             ? VK_FRONT_FACE_CLOCKWISE
             : VK_FRONT_FACE_COUNTER_CLOCKWISE;
}
}  // namespace

bool PipelineStore::init(VkDevice device) {
  std::vector<char> cacheData;
  wchar_t exePath[MAX_PATH]{};
  GetModuleFileNameW(nullptr, exePath, MAX_PATH);
  const auto cachePath =
      std::filesystem::path(exePath).parent_path() / "kenshin_pipeline.cache";
  if (std::filesystem::exists(cachePath)) {
    std::ifstream in(cachePath, std::ios::binary);
    in.seekg(0, std::ios::end);
    const std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size > 0) {
      cacheData.resize(static_cast<usize>(size));
      in.read(cacheData.data(), size);
    }
  }
  const VkPipelineCacheCreateInfo cacheInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
      .initialDataSize = cacheData.size(),
      .pInitialData = cacheData.data(),
  };
  return KE_VK(vkCreatePipelineCache(device, &cacheInfo, nullptr, &m_cache));
}

void PipelineStore::shutdown(VkDevice device) {
  for (const Pipeline& p : m_slots) {
    if (p.pipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(device, p.pipeline, nullptr);
    if (p.layout != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(device, p.layout, nullptr);
  }
  m_slots.clear();
  for (const Fullscreen& f : m_fullscreen) {
    if (!f.valid) continue;
    vkDestroyPipeline(device, f.pipeline, nullptr);
    vkDestroyPipelineLayout(device, f.layout, nullptr);
    vkDestroyDescriptorPool(device, f.pool, nullptr);
    vkDestroyDescriptorSetLayout(device, f.setLayout, nullptr);
  }
  m_fullscreen.clear();
  for (const Fullscreen& f : m_nullRaster) {
    if (!f.valid) continue;
    vkDestroyPipeline(device, f.pipeline, nullptr);
    vkDestroyPipelineLayout(device, f.layout, nullptr);
    vkDestroyDescriptorPool(device, f.pool, nullptr);
    vkDestroyDescriptorSetLayout(device, f.setLayout, nullptr);
  }
  m_nullRaster.clear();
  for (const Compute& c : m_compute) {
    if (!c.valid) continue;
    vkDestroyPipeline(device, c.pipeline, nullptr);
    vkDestroyPipelineLayout(device, c.layout, nullptr);
    vkDestroyDescriptorPool(device, c.pool, nullptr);
    vkDestroyDescriptorSetLayout(device, c.setLayout, nullptr);
  }
  m_compute.clear();
  if (m_cache != VK_NULL_HANDLE) {
    // Persist this session's driver compilations next to the exe;
    // rebuilt pipelines hit the cache on the next launch.
    size_t bytes = 0;
    if (KE_VK(vkGetPipelineCacheData(device, m_cache, &bytes, nullptr)) &&
        bytes > 0) {
      std::vector<char> data(bytes);
      if (KE_VK(vkGetPipelineCacheData(device, m_cache, &bytes, data.data()))) {
        wchar_t exePath[MAX_PATH]{};
        GetModuleFileNameW(nullptr, exePath, MAX_PATH);
        const auto cachePath = std::filesystem::path(exePath).parent_path() /
                               "kenshin_pipeline.cache";
        std::ofstream out(cachePath, std::ios::binary);
        out.write(data.data(), static_cast<std::streamsize>(bytes));
      }
    }
    vkDestroyPipelineCache(device, m_cache, nullptr);
    m_cache = VK_NULL_HANDLE;
  }
}

u32 PipelineStore::createFullscreen(
    VkDevice device, VkFormat colorFormat,
    const IRenderBackend::FullscreenPipelineInfo& info) {
  if (info.vsSpirv == nullptr || info.vsBytes == 0 || info.fsSpirv == nullptr ||
      info.fsBytes == 0)
    return IRenderBackend::kInvalidPipelineHandle;

  const VkShaderModuleCreateInfo vsInfo{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = info.vsBytes,
      .pCode = info.vsSpirv,
  };
  VkShaderModule vs = VK_NULL_HANDLE;
  if (!KE_VK(vkCreateShaderModule(device, &vsInfo, nullptr, &vs)))
    return IRenderBackend::kInvalidPipelineHandle;
  const VkShaderModuleCreateInfo fsInfo{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = info.fsBytes,
      .pCode = info.fsSpirv,
  };
  VkShaderModule fs = VK_NULL_HANDLE;
  if (!KE_VK(vkCreateShaderModule(device, &fsInfo, nullptr, &fs))) {
    vkDestroyShaderModule(device, vs, nullptr);
    return IRenderBackend::kInvalidPipelineHandle;
  }

  const VkPipelineShaderStageCreateInfo stages[2] = {
      VkPipelineShaderStageCreateInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT,
          .module = vs,
          .pName = "main",
      },
      VkPipelineShaderStageCreateInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
          .module = fs,
          .pName = "main",
      },
  };

  // SSBO set layout from the FS's reflected bindings; ONE set allocated
  // now and re-written per draw (buffers may change every frame). The
  // pool must cover the LAYOUT's totals (bindings x array size), not
  // just what a draw writes.
  u32 bindingCount = 0;
  const VkDescriptorSetLayout setLayout = DescriptorCache::createStorageLayout(
      device, info.fsSpirv, info.fsBytes / sizeof(u32), &bindingCount);
  if (setLayout == VK_NULL_HANDLE) {
    vkDestroyShaderModule(device, vs, nullptr);
    vkDestroyShaderModule(device, fs, nullptr);
    return IRenderBackend::kInvalidPipelineHandle;
  }
  VkDescriptorPool pool = DescriptorCache::createStoragePool(
      device, setLayout, bindingCount, kMaxFramesInFlight);
  VkDescriptorSet sets[kMaxFramesInFlight] = {};
  const bool setOk = pool != VK_NULL_HANDLE &&
                     DescriptorCache::allocateSets(device, pool, setLayout,
                                                   sets, kMaxFramesInFlight);

  const VkPushConstantRange pushRange{
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      .offset = 0,
      .size = 128,
  };
  const VkPipelineLayoutCreateInfo layoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &setLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pushRange,
  };
  VkPipelineLayout layout = VK_NULL_HANDLE;
  const bool layoutOk =
      KE_VK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &layout));

  const VkPipelineVertexInputStateCreateInfo vertexInput{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
  };
  const VkPipelineInputAssemblyStateCreateInfo inputAssembly{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
  };
  const VkPipelineViewportStateCreateInfo viewportState{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .scissorCount = 1,
  };
  const VkPipelineRasterizationStateCreateInfo rasterization{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .lineWidth = 1.0f,
  };
  const VkPipelineMultisampleStateCreateInfo multisample{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
  };
  // The VisBuffer already resolved visibility+depth: no depth test/write.
  const VkPipelineDepthStencilStateCreateInfo depthStencil{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
  };
  const VkPipelineColorBlendAttachmentState blendAttachment{
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
  };
  const VkPipelineColorBlendStateCreateInfo colorBlend{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &blendAttachment,
  };
  const VkDynamicState dynamicStates[2] = {VK_DYNAMIC_STATE_VIEWPORT,
                                           VK_DYNAMIC_STATE_SCISSOR};
  const VkPipelineDynamicStateCreateInfo dynamicState{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2,
      .pDynamicStates = dynamicStates,
  };
  const VkPipelineRenderingCreateInfoKHR renderingInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
      // The decode pass draws inside the frame scope's viewport pass,
      // which always attaches the D32 depth companion (no depth TEST in
      // the pipeline, but the formats must still match).
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &colorFormat,
      .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT,
  };

  VkPipeline pipeline = VK_NULL_HANDLE;
  bool pipeOk = false;
  if (layoutOk) {
    const VkGraphicsPipelineCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &renderingInfo,
        .stageCount = 2,
        .pStages = stages,
        .pVertexInputState = &vertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depthStencil,
        .pColorBlendState = &colorBlend,
        .pDynamicState = &dynamicState,
        .layout = layout,
    };
    pipeOk = KE_VK(vkCreateGraphicsPipelines(device, m_cache, 1, &createInfo,
                                             nullptr, &pipeline));
  }
  vkDestroyShaderModule(device, vs, nullptr);
  vkDestroyShaderModule(device, fs, nullptr);

  if (!setOk || !layoutOk || !pipeOk) {
    if (pipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(device, pipeline, nullptr);
    if (layout != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(device, layout, nullptr);
    if (pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, pool, nullptr);
    if (setLayout != VK_NULL_HANDLE)
      vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
    KENSHIN_ERROR("[vulkan] fullscreen pipeline creation failed");
    return IRenderBackend::kInvalidPipelineHandle;
  }

  m_fullscreen.emplace_back();
  Fullscreen& entry = m_fullscreen.back();
  entry.layout = layout;
  entry.pipeline = pipeline;
  entry.setLayout = setLayout;
  entry.pool = pool;
  for (u32 i = 0; i < kMaxFramesInFlight; ++i) entry.sets[i] = sets[i];
  entry.valid = true;
  return static_cast<u32>(m_fullscreen.size());
}

u32 PipelineStore::createFullscreenVariant(
    VkDevice device, const IRenderBackend::FullscreenPipelineInfo& info,
    const VkFormat* colorFormats, u32 colorCount, VkFormat depthFormat,
    bool mixedLayout, bool premultipliedBlend) {
  if (info.vsSpirv == nullptr || info.vsBytes == 0 || info.fsSpirv == nullptr ||
      info.fsBytes == 0)
    return IRenderBackend::kInvalidPipelineHandle;

  const VkShaderModuleCreateInfo vsInfo{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = info.vsBytes,
      .pCode = info.vsSpirv,
  };
  VkShaderModule vs = VK_NULL_HANDLE;
  if (!KE_VK(vkCreateShaderModule(device, &vsInfo, nullptr, &vs)))
    return IRenderBackend::kInvalidPipelineHandle;
  const VkShaderModuleCreateInfo fsInfo{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = info.fsBytes,
      .pCode = info.fsSpirv,
  };
  VkShaderModule fs = VK_NULL_HANDLE;
  if (!KE_VK(vkCreateShaderModule(device, &fsInfo, nullptr, &fs))) {
    vkDestroyShaderModule(device, vs, nullptr);
    return IRenderBackend::kInvalidPipelineHandle;
  }

  const VkPipelineShaderStageCreateInfo stages[2] = {
      VkPipelineShaderStageCreateInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT,
          .module = vs,
          .pName = "main",
      },
      VkPipelineShaderStageCreateInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
          .module = fs,
          .pName = "main",
      },
  };

  VkDescriptorType types[DescriptorCache::kMaxMixedBindings] = {};
  u32 bindingCount = 0;
  VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
  if (mixedLayout) {
    setLayout = DescriptorCache::createMixedLayout(
        device, info.fsSpirv, info.fsBytes / sizeof(u32), types, &bindingCount);
  } else {
    setLayout = DescriptorCache::createStorageLayout(
        device, info.fsSpirv, info.fsBytes / sizeof(u32), &bindingCount);
    for (u32 b = 0; b < bindingCount && b < DescriptorCache::kMaxMixedBindings;
         ++b)
      types[b] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  }
  if (setLayout == VK_NULL_HANDLE) {
    vkDestroyShaderModule(device, vs, nullptr);
    vkDestroyShaderModule(device, fs, nullptr);
    return IRenderBackend::kInvalidPipelineHandle;
  }
  VkDescriptorPool pool =
      mixedLayout ? DescriptorCache::createMixedPool(
                        device, types, bindingCount, kMaxFramesInFlight)
                  : DescriptorCache::createStoragePool(
                        device, setLayout, bindingCount, kMaxFramesInFlight);
  VkDescriptorSet sets[kMaxFramesInFlight] = {};
  const bool setOk = pool != VK_NULL_HANDLE &&
                     DescriptorCache::allocateSets(device, pool, setLayout,
                                                   sets, kMaxFramesInFlight);

  const VkPushConstantRange pushRange{
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      .offset = 0,
      .size = 128,
  };
  const VkPipelineLayoutCreateInfo layoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &setLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pushRange,
  };
  VkPipelineLayout layout = VK_NULL_HANDLE;
  const bool layoutOk =
      KE_VK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &layout));

  const VkPipelineVertexInputStateCreateInfo vertexInput{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
  };
  const VkPipelineInputAssemblyStateCreateInfo inputAssembly{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
  };
  const VkPipelineViewportStateCreateInfo viewportState{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .scissorCount = 1,
  };
  const VkPipelineRasterizationStateCreateInfo rasterization{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .lineWidth = 1.0f,
  };
  const VkPipelineMultisampleStateCreateInfo multisample{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
  };
  const VkPipelineDepthStencilStateCreateInfo depthStencil{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
  };
  VkPipelineColorBlendAttachmentState blendAttachments[2] = {};
  for (u32 a = 0; a < colorCount; ++a) {
    blendAttachments[a].colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    if (premultipliedBlend) {
      blendAttachments[a].blendEnable = VK_TRUE;
      blendAttachments[a].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
      blendAttachments[a].dstColorBlendFactor =
          VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      blendAttachments[a].colorBlendOp = VK_BLEND_OP_ADD;
      blendAttachments[a].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      blendAttachments[a].dstAlphaBlendFactor =
          VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      blendAttachments[a].alphaBlendOp = VK_BLEND_OP_ADD;
    }
  }
  const VkPipelineColorBlendStateCreateInfo colorBlend{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = colorCount,
      .pAttachments = blendAttachments,
  };
  const VkDynamicState dynamicStates[2] = {VK_DYNAMIC_STATE_VIEWPORT,
                                           VK_DYNAMIC_STATE_SCISSOR};
  const VkPipelineDynamicStateCreateInfo dynamicState{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2,
      .pDynamicStates = dynamicStates,
  };
  const VkPipelineRenderingCreateInfoKHR renderingInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
      .colorAttachmentCount = colorCount,
      .pColorAttachmentFormats = colorFormats,
      .depthAttachmentFormat = depthFormat,
  };

  VkPipeline pipeline = VK_NULL_HANDLE;
  bool pipeOk = false;
  if (layoutOk) {
    const VkGraphicsPipelineCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &renderingInfo,
        .stageCount = 2,
        .pStages = stages,
        .pVertexInputState = &vertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depthStencil,
        .pColorBlendState = &colorBlend,
        .pDynamicState = &dynamicState,
        .layout = layout,
    };
    pipeOk = KE_VK(vkCreateGraphicsPipelines(device, m_cache, 1, &createInfo,
                                             nullptr, &pipeline));
  }
  vkDestroyShaderModule(device, vs, nullptr);
  vkDestroyShaderModule(device, fs, nullptr);

  if (!setOk || !layoutOk || !pipeOk) {
    if (pipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(device, pipeline, nullptr);
    if (layout != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(device, layout, nullptr);
    if (pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, pool, nullptr);
    if (setLayout != VK_NULL_HANDLE)
      vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
    KENSHIN_ERROR("[vulkan] fullscreen-variant pipeline creation failed");
    return IRenderBackend::kInvalidPipelineHandle;
  }

  m_fullscreen.emplace_back();
  Fullscreen& entry = m_fullscreen.back();
  entry.layout = layout;
  entry.pipeline = pipeline;
  entry.setLayout = setLayout;
  entry.pool = pool;
  for (u32 i = 0; i < kMaxFramesInFlight; ++i) entry.sets[i] = sets[i];
  entry.valid = true;
  for (u32 b = 0; b < bindingCount && b < DescriptorCache::kMaxMixedBindings;
       ++b)
    entry.bindingTypes[b] = types[b];
  entry.bindingCount = bindingCount;
  return static_cast<u32>(m_fullscreen.size());
}

u32 PipelineStore::createGBuffer(
    VkDevice device, const IRenderBackend::FullscreenPipelineInfo& info) {
  // GBuffer B is sRGB-encoded: fsGBuffer writes LINEAR base color, and
  // an 8-bit UNORM store of linear albedo bands the darks (linear 0.02
  // sits 5 codes above black - the dark camo). The alpha channel (the
  // packed roughness/metallic bits) is linear in an _SRGB format.
  static const VkFormat kFormats[2] = {VK_FORMAT_R16G16B16A16_SFLOAT,
                                       VK_FORMAT_R8G8B8A8_SRGB};
  // M8.2: MIXED layout - fsGBuffer samples the albedo Texture2DArray
  // (binding 17/18: SAMPLED_IMAGE + SAMPLER) alongside its SSBOs.
  return createFullscreenVariant(device, info, kFormats, 2, VK_FORMAT_UNDEFINED,
                                 true);
}

u32 PipelineStore::createLighting(
    VkDevice device, const IRenderBackend::FullscreenPipelineInfo& info) {
  // M8.3: linear HDR into the per-slot RGBA16F intermediate; the
  // format was gate-checked at initialize (the GBuffer A target is the
  // same RGBA16F).
  static const VkFormat kHdr = VK_FORMAT_R16G16B16A16_SFLOAT;
  return createFullscreenVariant(device, info, &kHdr, 1, VK_FORMAT_UNDEFINED,
                                 true);
}

u32 PipelineStore::createTonemap(
    VkDevice device, VkFormat colorFormat,
    const IRenderBackend::FullscreenPipelineInfo& info) {
  return createFullscreenVariant(device, info, &colorFormat, 1,
                                 VK_FORMAT_UNDEFINED, true);
}

u32 PipelineStore::createTaa(
    VkDevice device, const IRenderBackend::FullscreenPipelineInfo& info) {
  // RGBA16F color (the resolve target), mixed sampler+SSBO layout
  // (3 Sampler2D + the matrices SSBO - a push block would need 160B;
  // the guaranteed push limit is 128).
  static const VkFormat kFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
  return createFullscreenVariant(device, info, &kFormat, 1, VK_FORMAT_UNDEFINED,
                                 true);
}

u32 PipelineStore::createTranslucent(
    VkDevice device, const IRenderBackend::FullscreenPipelineInfo& info) {
  static const VkFormat kHdr = VK_FORMAT_R16G16B16A16_SFLOAT;
  return createFullscreenVariant(device, info, &kHdr, 1, VK_FORMAT_UNDEFINED,
                                 true, true);
}

u32 PipelineStore::createShadowDecode(
    VkDevice device, const IRenderBackend::FullscreenPipelineInfo& info) {
  static const VkFormat kFormat = VK_FORMAT_R32_SFLOAT;
  return createFullscreenVariant(device, info, &kFormat, 1, VK_FORMAT_UNDEFINED,
                                 false);
}

u32 PipelineStore::createNullRaster(
    VkDevice device, const IRenderBackend::FullscreenPipelineInfo& info) {
  if (info.vsSpirv == nullptr || info.vsBytes == 0 || info.fsSpirv == nullptr ||
      info.fsBytes == 0)
    return IRenderBackend::kInvalidPipelineHandle;

  const VkShaderModuleCreateInfo vsInfo{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = info.vsBytes,
      .pCode = info.vsSpirv,
  };
  VkShaderModule vs = VK_NULL_HANDLE;
  if (!KE_VK(vkCreateShaderModule(device, &vsInfo, nullptr, &vs)))
    return IRenderBackend::kInvalidPipelineHandle;
  const VkShaderModuleCreateInfo fsInfo{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = info.fsBytes,
      .pCode = info.fsSpirv,
  };
  VkShaderModule fs = VK_NULL_HANDLE;
  if (!KE_VK(vkCreateShaderModule(device, &fsInfo, nullptr, &fs))) {
    vkDestroyShaderModule(device, vs, nullptr);
    return IRenderBackend::kInvalidPipelineHandle;
  }

  const VkPipelineShaderStageCreateInfo stages[2] = {
      VkPipelineShaderStageCreateInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT,
          .module = vs,
          .pName = "main",
      },
      VkPipelineShaderStageCreateInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
          .module = fs,
          .pName = "main",
      },
  };

  // M6.3: the layout must cover the UNION of the two stages' bindings -
  // Slang strips unused file-scope SSBOs from each entry point's SPIR-V,
  // so a VS-only read (raster_hw's paged tables) never appears in the
  // FS reflection. The layout shape is dense all-SSBO regardless of
  // which module it reflects from: keep whichever stage reflects more.
  u32 bindingCount = 0;
  VkDescriptorSetLayout setLayout = DescriptorCache::createStorageLayout(
      device, info.fsSpirv, info.fsBytes / sizeof(u32), &bindingCount);
  if (setLayout != VK_NULL_HANDLE) {
    u32 vsBindingCount = 0;
    const VkDescriptorSetLayout vsLayout = DescriptorCache::createStorageLayout(
        device, info.vsSpirv, info.vsBytes / sizeof(u32), &vsBindingCount);
    if (vsLayout != VK_NULL_HANDLE) {
      if (vsBindingCount > bindingCount) {
        vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
        setLayout = vsLayout;
        bindingCount = vsBindingCount;
      } else {
        vkDestroyDescriptorSetLayout(device, vsLayout, nullptr);
      }
    }
  }
  if (setLayout == VK_NULL_HANDLE) {
    vkDestroyShaderModule(device, vs, nullptr);
    vkDestroyShaderModule(device, fs, nullptr);
    return IRenderBackend::kInvalidPipelineHandle;
  }
  VkDescriptorPool pool = DescriptorCache::createStoragePool(
      device, setLayout, bindingCount, kMaxFramesInFlight);
  VkDescriptorSet sets[kMaxFramesInFlight] = {};
  const bool setOk = pool != VK_NULL_HANDLE &&
                     DescriptorCache::allocateSets(device, pool, setLayout,
                                                   sets, kMaxFramesInFlight);

  const VkPipelineLayoutCreateInfo layoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &setLayout,
  };
  VkPipelineLayout layout = VK_NULL_HANDLE;
  const bool layoutOk =
      KE_VK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &layout));

  const VkPipelineVertexInputStateCreateInfo vertexInput{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
  };
  const VkPipelineInputAssemblyStateCreateInfo inputAssembly{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
  };
  const VkPipelineViewportStateCreateInfo viewportState{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .scissorCount = 1,
  };
  // Cull BACK / front CCW: the same half the SW rasterizer drops under
  // the Y-flipped projection (the baseline pipeline's convention).
  // Conservative OVERESTIMATE (P0 seam fix): the fixed-function
  // rasterizer generates fragments from the UNSNAPPED triangle while
  // the PS's coverage test judges the 1/256-snapped one - pixels the
  // snapped triangle owns but the unsnapped one misses by <1/256px got
  // NO fragment at all (single-pixel HW/SW seam holes). Overestimate
  // rasterization widens fragment generation; the PS test still
  // rejects everything outside the snapped triangle, so coverage
  // lands exactly on the SW-agreed quantized footprint.
  const VkPipelineRasterizationConservativeStateCreateInfoEXT conserv{
      .sType =
          VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT,
      .conservativeRasterizationMode =
          VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT,
  };
  const VkPipelineRasterizationStateCreateInfo rasterization{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .pNext = info.conservativeRaster ? &conserv : nullptr,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_BACK_BIT,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = 1.0f,
  };
  const VkPipelineMultisampleStateCreateInfo multisample{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
  };
  const VkPipelineDepthStencilStateCreateInfo depthStencil{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
  };
  const VkPipelineColorBlendStateCreateInfo colorBlend{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
  };
  const VkDynamicState dynamicStates[2] = {VK_DYNAMIC_STATE_VIEWPORT,
                                           VK_DYNAMIC_STATE_SCISSOR};
  const VkPipelineDynamicStateCreateInfo dynamicState{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2,
      .pDynamicStates = dynamicStates,
  };
  // ZERO attachments: the only output is the VisBuffer SSBO atomic in
  // the PS (UE's null-targets form - one output buffer, no OM state,
  // async-compute-overlap friendly).
  const VkPipelineRenderingCreateInfoKHR renderingInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
  };

  VkPipeline pipeline = VK_NULL_HANDLE;
  bool pipeOk = false;
  if (layoutOk) {
    const VkGraphicsPipelineCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &renderingInfo,
        .stageCount = 2,
        .pStages = stages,
        .pVertexInputState = &vertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depthStencil,
        .pColorBlendState = &colorBlend,
        .pDynamicState = &dynamicState,
        .layout = layout,
    };
    pipeOk = KE_VK(vkCreateGraphicsPipelines(device, m_cache, 1, &createInfo,
                                             nullptr, &pipeline));
  }
  vkDestroyShaderModule(device, vs, nullptr);
  vkDestroyShaderModule(device, fs, nullptr);

  if (!setOk || !layoutOk || !pipeOk) {
    if (pipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(device, pipeline, nullptr);
    if (layout != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(device, layout, nullptr);
    if (pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, pool, nullptr);
    if (setLayout != VK_NULL_HANDLE)
      vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
    KENSHIN_ERROR("[vulkan] null-raster pipeline creation failed");
    return IRenderBackend::kInvalidPipelineHandle;
  }

  m_nullRaster.emplace_back();
  Fullscreen& entry = m_nullRaster.back();
  entry.layout = layout;
  entry.pipeline = pipeline;
  entry.setLayout = setLayout;
  entry.pool = pool;
  for (u32 i = 0; i < kMaxFramesInFlight; ++i) entry.sets[i] = sets[i];
  entry.valid = true;
  return static_cast<u32>(m_nullRaster.size());
}

u32 PipelineStore::createCompute(
    VkDevice device, const IRenderBackend::ComputePipelineInfo& info) {
  if (info.csSpirv == nullptr || info.csBytes == 0)
    return IRenderBackend::kInvalidPipelineHandle;

  const VkShaderModuleCreateInfo csInfo{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = info.csBytes,
      .pCode = info.csSpirv,
  };
  VkShaderModule module = VK_NULL_HANDLE;
  if (!KE_VK(vkCreateShaderModule(device, &csInfo, nullptr, &module)))
    return IRenderBackend::kInvalidPipelineHandle;

  // SSBO set layout from reflection; one set per frame slot, re-written
  // by that slot's dispatch.
  u32 bindingCount = 0;
  const VkDescriptorSetLayout setLayout = DescriptorCache::createStorageLayout(
      device, info.csSpirv, info.csBytes / sizeof(u32), &bindingCount);
  if (setLayout == VK_NULL_HANDLE) {
    vkDestroyShaderModule(device, module, nullptr);
    return IRenderBackend::kInvalidPipelineHandle;
  }
  VkDescriptorPool pool = DescriptorCache::createStoragePool(
      device, setLayout, bindingCount, kMaxFramesInFlight);
  VkDescriptorSet sets[kMaxFramesInFlight] = {};
  const bool setOk = pool != VK_NULL_HANDLE &&
                     DescriptorCache::allocateSets(device, pool, setLayout,
                                                   sets, kMaxFramesInFlight);

  const VkPipelineLayoutCreateInfo layoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &setLayout,
  };
  VkPipelineLayout layout = VK_NULL_HANDLE;
  const bool layoutOk =
      KE_VK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &layout));

  VkPipeline pipeline = VK_NULL_HANDLE;
  bool pipeOk = false;
  if (layoutOk) {
    const VkComputePipelineCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage =
            VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = module,
                .pName = "main",
            },
        .layout = layout,
    };
    pipeOk = KE_VK(vkCreateComputePipelines(device, m_cache, 1, &createInfo,
                                            nullptr, &pipeline));
  }
  vkDestroyShaderModule(device, module, nullptr);

  if (!setOk || !layoutOk || !pipeOk) {
    if (pipeline != VK_NULL_HANDLE)
      vkDestroyPipeline(device, pipeline, nullptr);
    if (layout != VK_NULL_HANDLE)
      vkDestroyPipelineLayout(device, layout, nullptr);
    if (pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, pool, nullptr);
    if (setLayout != VK_NULL_HANDLE)
      vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
    KENSHIN_ERROR("[vulkan] compute pipeline creation failed");
    return IRenderBackend::kInvalidPipelineHandle;
  }

  m_compute.emplace_back();
  Compute& entry = m_compute.back();
  entry.layout = layout;
  entry.pipeline = pipeline;
  entry.setLayout = setLayout;
  entry.pool = pool;
  for (u32 i = 0; i < kMaxFramesInFlight; ++i) entry.sets[i] = sets[i];
  entry.valid = true;
  return static_cast<u32>(m_compute.size());
}

PipelineStore::Pipeline PipelineStore::detach(u32 handle) {
  if (handle == 0 || handle > m_slots.size()) return {};
  Pipeline raw = m_slots[handle - 1];
  m_slots[handle - 1] = {};
  return raw;
}

bool PipelineStore::valid(u32 handle) const {
  return handle != 0 && handle <= m_slots.size() &&
         m_slots[handle - 1].pipeline != VK_NULL_HANDLE;
}

VkPipeline PipelineStore::pipeline(u32 handle) const {
  return m_slots[handle - 1].pipeline;
}

VkPipelineLayout PipelineStore::layout(u32 handle) const {
  return m_slots[handle - 1].layout;
}

u32 PipelineStore::create(VkDevice device, VkFormat colorFormat,
                          const IRenderBackend::GraphicsPipelineInfo& info) {
  if (info.vsSpirv == nullptr || info.vsBytes == 0 || info.fsSpirv == nullptr ||
      info.fsBytes == 0)
    return IRenderBackend::kInvalidPipelineHandle;

  const VkPushConstantRange pushRange{
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      .offset = 0,
      .size =
          112,  // mat4 viewProj (64) + kd (16) + lightDir (16) + camPos (16)
  };
  const VkPipelineLayoutCreateInfo layoutInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pushRange,
  };
  VkPipelineLayout layout = VK_NULL_HANDLE;
  KE_VK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &layout));

  const VkShaderModuleCreateInfo vsInfo{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = info.vsBytes,
      .pCode = info.vsSpirv,
  };
  VkShaderModule vs = VK_NULL_HANDLE;
  if (!KE_VK(vkCreateShaderModule(device, &vsInfo, nullptr, &vs))) {
    vkDestroyPipelineLayout(device, layout, nullptr);
    return IRenderBackend::kInvalidPipelineHandle;
  }
  const VkShaderModuleCreateInfo fsInfo{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = info.fsBytes,
      .pCode = info.fsSpirv,
  };
  VkShaderModule fs = VK_NULL_HANDLE;
  if (!KE_VK(vkCreateShaderModule(device, &fsInfo, nullptr, &fs))) {
    vkDestroyShaderModule(device, vs, nullptr);
    vkDestroyPipelineLayout(device, layout, nullptr);
    return IRenderBackend::kInvalidPipelineHandle;
  }

  const VkPipelineShaderStageCreateInfo stages[2] = {
      VkPipelineShaderStageCreateInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT,
          .module = vs,
          .pName =
              "main",  // Slang emits OpEntryPoint "main" regardless of function name
      },
      VkPipelineShaderStageCreateInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
          .module = fs,
          .pName = "main",
      },
  };

  // SOA vertex input: three bindings (positions / normals / uvs).
  const VkVertexInputBindingDescription bindings[3] = {
      {0, 12, VK_VERTEX_INPUT_RATE_VERTEX},
      {1, 12, VK_VERTEX_INPUT_RATE_VERTEX},
      {2, 8, VK_VERTEX_INPUT_RATE_VERTEX},
  };
  const VkVertexInputAttributeDescription attributes[3] = {
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
      {1, 1, VK_FORMAT_R32G32B32_SFLOAT, 0},
      {2, 2, VK_FORMAT_R32G32_SFLOAT, 0},
  };
  const VkPipelineRenderingCreateInfoKHR renderingInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &colorFormat,
      .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT,
  };
  const VkDynamicState dynamicStates[2] = {VK_DYNAMIC_STATE_VIEWPORT,
                                           VK_DYNAMIC_STATE_SCISSOR};
  const VkPipelineDynamicStateCreateInfo dynamicState{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2,
      .pDynamicStates = dynamicStates,
  };

  // State structs need named locals (designated initializers of
  // temporaries can't be address-taken).
  const VkPipelineVertexInputStateCreateInfo vertexInput{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 3,
      .pVertexBindingDescriptions = bindings,
      .vertexAttributeDescriptionCount = 3,
      .pVertexAttributeDescriptions = attributes,
  };
  const VkPipelineInputAssemblyStateCreateInfo inputAssembly{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
  };
  const VkPipelineViewportStateCreateInfo viewportState{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .scissorCount = 1,
  };
  const VkPipelineRasterizationStateCreateInfo rasterization{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = toVk(info.cullMode),
      .frontFace = toVk(info.frontFace),
      .lineWidth = 1.0f,
  };
  const VkPipelineMultisampleStateCreateInfo multisample{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
  };
  const VkPipelineDepthStencilStateCreateInfo depthStencil{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = info.depthTest ? VK_TRUE : VK_FALSE,
      .depthWriteEnable = info.depthWrite ? VK_TRUE : VK_FALSE,
      .depthCompareOp = toVk(info.depthCompare),
      .minDepthBounds = 0.0f,
      .maxDepthBounds = 1.0f,
  };
  const VkPipelineColorBlendAttachmentState blendAttachment{
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
  };
  const VkPipelineColorBlendStateCreateInfo colorBlend{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &blendAttachment,
  };

  const VkGraphicsPipelineCreateInfo createInfo{
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &renderingInfo,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vertexInput,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState = &viewportState,
      .pRasterizationState = &rasterization,
      .pMultisampleState = &multisample,
      .pDepthStencilState = &depthStencil,
      .pColorBlendState = &colorBlend,
      .pDynamicState = &dynamicState,
      .layout = layout,
  };
  VkPipeline pipeline = VK_NULL_HANDLE;
  if (!KE_VK(vkCreateGraphicsPipelines(device, m_cache, 1, &createInfo, nullptr,
                                       &pipeline))) {
    KENSHIN_ERROR("[vulkan] graphics pipeline creation failed");
    vkDestroyShaderModule(device, vs, nullptr);
    vkDestroyShaderModule(device, fs, nullptr);
    vkDestroyPipelineLayout(device, layout, nullptr);
    return IRenderBackend::kInvalidPipelineHandle;
  }
  vkDestroyShaderModule(device, vs, nullptr);
  vkDestroyShaderModule(device, fs, nullptr);

  // Reuse a freed slot when one exists; handle = index + 1.
  u32 handle = 0;
  for (usize i = 0; i < m_slots.size(); ++i) {
    if (m_slots[i].pipeline == VK_NULL_HANDLE) {
      m_slots[i] = {layout, pipeline};
      handle = static_cast<u32>(i + 1);
      break;
    }
  }
  if (handle == 0) {
    m_slots.push_back({layout, pipeline});
    handle = static_cast<u32>(m_slots.size());
  }
  char name[32];
  snprintf(name, sizeof(name), "graphics-pipeline-%u", handle);
  vkutil::setName(device, VK_OBJECT_TYPE_PIPELINE, (uint64_t)pipeline, name);
  return handle;
}

}  // namespace backend

KENSHIN_NS_END
