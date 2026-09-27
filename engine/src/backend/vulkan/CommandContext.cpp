#include "CommandContext.h"

#include "Vk.h"

KENSHIN_NS_BEGIN

namespace backend {

bool CommandContext::begin(VkCommandBuffer cmd) {
  m_cmd = cmd;
  m_passOpen = false;
  if (!KE_VK(vkResetCommandBuffer(m_cmd, 0))) return false;
  const VkCommandBufferBeginInfo beginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  return KE_VK(vkBeginCommandBuffer(m_cmd, &beginInfo));
}

bool CommandContext::end() {
  endPass();
  return KE_VK(vkEndCommandBuffer(m_cmd));
}

void CommandContext::beginViewportPass(VkImageView color, VkImageView depth,
                                       u32 width, u32 height) {
  const VkRenderingAttachmentInfo colorAttachment{
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = color,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,  // cleared in beginFrame
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
  };
  const VkClearValue depthClear{.depthStencil = {1.0f, 0}};
  const VkRenderingAttachmentInfo depthAttachment{
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = depth,
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,  // once per frame, on open
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue = depthClear,
  };
  const VkRenderingInfo renderingInfo{
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {0, 0, width, height},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &colorAttachment,
      .pDepthAttachment = &depthAttachment,
  };
  vkCmdBeginRendering(m_cmd, &renderingInfo);

  const VkViewport viewport{
      .x = 0.0f,
      .y = 0.0f,
      .width = static_cast<f32>(width),
      .height = static_cast<f32>(height),
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
  };
  const VkRect2D scissor{{0, 0}, {width, height}};
  vkCmdSetViewport(m_cmd, 0, 1, &viewport);
  vkCmdSetScissor(m_cmd, 0, 1, &scissor);
  m_passOpen = true;
}

void CommandContext::beginNullPass(u32 width, u32 height) {
  const VkRenderingInfo renderingInfo{
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {0, 0, width, height},
      .layerCount = 1,
  };
  vkCmdBeginRendering(m_cmd, &renderingInfo);

  const VkViewport viewport{
      .x = 0.0f,
      .y = 0.0f,
      .width = static_cast<f32>(width),
      .height = static_cast<f32>(height),
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
  };
  const VkRect2D scissor{{0, 0}, {width, height}};
  vkCmdSetViewport(m_cmd, 0, 1, &viewport);
  vkCmdSetScissor(m_cmd, 0, 1, &scissor);
  m_passOpen = true;
}

void CommandContext::beginColorPass(VkImageView color, u32 width, u32 height) {
  beginColorPass(color, width, height, VK_ATTACHMENT_LOAD_OP_CLEAR,
                 VkRect2D{{0, 0}, {width, height}}, 0.0f);
}

void CommandContext::beginColorPass(VkImageView color, u32 width, u32 height,
                                    VkAttachmentLoadOp loadOp,
                                    const VkRect2D& scissorRect, f32 clearValue) {
  const VkClearValue zero{.color = {{clearValue, clearValue, clearValue, clearValue}}};
  const VkRenderingAttachmentInfo attachment{
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = color,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = loadOp,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue = zero,
  };
  const VkRenderingInfo renderingInfo{
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = scissorRect,
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &attachment,
  };
  vkCmdBeginRendering(m_cmd, &renderingInfo);

  const VkViewport viewport{
      .x = 0.0f,
      .y = 0.0f,
      .width = static_cast<f32>(width),
      .height = static_cast<f32>(height),
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
  };
  vkCmdSetViewport(m_cmd, 0, 1, &viewport);
  vkCmdSetScissor(m_cmd, 0, 1, &scissorRect);
  m_passOpen = true;
}

void CommandContext::beginGBufferPass(VkImageView colorA, VkImageView colorB,
                                      u32 width, u32 height) {
  const VkClearValue zero{.color = {{0.0f, 0.0f, 0.0f, 0.0f}}};
  const VkRenderingAttachmentInfo attachments[2] = {
      VkRenderingAttachmentInfo{
          .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
          .imageView = colorA,
          .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
          .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
          .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
          .clearValue = zero,
      },
      VkRenderingAttachmentInfo{
          .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
          .imageView = colorB,
          .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
          .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
          .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
          .clearValue = zero,
      },
  };
  const VkRenderingInfo renderingInfo{
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {0, 0, width, height},
      .layerCount = 1,
      .colorAttachmentCount = 2,
      .pColorAttachments = attachments,
  };
  vkCmdBeginRendering(m_cmd, &renderingInfo);

  const VkViewport viewport{
      .x = 0.0f,
      .y = 0.0f,
      .width = static_cast<f32>(width),
      .height = static_cast<f32>(height),
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
  };
  const VkRect2D scissor{{0, 0}, {width, height}};
  vkCmdSetViewport(m_cmd, 0, 1, &viewport);
  vkCmdSetScissor(m_cmd, 0, 1, &scissor);
  m_passOpen = true;
}

void CommandContext::endPass() {
  if (m_passOpen) {
    vkCmdEndRendering(m_cmd);
    m_passOpen = false;
  }
}

void CommandContext::pipelineBarrier(VkCommandBuffer cmd,
                                     const VkImageMemoryBarrier2& barrier) {
  const VkDependencyInfo dep{
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &barrier,
  };
  vkCmdPipelineBarrier2(cmd, &dep);
}

VkImageMemoryBarrier2 CommandContext::imageBarrier(
    VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
    VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess, u32 baseMip,
    u32 mipCount, u32 baseLayer, u32 layerCount) {
  return VkImageMemoryBarrier2{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
      .dstStageMask = dstStage,
      .dstAccessMask = dstAccess,
      .oldLayout = oldLayout,
      .newLayout = newLayout,
      .image = image,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, baseMip, mipCount,
                           baseLayer, layerCount},
  };
}

void CommandContext::clearColorImage(VkCommandBuffer cmd, VkImage image,
                                     VkImageLayout layout, const f32 color[4]) {
  VkClearColorValue clear{};
  clear.float32[0] = color[0];
  clear.float32[1] = color[1];
  clear.float32[2] = color[2];
  clear.float32[3] = color[3];
  const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  vkCmdClearColorImage(cmd, image, layout, &clear, 1, &range);
}

}  // namespace backend

KENSHIN_NS_END
