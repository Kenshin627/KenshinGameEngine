#include "CommandContext.h"

#include "Vk.h"

KENSHIN_NS_BEGIN

namespace backend {

bool CommandContext::begin(VkCommandBuffer cmd) {
  m_cmd = cmd;
  if (!KE_VK(vkResetCommandBuffer(m_cmd, 0))) return false;
  const VkCommandBufferBeginInfo beginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  return KE_VK(vkBeginCommandBuffer(m_cmd, &beginInfo));
}

bool CommandContext::end() {
  // m_cmd stays valid until the next begin(): the submit path still reads
  // it after recording ends.
  return KE_VK(vkEndCommandBuffer(m_cmd));
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
    VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
  return VkImageMemoryBarrier2{
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
      .dstStageMask = dstStage,
      .dstAccessMask = dstAccess,
      .oldLayout = oldLayout,
      .newLayout = newLayout,
      .image = image,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
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
