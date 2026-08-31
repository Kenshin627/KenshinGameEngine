import re

# ---- VulkanBackend.cpp: baseline mesh implementation ----
p = 'engine/src/backend/VulkanBackend.cpp'
s = open(p, encoding='utf-8').read()

# Impl members
old = """        // Full-frame capture (UI included; dev/visual checks only)
        bool captureFull = false;
        bool fullCaptured = false;
        VkBuffer captureStaging = VK_NULL_HANDLE;
        VmaAllocation captureAlloc = VK_NULL_HANDLE;
        u32 captureWidth = 0, captureHeight = 0;
    };"""
new = """        // Full-frame capture (UI included; dev/visual checks only)
        bool captureFull = false;
        bool fullCaptured = false;
        VkBuffer captureStaging = VK_NULL_HANDLE;
        VmaAllocation captureAlloc = VK_NULL_HANDLE;
        u32 captureWidth = 0, captureHeight = 0;

        // Baseline mesh path (S2.2)
        struct MeshResource
        {
            VkBuffer positions = VK_NULL_HANDLE;
            VmaAllocation positionsAlloc = VK_NULL_HANDLE;
            VkBuffer normals = VK_NULL_HANDLE;
            VmaAllocation normalsAlloc = VK_NULL_HANDLE;
            VkBuffer uvs = VK_NULL_HANDLE;
            VmaAllocation uvsAlloc = VK_NULL_HANDLE;
            VkBuffer indices = VK_NULL_HANDLE;
            VmaAllocation indicesAlloc = VK_NULL_HANDLE;
            u32 vertexCount = 0;
            u32 indexCount = 0;
        };
        std::vector<MeshResource> meshes; // slot = handle - 1
        VkPipelineLayout baselineLayout = VK_NULL_HANDLE;
        VkPipeline baselinePipeline = VK_NULL_HANDLE;
        std::vector<u32> baselineVs, baselineFs; // SPIR-V staged by init call
    };"""
assert old in s
s = s.replace(old, new)

# beginFrame: after clear, transition to COLOR_ATTACHMENT (renders happen
# between begin/end; SHADER_READ_ONLY moves to endFrame)
old = """        // offscreen: TRANSFER_DST -> SHADER_READ_ONLY (sampled by the ImGui
        // viewport panel later in this same command buffer).
        recordBarrier(slot.commandBuffer,
                      imageBarrier(slot.offscreen.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_READ_BIT));"""
new = """        // offscreen: TRANSFER_DST -> COLOR_ATTACHMENT (passes between
        // begin/end render here; endFrame moves it to SHADER_READ_ONLY).
        recordBarrier(slot.commandBuffer,
                      imageBarrier(slot.offscreen.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                   VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT));
        ensureViewportDepth(info.width, info.height);
        // Depth image: UNDEFINED -> DEPTH_ATTACHMENT (cleared by first draw's
        // rendering begin; keep a matching record each frame).
        recordBarrier(slot.commandBuffer,
                      VkImageMemoryBarrier2 {
                          .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                          .srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                          .srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                          .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                          .dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                           VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                          .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                          .newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                          .image = m_impl->depthImage,
                          .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
                      });"""
assert old in s
s = s.replace(old, new)

# endFrame: transition to SHADER_READ_ONLY before ImGui (if any mesh drew)
old = """        ImDrawData* drawData = nullptr;
        if (impl.imguiInited)
        {
            ImGui::Render();
            drawData = ImGui::GetDrawData();
        }"""
new = """        // offscreen: COLOR_ATTACHMENT -> SHADER_READ_ONLY for the ImGui sample.
        recordBarrier(slot.commandBuffer,
                      imageBarrier(slot.offscreen.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                   VK_ACCESS_2_SHADER_READ_BIT));

        ImDrawData* drawData = nullptr;
        if (impl.imguiInited)
        {
            ImGui::Render();
            drawData = ImGui::GetDrawData();
        }"""
assert old in s
s = s.replace(old, new)

# shutdown: destroy meshes + pipeline + depth
old = """        m_impl->frameSync.shutdown(m_impl->device);"""
new = """        for (auto& mesh : m_impl->meshes)
            destroyMeshBuffers(mesh);
        m_impl->meshes.clear();
        if (m_impl->baselinePipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(m_impl->device.device, m_impl->baselinePipeline, nullptr);
        if (m_impl->baselineLayout != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(m_impl->device.device, m_impl->baselineLayout, nullptr);
        if (m_impl->depthImage != VK_NULL_HANDLE)
        {
            vkDestroyImageView(m_impl->device.device, m_impl->depthView, nullptr);
            vmaDestroyImage(m_impl->device.allocator, m_impl->depthImage,
                            m_impl->depthAlloc);
        }
        m_impl->frameSync.shutdown(m_impl->device);"""
assert old in s
s = s.replace(old, new)

# Impl depth members (append to Impl before meshes)
old = """        VkPipelineLayout baselineLayout = VK_NULL_HANDLE;"""
new = """        VkImage depthImage = VK_NULL_HANDLE;
        VmaAllocation depthAlloc = VK_NULL_HANDLE;
        VkImageView depthView = VK_NULL_HANDLE;
        u32 depthWidth = 0, depthHeight = 0;
        bool meshDrewThisFrame = false;
        VkPipelineLayout baselineLayout = VK_NULL_HANDLE;"""
assert old in s
s = s.replace(old, new)

open(p, 'w', encoding='utf-8', newline='\n').write(s)
print("part1 ok")
