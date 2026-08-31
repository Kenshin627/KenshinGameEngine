#pragma once

// ── Vulkan backend (public face, no Vulkan types exposed) ───────

#include "kenshin/backend/RenderBackend.h"

KENSHIN_NS_BEGIN

namespace backend
{

    class VulkanBackend final : public IRenderBackend
    {
    public:
        VulkanBackend();
        ~VulkanBackend() override;

        VulkanBackend(const VulkanBackend&) = delete;
        VulkanBackend& operator=(const VulkanBackend&) = delete;

        bool initialize(const BackendInitParams& params) override;
        void shutdown() override;
        const BackendCaps& caps() const override { return m_caps; }

        bool beginFrame(const FrameBeginInfo& info) override;
        void endFrame() override;
        bool readbackFrame(std::vector<u8>& rgbaOut, u32& width, u32& height) override;
        bool runComputeOneshot(const Job& job,
                               std::vector<u8> outputs[Job::kMaxBuffers]) override;
        bool initImGuiBackend(void* hwnd, u32 width, u32 height) override;
        void* viewportImGuiTexture() override;
        void requestFullFrameCapture() override;
        bool readbackFullFrame(std::vector<u8>& rgbaOut, u32& width, u32& height) override;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
        BackendCaps m_caps;

        /// (Re)create ImGui descriptor sets when offscreen targets change.
        void ensureViewportTextures();
        /// (Re)create the full-frame capture staging buffer on size change.
        void ensureCaptureStaging(u32 width, u32 height);
    };

} // namespace backend

KENSHIN_NS_END
