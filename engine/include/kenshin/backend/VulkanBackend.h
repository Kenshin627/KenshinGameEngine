#pragma once

// ── Vulkan backend (public face, no Vulkan types exposed) ───────

#include "kenshin/backend/RenderBackend.h"

KENSHIN_NS_BEGIN

namespace backend {

class VulkanBackend final : public IRenderBackend {
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
  bool readbackFrame(std::vector<u8>& rgbaOut, u32& width,
                     u32& height) override;
  bool runComputeOneshot(const Job& job,
                         std::vector<u8> outputs[Job::kMaxBuffers]) override;
  bool initImGuiBackend(void* hwnd, u32 width, u32 height) override;
  void* viewportImGuiTexture() override;
  void requestFullFrameCapture() override;
  bool readbackFullFrame(std::vector<u8>& rgbaOut, u32& width,
                         u32& height) override;
  u32 createMeshBuffer(const MeshUpload& upload) override;
  void destroyMeshBuffer(u32 mesh) override;
  u32 createGraphicsPipeline(const GraphicsPipelineInfo& info) override;
  void destroyGraphicsPipeline(u32 pipeline) override;
  void draw(u32 pipeline, u32 mesh, const DrawData& data) override;
  GpuTimings gpuTimings() const override { return m_gpuTimings; }
  u32 createBuffer(const BufferDesc& desc) override;
  void destroyBuffer(u32 buffer) override;
  bool readbackBuffer(u32 buffer, usize offset, usize bytes,
                      std::vector<u8>& out) override;
  u32 createFullscreenPipeline(const FullscreenPipelineInfo& info) override;
  u32 createNullPipeline(const FullscreenPipelineInfo& info) override;
  void destroyNullPipeline(u32 pipeline) override;
  void drawNullIndirect(u32 pipeline, const u32* bufferHandles, u32 bufferCount,
                        u32 argsBuffer, usize argsOffset, u32 width,
                        u32 height) override;
  void barrierIndirectArgs(u32 buffer) override;
  void barrierGraphicsRw(u32 buffer) override;
  void barrierGraphicsToCompute(u32 buffer) override;
  void drawFullscreen(u32 pipeline, const u32* bufferHandles, u32 bufferCount,
                      const void* push, usize pushBytes) override;
  u32 createComputePipeline(const ComputePipelineInfo& info) override;
  void destroyComputePipeline(u32 pipeline) override;
  void dispatchCompute(u32 pipeline, const u32* bufferHandles, u32 bufferCount,
                       u32 groupsX, u32 groupsY, u32 groupsZ) override;
  void bufferBarrier(u32 buffer, bool toFragment) override;
  void fillBuffer(u32 buffer, u32 value) override;
  void dispatchComputeIndirect(u32 pipeline, const u32* bufferHandles,
                               u32 bufferCount, u32 countBuffer,
                               usize countOffset) override;
  void* mappedPtr(u32 buffer) override;
  usize bufferSize(u32 buffer) const override;
  u32 currentFrameSlot() const override;
  u32 lastCompletedFrameSlot() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
  BackendCaps m_caps;
  bool m_shutdownDone = false;
  GpuTimings m_gpuTimings;

  /// (Re)create ImGui descriptor sets when offscreen targets change.
  void ensureViewportTextures();
  /// (Re)create the full-frame capture staging buffer on size change.
  void ensureCaptureStaging(u32 width, u32 height);
  void ensureViewportDepth(u32 width, u32 height);
};

}  // namespace backend

KENSHIN_NS_END
