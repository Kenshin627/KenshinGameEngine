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

  FrameStatus beginFrame(const FrameBeginInfo& info) override;
  void endFrame() override;
  bool readbackFrame(std::vector<u8>& rgbaOut, u32& width,
                     u32& height) override;
  bool runComputeOneshot(const Job& job,
                         std::vector<u8> outputs[Job::kMaxBuffers]) override;
  bool initImGuiBackend(u32 width, u32 height,
                        std::function<void()> platformNewFrame) override;
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
  u32 snapshotHostBuffer(u32 buffer) override;
  void destroyBuffer(u32 buffer) override;
  bool readbackBuffer(u32 buffer, usize offset, usize bytes,
                      std::vector<u8>& out) override;
  u32 createFullscreenPipeline(const FullscreenPipelineInfo& info) override;
  u32 createTexture(const TextureDesc& desc) override;
  u32 bindlessSlot(u32 texture) const override;
  u32 bindlessUsed() const override;
  u32 createTextureArray(const TextureArrayDesc& desc) override;
  void destroyTexture(u32 texture) override;
  u32 createGBufferPipeline(const FullscreenPipelineInfo& info) override;
  void drawGBuffer(u32 pipeline, const u32* bufferHandles, u32 bufferCount,
                   const void* push, usize pushBytes, u32 width,
                   u32 height) override;
  u32 createLightingPipeline(const FullscreenPipelineInfo& info) override;
  void drawLighting(u32 pipeline, const u32* bufferHandles, u32 bufferCount,
                    const void* push, usize pushBytes, u32 width, u32 height,
                    u32 envTexture) override;
  u32 createTonemapPipeline(const FullscreenPipelineInfo& info) override;
  void drawTonemap(u32 pipeline, const void* push, usize pushBytes, u32 width,
                   u32 height, bool fromTaa = false,
                   u32 exposureState = kInvalidBufferHandle,
                   u32 envTexture = kInvalidTextureHandle) override;
  u32 createExposurePipeline(const ComputePipelineInfo& info) override;
  bool drawExposure(u32 histogramPipeline, u32 adaptPipeline, u32 histogram,
                    u32 state, u32 envTexture, bool fromTaa, u32 width,
                    u32 height, const void* push, usize pushBytes) override;
  u32 createTaaPipeline(const FullscreenPipelineInfo& info) override;
  u32 createBloomPipeline(const FullscreenPipelineInfo& info,
                          bool additive) override;
  bool drawBloom(const u32* downPipelines, const u32* upPipelines, u32 levels,
                 bool fromTaa, u32 width, u32 height) override;
  u32 createTranslucentPipeline(const FullscreenPipelineInfo& info) override;
  void drawTranslucent(u32 pipeline, const u32* bufferHandles, u32 bufferCount,
                       u32 envTexture, const void* push,
                       usize pushBytes, u32 vertexCount, u32 instanceCount,
                       u32 width, u32 height) override;
  void drawTaa(u32 pipeline, const u32* bufferHandles, u32 bufferCount,
               const void* push, usize pushBytes, u32 width,
               u32 height) override;
  u32 createShadowDecodePipeline(const FullscreenPipelineInfo& info) override;
  void drawShadowDecode(u32 pipeline, const u32* bufferHandles, u32 bufferCount,
                        const void* push, usize pushBytes, u32 width,
                        u32 height, u32 level, bool shadowEnabled,
                        const u32 scissorRect[4] = nullptr) override;
  u32 createNullPipeline(const FullscreenPipelineInfo& info) override;
  void destroyNullPipeline(u32 pipeline) override;
  void drawNullIndirect(u32 pipeline, const u32* bufferHandles, u32 bufferCount,
                        u32 argsBuffer, usize argsOffset, u32 width,
                        u32 height) override;
  void barrierIndirectArgs(u32 buffer) override;
  void barrierGraphicsRw(u32 buffer) override;
  void barrierGraphicsToCompute(u32 buffer) override;
  bool beginChain() override;
  void submitChain() override;
  void chainWaitGraphics() override;
  void writeChainTimestamp(u32 index) override;
  bool chainOnComputeQueue() const override;
  void setAsyncChainEnabled(bool enabled) override;
  u32 createPageSlotPool(u32 slotCount, u32 slotBytes) override;
  bool beginPageUploads(u32 slotPool) override;
  bool copyPageToSlot(const void* hostData, u32 slot, u32 bytes) override;
  u64 submitPageUploads() override;
  bool waitPageUploads(u64 value) override;
  void setTransferQueueEnabled(bool enabled) override;
  bool pageUploadsOnTransferQueue() const override;
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

  /// M6.2: transfer-queue page-upload resources (pools, timeline,
  /// staging arena). Eagerly created - cheap and always testable.
  bool initPageStreaming();
  /// Register an uploaded image under a texture handle (bindless slot
  /// or kNoTexture).
  u32 storeTexture(struct GpuImage&& image, u32 slot);

  /// (Re)create ImGui descriptor sets when offscreen targets change.
  void ensureViewportTextures();
  /// (Re)create the full-frame capture staging buffer on size change.
  void ensureCaptureStaging(u32 width, u32 height);
  void ensureViewportDepth(u32 width, u32 height);
};

}  // namespace backend

KENSHIN_NS_END
