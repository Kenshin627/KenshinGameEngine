#pragma once

// ── Render backend abstraction (S1.2) ───────────────────────────
// Backend is the ONLY module allowed to make raw GPU API calls
// (renderEngine discipline). Consumers never see Vulkan types here.

#include <memory>
#include <vector>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace backend {

/// Per-frame render parameters (S1.3: clear only; grows over stages).
struct FrameBeginInfo {
  u32 width = 0;
  u32 height = 0;
  f32 clearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
};

enum class GpuType {
  Unknown,
  Integrated,
  Discrete,
  Virtual,
  Cpu,
};

/// Device capability report, filled during initialize().
struct BackendCaps {
  char deviceName[256] = {};
  GpuType gpuType = GpuType::Unknown;

  u32 apiVersionMajor = 0;
  u32 apiVersionMinor = 0;
  u32 apiVersionPatch = 0;
  u32 driverVersionRaw = 0;

  u64 deviceLocalMemoryMB = 0;

  // Features critical to the virtual geometry pipeline.
  // shaderBufferInt64Atomics backs the 64-bit atomic VisBuffer (S4.4).
  bool supportsShaderBufferInt64Atomics = false;
  bool enabledShaderBufferInt64Atomics = false;
  bool supportsTimelineSemaphore = false;
  bool enabledTimelineSemaphore = false;

  bool presentSupported = false;

  u32 graphicsFamily = 0xFFFFFFFFu;
  u32 computeFamily = 0xFFFFFFFFu;
  u32 transferFamily = 0xFFFFFFFFu;
  bool graphicsAndComputeSameFamily = false;
  bool dedicatedTransferFamily = false;
};

struct BackendInitParams {
  /// Native window handle (HWND); enables the present-support check.
  void* nativeWindowHandle = nullptr;
  bool enableValidation = false;
};

class IRenderBackend {
 public:
  virtual ~IRenderBackend() = default;

  virtual bool initialize(const BackendInitParams& params) = 0;
  virtual void shutdown() = 0;
  virtual const BackendCaps& caps() const = 0;

  // ---- frame loop (S1.3) ------------------------------------------
  /// Begin a frame: waits for the in-flight slot, (re)creates render
  /// targets on size change, acquires the swapchain image and records
  /// the clear into the frame command buffer.
  virtual bool beginFrame(const FrameBeginInfo& info) = 0;

  /// End the frame: blits the offscreen target to the swapchain image,
  /// submits and presents.
  virtual void endFrame() = 0;

  /// Blocking readback of the most recently ended frame's offscreen
  /// target. Output: RGBA8, rows top-down. Returns false on failure.
  virtual bool readbackFrame(std::vector<u8>& rgbaOut, u32& width,
                             u32& height) = 0;

  // ---- one-shot compute (S1.4) -------------------------------------
  /// A self-contained compute job: uploads buffers, dispatches once,
  /// waits for completion and copies marked buffers back to the host.
  /// Bindings map to SSBO set=0, binding=i. Used for shader smoke tests
  /// and offline utilities; per-frame in-frame compute arrives in Phase 4.
  struct Job {
    // 8 slots: the cluster-cull pass binds params + clusters + groups +
    // work items + instances + outputs; 4 was the S1.4 smoke-test shape.
    static constexpr u32 kMaxBuffers = 11;  // 10 = the HW pair list (M4.2)
    struct Buffer {
      const void* initData = nullptr;  // host data copied in (may be null)
      usize size = 0;                  // bytes, 0 = slot unused
      bool readBack = false;
    };
    const u32* spirv = nullptr;
    usize spirvBytes = 0;
    const char* entryPoint = "main";
    u32 groupCountX = 1;
    Buffer buffers[kMaxBuffers] = {};
  };
  /// Returns false on failure. For each buffer with readBack, outputs[i]
  /// receives its bytes.
  virtual bool runComputeOneshot(const Job& job,
                                 std::vector<u8> outputs[Job::kMaxBuffers]) = 0;

  // ---- ImGui integration (S1.5) ------------------------------------
  /// Initialize the ImGui Vulkan + Win32 backends. The caller must have
  /// created the ImGui context. The application then draws widgets
  /// between beginFrame() and endFrame().
  virtual bool initImGuiBackend(void* hwnd, u32 width, u32 height) = 0;

  /// Texture id (opaque descriptor-set handle) of the CURRENT frame
  /// slot's viewport target for ImGui::Image. Valid between
  /// beginFrame() and endFrame().
  virtual void* viewportImGuiTexture() = 0;

  /// Request a capture of the NEXT endFrame's final swapchain image
  /// (UI included). Read it back with readbackFullFrame().
  /// Unlike readbackFrame this is NOT deterministic (UI content).
  virtual void requestFullFrameCapture() = 0;
  virtual bool readbackFullFrame(std::vector<u8>& rgbaOut, u32& width,
                                 u32& height) = 0;

  // ---- baseline mesh path (S2.2) -----------------------------------
  /// Handle to an uploaded mesh (device-local vertex + index buffers).
  /// Vertices are SOA: three separate bindings (positions 3f, normals
  /// 3f, uvs 2f) matching kenshin::assets::MeshData.
  static constexpr u32 kInvalidMeshHandle = 0xFFFFFFFFu;
  struct MeshUpload {
    const f32* positions = nullptr;  // xyz per vertex
    const f32* normals = nullptr;    // xyz per vertex
    const f32* uvs = nullptr;        // uv per vertex
    u32 vertexCount = 0;
    const u32* indices = nullptr;
    u32 indexCount = 0;
  };
  /// Upload a mesh (staging copy, one-time). Returns handle or kInvalidMeshHandle.
  virtual u32 createMeshBuffer(const MeshUpload& upload) = 0;
  virtual void destroyMeshBuffer(u32 mesh) = 0;

  // ---- graphics pipelines (renderer layer defines, backend translates) ---
  /// Pipeline state enums — renderer-layer policy; translated to the
  /// native API inside the backend (UE RHI / Godot RenderingDevice cut).
  enum class CompareOp {
    Never,
    Less,
    Equal,
    LessOrEqual,
    Greater,
    NotEqual,
    GreaterOrEqual,
    Always
  };
  enum class CullMode { None, Back, Front };
  enum class FrontFace { CounterClockwise, Clockwise };

  /// One graphics pipeline = a shader pair + fixed-function state.
  /// Vertex input is the engine-wide SOA PNU layout (see MeshUpload).
  struct GraphicsPipelineInfo {
    const u32* vsSpirv = nullptr;
    usize vsBytes = 0;
    const u32* fsSpirv = nullptr;
    usize fsBytes = 0;
    bool depthTest = true;
    bool depthWrite = true;
    CompareOp depthCompare = CompareOp::Less;
    CullMode cullMode = CullMode::Back;
    FrontFace frontFace = FrontFace::CounterClockwise;
  };
  static constexpr u32 kInvalidPipelineHandle = 0xFFFFFFFFu;
  /// Compile SPIR-V + build the pipeline. Returns handle or
  /// kInvalidPipelineHandle on failure.
  virtual u32 createGraphicsPipeline(const GraphicsPipelineInfo& info) = 0;
  virtual void destroyGraphicsPipeline(u32 pipeline) = 0;

  /// Per-draw push payload (row-major 4x4 as the Slang shaders read it).
  struct DrawData {
    f32 viewProj[16];  // column-major mat4 (glm layout)
    f32 kd[4];         // material diffuse RGBA
    f32 lightDir[4];   // TO light, world
    f32 camPos[4];     // world-space camera position (specular)
  };
  /// Record an opaque draw into the CURRENT frame's viewport target.
  /// Must be called between beginFrame() and endFrame(); depth-tested.
  virtual void draw(u32 pipeline, u32 mesh, const DrawData& data) = 0;

  // ---- general device buffers (GPU Scene / compute paths) -----------
  /// Usage flags (bitfield). Storage = SSBO binding (the virtual geometry
  /// pipeline's currency); Uniform for small constant blocks; TransferSrc
  /// lets a buffer feed later copies.
  enum BufferUsage : u32 {
    BufferStorage = 0x1,
    BufferUniform = 0x2,
    BufferTransferSrc = 0x4,
    BufferHostVisible = 0x8,  // persistently mapped (CPU-written params)
    BufferIndirect = 0x10,    // dispatch-indirect command source
    BufferTransferDst = 0x20  // fillBuffer / copy target
  };
  struct BufferDesc {
    usize size = 0;
    u32 usage = BufferStorage;
    const void* initialData = nullptr;  // one-shot upload when non-null
    const char* debugName = nullptr;    // RenderDoc-visible when set
  };
  static constexpr u32 kInvalidBufferHandle = 0xFFFFFFFFu;
  /// Create a device-local buffer (optionally filled). Returns handle or
  /// kInvalidBufferHandle on failure.
  virtual u32 createBuffer(const BufferDesc& desc) = 0;
  /// Deferred destruction (safe while frames are in flight).
  virtual void destroyBuffer(u32 buffer) = 0;
  /// Blocking readback of a byte range (verification paths). Returns false
  /// on failure.
  virtual bool readbackBuffer(u32 buffer, usize offset, usize bytes,
                              std::vector<u8>& out) = 0;

  // ---- fullscreen pass (VisBuffer decode path, S4.5) ----------------
  /// A fullscreen-triangle graphics pipeline: no vertex input (positions
  /// generated in the VS), depth test/write OFF (the VisBuffer already
  /// resolved visibility), and a descriptor set layout derived from the
  /// FS's declared SSBO bindings (binding i <- buffer handle i).
  /// Push constants: one 128-byte block visible to VS|FS.
  struct FullscreenPipelineInfo {
    const u32* vsSpirv = nullptr;
    usize vsBytes = 0;
    const u32* fsSpirv = nullptr;
    usize fsBytes = 0;
    /// Null pipelines only: rasterize CONSERVATIVELY (overestimate) so
    /// every pixel the PS's exact quantized test can own gets a
    /// fragment (the fixed-function unsnapped rasterization misses
    /// snapped-but-not-unsnapped pixels - single-pixel HW/SW seams).
    bool conservativeRaster = false;
  };
  virtual u32 createFullscreenPipeline(const FullscreenPipelineInfo& info) = 0;
  /// M4.2 hardware raster path: VS+FS with ZERO attachments (the PS
  /// writes the VisBuffer SSBO atomically). Same info shape as the
  /// fullscreen pipeline (vsMain/psMain from one .slang file).
  virtual u32 createNullPipeline(const FullscreenPipelineInfo& info) = 0;
  virtual void destroyNullPipeline(u32 pipeline) = 0;
  /// Draw the HW-routed clusters: opens a no-attachment rendering
  /// scope, binds `pipeline`'s SSBO set (buffers[i] -> binding i, the
  /// FS's reflected layout) and issues ONE indirect draw reading the
  /// VkDrawIndirectCommand at argsBuffer[argsOffset] (the chain's
  /// count_to_draw conversion writes it). Self-contained: closes its
  /// own scope.
  virtual void drawNullIndirect(u32 pipeline, const u32* bufferHandles,
                                u32 bufferCount, u32 argsBuffer,
                                usize argsOffset, u32 width, u32 height) = 0;
  /// M4.2 barrier variants: compute -> indirect-draw-args read, and
  /// compute -> vertex/fragment read+write (the HW pass reads the pair
  /// lists in its VS and atomically writes the VisBuffer in its PS).
  virtual void barrierIndirectArgs(u32 buffer) = 0;
  virtual void barrierGraphicsRw(u32 buffer) = 0;
  /// ...and the way back: fragment-written -> compute read/write.
  virtual void barrierGraphicsToCompute(u32 buffer) = 0;

  /// M4.3 async compute: record the cull/raster chain onto the
  /// dedicated compute queue so it overlaps the graphics queue's
  /// decode/present work (UE5's layout: SW chain on async compute,
  /// HW raster + decode on graphics, both atomically writing the same
  /// VisBuffer). v1: ONE chain command buffer per frame (cull ->
  /// SW raster -> post -> final HZB) submitted as one compute
  /// submission; the graphics frame waits its completion before the
  /// HW draw + decode - frame N's chain overlaps frame N-1's
  /// graphics work (pipelining). Same-frame HW||SW concurrency needs
  /// the cull/raster submission split + a second graphics submission
  /// (v2). Every call returns false when no async compute queue
  /// exists - the caller then records everything onto the graphics
  /// queue exactly as before (the chain's fallback).
  virtual bool beginChain() = 0;
  virtual void submitChain() = 0;
  /// True when the last submitChain ran on the compute queue (the
  /// frame submission then waits the chain's completion).
  virtual bool chainOnComputeQueue() const = 0;
  /// M4.4 A/B: force the chain back onto the graphics queue even
  /// when an async compute queue exists (frame-time comparisons).
  virtual void setAsyncChainEnabled(bool enabled) = 0;

  // ---- page streaming uploads (M6.2, P2) ----------------------------
  /// Fixed-size page slots in ONE device-local buffer, CONCURRENT
  /// across the graphics/compute/transfer families so the transfer
  /// queue fills slots while frames read them. Mirrors the chain
  /// pattern: the backend owns the upload command pools, the queue
  /// submission and its OWN timeline - callers only record batches.
  ///
  /// Residency protocol (the streaming manager's side of the
  /// contract): a page becomes visible to frames only after the
  /// caller flips its page-table entry, and that flip happens only
  /// after waitPageUploads(value) of the batch that filled the slot.
  /// Slot REUSE (evict page q, upload page p) additionally needs the
  /// old readers drained - every batch submission waits the frame
  /// timeline's last signaled value, so recording batches mid-frame
  /// (after beginFrame's pacing wait) costs nothing extra.
  ///
  /// Create the slot pool: slotCount x slotBytes device-local bytes
  /// (Storage | TransferDst). Returns a REGULAR buffer handle -
  /// shaders bind it, readbackBuffer verifies it.
  virtual u32 createPageSlotPool(u32 slotCount, u32 slotBytes) = 0;
  /// Open a page-upload batch targeting `slotPool` (a createPageSlotPool
  /// handle). Waits the previous batch's completion first (its staging
  /// bytes may still be in flight), then resets the batch command
  /// buffer + staging arena. Main thread only - recorded at frame
  /// boundaries while no other recording is active. Returns false when
  /// a batch is already open or the pool handle is invalid.
  virtual bool beginPageUploads(u32 slotPool) = 0;
  /// memcpy `bytes` (<= the pool's slotBytes) into the batch's staging
  /// arena and record the copy into slot `slot`. Returns false when
  /// the arena is full or arguments are invalid - close the batch with
  /// submitPageUploads and continue in a fresh one.
  virtual bool copyPageToSlot(const void* hostData, u32 slot, u32 bytes) = 0;
  /// End + submit the batch on the transfer queue (dedicated DMA
  /// family) - or the graphics queue when none exists /
  /// setTransferQueueEnabled(false). Returns the signaled timeline
  /// value (0 = empty batch or failure); waitPageUploads(value) proves
  /// the slots' new contents.
  virtual u64 submitPageUploads() = 0;
  /// Host-block until the page-upload timeline reaches `value`.
  virtual bool waitPageUploads(u64 value) = 0;
  /// A/B + hardware fallback: force batches onto the graphics queue.
  /// Takes effect from the NEXT beginPageUploads.
  virtual void setTransferQueueEnabled(bool enabled) = 0;
  /// True when batches go to the transfer queue (the transfer family
  /// is distinct from graphics - a dedicated DMA queue here; the
  /// graphics-queue fallback covers everything else).
  virtual bool pageUploadsOnTransferQueue() const = 0;
  /// Draw the fullscreen triangle in the CURRENT frame scope.
  /// bufferHandles[i] binds to SSBO binding i (as declared by the FS);
  /// `push` is copied into the 128-byte push block.
  virtual void drawFullscreen(u32 pipeline, const u32* bufferHandles,
                              u32 bufferCount, const void* push,
                              usize pushBytes) = 0;

  // ---- in-frame compute (链入帧化) ------------------------------------
  /// A persistent compute pipeline: descriptor layout derived from the
  /// SPIR-V's declared SSBO bindings (binding i <- buffer handle i in
  /// dispatchCompute). Created once, dispatched every frame.
  struct ComputePipelineInfo {
    const u32* csSpirv = nullptr;
    usize csBytes = 0;
  };
  virtual u32 createComputePipeline(const ComputePipelineInfo& info) = 0;
  virtual void destroyComputePipeline(u32 pipeline) = 0;
  /// Record a dispatch into the CURRENT frame's command buffer. Writes
  /// this frame slot's descriptor set (like drawFullscreen) - safe to
  /// re-dispatch with changing buffer contents every frame.
  virtual void dispatchCompute(u32 pipeline, const u32* bufferHandles,
                               u32 bufferCount, u32 groupsX, u32 groupsY,
                               u32 groupsZ) = 0;
  /// Barrier a buffer from compute writes to its next consumer (compute
  /// or fragment reads); call between passes that hand data over.
  virtual void bufferBarrier(u32 buffer, bool toFragment) = 0;
  /// vkCmdFillBuffer in the current frame (e.g. VisBuffer zeroing).
  virtual void fillBuffer(u32 buffer, u32 value) = 0;
  /// GPU-sized dispatch: the command's (x,y,z) come from `countBuffer`
  /// at `offset` (VkDispatchIndirectCommand = 3 x u32, written by a
  /// previous pass). The Nanite-style GPU-driven chain.
  virtual void dispatchComputeIndirect(u32 pipeline, const u32* bufferHandles,
                                       u32 bufferCount, u32 countBuffer,
                                       usize countOffset) = 0;
  /// Persistently mapped host pointer of a BufferHostVisible buffer.
  virtual void* mappedPtr(u32 buffer) = 0;
  /// Byte size of a live buffer (validation guards / parity checks).
  virtual usize bufferSize(u32 buffer) const = 0;
  /// The frame slot currently being recorded (0..framesInFlight-1);
  /// per-slot resources (chain intermediates, VisBuffer) key off it.
  virtual u32 currentFrameSlot() const = 0;
  /// The slot of the most recently ENDED frame (post-frame readbacks
  /// must read THIS slot - currentFrameSlot has already advanced).
  virtual u32 lastCompletedFrameSlot() const = 0;

  // ---- GPU timing (debug overlay readout) ---------------------------
  /// Last completed frame's GPU timings. Zeroes when unsupported (queue
  /// family without timestampValidBits).
  struct GpuTimings {
    f32 frameMs = 0.0f;   // whole submission (first to last timestamp)
    f32 opaqueMs = 0.0f;  // first timestamp to viewport scope close
  };
  virtual GpuTimings gpuTimings() const = 0;
};

/// Factory implemented by the Vulkan backend (src/backend/).
std::unique_ptr<IRenderBackend> createVulkanBackend();

}  // namespace backend

KENSHIN_NS_END
