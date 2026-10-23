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
  /// Returns a SLOT-OWNED buffer whose bytes are a record-time
  /// snapshot of `buffer` (host-visible only). Bind this instead of a
  /// shared host buffer whose contents change per frame: the CPU
  /// rewrite at record time can tear the PREVIOUS frame's still-
  /// executing read (the M9 TAA jitter nondeterminism - a shared
  /// params block with a per-phase jitter word flickered the resolve
  /// run-to-run; everything single-frame looked byte-exact because
  /// memoryless paths converge). The passthrough default keeps
  /// headless/test backends unchanged.
  virtual u32 snapshotHostBuffer(u32 buffer) { return buffer; }
  /// Deferred destruction (safe while frames are in flight).
  virtual void destroyBuffer(u32 buffer) = 0;
  /// Blocking readback of a byte range (verification paths). Returns false
  /// on failure.
  virtual bool readbackBuffer(u32 buffer, usize offset, usize bytes,
                              std::vector<u8>& out) = 0;

  // ---- textures (M8.2) ---------------------------------------------
  /// Handle to an uploaded immutable texture array.
  static constexpr u32 kInvalidTextureHandle = 0xFFFFFFFFu;
  struct TextureArrayDesc {
    u32 width = 0;
    u32 height = 0;
    u32 layers = 1;  // Texture2DArray layers (>= 1; 1 = plain 2D)
    /// Top-down RGBA8 rows (the BmpReader output contract), layers
    /// consecutive: width*height*4 bytes each. The mip chain is
    /// generated on the GPU (linear blit downsample); mip count =
    /// floor(log2(max(width, height))) + 1.
    /// hdr: RGBA16F f32 rows instead (the IBL environment probe) -
    /// width*height*4*sizeof(f32) bytes per layer.
    bool hdr = false;
    const u8* rgba = nullptr;
    usize bytes = 0;  // width*height*(hdr ? 16 : 4)*layers
    const char* debugName = nullptr;
  };
  /// One-shot upload of a texture array: staging copy of every layer's
  /// mip 0, GPU blit mip chain, final transition to shader-read layout.
  /// Immutable afterwards. All layers must share dimensions (the
  /// Texture2DArray contract; enforced by the caller).
  virtual u32 createTextureArray(const TextureArrayDesc& desc) = 0;
  /// Destroy a texture array. Static scene resources only: the
  /// implementation drains the device first (no per-frame retire
  /// machinery for textures in v1).
  virtual void destroyTexture(u32 texture) = 0;

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
  /// M8.1 deferred shading - the GBuffer decode pass: VS+FS with TWO
  /// color attachments (A = RGBA16F normal + euclidean distance, B =
  /// RGBA8 base color) and no depth. The FS entry writes SV_Target0/1.
  virtual u32 createGBufferPipeline(const FullscreenPipelineInfo& info) = 0;
  /// Draw into THIS frame slot's GBuffer targets: barriers them into
  /// color-attachment layout, opens the 2-attachment scope (cleared to
  /// zero - the lighting pass reads alpha==0 as background), binds the
  /// pipeline's MIXED set and draws the fullscreen triangle. Binding
  /// contract (M8.2): reflected STORAGE_BUFFER bindings take
  /// `bufferHandles` in ascending order; the reflected SAMPLED_IMAGE
  /// binding gets `albedoTexture`'s view (an invalid handle falls back
  /// to a backend-internal 1x1 dummy so the set is always validly
  /// written) and the SAMPLER binding the backend's albedo sampler.
  /// Self-contained: closes its scope and barriers the targets to
  /// shader-read for the lighting pass.
  virtual void drawGBuffer(u32 pipeline, const u32* bufferHandles,
                           u32 bufferCount, u32 albedoTexture, const void* push,
                           usize pushBytes, u32 width, u32 height) = 0;
  /// M8.1 deferred shading - the lighting pass: fullscreen VS+FS whose
  /// set layout is MIXED - reflected Sampler2D bindings (in ascending
  /// binding order) are bound to this frame slot's GBuffer views (A
  /// first, then B); the reflected StorageBuffer bindings take
  /// `bufferHandles` in ascending order (lights, materials, env).
  /// M8.3: renders LINEAR HDR into this frame slot's RGBA16F target
  /// (no depth; background = exact zero, the tonemap sentinel).
  virtual u32 createLightingPipeline(const FullscreenPipelineInfo& info) = 0;
  virtual void drawLighting(u32 pipeline, const u32* bufferHandles,
                            u32 bufferCount, const void* push, usize pushBytes,
                            u32 width, u32 height, u32 envTexture) = 0;
  /// M8.3 tonemap pass: samples this frame slot's HDR target (bound to
  /// the reflected Sampler2D bindings in ascending order) and writes
  /// the exposure/tonemapped LDR into the frame color target via a
  /// depth-less color scope. Follows drawLighting every deferred frame.
  virtual u32 createTonemapPipeline(const FullscreenPipelineInfo& info) = 0;
  /// `exposureState` binds to the shader's SSBO slot (exposure.slang's
  /// state, or the TonemapPass dummy when auto exposure is off);
  /// `envTexture` to the third sampler (the sky background; the dummy
  /// texture array when invalid).
  virtual void drawTonemap(u32 pipeline, const void* push, usize pushBytes,
                           u32 width, u32 height, bool fromTaa = false,
                           u32 exposureState = kInvalidBufferHandle,
                           u32 envTexture = kInvalidTextureHandle) = 0;
  /// M9 TAA resolve: samples this slot's HDR + GBuffer A + the history
  /// ping-pong pair (all backend-internal) and writes the resolved
  /// RGBA16F; the SSBO `bufferHandles` carry the matrices/blend block
  /// (a push block would exceed the guaranteed 128B). The tonemap pass
  /// that follows with fromTaa=true samples the fresh resolve.
  /// Translucent forward pass (glTF BLEND): a vertex-pulled triangle
  /// list blended PREMULTIPLIED (ONE / ONE_MINUS_SRC_ALPHA, alpha
  /// accumulating coverage) into this slot's HDR target after
  /// drawLighting; no depth attachment (the shader tests the VisBuffer
  /// depth itself), no culling. Buffer handles fill the reflected
  /// STORAGE_BUFFER bindings in ascending order, `envTexture` the
  /// combined image sampler, `albedoTexture` the SAMPLED_IMAGE (+ the
  /// albedo SAMPLER). Defaults: unsupported (no-op).
  virtual u32 createTranslucentPipeline(const FullscreenPipelineInfo& info) {
    (void)info;
    return kInvalidPipelineHandle;
  }
  virtual void drawTranslucent(u32 pipeline, const u32* bufferHandles,
                               u32 bufferCount, u32 envTexture,
                               u32 albedoTexture, const void* push,
                               usize pushBytes, u32 vertexCount,
                               u32 instanceCount, u32 width, u32 height) {
    (void)pipeline, (void)bufferHandles, (void)bufferCount, (void)envTexture;
    (void)albedoTexture;
    (void)push, (void)pushBytes, (void)vertexCount, (void)instanceCount;
    (void)width, (void)height;
  }
  /// Bloom (bloom.slang): `additive` picks the upsample variant.
  virtual u32 createBloomPipeline(const FullscreenPipelineInfo& info,
                                  bool additive) {
    (void)info, (void)additive;
    return kInvalidPipelineHandle;
  }
  /// Record the whole bloom chain over this frame's resolved HDR (the
  /// TAA output when `fromTaa`, else the slot HDR): `levels` 13-tap
  /// downsamples into backend-owned half-resolution targets (one
  /// pipeline instance per pass - a descriptor set is written once per
  /// frame), then levels-1 tent upsamples blended additively into the
  /// next larger level. A following drawTonemap samples the top level.
  /// Returns false when nothing was recorded (tonemap then gets no
  /// bloom).
  virtual bool drawBloom(const u32* downPipelines, const u32* upPipelines,
                         u32 levels, bool fromTaa, u32 width, u32 height) {
    (void)downPipelines, (void)upPipelines, (void)levels, (void)fromTaa;
    (void)width, (void)height;
    return false;
  }
  virtual u32 createTaaPipeline(const FullscreenPipelineInfo& info) = 0;
  virtual void drawTaa(u32 pipeline, const u32* bufferHandles, u32 bufferCount,
                       const void* push, usize pushBytes, u32 width,
                       u32 height) = 0;
  /// M9.1 shadow depth decode: the fsShadowDepth entry (the shadow
  /// chain's VisBuffer decoded from the light's view) rendered into the
  /// backend's persistent R32F shadow target (created on first use,
  /// format-gated at initialize; 2048x2048 v1). SSBO handles like
  /// drawFullscreen; the target cycles COLOR_ATTACHMENT -> SHADER_READ
  /// and drawLighting binds it to the third reflected Sampler2D
  /// binding with a NEAREST sampler (depth comparison must not filter
  /// linearly). `shadowEnabled` false parks the target in SHADER_READ
  /// so lighting reads a cleared-to-one map (fully lit).
  virtual u32 createShadowDecodePipeline(
      const FullscreenPipelineInfo& info) = 0;
  /// scissorRect = inclusive pixel bounds [x0,y0,x1,y1] limiting the
  /// rewrite (Stage C page cache: outside LOADs its prior content);
  /// nullptr = the whole target, CLEARed to far (1.0).
  virtual void drawShadowDecode(u32 pipeline, const u32* bufferHandles,
                                u32 bufferCount, const void* push,
                                usize pushBytes, u32 width, u32 height,
                                u32 level, bool shadowEnabled,
                                const u32 scissorRect[4] = nullptr) = 0;
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
  /// Make THIS frame's chain submission wait the PREVIOUS frame's
  /// graphics submission (call before submitChain). Needed when the
  /// previous frame wrote chain inputs on the graphics queue - the HZB
  /// rebuilt after the HW raster draw. No-op without a separate queue.
  virtual void chainWaitGraphics() {}
  /// Write timestamp `index` (0..6) of this frame slot's CHAIN query
  /// pool onto the command buffer currently recording the chain (the
  /// async compute buffer, or the graphics frame buffer on the
  /// fallback path). The first call of a frame resets the pool. A
  /// no-op when the recording queue's family reports no timestamp
  /// bits. Segment 0..6 bracket cull / SW raster / HZB#1 / post /
  /// HZB#2 / tail; the harvested deltas surface in
  /// gpuTimings().chainMs (0.0 when unsupported).
  virtual void writeChainTimestamp(u32 index) = 0;
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
  /// Auto exposure (exposure.slang): a mixed-layout compute pipeline -
  /// sampler + SSBO bindings from reflection, 128-byte compute push.
  virtual u32 createExposurePipeline(const ComputePipelineInfo& info) {
    (void)info;
    return kInvalidPipelineHandle;
  }
  /// Record the exposure passes over this frame's resolved HDR (the TAA
  /// output when `fromTaa`): clear `histogram`, the histogram pass
  /// (16x16 groups over width x height), then the one-group adapt pass
  /// writing `state`, which is made visible to the following
  /// drawTonemap's fragment reads. Binding i of either pipeline takes
  /// the scene image (sampler 0), `envTexture` (sampler 3, arrayed view;
  /// ~0u = the scene image - the shader then never samples it) or
  /// {-, histogram, state}[i] (SSBO); the same `push` feeds both.
  /// Returns false when nothing was recorded.
  virtual bool drawExposure(u32 histogramPipeline, u32 adaptPipeline,
                            u32 histogram, u32 state, u32 envTexture,
                            bool fromTaa, u32 width, u32 height,
                            const void* push, usize pushBytes) {
    (void)histogramPipeline, (void)adaptPipeline, (void)histogram;
    (void)state, (void)envTexture, (void)fromTaa, (void)width, (void)height;
    (void)push, (void)pushBytes;
    return false;
  }
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
    /// Chain segment times (GPU execution only - async-engine
    /// scheduling/signal latency sits OUTSIDE timestamps; wall-clock
    /// stays the primary cross-variant metric, cf. the M5.3 lesson):
    /// [0] cull, [1] SW main raster, [2] HZB#1, [3] post cull+raster,
    /// [4] HZB#2, [5] tail (barriers + page touch). Zeroed (and
    /// chainValid = false) without timestamp support or before the
    /// first harvest.
    f32 chainMs[6] = {};
    bool chainValid = false;
  };
  virtual GpuTimings gpuTimings() const = 0;
};

/// Factory implemented by the Vulkan backend (src/backend/).
std::unique_ptr<IRenderBackend> createVulkanBackend();

}  // namespace backend

KENSHIN_NS_END
