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
    static constexpr u32 kMaxBuffers = 8;
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
    BufferTransferSrc = 0x4
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
  };
  virtual u32 createFullscreenPipeline(const FullscreenPipelineInfo& info) = 0;
  /// Draw the fullscreen triangle in the CURRENT frame scope.
  /// bufferHandles[i] binds to SSBO binding i (as declared by the FS);
  /// `push` is copied into the 128-byte push block.
  virtual void drawFullscreen(u32 pipeline, const u32* bufferHandles,
                              u32 bufferCount, const void* push,
                              usize pushBytes) = 0;

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
