#pragma once

// ── Render backend abstraction (S1.2) ───────────────────────────
// Backend is the ONLY module allowed to make raw GPU API calls
// (renderEngine discipline). Consumers never see Vulkan types here.

#include "kenshin/Common.h"

#include <memory>
#include <vector>

KENSHIN_NS_BEGIN

namespace backend
{

    /// Per-frame render parameters (S1.3: clear only; grows over stages).
    struct FrameBeginInfo
    {
        u32 width  = 0;
        u32 height = 0;
        f32 clearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    };

    enum class GpuType
    {
        Unknown,
        Integrated,
        Discrete,
        Virtual,
        Cpu,
    };

    /// Device capability report, filled during initialize().
    struct BackendCaps
    {
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
        bool enabledShaderBufferInt64Atomics  = false;
        bool supportsTimelineSemaphore = false;
        bool enabledTimelineSemaphore  = false;

        bool presentSupported = false;

        u32 graphicsFamily = 0xFFFFFFFFu;
        u32 computeFamily  = 0xFFFFFFFFu;
        u32 transferFamily = 0xFFFFFFFFu;
        bool graphicsAndComputeSameFamily = false;
        bool dedicatedTransferFamily = false;
    };

    struct BackendInitParams
    {
        /// Native window handle (HWND); enables the present-support check.
        void* nativeWindowHandle = nullptr;
        bool enableValidation = false;
    };

    class IRenderBackend
    {
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
        virtual bool readbackFrame(std::vector<u8>& rgbaOut, u32& width, u32& height) = 0;

        // ---- one-shot compute (S1.4) -------------------------------------
        /// A self-contained compute job: uploads buffers, dispatches once,
        /// waits for completion and copies marked buffers back to the host.
        /// Bindings map to SSBO set=0, binding=i. Used for shader smoke tests
        /// and offline utilities; per-frame in-frame compute arrives in Phase 4.
        struct Job
        {
            static constexpr u32 kMaxBuffers = 4;
            struct Buffer
            {
                const void* initData = nullptr; // host data copied in (may be null)
                usize size = 0;                 // bytes, 0 = slot unused
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
        virtual bool readbackFullFrame(std::vector<u8>& rgbaOut, u32& width, u32& height) = 0;
    };

    /// Factory implemented by the Vulkan backend (src/backend/).
    std::unique_ptr<IRenderBackend> createVulkanBackend();

} // namespace backend

KENSHIN_NS_END
