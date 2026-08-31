#pragma once

// ── Render backend abstraction (S1.2) ───────────────────────────
// Backend is the ONLY module allowed to make raw GPU API calls
// (renderEngine discipline). Consumers never see Vulkan types here.

#include "kenshin/Common.h"

#include <memory>

KENSHIN_NS_BEGIN

namespace backend
{

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
    };

    /// Factory implemented by the Vulkan backend (src/backend/).
    std::unique_ptr<IRenderBackend> createVulkanBackend();

} // namespace backend

KENSHIN_NS_END
