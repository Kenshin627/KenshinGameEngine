#include "kenshin/backend/VulkanBackend.h"

#include "vulkan/VulkanContext.h"
#include "vulkan/VulkanDevice.h"
#include "kenshin/core/Log.h"

#include <cstdio>
#include <cstring>

KENSHIN_NS_BEGIN

namespace backend
{

    struct VulkanBackend::Impl
    {
        VulkanContext context;
        VulkanDevice device;
        bool presentSupported = false;
    };

    VulkanBackend::VulkanBackend()
        : m_impl(std::make_unique<Impl>())
    {
    }

    VulkanBackend::~VulkanBackend()
    {
        shutdown();
    }

    bool VulkanBackend::initialize(const BackendInitParams& params)
    {
        const bool needSurface = params.nativeWindowHandle != nullptr;
        if (!m_impl->context.init(params.enableValidation, needSurface))
            return false;

        if (params.nativeWindowHandle)
            m_impl->presentSupported = m_impl->context.checkPresentSupport(params.nativeWindowHandle);

        if (!m_impl->device.init(m_impl->context))
            return false;

        // ---- capability report -------------------------------------------
        BackendCaps& caps = m_caps;
        std::snprintf(caps.deviceName, sizeof(caps.deviceName), "%s",
                      m_impl->context.props.deviceName);
        caps.gpuType = [](VkPhysicalDeviceType type) {
            switch (type)
            {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return GpuType::Discrete;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return GpuType::Integrated;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return GpuType::Virtual;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:            return GpuType::Cpu;
            default:                                     return GpuType::Unknown;
            }
        }(m_impl->context.props.deviceType);

        caps.apiVersionMajor = VK_API_VERSION_MAJOR(m_impl->context.props.apiVersion);
        caps.apiVersionMinor = VK_API_VERSION_MINOR(m_impl->context.props.apiVersion);
        caps.apiVersionPatch = VK_API_VERSION_PATCH(m_impl->context.props.apiVersion);
        caps.driverVersionRaw = m_impl->context.props.driverVersion;

        u64 localMB = 0;
        const auto& memProps = m_impl->context.memProps;
        for (u32 i = 0; i < memProps.memoryHeapCount; ++i)
            if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                localMB += memProps.memoryHeaps[i].size / (1024 * 1024);
        caps.deviceLocalMemoryMB = localMB;

        caps.supportsShaderBufferInt64Atomics = m_impl->context.supportsBufferInt64Atomics;
        caps.enabledShaderBufferInt64Atomics  = m_impl->device.enabledBufferInt64Atomics;
        caps.supportsTimelineSemaphore = m_impl->context.supportsTimelineSemaphore;
        caps.enabledTimelineSemaphore  = m_impl->device.enabledTimelineSemaphore;

        caps.presentSupported = m_impl->presentSupported;
        caps.graphicsFamily = m_impl->context.graphicsFamily;
        caps.computeFamily  = m_impl->context.computeFamily;
        caps.graphicsAndComputeSameFamily =
            m_impl->context.graphicsFamily == m_impl->context.computeFamily;

        static const char* kGpuTypeNames[] = { "unknown", "integrated", "discrete", "virtual", "cpu" };
        KENSHIN_INFO("[vulkan] GPU: {} (api {}.{}, {}, {}MB device-local)",
                     caps.deviceName, caps.apiVersionMajor, caps.apiVersionMinor,
                     kGpuTypeNames[static_cast<int>(caps.gpuType)], caps.deviceLocalMemoryMB);
        KENSHIN_INFO("[vulkan] feature shaderBufferInt64Atomics = {}{}",
                     caps.supportsShaderBufferInt64Atomics ? "supported" : "NOT supported",
                     caps.enabledShaderBufferInt64Atomics ? " (enabled)" : "");
        KENSHIN_INFO("[vulkan] feature timelineSemaphore = {}{}",
                     caps.supportsTimelineSemaphore ? "supported" : "NOT supported",
                     caps.enabledTimelineSemaphore ? " (enabled)" : "");
        KENSHIN_INFO("[vulkan] queues: graphics={} compute={}{}{}",
                     caps.graphicsFamily, caps.computeFamily,
                     caps.graphicsAndComputeSameFamily ? " (same family)" : "",
                     params.nativeWindowHandle
                         ? (caps.presentSupported ? ", present=ok" : ", present=NOT supported")
                         : ", present=unchecked (no window)");
        return true;
    }

    void VulkanBackend::shutdown()
    {
        if (m_impl)
        {
            m_impl->device.shutdown();
            m_impl->context.shutdown();
        }
    }

    std::unique_ptr<IRenderBackend> createVulkanBackend()
    {
        return std::make_unique<VulkanBackend>();
    }

} // namespace backend

KENSHIN_NS_END
