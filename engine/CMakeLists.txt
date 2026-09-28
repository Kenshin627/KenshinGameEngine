cmake_minimum_required(VERSION 3.24)

# ── KENSHIN engine project ──────────────────────────────────────
# Structure follows renderEngine: include/ke (internal headers) mirrors src/,
# and backend/ is the only module allowed to make raw GPU API calls (from S1.2 on).

project(KENSHIN VERSION 0.1.0 LANGUAGES CXX)

# Git short SHA (for the version banner; falls back to "unknown" without git)
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE KENSHIN_GIT_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT KENSHIN_GIT_SHA)
    set(KENSHIN_GIT_SHA "unknown")
endif()

add_library(kenshin_engine STATIC
    src/assets/GltfLoader.cpp
    src/assets/ObjLoader.cpp
    src/geometry/ClusterDag.cpp
    src/geometry/ClusterQuant.cpp
    src/geometry/ImposterBaker.cpp
    src/geometry/LodCut.cpp
    src/geometry/shapes/BoxGeometry.cpp
    src/geometry/shapes/CylinderGeometry.cpp
    src/geometry/shapes/PlaneGeometry.cpp
    src/geometry/shapes/SphereGeometry.cpp
    src/geometry/shapes/TorusGeometry.cpp
    src/geometry/shapes/TorusKnotGeometry.cpp
    src/geometry/KGeoFile.cpp
    src/geometry/PagePacker.cpp
    src/geometry/ClusterLodBuilder.cpp
    # vendored ImGui backends, compiled here so vk calls resolve through volk
    ${CMAKE_SOURCE_DIR}/ThirdParty/imgui-backends/imgui_impl_vulkan.cpp
    ${CMAKE_SOURCE_DIR}/ThirdParty/imgui-backends/imgui_impl_win32.cpp
    src/Version.cpp
    src/backend/VulkanBackend.cpp
    src/backend/vulkan/FrameSync.cpp
    src/backend/vulkan/GpuImage.cpp
    src/backend/vulkan/Swapchain.cpp
    src/backend/vulkan/VulkanContext.cpp
    src/backend/vulkan/VulkanDevice.cpp
    src/backend/vulkan/CommandContext.cpp
    src/backend/vulkan/ComputeOneShot.cpp
    src/backend/vulkan/Readback.cpp
    src/backend/vulkan/PipelineStore.cpp
    src/backend/vulkan/MeshStore.cpp
    src/backend/vulkan/BufferStore.cpp
    src/backend/vulkan/DescriptorCache.cpp
    src/backend/vulkan/UploadEngine.cpp
    src/core/Input.cpp
    src/core/Log.cpp
    src/core/Timer.cpp
    src/core/Window.cpp
    src/DebugOverlay.cpp
    src/Engine.cpp
    src/manager/MeshPool.cpp
    src/manager/ShaderManager.cpp
    src/manager/TextureManager.cpp
    src/renderer/BaselineForwardPipeline.cpp
    src/renderer/Renderer.cpp
    src/renderer/VirtualGeometryScene.cpp
    src/renderer/InstanceCullPass.cpp
    src/renderer/ClusterCullPass.cpp
    src/renderer/SoftRasterPass.cpp
    src/renderer/LightingPass.cpp
    src/renderer/TaaPass.cpp
    src/renderer/TonemapPass.cpp
    src/renderer/MaterialDecodePass.cpp
    src/renderer/VisBufferChain.cpp
    src/renderer/HzbPass.cpp
    src/renderer/StreamingManager.cpp
    src/scene/Camera.cpp
    src/scene/CameraController.cpp
    src/util/BmpReader.cpp
    src/util/BmpWriter.cpp
    src/math/ray.cpp
)
add_library(kenshin::engine ALIAS kenshin_engine)

# List headers as sources so the VS solution explorer shows them, and group
# sources+headers by their real folder tree (filters mirror the layout).
file(GLOB_RECURSE KENSHIN_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
target_sources(kenshin_engine PRIVATE ${KENSHIN_HEADERS})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/include PREFIX "Header Files" FILES ${KENSHIN_HEADERS})
file(GLOB_RECURSE KENSHIN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/src PREFIX "Source Files" FILES ${KENSHIN_SOURCES})

target_compile_features(kenshin_engine PUBLIC cxx_std_20)

target_include_directories(kenshin_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(kenshin_engine PUBLIC
    KENSHIN_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    KENSHIN_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    KENSHIN_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    KENSHIN_GIT_SHA="${KENSHIN_GIT_SHA}"
    $<$<CONFIG:Debug>:KENSHIN_BUILD_DEBUG=1>
)

if(MSVC)
    target_compile_options(kenshin_engine PRIVATE
        /W4 /permissive- /utf-8 /MP
        /w14242   # conversion from 'int' to 'size_t', possible loss of data
        /w14254   # operator conversion, possible loss of data
        /w14263   # member function hides virtual member in base class
        /w14265   # class has virtual functions but destructor is not virtual
        /w14287   # unsigned/signed mismatch
        /w14296   # expression is always false
        /w14545   # expression before comma has no effect
        /w14640   # thread-unsafe static member initialization
    )
endif()

# Public deps (their headers appear in our public headers -> PUBLIC)
find_package(spdlog CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
target_link_libraries(kenshin_engine PUBLIC spdlog::spdlog glm::glm imgui::imgui)

# Vulkan backend deps (kept PRIVATE: raw API types never leave the backend)
find_package(volk CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(unofficial-spirv-reflect CONFIG REQUIRED)
find_package(SPIRV-Headers CONFIG REQUIRED)
target_link_libraries(kenshin_engine PRIVATE
    volk::volk volk::volk_headers GPUOpen::VulkanMemoryAllocator
    unofficial::spirv-reflect SPIRV-Headers::SPIRV-Headers)
# Use the spirv-headers copy of spirv.h instead of the source-tree-relative
# default inside spirv_reflect.h.
target_compile_definitions(kenshin_engine PRIVATE SPIRV_REFLECT_USE_SYSTEM_SPIRV_H)
target_compile_definitions(kenshin_engine PRIVATE
    VK_NO_PROTOTYPES           # volk supplies all vk entry points
    VK_USE_PLATFORM_WIN32_KHR  # Win32 surface for present support check
    # Static VMA function table (filled explicitly from volk in
    # VulkanDevice). Newer vcpkg VMA defaults this to 1 and then demands
    # vkGetInstanceProcAddr/vkGetDeviceProcAddr in the table.
    VMA_DYNAMIC_VULKAN_FUNCTIONS=0
)

# Slang (vendored binary release; SPIR-V runtime compilation)
target_include_directories(kenshin_engine SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/ThirdParty/slang/include
    ${CMAKE_SOURCE_DIR}/ThirdParty/imgui-backends
    ${CMAKE_SOURCE_DIR}/ThirdParty
)
# Vendored meshoptimizer (master): vcpkg 1.1.1 lacks SimplifyPreserveFolds
# and other new flags clusterlod.h depends on.
file(GLOB KENSHIN_MEMOPT_SOURCES ${CMAKE_SOURCE_DIR}/ThirdParty/meshoptimizer/*.cpp)
target_sources(kenshin_engine PRIVATE ${KENSHIN_MEMOPT_SOURCES})
set_source_files_properties(${KENSHIN_MEMOPT_SOURCES} PROPERTIES COMPILE_OPTIONS "/W0")
set_source_files_properties(
    ${CMAKE_SOURCE_DIR}/ThirdParty/imgui-backends/imgui_impl_vulkan.cpp
    ${CMAKE_SOURCE_DIR}/ThirdParty/imgui-backends/imgui_impl_win32.cpp
    ${CMAKE_SOURCE_DIR}/ThirdParty/stb/stb_image_impl.cpp
    PROPERTIES COMPILE_OPTIONS "/W0")
# Vendored stb_image (glTF albedo decode; the BmpReader stays for the
# engine's own write-format roundtrips).
target_sources(kenshin_engine PRIVATE ${CMAKE_SOURCE_DIR}/ThirdParty/stb/stb_image_impl.cpp)
target_compile_definitions(kenshin_engine PRIVATE
    IMGUI_IMPL_VULKAN_USE_VOLK   # vk entry points via volk's loaded pointers
)
target_link_libraries(kenshin_engine PRIVATE
    ${CMAKE_SOURCE_DIR}/ThirdParty/slang/lib/slang.lib
)
