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
    src/Version.cpp
    src/backend/VulkanBackend.cpp
    src/backend/vulkan/FrameSync.cpp
    src/backend/vulkan/GpuImage.cpp
    src/backend/vulkan/Swapchain.cpp
    src/backend/vulkan/VulkanContext.cpp
    src/backend/vulkan/VulkanDevice.cpp
    src/core/Input.cpp
    src/core/Log.cpp
    src/core/Timer.cpp
    src/core/Window.cpp
    src/manager/ShaderManager.cpp
    src/util/BmpWriter.cpp
)
add_library(kenshin::engine ALIAS kenshin_engine)

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
target_link_libraries(kenshin_engine PUBLIC spdlog::spdlog glm::glm)

# Vulkan backend deps (kept PRIVATE: raw API types never leave the backend)
find_package(volk CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
target_link_libraries(kenshin_engine PRIVATE
    volk::volk volk::volk_headers GPUOpen::VulkanMemoryAllocator)
target_compile_definitions(kenshin_engine PRIVATE
    VK_NO_PROTOTYPES           # volk supplies all vk entry points
    VK_USE_PLATFORM_WIN32_KHR  # Win32 surface for present support check
)

# Slang (vendored binary release; SPIR-V runtime compilation)
target_include_directories(kenshin_engine SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/ThirdParty/slang/include
)
target_link_libraries(kenshin_engine PRIVATE
    ${CMAKE_SOURCE_DIR}/ThirdParty/slang/lib/slang.lib
)
