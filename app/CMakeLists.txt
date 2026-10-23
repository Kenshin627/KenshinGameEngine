cmake_minimum_required(VERSION 3.24)

# ── KENSHINApp application project ──────────────────────────────
# The demo executable: grows into the full demo scene over the stages
# (--frames/--screenshot/--cam).

project(KENSHINApp VERSION 0.1.0 LANGUAGES CXX)

add_executable(KENSHINDemo
    src/main.cpp
    src/DemoScene.cpp
    src/KgeoPreview.cpp
)

target_link_libraries(KENSHINDemo PRIVATE kenshin::engine)

# stb_image.h (the implementation compiles once inside the engine lib)
target_include_directories(KENSHINDemo SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/ThirdParty)

kenshin_target_defaults(KENSHINDemo)

# Dev: ship the Vulkan validation layer next to the executable. The layer comes
# from the vcpkg manifest (vulkan-validationlayers); the app points
# VK_LAYER_PATH at its own directory at startup so the loader discovers it.
set(KE_VKG_LAYER_DIR ${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/bin)
if(EXISTS ${KE_VKG_LAYER_DIR}/VkLayer_khronos_validation.json)
    add_custom_command(TARGET KENSHINDemo POST_BUILD
        # mimalloc.dll: the vcpkg-built layer links against it
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${KE_VKG_LAYER_DIR}/VkLayer_khronos_validation.dll
            ${KE_VKG_LAYER_DIR}/VkLayer_khronos_validation.json
            ${KE_VKG_LAYER_DIR}/mimalloc.dll
            ${KE_VKG_LAYER_DIR}/mimalloc-redirect.dll
            $<TARGET_FILE_DIR:KENSHINDemo>
        COMMENT "Copying Vulkan validation layer next to KENSHINDemo")
endif()

kenshin_deploy_runtime(KENSHINDemo)
