cmake_minimum_required(VERSION 3.24)

# ── Tests project ───────────────────────────────────────────────
# smoke: basic smoke tests (S0.2)
# Later: geometry (builder structural assertions, S3), cpu_reference_* (GPU diff references, S4)

project(KENSHINTests VERSION 0.1.0 LANGUAGES CXX)

find_package(Catch2 CONFIG REQUIRED)

add_executable(kenshin_tests
    src/assets_tests.cpp
    src/clod_tests.cpp
    src/cpu_reference_cluster_cull.cpp
    src/cpu_reference_raster.cpp
    src/cpu_reference_cull.cpp
    src/crc32_tests.cpp
    src/dag_tests.cpp
    src/engine_tests.cpp
    src/kgeo_tests.cpp
    src/lodcut_tests.cpp
    src/shader_reflection_tests.cpp
    src/shapes_tests.cpp
    src/smoke.cpp
    src/vgchain_tests.cpp
    src/hzb_tests.cpp
    src/crash_main.cpp
)

# crash_main.cpp provides main() with the SEH crash filter (the app's
# crashLogFilter twin); Catch2WithMain would carry a second main.
target_link_libraries(kenshin_tests PRIVATE kenshin::engine Catch2::Catch2)

# Headless Slang compilation in tests: the shader tree lives in the source
# dir and the Slang runtime DLLs must sit next to the test executable.
target_compile_definitions(kenshin_tests PRIVATE
    KENSHIN_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
set(KE_SLANG_BIN ${CMAKE_SOURCE_DIR}/ThirdParty/slang/bin)
if(EXISTS ${KE_SLANG_BIN}/slang.dll)
    add_custom_command(TARGET kenshin_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${KE_SLANG_BIN}/slang.dll
            ${KE_SLANG_BIN}/slang-compiler.dll
            $<TARGET_FILE_DIR:kenshin_tests>
        COMMENT "Copying Slang runtime DLLs next to kenshin_tests")
endif()

if(MSVC)
    target_compile_options(kenshin_tests PRIVATE /W4 /permissive- /utf-8 /MP)
endif()

# ── GPU render regression net ──────────────────────────────────────
# Byte-identical screenshot vs a reference under tests/data/ (regeneration
# instructions at the top of render_regression.cmake). These run the real
# GPU: deterministic on one machine/driver, intentionally binary-diff.
add_test(NAME render_baseline
    COMMAND ${CMAKE_COMMAND}
        -DDEMO=$<TARGET_FILE:KENSHINDemo>
        -DBUILDER=$<TARGET_FILE:kgeo_builder>
        -DREF=${CMAKE_CURRENT_SOURCE_DIR}/data/render_baseline_ref.bmp
        -DOUT=${CMAKE_CURRENT_BINARY_DIR}/render_baseline_out.bmp
        -DEXEDIR=$<TARGET_FILE_DIR:KENSHINDemo>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/render_regression.cmake)
add_test(NAME render_kgeo
    COMMAND ${CMAKE_COMMAND}
        -DDEMO=$<TARGET_FILE:KENSHINDemo>
        -DBUILDER=$<TARGET_FILE:kgeo_builder>
        -DREF=${CMAKE_CURRENT_SOURCE_DIR}/data/render_kgeo_ref.bmp
        -DOUT=${CMAKE_CURRENT_BINARY_DIR}/render_kgeo_out.bmp
        -DEXEDIR=$<TARGET_FILE_DIR:KENSHINDemo>
        -DKGEO=1 -DPROC=knot -DLOD=1
        -P ${CMAKE_CURRENT_SOURCE_DIR}/render_regression.cmake)
set_tests_properties(render_baseline render_kgeo PROPERTIES LABELS "gpu")

add_test(NAME kenshin_tests COMMAND kenshin_tests)
set_tests_properties(kenshin_tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}   # resolve tests/assets relative paths
)
