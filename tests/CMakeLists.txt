cmake_minimum_required(VERSION 3.24)

# ── Tests project ───────────────────────────────────────────────
# smoke: basic smoke tests (S0.2)
# Later: geometry (builder structural assertions, S3), cpu_reference_* (GPU diff references, S4)

project(KENSHINTests VERSION 0.1.0 LANGUAGES CXX)

find_package(Catch2 CONFIG REQUIRED)

add_executable(kenshin_tests
    src/assets_tests.cpp
    src/bindless_tests.cpp
    src/clod_tests.cpp
    src/cpu_reference_cluster_cull.cpp
    src/cpu_reference_raster.cpp
    src/cpu_reference_cull.cpp
    src/clusterquant_tests.cpp
    src/cut_coverage_tests.cpp
    src/crc32_tests.cpp
    src/dag_tests.cpp
    src/ecs_tests.cpp
    src/engine_tests.cpp
    src/gltf_import_tests.cpp
    src/kgeo_tests.cpp
    src/lodcut_tests.cpp
    src/pagestream_tests.cpp
    src/p0_repro.cpp
    src/shader_reflection_tests.cpp
    src/serialization_tests.cpp
    src/shapes_tests.cpp
    src/smoke.cpp
    src/texture_tests.cpp
    src/input_tests.cpp
    src/vgchain_tests.cpp
    src/hzb_tests.cpp
    src/imposter_tests.cpp
    src/kgeo_imposter_tests.cpp
    src/crash_main.cpp
)

# crash_main.cpp provides main() with the SEH crash filter (the app's
# crashLogFilter twin); Catch2WithMain would carry a second main.
target_link_libraries(kenshin_tests PRIVATE kenshin::engine Catch2::Catch2)
# meshopt inline helpers (quantizeSnorm) used by the quant tests
target_include_directories(kenshin_tests PRIVATE ${CMAKE_SOURCE_DIR}/ThirdParty)

# Headless Slang compilation in tests: the shader tree lives in the source
# dir and the Slang runtime DLLs must sit next to the test executable.
target_compile_definitions(kenshin_tests PRIVATE
    KENSHIN_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
kenshin_deploy_runtime(kenshin_tests)

kenshin_target_defaults(kenshin_tests)

# ── GPU render regression net ──────────────────────────────────────
# Byte-identical screenshot vs a reference under tests/data/ (regeneration
# instructions at the top of render_regression.cmake). These run the real
# GPU: deterministic on one machine/driver, intentionally binary-diff.
add_test(NAME render_baseline
    COMMAND ${CMAKE_COMMAND}
        -DDEMO=$<TARGET_FILE:KENSHINDemo>
        -DBUILDER=$<TARGET_FILE:kgeo_builder>
        -DBMPDIFF=$<TARGET_FILE:bmpdiff>
        -DREF=${CMAKE_CURRENT_SOURCE_DIR}/data/render_baseline_ref.bmp
        -DOUT=${CMAKE_CURRENT_BINARY_DIR}/render_baseline_out.bmp
        -DEXEDIR=$<TARGET_FILE_DIR:KENSHINDemo>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/render_regression.cmake)
add_test(NAME render_kgeo
    COMMAND ${CMAKE_COMMAND}
        -DDEMO=$<TARGET_FILE:KENSHINDemo>
        -DBUILDER=$<TARGET_FILE:kgeo_builder>
        -DBMPDIFF=$<TARGET_FILE:bmpdiff>
        -DREF=${CMAKE_CURRENT_SOURCE_DIR}/data/render_kgeo_ref.bmp
        -DOUT=${CMAKE_CURRENT_BINARY_DIR}/render_kgeo_out.bmp
        -DEXEDIR=$<TARGET_FILE_DIR:KENSHINDemo>
        -DKGEO=1 -DPROC=knot -DLOD=1
        # v5 quantized grid -> near-tie depth flips across runs; M9.1
        # shadow-edge sampling amplifies the tie jitter (measured ~40
        # px run-to-run vs 5 without shadows) - 64 keeps real
        # regressions (thousands of px) loud while absorbing the jitter.
        -DTOLERANCE=64
        -P ${CMAKE_CURRENT_SOURCE_DIR}/render_regression.cmake)
# HW-path anchor: same reference, --no-hw. Route and no-hw render
# byte-identically (verified after a13ba83), so both tests sharing one
# reference pin that equivalence - a HW-only regression splits them.
add_test(NAME render_kgeo_nohw
    COMMAND ${CMAKE_COMMAND}
        -DDEMO=$<TARGET_FILE:KENSHINDemo>
        -DBUILDER=$<TARGET_FILE:kgeo_builder>
        -DBMPDIFF=$<TARGET_FILE:bmpdiff>
        -DREF=${CMAKE_CURRENT_SOURCE_DIR}/data/render_kgeo_ref.bmp
        -DOUT=${CMAKE_CURRENT_BINARY_DIR}/render_kgeo_nohw_out.bmp
        -DEXEDIR=$<TARGET_FILE_DIR:KENSHINDemo>
        -DKGEO=1 -DPROC=knot -DLOD=1 -DNOHW=1
        -DTOLERANCE=64
        -P ${CMAKE_CURRENT_SOURCE_DIR}/render_regression.cmake)
# M8.2: the albedo variant - same knot asset built with --albedo (the
# committed checker_64.bmp), pinning the full texture path (upload +
# blit mips + analytic-lod sampling + material table) deterministically.
add_test(NAME render_kgeo_albedo
    COMMAND ${CMAKE_COMMAND}
        -DDEMO=$<TARGET_FILE:KENSHINDemo>
        -DBUILDER=$<TARGET_FILE:kgeo_builder>
        -DBMPDIFF=$<TARGET_FILE:bmpdiff>
        -DREF=${CMAKE_CURRENT_SOURCE_DIR}/data/render_kgeo_albedo_ref.bmp
        -DOUT=${CMAKE_CURRENT_BINARY_DIR}/render_kgeo_albedo_out.bmp
        -DEXEDIR=$<TARGET_FILE_DIR:KENSHINDemo>
        -DKGEO=1 -DPROC=knot -DLOD=1
        -DALBEDO=${CMAKE_CURRENT_SOURCE_DIR}/assets/checker_64.bmp
        -DTOLERANCE=64
        -P ${CMAKE_CURRENT_SOURCE_DIR}/render_regression.cmake)
set_tests_properties(render_baseline render_kgeo render_kgeo_nohw
    render_kgeo_albedo PROPERTIES LABELS "gpu")

add_test(NAME kenshin_tests COMMAND kenshin_tests)

# ── Crash handler (roadmap 0.2): a deliberate fault must produce a
# symbolized report and still fail the process.
add_executable(crash_probe crash/crash_probe.cpp)
target_link_libraries(crash_probe PRIVATE kenshin::engine)
kenshin_target_defaults(crash_probe)
add_test(NAME crash_handler
    COMMAND ${CMAKE_COMMAND} -DPROBE=$<TARGET_FILE:crash_probe>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/crash/check_crash.cmake)

# ── Include-path case check: Linux filesystems are case-sensitive.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME include_case
        COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tools/check_include_case.py)
endif()
set_tests_properties(kenshin_tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}   # resolve tests/assets relative paths
)
