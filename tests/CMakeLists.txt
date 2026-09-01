cmake_minimum_required(VERSION 3.24)

# ── Tests project ───────────────────────────────────────────────
# smoke: basic smoke tests (S0.2)
# Later: geometry (builder structural assertions, S3), cpu_reference_* (GPU diff references, S4)

project(KENSHINTests VERSION 0.1.0 LANGUAGES CXX)

find_package(Catch2 CONFIG REQUIRED)

add_executable(kenshin_tests
    src/assets_tests.cpp
    src/clod_tests.cpp
    src/dag_tests.cpp
    src/kgeo_tests.cpp
    src/lodcut_tests.cpp
    src/shapes_tests.cpp
    src/smoke.cpp
)

target_link_libraries(kenshin_tests PRIVATE kenshin::engine Catch2::Catch2WithMain)

if(MSVC)
    target_compile_options(kenshin_tests PRIVATE /W4 /permissive- /utf-8 /MP)
endif()

add_test(NAME kenshin_tests COMMAND kenshin_tests)
set_tests_properties(kenshin_tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}   # resolve tests/assets relative paths
)
