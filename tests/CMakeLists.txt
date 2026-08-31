cmake_minimum_required(VERSION 3.24)

# ── Tests project ───────────────────────────────────────────────
# smoke: basic smoke tests (S0.2)
# Later: geometry (builder structural assertions, S3), cpu_reference_* (GPU diff references, S4)

project(KEngineTests VERSION 0.1.0 LANGUAGES CXX)

find_package(Catch2 CONFIG REQUIRED)

add_executable(ke_tests
    src/smoke.cpp
)

target_link_libraries(ke_tests PRIVATE ke::engine Catch2::Catch2WithMain)

if(MSVC)
    target_compile_options(ke_tests PRIVATE /W4 /permissive- /utf-8 /MP)
endif()

add_test(NAME ke_tests COMMAND ke_tests)
