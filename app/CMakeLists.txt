cmake_minimum_required(VERSION 3.24)

# ── KEngineApp application project ──────────────────────────────
# The demo executable: grows into the full demo scene over the stages
# (--frames/--screenshot/--cam).

project(KEngineApp VERSION 0.1.0 LANGUAGES CXX)

add_executable(KEngineDemo
    src/main.cpp
)

target_link_libraries(KEngineDemo PRIVATE ke::engine)

if(MSVC)
    target_compile_options(KEngineDemo PRIVATE /W4 /permissive- /utf-8 /MP)
endif()
