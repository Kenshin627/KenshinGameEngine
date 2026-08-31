cmake_minimum_required(VERSION 3.24)

# ── KENSHINApp application project ──────────────────────────────
# The demo executable: grows into the full demo scene over the stages
# (--frames/--screenshot/--cam).

project(KENSHINApp VERSION 0.1.0 LANGUAGES CXX)

add_executable(KENSHINDemo
    src/main.cpp
)

target_link_libraries(KENSHINDemo PRIVATE kenshin::engine)

if(MSVC)
    target_compile_options(KENSHINDemo PRIVATE /W4 /permissive- /utf-8 /MP)
endif()
