cmake_minimum_required(VERSION 3.24)

# ── KEngineApp 应用项目 ──────────────────────────────────────────
# demo 可执行：随阶段推进长成完整 demo 场景（--frames/--screenshot/--cam）。

project(KEngineApp VERSION 0.1.0 LANGUAGES CXX)

add_executable(KEngineDemo
    src/main.cpp
)

target_link_libraries(KEngineDemo PRIVATE ke::engine)

if(MSVC)
    target_compile_options(KEngineDemo PRIVATE /W4 /permissive- /utf-8 /MP)
endif()
