cmake_minimum_required(VERSION 3.24)

# ── Tools project ───────────────────────────────────────────────
project(KenshinTools VERSION 0.1.0 LANGUAGES CXX)

add_executable(kgeo_builder kgeo_builder/main.cpp)
target_link_libraries(kgeo_builder PRIVATE kenshin::engine)
if(MSVC)
    target_compile_options(kgeo_builder PRIVATE /W4 /permissive- /utf-8 /MP)
endif()
