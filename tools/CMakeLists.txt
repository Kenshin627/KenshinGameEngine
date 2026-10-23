cmake_minimum_required(VERSION 3.24)

# ── Tools project ───────────────────────────────────────────────
project(KenshinTools VERSION 0.1.0 LANGUAGES CXX)

add_executable(kgeo_builder kgeo_builder/main.cpp)
target_link_libraries(kgeo_builder PRIVATE kenshin::engine)
kenshin_target_defaults(kgeo_builder)

# Pixel diff for tests/render_regression.cmake (replaces the CMake
# per-pixel hex loop that took minutes on a mismatching frame).
add_executable(bmpdiff bmpdiff/main.cpp)
target_link_libraries(bmpdiff PRIVATE kenshin::engine)
kenshin_target_defaults(bmpdiff)
