cmake_minimum_required(VERSION 3.24)

project(KEngineWorkspace VERSION 0.1.0 LANGUAGES CXX)

# Workspace: engine (lib) / app (executable) / tools (CLI) / tests (Catch2).
# Build presets live in CMakePresets.json (MSVC + vcpkg manifest).

enable_testing()

add_subdirectory(engine)
add_subdirectory(app)
add_subdirectory(tools)
add_subdirectory(tests)
