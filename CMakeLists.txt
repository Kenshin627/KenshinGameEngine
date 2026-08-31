cmake_minimum_required(VERSION 3.24)

project(KEngineWorkspace VERSION 0.1.0 LANGUAGES CXX)

# 工作区：engine（库）/ app（可执行）/ tools（CLI）/ tests（Catch2）四个项目。
# 构建预设见 CMakePresets.json（MSVC + vcpkg manifest）。

enable_testing()

add_subdirectory(engine)
add_subdirectory(app)
add_subdirectory(tools)
add_subdirectory(tests)
