cmake_minimum_required(VERSION 3.24)

# ── KEngine 引擎项目 ─────────────────────────────────────────────
# 结构参照 renderEngine：include/ke（内部公共头）与 src（实现）镜像，
# backend/ 是唯一允许裸 GPU API 调用的模块（S1.2 起）。

project(KEngine VERSION 0.1.0 LANGUAGES CXX)

# git 短 SHA（供版本横幅；无 git 环境时回退 "unknown"）
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE KE_GIT_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT KE_GIT_SHA)
    set(KE_GIT_SHA "unknown")
endif()

add_library(ke_engine STATIC
    src/Version.cpp
    src/core/Input.cpp
    src/core/Log.cpp
    src/core/Timer.cpp
    src/core/Window.cpp
)
add_library(ke::engine ALIAS ke_engine)

target_compile_features(ke_engine PUBLIC cxx_std_20)

target_include_directories(ke_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(ke_engine PUBLIC
    KE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    KE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    KE_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    KE_GIT_SHA="${KE_GIT_SHA}"
    $<$<CONFIG:Debug>:KE_BUILD_DEBUG=1>
)

if(MSVC)
    target_compile_options(ke_engine PRIVATE
        /W4 /permissive- /utf-8 /MP
        /w14242   # int 与 size_t 转换
        /w14254   # 运算符优先级转换截断
        /w14263   # 成员函数覆盖但非虚
        /w14265   # 类有虚函数但析构非虚
        /w14287   # 无符号/有符号不匹配
        /w14296   # 表达式恒为 false
        /w14545   # 内联前的逗号表达式
        /w14640   # 谓词安全警告
    )
endif()

# 公共依赖（头文件出现在公共头里 → PUBLIC）
find_package(spdlog CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
target_link_libraries(ke_engine PUBLIC spdlog::spdlog glm::glm)
