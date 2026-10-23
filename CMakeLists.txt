cmake_minimum_required(VERSION 3.24)

# vcpkg toolchain, located BEFORE project(): an explicit
# -DCMAKE_TOOLCHAIN_FILE wins, then $VCPKG_ROOT, then the usual install
# locations - the presets stay machine-independent.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(_kenshin_vcpkg_roots "$ENV{VCPKG_ROOT}" "C:/vcpkg" "$ENV{HOME}/vcpkg"
                           "/opt/vcpkg" "/usr/local/vcpkg")
  foreach(_root IN LISTS _kenshin_vcpkg_roots)
    if(_root AND EXISTS "${_root}/scripts/buildsystems/vcpkg.cmake")
      set(CMAKE_TOOLCHAIN_FILE "${_root}/scripts/buildsystems/vcpkg.cmake"
          CACHE FILEPATH "vcpkg toolchain")
      break()
    endif()
  endforeach()
  if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    message(FATAL_ERROR "vcpkg not found: set VCPKG_ROOT or pass "
                        "-DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake")
  endif()
endif()

project(KENSHINWorkspace VERSION 0.1.0 LANGUAGES CXX)

# Workspace: engine (lib) / app (executable) / tools (CLI) / tests (Catch2).
# Build presets live in CMakePresets.json (vcpkg manifest; MSVC, GCC,
# Clang, AppleClang).

enable_testing()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(KenshinRuntime)
include(KenshinWarnings)

add_subdirectory(engine)
add_subdirectory(app)
add_subdirectory(tools)
add_subdirectory(tests)
