# kenshin_deploy_runtime(<target>): put the runtime libraries an executable
# loads DYNAMICALLY next to it after each build.
#
# vcpkg's app-local deployment only follows the import table; Slang's
# compiler backend (slang-compiler) is loaded by slang at run time, so it
# is copied explicitly. Linux/macOS find them
# through the vcpkg lib directory on the build RPATH.
function(kenshin_deploy_runtime target)
  if(NOT WIN32)
    return()
  endif()
  set(_bin "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin")
  # SPIR-V generation needs only these two. slang-llvm (105 MB, CPU/LLVM
  # targets), slang-glslang, slang-glsl-module and slang-rt are never
  # loaded for SPIR-V output - shipping them would bloat every export.
  set(_slang_dlls "${_bin}/slang.dll" "${_bin}/slang-compiler.dll")
  foreach(_dll IN LISTS _slang_dlls)
    if(NOT EXISTS "${_dll}")
      message(FATAL_ERROR "kenshin_deploy_runtime: missing ${_dll}")
    endif()
  endforeach()
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_slang_dlls}
            $<TARGET_FILE_DIR:${target}>
    COMMENT "Deploying Slang runtime libraries next to ${target}")
endfunction()
