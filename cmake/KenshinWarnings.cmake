# Compiler-neutral target defaults.
#
# kenshin_target_defaults(<target>): the project's warning level and
# source-encoding flags for MSVC, GCC and Clang/AppleClang alike.
# KENSHIN_NO_WARNINGS: a per-source COMPILE_OPTIONS value that silences
# vendored third-party code under every compiler (a raw "/W0" is a FILE
# NAME to GCC).

set(KENSHIN_NO_WARNINGS "$<IF:$<CXX_COMPILER_ID:MSVC>,/W0,-w>")

function(kenshin_target_defaults target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8 /MP)
    # Standard C (fopen/sscanf/getenv) is the portable API; MSVC's C4996
    # deprecation pushes the Windows-only *_s variants instead.
    target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
    # Release builds keep symbols (crash reports from shipped builds are
    # unreadable without them): /Z7 = debug info in the objects (safe with
    # /MP), /DEBUG writes the PDB, /OPT:REF,ICF restores the size
    # optimizations /DEBUG would otherwise turn off.
    target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:/Z7>)
    get_target_property(_ttype ${target} TYPE)
    if(NOT _ttype STREQUAL "STATIC_LIBRARY")
      target_link_options(${target} PRIVATE
        $<$<CONFIG:Release>:/DEBUG /OPT:REF /OPT:ICF>)
    endif()
  else()
    # Sources are UTF-8 already (GCC/Clang default); -Wall -Wextra is the
    # /W4 counterpart.
    target_compile_options(${target} PRIVATE -Wall -Wextra)
  endif()
  # Readable crash reports on Linux/macOS: backtrace_symbols() names only
  # symbols in the dynamic table.
  get_target_property(_type ${target} TYPE)
  if(_type STREQUAL "EXECUTABLE" AND NOT WIN32)
    target_link_options(${target} PRIVATE -rdynamic)
  endif()
endfunction()
