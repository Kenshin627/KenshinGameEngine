# GPU render regression net (RHI v2 item ②): run the demo headless and
# byte-compare the screenshot against a reference checked into tests/data.
#
# Called from tests/CMakeLists.txt via add_test with:
#   -DDEMO=<path to KENSHINDemo>  -DBUILDER=<path to kgeo_builder>
#   -DREF=<reference bmp>         -DOUT=<output bmp>
#   -DEXEDIR=<exe dir: imgui.ini lives there>
#   [-DKGEO=1 -DPROC=knot -DLOD=1]
#
# Reference regeneration (after an INTENDED rendering change):
#   1. cmake --build --preset windows-msvc-release
#   2. rm build/windows-msvc-release/app/Release/imgui.ini
#   3. run the demo command shown in the failure message
#   4. copy the output bmp over tests/data/render_<case>_ref.bmp
# Reference updates go through review like any code change - the diff
# being binary is the point: any unintended pixel change fails this test.

file(REMOVE "${EXEDIR}/imgui.ini")

if(KGEO)
  set(KGEO_FILE "${OUT}.kgeo")
  execute_process(
    COMMAND "${BUILDER}" --proc "${PROC}" -o "${KGEO_FILE}"
    RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "kgeo_builder --proc ${PROC} failed (${rc})")
  endif()
  set(EXTRA_ARGS --kgeo "${KGEO_FILE}" --lod-threshold "${LOD}")
else()
  set(EXTRA_ARGS)
endif()

execute_process(
  COMMAND "${DEMO}" --frames 6 --size 1280x720 ${EXTRA_ARGS}
          --screenshot "${OUT}"
  RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "KENSHINDemo exited with ${rc}")
endif()

if(NOT EXISTS "${OUT}")
  message(FATAL_ERROR "screenshot not written: ${OUT}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files "${OUT}" "${REF}"
  RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR
    "render regression: screenshot differs from reference.\n"
    "  got:      ${OUT}\n"
    "  expected: ${REF}\n"
    "If this change is INTENDED, regenerate the reference (see the "
    "instructions at the top of tests/render_regression.cmake) and commit "
    "it together with the code change.")
endif()
