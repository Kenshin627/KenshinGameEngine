# GPU render regression net (RHI v2 item ②): run the demo headless and
# byte-compare the screenshot against a reference checked into tests/data.
#
# Called from tests/CMakeLists.txt via add_test with:
#   -DDEMO=<path to KENSHINDemo>  -DBUILDER=<path to kgeo_builder>
#   -DREF=<reference bmp>         -DOUT=<output bmp>
#   -DEXEDIR=<exe dir: imgui.ini lives there>
#   [-DKGEO=1 -DPROC=knot -DLOD=1]
#   [-DNOHW=1  render with --no-hw]
#
# The NOHW variant anchors the HW-raster path: routing (default) and
# --no-hw render byte-identically, so BOTH ctests compare against the
# SAME reference - a regression that only touches the HW draw trips
# the routed test while the no-hw test stays green (and a reference
# regenerated on a broken build fails one of the two).
#
# Reference regeneration (after an INTENDED rendering change):
#   1. cmake --build --preset windows-msvc-release
#   2. rm build/windows-msvc-release/app/Release/imgui.ini
#   3. run the demo command shown in the failure message
#   4. copy the output bmp over tests/data/render_<case>_ref.bmp
# Reference updates go through review like any code change - the diff
# being binary is the point: any unintended pixel change fails this test.

file(REMOVE "${EXEDIR}/imgui.ini")

# ALBEDO (M8.2): the texture reference goes through the BUILDER (-
# -albedo) so the .kgeo carries a material table with a real texture;
# the builder copies the BMP next to the generated .kgeo. NEVER append
# --albedo to the demo args: the demo has no such flag and silently
# drops unknown flags (that masking is why the first version of this
# test passed while exercising nothing).
if(ALBEDO)
  set(BUILDER_ARGS --albedo "${ALBEDO}")
else()
  set(BUILDER_ARGS)
endif()

if(KGEO)
  set(KGEO_FILE "${OUT}.kgeo")
  execute_process(
    COMMAND "${BUILDER}" --proc "${PROC}" ${BUILDER_ARGS} -o "${KGEO_FILE}"
    RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "kgeo_builder --proc ${PROC} failed (${rc})")
  endif()
  set(EXTRA_ARGS --kgeo "${KGEO_FILE}" --lod-threshold "${LOD}")
else()
  set(EXTRA_ARGS)
endif()

if(NOHW)
  list(APPEND EXTRA_ARGS --no-hw)
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

# TOLERANCE (optional): pixel budget for run-to-run differences. v5
# quantization snaps positions onto a global grid, so near-tie depths
# tie far more often and the VisBuffer atomic max's write order flips a
# handful of pixels per run. Default 0 = the old byte-exact semantics
# (the baseline case); the quantized kgeo case passes a small budget.
if(NOT DEFINED TOLERANCE)
  set(TOLERANCE 0)
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files "${OUT}" "${REF}"
  RESULT_VARIABLE rc)
if((NOT rc EQUAL 0) AND (TOLERANCE GREATER 0))
  # Byte-pair (BGR) diff count over same-size BMPs.
  file(READ "${OUT}" outHex HEX)
  file(READ "${REF}" refHex HEX)
  string(LENGTH "${outHex}" outLen)
  string(LENGTH "${refHex}" refLen)
  if(NOT outLen STREQUAL refLen)
    message(FATAL_ERROR "render regression: size mismatch.")
  endif()
  set(diff 0)
  set(i 54)  # skip the BMP header (14) + info (40)
  while(i LESS outLen)
    string(SUBSTRING "${outHex}" ${i} 6 outPx)
    string(SUBSTRING "${refHex}" ${i} 6 refPx)
    if(NOT outPx STREQUAL refPx)
      math(EXPR diff "${diff} + 1")
      if(diff GREATER ${TOLERANCE})
        message(FATAL_ERROR
          "render regression: over ${diff} pixels differ (budget ${TOLERANCE}).")
      endif()
    endif()
    math(EXPR i "${i} + 6")
  endwhile()
elseif(NOT rc EQUAL 0)
  message(FATAL_ERROR
    "render regression: screenshot differs from reference.\n"
    "  got:      ${OUT}\n"
    "  expected: ${REF}\n"
    "If this change is INTENDED, regenerate the reference (see the "
    "instructions at the top of tests/render_regression.cmake) and commit "
    "it together with the code change.")
endif()
