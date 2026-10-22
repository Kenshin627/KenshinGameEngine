# GPU render regression net (RHI v2 item ②): run the demo headless and
# byte-compare the screenshot against a reference checked into tests/data.
#
# Called from tests/CMakeLists.txt via add_test with:
#   -DDEMO=<path to KENSHINDemo>  -DBUILDER=<path to kgeo_builder>
#   -DBMPDIFF=<path to bmpdiff>
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
#   3. run ctest (or this script) - the failure names the output bmp
#      and a .diff.bmp (differing pixels in red, |delta| x 8)
#   4. copy the output bmp over tests/data/render_<case>_ref.bmp
# Reference updates go through review like any code change - the diff
# being binary is the point: any unintended pixel change fails this test.

file(REMOVE "${EXEDIR}/imgui.ini")

# Absolute paths: the demo runs in OUT's directory (see below), so a
# relative path from a manual `cmake -P` run must not shift.
foreach(var DEMO BUILDER BMPDIFF OUT REF ALBEDO)
  if(${var})
    get_filename_component(${var} "${${var}}" ABSOLUTE)
  endif()
endforeach()

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

# Pinned working directory: the demo falls back to ./env.hdr for its
# IBL, so a run from a directory holding one (the repo root does) would
# render - and regenerate references - under a different environment.
get_filename_component(OUT_DIR "${OUT}" DIRECTORY)
if(EXISTS "${OUT_DIR}/env.hdr")
  message(FATAL_ERROR "${OUT_DIR}/env.hdr would change the test's IBL")
endif()
execute_process(
  COMMAND "${DEMO}" --frames 6 --size 1280x720 ${EXTRA_ARGS}
          --screenshot "${OUT}"
  WORKING_DIRECTORY "${OUT_DIR}"
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

# bmpdiff (tools/bmpdiff): pixel count + max delta + bbox in well under
# a second, and a diff image next to the output on mismatch. (The old
# per-pixel CMake hex loop took 10+ minutes on a differing frame.)
set(DIFF_IMAGE "${OUT}.diff.bmp")
file(REMOVE "${DIFF_IMAGE}")
execute_process(
  COMMAND "${BMPDIFF}" "${OUT}" "${REF}" --tolerance "${TOLERANCE}"
          --diff "${DIFF_IMAGE}"
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE diffOut
  ERROR_VARIABLE diffErr)
string(STRIP "${diffOut}${diffErr}" diffReport)
if(rc EQUAL 0)
  message(STATUS "${diffReport}")
elseif(rc EQUAL 1)
  message(FATAL_ERROR
    "render regression: screenshot differs from reference.
"
    "  ${diffReport}
"
    "  got:      ${OUT}
"
    "  expected: ${REF}
"
    "  diff:     ${DIFF_IMAGE}
"
    "If this change is INTENDED, regenerate the reference (see the "
    "instructions at the top of tests/render_regression.cmake) and commit "
    "it together with the code change.")
else()
  message(FATAL_ERROR "render regression: bmpdiff failed (${rc}): "
                      "${diffReport}")
endif()
