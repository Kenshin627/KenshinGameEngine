# Project-scene render tests (roadmap 1.3d): KENSHINDemo --project on a
# fresh copy of tests/data/project (the asset database writes its cache
# and metas inside the project, never into the source tree).
#
#   -DDEMO=<KENSHINDemo>  -DBMPDIFF=<bmpdiff>  -DPROJECT=<tests/data/project>
#   -DOUT=<output bmp>    -DEXEDIR=<exe dir: imgui.ini lives there>
#   [-DTOLERANCE=<px>]
#   Render:  -DREF=<reference bmp>
#   Reload:  -DEDIT_SRC=<file> -DEDIT_DST=<project-relative path>
#            The run rewrites EDIT_DST with EDIT_SRC at frame 8 (polling
#            every frame) and must end on the SAME image as a cold start
#            of the edited project - hot reload == a fresh load.
#
# Every run fails on a validation error. Reference regeneration as in
# render_regression.cmake (copy the output bmp over the reference).

file(REMOVE "${EXEDIR}/imgui.ini")
foreach(var DEMO BMPDIFF PROJECT OUT REF EDIT_SRC)
  if(${var})
    get_filename_component(${var} "${${var}}" ABSOLUTE)
  endif()
endforeach()
if(NOT DEFINED TOLERANCE)
  set(TOLERANCE 0)
endif()
get_filename_component(OUT_DIR "${OUT}" DIRECTORY)
if(EXISTS "${OUT_DIR}/env.hdr")
  message(FATAL_ERROR "${OUT_DIR}/env.hdr would change the test's IBL")
endif()

# 16 frames, TAA off (the swap lands mid-run; with TAA the history of
# the old frames would differ from a cold start's), fixed camera.
set(COMMON_ARGS --frames 16 --size 1280x720 --cam 30,20,3.2,0,0,0.4 --no-taa
                --watch-interval 0)

# run(<name> <project copy> <screenshot> [extra args...])
function(run_demo name dir shot)
  execute_process(
    COMMAND "${DEMO}" --project "${dir}" ${COMMON_ARGS} ${ARGN} --screenshot "${shot}"
    WORKING_DIRECTORY "${OUT_DIR}"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE log
    ERROR_VARIABLE err)
  file(WRITE "${shot}.log" "${log}${err}")
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "${name}: KENSHINDemo exited with ${rc} (log: ${shot}.log)")
  endif()
  if("${log}${err}" MATCHES "\\[error\\] \\[vulkan\\]\\[validation\\]")
    message(FATAL_ERROR "${name}: validation errors (log: ${shot}.log)")
  endif()
  if(NOT EXISTS "${shot}")
    message(FATAL_ERROR "${name}: screenshot not written: ${shot}")
  endif()
  set(RUN_LOG "${log}${err}" PARENT_SCOPE)
endfunction()

function(fresh_copy dir)
  file(REMOVE_RECURSE "${dir}")
  file(COPY "${PROJECT}/" DESTINATION "${dir}")
endfunction()

function(compare got want what)
  set(diff "${got}.diff.bmp")
  file(REMOVE "${diff}")
  execute_process(
    COMMAND "${BMPDIFF}" "${got}" "${want}" --tolerance "${TOLERANCE}" --diff "${diff}"
    RESULT_VARIABLE rc OUTPUT_VARIABLE o ERROR_VARIABLE e)
  string(STRIP "${o}${e}" report)
  if(rc EQUAL 0)
    message(STATUS "${report}")
  elseif(rc EQUAL 1)
    message(FATAL_ERROR "${what}:\n  ${report}\n  got:      ${got}\n  expected: ${want}\n"
                        "  diff:     ${diff}")
  else()
    message(FATAL_ERROR "bmpdiff failed (${rc}): ${report}")
  endif()
endfunction()

set(LIVE "${OUT}.project")
fresh_copy("${LIVE}")
if(NOT EDIT_SRC)
  run_demo(render "${LIVE}" "${OUT}")
  compare("${OUT}" "${REF}" "project render differs from the reference")
  return()
endif()

# The hot-reload run.
run_demo(reload "${LIVE}" "${OUT}" --touch 8 "${EDIT_SRC}" "${LIVE}/${EDIT_DST}")
if(NOT RUN_LOG MATCHES "\\[hotreload\\] #1:")
  message(FATAL_ERROR "reload: no hot reload was applied (log: ${OUT}.log)")
endif()
if(RUN_LOG MATCHES "\\[hotreload\\] #1:[^\n]* [1-9][0-9]* failed")
  message(FATAL_ERROR "reload: the re-import failed (log: ${OUT}.log)")
endif()
string(REGEX MATCH "\\[hotreload\\] #1:[^\n]*" applied "${RUN_LOG}")
message(STATUS "${applied}")

# The cold start of the edited project.
set(COLD "${OUT}.cold")
fresh_copy("${COLD}")
file(COPY_FILE "${EDIT_SRC}" "${COLD}/${EDIT_DST}")
run_demo(cold "${COLD}" "${OUT}.cold.bmp")
compare("${OUT}" "${OUT}.cold.bmp" "the hot-reloaded frame differs from a cold start")
