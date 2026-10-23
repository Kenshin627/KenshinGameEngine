# Run crash_probe and assert the crash handler's report:
#   - the process fails (the OS default disposition still applies),
#   - stderr has "=== FATAL" and the faulting frame's symbol.
# -DPROBE=<path to crash_probe>
execute_process(COMMAND "${PROBE}" --no-crash RESULT_VARIABLE rc0
                OUTPUT_VARIABLE out0)
if(NOT rc0 EQUAL 0 OR NOT out0 MATCHES "alive")
  message(FATAL_ERROR "crash_probe --no-crash failed (${rc0})")
endif()
execute_process(COMMAND "${PROBE}" RESULT_VARIABLE rc ERROR_VARIABLE err
                OUTPUT_VARIABLE out)
if(rc EQUAL 0)
  message(FATAL_ERROR "crash_probe did not crash (rc 0)")
endif()
if(NOT err MATCHES "=== FATAL")
  message(FATAL_ERROR "no crash report on stderr (rc ${rc}):\n${err}")
endif()
if(NOT err MATCHES "kenshinCrashProbeFault")
  message(FATAL_ERROR "crash report lacks the faulting frame:\n${err}")
endif()
message(STATUS "crash report ok (rc ${rc})")
