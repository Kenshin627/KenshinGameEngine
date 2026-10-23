// Crash handler probe (roadmap 0.2): installs the engine crash handler
// and faults on purpose; tests/crash/check_crash.cmake asserts the
// report. A noinline frame with a distinctive name must show up in the
// symbolized stack.
#include <cstdio>
#include <cstring>

#include "kenshin/platform/CrashHandler.h"

#if defined(_MSC_VER)
#define KENSHIN_NOINLINE __declspec(noinline)
#else
#define KENSHIN_NOINLINE __attribute__((noinline))
#endif

volatile int* g_null = nullptr;

// External linkage: POSIX backtrace() names only dynamic symbols
// (executables link with -rdynamic - see kenshin_target_defaults).
KENSHIN_NOINLINE void kenshinCrashProbeFault() {
  *g_null = 42;  // access violation / SIGSEGV
}

int main(int argc, char** argv) {
  kenshin::platform::installCrashHandler();
  if (argc > 1 && std::strcmp(argv[1], "--no-crash") == 0) {
    std::puts("crash_probe: alive");
    return 0;
  }
  kenshinCrashProbeFault();
  return 0;
}
