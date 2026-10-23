#pragma once

// ── CrashHandler — symbolized stack on fatal errors ─────────────────
// installCrashHandler() makes a crash print "=== FATAL <reason>" and a
// stack trace to stderr before the process dies (then the OS default
// takes over: crash dialog / core dump / exit code are unchanged).
//   Windows: unhandled SEH exceptions, dbghelp symbolization (PDBs next
//            to the executable).
//   Linux/macOS: SIGSEGV/SIGBUS/SIGILL/SIGFPE/SIGABRT, backtrace() on an
//            alternate signal stack (stack overflows still report).
// Install once, early in main().

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace platform {

void installCrashHandler();

}  // namespace platform

KENSHIN_NS_END
