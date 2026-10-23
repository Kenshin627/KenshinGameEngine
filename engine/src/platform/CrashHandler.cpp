#include "kenshin/platform/CrashHandler.h"

#include <cstdio>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
// dbghelp.h after Windows.h (it relies on its types).
#include <dbghelp.h>
#else
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#endif

KENSHIN_NS_BEGIN

namespace platform {

#ifdef _WIN32

namespace {

LONG WINAPI crashFilter(EXCEPTION_POINTERS* ep) {
  HANDLE proc = GetCurrentProcess();
  SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
  SymInitialize(proc, nullptr, TRUE);
  {
    // PDBs ship next to the executable (exported builds have no build
    // tree to point at).
    wchar_t exePath[MAX_PATH]{};
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    if (wchar_t* slash = wcsrchr(exePath, L'\\')) *slash = L'\0';
    SymSetSearchPathW(proc, exePath);
  }
  std::fprintf(stderr, "=== FATAL unhandled exception code=%08lx addr=%p\n",
               ep->ExceptionRecord->ExceptionCode,
               ep->ExceptionRecord->ExceptionAddress);
  // Walk from the FAULTING context, not this filter's own stack
  // (CaptureStackBackTrace here yields no usable frames): the first
  // frame is the instruction that faulted.
  CONTEXT ctx = *ep->ContextRecord;
  STACKFRAME64 frame{};
#if defined(_M_X64)
  const DWORD machine = IMAGE_FILE_MACHINE_AMD64;
  frame.AddrPC.Offset = ctx.Rip;
  frame.AddrFrame.Offset = ctx.Rbp;
  frame.AddrStack.Offset = ctx.Rsp;
#elif defined(_M_ARM64)
  const DWORD machine = IMAGE_FILE_MACHINE_ARM64;
  frame.AddrPC.Offset = ctx.Pc;
  frame.AddrFrame.Offset = ctx.Fp;
  frame.AddrStack.Offset = ctx.Sp;
#else
#error "unsupported Windows architecture"
#endif
  frame.AddrPC.Mode = AddrModeFlat;
  frame.AddrFrame.Mode = AddrModeFlat;
  frame.AddrStack.Mode = AddrModeFlat;
  HANDLE thread = GetCurrentThread();
  for (int i = 0; i < 64; ++i) {
    if (!StackWalk64(machine, proc, thread, &frame, &ctx, nullptr,
                     SymFunctionTableAccess64, SymGetModuleBase64, nullptr))
      break;
    const DWORD64 pc = frame.AddrPC.Offset;
    if (pc == 0) break;
    char buf[sizeof(SYMBOL_INFO) + 256]{};
    auto* sym = reinterpret_cast<SYMBOL_INFO*>(buf);
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen = 255;
    DWORD64 disp = 0;
    if (SymFromAddr(proc, pc, &disp, sym))
      std::fprintf(stderr, "  #%02d %s+%llu\n", i, sym->Name,
                   static_cast<unsigned long long>(disp));
    else
      std::fprintf(stderr, "  #%02d %016llx\n", i,
                   static_cast<unsigned long long>(pc));
  }
  std::fflush(stderr);
  return EXCEPTION_CONTINUE_SEARCH;  // the OS default continues
}

}  // namespace

void installCrashHandler() { SetUnhandledExceptionFilter(crashFilter); }

#else  // POSIX

namespace {

// Async-signal-safe output only: write(2) and backtrace_symbols_fd.
void writeStr(const char* s) {
  const ssize_t ignored = ::write(STDERR_FILENO, s, std::strlen(s));
  (void)ignored;
}

const char* signalName(int sig) {
  switch (sig) {
    case SIGSEGV:
      return "SIGSEGV";
    case SIGBUS:
      return "SIGBUS";
    case SIGILL:
      return "SIGILL";
    case SIGFPE:
      return "SIGFPE";
    case SIGABRT:
      return "SIGABRT";
    default:
      return "signal";
  }
}

void onFatalSignal(int sig) {
  writeStr("=== FATAL ");
  writeStr(signalName(sig));
  writeStr("\n");
  void* frames[64];
  const int n = backtrace(frames, 64);
  backtrace_symbols_fd(frames, n, STDERR_FILENO);
  // Restore the default disposition and re-raise: core dump / exit code
  // behave as if no handler were installed.
  signal(sig, SIG_DFL);
  raise(sig);
}

}  // namespace

void installCrashHandler() {
  // Alternate stack: a stack overflow's SIGSEGV still has room to run.
  static char altStack[64 * 1024];
  stack_t ss{};
  ss.ss_sp = altStack;
  ss.ss_size = sizeof(altStack);
  sigaltstack(&ss, nullptr);
  // backtrace() may allocate on its first call - warm it up outside the
  // signal handler.
  void* warm[1];
  backtrace(warm, 1);
  struct sigaction sa {};
  sa.sa_handler = onFatalSignal;
  sa.sa_flags = SA_ONSTACK | SA_RESETHAND;
  sigemptyset(&sa.sa_mask);
  for (int sig : {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT})
    sigaction(sig, &sa, nullptr);
}

#endif

}  // namespace platform

KENSHIN_NS_END
