// SEH crash filter for the test binary (the app's crashLogFilter
// twin): symbolized stack on any unhandled exception - the M4.2
// force-HW test SIGSEGVs and Catch2 reports only "fatal error".
#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <cstdio>
#include <filesystem>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")

static LONG WINAPI crashFilter(EXCEPTION_POINTERS* ep) {
  HANDLE proc = GetCurrentProcess();
  SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
  SymInitialize(proc, nullptr, TRUE);
  {
    wchar_t exePath[MAX_PATH]{};
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    SymSetSearchPathW(proc,
                      std::filesystem::path(exePath).parent_path().c_str());
  }
  void* frames[64]{};
  const USHORT n = CaptureStackBackTrace(0, 64, frames, nullptr);
  fprintf(stderr, "=== UNHANDLED EXCEPTION code=%08lx addr=%p\n",
          ep->ExceptionRecord->ExceptionCode,
          ep->ExceptionRecord->ExceptionAddress);
  for (USHORT i = 0; i < n; ++i) {
    char buf[sizeof(SYMBOL_INFO) + 256]{};
    auto* sym = reinterpret_cast<SYMBOL_INFO*>(buf);
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen = 255;
    DWORD64 disp = 0;
    if (SymFromAddr(proc, reinterpret_cast<DWORD64>(frames[i]), &disp, sym))
      fprintf(stderr, "  #%02u %s+%llu\n", i, sym->Name,
              static_cast<unsigned long long>(disp));
    else
      fprintf(stderr, "  #%02u %p\n", i, frames[i]);
  }
  fflush(stderr);
  return EXCEPTION_CONTINUE_SEARCH;
}

int main(int argc, char** argv) {
  SetUnhandledExceptionFilter(crashFilter);
  return Catch::Session().run(argc, argv);
}
