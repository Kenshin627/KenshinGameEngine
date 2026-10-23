// Test runner main: the engine's crash handler gives a symbolized stack
// on fatal errors (Catch2 alone reports only "fatal error" - the M4.2
// force-HW test SIGSEGV was triaged this way).
#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>

#include "kenshin/platform/CrashHandler.h"

int main(int argc, char** argv) {
  kenshin::platform::installCrashHandler();
  return Catch::Session().run(argc, argv);
}
