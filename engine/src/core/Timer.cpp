#include "kenshin/core/Timer.h"

#include <chrono>

KENSHIN_NS_BEGIN

// steady_clock: monotonic on every platform (QueryPerformanceCounter on
// Windows, clock_gettime(CLOCK_MONOTONIC) on Linux/macOS).
using Clock = std::chrono::steady_clock;

const double Timer::kInvFrequency =
    static_cast<double>(Clock::period::num) /
    static_cast<double>(Clock::period::den);

double getAbsoluteSeconds() {
  return std::chrono::duration<double>(Clock::now().time_since_epoch())
      .count();
}

i64 Timer::ticksNow() {
  return static_cast<i64>(Clock::now().time_since_epoch().count());
}

KENSHIN_NS_END
