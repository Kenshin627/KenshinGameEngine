#pragma once

// ── High-resolution timer (QueryPerformanceCounter) ────────────

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

/// Absolute time in seconds (monotonic, process-wide consistent).
double getAbsoluteSeconds();

/// Interval timer: starts on construction, reset() to zero, elapsed*() to read.
class Timer {
 public:
  Timer() : m_startTicks(ticksNow()) {}

  void reset() { m_startTicks = ticksNow(); }

  double elapsedSeconds() const {
    return (ticksNow() - m_startTicks) * kInvFrequency;
  }

  double elapsedMilliseconds() const { return elapsedSeconds() * 1000.0; }

 private:
  static i64 ticksNow();
  static const double kInvFrequency;

  i64 m_startTicks;
};

KENSHIN_NS_END
