#pragma once

// ── 高精度计时器（QueryPerformanceCounter）─────────────────────

#include "ke/Common.h"

KE_NS_BEGIN

/// 绝对时间（秒，单调递增，进程内一致）。
double getAbsoluteSeconds();

/// 区间计时器：构造即开始，reset 归零，elapsed 取秒。
class Timer
{
public:
    Timer()
        : m_startTicks(ticksNow())
    {
    }

    void reset() { m_startTicks = ticksNow(); }

    double elapsedSeconds() const
    {
        return (ticksNow() - m_startTicks) * kInvFrequency;
    }

    double elapsedMilliseconds() const { return elapsedSeconds() * 1000.0; }

private:
    static i64 ticksNow();
    static const double kInvFrequency;

    i64 m_startTicks;
};

KE_NS_END
