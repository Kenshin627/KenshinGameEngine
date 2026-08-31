#include "ke/core/Timer.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

KE_NS_BEGIN

namespace
{
    const double kFrequencyInv = [] {
        LARGE_INTEGER freq {};
        QueryPerformanceFrequency(&freq);
        return 1.0 / static_cast<double>(freq.QuadPart);
    }();
} // namespace

const double Timer::kInvFrequency = kFrequencyInv;

double getAbsoluteSeconds()
{
    LARGE_INTEGER counter {};
    QueryPerformanceCounter(&counter);
    return static_cast<double>(counter.QuadPart) * kFrequencyInv;
}

i64 Timer::ticksNow()
{
    LARGE_INTEGER counter {};
    QueryPerformanceCounter(&counter);
    return counter.QuadPart;
}

KE_NS_END
