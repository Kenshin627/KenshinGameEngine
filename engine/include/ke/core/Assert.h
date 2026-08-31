#pragma once

// ── Assertions ──────────────────────────────────────────────────
// KE_ASSERT: debug-only (compiles to nothing under NDEBUG); logs and breaks on failure.
// KE_VERIFY: expression is always evaluated; asserts in debug (for side-effectful checks).

#include "ke/core/Log.h"

#if defined(NDEBUG)
    #define KE_ASSERT(expr)                 ((void)0)
    #define KE_ASSERT_MSG(expr, fmt, ...)   ((void)0)
#else
    #define KE_ASSERT(expr)                                                     \
        do                                                                      \
        {                                                                       \
            if (!(expr))                                                        \
            {                                                                   \
                KE_ERROR("Assertion failed: {} ({}:{})", #expr, __FILE__, __LINE__); \
                __debugbreak();                                                 \
            }                                                                   \
        } while (0)
    #define KE_ASSERT_MSG(expr, fmt, ...)                                       \
        do                                                                      \
        {                                                                       \
            if (!(expr))                                                        \
            {                                                                   \
                KE_ERROR("Assertion failed: {} — " fmt " ({}:{})",              \
                         #expr, ##__VA_ARGS__, __FILE__, __LINE__);             \
                __debugbreak();                                                 \
            }                                                                   \
        } while (0)
#endif

#define KE_VERIFY(expr)                                                         \
    do                                                                          \
    {                                                                           \
        const bool _ke_verify_result_ = static_cast<bool>(expr);                \
        if (!_ke_verify_result_)                                                \
        {                                                                       \
            KE_ERROR("Verify failed: {} ({}:{})", #expr, __FILE__, __LINE__);   \
            KE_ASSERT(false && #expr);                                          \
        }                                                                       \
    } while (0)
