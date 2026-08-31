#pragma once

// ── 断言 ─────────────────────────────────────────────────────────
// KE_ASSERT：仅 Debug 生效（NDEBUG 下编译为空），失败时打日志并断点。
// KE_VERIFY：任何配置都求值表达式，Debug 下附带断言（用于"必须执行副作用"的检查）。

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
