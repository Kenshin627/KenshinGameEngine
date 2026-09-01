#pragma once

// ── Assertions ──────────────────────────────────────────────────
// KENSHIN_ASSERT: debug-only (compiles to nothing under NDEBUG); logs and breaks on failure.
// KENSHIN_VERIFY: expression is always evaluated; asserts in debug (for side-effectful checks).

#include "kenshin/core/Log.h"

#if defined(NDEBUG)
#define KENSHIN_ASSERT(expr) ((void)0)
#define KENSHIN_ASSERT_MSG(expr, fmt, ...) ((void)0)
#else
#define KENSHIN_ASSERT(expr)                                         \
  do {                                                               \
    if (!(expr)) {                                                   \
      KENSHIN_ERROR("Assertion failed: {} ({}:{})", #expr, __FILE__, \
                    __LINE__);                                       \
      __debugbreak();                                                \
    }                                                                \
  } while (0)
#define KENSHIN_ASSERT_MSG(expr, fmt, ...)                           \
  do {                                                               \
    if (!(expr)) {                                                   \
      KENSHIN_ERROR("Assertion failed: {} — " fmt " ({}:{})", #expr, \
                    ##__VA_ARGS__, __FILE__, __LINE__);              \
      __debugbreak();                                                \
    }                                                                \
  } while (0)
#endif

#define KENSHIN_VERIFY(expr)                                                 \
  do {                                                                       \
    const bool _kenshin_verify_result_ = static_cast<bool>(expr);            \
    if (!_kenshin_verify_result_) {                                          \
      KENSHIN_ERROR("Verify failed: {} ({}:{})", #expr, __FILE__, __LINE__); \
      KENSHIN_ASSERT(false && #expr);                                        \
    }                                                                        \
  } while (0)
