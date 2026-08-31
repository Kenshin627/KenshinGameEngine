#pragma once

// ── Logging (thin wrapper over spdlog) ──────────────────────────
// Colored console + rotating file (log/ke.log, 5MB x 3).
// Usage: call ke::log::init() once, then KE_INFO/KE_WARN/KE_ERROR(fmt, args...).

#include "ke/Common.h"

#include <spdlog/spdlog.h>

KE_NS_BEGIN
namespace log
{
    /// Initialize logging (idempotent): console + log/ke.log.
    /// @param logDir log directory, defaults to "log".
    void init(const char* logDir = "log");

    /// Release log resources (optional; safe to skip).
    void shutdown();

    /// Current logger (never null; falls back to spdlog's default before init).
    spdlog::logger* logger();
}
KE_NS_END

#define KE_TRACE(...) ::ke::log::logger()->trace(__VA_ARGS__)
#define KE_INFO(...)  ::ke::log::logger()->info(__VA_ARGS__)
#define KE_WARN(...)  ::ke::log::logger()->warn(__VA_ARGS__)
#define KE_ERROR(...) ::ke::log::logger()->error(__VA_ARGS__)
