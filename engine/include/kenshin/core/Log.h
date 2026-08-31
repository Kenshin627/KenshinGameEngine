#pragma once

// ── Logging (thin wrapper over spdlog) ──────────────────────────
// Colored console + rotating file (log/kenshin.log, 5MB x 3).
// Usage: call kenshin::log::init() once, then KENSHIN_INFO/KENSHIN_WARN/KENSHIN_ERROR(fmt, args...).

#include "kenshin/Common.h"

#include <spdlog/spdlog.h>

KENSHIN_NS_BEGIN
namespace log
{
    /// Initialize logging (idempotent): console + log/kenshin.log.
    /// @param logDir log directory, defaults to "log".
    void init(const char* logDir = "log");

    /// Release log resources (optional; safe to skip).
    void shutdown();

    /// Current logger (never null; falls back to spdlog's default before init).
    spdlog::logger* logger();
}
KENSHIN_NS_END

#define KENSHIN_TRACE(...) ::kenshin::log::logger()->trace(__VA_ARGS__)
#define KENSHIN_INFO(...)  ::kenshin::log::logger()->info(__VA_ARGS__)
#define KENSHIN_WARN(...)  ::kenshin::log::logger()->warn(__VA_ARGS__)
#define KENSHIN_ERROR(...) ::kenshin::log::logger()->error(__VA_ARGS__)
