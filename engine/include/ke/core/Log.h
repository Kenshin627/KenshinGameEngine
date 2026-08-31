#pragma once

// ── 日志（spdlog 薄封装）────────────────────────────────────────
// 彩色控制台 + 滚动文件（log/ke.log，5MB × 3 轮转）。
// 用法：ke::log::init() 一次，之后 KE_INFO/KE_WARN/KE_ERROR(fmt, args...)。

#include "ke/Common.h"

#include <spdlog/spdlog.h>

KE_NS_BEGIN
namespace log
{
    /// 初始化日志（幂等）：控制台 + log/ke.log。
    /// @param logDir 日志目录，默认 "log"。
    void init(const char* logDir = "log");

    /// 释放日志资源（程序退出前调用；不调用也能安全收尾）。
    void shutdown();

    /// 当前 logger（永不为 null；未 init 时返回 spdlog 默认 logger）。
    spdlog::logger* logger();
}
KE_NS_END

#define KE_TRACE(...) ::ke::log::logger()->trace(__VA_ARGS__)
#define KE_INFO(...)  ::ke::log::logger()->info(__VA_ARGS__)
#define KE_WARN(...)  ::ke::log::logger()->warn(__VA_ARGS__)
#define KE_ERROR(...) ::ke::log::logger()->error(__VA_ARGS__)
