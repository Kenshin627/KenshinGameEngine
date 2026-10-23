#pragma once

// ── Paths — portable process/filesystem locations ───────────────────
// The platform layer's answer to "where am I running from" (pipeline
// cache, imgui.ini, the validation layer shipped next to the exe) and a
// startup-only environment setter. SDL3 underneath; no OS headers leak.

#include <filesystem>
#include <string>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace platform {

/// Directory holding the running executable (macOS bundles: the
/// bundle's Resources directory, SDL's convention). Empty on failure.
std::filesystem::path executableDir();

/// Set a process environment variable. Only call before worker threads
/// start (POSIX setenv is not thread-safe) - e.g. VK_LAYER_PATH ahead of
/// Vulkan instance creation.
bool setEnvironmentVariable(const std::string& name, const std::string& value);

}  // namespace platform

KENSHIN_NS_END
