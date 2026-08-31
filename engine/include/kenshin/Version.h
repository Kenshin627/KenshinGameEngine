#pragma once

// ── Version info ────────────────────────────────────────────────
// Macros are injected by engine/CMakeLists.txt: KENSHIN_VERSION_MAJOR/MINOR/PATCH,
// KENSHIN_GIT_SHA, KENSHIN_BUILD_DEBUG.

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

/// Version string in the form "0.1.0+ab12cd3" (semver + git short SHA).
const char* getVersionString();

/// "Debug" / "Release"
const char* getBuildConfig();

/// Startup banner: `KENSHIN 0.1.0+ab12cd3 (Debug build)`
const char* getBanner();

KENSHIN_NS_END
