#pragma once

// ── Version info ────────────────────────────────────────────────
// Macros are injected by engine/CMakeLists.txt: KE_VERSION_MAJOR/MINOR/PATCH,
// KE_GIT_SHA, KE_BUILD_DEBUG.

#include "ke/Common.h"

KE_NS_BEGIN

/// Version string in the form "0.1.0+ab12cd3" (semver + git short SHA).
const char* getVersionString();

/// "Debug" / "Release"
const char* getBuildConfig();

/// Startup banner: `KEngine 0.1.0+ab12cd3 (Debug build)`
const char* getBanner();

KE_NS_END
