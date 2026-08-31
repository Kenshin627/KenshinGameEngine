#pragma once

// ── 版本信息 ────────────────────────────────────────────────────
// 宏由 engine/CMakeLists.txt 注入：KE_VERSION_MAJOR/MINOR/PATCH、
// KE_GIT_SHA、KE_BUILD_DEBUG。

#include "ke/Common.h"

KE_NS_BEGIN

/// "0.1.0+ab12cd3" 形式的版本串（版本号 + git 短 SHA）
const char* getVersionString();

/// "Debug" / "Release"
const char* getBuildConfig();

/// 启动横幅：`KEngine 0.1.0+ab12cd3 (Debug build)`
const char* getBanner();

KE_NS_END
