#include "kenshin/Version.h"

#include <cstdio>

#ifndef KENSHIN_VERSION_MAJOR
#define KENSHIN_VERSION_MAJOR 0
#endif
#ifndef KENSHIN_VERSION_MINOR
#define KENSHIN_VERSION_MINOR 0
#endif
#ifndef KENSHIN_VERSION_PATCH
#define KENSHIN_VERSION_PATCH 0
#endif
#ifndef KENSHIN_GIT_SHA
#define KENSHIN_GIT_SHA "unknown"
#endif

KENSHIN_NS_BEGIN

namespace
{
    char s_versionString[64];
    char s_banner[96];
    bool s_initialized = false;

    void ensureInitialized()
    {
        if (s_initialized)
            return;
        std::snprintf(s_versionString, sizeof(s_versionString),
                      "%d.%d.%d+%s",
                      KENSHIN_VERSION_MAJOR, KENSHIN_VERSION_MINOR, KENSHIN_VERSION_PATCH,
                      KENSHIN_GIT_SHA);
        std::snprintf(s_banner, sizeof(s_banner),
                      "KENSHIN %s (%s build)",
                      s_versionString, getBuildConfig());
        s_initialized = true;
    }
} // namespace

const char* getVersionString()
{
    ensureInitialized();
    return s_versionString;
}

const char* getBuildConfig()
{
#if defined(KENSHIN_BUILD_DEBUG) || defined(_DEBUG)
    return "Debug";
#else
    return "Release";
#endif
}

const char* getBanner()
{
    ensureInitialized();
    return s_banner;
}

KENSHIN_NS_END
