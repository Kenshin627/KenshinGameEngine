#include "ke/Version.h"

#include <cstdio>

#ifndef KE_VERSION_MAJOR
#define KE_VERSION_MAJOR 0
#endif
#ifndef KE_VERSION_MINOR
#define KE_VERSION_MINOR 0
#endif
#ifndef KE_VERSION_PATCH
#define KE_VERSION_PATCH 0
#endif
#ifndef KE_GIT_SHA
#define KE_GIT_SHA "unknown"
#endif

KE_NS_BEGIN

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
                      KE_VERSION_MAJOR, KE_VERSION_MINOR, KE_VERSION_PATCH,
                      KE_GIT_SHA);
        std::snprintf(s_banner, sizeof(s_banner),
                      "KEngine %s (%s build)",
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
#if defined(KE_BUILD_DEBUG) || defined(_DEBUG)
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

KE_NS_END
