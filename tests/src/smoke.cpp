// Smoke tests: basic arithmetic / version strings / log formatting / timer (S0.2 acceptance).

#include <catch2/catch_test_macros.hpp>

#include "ke/Version.h"
#include "ke/core/Log.h"
#include "ke/core/Timer.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>

TEST_CASE("arithmetic smoke", "[smoke]")
{
    REQUIRE(1 + 1 == 2);
}

TEST_CASE("version banner is populated", "[smoke][version]")
{
    REQUIRE(std::strlen(ke::getVersionString()) > 0);
    const std::string banner = ke::getBanner();
    REQUIRE(banner.starts_with("KEngine "));
    REQUIRE(banner.find(" build)") != std::string::npos);
}

TEST_CASE("log formats and falls back safely", "[smoke][log]")
{
    // Before init: must not crash, falls back to the default logger
    KE_INFO("pre-init message {}", 0);

    ke::log::init();
    KE_INFO("formatted message {}", 42);
    KE_WARN("warn with {} {}", "two", 2);

    // The log file should exist at CWD/log/ke.log
    REQUIRE(std::filesystem::exists("log/ke.log"));

    ke::log::shutdown();
}

TEST_CASE("timer is monotonic", "[smoke][timer]")
{
    const double t0 = ke::getAbsoluteSeconds();
    ke::Timer t;
    double prev = 0.0;
    for (int i = 0; i < 1000; ++i)
    {
        const double e = t.elapsedSeconds();
        REQUIRE(e >= prev);
        prev = e;
    }
    REQUIRE(prev >= 0.0);
    REQUIRE(ke::getAbsoluteSeconds() >= t0);
    CHECK_FALSE(std::isnan(prev));
}
