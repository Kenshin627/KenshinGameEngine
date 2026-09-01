// Smoke tests: basic arithmetic / version strings / log formatting / timer (S0.2 acceptance).

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string>

#include "kenshin/Version.h"
#include "kenshin/core/Log.h"
#include "kenshin/core/Timer.h"

TEST_CASE("arithmetic smoke", "[smoke]") { REQUIRE(1 + 1 == 2); }

TEST_CASE("version banner is populated", "[smoke][version]") {
  REQUIRE(std::strlen(kenshin::getVersionString()) > 0);
  const std::string banner = kenshin::getBanner();
  REQUIRE(banner.starts_with("KENSHIN "));
  REQUIRE(banner.find(" build)") != std::string::npos);
}

TEST_CASE("log formats and falls back safely", "[smoke][log]") {
  // Before init: must not crash, falls back to the default logger
  KENSHIN_INFO("pre-init message {}", 0);

  kenshin::log::init();
  KENSHIN_INFO("formatted message {}", 42);
  KENSHIN_WARN("warn with {} {}", "two", 2);

  // The log file should exist at CWD/log/kenshin.log
  REQUIRE(std::filesystem::exists("log/kenshin.log"));

  kenshin::log::shutdown();
}

TEST_CASE("timer is monotonic", "[smoke][timer]") {
  const double t0 = kenshin::getAbsoluteSeconds();
  kenshin::Timer t;
  double prev = 0.0;
  for (int i = 0; i < 1000; ++i) {
    const double e = t.elapsedSeconds();
    REQUIRE(e >= prev);
    prev = e;
  }
  REQUIRE(prev >= 0.0);
  REQUIRE(kenshin::getAbsoluteSeconds() >= t0);
  CHECK_FALSE(std::isnan(prev));
}
