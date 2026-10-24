#include "AppArgs.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>

#include "kenshin/platform/Paths.h"

namespace demo {

Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--no-validation") == 0)
      args.validation = false;
    else if (std::strcmp(argv[i], "--stream-budget") == 0 && i + 1 < argc)
      args.streamBudgetMb = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--no-hw") == 0)
      args.noHw = true;
    else if (std::strcmp(argv[i], "--route-hw") == 0)
      args.routeHw = true;
    else if (std::strcmp(argv[i], "--force-hw") == 0)
      args.forceHw = true;
    else if (std::strcmp(argv[i], "--exposure") == 0 && i + 1 < argc)
      args.exposure = static_cast<float>(std::atof(argv[++i]));
    else if (std::strcmp(argv[i], "--env-intensity") == 0 && i + 1 < argc)
      args.envIntensity = static_cast<float>(std::atof(argv[++i]));
    else if (std::strcmp(argv[i], "--taa-sharpen") == 0 && i + 1 < argc)
      args.taaSharpen = static_cast<float>(std::atof(argv[++i]));
    else if (std::strcmp(argv[i], "--env") == 0 && i + 1 < argc)
      args.envPath = argv[++i];
    else if (std::strcmp(argv[i], "--no-auto-exposure") == 0)
      args.noAutoExposure = true;
    else if (std::strcmp(argv[i], "--exposure-bias") == 0 && i + 1 < argc)
      args.exposureBias = static_cast<float>(std::atof(argv[++i]));
    else if (std::strcmp(argv[i], "--no-emissive-pass-through") == 0)
      args.noEmissivePass = true;
    else if (std::strcmp(argv[i], "--no-sky") == 0)
      args.noSky = true;
    else if (std::strcmp(argv[i], "--sky-blur") == 0 && i + 1 < argc)
      args.skyBlur = static_cast<float>(std::atof(argv[++i]));
    else if (std::strcmp(argv[i], "--no-bloom") == 0)
      args.noBloom = true;
    else if (std::strcmp(argv[i], "--bloom-strength") == 0 && i + 1 < argc)
      args.bloomStrength = static_cast<float>(std::atof(argv[++i]));
    else if (std::strcmp(argv[i], "--emissive-strength") == 0 && i + 1 < argc)
      args.emissiveStrength = static_cast<float>(std::atof(argv[++i]));
    else if (std::strcmp(argv[i], "--hw-threshold") == 0 && i + 1 < argc)
      args.hwThresholdPx = static_cast<float>(std::atof(argv[++i]));
    else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc)
      args.frames = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc)
      std::sscanf(argv[++i], "%dx%d", &args.width, &args.height);
    else if (std::strcmp(argv[i], "--kgeo") == 0 && i + 1 < argc)
      args.kgeo = argv[++i];
    else if (std::strcmp(argv[i], "--lod-threshold") == 0 && i + 1 < argc)
      args.lodThreshold = static_cast<float>(std::atof(argv[++i]));
    else if (std::strcmp(argv[i], "--screenshot") == 0 && i + 1 < argc)
      args.screenshot = argv[++i];
    else if (std::strcmp(argv[i], "--screenshot-ui") == 0 && i + 1 < argc)
      args.screenshotUi = argv[++i];
    else if (std::strcmp(argv[i], "--virtual") == 0)
      args.virtualPath = true;
    else if (std::strcmp(argv[i], "--baseline") == 0)
      args.baseline = true;  // no-op: CPU-cut path retired at M4.4
    else if (std::strcmp(argv[i], "--checker") == 0)
      args.checker = true;
    else if (std::strcmp(argv[i], "--no-occ") == 0)
      args.noOcclusion = true;
    else if (std::strcmp(argv[i], "--no-async") == 0)
      args.noAsync = true;
    else if (std::strcmp(argv[i], "--cull-linear") == 0)
      args.cullLinear = true;
    else if (std::strcmp(argv[i], "--cull-persistent") == 0)
      args.cullPersistent = true;
    else if (std::strcmp(argv[i], "--raster-opt") == 0)
      args.rasterOpt = true;
    else if (std::strcmp(argv[i], "--raster-base") == 0)
      args.rasterBase = true;
    else if (std::strcmp(argv[i], "--raster-probe") == 0)
      args.rasterProbe = true;
    else if (std::strcmp(argv[i], "--hw-px-stats") == 0)
      args.hwPxStats = true;
    else if (std::strcmp(argv[i], "--no-shadows") == 0)
      args.noShadows = true;
    else if (std::strcmp(argv[i], "--no-taa") == 0)
      args.noTaa = true;
    else if (std::strcmp(argv[i], "--taa") == 0)
      args.forceTaa = true;
    else if (std::strcmp(argv[i], "--imposter-debug") == 0)
      args.imposterDebug = true;
    else if (std::strcmp(argv[i], "--no-imposter") == 0)
      args.imposterOff = true;
    else if (std::strcmp(argv[i], "--ground") == 0 && i + 1 < argc)
      args.ground = argv[++i];
    else if (std::strcmp(argv[i], "--mesh") == 0 && i + 1 < argc)
      args.meshes.push_back(argv[++i]);
    else if (std::strcmp(argv[i], "--view") == 0 && i + 1 < argc)
      args.viewMode = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--multi") == 0)
      args.multi = true;
    else if (std::strcmp(argv[i], "--dense") == 0)
      args.dense = true;
    else if (std::strcmp(argv[i], "--grid") == 0 && i + 1 < argc)
      args.grid = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--wall") == 0)
      args.wall = true;
    else if (std::strcmp(argv[i], "--project") == 0 && i + 1 < argc)
      args.project = argv[++i];
    else if (std::strcmp(argv[i], "--scene") == 0 && i + 1 < argc)
      args.scene = argv[++i];
    else if (std::strcmp(argv[i], "--watch-interval") == 0 && i + 1 < argc)
      args.watchInterval = static_cast<float>(std::atof(argv[++i]));
    else if (std::strcmp(argv[i], "--touch") == 0 && i + 3 < argc) {
      args.touchFrame = std::atoi(argv[++i]);
      args.touchSrc = argv[++i];
      args.touchDst = argv[++i];
    }
    else if (std::strcmp(argv[i], "--cam") == 0 && i + 1 < argc) {
      const int n = std::sscanf(argv[++i], "%f,%f,%f,%f,%f,%f", &args.camYaw,
                             &args.camPitch, &args.camDist, &args.camTarget[0],
                             &args.camTarget[1], &args.camTarget[2]);
      args.hasCam = n >= 3;  // target triple optional
    }
  }
  if (args.width <= 0 || args.height <= 0) {
    args.width = 1600;
    args.height = 900;
  }
  return args;
}

std::string findShaderRoot() {
  auto dir = kenshin::platform::executableDir();
  const std::filesystem::path candidates[] = {
      dir / ".." / ".." / ".." / ".." / "engine" / "shaders",
      std::filesystem::path("engine/shaders"),
  };
  for (const auto& candidate : candidates) {
    std::error_code ec;
    if (std::filesystem::exists(candidate / "test_atomic.slang", ec))
      return std::filesystem::weakly_canonical(candidate, ec).string();
  }
  return "engine/shaders";
}

}  // namespace demo
