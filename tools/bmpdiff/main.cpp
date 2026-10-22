// ── bmpdiff — pixel diff for the render regression net ──────────────
// Replaces render_regression.cmake's hex-string loop (a CMake SUBSTRING
// per pixel: 10+ minutes on a 1280x720 frame whenever the screenshot
// differs from its reference - indistinguishable from a hang).
//
//   bmpdiff <out.bmp> <ref.bmp> [--tolerance N] [--diff <diff.bmp>]
//
// Counts pixels whose RGB differs in ANY channel (the old loop's unit),
// reports the max channel delta and the differing bounding box, and
// optionally writes a diff image: the reference dimmed to 25% grey with
// differing pixels in red, |delta| x 8 (a 1-code-value drift is visible).
// Exit: 0 = at most N pixels differ, 1 = over budget, 2 = bad input.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "kenshin/util/BmpReader.h"
#include "kenshin/util/BmpWriter.h"

namespace {

using kenshin::u32;
using kenshin::u8;
using kenshin::usize;

int usage() {
  std::fprintf(stderr,
               "usage: bmpdiff <out.bmp> <ref.bmp> [--tolerance N] "
               "[--diff <diff.bmp>]\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) return usage();
  const std::string outPath = argv[1], refPath = argv[2];
  long long tolerance = 0;
  std::string diffPath;
  for (int i = 3; i < argc; ++i) {
    if (std::strcmp(argv[i], "--tolerance") == 0 && i + 1 < argc) {
      tolerance = std::atoll(argv[++i]);
    } else if (std::strcmp(argv[i], "--diff") == 0 && i + 1 < argc) {
      diffPath = argv[++i];
    } else {
      std::fprintf(stderr, "bmpdiff: unknown argument '%s'\n", argv[i]);
      return usage();
    }
  }

  kenshin::util::ImageRgba out, ref;
  std::string log;
  if (!kenshin::util::readBmp24(outPath, out, log)) {
    std::fprintf(stderr, "bmpdiff: cannot read %s: %s\n", outPath.c_str(),
                 log.c_str());
    return 2;
  }
  if (!kenshin::util::readBmp24(refPath, ref, log)) {
    std::fprintf(stderr, "bmpdiff: cannot read %s: %s\n", refPath.c_str(),
                 log.c_str());
    return 2;
  }
  if (out.width != ref.width || out.height != ref.height) {
    std::fprintf(stderr, "bmpdiff: size mismatch: %ux%u vs reference %ux%u\n",
                 out.width, out.height, ref.width, ref.height);
    return 2;
  }

  const u32 w = ref.width, h = ref.height;
  long long differing = 0;
  int maxDelta = 0;
  u32 x0 = w, y0 = h, x1 = 0, y1 = 0;
  std::vector<u8> diff;
  if (!diffPath.empty()) diff.resize(usize(w) * h * 4);
  for (u32 y = 0; y < h; ++y) {
    for (u32 x = 0; x < w; ++x) {
      const usize p = (usize(y) * w + x) * 4;
      int d = 0;
      for (int c = 0; c < 3; ++c)
        d = std::max(d, std::abs(int(out.rgba[p + c]) - int(ref.rgba[p + c])));
      if (d > 0) {
        ++differing;
        maxDelta = std::max(maxDelta, d);
        x0 = std::min(x0, x);
        y0 = std::min(y0, y);
        x1 = std::max(x1, x);
        y1 = std::max(y1, y);
      }
      if (!diff.empty()) {
        if (d > 0) {
          diff[p + 0] = u8(std::min(255, 64 + d * 8));
          diff[p + 1] = 0;
          diff[p + 2] = 0;
        } else {
          const int grey =
              (int(ref.rgba[p]) + int(ref.rgba[p + 1]) + int(ref.rgba[p + 2])) /
              12;  // mean / 4
          diff[p + 0] = diff[p + 1] = diff[p + 2] = u8(grey);
        }
        diff[p + 3] = 255;
      }
    }
  }

  const bool pass = differing <= tolerance;
  if (differing == 0) {
    std::printf("bmpdiff: identical (%ux%u)\n", w, h);
  } else {
    std::printf(
        "bmpdiff: %lld of %llu pixels differ (budget %lld), max channel "
        "delta %d, bbox [%u,%u]-[%u,%u]\n",
        differing, static_cast<unsigned long long>(usize(w) * h), tolerance,
        maxDelta, x0, y0, x1, y1);
  }
  if (!diff.empty() && differing > 0) {
    if (!kenshin::util::writeBmp24(diffPath, diff, w, h))
      std::fprintf(stderr, "bmpdiff: cannot write %s\n", diffPath.c_str());
  }
  return pass ? 0 : 1;
}
