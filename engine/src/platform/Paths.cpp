#include "kenshin/platform/Paths.h"

#include <SDL3/SDL_filesystem.h>
#include <SDL3/SDL_stdinc.h>

KENSHIN_NS_BEGIN

namespace platform {

std::filesystem::path executableDir() {
  // SDL owns the returned string (cached for the process lifetime).
  const char* base = SDL_GetBasePath();
  if (base == nullptr) return {};
  return std::filesystem::path(
      std::u8string(reinterpret_cast<const char8_t*>(base)));
}

bool setEnvironmentVariable(const std::string& name, const std::string& value) {
  return SDL_setenv_unsafe(name.c_str(), value.c_str(), 1) == 0;
}

}  // namespace platform

KENSHIN_NS_END
