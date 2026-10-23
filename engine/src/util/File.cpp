#include "kenshin/util/File.h"

#include <string_view>

KENSHIN_NS_BEGIN

namespace util {

std::filesystem::path pathFromUtf8(const std::string& utf8) {
  return std::filesystem::path(
      std::u8string_view(reinterpret_cast<const char8_t*>(utf8.data()),
                         utf8.size()));
}

std::FILE* openFile(const std::filesystem::path& path, const char* mode) {
  if (mode == nullptr) return nullptr;
#ifdef _WIN32
  // Wide APIs: the path's native form is UTF-16 - no code-page loss.
  std::wstring wmode;
  for (const char* m = mode; *m; ++m) wmode.push_back(static_cast<wchar_t>(*m));
  std::FILE* f = nullptr;
  if (_wfopen_s(&f, path.c_str(), wmode.c_str()) != 0) return nullptr;
  return f;
#else
  return std::fopen(path.c_str(), mode);
#endif
}

std::FILE* openFile(const char* utf8Path, const char* mode) {
  if (utf8Path == nullptr) return nullptr;
  return openFile(pathFromUtf8(utf8Path), mode);
}

std::FILE* openFile(const std::string& utf8Path, const char* mode) {
  return openFile(pathFromUtf8(utf8Path), mode);
}

}  // namespace util

KENSHIN_NS_END
