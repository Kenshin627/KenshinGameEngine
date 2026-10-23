#include "kenshin/util/File.h"

#include <string_view>
#include <system_error>

#ifdef _WIN32
#include <io.h>  // _commit, _fileno
#else
#include <unistd.h>  // fsync
#endif

KENSHIN_NS_BEGIN

namespace util {

std::filesystem::path pathFromUtf8(const std::string& utf8) {
  return std::filesystem::path(
      std::u8string_view(reinterpret_cast<const char8_t*>(utf8.data()),
                         utf8.size()));
}

std::string pathToUtf8(const std::filesystem::path& path) {
  const std::u8string u8 = path.u8string();
  return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
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

bool readFile(const std::filesystem::path& path, std::string& out) {
  std::FILE* f = openFile(path, "rb");
  if (!f) return false;
  out.clear();
  char buf[1 << 16];
  bool ok = true;
  for (;;) {
    const usize n = std::fread(buf, 1, sizeof(buf), f);
    out.append(buf, n);
    if (n < sizeof(buf)) {
      ok = std::ferror(f) == 0;
      break;
    }
  }
  std::fclose(f);
  return ok;
}

bool writeFileAtomic(const std::filesystem::path& path, std::string_view data,
                     std::string* error) {
  auto fail = [&](const std::string& why) {
    if (error) *error = why;
    return false;
  };
  std::filesystem::path tmp = path;
  tmp += ".tmp";
  std::FILE* f = openFile(tmp, "wb");
  if (!f) return fail("cannot create " + pathToUtf8(tmp));
  bool ok = std::fwrite(data.data(), 1, data.size(), f) == data.size();
  ok = std::fflush(f) == 0 && ok;
#ifdef _WIN32
  ok = _commit(_fileno(f)) == 0 && ok;
#else
  ok = fsync(fileno(f)) == 0 && ok;
#endif
  ok = std::fclose(f) == 0 && ok;
  std::error_code ec;
  if (!ok) {
    std::filesystem::remove(tmp, ec);
    return fail("write failed (disk full or I/O error)");
  }
  // Replaces an existing target on every platform (MoveFileEx with
  // REPLACE_EXISTING on Windows, rename(2) elsewhere).
  std::filesystem::rename(tmp, path, ec);
  if (ec) {
    const std::string why = "cannot replace the target file: " + ec.message();
    std::filesystem::remove(tmp, ec);
    return fail(why);
  }
  return true;
}

}  // namespace util

KENSHIN_NS_END
