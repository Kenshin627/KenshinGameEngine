#pragma once

// ── File — portable C-stdio file opening ────────────────────────────
// fopen_s is MSVC/Annex-K only, and plain fopen with a UTF-8 path breaks
// on Windows (it goes through the ANSI code page: a user folder named
// in Chinese fails to open). openFile takes UTF-8 / std::filesystem
// paths and opens them correctly on every platform. Returns nullptr on
// failure; the caller fcloses.

#include <cstdio>
#include <filesystem>
#include <string>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace util {

std::FILE* openFile(const std::filesystem::path& path, const char* mode);
/// `utf8Path` is interpreted as UTF-8 (the engine's string encoding).
std::FILE* openFile(const char* utf8Path, const char* mode);
std::FILE* openFile(const std::string& utf8Path, const char* mode);

/// UTF-8 string -> filesystem path (std::filesystem::path(std::string)
/// uses the native narrow encoding, which is not UTF-8 on Windows).
std::filesystem::path pathFromUtf8(const std::string& utf8);

}  // namespace util

KENSHIN_NS_END
