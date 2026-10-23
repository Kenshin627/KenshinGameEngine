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
#include <string_view>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace util {

std::FILE* openFile(const std::filesystem::path& path, const char* mode);
/// `utf8Path` is interpreted as UTF-8 (the engine's string encoding).
std::FILE* openFile(const char* utf8Path, const char* mode);
std::FILE* openFile(const std::string& utf8Path, const char* mode);

/// Whole file as bytes (no newline translation). False if it cannot be
/// opened or read.
bool readFile(const std::filesystem::path& path, std::string& out);

/// Replace `path` with `data` atomically: written to a sibling temp
/// file, flushed to disk, then renamed over the target - a crash or a
/// full disk leaves either the old file or the new one, never half of
/// each. On failure `error` (if given) says why and the target is
/// untouched.
bool writeFileAtomic(const std::filesystem::path& path, std::string_view data,
                     std::string* error = nullptr);

/// 64-bit position / seek (fseek/ftell use `long`: 32 bits on Windows,
/// and files cross 2 GB at scale). seek64 returns true on success; tell64
/// returns -1 on failure. `whence` = SEEK_SET / SEEK_CUR / SEEK_END.
i64 tell64(std::FILE* file);
bool seek64(std::FILE* file, i64 offset, int whence);

/// UTF-8 string -> filesystem path (std::filesystem::path(std::string)
/// uses the native narrow encoding, which is not UTF-8 on Windows).
std::filesystem::path pathFromUtf8(const std::string& utf8);
/// The inverse: a path as UTF-8 text (messages, files).
std::string pathToUtf8(const std::filesystem::path& path);

}  // namespace util

KENSHIN_NS_END
