#pragma once

// ── FileWatcher — portable polling change detection (roadmap 1.3d) ──
// Snapshots (size, mtime) of every regular file under a root and
// reports what changed since the previous poll. Polling instead of
// ReadDirectoryChangesW / inotify / FSEvents: one code path on all
// three platforms, no event-queue overflow, network drives work, and
// a stat walk of a few thousand files costs well under a millisecond
// per hundred files. The caller picks the rate (the asset manager polls
// at 4 Hz).
//
// A write that keeps both size and mtime (a same-size rewrite inside
// the filesystem's mtime granularity) is not seen: the asset database's
// content hash catches it at the next import instead.

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN
namespace core {

class FileWatcher {
 public:
  enum class ChangeKind : u8 { Added, Modified, Removed };
  struct Change {
    std::string path;  // relative to the root, '/'-separated (UTF-8)
    ChangeKind kind = ChangeKind::Modified;
  };

  /// Watches `root` recursively; the current state is the baseline (no
  /// changes are reported for it). A missing root watches nothing until
  /// it appears.
  void watch(const std::filesystem::path& root);
  /// Rescans; returns the changes since the last poll, sorted by path.
  std::vector<Change> poll();

  const std::filesystem::path& root() const { return m_root; }
  usize fileCount() const { return m_files.size(); }

 private:
  struct Stamp {
    u64 size = 0;
    i64 mtime = 0;
  };
  std::unordered_map<std::string, Stamp> scan() const;

  std::filesystem::path m_root;
  std::unordered_map<std::string, Stamp> m_files;
};

}  // namespace core
KENSHIN_NS_END
