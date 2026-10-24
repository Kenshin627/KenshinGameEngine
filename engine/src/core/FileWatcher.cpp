#include "kenshin/core/FileWatcher.h"

#include <algorithm>

KENSHIN_NS_BEGIN
namespace core {

namespace fs = std::filesystem;

void FileWatcher::watch(const fs::path& root) {
  m_root = root;
  m_files = scan();
}

std::unordered_map<std::string, FileWatcher::Stamp> FileWatcher::scan() const {
  std::unordered_map<std::string, Stamp> files;
  std::error_code ec;
  if (m_root.empty() || !fs::is_directory(m_root, ec)) return files;
  // A file vanishing mid-walk (an editor's save-by-rename) is skipped
  // this round and shows up at its final state next round.
  for (fs::recursive_directory_iterator it(m_root, fs::directory_options::skip_permission_denied, ec),
       end;
       !ec && it != end; it.increment(ec)) {
    std::error_code fe;
    if (!it->is_regular_file(fe) || fe) continue;
    Stamp s;
    s.size = it->file_size(fe);
    if (fe) continue;
    const auto t = it->last_write_time(fe);
    if (fe) continue;
    s.mtime = static_cast<i64>(t.time_since_epoch().count());
    const std::u8string rel = it->path().lexically_relative(m_root).generic_u8string();
    files.emplace(std::string(reinterpret_cast<const char*>(rel.data()), rel.size()), s);
  }
  return files;
}

std::vector<FileWatcher::Change> FileWatcher::poll() {
  std::unordered_map<std::string, Stamp> now = scan();
  std::vector<Change> changes;
  for (const auto& [path, stamp] : now) {
    const auto it = m_files.find(path);
    if (it == m_files.end())
      changes.push_back({path, ChangeKind::Added});
    else if (it->second.size != stamp.size || it->second.mtime != stamp.mtime)
      changes.push_back({path, ChangeKind::Modified});
  }
  for (const auto& [path, stamp] : m_files)
    if (!now.count(path)) changes.push_back({path, ChangeKind::Removed});
  std::sort(changes.begin(), changes.end(),
            [](const Change& a, const Change& b) { return a.path < b.path; });
  m_files = std::move(now);
  return changes;
}

}  // namespace core
KENSHIN_NS_END
