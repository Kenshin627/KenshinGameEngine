#pragma once

// ── JSON for engine files (roadmap 1.2) ─────────────────────────
// Scene, prefab and project files are text JSON so they diff, merge and
// are directly editable by people and agents. Two rules make them
// stable under version control:
//   * canonical output - insertion-ordered keys, 2-space indent, short
//     scalar arrays on one line, '\n' endings, floats in their shortest
//     exact form - so save(load(file)) reproduces the file byte-for-byte;
//   * every file starts with {"format": ..., "version": N}, and older
//     versions are upgraded by an explicit chain of migrations.

#include <functional>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <vector>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN
namespace serialization {

/// Insertion-ordered: keys come out in the order they were written.
using Json = nlohmann::ordered_json;

/// The shortest decimal that reads back as exactly `v` (stored as a
/// double so the text is "0.1", not "0.10000000149011612").
Json floatToJson(f32 v);

/// Canonical text (see the header comment), ending in a newline.
std::string writeJson(const Json& value);

/// Parse; on malformed text returns false with a line/column message.
bool parseJson(std::string_view text, Json& out, std::string* error);

/// Upgrades documents of one file format from older versions to
/// `currentVersion`, one step at a time.
class SchemaMigrator {
 public:
  using Step = std::function<bool(Json& doc, std::string& error)>;

  SchemaMigrator(std::string format, i32 currentVersion)
      : m_format(std::move(format)), m_current(currentVersion) {}

  /// The step that turns a version `fromVersion` document into
  /// `fromVersion + 1`. Steps must be added for every version below
  /// current that should stay loadable.
  void add(i32 fromVersion, std::string description, Step step);

  const std::string& format() const { return m_format; }
  i32 currentVersion() const { return m_current; }

  /// Checks the header (format name, integer version), applies the
  /// steps in order and stamps the new version. Fails - naming the
  /// reason - for a wrong format, a version newer than current (written
  /// by a newer engine) or a version with no migration path. `doc` may
  /// be partially migrated on failure; callers discard it.
  bool migrate(Json& doc, std::string& error) const;

 private:
  struct Migration {
    i32 from;
    std::string description;
    Step step;
  };
  std::string m_format;
  i32 m_current;
  std::vector<Migration> m_steps;
};

}  // namespace serialization
KENSHIN_NS_END
