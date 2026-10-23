#include "kenshin/scene/ProjectFile.h"

#include <format>

#include "kenshin/util/File.h"

KENSHIN_NS_BEGIN
namespace scene {

using serialization::Json;

const serialization::SchemaMigrator& projectMigrator() {
  static const serialization::SchemaMigrator m(std::string(kProjectFormat), kProjectVersion);
  return m;
}

bool isProjectRelativePath(std::string_view path) {
  if (path.empty() || path.front() == '/' || path.find('\\') != std::string_view::npos ||
      path.find(':') != std::string_view::npos)  // "C:..." or a URL scheme
    return false;
  usize start = 0;
  while (start <= path.size()) {
    const usize end = std::min(path.find('/', start), path.size());
    const std::string_view segment = path.substr(start, end - start);
    if (segment.empty() || segment == "." || segment == "..") return false;
    start = end + 1;
  }
  return true;
}

std::string saveProject(const ProjectDesc& project) {
  Json doc = Json::object();
  doc["format"] = std::string(kProjectFormat);
  doc["version"] = kProjectVersion;
  doc["name"] = project.name;
  doc["startup_scene"] = project.startupScene;
  return serialization::writeJson(doc);
}

bool loadProject(std::string_view text, ProjectDesc& out, std::string* error,
                 const serialization::SchemaMigrator* migrator) {
  auto fail = [&](const std::string& why) {
    if (error) *error = why;
    return false;
  };
  Json doc;
  std::string why;
  if (!serialization::parseJson(text, doc, &why)) return fail(why);
  if (!(migrator ? *migrator : projectMigrator()).migrate(doc, why)) return fail(why);

  ProjectDesc p;
  bool haveName = false;
  for (auto it = doc.begin(); it != doc.end(); ++it) {
    const std::string& k = it.key();
    if (k == "format" || k == "version") continue;
    if (k == "name") {
      if (!it->is_string() || it->get<std::string>().empty())
        return fail("\"name\" must be a non-empty string");
      p.name = it->get<std::string>();
      haveName = true;
    } else if (k == "startup_scene") {
      if (!it->is_string()) return fail("\"startup_scene\" must be a string");
      p.startupScene = it->get<std::string>();
      if (!p.startupScene.empty() && !isProjectRelativePath(p.startupScene))
        return fail(std::format(
            "\"startup_scene\" \"{}\" must be a relative path inside the project "
            "('/'-separated, no '..')",
            p.startupScene));
    } else {
      return fail(std::format("unknown key \"{}\"", k));
    }
  }
  if (!haveName) return fail("missing \"name\"");
  out = std::move(p);
  return true;
}

bool saveProjectFile(const std::filesystem::path& path, const ProjectDesc& project,
                     std::string* error) {
  return util::writeFileAtomic(path, saveProject(project), error);
}

bool loadProjectFile(const std::filesystem::path& path, ProjectDesc& out,
                     std::string* error) {
  std::string text;
  if (!util::readFile(path, text)) {
    if (error) *error = "cannot read " + util::pathToUtf8(path);
    return false;
  }
  std::string why;
  if (!loadProject(text, out, &why)) {
    if (error) *error = util::pathToUtf8(path) + ": " + why;
    return false;
  }
  return true;
}

}  // namespace scene
KENSHIN_NS_END
