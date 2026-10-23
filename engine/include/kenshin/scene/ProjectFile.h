#pragma once

// ── Project file (roadmap 1.2) ──────────────────────────────────
// <project>/project.kproject - the root of a game project:
//
//   {
//     "format": "kenshin.project",
//     "version": 1,
//     "name": "Tetris",
//     "startup_scene": "scenes/main.kscene"
//   }
//
// Paths inside a project are relative to its folder, '/'-separated and
// may not leave it (an exported game must be self-contained). The asset
// database (1.3) and settings grow here through schema versions.

#include <filesystem>
#include <string>
#include <string_view>

#include "kenshin/serialization/Json.h"

KENSHIN_NS_BEGIN
namespace scene {

inline constexpr i32 kProjectVersion = 1;
inline constexpr std::string_view kProjectFormat = "kenshin.project";
inline constexpr std::string_view kProjectFileName = "project.kproject";

struct ProjectDesc {
  std::string name;
  std::string startupScene;  // project-relative; empty = none yet
};

const serialization::SchemaMigrator& projectMigrator();

/// True for a non-empty relative '/'-separated path with no "." / ".."
/// segments, empty segments, backslashes or drive/root prefix.
bool isProjectRelativePath(std::string_view path);

std::string saveProject(const ProjectDesc& project);
/// Strict (unknown keys, empty name, bad paths are errors); `out` is
/// untouched on failure.
bool loadProject(std::string_view text, ProjectDesc& out, std::string* error,
                 const serialization::SchemaMigrator* migrator = nullptr);

bool saveProjectFile(const std::filesystem::path& path, const ProjectDesc& project,
                     std::string* error);
bool loadProjectFile(const std::filesystem::path& path, ProjectDesc& out,
                     std::string* error);

}  // namespace scene
KENSHIN_NS_END
