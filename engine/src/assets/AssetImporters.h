#pragma once

// Internal to the asset database: per-type importers (AssetImporters.cpp).

#include <filesystem>
#include <string>
#include <vector>

#include "kenshin/assets/AssetDatabase.h"

KENSHIN_NS_BEGIN
namespace assets::detail {

/// Bumped whenever an importer's output for the same input changes.
u32 baseImporterVersion(AssetType type);

/// Fills defaults and rejects unknown keys / bad values; `out` is the
/// canonical (complete, fixed key order) settings object.
bool normalizeSettings(AssetType type, const serialization::Json& in, serialization::Json& out,
                       std::string& error);

struct ImportContext {
  core::Guid guid;
  AssetType type = AssetType::Texture;
  std::filesystem::path source;  // absolute
  const serialization::Json* settings = nullptr;  // normalized
  std::filesystem::path outDir;  // empty temp folder to write artifacts into
  const ecs::ComponentRegistry* registry = nullptr;
  // Outputs.
  std::vector<Artifact> artifacts;
  std::vector<std::filesystem::path> dependencies;  // absolute, besides `source`
  std::vector<std::string> notes;
};

bool runImporter(ImportContext& ctx, std::string& error);

}  // namespace assets::detail
KENSHIN_NS_END
