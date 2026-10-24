#pragma once

// ── Asset database (roadmap 1.3c) ───────────────────────────────
// A project folder's source assets, their identities and their cooked
// artifacts:
//
//   MyGame/project.kproject
//   MyGame/assets/...            sources + one .meta sidecar each (VCS)
//   MyGame/.kenshin/cache/<guid>/ artifacts + import.json (disposable)
//
// * Identity: every managed source has a .meta (JSON: guid, type,
//   importer settings). Moving a file with its .meta keeps every
//   reference. Sub-assets (a model's meshes, materials, textures,
//   prefab) get name-based Guids derived from (asset Guid, key), so
//   re-importing - even from an empty cache - reproduces them.
// * Repairs on scan:
//   - a source without a .meta gets one; if an orphaned .meta's last
//     import matches the file's content it is ADOPTED (a move that
//     left the .meta behind keeps its Guid);
//   - a duplicated Guid (a copied .meta) is re-issued to the copy;
//   - a corrupt .meta is reported and never overwritten.
// * Staleness: an import is current while the importer version, the
//   settings and every source file (content, checked by size + mtime
//   first, SHA-1 when they changed) match its import record. Imports
//   build into a temp folder and replace the old artifacts only on
//   success.
//
// Types by extension: texture (png jpg jpeg tga bmp hdr ktx2), model
// (gltf glb obj), material (kmat), scene (kscene), prefab (kprefab).
// Other files are not managed (a .gltf's .bin is a model dependency).

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "kenshin/core/Guid.h"
#include "kenshin/ecs/Reflection.h"
#include "kenshin/scene/ProjectFile.h"
#include "kenshin/serialization/Json.h"

KENSHIN_NS_BEGIN
namespace assets {

enum class AssetType : u8 { Texture, Model, Material, Scene, Prefab };
std::string_view assetTypeName(AssetType type);
std::optional<AssetType> assetTypeFromName(std::string_view name);
/// By file extension (case-insensitive); nullopt = not managed.
std::optional<AssetType> assetTypeForPath(const std::filesystem::path& path);

enum class ArtifactType : u8 { Texture, Mesh, TranslucentMesh, Material, Prefab, Scene };
std::string_view artifactTypeName(ArtifactType type);

/// One cooked output. Single-artifact assets have key "main" and the
/// asset's own Guid; a model's sub-assets carry their key and derived
/// Guid.
struct Artifact {
  std::string key;
  core::Guid guid;
  ArtifactType type = ArtifactType::Texture;
  std::string name;  // display name
  std::string file;  // file name inside the asset's cache folder
};

struct AssetRecord {
  core::Guid guid;
  std::string path;  // project-relative, '/'-separated ("assets/x.png")
  AssetType type = AssetType::Texture;
  serialization::Json settings;  // the full (defaulted) importer settings
  bool imported = false;         // a successful import record exists
  std::vector<Artifact> artifacts;
  std::vector<std::string> notes;  // the importer's notes
  std::string error;               // the last failed import (empty = none)
};

struct ScanReport {
  std::vector<std::string> createdMetas;    // new assets
  std::vector<std::string> adoptedMoves;    // "old/path -> new/path"
  std::vector<std::string> reissuedGuids;   // copies that got a fresh Guid
  std::vector<std::string> orphanedMetas;   // .meta without its source (kept)
  std::vector<std::string> corruptMetas;    // "path: reason" (asset skipped)
};

struct ImportReport {
  std::vector<std::string> imported;  // paths
  std::vector<std::string> failed;    // "path: reason"
  usize upToDate = 0;
};

struct AssetDatabaseOptions {
  /// Validates scene and prefab assets (they fail to import without).
  const ecs::ComponentRegistry* registry = nullptr;
  /// Added to every importer version - tests simulate an engine update.
  u32 importerVersionSalt = 0;
};

class AssetDatabase {
 public:
  static constexpr std::string_view kMetaFormat = "kenshin.meta";
  static constexpr std::string_view kImportFormat = "kenshin.import";
  static constexpr const char* kCacheDir = ".kenshin/cache";

  /// Derived sub-asset identity (RFC 4122 v5 of the key under the asset).
  static core::Guid subAssetGuid(const core::Guid& asset, std::string_view key) {
    return core::Guid::fromName(asset, key);
  }
  /// The default importer settings for a new asset at `path`.
  static serialization::Json defaultSettings(AssetType type, std::string_view path);

  /// Reads project.kproject and scans assets/ (see refresh()).
  bool open(const std::filesystem::path& projectDir, const AssetDatabaseOptions& options,
            std::string* error, ScanReport* report = nullptr);
  /// Rescans: creates/repairs .meta files and reloads import records.
  ScanReport refresh();

  /// Imports every stale (or, with `force`, every) asset.
  ImportReport importAll(bool force = false);
  bool reimport(const core::Guid& asset, std::string* error);
  bool isStale(const core::Guid& asset) const;

  /// Validates and stores new settings (written to the .meta); the
  /// asset becomes stale.
  bool setSettings(const core::Guid& asset, const serialization::Json& settings,
                   std::string* error);

  const AssetRecord* find(const core::Guid& asset) const;
  const AssetRecord* findByPath(std::string_view projectRelativePath) const;
  /// All assets, sorted by path.
  std::vector<const AssetRecord*> assets() const;

  struct Resolved {
    const AssetRecord* asset;
    const Artifact* artifact;
  };
  /// An asset or sub-asset Guid -> its record and artifact (imported only).
  std::optional<Resolved> resolve(const core::Guid& guid) const;
  /// The cooked file of an asset or sub-asset; empty if not imported.
  std::filesystem::path artifactPath(const core::Guid& guid) const;

  /// Removes cache folders of assets that no longer exist (and failed
  /// imports' temp folders). Returns how many were removed.
  usize collectGarbage();

  const std::filesystem::path& projectDir() const { return m_root; }
  std::filesystem::path cacheDir() const { return m_root / kCacheDir; }
  const scene::ProjectDesc& project() const { return m_project; }

 private:
  struct SourceStamp {
    std::string path;  // project-relative
    u64 size = 0;
    i64 mtime = 0;
    std::string sha1;
  };
  struct ImportRecord {
    AssetType type;
    u32 importerVersion = 0;
    std::string settingsHash;
    std::vector<SourceStamp> sources;
    std::vector<Artifact> artifacts;
    std::vector<std::string> notes;
  };

  bool importOne(AssetRecord& asset, std::string& error);
  bool loadRecord(const core::Guid& guid, ImportRecord& out) const;
  bool recordCurrent(const AssetRecord& asset, const ImportRecord& record) const;
  u32 importerVersion(AssetType type) const;
  std::filesystem::path abs(std::string_view projectRelative) const;
  std::string rel(const std::filesystem::path& absolute) const;

  std::filesystem::path m_root;
  AssetDatabaseOptions m_options;
  scene::ProjectDesc m_project;
  std::vector<AssetRecord> m_assets;  // sorted by path
  std::unordered_map<core::Guid, usize> m_byGuid;
  std::unordered_map<core::Guid, std::pair<usize, usize>> m_bySubGuid;  // asset, artifact
};

}  // namespace assets
KENSHIN_NS_END
