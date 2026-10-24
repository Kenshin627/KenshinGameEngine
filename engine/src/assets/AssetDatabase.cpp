#include "kenshin/assets/AssetDatabase.h"

#include <algorithm>
#include <format>
#include <map>
#include <set>

#include "AssetImporters.h"
#include "kenshin/core/Log.h"
#include "kenshin/core/Sha1.h"
#include "kenshin/util/File.h"

KENSHIN_NS_BEGIN
namespace assets {

namespace fs = std::filesystem;
using serialization::Json;

// ── Types ──────────────────────────────────────────────────────

std::string_view assetTypeName(AssetType type) {
  switch (type) {
    case AssetType::Texture: return "texture";
    case AssetType::Model: return "model";
    case AssetType::Material: return "material";
    case AssetType::Scene: return "scene";
    case AssetType::Prefab: return "prefab";
  }
  return "?";
}

std::optional<AssetType> assetTypeFromName(std::string_view name) {
  for (AssetType t : {AssetType::Texture, AssetType::Model, AssetType::Material, AssetType::Scene,
                      AssetType::Prefab})
    if (assetTypeName(t) == name) return t;
  return std::nullopt;
}

std::optional<AssetType> assetTypeForPath(const fs::path& path) {
  std::string ext = util::pathToUtf8(path.extension());
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".tga" || ext == ".bmp" ||
      ext == ".hdr" || ext == ".ktx2")
    return AssetType::Texture;
  if (ext == ".gltf" || ext == ".glb" || ext == ".obj") return AssetType::Model;
  if (ext == ".kmat") return AssetType::Material;
  if (ext == ".kscene") return AssetType::Scene;
  if (ext == ".kprefab") return AssetType::Prefab;
  return std::nullopt;
}

std::string_view artifactTypeName(ArtifactType type) {
  switch (type) {
    case ArtifactType::Texture: return "texture";
    case ArtifactType::Mesh: return "mesh";
    case ArtifactType::TranslucentMesh: return "translucent_mesh";
    case ArtifactType::Material: return "material";
    case ArtifactType::Prefab: return "prefab";
    case ArtifactType::Scene: return "scene";
  }
  return "?";
}

namespace {

std::optional<ArtifactType> artifactTypeFromName(std::string_view n) {
  for (ArtifactType t : {ArtifactType::Texture, ArtifactType::Mesh, ArtifactType::TranslucentMesh,
                         ArtifactType::Material, ArtifactType::Prefab, ArtifactType::Scene})
    if (artifactTypeName(t) == n) return t;
  return std::nullopt;
}

const serialization::SchemaMigrator& metaMigrator() {
  static const serialization::SchemaMigrator m(std::string(AssetDatabase::kMetaFormat), 1);
  return m;
}
const serialization::SchemaMigrator& importMigrator() {
  static const serialization::SchemaMigrator m(std::string(AssetDatabase::kImportFormat), 1);
  return m;
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return s;
}

/// Streams a file through SHA-1; empty on failure.
std::string fileSha1(const fs::path& p) {
  std::FILE* f = util::openFile(p, "rb");
  if (!f) return {};
  core::Sha1 sha;
  char buf[1 << 16];
  usize n = 0;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) sha.update(buf, n);
  std::fclose(f);
  return core::Sha1::hex(sha.finish());
}

bool statFile(const fs::path& p, u64& size, i64& mtime) {
  std::error_code ec;
  size = fs::file_size(p, ec);
  if (ec) return false;
  const auto t = fs::last_write_time(p, ec);
  if (ec) return false;
  mtime = static_cast<i64>(t.time_since_epoch().count());
  return true;
}

std::string settingsHash(const Json& settings) {
  return core::Sha1::hex(core::Sha1::of(serialization::writeJson(settings)));
}

struct MetaFile {
  core::Guid guid;
  AssetType type;
  Json settings;
};

bool readMeta(const fs::path& metaPath, AssetType expected, MetaFile& out, std::string& error) {
  std::string text;
  if (!util::readFile(metaPath, text)) {
    error = "unreadable";
    return false;
  }
  Json doc;
  if (!serialization::parseJson(text, doc, &error)) return false;
  if (!metaMigrator().migrate(doc, error)) return false;
  MetaFile m{};
  bool haveGuid = false, haveType = false;
  Json settings = Json::object();
  for (auto it = doc.begin(); it != doc.end(); ++it) {
    const std::string& k = it.key();
    if (k == "format" || k == "version") continue;
    if (k == "guid") {
      const auto g = it->is_string() ? core::Guid::parse(it->get<std::string>()) : std::nullopt;
      if (!g || g->isNull()) {
        error = "guid must be a non-null guid string";
        return false;
      }
      m.guid = *g;
      haveGuid = true;
    } else if (k == "type") {
      const auto t = it->is_string() ? assetTypeFromName(it->get<std::string>()) : std::nullopt;
      if (!t) {
        error = "type must be one of [texture, model, material, scene, prefab]";
        return false;
      }
      m.type = *t;
      haveType = true;
    } else if (k == "settings") {
      settings = *it;
    } else {
      error = std::format("unknown key \"{}\"", k);
      return false;
    }
  }
  if (!haveGuid) {
    error = "missing \"guid\"";
    return false;
  }
  if (!haveType) {
    error = "missing \"type\"";
    return false;
  }
  if (m.type != expected) {
    error = std::format("type \"{}\" does not match the file (a {})", assetTypeName(m.type),
                        assetTypeName(expected));
    return false;
  }
  std::string why;
  if (!detail::normalizeSettings(m.type, settings, m.settings, why)) {
    error = why;
    return false;
  }
  out = std::move(m);
  return true;
}

bool writeMeta(const fs::path& metaPath, const core::Guid& guid, AssetType type,
               const Json& settings, std::string* error) {
  Json doc = Json::object();
  doc["format"] = std::string(AssetDatabase::kMetaFormat);
  doc["version"] = 1;
  doc["guid"] = guid.toString();
  doc["type"] = std::string(assetTypeName(type));
  doc["settings"] = settings;
  return util::writeFileAtomic(metaPath, serialization::writeJson(doc), error);
}

fs::path metaPathOf(const fs::path& source) {
  fs::path m = source;
  m += ".meta";
  return m;
}

}  // namespace

// ── Settings ───────────────────────────────────────────────────

Json AssetDatabase::defaultSettings(AssetType type, std::string_view path) {
  Json in = Json::object();
  if (type == AssetType::Texture) {
    // A conservative name heuristic for NEW assets (the result is
    // written to the .meta, where it can be changed).
    const std::string stem = lower(util::pathToUtf8(util::pathFromUtf8(std::string(path)).stem()));
    auto has = [&](std::string_view token) { return stem.find(token) != std::string::npos; };
    if (has("normal") || has("nrm") || (stem.size() > 2 && stem.ends_with("_n")))
      in["normal_map"] = true;
    else if (has("rough") || has("metal") || has("orm") || has("occlusion") || has("_ao") ||
             has("_mr") || has("height") || has("mask") || has("_disp"))
      in["color_space"] = "linear";
  }
  Json out;
  std::string error;
  detail::normalizeSettings(type, in, out, error);
  return out;
}

u32 AssetDatabase::importerVersion(AssetType type) const {
  return detail::baseImporterVersion(type) + m_options.importerVersionSalt;
}

// ── Paths ──────────────────────────────────────────────────────

fs::path AssetDatabase::abs(std::string_view projectRelative) const {
  return m_root / util::pathFromUtf8(std::string(projectRelative));
}

std::string AssetDatabase::rel(const fs::path& absolute) const {
  // Project-relative with '/' separators on every platform (the form
  // .meta adoption, records and findByPath compare).
  std::error_code ec;
  fs::path r = fs::relative(absolute, m_root, ec);
  if (ec || r.empty()) r = absolute;
  const std::u8string g = r.generic_u8string();
  return std::string(reinterpret_cast<const char*>(g.data()), g.size());
}

// ── Open / scan ────────────────────────────────────────────────

bool AssetDatabase::open(const fs::path& projectDir, const AssetDatabaseOptions& options,
                         std::string* error, ScanReport* report) {
  m_root = projectDir;
  m_options = options;
  m_assets.clear();
  m_byGuid.clear();
  m_bySubGuid.clear();
  std::string why;
  if (!scene::loadProjectFile(m_root / scene::kProjectFileName, m_project, &why)) {
    if (error) *error = why;
    return false;
  }
  ScanReport r = refresh();
  if (report) *report = std::move(r);
  return true;
}

ScanReport AssetDatabase::refresh() {
  ScanReport report;
  const fs::path assetsDir = m_root / "assets";
  // Every managed source and every .meta under assets/ (hidden entries
  // skipped), keyed by project-relative path.
  std::map<std::string, fs::path> sources, metas;
  std::error_code ec;
  if (fs::is_directory(assetsDir, ec)) {
    for (auto it = fs::recursive_directory_iterator(assetsDir, ec);
         !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
      const std::string name = util::pathToUtf8(it->path().filename());
      if (!name.empty() && name[0] == '.') {
        if (it->is_directory(ec)) it.disable_recursion_pending();
        continue;
      }
      if (!it->is_regular_file(ec)) continue;
      const std::string relPath = rel(it->path());
      if (lower(name).ends_with(".meta"))
        metas.emplace(relPath.substr(0, relPath.size() - 5), it->path());
      else if (assetTypeForPath(it->path()))
        sources.emplace(relPath, it->path());
    }
  }

  // 1. Read every .meta that has its source; corrupt ones exclude the
  //    source from management (never regenerated over).
  struct Entry {
    std::string path;
    MetaFile meta;
  };
  std::vector<Entry> entries;
  std::set<std::string> handled;
  for (const auto& [path, metaFile] : metas) {
    const auto src = sources.find(path);
    if (src == sources.end()) continue;  // orphan (step 2)
    handled.insert(path);
    MetaFile m;
    std::string why;
    if (!readMeta(metaFile, *assetTypeForPath(src->second), m, why)) {
      report.corruptMetas.push_back(path + ".meta: " + why);
      continue;
    }
    entries.push_back({path, std::move(m)});
  }

  // 2. Orphaned metas (their source is gone) - kept on disk; one whose
  //    last import's content matches a NEW source is adopted (a move
  //    that left the .meta behind).
  struct Orphan {
    std::string path;
    MetaFile meta;
    ImportRecord record;
    bool hasRecord = false;
    bool adopted = false;
  };
  std::vector<Orphan> orphans;
  for (const auto& [path, metaFile] : metas) {
    if (sources.count(path)) continue;
    const auto type = assetTypeForPath(util::pathFromUtf8(path));
    Orphan o;
    o.path = path;
    std::string why;
    if (!type || !readMeta(metaFile, *type, o.meta, why)) {
      report.orphanedMetas.push_back(path + ".meta");
      continue;
    }
    o.hasRecord = loadRecord(o.meta.guid, o.record) && !o.record.sources.empty();
    orphans.push_back(std::move(o));
  }
  std::set<core::Guid> metaGuids;
  for (const Entry& e : entries) metaGuids.insert(e.meta.guid);

  // 3. Sources without a .meta: adopt an orphan or create a new asset.
  for (const auto& [path, file] : sources) {
    if (handled.count(path)) continue;
    const AssetType type = *assetTypeForPath(file);
    std::string sha;
    Orphan* match = nullptr;
    u64 size = 0;
    i64 mtime = 0;
    statFile(file, size, mtime);
    for (Orphan& o : orphans) {
      if (o.adopted || !o.hasRecord || o.meta.type != type || o.record.sources[0].size != size ||
          metaGuids.count(o.meta.guid))
        continue;
      if (sha.empty()) sha = fileSha1(file);
      if (o.record.sources[0].sha1 == sha) {
        match = &o;
        break;
      }
    }
    MetaFile m;
    if (match) {
      m = match->meta;
      match->adopted = true;
      report.adoptedMoves.push_back(match->path + " -> " + path);
      std::error_code rmEc;
      fs::remove(metaPathOf(abs(match->path)), rmEc);  // the meta moves with it
    } else {
      m.guid = core::Guid::generate();
      m.type = type;
      m.settings = defaultSettings(type, path);
      report.createdMetas.push_back(path);
    }
    std::string why;
    if (!writeMeta(metaPathOf(file), m.guid, m.type, m.settings, &why)) {
      report.corruptMetas.push_back(path + ".meta: cannot write (" + why + ")");
      continue;
    }
    metaGuids.insert(m.guid);
    entries.push_back({path, std::move(m)});
  }
  for (const Orphan& o : orphans)
    if (!o.adopted) report.orphanedMetas.push_back(o.path + ".meta");

  // 4. Duplicate Guids (a copied .meta): the asset its import record
  //    names keeps the Guid (else the first path); copies get new ones.
  std::map<core::Guid, std::vector<Entry*>> byGuid;
  for (Entry& e : entries) byGuid[e.meta.guid].push_back(&e);
  for (auto& [guid, group] : byGuid) {
    if (group.size() < 2) continue;
    ImportRecord rec;
    Entry* keeper = group.front();
    if (loadRecord(guid, rec) && !rec.sources.empty())
      for (Entry* e : group)
        if (e->path == rec.sources[0].path) keeper = e;
    for (Entry* e : group) {
      if (e == keeper) continue;
      e->meta.guid = core::Guid::generate();
      std::string why;
      writeMeta(metaPathOf(abs(e->path)), e->meta.guid, e->meta.type, e->meta.settings, &why);
      report.reissuedGuids.push_back(e->path);
      KENSHIN_WARN("[assets] {} duplicated the guid of {} - re-issued {}", e->path, keeper->path,
                   e->meta.guid.toString());
    }
  }

  // 5. Build the records (sorted by path) and attach import results.
  std::sort(entries.begin(), entries.end(),
            [](const Entry& a, const Entry& b) { return a.path < b.path; });
  std::vector<AssetRecord> previous = std::move(m_assets);
  m_assets.clear();
  m_byGuid.clear();
  m_bySubGuid.clear();
  for (Entry& e : entries) {
    AssetRecord r;
    r.guid = e.meta.guid;
    r.path = e.path;
    r.type = e.meta.type;
    r.settings = std::move(e.meta.settings);
    ImportRecord rec;
    if (loadRecord(r.guid, rec)) {
      r.imported = true;
      r.artifacts = rec.artifacts;
      r.notes = rec.notes;
    }
    for (const AssetRecord& p : previous)
      if (p.guid == r.guid) r.error = p.error;
    m_assets.push_back(std::move(r));
  }
  for (usize i = 0; i < m_assets.size(); ++i) {
    m_byGuid.emplace(m_assets[i].guid, i);
    for (usize a = 0; a < m_assets[i].artifacts.size(); ++a)
      m_bySubGuid.emplace(m_assets[i].artifacts[a].guid, std::make_pair(i, a));
  }
  return report;
}

// ── Import records ─────────────────────────────────────────────

bool AssetDatabase::loadRecord(const core::Guid& guid, ImportRecord& out) const {
  std::string text;
  if (!util::readFile(cacheDir() / guid.toString() / "import.json", text)) return false;
  Json doc;
  std::string why;
  if (!serialization::parseJson(text, doc, &why) || !importMigrator().migrate(doc, why)) return false;
  try {
    ImportRecord r;
    const auto type = assetTypeFromName(doc.at("type").get<std::string>());
    if (!type) return false;
    r.type = *type;
    r.importerVersion = doc.at("importer_version").get<u32>();
    r.settingsHash = doc.at("settings_hash").get<std::string>();
    for (const Json& s : doc.at("sources"))
      r.sources.push_back({s.at("path").get<std::string>(), s.at("size").get<u64>(),
                           s.at("mtime").get<i64>(), s.at("sha1").get<std::string>()});
    for (const Json& a : doc.at("artifacts")) {
      const auto t = artifactTypeFromName(a.at("type").get<std::string>());
      const auto g = core::Guid::parse(a.at("guid").get<std::string>());
      if (!t || !g) return false;
      r.artifacts.push_back({a.at("key").get<std::string>(), *g, *t, a.at("name").get<std::string>(),
                             a.at("file").get<std::string>()});
    }
    for (const Json& n : doc.at("notes")) r.notes.push_back(n.get<std::string>());
    out = std::move(r);
    return true;
  } catch (const Json::exception&) {
    return false;  // a damaged record is simply stale
  }
}

bool AssetDatabase::recordCurrent(const AssetRecord& asset, const ImportRecord& rec) const {
  if (rec.type != asset.type || rec.importerVersion != importerVersion(asset.type) ||
      rec.settingsHash != settingsHash(asset.settings) || rec.sources.empty())
    return false;
  for (usize i = 0; i < rec.sources.size(); ++i) {
    const SourceStamp& s = rec.sources[i];
    // The asset may have moved since (with its .meta): its own content
    // is checked at the CURRENT path.
    const fs::path p = i == 0 ? abs(asset.path) : abs(s.path);
    u64 size = 0;
    i64 mtime = 0;
    if (!statFile(p, size, mtime) || size != s.size) return false;
    if (mtime != s.mtime && fileSha1(p) != s.sha1) return false;
  }
  for (const Artifact& a : rec.artifacts) {
    std::error_code ec;
    if (!fs::is_regular_file(cacheDir() / asset.guid.toString() / util::pathFromUtf8(a.file), ec))
      return false;
  }
  return true;
}

bool AssetDatabase::isStale(const core::Guid& guid) const {
  const AssetRecord* a = find(guid);
  if (!a) return false;
  ImportRecord rec;
  return !loadRecord(guid, rec) || !recordCurrent(*a, rec);
}

// ── Import ─────────────────────────────────────────────────────

bool AssetDatabase::importOne(AssetRecord& asset, std::string& error) {
  const std::string id = asset.guid.toString();
  const fs::path finalDir = cacheDir() / id;
  const fs::path tmp = cacheDir() / (id + ".tmp");
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp, ec);
  if (ec) {
    error = "cannot create " + util::pathToUtf8(tmp);
    return false;
  }
  detail::ImportContext ctx;
  ctx.guid = asset.guid;
  ctx.type = asset.type;
  ctx.source = abs(asset.path);
  ctx.settings = &asset.settings;
  ctx.outDir = tmp;
  ctx.registry = m_options.registry;
  if (!detail::runImporter(ctx, error)) {
    fs::remove_all(tmp, ec);
    return false;  // the previous artifacts stay in place
  }
  // The record: what was read (stamped) and what was written.
  Json doc = Json::object();
  doc["format"] = std::string(kImportFormat);
  doc["version"] = 1;
  doc["type"] = std::string(assetTypeName(asset.type));
  doc["importer_version"] = importerVersion(asset.type);
  doc["settings_hash"] = settingsHash(asset.settings);
  Json sources = Json::array();
  std::vector<fs::path> files{ctx.source};
  for (const fs::path& d : ctx.dependencies)
    if (std::find(files.begin(), files.end(), d) == files.end()) files.push_back(d);
  for (const fs::path& f : files) {
    u64 size = 0;
    i64 mtime = 0;
    if (!statFile(f, size, mtime)) continue;  // optional dependency absent
    Json s = Json::object();
    s["path"] = rel(f);
    s["size"] = size;
    s["mtime"] = mtime;
    s["sha1"] = fileSha1(f);
    sources.push_back(std::move(s));
  }
  doc["sources"] = std::move(sources);
  Json artifacts = Json::array();
  for (const Artifact& a : ctx.artifacts) {
    Json j = Json::object();
    j["key"] = a.key;
    j["guid"] = a.guid.toString();
    j["type"] = std::string(artifactTypeName(a.type));
    j["name"] = a.name;
    j["file"] = a.file;
    artifacts.push_back(std::move(j));
  }
  doc["artifacts"] = std::move(artifacts);
  Json notes = Json::array();
  for (const std::string& n : ctx.notes) notes.push_back(n);
  doc["notes"] = std::move(notes);
  std::string why;
  if (!util::writeFileAtomic(tmp / "import.json", serialization::writeJson(doc), &why)) {
    fs::remove_all(tmp, ec);
    error = why;
    return false;
  }
  fs::remove_all(finalDir, ec);
  fs::rename(tmp, finalDir, ec);
  if (ec) {
    error = "cannot move the import into place: " + ec.message();
    fs::remove_all(tmp, ec);
    return false;
  }
  asset.imported = true;
  asset.artifacts = std::move(ctx.artifacts);
  asset.notes = std::move(ctx.notes);
  asset.error.clear();
  return true;
}

ImportReport AssetDatabase::importAll(bool force) {
  ImportReport report;
  for (AssetRecord& a : m_assets) {
    if (!force && !isStale(a.guid)) {
      ++report.upToDate;
      continue;
    }
    std::string why;
    if (importOne(a, why)) {
      report.imported.push_back(a.path);
    } else {
      a.error = why;
      report.failed.push_back(a.path + ": " + why);
      KENSHIN_ERROR("[assets] import failed: {}: {}", a.path, why);
    }
  }
  // Sub-asset index.
  m_bySubGuid.clear();
  for (usize i = 0; i < m_assets.size(); ++i)
    for (usize a = 0; a < m_assets[i].artifacts.size(); ++a)
      m_bySubGuid.emplace(m_assets[i].artifacts[a].guid, std::make_pair(i, a));
  return report;
}

bool AssetDatabase::reimport(const core::Guid& guid, std::string* error) {
  const auto it = m_byGuid.find(guid);
  if (it == m_byGuid.end()) {
    if (error) *error = "unknown asset " + guid.toString();
    return false;
  }
  AssetRecord& a = m_assets[it->second];
  std::string why;
  const bool ok = importOne(a, why);
  a.error = ok ? std::string() : why;
  for (auto s = m_bySubGuid.begin(); s != m_bySubGuid.end();)
    s = s->second.first == it->second ? m_bySubGuid.erase(s) : std::next(s);
  for (usize k = 0; k < a.artifacts.size(); ++k)
    m_bySubGuid.emplace(a.artifacts[k].guid, std::make_pair(it->second, k));
  if (!ok && error) *error = why;
  return ok;
}

bool AssetDatabase::setSettings(const core::Guid& guid, const Json& settings, std::string* error) {
  const auto it = m_byGuid.find(guid);
  if (it == m_byGuid.end()) {
    if (error) *error = "unknown asset " + guid.toString();
    return false;
  }
  AssetRecord& a = m_assets[it->second];
  Json normalized;
  std::string why;
  if (!detail::normalizeSettings(a.type, settings, normalized, why) ||
      !writeMeta(metaPathOf(abs(a.path)), a.guid, a.type, normalized, &why)) {
    if (error) *error = why;
    return false;
  }
  a.settings = std::move(normalized);
  return true;
}

// ── Queries ────────────────────────────────────────────────────

const AssetRecord* AssetDatabase::find(const core::Guid& guid) const {
  const auto it = m_byGuid.find(guid);
  return it == m_byGuid.end() ? nullptr : &m_assets[it->second];
}

const AssetRecord* AssetDatabase::findByPath(std::string_view path) const {
  for (const AssetRecord& a : m_assets)
    if (a.path == path) return &a;
  return nullptr;
}

std::vector<const AssetRecord*> AssetDatabase::assets() const {
  std::vector<const AssetRecord*> out;
  for (const AssetRecord& a : m_assets) out.push_back(&a);
  return out;
}

std::optional<AssetDatabase::Resolved> AssetDatabase::resolve(const core::Guid& guid) const {
  if (const auto s = m_bySubGuid.find(guid); s != m_bySubGuid.end()) {
    const AssetRecord& a = m_assets[s->second.first];
    return Resolved{&a, &a.artifacts[s->second.second]};
  }
  return std::nullopt;
}

fs::path AssetDatabase::artifactPath(const core::Guid& guid) const {
  const auto r = resolve(guid);
  if (!r) return {};
  return cacheDir() / r->asset->guid.toString() / util::pathFromUtf8(r->artifact->file);
}

usize AssetDatabase::collectGarbage() {
  usize removed = 0;
  std::error_code ec;
  if (!fs::is_directory(cacheDir(), ec)) return 0;
  std::vector<fs::path> doomed;
  for (const auto& e : fs::directory_iterator(cacheDir(), ec)) {
    const std::string name = util::pathToUtf8(e.path().filename());
    const auto g = core::Guid::parse(name);
    if (!e.is_directory(ec) || !g || !m_byGuid.count(*g)) doomed.push_back(e.path());
  }
  for (const fs::path& p : doomed) {
    fs::remove_all(p, ec);
    if (!ec) ++removed;
  }
  return removed;
}

}  // namespace assets
KENSHIN_NS_END
