#include "kenshin/assets/GltfLoader.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <map>

#pragma warning(push, 0)
#include "stb/stb_image.h"
#pragma warning(pop)

KENSHIN_NS_BEGIN

namespace assets {

bool loadGltfImpl(const std::string& path, MeshData& mesh, GltfImage* albedo,
                  std::string& log, std::vector<GltfMaterial>* materials,
                  std::vector<GltfImage>* images);

namespace {

// ---- minimal JSON DOM -------------------------------------------

struct JVal {
  enum Type : u8 { Null, Bool, Num, Str, Arr, Obj };
  Type type = Null;
  bool boolean = false;
  f64 number = 0.0;
  std::string text;                                   // Str
  std::vector<JVal> items;                            // Arr
  std::vector<std::pair<std::string, JVal>> members;  // Obj

  const JVal* find(const char* key) const {
    if (type != Obj) return nullptr;
    for (const auto& kv : members)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
  size_t size() const { return items.size(); }
  const JVal& at(size_t i) const { return items[i]; }
  f64 numOr(const char* key, f64 dflt) const {
    const JVal* v = find(key);
    return v && v->type == Num ? v->number : dflt;
  }
};

struct Json {
  const char* p = nullptr;
  const char* end = nullptr;
  std::string err;

  bool fail(const char* why) {
    if (err.empty()) err = why;
    return false;
  }
  void skipWs() {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
      ++p;
  }
  bool literal(const char* s, usize n) {
    if (usize(end - p) < n || std::memcmp(p, s, n) != 0) return false;
    p += n;
    return true;
  }

  bool parseString(std::string& out) {
    if (p >= end || *p != '"') return fail("expected string");
    ++p;
    out.clear();
    while (p < end && *p != '"') {
      if (*p == '\\') {
        ++p;
        if (p >= end) return fail("bad escape");
        switch (*p) {
          case '"':
            out += '"';
            break;
          case '\\':
            out += '\\';
            break;
          case '/':
            out += '/';
            break;
          case 'b':
            out += '\b';
            break;
          case 'f':
            out += '\f';
            break;
          case 'n':
            out += '\n';
            break;
          case 'r':
            out += '\r';
            break;
          case 't':
            out += '\t';
            break;
          case 'u': {
            // Keep the escape verbatim - glTF keys/URIs are ASCII and
            // the codepoints we care about never arrive escaped.
            if (end - p < 5) return fail("bad \\u escape");
            out += *p;
            out += p[1];
            out += p[2];
            out += p[3];
            out += p[4];
            p += 4;
            break;
          }
          default:
            return fail("bad escape char");
        }
        ++p;
      } else {
        out += *p++;
      }
    }
    if (p >= end) return fail("unterminated string");
    ++p;  // closing quote
    return true;
  }

  bool parseValue(JVal& out, int depth) {
    if (depth > 64) return fail("json too deep");
    skipWs();
    if (p >= end) return fail("unexpected end");
    if (*p == '{') {
      ++p;
      out.type = JVal::Obj;
      skipWs();
      if (p < end && *p == '}') {
        ++p;
        return true;
      }
      while (true) {
        skipWs();
        std::string key;
        if (!parseString(key)) return false;
        skipWs();
        if (p >= end || *p != ':') return fail("expected ':'");
        ++p;
        JVal v;
        v.type = JVal::Null;
        if (!parseValue(v, depth + 1)) return false;
        out.members.emplace_back(std::move(key), std::move(v));
        skipWs();
        if (p < end && *p == ',') {
          ++p;
          continue;
        }
        if (p < end && *p == '}') {
          ++p;
          return true;
        }
        return fail("expected ',' or '}'");
      }
    }
    if (*p == '[') {
      ++p;
      out.type = JVal::Arr;
      skipWs();
      if (p < end && *p == ']') {
        ++p;
        return true;
      }
      while (true) {
        JVal v;
        v.type = JVal::Null;
        if (!parseValue(v, depth + 1)) return false;
        out.items.push_back(std::move(v));
        skipWs();
        if (p < end && *p == ',') {
          ++p;
          continue;
        }
        if (p < end && *p == ']') {
          ++p;
          return true;
        }
        return fail("expected ',' or ']'");
      }
    }
    if (*p == '"') {
      out.type = JVal::Str;
      return parseString(out.text);
    }
    if (literal("true", 4)) {
      out.type = JVal::Bool;
      out.boolean = true;
      return true;
    }
    if (literal("false", 5)) {
      out.type = JVal::Bool;
      out.boolean = false;
      return true;
    }
    if (literal("null", 4)) return true;  // stays Null
    // Number: strtod over the remaining span.
    out.type = JVal::Num;
    char* numEnd = nullptr;
    out.number = std::strtod(p, &numEnd);
    if (numEnd == p) return fail("bad token");
    p = numEnd;
    return true;
  }
};

bool parseJson(const std::string& text, JVal& root, std::string& err) {
  Json j{text.data(), text.data() + text.size()};
  if (!j.parseValue(root, 0)) {
    err = "json: " + (j.err.empty() ? "parse failed" : j.err);
    return false;
  }
  return true;
}

// ---- small utilities ---------------------------------------------

bool readFile(const std::string& path, std::vector<u8>& out) {
  FILE* f = nullptr;
  if (fopen_s(&f, path.c_str(), "rb") != 0 || !f) return false;
  char buf[4096];
  size_t n = 0;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0)
    out.insert(out.end(), buf, buf + n);
  std::fclose(f);
  return true;
}

/// Checked f64 -> u32 (a hostile 1e30 in a JSON number makes the bare
/// cast UB; wrapped values could also land on wrong-but-in-bounds
/// offsets silently).
bool checkedU32(f64 v, u32& out) {
  if (!(v >= 0.0) || v > 4294967295.0) return false;
  out = u32(v);
  return true;
}

u32 base64Nibble(char c) {
  if (c >= 'A' && c <= 'Z') return u32(c - 'A');
  if (c >= 'a' && c <= 'z') return u32(c - 'a' + 26);
  if (c >= '0' && c <= '9') return u32(c - '0' + 52);
  if (c == '+') return 62;
  if (c == '/') return 63;
  return 0xFFFFFFFFu;
}

bool decodeBase64(const std::string& in, std::vector<u8>& out) {
  out.clear();
  u32 acc = 0;
  int bits = 0;
  for (char c : in) {
    if (c == '=' || c == '\n' || c == '\r') continue;
    const u32 v = base64Nibble(c);
    if (v == 0xFFFFFFFFu) return false;
    acc = (acc << 6) | v;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(u8((acc >> bits) & 0xFF));
    }
  }
  return true;
}

/// Untrusted-model URI guard: percent-decode, then refuse absolute
/// paths and parent-directory escapes (an image or buffer URI must
/// stay inside the model's own directory tree).
bool safeUri(const std::string& dir, const std::string& raw, std::string& log) {
  std::string uri;
  for (usize i = 0; i < raw.size(); ++i) {
    if (raw[i] == '%' && i + 2 < raw.size()) {
      const auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
      };
      const int hi = hex(raw[i + 1]), lo = hex(raw[i + 2]);
      if (hi >= 0 && lo >= 0) {
        uri += char(hi * 16 + lo);
        i += 2;
        continue;
      }
    }
    uri += raw[i];
  }
  if (uri.find(':') != std::string::npos && uri.find(':') < 2) {
    log = "uri with scheme refused: " + uri;
    return false;
  }
  const std::filesystem::path p(uri);
  if (p.is_absolute()) {
    log = "absolute uri refused: " + uri;
    return false;
  }
  for (const auto& seg : p)
    if (seg == "..") {
      log = "parent-directory uri refused: " + uri;
      return false;
    }
  return true;
}

// ---- glTF context ------------------------------------------------

struct Accessor {
  u32 componentType = 0;  // 5120..5126
  u32 count = 0;
  u32 perElement = 0;  // SCALAR=1 .. MAT4=16
  u32 bufferView = 0xFFFFFFFFu;
  u32 byteOffset = 0;
  bool sparse = false;
};

u32 componentSize(u32 t) {
  switch (t) {
    case 5120:
    case 5121:
      return 1;
    case 5122:
    case 5123:
      return 2;
    default:
      return 4;  // 5125 uint, 5126 float
  }
}

struct Gltf {
  const JVal* accessors = nullptr;
  const JVal* bufferViews = nullptr;
  const JVal* images = nullptr;
  const JVal* textures = nullptr;
  const JVal* materials = nullptr;
  const JVal* meshes = nullptr;
  const JVal* nodes = nullptr;
  const JVal* scenes = nullptr;

  std::vector<std::vector<u8>> buffers;  // decoded (BIN chunk or files)
  std::string dir;                       // for relative uri resolution
  std::string log;

  const JVal* arrayOrNull(const JVal& root, const char* key) {
    const JVal* v = root.find(key);
    return v && v->type == JVal::Arr ? v : nullptr;
  }

  /// Raw accessor bytes (base of the bufferView + accessor offset) +
  /// the bytes remaining in that buffer (bounds checks).
  const u8* accessorBase(const Accessor& a, usize* remaining) const {
    if (a.bufferView >= bufferViews->size()) return nullptr;
    const JVal& bv = bufferViews->at(a.bufferView);
    u32 buffer = 0, bvOff = 0, bvLen = 0;
    if (!checkedU32(bv.numOr("buffer", 0), buffer) ||
        !checkedU32(bv.numOr("byteOffset", 0), bvOff) ||
        !checkedU32(bv.numOr("byteLength", 0), bvLen))
      return nullptr;
    if (buffer >= buffers.size()) return nullptr;
    const std::vector<u8>& data = buffers[buffer];
    const usize base = usize(bvOff) + a.byteOffset;
    if (base >= data.size()) return nullptr;
    if (remaining) {
      *remaining = data.size() - base;
      const usize viewEnd = usize(bvOff) + bvLen;
      if (viewEnd <= data.size())
        *remaining = std::min<usize>(*remaining, viewEnd - base);
    }
    return data.data() + base;
  }
};

bool parseAccessor(const JVal& j, Accessor& a, std::string& log) {
  if (!checkedU32(j.numOr("componentType", 0), a.componentType) ||
      !checkedU32(j.numOr("count", 0), a.count)) {
    log = "accessor count/componentType out of range";
    return false;
  }
  const JVal* t = j.find("type");
  if (!t || t->type != JVal::Str) {
    log = "accessor without type";
    return false;
  }
  if (t->text == "SCALAR")
    a.perElement = 1;
  else if (t->text == "VEC2")
    a.perElement = 2;
  else if (t->text == "VEC3")
    a.perElement = 3;
  else if (t->text == "VEC4")
    a.perElement = 4;
  else if (t->text == "MAT2")
    a.perElement = 4;
  else if (t->text == "MAT3")
    a.perElement = 9;
  else if (t->text == "MAT4")
    a.perElement = 16;
  else {
    log = "unsupported accessor type " + t->text;
    return false;
  }
  const f64 bvNum = j.numOr("bufferView", 4294967295.0);
  const f64 offNum = j.numOr("byteOffset", 0.0);
  if (offNum < 0.0 || bvNum < 0.0 || bvNum > 4294967295.0 ||
      offNum > 4294967295.0) {
    log = "accessor bufferView/byteOffset out of range";
    return false;
  }
  a.bufferView = u32(bvNum);
  a.byteOffset = u32(offNum);
  a.sparse = j.find("sparse") != nullptr;
  if (a.sparse) {
    log = "sparse accessors are not supported (v1)";
    return false;
  }
  return true;
}

// Component counts (plain overload resolution; no template magic).
usize sizeOfVec(glm::vec2) { return 2; }
usize sizeOfVec(glm::vec3) { return 3; }

/// glTF references accessors by INDEX everywhere (attributes are
/// {"POSITION": 0} - the number, not the object).
const JVal* accessorAt(const Gltf& g, const JVal* ref, std::string& log) {
  if (!ref || ref->type != JVal::Num) {
    log = "accessor reference is not a number";
    return nullptr;
  }
  if (ref->number < 0 || usize(ref->number) >= g.accessors->size()) {
    log = "accessor index out of range";
    return nullptr;
  }
  return &g.accessors->at(usize(ref->number));
}

/// Read a FLOAT attribute stream (POSITION/NORMAL/TEXCOORD); v1 only
/// accepts componentType 5126 (the overwhelmingly common case for
/// these attributes - anything else gets a loud reject, not garbage).
template <typename Vec>
bool readFloatVec(Gltf& g, const JVal& acc, std::vector<Vec>& out) {
  Accessor a;
  if (!parseAccessor(acc, a, g.log)) return false;
  if (a.componentType != 5126 || a.perElement != sizeOfVec(Vec{})) {
    g.log = "attribute accessor must be FLOAT VEC" +
            std::to_string(sizeOfVec(Vec{}));
    return false;
  }
  usize remaining = 0;
  const u8* base = g.accessorBase(a, &remaining);
  if (!base) {
    g.log = "accessor out of buffer bounds";
    return false;
  }
  const usize stride = usize(a.count) * a.perElement * 4;
  if (stride > remaining) {
    g.log = "accessor data overruns its buffer";
    return false;
  }
  out.resize(a.count);
  // Byte-wise copy into the (properly aligned) std::vector - the
  // source bytes may be arbitrarily aligned inside the buffer.
  if (stride > 0) {
    std::vector<u8> tmp(stride);
    std::memcpy(tmp.data(), base, stride);
    std::memcpy(out.data(), tmp.data(), stride);
  }
  return true;
}

/// Read an index stream as u32 (accepts USHORT/UINT/UBYTE).
bool readIndices(Gltf& g, const JVal& acc, std::vector<u32>& out) {
  Accessor a;
  if (!parseAccessor(acc, a, g.log)) return false;
  usize remaining = 0;
  const u8* base = g.accessorBase(a, &remaining);
  if (!base) {
    g.log = "index accessor out of buffer bounds";
    return false;
  }
  const usize esz = componentSize(a.componentType);
  if (usize(a.count) * esz > remaining) {
    g.log = "index data overruns its buffer";
    return false;
  }
  out.resize(a.count);
  for (u32 i = 0; i < a.count; ++i) {
    u32 v = 0;
    // memcpy loads: the buffer is a byte vector - a hostile file owes
    // us no alignment for the reinterpret_cast route.
    if (a.componentType == 5123) {
      std::memcpy(&v, base + usize(i) * 2, 2);
    } else if (a.componentType == 5125) {
      std::memcpy(&v, base + usize(i) * 4, 4);
    } else if (a.componentType == 5121) {
      v = base[i];
    } else {
      g.log = "unsupported index componentType";
      return false;
    }
    out[i] = v;
  }
  return true;
}

/// Node local transform (matrix or TRS), column-major like glm.
glm::mat4 nodeMatrix(const JVal& n) {
  glm::mat4 m(1.0f);
  if (const JVal* mat = n.find("matrix");
      mat && mat->type == JVal::Arr && mat->size() == 16) {
    for (int c = 0; c < 4; ++c)
      for (int r = 0; r < 4; ++r)
        m[c][r] = f32(mat->at(usize(c) * 4 + usize(r)).number);
    return m;
  }
  glm::vec3 t(0.0f), s(1.0f);
  glm::quat q(1.0f, 0.0f, 0.0f, 0.0f);
  if (const JVal* v = n.find("translation"); v && v->size() == 3)
    t = glm::vec3(f32(v->at(0).number), f32(v->at(1).number),
                  f32(v->at(2).number));
  if (const JVal* v = n.find("scale"); v && v->size() == 3)
    s = glm::vec3(f32(v->at(0).number), f32(v->at(1).number),
                  f32(v->at(2).number));
  if (const JVal* v = n.find("rotation"); v && v->size() == 4)
    q = glm::quat(f32(v->at(3).number), f32(v->at(0).number),
                  f32(v->at(1).number), f32(v->at(2).number));
  return glm::translate(glm::mat4(1.0f), t) * glm::mat4_cast(q) *
         glm::scale(glm::mat4(1.0f), s);
}

void collectMeshNodes(Gltf& g, u32 nodeIdx, const glm::mat4& parent,
                      std::vector<std::pair<u32, glm::mat4>>& out,
                      std::vector<bool>& visited, int depth) {
  // Depth caps hostile chains; the visited set kills cycles AND the
  // exponential re-walk of shared-child DAGs; the count cap bounds the
  // collected list regardless.
  if (depth > 256 || out.size() > 4096 || visited[nodeIdx]) return;
  visited[nodeIdx] = true;
  const JVal& node = g.nodes->at(nodeIdx);
  const glm::mat4 m = parent * nodeMatrix(node);
  if (const JVal* mesh = node.find("mesh"); mesh && mesh->type == JVal::Num) {
    u32 meshIdx = 0;
    // The mesh index bounds check MUST happen here - JVal::at is the
    // unchecked operator[] and a hostile index type-confuses the DOM.
    if (!checkedU32(mesh->number, meshIdx) || meshIdx >= g.meshes->size()) {
      g.log = "node mesh index out of range";
      return;
    }
    out.emplace_back(meshIdx, m);
  }
  if (const JVal* kids = node.find("children"); kids && kids->type == JVal::Arr)
    for (const JVal& k : kids->items) {
      u32 child = 0;
      if (k.type == JVal::Num && checkedU32(k.number, child) &&
          child < g.nodes->size())
        collectMeshNodes(g, child, m, out, visited, depth + 1);
    }
}

/// Decode one glTF image (embedded bufferView / file URI / base64).
bool decodeImage(const Gltf& g, const JVal& img, GltfImage& out,
                 std::string& log) {
  std::vector<u8> bytes;
  std::string name = "albedo";
  if (const JVal* nm = img.find("name");
      nm && nm->type == JVal::Str && !nm->text.empty())
    name = nm->text;  // the builder's re-encode filename hint
  if (const JVal* bv = img.find("bufferView"); bv && bv->type == JVal::Num) {
    u32 bvi = 0;
    if (!checkedU32(bv->number, bvi) || bvi >= g.bufferViews->size()) {
      log = "image bufferView out of range";
      return false;
    }
    const JVal& b = g.bufferViews->at(bvi);
    u32 buffer = 0, off = 0, len = 0;
    if (!checkedU32(b.numOr("buffer", 0), buffer) ||
        !checkedU32(b.numOr("byteOffset", 0), off) ||
        !checkedU32(b.numOr("byteLength", 0), len)) {
      log = "image bufferView fields out of range";
      return false;
    }
    // 64-bit adds: the u32 (off + len) wraps on hostile headers.
    if (buffer >= g.buffers.size() ||
        usize(off) + usize(len) > g.buffers[buffer].size()) {
      log = "image bufferView out of bounds";
      return false;
    }
    bytes.assign(g.buffers[buffer].data() + off,
                 g.buffers[buffer].data() + off + len);
    if (const JVal* mime = img.find("mimeType");
        mime && mime->type == JVal::Str && mime->text != "image/png" &&
        mime->text != "image/jpeg" && mime->text != "image/bmp" &&
        mime->text != "image/tga")
      log = "note: unofficial mimeType " + mime->text + " (stb sniffs)";
  } else if (const JVal* uri = img.find("uri"); uri && uri->type == JVal::Str) {
    const std::string& u = uri->text;
    if (u.rfind("data:", 0) == 0) {
      const auto comma = u.find(',');
      if (comma == std::string::npos ||
          !decodeBase64(u.substr(comma + 1), bytes)) {
        log = "bad data-uri image";
        return false;
      }
    } else {
      if (!safeUri(g.dir, u, log)) return false;
      if (!readFile((std::filesystem::path(g.dir) / u).string(), bytes)) {
        log = "image file not found: " + u;
        return false;
      }
      name = std::filesystem::path(u).stem().string();
    }
  } else {
    log = "image without bufferView or uri";
    return false;
  }
  int w = 0, h = 0, comp = 0;
  stbi_uc* rgba =
      stbi_load_from_memory(bytes.data(), int(bytes.size()), &w, &h, &comp, 4);
  if (!rgba) {
    log = std::string("image decode failed: ") +
          (stbi_failure_reason() ? stbi_failure_reason() : "?");
    return false;
  }
  out.name = name;
  out.width = w;
  out.height = h;
  out.rgba.assign(rgba, rgba + usize(w) * h * 4);
  stbi_image_free(rgba);
  return true;
}

}  // namespace

bool loadGltf(const std::string& path, MeshData& mesh, GltfImage* albedo,
              std::string& log, std::vector<GltfMaterial>* materials,
              std::vector<GltfImage>* images) {
  try {
    return loadGltfImpl(path, mesh, albedo, log, materials, images);
  } catch (const std::bad_alloc&) {
    // The header contract: never throw - a hostile 20 MB "[0,0,...]"
    // JSON amplifies to >1 GB of DOM and would terminate the process.
    mesh = MeshData{};
    if (albedo) *albedo = GltfImage{};
    if (materials) materials->clear();
    if (images) images->clear();
    log = "out of memory (hostile or oversized file)";
    return false;
  }
}

bool loadGltfImpl(const std::string& path, MeshData& mesh, GltfImage* albedo,
                  std::string& log, std::vector<GltfMaterial>* materials,
                  std::vector<GltfImage>* images) {
  mesh = MeshData{};
  std::vector<u8> file;
  if (!readFile(path, file)) {
    log = "cannot open: " + path;
    return false;
  }
  if (file.size() < 12) {
    log = "file too small";
    return false;
  }
  namespace fs = std::filesystem;
  const std::string ext = fs::path(path).extension().string();
  const bool isGlb = std::memcmp(file.data(), "glTF", 4) == 0 || ext == ".glb";

  std::string jsonText;
  Gltf g;
  g.dir = fs::path(path).parent_path().string();

  if (isGlb) {
    // GLB: 12-byte header + chunks (length, type, payload). JSON first,
    // then the optional BIN chunk (buffer 0).
    const u8* p = file.data();
    const u8* end = p + file.size();
    u32 version = 0, total = 0;
    std::memcpy(&version, p + 4, 4);
    std::memcpy(&total, p + 8, 4);
    if (version != 2) {
      log = "only GLB version 2";
      return false;
    }
    if (total > file.size()) {
      log = "glb length exceeds file";
      return false;
    }
    end = p + total;
    p += 12;
    bool haveBin = false;
    while (p + 8 <= end) {
      u32 len = 0, type = 0;
      std::memcpy(&len, p, 4);
      std::memcpy(&type, p + 4, 4);
      p += 8;
      if (usize(end - p) < len) {
        log = "glb chunk overruns file";
        return false;
      }
      if (type == 0x4E4F534A) {  // "JSON"
        jsonText.assign(reinterpret_cast<const char*>(p), len);
      } else if (type == 0x004E4942) {  // "BIN\0"
        g.buffers.emplace_back(p, p + len);
        haveBin = true;
      }
      p += len + ((4 - (len % 4)) % 4);  // chunks are 4-byte aligned
    }
    if (jsonText.empty()) {
      log = "glb without a JSON chunk";
      return false;
    }
    if (!haveBin) g.buffers.emplace_back();  // buffers[0] must exist
  } else {
    jsonText.assign(reinterpret_cast<const char*>(file.data()), file.size());
  }

  JVal root;
  if (!parseJson(jsonText, root, log)) return false;
  if (root.type != JVal::Obj) {
    log = "glTF root is not an object";
    return false;
  }

  // Buffers (.gltf): buffer 0 may be the GLB BIN; others are URIs.
  if (const JVal* bufs = root.find("buffers");
      bufs && bufs->type == JVal::Arr) {
    for (usize i = 0; i < bufs->size(); ++i) {
      if (isGlb && i == 0) continue;  // the BIN chunk already loaded
      const JVal& b = bufs->at(i);
      const JVal* uri = b.find("uri");
      if (!uri || uri->type != JVal::Str || uri->text.rfind("data:", 0) == 0) {
        log = "buffer " + std::to_string(i) +
              " has no file uri (data-uri buffers unsupported, v1)";
        return false;
      }
      if (!safeUri(g.dir, uri->text, log)) return false;
      std::vector<u8> data;
      if (!readFile((fs::path(g.dir) / uri->text).string(), data)) {
        log = "buffer file not found: " + uri->text;
        return false;
      }
      g.buffers.push_back(std::move(data));
    }
  }

  g.accessors = g.arrayOrNull(root, "accessors");
  g.bufferViews = g.arrayOrNull(root, "bufferViews");
  g.images = g.arrayOrNull(root, "images");
  g.textures = g.arrayOrNull(root, "textures");
  g.materials = g.arrayOrNull(root, "materials");
  g.meshes = g.arrayOrNull(root, "meshes");
  g.nodes = g.arrayOrNull(root, "nodes");
  g.scenes = g.arrayOrNull(root, "scenes");
  if (!g.accessors || !g.bufferViews || !g.meshes || !g.nodes) {
    log = "missing accessors/bufferViews/meshes/nodes";
    return false;
  }

  // Scene walk: every mesh with its accumulated node transform. The
  // top-level "scene" index picks the default scene; the fallback only
  // walks TRUE roots (nodes no other node parents - walking every node
  // would merge child meshes twice).
  std::vector<std::pair<u32, glm::mat4>> meshNodes;
  std::vector<bool> visited(g.nodes->size(), false);
  u32 sceneIdx = 0;
  checkedU32(root.numOr("scene", 0.0), sceneIdx);
  if (g.scenes && sceneIdx < g.scenes->size()) {
    const JVal* roots = g.scenes->at(sceneIdx).find("nodes");
    if (roots)
      for (const JVal& n : roots->items) {
        u32 idx = 0;
        if (n.type == JVal::Num && checkedU32(n.number, idx) &&
            idx < g.nodes->size())
          collectMeshNodes(g, idx, glm::mat4(1.0f), meshNodes, visited, 0);
      }
  }
  if (meshNodes.empty()) {
    std::vector<bool> isChild(g.nodes->size(), false);
    for (const JVal& n : g.nodes->items)
      if (const JVal* kids = n.find("children");
          kids && kids->type == JVal::Arr)
        for (const JVal& k : kids->items) {
          u32 c = 0;
          if (k.type == JVal::Num && checkedU32(k.number, c) &&
              c < g.nodes->size())
            isChild[c] = true;
        }
    for (usize n = 0; n < g.nodes->size(); ++n)
      if (!isChild[n])
        collectMeshNodes(g, u32(n), glm::mat4(1.0f), meshNodes, visited, 0);
  }
  if (!g.log.empty() && g.log.rfind("node mesh", 0) == 0) {
    log = g.log;
    return false;
  }
  if (meshNodes.empty()) {
    log = "no meshes in the scene";
    return false;
  }

  // Merge all primitives of all meshes (v1 single MeshData). Normals
  // transform IN THIS PASS (the old second pass walked the node list
  // again and could desync its cursor against skipped primitives - an
  // OOB write on hostile files).
  std::vector<glm::vec3> positions;
  std::vector<glm::vec2> uvs;
  std::vector<glm::vec3> normals;
  std::vector<u32> indices;
  std::vector<u32> triMaterial;  // per-triangle slot into usedMaterials
  bool anyUvs = true;
  bool allNormals = true;
  int firstMaterial = -1;
  // Multi-material registry: every used material in first-use order
  // (a slot per primitive), images deduplicated by glTF image index.
  std::vector<GltfMaterial> usedMaterials;
  std::vector<GltfImage> usedImages;
  std::vector<int> imageOfMaterial;   // -1 = factor material
  std::map<u32, u32> slotOfMaterial;  // glTF material index -> slot
  std::map<u32, u32> imgSlotOf;       // glTF image index -> usedImages idx

  // glTF is right-handed Y-up by spec; the engine is right-handed
  // Z-up. R_x(+90) column-major below: (x,y,z)_glTF -> (x,-z,y) -
  // glTF up lands on engine +Z, handedness preserved (det = +1).
  const glm::mat4 rootFix(1.0f, 0.0f, 0.0f, 0.0f,   // col0: x -> x
                          0.0f, 0.0f, 1.0f, 0.0f,   // col1: y -> +Z
                          0.0f, -1.0f, 0.0f, 0.0f,  // col2: z -> -Y
                          0.0f, 0.0f, 0.0f, 1.0f);

  for (const auto& [meshIdx, nodeMraw] : meshNodes) {
    const JVal& m = g.meshes->at(meshIdx);
    const JVal* prims = m.find("primitives");
    if (!prims) continue;
    const glm::mat4 nodeM = rootFix * nodeMraw;
    const glm::mat3 normalM = glm::inverseTranspose(glm::mat3(nodeM));
    for (const JVal& prim : prims->items) {
      u32 mode = 4;
      checkedU32(prim.numOr("mode", 4.0), mode);
      if (mode != 4) {
        g.log = "note: skipped a non-triangle primitive";
        continue;
      }
      const JVal* attrs = prim.find("attributes");
      if (!attrs) continue;
      const JVal* posAccObj = accessorAt(g, attrs->find("POSITION"), g.log);
      if (!posAccObj) continue;
      std::vector<glm::vec3> pLocal;
      if (!readFloatVec(g, *posAccObj, pLocal)) {
        log = "POSITION: " + g.log;
        return false;
      }

      // glTF: attribute counts equal POSITION's count - mismatches are
      // corrupt files, reject loud (padding would poison the merge).
      std::vector<glm::vec2> uvLocal;
      const JVal* uvAcc = accessorAt(g, attrs->find("TEXCOORD_0"), g.log);
      if (uvAcc) {
        if (!readFloatVec(g, *uvAcc, uvLocal)) {
          log = "TEXCOORD_0: " + g.log;
          return false;
        }
        if (uvLocal.size() != pLocal.size()) {
          log = "TEXCOORD_0 count != POSITION count";
          return false;
        }
      } else {
        anyUvs = false;
      }
      std::vector<glm::vec3> nLocal;
      const JVal* nrmAcc = accessorAt(g, attrs->find("NORMAL"), g.log);
      if (nrmAcc) {
        if (!readFloatVec(g, *nrmAcc, nLocal)) {
          log = "NORMAL: " + g.log;
          return false;
        }
        if (nLocal.size() != pLocal.size()) {
          log = "NORMAL count != POSITION count";
          return false;
        }
      } else {
        allNormals = false;
      }

      // Index VALUES validated against the vertex count here - a
      // hostile file with an out-of-range index would otherwise read
      // (computed normals) and write OOB.
      std::vector<u32> idx;
      if (const JVal* idxRef = prim.find("indices")) {
        const JVal* idxAcc = accessorAt(g, idxRef, g.log);
        if (!idxAcc) {
          log = "indices: " + g.log;
          return false;
        }
        if (!readIndices(g, *idxAcc, idx)) {
          log = "indices: " + g.log;
          return false;
        }
        for (u32 i : idx)
          if (i >= pLocal.size()) {
            log = "index value out of vertex range";
            return false;
          }
      } else {
        idx.resize(pLocal.size());
        for (u32 i = 0; i < pLocal.size(); ++i) idx[i] = i;
      }

      const u32 base = u32(positions.size());
      for (const glm::vec3& v : pLocal)
        positions.push_back(glm::vec3(nodeM * glm::vec4(v, 1.0f)));
      if (uvAcc)
        uvs.insert(uvs.end(), uvLocal.begin(), uvLocal.end());
      else
        uvs.resize(uvs.size() + pLocal.size(), glm::vec2(0.0f));
      if (nrmAcc) {
        for (const glm::vec3& n : nLocal)
          normals.push_back(glm::normalize(normalM * n));
      } else {
        normals.resize(normals.size() + pLocal.size(), glm::vec3(0.0f));
      }
      for (u32 i : idx) indices.push_back(base + i);

      // This primitive's material -> registry slot; every triangle
      // gets tagged (cluster majority assignment happens in the DAG).
      u32 matSlot = 0;
      const JVal* mat = prim.find("material");
      u32 mi = 0;
      if (mat && mat->type == JVal::Num && checkedU32(mat->number, mi) &&
          g.materials && mi < g.materials->size()) {
        if (firstMaterial < 0) firstMaterial = int(mi);
        auto it = slotOfMaterial.find(mi);
        if (it == slotOfMaterial.end()) {
          // New slot: pull the PBR params + base-color texture ref.
          const JVal& jm = g.materials->at(mi);
          GltfMaterial gm;
          int imgIdx = -1;
          if (const JVal* pbr = jm.find("pbrMetallicRoughness")) {
            if (const JVal* f = pbr->find("baseColorFactor");
                f && f->size() >= 3)
              gm.kd = glm::vec3(f32(f->at(0).number), f32(f->at(1).number),
                                f32(f->at(2).number));
            gm.roughness = f32(pbr->numOr("roughnessFactor", 1.0));
            gm.metallic = f32(pbr->numOr("metallicFactor", 1.0));
            if (const JVal* bt = pbr->find("baseColorTexture");
                bt && g.textures && g.images) {
              u32 texIdx = 0xFFFFFFFFu, imgIdxU = 0xFFFFFFFFu;
              if (checkedU32(bt->numOr("index", 4294967295.0), texIdx) &&
                  texIdx < g.textures->size() &&
                  checkedU32(
                      g.textures->at(texIdx).numOr("source", 4294967295.0),
                      imgIdxU) &&
                  imgIdxU < g.images->size())
                imgIdx = int(imgIdxU);
            }
          }
          if (imgIdx >= 0) {
            auto imgIt = imgSlotOf.find(u32(imgIdx));
            if (imgIt == imgSlotOf.end()) {
              GltfImage dec;
              std::string imgLog;
              if (decodeImage(g, g.images->at(u32(imgIdx)), dec, imgLog))
                imgIt = imgSlotOf.emplace(u32(imgIdx), u32(usedImages.size()))
                            .first,
                usedImages.push_back(std::move(dec));
              else
                log = log.empty() ? imgLog : log + "; " + imgLog;
            }
            if (imgSlotOf.count(u32(imgIdx))) {
              gm.image = int(imgSlotOf.at(u32(imgIdx)));
              if (usedImages[gm.image].name.empty())
                usedImages[gm.image].name =
                    "mat" + std::to_string(usedMaterials.size());
            }
          }
          matSlot = u32(usedMaterials.size());
          usedMaterials.push_back(gm);
          imageOfMaterial.push_back(gm.image);
          slotOfMaterial.emplace(mi, matSlot);
        } else {
          matSlot = it->second;
        }
      }
      triMaterial.resize(triMaterial.size() + idx.size() / 3, matSlot);
    }
  }

  if (positions.empty() || indices.size() < 3) {
    log = "no triangle geometry found";
    return false;
  }
  if (!allNormals) {
    // Any primitive missing normals: compute smooth face-accumulation
    // normals for the whole merged mesh (mixing transformed and
    // computed normals across primitives would shade inconsistently).
    std::fill(normals.begin(), normals.end(), glm::vec3(0.0f));
    for (usize t = 0; t + 2 < indices.size(); t += 3) {
      const glm::vec3& a = positions[indices[t]];
      const glm::vec3& b = positions[indices[t + 1]];
      const glm::vec3& c = positions[indices[t + 2]];
      const glm::vec3 fn = glm::cross(b - a, c - a);
      normals[indices[t]] += fn;
      normals[indices[t + 1]] += fn;
      normals[indices[t + 2]] += fn;
    }
  }
  // Degenerate (zero) normals normalize to NaN - give them a safe
  // fallback direction instead of poisoning the pipeline.
  for (glm::vec3& n : normals)
    n = glm::dot(n, n) > 1e-20f ? glm::normalize(n)
                                : glm::vec3(0.0f, 0.0f, 1.0f);

  // Into MeshData (SOA + u32 indices).
  mesh.positions.reserve(positions.size() * 3);
  for (const glm::vec3& v : positions)
    mesh.positions.insert(mesh.positions.end(), {v.x, v.y, v.z});
  mesh.normals.reserve(normals.size() * 3);
  for (const glm::vec3& v : normals)
    mesh.normals.insert(mesh.normals.end(), {v.x, v.y, v.z});
  mesh.indices = indices;
  if (anyUvs) {
    mesh.uvs.reserve(uvs.size() * 2);
    for (const glm::vec2& v : uvs) mesh.uvs.insert(mesh.uvs.end(), {v.x, v.y});
  }

  // Material (the first referenced one fills MeshData.material +
  // the `albedo` out for single-material consumers; the full set goes
  // to the multi-material outs below).
  if (firstMaterial >= 0 && slotOfMaterial.count(u32(firstMaterial))) {
    const u32 s = slotOfMaterial.at(u32(firstMaterial));
    mesh.material.kd = usedMaterials[s].kd;
    mesh.material.ka = glm::vec3(0.0f);
    mesh.material.roughness = usedMaterials[s].roughness;
    mesh.material.metallic = usedMaterials[s].metallic;
    if (albedo && imageOfMaterial[s] >= 0)
      *albedo = usedImages[u32(imageOfMaterial[s])];
  }
  if (usedMaterials.size() > 1) {
    mesh.triMaterial = std::move(triMaterial);
  } else if (!triMaterial.empty()) {
    // Single material: keep MeshData's "empty = material 0" shape.
    triMaterial.clear();
  }
  if (materials) *materials = std::move(usedMaterials);
  if (images) *images = std::move(usedImages);
  if (!anyUvs && albedo) *albedo = GltfImage{};  // no UVs -> untextured
  if (!anyUvs)
    log = log.empty() ? "note: no TEXCOORD_0 - untextured"
                      : log + "; note: no TEXCOORD_0 - untextured";
  return true;
}

}  // namespace assets

KENSHIN_NS_END
