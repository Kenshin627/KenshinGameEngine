#include "kenshin/assets/GltfLoader.h"
#include "kenshin/assets/GltfScene.h"
#include "kenshin/util/File.h"

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

#ifdef _MSC_VER

#pragma warning(push, 0)

#endif
#include "stb/stb_image.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif
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

  /// Reads the 4 hex digits after `p` (which points at 'u'); leaves `p`
  /// on the last digit.
  bool hex4(u32& out) {
    if (end - p < 5) return false;
    out = 0;
    for (int i = 1; i <= 4; ++i) {
      const char c = p[i];
      const int v = c >= '0' && c <= '9'   ? c - '0'
                    : c >= 'a' && c <= 'f' ? c - 'a' + 10
                    : c >= 'A' && c <= 'F' ? c - 'A' + 10
                                           : -1;
      if (v < 0) return false;
      out = out * 16 + u32(v);
    }
    p += 4;
    return true;
  }
  static void appendUtf8(std::string& s, u32 cp) {
    if (cp < 0x80) {
      s += char(cp);
    } else if (cp < 0x800) {
      s += char(0xC0 | (cp >> 6));
      s += char(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      s += char(0xE0 | (cp >> 12));
      s += char(0x80 | ((cp >> 6) & 0x3F));
      s += char(0x80 | (cp & 0x3F));
    } else {
      s += char(0xF0 | (cp >> 18));
      s += char(0x80 | ((cp >> 12) & 0x3F));
      s += char(0x80 | ((cp >> 6) & 0x3F));
      s += char(0x80 | (cp & 0x3F));
    }
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
            // \uXXXX (+ a low surrogate after a high one) -> UTF-8: node
            // and material names are user text in any script.
            u32 cp = 0;
            if (!hex4(cp)) return fail("bad \\u escape");
            if (cp >= 0xD800 && cp <= 0xDBFF) {
              u32 lo = 0;
              if (end - p < 7 || p[1] != '\\' || p[2] != 'u') return fail("lone surrogate");
              p += 2;
              if (!hex4(lo) || lo < 0xDC00 || lo > 0xDFFF) return fail("bad surrogate pair");
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
              return fail("lone surrogate");
            }
            appendUtf8(out, cp);
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
  // GLB pads its JSON chunk with spaces (some exporters use NULs);
  // anything else after the root value means a corrupt or concatenated
  // file.
  while (j.p < j.end && (*j.p == ' ' || *j.p == '\t' || *j.p == '\n' || *j.p == '\r' ||
                         *j.p == '\0'))
    ++j.p;
  if (j.p != j.end) {
    err = "json: trailing data after the root value";
    return false;
  }
  return true;
}

// ---- small utilities ---------------------------------------------

bool readFile(const std::string& path, std::vector<u8>& out) {
  FILE* f = nullptr;
  if ((f = kenshin::util::openFile(path.c_str(), "rb")) == nullptr) return false;
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

/// Untrusted-model URI guard: percent-decode, then refuse schemes,
/// absolute paths and parent-directory escapes (an image or buffer URI
/// must stay inside the model's own directory tree). `file` receives
/// the DECODED path under `dir` - the one to open (a spec-encoded
/// "my%20image.png" names the file "my image.png").
bool resolveUri(const std::string& dir, const std::string& raw,
                std::filesystem::path& file, std::string& log) {
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
  if (uri.find(':') != std::string::npos) {  // http:, file:, C:, ...
    log = "uri with scheme or drive refused: " + uri;
    return false;
  }
  const std::filesystem::path p = util::pathFromUtf8(uri);
  if (p.is_absolute()) {
    log = "absolute uri refused: " + uri;
    return false;
  }
  for (const auto& seg : p)
    if (seg == "..") {
      log = "parent-directory uri refused: " + uri;
      return false;
    }
  file = util::pathFromUtf8(dir) / p;
  return true;
}

// ---- glTF context ------------------------------------------------

struct Accessor {
  u32 componentType = 0;  // 5120..5126
  u32 count = 0;
  u32 perElement = 0;  // SCALAR=1 .. MAT4=16
  u32 bufferView = 0xFFFFFFFFu;  // none = all zeros (+ sparse values)
  u32 byteOffset = 0;
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
  /// Every file read besides the document itself (buffers, images):
  /// the asset database's dependency list.
  mutable std::vector<std::string> externalFiles;

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
  return true;
}

/// The accessor's elements as densely packed bytes (1.3e): the
/// bufferView's byteStride resolved (interleaved vertex data), sparse
/// substitutions applied, everything bounds-checked. Without a
/// bufferView the elements start as zeros (glTF 3.6.2.3).
bool accessorBytes(const Gltf& g, const JVal& j, const Accessor& a, std::vector<u8>& out,
                   std::string& log) {
  const usize elem = usize(componentSize(a.componentType)) * a.perElement;
  out.assign(usize(a.count) * elem, 0);
  if (a.bufferView != 0xFFFFFFFFu) {
    usize remaining = 0;
    const u8* base = g.accessorBase(a, &remaining);
    if (!base) {
      log = "accessor out of buffer bounds";
      return false;
    }
    u32 stride = 0;
    if (!checkedU32(g.bufferViews->at(a.bufferView).numOr("byteStride", 0), stride)) {
      log = "bufferView byteStride out of range";
      return false;
    }
    if (stride == 0) stride = u32(elem);
    if (stride < elem) {
      log = "bufferView byteStride is smaller than the accessor's element";
      return false;
    }
    if (a.count > 0 && usize(a.count - 1) * stride + elem > remaining) {
      log = "accessor data overruns its buffer";
      return false;
    }
    // Byte-wise copies: the source may be arbitrarily aligned.
    for (u32 i = 0; i < a.count; ++i)
      std::memcpy(out.data() + usize(i) * elem, base + usize(i) * stride, elem);
  }
  const JVal* sp = j.find("sparse");
  if (sp == nullptr) return true;
  u32 n = 0;
  if (!checkedU32(sp->numOr("count", 0), n) || n == 0 || n > a.count) {
    log = "sparse count out of range";
    return false;
  }
  const JVal* si = sp->find("indices");
  const JVal* sv = sp->find("values");
  if (!si || !sv || si->type != JVal::Obj || sv->type != JVal::Obj) {
    log = "sparse accessor without indices/values";
    return false;
  }
  // Both are tightly packed bufferView ranges (no byteStride by spec).
  const auto view = [&](const JVal& v, Accessor& r, usize bytes) -> const u8* {
    if (!checkedU32(v.numOr("bufferView", 4294967295.0), r.bufferView) ||
        r.bufferView >= g.bufferViews->size() ||
        !checkedU32(v.numOr("byteOffset", 0), r.byteOffset))
      return nullptr;
    usize remaining = 0;
    const u8* p = g.accessorBase(r, &remaining);
    return p && bytes <= remaining ? p : nullptr;
  };
  Accessor ia;
  if (!checkedU32(si->numOr("componentType", 0), ia.componentType) ||
      (ia.componentType != 5121 && ia.componentType != 5123 && ia.componentType != 5125)) {
    log = "sparse indices must be UNSIGNED_BYTE/SHORT/INT";
    return false;
  }
  const usize isz = componentSize(ia.componentType);
  const u8* ib = view(*si, ia, usize(n) * isz);
  Accessor va;
  const u8* vb = view(*sv, va, usize(n) * elem);
  if (!ib || !vb) {
    log = "sparse indices/values out of buffer bounds";
    return false;
  }
  u32 prev = 0;
  for (u32 k = 0; k < n; ++k) {
    u32 idx = 0;
    std::memcpy(&idx, ib + usize(k) * isz, isz);  // little-endian, zero-extended
    if (idx >= a.count || (k > 0 && idx <= prev)) {
      log = "sparse indices must be increasing and inside the accessor";
      return false;
    }
    prev = idx;
    std::memcpy(out.data() + usize(idx) * elem, vb + usize(k) * elem, elem);
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
  std::vector<u8> bytes;
  if (!accessorBytes(g, acc, a, bytes, g.log)) return false;
  out.resize(a.count);
  if (!bytes.empty()) std::memcpy(out.data(), bytes.data(), bytes.size());
  return true;
}

/// Read an index stream as u32 (accepts USHORT/UINT/UBYTE).
bool readIndices(Gltf& g, const JVal& acc, std::vector<u32>& out) {
  Accessor a;
  if (!parseAccessor(acc, a, g.log)) return false;
  if (a.perElement != 1) {
    g.log = "index accessor must be SCALAR";
    return false;
  }
  std::vector<u8> bytes;
  if (!accessorBytes(g, acc, a, bytes, g.log)) return false;
  const u8* base = bytes.data();
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

/// The encoded bytes of one glTF image (embedded bufferView / file URI
/// / base64 data URI) + a file-stem name suggestion and its mimeType.
bool imageBytes(const Gltf& g, const JVal& img, std::vector<u8>& bytes,
                std::string& name, std::string& mime, std::string& log) {
  name = "albedo";
  mime.clear();
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
    if (const JVal* mt = img.find("mimeType"); mt && mt->type == JVal::Str) {
      mime = mt->text;
      if (mime != "image/png" && mime != "image/jpeg" && mime != "image/bmp" &&
          mime != "image/tga")
        log = "note: unofficial mimeType " + mime + " (stb sniffs)";
    }
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
      std::filesystem::path file;
      if (!resolveUri(g.dir, u, file, log)) return false;
      if (!readFile(util::pathToUtf8(file), bytes)) {
        log = "image file not found: " + u;
        return false;
      }
      g.externalFiles.push_back(util::pathToUtf8(file));
      name = util::pathToUtf8(file.stem());
    }
  } else {
    log = "image without bufferView or uri";
    return false;
  }
  return true;
}

/// Decode one glTF image (embedded bufferView / file URI / base64).
bool decodeImage(const Gltf& g, const JVal& img, GltfImage& out,
                 std::string& log) {
  std::vector<u8> bytes;
  std::string name, mime;
  if (!imageBytes(g, img, bytes, name, mime, log)) return false;
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

/// One triangle primitive's LOCAL attribute streams (glTF frame).
struct PrimitiveData {
  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> normals;  // empty = absent
  std::vector<glm::vec2> uvs;      // empty = absent
  std::vector<u32> indices;        // validated against the vertex count
  bool hasNormals = false;
  bool hasUvs = false;
};

enum class PrimitiveResult { Ok, Skipped, Error };

/// Reads a primitive's POSITION / NORMAL / TEXCOORD_0 / indices. Non-
/// triangle primitives and ones without positions are Skipped (with a
/// note in g.log); malformed data is an Error (reason in `log`).
PrimitiveResult readPrimitive(Gltf& g, const JVal& prim, PrimitiveData& out,
                              std::string& log) {
  out = PrimitiveData{};
  u32 mode = 4;
  checkedU32(prim.numOr("mode", 4.0), mode);
  if (mode != 4) {
    g.log = "note: skipped a non-triangle primitive";
    return PrimitiveResult::Skipped;
  }
  const JVal* attrs = prim.find("attributes");
  if (!attrs) return PrimitiveResult::Skipped;
  const JVal* posAccObj = accessorAt(g, attrs->find("POSITION"), g.log);
  if (!posAccObj) return PrimitiveResult::Skipped;
  if (!readFloatVec(g, *posAccObj, out.positions)) {
    log = "POSITION: " + g.log;
    return PrimitiveResult::Error;
  }
  // glTF: attribute counts equal POSITION's count - mismatches are
  // corrupt files, reject loud (padding would poison the merge).
  const JVal* uvAcc = accessorAt(g, attrs->find("TEXCOORD_0"), g.log);
  if (uvAcc) {
    if (!readFloatVec(g, *uvAcc, out.uvs)) {
      log = "TEXCOORD_0: " + g.log;
      return PrimitiveResult::Error;
    }
    if (out.uvs.size() != out.positions.size()) {
      log = "TEXCOORD_0 count != POSITION count";
      return PrimitiveResult::Error;
    }
    out.hasUvs = true;
  }
  const JVal* nrmAcc = accessorAt(g, attrs->find("NORMAL"), g.log);
  if (nrmAcc) {
    if (!readFloatVec(g, *nrmAcc, out.normals)) {
      log = "NORMAL: " + g.log;
      return PrimitiveResult::Error;
    }
    if (out.normals.size() != out.positions.size()) {
      log = "NORMAL count != POSITION count";
      return PrimitiveResult::Error;
    }
    out.hasNormals = true;
  }
  // Index VALUES validated against the vertex count here - a hostile
  // file with an out-of-range index would otherwise read (computed
  // normals) and write OOB.
  if (const JVal* idxRef = prim.find("indices")) {
    const JVal* idxAcc = accessorAt(g, idxRef, g.log);
    if (!idxAcc) {
      log = "indices: " + g.log;
      return PrimitiveResult::Error;
    }
    if (!readIndices(g, *idxAcc, out.indices)) {
      log = "indices: " + g.log;
      return PrimitiveResult::Error;
    }
    for (u32 i : out.indices)
      if (i >= out.positions.size()) {
        log = "index value out of vertex range";
        return PrimitiveResult::Error;
      }
  } else {
    out.indices.resize(out.positions.size());
    for (u32 i = 0; i < out.positions.size(); ++i) out.indices[i] = i;
  }
  return PrimitiveResult::Ok;
}

/// Reads the file and parses the JSON (+ GLB chunks, + external
/// buffers) into `root`/`g`. `root` must outlive `g` (g points into it).
bool openDocument(const std::string& path, JVal& root, Gltf& g, std::string& log) {
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
  g.dir = util::pathToUtf8(util::pathFromUtf8(path).parent_path());

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
      // Chunks are 4-byte aligned; a hostile length near the end must
      // not step the cursor past `end`.
      const usize pad = (4 - (len % 4)) % 4;
      if (usize(end - p) < usize(len) + pad) break;
      p += len + pad;
    }
    if (jsonText.empty()) {
      log = "glb without a JSON chunk";
      return false;
    }
    if (!haveBin) g.buffers.emplace_back();  // buffers[0] must exist
  } else {
    jsonText.assign(reinterpret_cast<const char*>(file.data()), file.size());
  }

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
      fs::path bufferFile;
      if (!resolveUri(g.dir, uri->text, bufferFile, log)) return false;
      std::vector<u8> data;
      if (!readFile(util::pathToUtf8(bufferFile), data)) {
        log = "buffer file not found: " + uri->text;
        return false;
      }
      g.externalFiles.push_back(util::pathToUtf8(bufferFile));
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
  JVal root;
  Gltf g;
  if (!openDocument(path, root, g, log)) return false;

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
      PrimitiveData pd;
      const PrimitiveResult pr = readPrimitive(g, prim, pd, log);
      if (pr == PrimitiveResult::Error) return false;
      if (pr == PrimitiveResult::Skipped) continue;
      const std::vector<glm::vec3>& pLocal = pd.positions;
      const std::vector<glm::vec2>& uvLocal = pd.uvs;
      const std::vector<glm::vec3>& nLocal = pd.normals;
      const std::vector<u32>& idx = pd.indices;
      if (!pd.hasUvs) anyUvs = false;
      if (!pd.hasNormals) allNormals = false;

      const u32 base = u32(positions.size());
      for (const glm::vec3& v : pLocal)
        positions.push_back(glm::vec3(nodeM * glm::vec4(v, 1.0f)));
      if (pd.hasUvs)
        uvs.insert(uvs.end(), uvLocal.begin(), uvLocal.end());
      else
        uvs.resize(uvs.size() + pLocal.size(), glm::vec2(0.0f));
      if (pd.hasNormals) {
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
          // glTF: baseColorFactor defaults to WHITE (the texture alone
          // is the albedo when no factor is given) - MeshData's 0.8
          // default would double-attenuate textured materials into a
          // washed warm gray after lighting.
          GltfMaterial gm;
          gm.kd = glm::vec3(1.0f);
          if (const JVal* ds = jm.find("doubleSided");
              ds && ds->type == JVal::Bool)
            gm.doubleSided = ds->boolean;
          int imgIdx = -1;
          if (const JVal* pbr = jm.find("pbrMetallicRoughness")) {
            if (const JVal* f = pbr->find("baseColorFactor");
                f && f->size() >= 3) {
              gm.kd = glm::vec3(f32(f->at(0).number), f32(f->at(1).number),
                                f32(f->at(2).number));
              if (f->size() >= 4) gm.alpha = f32(f->at(3).number);
            }
            // Only BLEND carries alpha; OPAQUE/MASK ignore the factor's
            // w per spec.
            if (const JVal* am = jm.find("alphaMode");
                !(am && am->type == JVal::Str && am->text == "BLEND"))
              gm.alpha = 1.0f;
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
          auto slotForImage = [&](int imageIdx) -> int {
            if (imageIdx < 0 || !g.images) return -1;
            auto imgIt = imgSlotOf.find(u32(imageIdx));
            if (imgIt == imgSlotOf.end()) {
              GltfImage dec;
              std::string imgLog;
              if (decodeImage(g, g.images->at(u32(imageIdx)), dec, imgLog))
                imgIt = imgSlotOf.emplace(u32(imageIdx), u32(usedImages.size()))
                            .first,
                usedImages.push_back(std::move(dec));
              else
                log = log.empty() ? imgLog : log + "; " + imgLog;
            }
            if (!imgSlotOf.count(u32(imageIdx))) return -1;
            return int(imgSlotOf.at(u32(imageIdx)));
          };
          gm.image = slotForImage(imgIdx);
          if (gm.image >= 0 && usedImages[usize(gm.image)].name.empty())
            usedImages[usize(gm.image)].name =
                "mat" + std::to_string(usedMaterials.size());
          // metallicRoughnessTexture: roughness rides G, metallic B
          // (LINEAR data - never sRGB-decoded).
          if (const JVal* pbrMr = jm.find("pbrMetallicRoughness");
              pbrMr && g.textures && g.images) {
            if (const JVal* mr = pbrMr->find("metallicRoughnessTexture"); mr) {
              u32 mrTex = 0xFFFFFFFFu, mrImg = 0xFFFFFFFFu;
              if (checkedU32(mr->numOr("index", 4294967295.0), mrTex) &&
                  mrTex < g.textures->size() &&
                  checkedU32(
                      g.textures->at(mrTex).numOr("source", 4294967295.0),
                      mrImg) &&
                  mrImg < g.images->size()) {
                const int slot = slotForImage(int(mrImg));
                if (slot >= 0) {
                  gm.mrImage = slot;
                  if (usedImages[usize(slot)].name.empty())
                    usedImages[usize(slot)].name =
                        "mat" + std::to_string(usedMaterials.size()) + "_mr";
                }
              }
            }
          }
          if (const JVal* ef = jm.find("emissiveFactor"); ef && ef->size() >= 3)
            gm.emissive =
                glm::vec3(f32(ef->at(0).number), f32(ef->at(1).number),
                          f32(ef->at(2).number));
          if (const JVal* matExt = jm.find("extensions"))
            if (const JVal* es =
                    matExt->find("KHR_materials_emissive_strength"))
              gm.emissive *= f32(es->numOr("emissiveStrength", 1.0));
          if (!(gm.emissive.x >= 0.0f && gm.emissive.y >= 0.0f &&
                gm.emissive.z >= 0.0f && gm.emissive.x < 1e4f &&
                gm.emissive.y < 1e4f && gm.emissive.z < 1e4f))
            gm.emissive = glm::vec3(0.0f);  // hostile/garbage factor
          // normalTexture (tangent-space; LINEAR data). texCoord != 0 is
          // not supported (one UV set) - warn when the asset asks for it.
          if (const JVal* nt = jm.find("normalTexture");
              nt && g.textures && g.images) {
            u32 nTex = 0xFFFFFFFFu, nImg = 0xFFFFFFFFu;
            if (checkedU32(nt->numOr("index", 4294967295.0), nTex) &&
                nTex < g.textures->size() &&
                checkedU32(g.textures->at(nTex).numOr("source", 4294967295.0),
                           nImg) &&
                nImg < g.images->size()) {
              const int slot = slotForImage(int(nImg));
              if (slot >= 0) {
                gm.nrmImage = slot;
                if (usedImages[usize(slot)].name.empty())
                  usedImages[usize(slot)].name =
                      "mat" + std::to_string(usedMaterials.size()) + "_nrm";
              }
            }
          }
          // occlusionTexture (AO in R; LINEAR). glTF: ao' = 1 +
          // strength * (ao - 1), applied to INDIRECT light only.
          if (const JVal* ot = jm.find("occlusionTexture");
              ot && g.textures && g.images) {
            u32 oTex = 0xFFFFFFFFu, oImg = 0xFFFFFFFFu;
            if (checkedU32(ot->numOr("index", 4294967295.0), oTex) &&
                oTex < g.textures->size() &&
                checkedU32(g.textures->at(oTex).numOr("source", 4294967295.0),
                           oImg) &&
                oImg < g.images->size()) {
              const int slot = slotForImage(int(oImg));
              if (slot >= 0) {
                gm.aoImage = slot;
                const f32 st = f32(ot->numOr("strength", 1.0));
                gm.aoStrength = st >= 0.0f ? std::min(st, 1.0f) : 0.0f;
                if (usedImages[usize(slot)].name.empty())
                  usedImages[usize(slot)].name =
                      "mat" + std::to_string(usedMaterials.size()) + "_ao";
              }
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

// ════════════════════════════════════════════════════════════════
// Scene-graph import (roadmap 1.3b) - see GltfScene.h.
// ════════════════════════════════════════════════════════════════

namespace {

/// glTF (right-handed Y-up) -> engine (right-handed Z-up): R_x(+90),
/// (x, y, z) -> (x, -z, y). A rotation, so R^-1 = R^T.
const glm::mat4 kGltfToEngine(1.0f, 0.0f, 0.0f, 0.0f,   // col0: x -> x
                              0.0f, 0.0f, 1.0f, 0.0f,   // col1: y -> +Z
                              0.0f, -1.0f, 0.0f, 0.0f,  // col2: z -> -Y
                              0.0f, 0.0f, 0.0f, 1.0f);

/// Extensions a file may REQUIRE and still load correctly here.
bool requiredExtensionSupported(const std::string& name) {
  return name == "KHR_materials_emissive_strength";
}

struct SceneBuilder {
  Gltf& g;
  const JVal& root;
  GltfScene& out;
  std::map<u32, int> meshOf;      // glTF mesh -> scene mesh (first use)
  std::map<u32, int> materialOf;  // glTF material -> scene material
  std::map<u32, int> imageOf;     // glTF image -> scene image (-1 = failed)
  std::string error;

  void note(const std::string& n) {
    if (std::find(out.notes.begin(), out.notes.end(), n) == out.notes.end())
      out.notes.push_back(n);
  }

  static std::string nameOr(const JVal& j, const std::string& fallback) {
    const JVal* n = j.find("name");
    return n && n->type == JVal::Str && !n->text.empty() ? n->text : fallback;
  }

  static backend::WrapMode wrapOf(f64 v) {
    if (v == 33071.0) return backend::WrapMode::Clamp;   // CLAMP_TO_EDGE
    if (v == 33648.0) return backend::WrapMode::Mirror;  // MIRRORED_REPEAT
    return backend::WrapMode::Repeat;                    // 10497 / default
  }

  int image(u32 gi) {
    if (const auto it = imageOf.find(gi); it != imageOf.end()) return it->second;
    GltfSceneImage img;
    std::string log;
    int index = -1;
    if (g.images && gi < g.images->size() &&
        imageBytes(g, g.images->at(gi), img.encoded, img.name, img.mimeType, log)) {
      if (img.name.empty() || img.name == "albedo")
        img.name = nameOr(g.images->at(gi), "image" + std::to_string(gi));
      index = static_cast<int>(out.images.size());
      out.images.push_back(std::move(img));
      if (!log.empty()) note(log);
    } else {
      note("image " + std::to_string(gi) + " skipped: " + (log.empty() ? "missing" : log));
    }
    imageOf.emplace(gi, index);
    return index;
  }

  /// A material's textureInfo -> image + sampler + UV set.
  GltfTextureRef texture(const JVal* info, const char* role) {
    GltfTextureRef ref;
    if (!info || info->type != JVal::Obj) return ref;
    u32 ti = 0;
    if (!checkedU32(info->numOr("index", -1.0), ti) || !g.textures || ti >= g.textures->size()) {
      note(std::string(role) + ": texture index out of range");
      return ref;
    }
    const JVal& tex = g.textures->at(ti);
    u32 src = 0;
    if (!checkedU32(tex.numOr("source", -1.0), src)) {
      // KHR_texture_basisu / EXT_texture_webp / ... put the source in
      // an extension: not decodable here.
      note(std::string(role) + ": texture without a core image source (extension) ignored");
      return ref;
    }
    ref.image = image(src);
    u32 si = 0;
    const JVal* samplers = root.find("samplers");
    if (checkedU32(tex.numOr("sampler", -1.0), si) && samplers && samplers->type == JVal::Arr &&
        si < samplers->size()) {
      const JVal& s = samplers->at(si);
      ref.sampler.wrapU = wrapOf(s.numOr("wrapS", 10497.0));
      ref.sampler.wrapV = wrapOf(s.numOr("wrapT", 10497.0));
      if (s.numOr("magFilter", 9729.0) == 9728.0)  // NEAREST
        ref.sampler.filter = backend::FilterMode::Nearest;
    }
    checkedU32(info->numOr("texCoord", 0.0), ref.texCoord);
    if (ref.texCoord != 0)
      note(std::string(role) + " uses TEXCOORD_" + std::to_string(ref.texCoord) +
           " (only TEXCOORD_0 is imported; W7)");
    if (const JVal* ext = info->find("extensions"); ext && ext->find("KHR_texture_transform"))
      note(std::string(role) + ": KHR_texture_transform ignored (W7)");
    return ref;
  }

  static f32 unit(f64 v, f32 fallback) {
    return std::isfinite(v) ? std::clamp(f32(v), 0.0f, 1.0f) : fallback;
  }

  int material(u32 gm) {
    if (const auto it = materialOf.find(gm); it != materialOf.end()) return it->second;
    const JVal& jm = g.materials->at(gm);
    GltfSceneMaterial m;
    m.name = nameOr(jm, "Material " + std::to_string(gm));
    if (const JVal* pbr = jm.find("pbrMetallicRoughness"); pbr && pbr->type == JVal::Obj) {
      if (const JVal* f = pbr->find("baseColorFactor"); f && f->type == JVal::Arr && f->size() == 4)
        for (int c = 0; c < 4; ++c) m.baseColor[c] = unit(f->at(usize(c)).number, 1.0f);
      m.metallic = unit(pbr->numOr("metallicFactor", 1.0), 1.0f);
      m.roughness = unit(pbr->numOr("roughnessFactor", 1.0), 1.0f);
      m.baseColorTexture = texture(pbr->find("baseColorTexture"), "baseColorTexture");
      m.metallicRoughnessTexture =
          texture(pbr->find("metallicRoughnessTexture"), "metallicRoughnessTexture");
    }
    if (const JVal* nt = jm.find("normalTexture")) {
      m.normalTexture = texture(nt, "normalTexture");
      const f64 sc = nt->numOr("scale", 1.0);
      m.normalScale = std::isfinite(sc) ? f32(sc) : 1.0f;
    }
    if (const JVal* ot = jm.find("occlusionTexture")) {
      m.occlusionTexture = texture(ot, "occlusionTexture");
      m.occlusionStrength = unit(ot->numOr("strength", 1.0), 1.0f);
    }
    m.emissiveTexture = texture(jm.find("emissiveTexture"), "emissiveTexture");
    if (const JVal* ef = jm.find("emissiveFactor"); ef && ef->type == JVal::Arr && ef->size() == 3)
      for (int c = 0; c < 3; ++c) m.emissive[c] = unit(ef->at(usize(c)).number, 0.0f);
    if (const JVal* ext = jm.find("extensions")) {
      if (const JVal* es = ext->find("KHR_materials_emissive_strength")) {
        const f64 k = es->numOr("emissiveStrength", 1.0);
        if (std::isfinite(k) && k >= 0.0 && k < 1e4) m.emissive *= f32(k);
      }
      for (const auto& [name, value] : ext->members)
        if (name != "KHR_materials_emissive_strength")
          note("material extension " + name + " ignored");
    }
    if (const JVal* am = jm.find("alphaMode"); am && am->type == JVal::Str) {
      if (am->text == "MASK") m.alphaMode = GltfSceneMaterial::AlphaMode::Mask;
      else if (am->text == "BLEND") m.alphaMode = GltfSceneMaterial::AlphaMode::Blend;
    }
    const f64 cutoff = jm.numOr("alphaCutoff", 0.5);
    m.alphaCutoff = std::isfinite(cutoff) && cutoff >= 0.0 ? f32(cutoff) : 0.5f;
    if (const JVal* ds = jm.find("doubleSided"); ds && ds->type == JVal::Bool)
      m.doubleSided = ds->boolean;
    const int index = static_cast<int>(out.materials.size());
    out.materials.push_back(std::move(m));
    materialOf.emplace(gm, index);
    return index;
  }

  /// glTF mesh -> scene mesh (primitives merged, mesh-local, engine
  /// frame). -1 when it has no triangle geometry; `error` set on
  /// malformed data.
  int mesh(u32 gmIndex) {
    if (const auto it = meshOf.find(gmIndex); it != meshOf.end()) return it->second;
    const JVal& jm = g.meshes->at(gmIndex);
    GltfSceneMesh sm;
    sm.name = nameOr(jm, "Mesh " + std::to_string(gmIndex));
    std::vector<glm::vec3> positions, normals;
    std::vector<glm::vec2> uvs;
    std::vector<u32> indices, triSlot;
    bool allNormals = true, allUvs = true;
    std::map<int, u32> slotOf;
    const glm::mat3 r(kGltfToEngine);
    if (const JVal* prims = jm.find("primitives"); prims && prims->type == JVal::Arr)
      for (const JVal& prim : prims->items) {
        PrimitiveData pd;
        std::string log;
        const PrimitiveResult res = readPrimitive(g, prim, pd, log);
        if (res == PrimitiveResult::Error) {
          error = sm.name + ": " + log;
          return -1;
        }
        if (res == PrimitiveResult::Skipped) {
          note("mesh '" + sm.name + "': a non-triangle or empty primitive was skipped");
          continue;
        }
        if (const JVal* attrs = prim.find("attributes")) {
          if (attrs->find("COLOR_0")) note("vertex colours (COLOR_0) ignored (W7)");
          if (attrs->find("JOINTS_0")) note("skinning attributes ignored (W7)");
        }
        if (prim.find("targets")) note("morph targets ignored");
        int matIndex = -1;
        u32 mi = 0;
        if (const JVal* mat = prim.find("material");
            mat && mat->type == JVal::Num && checkedU32(mat->number, mi) && g.materials &&
            mi < g.materials->size())
          matIndex = material(mi);
        const auto [slotIt, fresh] = slotOf.emplace(matIndex, u32(sm.slotMaterials.size()));
        if (fresh) sm.slotMaterials.push_back(matIndex);
        const u32 base = u32(positions.size());
        for (const glm::vec3& p : pd.positions) positions.push_back(r * p);
        if (pd.hasNormals)
          for (const glm::vec3& n : pd.normals) normals.push_back(r * n);
        else
          normals.resize(normals.size() + pd.positions.size(), glm::vec3(0.0f));
        if (pd.hasUvs)
          uvs.insert(uvs.end(), pd.uvs.begin(), pd.uvs.end());
        else
          uvs.resize(uvs.size() + pd.positions.size(), glm::vec2(0.0f));
        allNormals &= pd.hasNormals;
        allUvs &= pd.hasUvs;
        for (u32 i : pd.indices) indices.push_back(base + i);
        triSlot.resize(triSlot.size() + pd.indices.size() / 3, slotIt->second);
      }
    if (positions.empty() || indices.size() < 3) {
      note("mesh '" + sm.name + "' has no triangle geometry");
      meshOf.emplace(gmIndex, -1);
      return -1;
    }
    if (!allNormals) {  // smooth face-accumulated normals for the whole mesh
      std::fill(normals.begin(), normals.end(), glm::vec3(0.0f));
      for (usize t = 0; t + 2 < indices.size(); t += 3) {
        const glm::vec3 fn = glm::cross(positions[indices[t + 1]] - positions[indices[t]],
                                        positions[indices[t + 2]] - positions[indices[t]]);
        for (int k = 0; k < 3; ++k) normals[indices[t + k]] += fn;
      }
    }
    for (glm::vec3& n : normals)
      n = glm::dot(n, n) > 1e-20f ? glm::normalize(n) : glm::vec3(0.0f, 0.0f, 1.0f);
    MeshData& md = sm.mesh;
    md.positions.reserve(positions.size() * 3);
    for (const glm::vec3& v : positions) md.positions.insert(md.positions.end(), {v.x, v.y, v.z});
    md.normals.reserve(normals.size() * 3);
    for (const glm::vec3& v : normals) md.normals.insert(md.normals.end(), {v.x, v.y, v.z});
    if (allUvs)
      for (const glm::vec2& v : uvs) md.uvs.insert(md.uvs.end(), {v.x, v.y});
    else
      note("mesh '" + sm.name + "': a primitive without TEXCOORD_0 - the mesh is untextured");
    md.indices = std::move(indices);
    if (sm.slotMaterials.size() > 1) md.triMaterial = std::move(triSlot);
    const int index = static_cast<int>(out.meshes.size());
    out.meshes.push_back(std::move(sm));
    meshOf.emplace(gmIndex, index);
    return index;
  }

  /// Pre-order walk; returns false on malformed data (`error`).
  bool node(u32 gi, int parent, std::vector<bool>& visited, int depth) {
    if (depth > 256) {
      error = "node hierarchy deeper than 256 levels";
      return false;
    }
    if (visited[gi]) {
      note("node " + std::to_string(gi) + " is reachable twice (glTF requires a tree) - ignored");
      return true;
    }
    if (out.nodes.size() >= 65536) {
      error = "more than 65536 nodes";
      return false;
    }
    visited[gi] = true;
    const JVal& jn = g.nodes->at(gi);
    const int self = static_cast<int>(out.nodes.size());
    GltfSceneNode n;
    n.name = nameOr(jn, "Node " + std::to_string(gi));
    n.parent = parent;
    n.local = kGltfToEngine * nodeMatrix(jn) * glm::transpose(kGltfToEngine);
    out.nodes.push_back(std::move(n));
    if (parent >= 0)
      out.nodes[usize(parent)].children.push_back(self);
    else
      out.roots.push_back(self);
    if (const JVal* m = jn.find("mesh"); m && m->type == JVal::Num) {
      u32 mi = 0;
      if (!checkedU32(m->number, mi) || mi >= g.meshes->size()) {
        error = "node mesh index out of range";
        return false;
      }
      const int sm = mesh(mi);
      if (!error.empty()) return false;
      out.nodes[usize(self)].mesh = sm;
    }
    if (jn.find("skin")) note("skins ignored (W7)");
    if (jn.find("camera")) note("cameras ignored");
    if (const JVal* ext = jn.find("extensions"); ext && ext->find("KHR_lights_punctual"))
      note("KHR_lights_punctual lights ignored");
    if (const JVal* kids = jn.find("children"); kids && kids->type == JVal::Arr)
      for (const JVal& k : kids->items) {
        u32 c = 0;
        if (k.type != JVal::Num || !checkedU32(k.number, c) || c >= g.nodes->size()) {
          error = "node child index out of range";
          return false;
        }
        if (!node(c, self, visited, depth + 1)) return false;
      }
    return true;
  }
};

}  // namespace

bool loadGltfSceneImpl(const std::string& path, GltfScene& out, std::string& log) {
  out = GltfScene{};
  JVal root;
  Gltf g;
  if (!openDocument(path, root, g, log)) return false;
  if (const JVal* req = root.find("extensionsRequired"); req && req->type == JVal::Arr)
    for (const JVal& e : req->items)
      if (e.type == JVal::Str && !requiredExtensionSupported(e.text)) {
        log = "requires unsupported extension " + e.text;
        return false;
      }
  GltfScene scene;
  SceneBuilder b{g, root, scene, {}, {}, {}, {}};
  std::vector<bool> visited(g.nodes->size(), false);
  // The default scene ("scene", else scenes[0]); files without scenes
  // fall back to the true roots (nodes no other node parents).
  std::vector<u32> rootNodes;
  u32 sceneIdx = 0;
  checkedU32(root.numOr("scene", 0.0), sceneIdx);
  if (g.scenes && sceneIdx < g.scenes->size()) {
    if (const JVal* roots = g.scenes->at(sceneIdx).find("nodes"); roots && roots->type == JVal::Arr)
      for (const JVal& n : roots->items) {
        u32 idx = 0;
        if (n.type == JVal::Num && checkedU32(n.number, idx) && idx < g.nodes->size())
          rootNodes.push_back(idx);
      }
    if (g.scenes->size() > 1) b.note("only the default scene is imported");
  } else {
    std::vector<bool> isChild(g.nodes->size(), false);
    for (const JVal& n : g.nodes->items)
      if (const JVal* kids = n.find("children"); kids && kids->type == JVal::Arr)
        for (const JVal& k : kids->items) {
          u32 c = 0;
          if (k.type == JVal::Num && checkedU32(k.number, c) && c < g.nodes->size())
            isChild[c] = true;
        }
    for (u32 n = 0; n < g.nodes->size(); ++n)
      if (!isChild[n]) rootNodes.push_back(n);
  }
  for (const u32 r : rootNodes)
    if (!b.node(r, -1, visited, 0)) {
      log = b.error;
      return false;
    }
  if (root.find("animations")) b.note("animations ignored (W7)");
  if (scene.meshes.empty()) {
    log = "no triangle meshes in the default scene";
    return false;
  }
  scene.externalFiles = g.externalFiles;
  std::sort(scene.externalFiles.begin(), scene.externalFiles.end());
  scene.externalFiles.erase(std::unique(scene.externalFiles.begin(), scene.externalFiles.end()),
                            scene.externalFiles.end());
  out = std::move(scene);
  return true;
}

bool loadGltfScene(const std::string& path, GltfScene& out, std::string& log) {
  try {
    return loadGltfSceneImpl(path, out, log);
  } catch (const std::bad_alloc&) {
    out = GltfScene{};
    log = "out of memory (hostile or oversized file)";
    return false;
  }
}

}  // namespace assets

KENSHIN_NS_END
