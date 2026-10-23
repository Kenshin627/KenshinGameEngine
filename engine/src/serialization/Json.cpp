#include "kenshin/serialization/Json.h"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <format>
#include <unordered_set>

KENSHIN_NS_BEGIN
namespace serialization {

Json floatToJson(f32 v) {
  // to_chars(float) gives the shortest text that round-trips as float;
  // reading it as a double keeps that text when nlohmann prints the
  // double back (its dtoa is shortest-for-double).
  char buf[64];
  const auto r = std::to_chars(buf, buf + sizeof(buf), v);
  *r.ptr = '\0';
  const double d = std::strtod(buf, nullptr);
  // The reader computes float(d); should that ever differ from v (a
  // double-rounding corner), store v's exact double instead - longer
  // text, still exact.
  if (static_cast<f32>(d) != v && !std::isnan(v)) return Json(static_cast<double>(v));
  return Json(d);
}

namespace {
bool isScalar(const Json& j) { return !j.is_object() && !j.is_array(); }

std::string scalarText(const Json& j) {
  // Invalid UTF-8 cannot come from the engine's own strings; replace
  // rather than throw if it ever does.
  return j.dump(-1, ' ', false, Json::error_handler_t::replace);
}

void write(const Json& j, int indent, std::string& out) {
  const std::string pad(static_cast<usize>(indent) * 2, ' ');
  const std::string inner(static_cast<usize>(indent + 1) * 2, ' ');
  if (j.is_object()) {
    if (j.empty()) {
      out += "{}";
      return;
    }
    out += "{\n";
    usize i = 0;
    for (auto it = j.begin(); it != j.end(); ++it, ++i) {
      out += inner;
      out += scalarText(Json(it.key()));
      out += ": ";
      write(it.value(), indent + 1, out);
      out += i + 1 < j.size() ? ",\n" : "\n";
    }
    out += pad + "}";
  } else if (j.is_array()) {
    if (j.empty()) {
      out += "[]";
      return;
    }
    // Vectors, colors, quaternions: one line keeps files readable.
    bool inlineArray = j.size() <= 4;
    for (const Json& e : j) inlineArray = inlineArray && isScalar(e);
    if (inlineArray) {
      out += "[";
      for (usize i = 0; i < j.size(); ++i) {
        if (i) out += ", ";
        out += scalarText(j[i]);
      }
      out += "]";
      return;
    }
    out += "[\n";
    for (usize i = 0; i < j.size(); ++i) {
      out += inner;
      write(j[i], indent + 1, out);
      out += i + 1 < j.size() ? ",\n" : "\n";
    }
    out += pad + "]";
  } else {
    out += scalarText(j);
  }
}

/// Validation-only SAX pass (no DOM): nesting depth and duplicate
/// keys, which the DOM parser would silently collapse to one value.
/// Scopes are pooled so steady-state parsing does not allocate.
class StrictSax final : public nlohmann::json_sax<Json> {
 public:
  static constexpr usize kMaxDepth = 128;
  std::string error;

  bool null() override { return true; }
  bool boolean(bool) override { return true; }
  bool number_integer(number_integer_t) override { return true; }
  bool number_unsigned(number_unsigned_t) override { return true; }
  bool number_float(number_float_t, const string_t&) override { return true; }
  bool string(string_t&) override { return true; }
  bool binary(binary_t&) override { return true; }
  bool start_object(std::size_t) override { return open(); }
  bool end_object() override { return close(); }
  bool start_array(std::size_t) override { return open(); }
  bool end_array() override { return close(); }
  bool key(string_t& k) override {
    Scope& s = m_scopes[m_depth - 1];
    // Small objects (all of ours): a linear scan beats hashing.
    // Large ones switch to a set so hostile input stays linear.
    if (s.keys.size() < 32) {
      for (const std::string& seen : s.keys)
        if (seen == k) return duplicate(k);
      s.keys.push_back(k);
      if (s.keys.size() == 32) s.big.insert(s.keys.begin(), s.keys.end());
    } else if (!s.big.insert(k).second) {
      return duplicate(k);
    }
    return true;
  }
  bool parse_error(std::size_t, const std::string&, const nlohmann::detail::exception& e) override {
    error = e.what();
    return false;
  }

 private:
  struct Scope {
    std::vector<std::string> keys;
    std::unordered_set<std::string> big;
  };
  bool open() {  // arrays get an (unused, empty) scope too
    if (m_depth >= kMaxDepth) {
      error = std::format("nesting deeper than {} levels", kMaxDepth);
      return false;
    }
    if (m_scopes.size() <= m_depth) m_scopes.emplace_back();
    Scope& s = m_scopes[m_depth++];
    s.keys.clear();
    s.big.clear();
    return true;
  }
  bool close() {
    --m_depth;
    return true;
  }
  bool duplicate(const std::string& k) {
    error = std::format("duplicate key \"{}\"", k);
    return false;
  }
  std::vector<Scope> m_scopes;
  usize m_depth = 0;
};
}  // namespace

std::string writeJson(const Json& value) {
  std::string out;
  write(value, 0, out);
  out += '\n';
  return out;
}

bool parseJson(std::string_view text, Json& out, std::string* error) {
  // Two passes: the SAX check (strictness, ~cheap) then nlohmann's
  // plain DOM parser, which is far faster than its callback parser.
  StrictSax sax;
  try {
    if (!Json::sax_parse(text.begin(), text.end(), &sax, nlohmann::detail::input_format_t::json,
                         /*strict=*/true, /*ignore_comments=*/false)) {
      if (error) *error = sax.error;
      out = Json();
      return false;
    }
    out = Json::parse(text.begin(), text.end(), nullptr, /*allow_exceptions=*/true,
                      /*ignore_comments=*/false);
    return true;
  } catch (const Json::exception& e) {  // not expected after the SAX pass
    if (error) *error = e.what();
  }
  out = Json();
  return false;
}

void SchemaMigrator::add(i32 fromVersion, std::string description, Step step) {
  m_steps.push_back({fromVersion, std::move(description), std::move(step)});
}

bool SchemaMigrator::migrate(Json& doc, std::string& error) const {
  if (!doc.is_object()) {
    error = "document is not a JSON object";
    return false;
  }
  const auto format = doc.find("format");
  if (format == doc.end() || !format->is_string()) {
    error = "missing \"format\"";
    return false;
  }
  if (format->get<std::string>() != m_format) {
    error = std::format("format is \"{}\", expected \"{}\"", format->get<std::string>(),
                        m_format);
    return false;
  }
  const auto version = doc.find("version");
  if (version == doc.end() || !version->is_number_integer()) {
    error = "missing integer \"version\"";
    return false;
  }
  if (version->is_number_unsigned()
          ? version->get<u64>() > static_cast<u64>(m_current)
          : version->get<i64>() > m_current) {
    error = std::format("version {} is newer than this engine supports ({})", version->dump(),
                        m_current);
    return false;
  }
  if (version->get<i64>() < 0) {
    error = std::format("version {} is not valid", version->dump());
    return false;
  }
  i32 v = version->get<i32>();  // 0..current: fits
  while (v < m_current) {
    const Migration* step = nullptr;
    for (const Migration& m : m_steps)
      if (m.from == v) step = &m;
    if (!step) {
      error = std::format("no migration from version {} (current {})", v, m_current);
      return false;
    }
    std::string why;
    bool ok = false;
    try {
      ok = step->step(doc, why);
    } catch (const std::exception& e) {  // e.g. a Json type_error in a step
      why = e.what();
    }
    if (!ok) {
      error = std::format("migration {} -> {} ({}) failed: {}", v, v + 1,
                          step->description, why);
      return false;
    }
    ++v;
    doc["version"] = v;
  }
  return true;
}

}  // namespace serialization
KENSHIN_NS_END
