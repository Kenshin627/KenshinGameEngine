#include "kenshin/assets/Material.h"

#include <cmath>
#include <format>

KENSHIN_NS_BEGIN
namespace assets {

using serialization::Json;

namespace {

constexpr const char* kSlotNames[] = {"base_color", "metallic_roughness", "normal", "occlusion",
                                      "emissive"};

MaterialTextureSlot* slotOf(MaterialDesc& m, int i) {
  MaterialTextureSlot* slots[] = {&m.baseColorTexture, &m.metallicRoughnessTexture,
                                  &m.normalTexture, &m.occlusionTexture, &m.emissiveTexture};
  return slots[i];
}
const MaterialTextureSlot* slotOf(const MaterialDesc& m, int i) {
  return slotOf(const_cast<MaterialDesc&>(m), i);
}

const char* wrapName(backend::WrapMode w) {
  switch (w) {
    case backend::WrapMode::Repeat: return "repeat";
    case backend::WrapMode::Clamp: return "clamp";
    case backend::WrapMode::Mirror: return "mirror";
  }
  return "repeat";
}

Json floats(std::initializer_list<f32> v) {
  Json a = Json::array();
  for (const f32 x : v) a.push_back(serialization::floatToJson(x));
  return a;
}

/// A finite number in [lo, hi].
bool number(const Json& j, f32 lo, f32 hi, f32& out) {
  if (!j.is_number()) return false;
  const double d = j.get<double>();
  if (!std::isfinite(d) || d < lo || d > hi) return false;
  out = f32(d);
  return true;
}

bool vecOf(const Json& j, usize n, f32 lo, f32 hi, f32* out) {
  if (!j.is_array() || j.size() != n) return false;
  for (usize i = 0; i < n; ++i)
    if (!number(j[i], lo, hi, out[i])) return false;
  return true;
}

}  // namespace

const serialization::SchemaMigrator& materialMigrator() {
  static const serialization::SchemaMigrator m(std::string(kMaterialFormat), kMaterialVersion);
  return m;
}

Json materialToJson(const MaterialDesc& m) {
  Json doc = Json::object();
  doc["format"] = std::string(kMaterialFormat);
  doc["version"] = kMaterialVersion;
  doc["base_color"] = floats({m.baseColor.r, m.baseColor.g, m.baseColor.b, m.baseColor.a});
  doc["metallic"] = serialization::floatToJson(m.metallic);
  doc["roughness"] = serialization::floatToJson(m.roughness);
  doc["emissive"] = floats({m.emissive.r, m.emissive.g, m.emissive.b});
  static constexpr const char* kAlpha[] = {"opaque", "mask", "blend"};
  doc["alpha_mode"] = kAlpha[static_cast<int>(m.alphaMode)];
  doc["alpha_cutoff"] = serialization::floatToJson(m.alphaCutoff);
  doc["double_sided"] = m.doubleSided;
  doc["normal_scale"] = serialization::floatToJson(m.normalScale);
  doc["occlusion_strength"] = serialization::floatToJson(m.occlusionStrength);
  Json textures = Json::object();
  for (int i = 0; i < 5; ++i) {
    const MaterialTextureSlot& s = *slotOf(m, i);
    if (!s.valid()) continue;
    Json t = Json::object();
    t["texture"] = s.texture.toString();
    t["wrap_u"] = wrapName(s.sampler.wrapU);
    t["wrap_v"] = wrapName(s.sampler.wrapV);
    t["filter"] = s.sampler.filter == backend::FilterMode::Nearest ? "nearest" : "linear";
    textures[kSlotNames[i]] = std::move(t);
  }
  doc["textures"] = std::move(textures);
  return doc;
}

std::string saveMaterial(const MaterialDesc& m) { return serialization::writeJson(materialToJson(m)); }

bool materialFromJson(Json doc, MaterialDesc& out, std::string* error) {
  auto fail = [&](const std::string& why) {
    if (error) *error = why;
    return false;
  };
  std::string why;
  if (!materialMigrator().migrate(doc, why)) return fail(why);
  MaterialDesc m;
  for (auto it = doc.begin(); it != doc.end(); ++it) {
    const std::string& k = it.key();
    const Json& v = *it;
    if (k == "format" || k == "version") continue;
    if (k == "base_color") {
      if (!vecOf(v, 4, 0.0f, 1.0f, &m.baseColor.r)) return fail("base_color must be 4 numbers in [0, 1]");
    } else if (k == "metallic") {
      if (!number(v, 0.0f, 1.0f, m.metallic)) return fail("metallic must be a number in [0, 1]");
    } else if (k == "roughness") {
      if (!number(v, 0.0f, 1.0f, m.roughness)) return fail("roughness must be a number in [0, 1]");
    } else if (k == "emissive") {
      if (!vecOf(v, 3, 0.0f, 1e5f, &m.emissive.r)) return fail("emissive must be 3 numbers >= 0");
    } else if (k == "alpha_mode") {
      const std::string a = v.is_string() ? v.get<std::string>() : "";
      if (a == "opaque") m.alphaMode = MaterialDesc::AlphaMode::Opaque;
      else if (a == "mask") m.alphaMode = MaterialDesc::AlphaMode::Mask;
      else if (a == "blend") m.alphaMode = MaterialDesc::AlphaMode::Blend;
      else return fail("alpha_mode must be one of [opaque, mask, blend]");
    } else if (k == "alpha_cutoff") {
      if (!number(v, 0.0f, 1.0f, m.alphaCutoff)) return fail("alpha_cutoff must be a number in [0, 1]");
    } else if (k == "double_sided") {
      if (!v.is_boolean()) return fail("double_sided must be a boolean");
      m.doubleSided = v.get<bool>();
    } else if (k == "normal_scale") {
      if (!number(v, -100.0f, 100.0f, m.normalScale)) return fail("normal_scale must be a number");
    } else if (k == "occlusion_strength") {
      if (!number(v, 0.0f, 1.0f, m.occlusionStrength))
        return fail("occlusion_strength must be a number in [0, 1]");
    } else if (k == "textures") {
      if (!v.is_object()) return fail("textures must be an object");
      for (auto t = v.begin(); t != v.end(); ++t) {
        int idx = -1;
        for (int i = 0; i < 5; ++i)
          if (t.key() == kSlotNames[i]) idx = i;
        if (idx < 0)
          return fail(std::format("unknown texture slot \"{}\" (slots: base_color, "
                                  "metallic_roughness, normal, occlusion, emissive)",
                                  t.key()));
        const std::string at = "textures." + t.key();
        if (!t->is_object()) return fail(at + " must be an object");
        MaterialTextureSlot& s = *slotOf(m, idx);
        for (auto f = t->begin(); f != t->end(); ++f) {
          const std::string text = f->is_string() ? f->get<std::string>() : "";
          if (f.key() == "texture") {
            const auto g = core::Guid::parse(text);
            if (!g || g->isNull()) return fail(at + ".texture must be a non-null guid string");
            s.texture = *g;
          } else if (f.key() == "wrap_u" || f.key() == "wrap_v") {
            backend::WrapMode w;
            if (text == "repeat") w = backend::WrapMode::Repeat;
            else if (text == "clamp") w = backend::WrapMode::Clamp;
            else if (text == "mirror") w = backend::WrapMode::Mirror;
            else return fail(at + "." + f.key() + " must be one of [repeat, clamp, mirror]");
            (f.key() == "wrap_u" ? s.sampler.wrapU : s.sampler.wrapV) = w;
          } else if (f.key() == "filter") {
            if (text == "linear") s.sampler.filter = backend::FilterMode::Linear;
            else if (text == "nearest") s.sampler.filter = backend::FilterMode::Nearest;
            else return fail(at + ".filter must be one of [linear, nearest]");
          } else {
            return fail(std::format("{}: unknown key \"{}\"", at, f.key()));
          }
        }
        if (!s.valid()) return fail(at + " needs a \"texture\" guid");
      }
    } else {
      return fail(std::format("unknown key \"{}\"", k));
    }
  }
  out = m;
  return true;
}

bool loadMaterial(std::string_view text, MaterialDesc& out, std::string* error) {
  Json doc;
  std::string why;
  if (!serialization::parseJson(text, doc, &why)) {
    if (error) *error = why;
    return false;
  }
  return materialFromJson(std::move(doc), out, error);
}

}  // namespace assets
KENSHIN_NS_END
