#pragma once

// Test fixtures shared by the glTF import and asset database tests: a
// .bin/.gltf writer and the reference model (hierarchy, instancing, two
// materials incl. BLEND, a %20 file-URI image and an embedded one,
// samplers, and content the importer only notes).

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <string>
#include <vector>

#include "kenshin/util/File.h"
#include "test_png.h"

namespace testgltf {

using kenshin::f32;
using kenshin::u32;
using kenshin::u8;
using kenshin::usize;
namespace util = kenshin::util;
namespace fs = std::filesystem;

/// Accumulates a .bin buffer plus its bufferViews/accessors as JSON.
struct Bin {
  std::vector<u8> data;
  std::vector<std::string> views, accessors;

  int view(const void* bytes, usize n) {
    while (data.size() % 4) data.push_back(0);
    const usize at = data.size();
    data.insert(data.end(), static_cast<const u8*>(bytes), static_cast<const u8*>(bytes) + n);
    views.push_back("{\"buffer\":0,\"byteOffset\":" + std::to_string(at) +
                    ",\"byteLength\":" + std::to_string(n) + "}");
    return static_cast<int>(views.size()) - 1;
  }
  int floats(const std::vector<f32>& v, int comps) {
    const int bv = view(v.data(), v.size() * 4);
    static const char* kType[] = {"", "SCALAR", "VEC2", "VEC3", "VEC4"};
    accessors.push_back("{\"bufferView\":" + std::to_string(bv) +
                        ",\"componentType\":5126,\"count\":" + std::to_string(v.size() / comps) +
                        ",\"type\":\"" + kType[comps] + "\"}");
    return static_cast<int>(accessors.size()) - 1;
  }
  int indices(const std::vector<u32>& v) {
    const int bv = view(v.data(), v.size() * 4);
    accessors.push_back("{\"bufferView\":" + std::to_string(bv) +
                        ",\"componentType\":5125,\"count\":" + std::to_string(v.size()) +
                        ",\"type\":\"SCALAR\"}");
    return static_cast<int>(accessors.size()) - 1;
  }
  static std::string join(const std::vector<std::string>& v) {
    std::string s;
    for (usize i = 0; i < v.size(); ++i) s += (i ? "," : "") + v[i];
    return "[" + s + "]";
  }
};

struct TempDir {
  fs::path path;
  explicit TempDir(const std::string& name) : path(fs::temp_directory_path() / name) {
    fs::remove_all(path);
    fs::create_directories(path);
  }
  ~TempDir() { fs::remove_all(path); }
  void write(const std::string& file, const void* data, usize n) const {
    REQUIRE(util::writeFileAtomic(path / util::pathFromUtf8(file),
                                  std::string_view(static_cast<const char*>(data), n)));
  }
  void write(const std::string& file, const std::string& text) const {
    write(file, text.data(), text.size());
  }
  std::string str(const std::string& file) const {
    return util::pathToUtf8(path / util::pathFromUtf8(file));
  }
};

/// The reference scene: a TRS root with two children instancing one
/// two-primitive mesh (two materials, one BLEND), a matrix-transformed
/// node, a second root with a normal-less, UV-less mesh, an orphan node
/// outside the scene, two images (file URI with an escaped space and an
/// embedded bufferView), samplers, and content the importer only notes.
struct ReferenceModel {
  TempDir dir{"kenshin_gltf_import_test"};
  std::vector<u8> pngA = testpng::makePng(2, 2, 8, 6, std::vector<u32>(16, 200));
  std::vector<u8> pngB = testpng::makePng(1, 1, 8, 2, {128, 128, 255});
  std::string path;

  ReferenceModel() {
    Bin b;
    const int qPos = b.floats({0, 0, 0, 1, 0, 0, 0, 1, 0}, 3);
    const int qNrm = b.floats({0, 0, 1, 0, 0, 1, 0, 0, 1}, 3);
    const int qUv = b.floats({0, 0, 1, 0, 0, 1}, 2);
    const int qPos2 = b.floats({0, 0, 1, 1, 0, 1, 0, 1, 1}, 3);
    const int idx = b.indices({0, 1, 2});
    const int tPos = b.floats({0, 0, 0, 0, 0, 1, 1, 0, 0}, 3);
    const int tCol = b.floats({1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1}, 4);
    const int imgView = b.view(pngB.data(), pngB.size());
    dir.write("model.bin", b.data.data(), b.data.size());
    dir.write("tex one.png", pngA.data(), pngA.size());
    const std::string json = R"({
      "asset": {"version": "2.0"},
      "scene": 0,
      "scenes": [{"nodes": [0, 4]}],
      "nodes": [
        {"name": "Root", "translation": [1, 2, 3], "rotation": [0, 0.70710678, 0, 0.70710678],
         "scale": [2, 2, 2], "children": [1, 2]},
        {"name": "Left", "mesh": 0, "translation": [0, 1, 0]},
        {"name": "Right", "mesh": 0, "matrix": [1,0,0,0, 0,1,0,0, 0,0,1,0, 5,0,0,1]},
        {"name": "Orphan", "mesh": 1},
        {"name": "\u4e2d\u6587 \ud83d\ude00", "mesh": 1, "scale": [1, 3, 1]}
      ],
      "meshes": [
        {"name": "Quad", "primitives": [
          {"attributes": {"POSITION": )" + std::to_string(qPos) + R"(, "NORMAL": )" +
                             std::to_string(qNrm) + R"(, "TEXCOORD_0": )" + std::to_string(qUv) +
                             R"(}, "indices": )" + std::to_string(idx) + R"(, "material": 0},
          {"attributes": {"POSITION": )" + std::to_string(qPos2) + R"(, "NORMAL": )" +
                             std::to_string(qNrm) + R"(, "TEXCOORD_0": )" + std::to_string(qUv) +
                             R"(}, "indices": )" + std::to_string(idx) + R"(, "material": 1}]},
        {"name": "Tri", "primitives": [
          {"attributes": {"POSITION": )" + std::to_string(tPos) + R"(, "COLOR_0": )" +
                             std::to_string(tCol) + R"(}}]}
      ],
      "materials": [
        {"name": "Red",
         "pbrMetallicRoughness": {"baseColorFactor": [1, 0, 0, 1], "metallicFactor": 0.2,
           "roughnessFactor": 0.7, "baseColorTexture": {"index": 0},
           "metallicRoughnessTexture": {"index": 1}},
         "normalTexture": {"index": 2, "texCoord": 1, "scale": 0.5,
                           "extensions": {"KHR_texture_transform": {"scale": [2, 2]}}},
         "occlusionTexture": {"index": 1, "strength": 0.3}},
        {"name": "Glass", "alphaMode": "BLEND", "doubleSided": true,
         "pbrMetallicRoughness": {"baseColorFactor": [0.5, 0.5, 1, 0.25]},
         "emissiveFactor": [1, 0.5, 0],
         "extensions": {"KHR_materials_emissive_strength": {"emissiveStrength": 2}}}
      ],
      "textures": [
        {"source": 0, "sampler": 0},
        {"source": 0},
        {"source": 1, "sampler": 1}
      ],
      "samplers": [{"wrapS": 33071, "wrapT": 33648, "magFilter": 9728}, {}],
      "images": [
        {"uri": "tex%20one.png"},
        {"bufferView": )" + std::to_string(imgView) + R"(, "mimeType": "image/png", "name": "nrm"}
      ],
      "animations": [{"channels": [], "samplers": []}],
      "buffers": [{"uri": "model.bin", "byteLength": )" + std::to_string(b.data.size()) + R"(}],
      "bufferViews": )" + Bin::join(b.views) + R"(,
      "accessors": )" + Bin::join(b.accessors) + R"(
    })";
    dir.write("model.gltf", json);
    path = dir.str("model.gltf");
  }
};

}  // namespace testgltf
