#pragma once

// ── Minimal glTF 2.0 / GLB loader (v1) ───────────────────────────
// Hand-written JSON DOM + GLB chunk container + accessor decoding,
// the ObjLoader discipline (return false + log, never throw). Produces
// ONE merged MeshData (every mesh in the scene, node transforms
// applied, normals computed when absent) plus the decoded albedo
// image - the builder re-encodes it as a BMP next to its output so
// the whole downstream material/texture path stays unchanged.
//
// v1 scope: mode-4 (triangle) primitives; FLOAT positions/UVs,
// USHORT/UINT/UBYTE indices; buffers from the GLB BIN chunk or
// same-directory files (data-URI buffers rejected); images from
// embedded bufferViews, same-directory files or base64 data URIs
// (png/jpeg/bmp/tga - whatever stb sniffs); pbrMetallicRoughness
// (first referenced material); KHR extensions ignored with a log.

#include <string>
#include <vector>

#include "kenshin/Common.h"
#include "kenshin/assets/MeshData.h"

KENSHIN_NS_BEGIN

namespace assets {

/// A decoded glTF image (RGBA8, top-down rows - writeBmp24's input
/// contract so the builder can drop it straight to disk).
struct GltfImage {
  std::string name;  // file-stem suggestion for the re-encoded BMP
  int width = 0;
  int height = 0;
  std::vector<u8> rgba;
};

/// Load a .glb or .gltf. `albedo` (optional out) receives the first
/// material's base-color texture when present; it stays empty
/// otherwise. @return true on success (`log` carries the reason on
/// false).
bool loadGltf(const std::string& path, MeshData& mesh, GltfImage* albedo,
              std::string& log);

}  // namespace assets

KENSHIN_NS_END
