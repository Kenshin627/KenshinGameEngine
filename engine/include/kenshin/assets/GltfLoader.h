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

/// One USED glTF material (deduplicated, insertion order): PBR
/// params plus the base-color image (index into the `images` out of
/// loadGltf, -1 = untextured factor material).
struct GltfMaterial {
  glm::vec3 kd{0.8f, 0.8f, 0.8f};
  f32 roughness = 0.5f;
  f32 metallic = 0.0f;
  int image = -1;     // base-color image (index into `images` out)
  int mrImage = -1;   // metallicRoughness image (glTF packs roughness in
                      // G, metallic in B; LINEAR data - never sRGB-decoded)
  f32 alpha = 1.0f;   // baseColorFactor[3] when alphaMode == BLEND (the
                      // builder sinks BLEND geometry slightly behind the
                      // opaque surface - skintight plates otherwise lose
                      // their intended depth order to LOD simplification
                      // error and paint opaque blocks over the shell)
  int nrmImage = -1;  // normalTexture image (tangent space, LINEAR)
  bool doubleSided = false;  // glTF material.doubleSided (the builder
                             // appends reversed-winding copies: every
                             // raster path culls back faces)
};

/// Load a .glb or .gltf. `albedo` (optional out) receives the first
/// material's base-color texture when present; it stays empty
/// otherwise. `materials`/`images` (optional outs) receive the full
/// multi-material set - every used material in first-use order with
/// `mesh.triMaterial` tagging each triangle's material slot - so the
/// builder can emit a per-cluster material table. All coordinates are
/// rotated into the engine's Z-up right-handed frame (glTF is
/// right-handed Y-up by spec: R_x(+90), (x,y,z)_glTF ->
/// (x,-z,y)_engine). @return true on success (`log` carries the
/// reason on false).
bool loadGltf(const std::string& path, MeshData& mesh, GltfImage* albedo,
              std::string& log, std::vector<GltfMaterial>* materials = nullptr,
              std::vector<GltfImage>* images = nullptr);

}  // namespace assets

KENSHIN_NS_END
