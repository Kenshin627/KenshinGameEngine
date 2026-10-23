#include "kenshin/importer/KgeoMaterials.h"

#include <algorithm>

#include "kenshin/core/Log.h"
#include "kenshin/util/BmpWriter.h"
#include "kenshin/util/File.h"

KENSHIN_NS_BEGIN
namespace importer {

namespace fs = std::filesystem;
using assets::GltfImage;
using assets::GltfMaterial;
using geometry::ClusterDag;

namespace {

/// Box resample (8-bit RGB, opaque): composes a separate AO image into
/// its material's metallic-roughness image (ORM packing: AO rides R)
/// when the two differ in size.
GltfImage resampleTo(const GltfImage& im, int w, int h) {
  GltfImage out;
  out.name = im.name;
  out.width = w;
  out.height = h;
  out.rgba.assign(usize(w) * h * 4, 0);
  for (int y = 0; y < h; ++y) {
    const int sy0 = int(u64(y) * im.height / h);
    const int sy1 = std::max(int(u64(y + 1) * im.height / h), sy0 + 1);
    for (int x = 0; x < w; ++x) {
      const int sx0 = int(u64(x) * im.width / w);
      const int sx1 = std::max(int(u64(x + 1) * im.width / w), sx0 + 1);
      u32 r = 0, g = 0, b = 0, n = 0;
      for (int sy = sy0; sy < sy1 && sy < im.height; ++sy)
        for (int sx = sx0; sx < sx1 && sx < im.width; ++sx) {
          const u8* p = im.rgba.data() + (usize(sy) * im.width + sx) * 4;
          r += p[0];
          g += p[1];
          b += p[2];
          ++n;
        }
      u8* o = out.rgba.data() + (usize(y) * w + x) * 4;
      o[0] = u8(r / n);
      o[1] = u8(g / n);
      o[2] = u8(b / n);
      o[3] = 255;
    }
  }
  return out;
}

}  // namespace

bool writeSingleAlbedo(const GltfImage& albedo, const std::string& fallbackStem,
                       const fs::path& outDir, assets::MeshData& mesh, std::string& log) {
  const std::string stem = albedo.name.empty() ? fallbackStem : albedo.name;
  const fs::path bmp = outDir / (stem + "_albedo.bmp");
  if (!util::writeBmp24(bmp.string(), albedo.rgba, u32(albedo.width), u32(albedo.height))) {
    log = "albedo re-encode failed: " + bmp.string();
    return false;
  }
  mesh.material.diffuseTexture = bmp.string();
  KENSHIN_INFO("[import] gltf albedo {}x{} -> {}", albedo.width, albedo.height, bmp.string());
  return true;
}

bool buildKgeoMaterials(const assets::MeshData& mesh, const MaterialOverrides& ov,
                        std::vector<GltfMaterial>& gltfMaterials,
                        std::vector<GltfImage>& gltfImages, const fs::path& outDir,
                        ClusterDag& dag, std::string& albedoSource, std::string& log) {
  // The single-material table (v1). OBJ/MTL provides the defaults; the
  // overrides win. The albedo reference is stored as a bare FILE NAME -
  // the runtime resolves it against the .kgeo's directory.
  albedoSource.clear();
  ClusterDag::DagMaterial mat;
  if (mesh.material.kd[0] >= 0.0f)
    for (int c = 0; c < 3; ++c) mat.kd[c] = mesh.material.kd[c];
  if (ov.kd[0] >= 0.0f)
    for (int c = 0; c < 3; ++c) mat.kd[c] = ov.kd[c];
  if (ov.roughness >= 0.0f)
    mat.roughness = ov.roughness;
  else if (mesh.material.roughness != 0.5f)
    mat.roughness = mesh.material.roughness;
  if (ov.metallic >= 0.0f)
    mat.metallic = ov.metallic;
  else if (mesh.material.metallic != 0.0f)
    mat.metallic = mesh.material.metallic;
  if (!ov.albedo.empty())
    albedoSource = ov.albedo;
  else if (!mesh.material.diffuseTexture.empty())
    albedoSource = mesh.material.diffuseTexture;
  if (!albedoSource.empty()) mat.texPath = fs::path(albedoSource).filename().string();

  if (gltfMaterials.size() > 1) {
    // Multi-material table: one DagMaterial per used glTF material;
    // per-cluster ids came from the DAG's vertex-majority vote (do NOT
    // overwrite them). Overrides apply to every entry. Each textured
    // slot is re-encoded as its own BMP next to the output.
    std::vector<std::string> writtenFor(gltfImages.size());
    dag.materials.clear();
    bool failed = false;
    for (const auto& gm : gltfMaterials) {
      ClusterDag::DagMaterial dm = mat;
      for (int c = 0; c < 3; ++c) dm.kd[c] = gm.kd[c];
      dm.roughness = gm.roughness;
      dm.metallic = gm.metallic;
      if (ov.kd[0] >= 0.0f)
        for (int c = 0; c < 3; ++c) dm.kd[c] = ov.kd[c];
      if (ov.roughness >= 0.0f) dm.roughness = ov.roughness;
      if (ov.metallic >= 0.0f) dm.metallic = ov.metallic;
      // Each image keeps its own size: bindless textures (1.3a) lifted
      // the shared-array rule that used to box-resample them all to the
      // first image's size.
      auto imageFile = [&](int imageIdx, const char* suffix) -> std::string {
        if (imageIdx < 0 || imageIdx >= int(gltfImages.size())) return {};
        const GltfImage& img = gltfImages[usize(imageIdx)];
        if (img.width <= 0) return {};
        std::string& file = writtenFor[usize(imageIdx)];
        if (file.empty()) {
          const std::string stem =
              img.name.empty() ? "mat" + std::to_string(dag.materials.size()) : img.name;
          const fs::path bmp = outDir / (stem + suffix + ".bmp");
          if (!util::writeBmp24(bmp.string(), img.rgba, u32(img.width), u32(img.height))) {
            KENSHIN_ERROR("[import] texture re-encode failed: {}", bmp.string());
            failed = true;
            return {};
          }
          file = bmp.filename().string();
          KENSHIN_INFO("[import] gltf material {} tex {}x{} -> {}", dag.materials.size(),
                       img.width, img.height, file);
        }
        return file;
      };
      const std::string albedoFile = imageFile(gm.image, "_albedo");
      if (!albedoFile.empty()) dm.texPath = albedoFile;
      // AO rides the MR layer's R channel (glTF ORM packing). A
      // separate occlusion image is composed into a copy of the MR
      // image (or a G = B = 1 stand-in: factors x 1 = the factor-only
      // result).
      int mrIdx = gm.mrImage;
      if (gm.aoImage >= 0 && gm.aoImage != gm.mrImage && gm.aoImage < int(gltfImages.size())) {
        const auto& ao = gltfImages[usize(gm.aoImage)];
        GltfImage orm;
        if (mrIdx >= 0 && mrIdx < int(gltfImages.size())) {
          orm = gltfImages[usize(mrIdx)];
        } else {
          orm.width = ao.width;
          orm.height = ao.height;
          orm.rgba.assign(usize(ao.width) * ao.height * 4, 255);
        }
        const GltfImage aoFit = (ao.width == orm.width && ao.height == orm.height)
                                    ? ao
                                    : resampleTo(ao, orm.width, orm.height);
        for (usize i = 0; i + 3 < orm.rgba.size(); i += 4) orm.rgba[i] = aoFit.rgba[i];
        orm.name = "mat" + std::to_string(dag.materials.size()) + "_ao";
        gltfImages.push_back(std::move(orm));
        writtenFor.emplace_back();
        mrIdx = int(gltfImages.size()) - 1;
      }
      const std::string mrFile = imageFile(mrIdx, "_mr");
      if (!mrFile.empty()) dm.mrPath = mrFile;
      if (gm.aoImage >= 0 && !mrFile.empty()) dm.occlusion = gm.aoStrength;
      const std::string nrmFile = imageFile(gm.nrmImage, "_nrm");
      if (!nrmFile.empty()) dm.nrmPath = nrmFile;
      for (int c = 0; c < 3; ++c) dm.emissive[c] = gm.emissive[c];
      dag.materials.push_back(dm);
    }
    // A failed side file was already the old builder's silent "keep
    // going without that texture"; surface it as a warning only.
    if (failed) log = "some material textures could not be written (see errors)";
  } else {
    dag.materials = {mat};
  }
  if (dag.clusterMaterial.size() != dag.clusters.size())
    dag.clusterMaterial.assign(dag.clusters.size(), 0);
  return true;
}

bool copyAlbedoBesideOutput(const std::string& albedoSource, const fs::path& output,
                            std::string& log) {
  if (albedoSource.empty()) return true;
  std::error_code ec;
  const fs::path src = fs::path(albedoSource);
  if (!fs::exists(src, ec)) {
    log = "albedo not found: " + albedoSource + " (relative to CWD)";
    return false;
  }
  const fs::path dst = output.parent_path() / src.filename();
  // The glTF path may have written the BMP straight into the output dir
  // already - copying onto itself fails on Windows.
  if (fs::weakly_canonical(src, ec) != fs::weakly_canonical(dst, ec)) {
    if (!fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec)) {
      log = "albedo copy failed: " + src.string() + " -> " + dst.string() + " (" + ec.message() +
            ")";
      return false;
    }
    KENSHIN_INFO("[import] albedo: {} -> {}", src.string(), dst.string());
  }
  return true;
}

}  // namespace importer
KENSHIN_NS_END
