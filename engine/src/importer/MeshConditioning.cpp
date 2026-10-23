#include "kenshin/importer/MeshConditioning.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <glm/glm.hpp>
#include <map>
#include <tuple>

#include "kenshin/core/Log.h"
#include "kenshin/util/File.h"

KENSHIN_NS_BEGIN
namespace importer {

using assets::GltfMaterial;
using assets::MeshData;

namespace {

/// Squared distance from p to triangle abc (Ericson, Real-Time
/// Collision Detection 5.1.5 - closest point by Voronoi region).
f32 pointTriDist2(const glm::vec3& p, const glm::vec3& a, const glm::vec3& b,
                  const glm::vec3& c) {
  const glm::vec3 ab = b - a, ac = c - a, ap = p - a;
  const f32 d1 = glm::dot(ab, ap), d2 = glm::dot(ac, ap);
  if (d1 <= 0.0f && d2 <= 0.0f) return glm::dot(ap, ap);
  const glm::vec3 bp = p - b;
  const f32 d3 = glm::dot(ab, bp), d4 = glm::dot(ac, bp);
  if (d3 >= 0.0f && d4 <= d3) return glm::dot(bp, bp);
  const f32 vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
    const glm::vec3 q = a + ab * (d1 / (d1 - d3));
    return glm::dot(p - q, p - q);
  }
  const glm::vec3 cp = p - c;
  const f32 d5 = glm::dot(ab, cp), d6 = glm::dot(ac, cp);
  if (d6 >= 0.0f && d5 <= d6) return glm::dot(cp, cp);
  const f32 vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
    const glm::vec3 q = a + ac * (d2 / (d2 - d6));
    return glm::dot(p - q, p - q);
  }
  const f32 va = d3 * d6 - d5 * d4;
  if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
    const glm::vec3 q = b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    return glm::dot(p - q, p - q);
  }
  const f32 denom = 1.0f / (va + vb + vc);
  const glm::vec3 q = a + ab * (vb * denom) + ac * (vc * denom);
  return glm::dot(p - q, p - q);
}

bool isBlend(std::span<const GltfMaterial> materials, u32 m) {
  return m < materials.size() && materials[m].alpha < 0.999f;
}

}  // namespace

void recenter(MeshData& mesh, f32 fit) {
  if (mesh.positions.empty()) return;
  f32 mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
  for (usize v = 0; v < mesh.vertexCount(); ++v)
    for (int k = 0; k < 3; ++k) {
      const f32 p = mesh.positions[v * 3 + k];
      mn[k] = std::min(mn[k], p);
      mx[k] = std::max(mx[k], p);
    }
  const f32 cx = (mn[0] + mx[0]) * 0.5f, cy = (mn[1] + mx[1]) * 0.5f;
  const f32 size = std::max({mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]});
  const f32 scale = fit > 0.0f && size > 1e-9f ? fit / size : 1.0f;
  if (std::abs(cx) > 1e-6f || std::abs(cy) > 1e-6f || std::abs(mn[2]) > 1e-6f ||
      scale != 1.0f) {
    for (usize v = 0; v < mesh.vertexCount(); ++v) {
      mesh.positions[v * 3 + 0] = (mesh.positions[v * 3 + 0] - cx) * scale;
      mesh.positions[v * 3 + 1] = (mesh.positions[v * 3 + 1] - cy) * scale;
      mesh.positions[v * 3 + 2] = (mesh.positions[v * 3 + 2] - mn[2]) * scale;
    }
    KENSHIN_INFO(
        "[import] gltf recentered (-{:.2f},-{:.2f},-{:.2f}){} bbox {:.2f} -> "
        "{:.2f} units",
        cx, cy, mn[2], scale != 1.0f ? " + scaled" : "", size, size * scale);
  }
}

bool splitTranslucent(MeshData& mesh, std::span<const GltfMaterial> materials,
                      TranslucentMesh& out) {
  out.vertices.clear();
  if (mesh.triMaterial.empty()) return false;
  std::vector<u32> keepIdx, keepMat;
  keepIdx.reserve(mesh.indices.size());
  keepMat.reserve(mesh.triMaterial.size());
  const bool hasNrm = mesh.normals.size() == mesh.positions.size();
  const bool hasUv = mesh.uvs.size() == mesh.vertexCount() * 2;
  std::vector<f32>& tverts = out.vertices;
  for (usize t = 0; t < mesh.triMaterial.size(); ++t) {
    const u32 m = mesh.triMaterial[t];
    if (!isBlend(materials, m)) {
      for (int k = 0; k < 3; ++k) keepIdx.push_back(mesh.indices[t * 3 + k]);
      keepMat.push_back(m);
      continue;
    }
    for (int k = 0; k < 3; ++k) {
      const u32 v = mesh.indices[t * 3 + k];
      for (int c = 0; c < 3; ++c) tverts.push_back(mesh.positions[usize(v) * 3 + c]);
      tverts.push_back(f32(m));
      for (int c = 0; c < 3; ++c)
        tverts.push_back(hasNrm ? mesh.normals[usize(v) * 3 + c] : (c == 2 ? 1.0f : 0.0f));
      tverts.push_back(0.0f);
      tverts.push_back(hasUv ? mesh.uvs[usize(v) * 2] : 0.0f);
      tverts.push_back(hasUv ? mesh.uvs[usize(v) * 2 + 1] : 0.0f);
      tverts.push_back(0.0f);
      tverts.push_back(0.0f);
    }
  }
  if (tverts.empty()) return false;
  mesh.indices.swap(keepIdx);
  mesh.triMaterial.swap(keepMat);
  return true;
}

bool writeTranslucentMesh(const std::filesystem::path& path,
                          std::span<const GltfMaterial> materials, const TranslucentMesh& mesh,
                          std::string& log) {
  std::FILE* tf = util::openFile(path, "wb");
  bool ok = tf != nullptr;
  if (ok) {
    const u32 head[3] = {0x314D544Bu /* "KTM1" */, 2u, static_cast<u32>(materials.size())};
    ok = std::fwrite(head, sizeof(head), 1, tf) == 1;
    for (const auto& gm : materials) {
      const f32 rec[8] = {gm.kd.r, gm.kd.g, gm.kd.b, gm.alpha, gm.roughness, gm.metallic,
                          0.0f,    0.0f};
      ok = ok && std::fwrite(rec, sizeof(rec), 1, tf) == 1;
    }
    const u32 vcount = static_cast<u32>(mesh.vertices.size() / 12);
    ok = ok && std::fwrite(&vcount, sizeof(vcount), 1, tf) == 1 &&
         std::fwrite(mesh.vertices.data(), sizeof(f32), mesh.vertices.size(), tf) ==
             mesh.vertices.size();
    ok = std::fclose(tf) == 0 && ok;
  }
  if (!ok) {
    log = "translucent mesh write failed: " + util::pathToUtf8(path);
    return false;
  }
  KENSHIN_INFO("[import] translucent: {} BLEND tris -> {}", mesh.triangleCount(),
               util::pathToUtf8(path));
  return true;
}

void sinkHuggingBlend(MeshData& mesh, std::span<const GltfMaterial> materials, f32 fit) {
  bool anyBlend = false;
  for (const auto& gm : materials) anyBlend |= gm.alpha < 0.999f;
  if (!anyBlend || mesh.triMaterial.empty() || mesh.normals.size() != mesh.positions.size())
    return;
  const f32 kSink = fit > 0.0f ? fit * 0.0015f : 0.003f;
  std::vector<bool> sink(mesh.vertexCount(), false);
  for (usize t = 0; t < mesh.triMaterial.size(); ++t) {
    if (!isBlend(materials, mesh.triMaterial[t])) continue;
    for (int k = 0; k < 3; ++k) sink[mesh.indices[t * 3 + k]] = true;
  }
  // Only materials that HUG opaque geometry sink. Free-standing glass
  // (the F-35 visor stands well off its shell) sinking behind its own
  // front drove the visor rim under the shell's mounting parts. Decided
  // PER MATERIAL (>= 25% of its vertices within kHug of an opaque
  // triangle): a per-vertex sink bends a plate at the hug boundary.
  // Uniform grid of opaque triangles, each inserted into every cell its
  // kHug-expanded bounds touch, so a vertex checks its own cell only.
  const f32 kHug = fit > 0.0f ? fit * 0.005f : 0.01f;
  const f32 cell = kHug * 4.0f;
  std::map<std::tuple<int, int, int>, std::vector<u32>> grid;
  const auto P = [&](u32 v) {
    return glm::vec3(mesh.positions[usize(v) * 3], mesh.positions[usize(v) * 3 + 1],
                     mesh.positions[usize(v) * 3 + 2]);
  };
  const auto key = [&](f32 x) { return static_cast<int>(std::floor(x / cell)); };
  for (usize t = 0; t < mesh.triMaterial.size(); ++t) {
    if (isBlend(materials, mesh.triMaterial[t])) continue;
    const glm::vec3 a = P(mesh.indices[t * 3]), b = P(mesh.indices[t * 3 + 1]),
                    c = P(mesh.indices[t * 3 + 2]);
    const glm::vec3 lo = glm::min(a, glm::min(b, c)) - kHug;
    const glm::vec3 hi = glm::max(a, glm::max(b, c)) + kHug;
    const int x0 = key(lo.x), x1 = key(hi.x), y0 = key(lo.y), y1 = key(hi.y), z0 = key(lo.z),
              z1 = key(hi.z);
    if ((x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1) > 4096)
      continue;  // a huge opaque triangle cannot be "hugged" here
    for (int x = x0; x <= x1; ++x)
      for (int y = y0; y <= y1; ++y)
        for (int z = z0; z <= z1; ++z) grid[{x, y, z}].push_back(static_cast<u32>(t));
  }
  // Per-material vote: a vertex counts toward the BLEND material that
  // references it.
  std::vector<u32> vertMat(mesh.vertexCount(), 0xFFFFFFFFu);
  for (usize t = 0; t < mesh.triMaterial.size(); ++t) {
    const u32 m = mesh.triMaterial[t];
    if (!isBlend(materials, m)) continue;
    for (int k = 0; k < 3; ++k) vertMat[mesh.indices[t * 3 + k]] = m;
  }
  std::vector<usize> hugCount(materials.size(), 0), blendCount(materials.size(), 0);
  for (usize v = 0; v < sink.size(); ++v) {
    if (!sink[v] || vertMat[v] >= materials.size()) continue;
    ++blendCount[vertMat[v]];
    const glm::vec3 p = P(static_cast<u32>(v));
    const auto it = grid.find({key(p.x), key(p.y), key(p.z)});
    if (it == grid.end()) continue;
    for (const u32 t : it->second)
      if (pointTriDist2(p, P(mesh.indices[t * 3]), P(mesh.indices[t * 3 + 1]),
                        P(mesh.indices[t * 3 + 2])) <= kHug * kHug) {
        ++hugCount[vertMat[v]];
        break;
      }
  }
  for (usize m = 0; m < materials.size(); ++m) {
    if (blendCount[m] == 0) continue;
    const bool hugs = hugCount[m] * 4 >= blendCount[m];
    KENSHIN_INFO("[import] BLEND material {}: {}/{} verts hug opaque geometry -> {}", m,
                 hugCount[m], blendCount[m], hugs ? "sink" : "free-standing, kept in place");
    if (hugs) continue;
    for (usize v = 0; v < sink.size(); ++v)
      if (vertMat[v] == m) sink[v] = false;
  }
  usize sunk = 0;
  for (usize v = 0; v < sink.size(); ++v) {
    if (!sink[v]) continue;
    for (int k = 0; k < 3; ++k) mesh.positions[v * 3 + k] -= mesh.normals[v * 3 + k] * kSink;
    ++sunk;
  }
  if (sunk) KENSHIN_INFO("[import] BLEND sink: {} verts -{} units (normal)", sunk, kSink);
}

void separateLayers(MeshData& mesh, f32 fit) {
  if (mesh.triMaterial.empty() || mesh.normals.size() != mesh.positions.size()) return;
  const f32 kLayer = fit > 0.0f ? fit * 0.004f : 0.008f;
  const f32 cell = kLayer * 4.0f;
  const auto P = [&](u32 v) {
    return glm::vec3(mesh.positions[usize(v) * 3], mesh.positions[usize(v) * 3 + 1],
                     mesh.positions[usize(v) * 3 + 2]);
  };
  const auto key = [&](f32 x) { return static_cast<int>(std::floor(x / cell)); };
  std::map<std::tuple<int, int, int>, std::vector<u32>> grid;
  std::vector<u32> vertMat(mesh.vertexCount(), 0xFFFFFFFFu);
  for (usize t = 0; t < mesh.triMaterial.size(); ++t) {
    const u32 m = mesh.triMaterial[t];
    for (int k = 0; k < 3; ++k) vertMat[mesh.indices[t * 3 + k]] = m;
    const glm::vec3 a = P(mesh.indices[t * 3]), b = P(mesh.indices[t * 3 + 1]),
                    c = P(mesh.indices[t * 3 + 2]);
    const glm::vec3 lo = glm::min(a, glm::min(b, c)) - kLayer;
    const glm::vec3 hi = glm::max(a, glm::max(b, c)) + kLayer;
    const int x0 = key(lo.x), x1 = key(hi.x), y0 = key(lo.y), y1 = key(hi.y), z0 = key(lo.z),
              z1 = key(hi.z);
    if ((x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1) > 4096) continue;
    for (int x = x0; x <= x1; ++x)
      for (int y = y0; y <= y1; ++y)
        for (int z = z0; z <= z1; ++z) grid[{x, y, z}].push_back(static_cast<u32>(t));
  }
  bool multi = false;
  for (u32 m : mesh.triMaterial) multi |= m != mesh.triMaterial[0];
  usize moved = 0;
  std::vector<std::pair<usize, glm::vec3>> pushes;  // applied after the scan
  if (multi)
    for (usize v = 0; v < vertMat.size(); ++v) {
      if (vertMat[v] == 0xFFFFFFFFu) continue;
      const glm::vec3 p = P(static_cast<u32>(v));
      glm::vec3 n(mesh.normals[v * 3], mesh.normals[v * 3 + 1], mesh.normals[v * 3 + 2]);
      const f32 nl = glm::length(n);
      if (!(nl > 1e-6f)) continue;
      n /= nl;
      const auto it = grid.find({key(p.x), key(p.y), key(p.z)});
      if (it == grid.end()) continue;
      f32 gap = kLayer;
      for (const u32 t : it->second) {
        if (mesh.triMaterial[t] == vertMat[v]) continue;
        const glm::vec3 a = P(mesh.indices[t * 3]), b = P(mesh.indices[t * 3 + 1]),
                        c = P(mesh.indices[t * 3 + 2]);
        const f32 d2 = pointTriDist2(p, a, b, c);
        if (d2 >= gap * gap) continue;
        // In FRONT: the other surface lies on the normal's side (its
        // plane crosses the normal ray ahead of p).
        const glm::vec3 fn = glm::cross(b - a, c - a);
        const f32 denom = glm::dot(fn, n);
        if (std::abs(denom) < 1e-12f) continue;
        const f32 tHit = glm::dot(fn, a - p) / denom;
        if (tHit <= 0.0f) continue;
        gap = std::sqrt(d2);
      }
      if (gap < kLayer) pushes.emplace_back(v, n * (kLayer - gap));
    }
  for (const auto& [v, d] : pushes) {
    for (int k = 0; k < 3; ++k) mesh.positions[v * 3 + k] -= d[k];
    ++moved;
  }
  if (moved)
    KENSHIN_INFO("[import] layer separation: {} covered verts pushed back to a {} gap", moved,
                 kLayer);
}

void appendBackFaces(MeshData& mesh, std::span<const GltfMaterial> materials, f32 fit) {
  if (mesh.triMaterial.empty()) return;
  // The copy is pushed a hair along ITS normal (behind the original's
  // front side): bit-identical positions get welded by clusterlod's
  // position remap, every edge turns non-manifold and the simplifier
  // locks it.
  const f32 kSplit = fit > 0.0f ? fit * 1e-4f : 2e-4f;
  const usize triCount = mesh.triMaterial.size();
  const bool hasNrm = mesh.normals.size() == mesh.positions.size();
  const bool hasUv = mesh.uvs.size() == mesh.vertexCount() * 2;
  std::vector<u32> remap(mesh.vertexCount(), 0xFFFFFFFFu);
  usize added = 0;
  for (usize t = 0; t < triCount; ++t) {
    const u32 m = mesh.triMaterial[t];
    if (m >= materials.size() || !materials[m].doubleSided) continue;
    u32 tri[3];
    for (int k = 0; k < 3; ++k) {
      const u32 v = mesh.indices[t * 3 + k];
      if (remap[v] == 0xFFFFFFFFu) {
        remap[v] = static_cast<u32>(mesh.vertexCount());
        for (int c = 0; c < 3; ++c)
          mesh.positions.push_back(mesh.positions[usize(v) * 3 + c] -
                                   (hasNrm ? mesh.normals[usize(v) * 3 + c] * kSplit : 0.0f));
        if (hasNrm)
          for (int c = 0; c < 3; ++c) mesh.normals.push_back(-mesh.normals[usize(v) * 3 + c]);
        if (hasUv)
          for (int c = 0; c < 2; ++c) mesh.uvs.push_back(mesh.uvs[usize(v) * 2 + c]);
      }
      tri[k] = remap[v];
    }
    mesh.indices.push_back(tri[0]);  // reversed winding
    mesh.indices.push_back(tri[2]);
    mesh.indices.push_back(tri[1]);
    mesh.triMaterial.push_back(m);
    ++added;
  }
  if (added) KENSHIN_INFO("[import] doubleSided: +{} back-face triangles", added);
}

}  // namespace importer
KENSHIN_NS_END
