#include "kenshin/assets/ProceduralMesh.h"

#define _USE_MATH_DEFINES
#include <cmath>

#include <glm/gtc/constants.hpp>

KENSHIN_NS_BEGIN

namespace assets
{

    MeshData createUvSphere(int segsU, int segsV, f32 radius)
    {
        MeshData mesh;
        if (segsU < 3 || segsV < 2)
            return mesh;

        const int rings = segsV - 1; // interior ring rows (poles handled apart)
        mesh.positions.reserve(static_cast<usize>(2 + rings * segsU) * 3);
        mesh.uvs.reserve(static_cast<usize>(2 + rings * segsU) * 2);

        auto pushVertex = [&mesh, radius](f32 x, f32 y, f32 z, f32 u, f32 v) {
            mesh.positions.insert(mesh.positions.end(), {x, y, z});
            mesh.normals.insert(mesh.normals.end(), {x / radius, y / radius, z / radius});
            mesh.uvs.insert(mesh.uvs.end(), {u, v});
        };

        // North pole (v=1 at top).
        pushVertex(0, radius, 0, 0, 1);
        for (int r = 1; r <= rings; ++r)
        {
            const f32 phi = glm::pi<f32>() * static_cast<f32>(r) / segsV;
            const f32 y = radius * std::cos(phi);
            const f32 s = radius * std::sin(phi);
            for (int u = 0; u < segsU; ++u)
            {
                const f32 theta = 2.0f * glm::pi<f32>() * static_cast<f32>(u) / segsU;
                pushVertex(s * std::sin(theta), y, s * std::cos(theta),
                           static_cast<f32>(u) / segsU,
                           1.0f - static_cast<f32>(r) / segsV);
            }
        }
        // South pole (v=0 at bottom).
        pushVertex(0, -radius, 0, 0, 0);

        const auto interiorIndex = [segsU](int ring, int u) -> u32 {
            return 1 + static_cast<u32>(ring - 1) * segsU + (u % segsU);
        };

        // North cap: (pole, i+1, i) — CCW seen from outside.
        for (int u = 0; u < segsU; ++u)
        {
            const u32 a = 0;
            const u32 b = interiorIndex(1, u + 1);
            const u32 c = interiorIndex(1, u);
            mesh.indices.insert(mesh.indices.end(), {a, b, c});
        }
        // Body quads.
        for (int r = 1; r < rings; ++r)
        {
            for (int u = 0; u < segsU; ++u)
            {
                const u32 a = interiorIndex(r, u);
                const u32 b = interiorIndex(r, u + 1);
                const u32 c = interiorIndex(r + 1, u + 1);
                const u32 d = interiorIndex(r + 1, u);
                mesh.indices.insert(mesh.indices.end(), {a, b, c, a, c, d});
            }
        }
        // South cap.
        const u32 south = mesh.vertexCount() - 1;
        for (int u = 0; u < segsU; ++u)
        {
            const u32 a = south;
            const u32 b = interiorIndex(rings, u);
            const u32 c = interiorIndex(rings, u + 1);
            mesh.indices.insert(mesh.indices.end(), {a, b, c});
        }
        return mesh;
    }

    MeshData createTorusKnot(int p, int q, int tubeSegs, int ringSegs,
                             f32 radius, f32 tubeRadius)
    {
        MeshData mesh;
        if (p < 1 || q < 1 || tubeSegs < 3 || ringSegs < 3)
            return mesh;

        // Canonical (p,q) torus knot on a torus of major radius R and
        // minor radius r (Wikipedia parametrization): winds p times around
        // the Y axis while crossing the tube q times. The earlier three.js
        // transcription produced an elliptical rotating ring (constant x
        // amplitude 0.5R, pulsing z 0.5..1.5R) - a distorted knot.
        const f32 minorRadius = radius * 0.375f;
        auto curve = [&](f32 t, glm::vec3& pos) {
            const f32 cq = std::cos(q * t);
            const f32 ring = radius + minorRadius * cq;
            pos.x = ring * std::cos(p * t);
            pos.y = minorRadius * std::sin(q * t);
            pos.z = ring * std::sin(p * t);
        };

        const usize vertexTotal = static_cast<usize>(tubeSegs + 1) * (ringSegs + 1);
        mesh.positions.reserve(vertexTotal * 3);
        mesh.normals.reserve(vertexTotal * 3);
        mesh.uvs.reserve(vertexTotal * 2);

        // Parallel-transport frame with even seam-twist distribution:
        // winding-consistent along the whole tube by construction. (Both
        // earlier frames flipped orientation on segments where the tangent's
        // azimuth around the Y axis reversed - backface culling then ate
        // those strips: the "incomplete mesh" artifact, 42/256 segments.)
        auto tangentAt = [&](f32 t, glm::vec3& out) {
            glm::vec3 a, b;
            curve(t, a);
            curve(t + 0.01f, b);
            out = glm::normalize(b - a);
        };
        glm::vec3 t0;
        tangentAt(0.0f, t0);
        glm::vec3 seed = glm::vec3(0.0f, 1.0f, 0.0f);
        if (std::abs(glm::dot(seed, t0)) > 0.9f)
            seed = glm::vec3(1.0f, 0.0f, 0.0f);
        glm::vec3 prevBinormal =
            glm::normalize(glm::cross(t0, glm::cross(seed, t0)));

        // Measure holonomy: transport once around, compare with the start.
        glm::vec3 transported = prevBinormal;
        for (int i = 0; i < tubeSegs; ++i)
        {
            const f32 t = static_cast<f32>(i) / tubeSegs * glm::pi<f32>() * 2.0f;
            glm::vec3 tangent;
            tangentAt(t, tangent);
            transported = glm::normalize(
                transported - tangent * glm::dot(tangent, transported));
        }
        const f32 seamAngle = std::atan2(
            glm::dot(glm::cross(prevBinormal, transported), t0),
            glm::dot(prevBinormal, transported));
        const f32 twistPerStep = -seamAngle / static_cast<f32>(tubeSegs);

        glm::vec3 binormal = prevBinormal;
        for (int i = 0; i <= tubeSegs; ++i)
        {
            const f32 t = static_cast<f32>(i) / tubeSegs * glm::pi<f32>() * 2.0f;

            glm::vec3 p1;
            curve(t, p1);
            glm::vec3 tangent;
            tangentAt(t, tangent);

            // Transport the previous binormal onto this tangent plane...
            binormal = glm::normalize(binormal -
                                      tangent * glm::dot(tangent, binormal));
            // ...and apply the distributed seam twist so ring tubeSegs
            // coincides exactly with ring 0.
            const f32 a = twistPerStep * static_cast<f32>(i);
            const glm::vec3 side = glm::normalize(glm::cross(tangent, binormal));
            binormal = glm::normalize(binormal * std::cos(a) + side * std::sin(a));
            const glm::vec3 normal = glm::cross(binormal, tangent);

            for (int j = 0; j <= ringSegs; ++j)
            {
                const f32 v = static_cast<f32>(j) / ringSegs * glm::pi<f32>() * 2.0f;
                const f32 cx = -tubeRadius * std::cos(v);
                const f32 cy = tubeRadius * std::sin(v);

                const glm::vec3 pos = p1 + cx * normal + cy * binormal;
                mesh.positions.insert(mesh.positions.end(), {pos.x, pos.y, pos.z});
                const glm::vec3 nrm = glm::normalize(pos - p1);
                mesh.normals.insert(mesh.normals.end(), {nrm.x, nrm.y, nrm.z});
                mesh.uvs.insert(mesh.uvs.end(),
                                {static_cast<f32>(i) / tubeSegs,
                                 static_cast<f32>(j) / ringSegs});
            }
        }

        const auto idx = [&](int i, int j) -> u32 {
            return static_cast<u32>(i * (ringSegs + 1) + (j % (ringSegs + 1)));
        };
        for (int i = 0; i < tubeSegs; ++i)
        {
            for (int j = 0; j < ringSegs; ++j)
            {
                const u32 a = idx(i, j);
                const u32 b = idx(i + 1, j);
                const u32 c = idx(i + 1, j + 1);
                const u32 d = idx(i, j + 1);
                mesh.indices.insert(mesh.indices.end(), {a, b, c, a, c, d});
            }
        }
        return mesh;
    }

} // namespace assets

KENSHIN_NS_END
