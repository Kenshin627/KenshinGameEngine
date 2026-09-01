#include "DemoScene.h"

#include "kenshin/assets/ObjLoader.h"
#include "kenshin/core/Log.h"
#include "kenshin/geometry/shapes/BoxGeometry.h"
#include "kenshin/geometry/shapes/CylinderGeometry.h"
#include "kenshin/geometry/shapes/PlaneGeometry.h"
#include "kenshin/geometry/shapes/SphereGeometry.h"
#include "kenshin/geometry/shapes/TorusGeometry.h"
#include "kenshin/geometry/shapes/TorusKnotGeometry.h"

namespace demo
{

    bool loadMesh(const std::string& name, kenshin::assets::MeshData& out,
                  std::string& log)
    {
        if (name == "cube")
        {
            out = kenshin::geometry::BoxGeometry::create(1.0f, 1.0f, 1.0f);
            return true;
        }
        if (name == "sphere")
        {
            out = kenshin::geometry::SphereGeometry::create(64, 32, 1.0f);
            return true;
        }
        if (name == "knot")
        {
            out = kenshin::geometry::TorusKnotGeometry::create(2, 3, 256, 32, 1.0f);
            return true;
        }
        if (name == "torus")
        {
            out = kenshin::geometry::TorusGeometry::create(1.2f, 0.45f, 24, 96);
            return true;
        }
        if (name == "cone")
        {
            out = kenshin::geometry::ConeGeometry::create(1.0f, 2.0f, 48);
            return true;
        }
        if (name == "cylinder")
        {
            out = kenshin::geometry::CylinderGeometry::create(0.8f, 0.8f, 2.0f, 48);
            return true;
        }
        if (name == "plane")
        {
            out = kenshin::geometry::PlaneGeometry::create(3.0f, 3.0f, 12, 12);
            return true;
        }
        return kenshin::assets::loadObj(name, out, log);
    }

    std::vector<SceneShape> buildScene(const std::string& meshArg)
    {
        struct Def
        {
            const char* name;
            glm::vec3 kd;
        };
        const Def defs[] = {
            {"cube", glm::vec3(0.80f, 0.55f, 0.30f)},
            {"sphere", glm::vec3(0.75f, 0.30f, 0.35f)},
            {"torus", glm::vec3(0.30f, 0.65f, 0.80f)},
            {"cylinder", glm::vec3(0.55f, 0.75f, 0.35f)},
            {"cone", glm::vec3(0.80f, 0.80f, 0.30f)},
            {"knot", glm::vec3(0.80f, 0.70f, 0.35f)},
        };
        const size_t count = sizeof(defs) / sizeof(defs[0]);
        const kenshin::f32 spacing = 5.0f;
        const kenshin::f32 x0 = -0.5f * spacing * static_cast<kenshin::f32>(count - 1);
        const bool all = meshArg == "all";

        std::vector<SceneShape> scene;
        for (size_t i = 0; i < count; ++i)
        {
            if (!all && meshArg != defs[i].name)
                continue;
            scene.push_back({defs[i].name, kenshin::assets::kInvalidMeshHandle,
                             all ? glm::vec3(x0 + spacing * static_cast<kenshin::f32>(i),
                                             0.0f, 0.0f)
                                 : glm::vec3(0.0f),
                             defs[i].kd});
        }
        if (scene.empty())
        {
            // unrecognized --mesh name: treat as OBJ path, single instance
            scene.push_back({meshArg.c_str(), kenshin::assets::kInvalidMeshHandle,
                             glm::vec3(0.0f), glm::vec3(0.8f, 0.7f, 0.35f)});
        }
        return scene;
    }

    bool uploadScene(std::vector<SceneShape>& scene, kenshin::manager::MeshPool& pool)
    {
        bool any = false;
        for (auto& shape : scene)
        {
            kenshin::assets::MeshData m;
            std::string meshLog;
            if (!loadMesh(shape.name, m, meshLog))
            {
                KENSHIN_ERROR("mesh '{}' load failed: {}", shape.name, meshLog);
                continue;
            }
            shape.handle = pool.upload(m);
            KENSHIN_INFO("[baseline] mesh '{}' uploaded: {} verts / {} tris",
                         shape.name, m.vertexCount(), m.triangleCount());
            any = true;
        }
        return any;
    }

} // namespace demo
