import re
p = 'app/src/main.cpp'
s = open(p, encoding='utf-8').read()

# 1) loadDemoMesh: add torus / cone / cylinder / plane
old = '''        if (name == "knot")
        {
            out = kenshin::geometry::TorusKnotGeometry::create(2, 3, 256, 32, 1.0f);
            return true;
        }
        return kenshin::assets::loadObj(name, out, log);'''
new = '''        if (name == "knot")
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
        return kenshin::assets::loadObj(name, out, log);'''
assert old in s
s = s.replace(old, new)

# 2) single-mesh setup -> scene vector
old = '''    // ---- baseline mesh path (S2.2) --------------------------------------
    kenshin::assets::MeshData demoMesh;
    {
        std::string meshLog;
        if (!loadDemoMesh(args.mesh, demoMesh, meshLog))
        {
            KENSHIN_ERROR("mesh load failed: {}", meshLog);
            backend->shutdown();
            ImGui::DestroyContext();
            kenshin::log::shutdown();
            return 1;
        }
    }

    kenshin::manager::ShaderManager shaders;
    kenshin::u32 meshHandle = kenshin::backend::IRenderBackend::kInvalidMeshHandle;
    glm::vec3 meshKd(0.8f, 0.7f, 0.35f);
    if (shaders.init(findShaderRoot()))
    {
        std::vector<kenshin::u32> vsSpirv, fsSpirv;
        std::string shaderLog;
        if (shaders.compileFile("baseline.slang", vsSpirv, shaderLog, "vsMain") &&
            shaders.compileFile("baseline.slang", fsSpirv, shaderLog, "fsMain"))
        {
            backend->stageBaselineShaders(vsSpirv.data(), vsSpirv.size() * 4,
                                          fsSpirv.data(), fsSpirv.size() * 4);
            kenshin::backend::IRenderBackend::MeshUpload upload;
            upload.positions = demoMesh.positions.data();
            upload.normals = demoMesh.normals.data();
            upload.uvs = demoMesh.uvs.data();
            upload.vertexCount = demoMesh.vertexCount();
            upload.indices = demoMesh.indices.data();
            upload.indexCount = static_cast<kenshin::u32>(demoMesh.indices.size());
            meshHandle = backend->createMeshBuffer(upload);
            KENSHIN_INFO("[baseline] mesh '{}' uploaded: {} verts / {} tris (handle {})",
                         args.mesh, demoMesh.vertexCount(), demoMesh.triangleCount(),
                         meshHandle);
            meshKd = glm::vec3(demoMesh.material.kd);
        }
        else
        {
            KENSHIN_WARN("[baseline] shader compile failed; viewport shows only the "
                         "clear color. Log: {}", shaderLog);
        }
    }'''
new = '''    // ---- baseline demo scene (S2.2): one instance of every shape, spread
    // in a row on the X axis. --mesh <name> selects a single shape; --mesh
    // all (the default) lays out the full scene; --kgeo switches to the
    // virtual-geometry preview entirely.
    struct SceneShape
    {
        const char* name;
        kenshin::u32 handle;
        glm::vec3 offset;
        glm::vec3 kd;
    };
    std::vector<SceneShape> scene;
    if (args.kgeo.empty())
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
        const f32 spacing = 5.0f;
        const f32 x0 = -0.5f * spacing * static_cast<f32>(count - 1);
        const bool all = args.mesh == "all";
        for (size_t i = 0; i < count; ++i)
        {
            if (!all && args.mesh != defs[i].name)
                continue;
            scene.push_back({defs[i].name,
                             kenshin::backend::IRenderBackend::kInvalidMeshHandle,
                             all ? glm::vec3(x0 + spacing * static_cast<f32>(i),
                                             0.0f, 0.0f)
                                 : glm::vec3(0.0f),
                             defs[i].kd});
        }
        if (scene.empty())
        {
            // unrecognized --mesh name: treat as OBJ path, single instance
            scene.push_back({args.mesh.c_str(),
                             kenshin::backend::IRenderBackend::kInvalidMeshHandle,
                             glm::vec3(0.0f), glm::vec3(0.8f, 0.7f, 0.35f)});
        }
    }

    kenshin::manager::ShaderManager shaders;
    kenshin::u32 meshHandle = kenshin::backend::IRenderBackend::kInvalidMeshHandle;
    glm::vec3 meshKd(0.8f, 0.7f, 0.35f);
    if (shaders.init(findShaderRoot()))
    {
        std::vector<kenshin::u32> vsSpirv, fsSpirv;
        std::string shaderLog;
        if (shaders.compileFile("baseline.slang", vsSpirv, shaderLog, "vsMain") &&
            shaders.compileFile("baseline.slang", fsSpirv, shaderLog, "fsMain"))
        {
            backend->stageBaselineShaders(vsSpirv.data(), vsSpirv.size() * 4,
                                          fsSpirv.data(), fsSpirv.size() * 4);
            bool any = false;
            for (auto& shape : scene)
            {
                kenshin::assets::MeshData m;
                std::string meshLog;
                if (!loadDemoMesh(shape.name, m, meshLog))
                {
                    KENSHIN_ERROR("mesh '{}' load failed: {}", shape.name, meshLog);
                    continue;
                }
                kenshin::backend::IRenderBackend::MeshUpload upload;
                upload.positions = m.positions.data();
                upload.normals = m.normals.data();
                upload.uvs = m.uvs.data();
                upload.vertexCount = m.vertexCount();
                upload.indices = m.indices.data();
                upload.indexCount = static_cast<kenshin::u32>(m.indices.size());
                shape.handle = backend->createMeshBuffer(upload);
                KENSHIN_INFO("[baseline] mesh '{}' uploaded: {} verts / {} tris",
                             shape.name, m.vertexCount(), m.triangleCount());
                any = true;
            }
            if (any && !scene.empty())
                meshHandle = scene.front().handle;
        }
        else
        {
            KENSHIN_WARN("[baseline] shader compile failed; viewport shows only the "
                         "clear color. Log: {}", shaderLog);
        }
    }'''
assert old in s
s = s.replace(old, new)
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print("part A ok")
