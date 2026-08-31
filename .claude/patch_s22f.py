import re

p = 'app/src/main.cpp'
s = open(p, encoding='utf-8').read()

# After ImGui init + before the frame loop: load mesh, compile baseline shader,
# upload. Anchor on the camera-checked block start.
old = """    UiState ui;
    ui.viewportSize = ImVec2(static_cast<float>(window.width()),
                             static_cast<float>(window.height()));"""
new = """    // ---- baseline mesh path (S2.2) --------------------------------------
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
            backend->initBaselinePipeline(vsSpirv.data(), vsSpirv.size() * 4,
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
            KENSHIN_WARN("[baseline] shader compile failed; viewport shows only "
                         "the clear color. Log: {}", shaderLog);
        }
    }

    UiState ui;
    ui.viewportSize = ImVec2(static_cast<float>(window.width()),
                             static_cast<float>(window.height()));"""
assert old in s
s = s.replace(old, new)

# In the frame loop after camera update, draw the mesh
old = """        if (!cameraChecked && camera.isOrthonormal())
        {
            KENSHIN_INFO("[camera] view basis orthonormal: OK");
            cameraChecked = true;
        }

        backend->endFrame();"""
new = """        if (!cameraChecked && camera.isOrthonormal())
        {
            KENSHIN_INFO("[camera] view basis orthonormal: OK");
            cameraChecked = true;
        }

        // Baseline draw (S2.2): the mesh renders into the viewport target
        // between beginFrame and endFrame.
        if (meshHandle != kenshin::backend::IRenderBackend::kInvalidMeshHandle &&
            ui.viewportSize.x > 1.0f && ui.viewportSize.y > 1.0f)
        {
            const f32 aspect = ui.viewportSize.x / ui.viewportSize.y;
            const glm::mat4 viewProj = camera.projection(aspect) * camera.view();
            kenshin::backend::IRenderBackend::MeshDraw draw;
            draw.mesh = meshHandle;
            std::memcpy(draw.viewProj, &viewProj, sizeof(draw.viewProj));
            const f32 kdArr[4] = {meshKd.r, meshKd.g, meshKd.b, 1.0f};
            std::memcpy(draw.kd, kdArr, sizeof(draw.kd));
            backend->drawMesh(draw);
        }

        backend->endFrame();"""
assert old in s
s = s.replace(old, new)

# glm include
old = '#include "kenshin/scene/Camera.h"'
new = '#include "kenshin/scene/Camera.h"\n\n#include <glm/glm.hpp>'
assert old in s
s = s.replace(old, new)
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print("app part2 ok")
