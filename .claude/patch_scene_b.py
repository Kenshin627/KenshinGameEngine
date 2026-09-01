p = 'app/src/main.cpp'
s = open(p, encoding='utf-8').read()

old = '''        const kenshin::u32 activeMesh =
            kgeoMode ? kgeoPreview.meshHandle : meshHandle;'''
new = '''        const kenshin::u32 activeMesh =
            kgeoMode ? kgeoPreview.meshHandle : meshHandle;
        (void)activeMesh; // the scene / kgeo loops below issue the draws'''
assert old in s
s = s.replace(old, new)

old = '''        if (activeMesh != kenshin::backend::IRenderBackend::kInvalidMeshHandle &&
            ui.viewportSize.x > 1.0f && ui.viewportSize.y > 1.0f)
        {
            const kenshin::f32 aspect = ui.viewportSize.x / ui.viewportSize.y;
            // Slang push-constant matrices read the 64 bytes row-major;
            // GLM stores column-major, so upload the transpose.
            const glm::mat4 viewProj = glm::transpose(camera.projection(aspect) * camera.view());
            kenshin::backend::IRenderBackend::MeshDraw draw;
            draw.mesh = activeMesh;
            std::memcpy(draw.viewProj, &viewProj, sizeof(draw.viewProj));
            kenshin::f32 kdArr[4] = {meshKd.r, meshKd.g, meshKd.b, 1.0f};
            if (kgeoMode)
            {
                kdArr[0] = 0.75f; kdArr[1] = 0.75f; kdArr[2] = 0.78f; kdArr[3] = 1.0f;
            }
            std::memcpy(draw.kd, kdArr, sizeof(draw.kd));
            // Studio light from upper-left-front; camera position feeds the
            // Blinn-Phong specular half-vector.
            const kenshin::f32 light[4] = {0.35f, 0.8f, 0.45f, 0.0f};
            std::memcpy(draw.lightDir, light, sizeof(draw.lightDir));
            const kenshin::f32 camArr[4] = {camera.position.x, camera.position.y,
                                            camera.position.z, 0.0f};
            std::memcpy(draw.camPos, camArr, sizeof(draw.camPos));
            backend->drawMesh(draw);
        }'''
new = '''        const bool drawable =
            ui.viewportSize.x > 1.0f && ui.viewportSize.y > 1.0f;
        if (!kgeoMode && !args.depthProbe && drawable && !scene.empty())
        {
            const kenshin::f32 aspect = ui.viewportSize.x / ui.viewportSize.y;
            for (const auto& shape : scene)
            {
                if (shape.handle == kenshin::backend::IRenderBackend::kInvalidMeshHandle)
                    continue;
                // Slang push-constant matrices read the 64 bytes row-major;
                // GLM stores column-major, so upload the transpose.
                const glm::mat4 model = glm::translate(glm::mat4(1.0f), shape.offset);
                const glm::mat4 viewProj =
                    glm::transpose(camera.projection(aspect) * camera.view() * model);
                kenshin::backend::IRenderBackend::MeshDraw draw;
                draw.mesh = shape.handle;
                std::memcpy(draw.viewProj, &viewProj, sizeof(draw.viewProj));
                const kenshin::f32 kdArr[4] = {shape.kd.r, shape.kd.g, shape.kd.b, 1.0f};
                std::memcpy(draw.kd, kdArr, sizeof(draw.kd));
                // Studio light from upper-left-front; camera position feeds the
                // Blinn-Phong specular half-vector.
                const kenshin::f32 light[4] = {0.35f, 0.8f, 0.45f, 0.0f};
                std::memcpy(draw.lightDir, light, sizeof(draw.lightDir));
                const kenshin::f32 camArr[4] = {camera.position.x, camera.position.y,
                                                camera.position.z, 0.0f};
                std::memcpy(draw.camPos, camArr, sizeof(draw.camPos));
                backend->drawMesh(draw);
            }
        }
        else if (kgeoMode && drawable && kgeoPreview.meshHandle !=
                                           kenshin::backend::IRenderBackend::kInvalidMeshHandle)
        {
            const kenshin::f32 aspect = ui.viewportSize.x / ui.viewportSize.y;
            const glm::mat4 viewProj =
                glm::transpose(camera.projection(aspect) * camera.view());
            kenshin::backend::IRenderBackend::MeshDraw draw;
            draw.mesh = kgeoPreview.meshHandle;
            std::memcpy(draw.viewProj, &viewProj, sizeof(draw.viewProj));
            const kenshin::f32 kdArr[4] = {0.75f, 0.75f, 0.78f, 1.0f};
            std::memcpy(draw.kd, kdArr, sizeof(draw.kd));
            const kenshin::f32 light[4] = {0.35f, 0.8f, 0.45f, 0.0f};
            std::memcpy(draw.lightDir, light, sizeof(draw.lightDir));
            const kenshin::f32 camArr[4] = {camera.position.x, camera.position.y,
                                            camera.position.z, 0.0f};
            std::memcpy(draw.camPos, camArr, sizeof(draw.camPos));
            backend->drawMesh(draw);
        }'''
assert old in s
s = s.replace(old, new)
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print("part B ok")
