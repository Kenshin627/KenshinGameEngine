#include "KgeoPreview.h"

#include "kenshin/core/Log.h"

#pragma warning(push, 0)
#include <imgui.h>
#pragma warning(pop)

namespace demo
{

    bool loadKgeo(const std::string& path, KgeoPreview& preview, std::string& log)
    {
        if (!kenshin::geometry::readKGeo(path, preview.dag, log))
            return false;
        preview.needRebuild = true;
        return true;
    }

    /// Rebuild the drawable mesh from the current cut (baseline path):
    /// concatenate the cut clusters' local vertices/indices into one buffer.
    static void rebuildCutMesh(KgeoPreview& preview,
                               kenshin::backend::IRenderBackend* backend)
    {
        const auto& dag = preview.dag;
        kenshin::assets::MeshData mesh;
        for (kenshin::u32 c : preview.cut)
        {
            const auto& cluster = dag.clusters[c];
            const kenshin::u32 base = mesh.vertexCount();
            for (kenshin::u32 v = 0; v < cluster.vertexCount; ++v)
            {
                const kenshin::u32 src = cluster.vertexOffset + v;
                for (int k = 0; k < 3; ++k)
                    mesh.positions.push_back(dag.positions[src * 3 + k]);
                for (int k = 0; k < 3; ++k)
                    mesh.normals.push_back(dag.normals[src * 3 + k]);
                mesh.uvs.push_back(dag.uvs[src * 2]);
                mesh.uvs.push_back(dag.uvs[src * 2 + 1]);
            }
            for (kenshin::u32 i = 0; i < cluster.indexCount; ++i)
                mesh.indices.push_back(base + dag.indices[cluster.indexOffset + i]);
        }
        if (preview.meshHandle != kenshin::backend::IRenderBackend::kInvalidMeshHandle)
            backend->destroyMeshBuffer(preview.meshHandle);
        preview.meshHandle = kenshin::backend::IRenderBackend::kInvalidMeshHandle;
        if (mesh.indices.empty())
            return;
        kenshin::backend::IRenderBackend::MeshUpload upload;
        upload.positions = mesh.positions.data();
        upload.normals = mesh.normals.data();
        upload.uvs = mesh.uvs.data();
        upload.vertexCount = mesh.vertexCount();
        upload.indices = mesh.indices.data();
        upload.indexCount = static_cast<kenshin::u32>(mesh.indices.size());
        preview.meshHandle = backend->createMeshBuffer(upload);
    }

    void updateCut(KgeoPreview& preview, kenshin::backend::IRenderBackend* backend,
                   const kenshin::scene::Camera& camera,
                   const kenshin::scene::CameraController& controller,
                   kenshin::u32 viewportHeight)
    {
        std::vector<kenshin::u32> cut;
        const kenshin::f32 camPos[3] = {camera.position().x, camera.position().y,
                                        camera.position().z};
        kenshin::geometry::selectLodCut(preview.dag, preview.pixelThreshold,
                                        viewportHeight, controller.fovYDeg, camPos,
                                        cut);
        if (cut != preview.cut)
        {
            preview.cut = std::move(cut);
            preview.needRebuild = true;
            preview.lastStats =
                kenshin::geometry::summarizeCut(preview.dag, preview.cut);
            preview.history.push_back(
                static_cast<float>(preview.lastStats.triangleCount));
            if (preview.history.size() > 240)
                preview.history.erase(preview.history.begin());
            KENSHIN_INFO("[kgeo] cut: {} clusters / {} tris (levels {}..{}) "
                         "at {:.2f}px",
                         preview.lastStats.clusterCount,
                         preview.lastStats.triangleCount,
                         preview.lastStats.minLevel,
                         preview.lastStats.maxLevel,
                         preview.pixelThreshold);
        }
        if (preview.needRebuild)
        {
            rebuildCutMesh(preview, backend);
            preview.needRebuild = false;
        }
    }

    void drawLodPanel(KgeoPreview& preview)
    {
        ImGui::Begin("LOD Preview");
        ImGui::SliderFloat("pixel error", &preview.pixelThreshold, 0.05f, 32.0f,
                           "%.2f px", ImGuiSliderFlags_Logarithmic);
        ImGui::Checkbox("level coloring (shading off)", &preview.lodColors);
        const auto& st = preview.lastStats;
        ImGui::Text("cut: %u clusters / %u tris", st.clusterCount,
                    st.triangleCount);
        ImGui::Text("levels %d..%d", st.minLevel, st.maxLevel);
        ImGui::Text("source: %u tris", preview.dag.sourceTriangleCount);
        if (!preview.history.empty() &&
            ImGui::BeginChild("history", ImVec2(0, 80), true))
        {
            ImGui::PlotLines("##tris", preview.history.data(),
                             static_cast<int>(preview.history.size()), 0,
                             nullptr, 0.0f,
                             static_cast<float>(preview.dag.sourceTriangleCount),
                             ImVec2(0, 60));
            ImGui::EndChild();
        }
        ImGui::End();
    }

} // namespace demo
