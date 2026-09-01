#include "kenshin/renderer/BaselineForwardPipeline.h"

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/manager/ShaderManager.h"

#include <cstring>
#include <string>
#include <vector>

KENSHIN_NS_BEGIN

namespace renderer
{

    bool BaselineForwardPipeline::init(backend::IRenderBackend* /*backend*/,
                                       manager::ShaderManager& shaders)
    {
        std::string log;
        if (!shaders.compileFile("baseline.slang", m_vsSpirv, log, "vsMain") ||
            !shaders.compileFile("baseline.slang", m_fsSpirv, log, "fsMain"))
        {
            // Matching the long-standing demo behavior: a missing/broken
            // shader degrades to a clear-color viewport, the app keeps
            // running (the LOD pipeline is recompiled on next launch).
            KENSHIN_WARN("[baseline] shader compile failed; viewport shows only "
                         "the clear color. Log: {}", log);
            return true;
        }
        return true;
    }

    void BaselineForwardPipeline::ensurePipeline(backend::IRenderBackend* backend)
    {
        if (m_pipeline != 0xFFFFFFFFu || m_vsSpirv.empty() || m_fsSpirv.empty())
            return;

        // Pipeline policy (verified 2026-09-01): depth is [0,1] ZO with
        // near=0 (perspectiveRH_ZO + clear 1.0), so LESS keeps the nearest.
        // Meshes are CCW in object space and STAY CCW at the rasterizer: the
        // projection's clip-space Y negation and Vulkan's NDC->viewport Y
        // mapping each flip once — two flips cancel. (A single-flip
        // CLOCKWISE derivation was wrong, and closed meshes could not show
        // it — only the knot's thin tubes did.)
        backend::IRenderBackend::GraphicsPipelineInfo info;
        info.vsSpirv = m_vsSpirv.data();
        info.vsBytes = m_vsSpirv.size() * sizeof(u32);
        info.fsSpirv = m_fsSpirv.data();
        info.fsBytes = m_fsSpirv.size() * sizeof(u32);
        info.depthTest = true;
        info.depthWrite = true;
        info.depthCompare = backend::IRenderBackend::CompareOp::Less;
        info.cullMode = backend::IRenderBackend::CullMode::Back;
        info.frontFace = backend::IRenderBackend::FrontFace::CounterClockwise;
        m_pipeline = backend->createGraphicsPipeline(info);
        if (m_pipeline == backend::IRenderBackend::kInvalidPipelineHandle)
            KENSHIN_ERROR("[baseline] pipeline creation failed (viewport will "
                          "show only the clear color)");
    }

    void BaselineForwardPipeline::shutdown(backend::IRenderBackend* backend)
    {
        if (m_pipeline != 0xFFFFFFFFu)
            backend->destroyGraphicsPipeline(m_pipeline);
        m_pipeline = 0xFFFFFFFFu;
    }

    void BaselineForwardPipeline::execute(const FrameContext& ctx)
    {
        ensurePipeline(ctx.backend);
        if (m_pipeline == 0xFFFFFFFFu)
            return;

        for (const DrawItem& item : ctx.items)
        {
            if (item.mesh == assets::kInvalidMeshHandle)
                continue;
            // Slang push-constant matrices read the 64 bytes row-major;
            // GLM stores column-major, so upload the transpose.
            const glm::mat4 viewProj =
                glm::transpose(ctx.view->viewProj * item.model);
            const glm::vec4 kd(item.kd, 1.0f);
            const glm::vec4 light(ctx.view->lightDir, 0.0f);
            const glm::vec4 cam(ctx.view->camPos, 0.0f);

            backend::IRenderBackend::DrawData data;
            std::memcpy(data.viewProj, &viewProj, sizeof(data.viewProj));
            std::memcpy(data.kd, &kd, sizeof(data.kd));
            std::memcpy(data.lightDir, &light, sizeof(data.lightDir));
            std::memcpy(data.camPos, &cam, sizeof(data.camPos));
            ctx.backend->draw(m_pipeline, item.mesh, data);
        }
    }

} // namespace renderer

KENSHIN_NS_END
