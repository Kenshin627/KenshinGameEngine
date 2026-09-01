#pragma once

// ── Renderer — frame orchestrator + render data contract ─────────
// The renderer layer decides WHAT is drawn; the backend decides HOW.
// This header defines the pure-data contract (FrameView / DrawItem /
// FrameContext / IRenderPipeline) plus the Renderer subsystem that
// sequences frames and dispatches to the active pipeline strategy.
//
// Layering discipline (renderEngine / Filament / UE-Godot consensus):
// - Renderer holds no meshes, cameras or lights — the app assembles the
//   frame data (FrameView + DrawItems) and hands it over per frame.
// - A pipeline strategy (IRenderPipeline) owns the pass sequence and the
//   graphics-pipeline state policy; it draws through backend primitives.
// - This header mentions no backend type: apps never include the backend.

#include "kenshin/Common.h"
#include "kenshin/assets/MeshData.h"

#include <glm/glm.hpp>

#include <memory>
#include <span>
#include <vector>

KENSHIN_NS_BEGIN

class Engine;
namespace backend { class IRenderBackend; }
namespace manager { class ShaderManager; }

namespace renderer
{

    /// Per-frame view data (pure numbers — cameras stay on the scene side).
    struct FrameView
    {
        glm::mat4 viewProj {1.0f};
        glm::vec3 camPos {0.0f};
        glm::vec3 lightDir {0.0f, 0.0f, 1.0f}; // TO light, world
    };

    /// One drawable: mesh + world transform + minimal material.
    struct DrawItem
    {
        assets::MeshHandle mesh = assets::kInvalidMeshHandle;
        glm::mat4 model {1.0f};
        glm::vec3 kd {0.8f, 0.8f, 0.8f};
    };

    /// Everything a pipeline needs to render one frame (internal contract;
    /// pipeline code is engine-side and may use the backend).
    struct FrameContext
    {
        backend::IRenderBackend* backend = nullptr;
        const FrameView* view = nullptr;
        std::span<const DrawItem> items;
        u32 width = 0;
        u32 height = 0;
    };

    /// The rendering strategy for a frame: a stateless sequence of passes
    /// that consumes the draw stream through the backend. Implementations
    /// define their own graphics-pipeline state (shaders, depth, culling).
    class IRenderPipeline
    {
    public:
        virtual ~IRenderPipeline() = default;

        /// Compile shaders + create the backend pipeline objects.
        virtual bool init(backend::IRenderBackend* backend,
                          manager::ShaderManager& shaders) = 0;
        /// Destroy the backend pipeline objects (reverse of init).
        virtual void shutdown(backend::IRenderBackend* backend) = 0;
        /// Record the frame's passes. Called once per frame between the
        /// backend's beginFrame and endFrame.
        virtual void execute(const FrameContext& ctx) = 0;
    };

    /// Frame orchestrator subsystem. The app drives:
    ///   beginFrame(...) → overlay/UI between → submit(view, items) → endFrame()
    /// endFrame builds the FrameContext and dispatches to the active
    /// pipeline, then ends the backend frame (present stays there until a
    /// render graph arrives, see todoList).
    class Renderer
    {
    public:
        struct Config
        {
            bool validation = true; // enable the Vulkan validation layer
        };

        explicit Renderer(Config config = {}) : m_config(config) {}

        /// Subsystem hook: creates + registers the render backend, enforces
        /// the int64-atomics capability gate, then initializes the pipeline.
        bool onInit(Engine& engine);
        void onShutdown();

        /// Returns false when the backend could not begin (caller stops).
        bool beginFrame(u32 width, u32 height, const f32 clearColor[4]);
        void submit(const FrameView& view, std::span<const DrawItem> items);
        void endFrame();

        /// Opaque texture id of the current viewport target for the debug
        /// overlay (valid between beginFrame and endFrame).
        void* viewportTexture();

        // Verification channel (screenshots): frame content / incl. UI.
        void requestFrameCapture();
        bool readbackViewport(std::vector<u8>& rgbaOut, u32& width, u32& height);
        bool readbackFrame(std::vector<u8>& rgbaOut, u32& width, u32& height);

    private:
        Engine* m_engine = nullptr;
        backend::IRenderBackend* m_backend = nullptr;
        std::unique_ptr<IRenderPipeline> m_pipeline;
        Config m_config;

        FrameView m_frameView;
        std::vector<DrawItem> m_stream;
        u32 m_width = 0;
        u32 m_height = 0;
    };

} // namespace renderer

KENSHIN_NS_END
