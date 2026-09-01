#pragma once

// ── MeshPool — GPU mesh resource lifetime ────────────────────────
// The one entry point for uploading meshes to device memory. The scene
// layer holds HANDLES (the DrawItem currency); this pool owns the device
// buffers behind them, destroys everything live at engine shutdown
// (before device teardown), and is the natural M6 streaming host
// (paged upload / eviction / residency budget grow here).

#include "kenshin/Common.h"
#include "kenshin/assets/MeshData.h"

#include <vector>

KENSHIN_NS_BEGIN

class Engine;
namespace backend { class IRenderBackend; }

namespace manager
{

    class MeshPool
    {
    public:
        /// Subsystem hook: resolve the render backend from the registry.
        bool onInit(Engine& engine);

        /// Upload a mesh (staging copy, one-time). Returns the handle or
        /// kInvalidMeshHandle on failure.
        assets::MeshHandle upload(const assets::MeshData& mesh);

        /// Destroy one uploaded mesh (no-op for unknown handles).
        void destroy(assets::MeshHandle handle);

        /// Destroy every live mesh — runs before backend teardown (LIFO).
        void onShutdown();

    private:
        backend::IRenderBackend* m_backend = nullptr;
        std::vector<assets::MeshHandle> m_live;
    };

} // namespace manager

KENSHIN_NS_END
