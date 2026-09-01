#include "kenshin/manager/MeshPool.h"

#include "kenshin/Engine.h"
#include "kenshin/backend/RenderBackend.h"

KENSHIN_NS_BEGIN

namespace manager
{

    bool MeshPool::onInit(Engine& engine)
    {
        m_backend = &engine.get<backend::IRenderBackend>();
        return true;
    }

    assets::MeshHandle MeshPool::upload(const assets::MeshData& mesh)
    {
        if (!m_backend)
            return assets::kInvalidMeshHandle;
        backend::IRenderBackend::MeshUpload upload;
        upload.positions = mesh.positions.data();
        upload.normals = mesh.normals.data();
        upload.uvs = mesh.uvs.data();
        upload.vertexCount = mesh.vertexCount();
        upload.indices = mesh.indices.data();
        upload.indexCount = static_cast<u32>(mesh.indices.size());
        const assets::MeshHandle handle = m_backend->createMeshBuffer(upload);
        if (handle != assets::kInvalidMeshHandle)
            m_live.push_back(handle);
        return handle;
    }

    void MeshPool::destroy(assets::MeshHandle handle)
    {
        if (!m_backend || handle == assets::kInvalidMeshHandle)
            return;
        m_backend->destroyMeshBuffer(handle);
        for (usize i = 0; i < m_live.size(); ++i)
        {
            if (m_live[i] == handle)
            {
                m_live[i] = m_live.back();
                m_live.pop_back();
                return;
            }
        }
    }

    void MeshPool::onShutdown()
    {
        if (!m_backend)
            return;
        for (const assets::MeshHandle handle : m_live)
            m_backend->destroyMeshBuffer(handle);
        m_live.clear();
        m_backend = nullptr;
    }

} // namespace manager

KENSHIN_NS_END
