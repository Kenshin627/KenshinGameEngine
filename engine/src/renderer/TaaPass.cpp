#include "kenshin/renderer/TaaPass.h"

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/manager/ShaderManager.h"

KENSHIN_NS_BEGIN

namespace renderer {

bool TaaPass::init(backend::IRenderBackend& backend,
                   manager::ShaderManager& shaders) {
  std::vector<u32> vs, fs;
  std::string log;
  if (!shaders.compileFile("taa.slang", vs, log, "vsMain") ||
      !shaders.compileFile("taa.slang", fs, log, "fsMain")) {
    KENSHIN_ERROR("[taa] shader compile failed: {}", log);
    return false;
  }
  backend::IRenderBackend::FullscreenPipelineInfo info;
  info.vsSpirv = vs.data();
  info.vsBytes = vs.size() * sizeof(u32);
  info.fsSpirv = fs.data();
  info.fsBytes = fs.size() * sizeof(u32);
  m_pipeline = backend.createTaaPipeline(info);
  if (m_pipeline == backend::IRenderBackend::kInvalidPipelineHandle) {
    KENSHIN_ERROR("[taa] pipeline creation failed");
    return false;
  }
  backend::IRenderBackend::BufferDesc desc;
  desc.size = 160;  // 2 mat4 + 2 vec4, std430-friendly
  desc.usage = backend::IRenderBackend::BufferStorage |
               backend::IRenderBackend::BufferHostVisible;
  desc.initialData = nullptr;
  desc.debugName = "taa-params";
  m_params = backend.createBuffer(desc);
  if (m_params == backend::IRenderBackend::kInvalidBufferHandle) {
    KENSHIN_ERROR("[taa] params buffer creation failed");
    return false;
  }
  return true;
}

void TaaPass::draw(backend::IRenderBackend& backend, const TaaFrame& frame,
                   u32 width, u32 height) {
  if (m_pipeline == ~0u || m_params == ~0u || width == 0 || height == 0)
    return;
  // taa.slang's TaaParams, glm raw (std430: mat4 rows at 16B).
  auto* p = static_cast<glm::vec4*>(backend.mappedPtr(m_params));
  const glm::vec4* inv = reinterpret_cast<const glm::vec4*>(&frame.invViewProj);
  const glm::vec4* prev = reinterpret_cast<const glm::vec4*>(&frame.prevViewProj);
  for (int r = 0; r < 4; ++r) p[r] = inv[r];
  for (int r = 0; r < 4; ++r) p[4 + r] = prev[r];
  p[8] = glm::vec4(frame.camPos, frame.blend);
  p[9] = glm::vec4(1.0f / static_cast<f32>(width),
                   1.0f / static_cast<f32>(height), 0.0f, 0.0f);
  const u32 handles[1] = {m_params};
  backend.drawTaa(m_pipeline, handles, 1, nullptr, 0, width, height);
}

void TaaPass::onShutdown(backend::IRenderBackend& backend) {
  if (m_params != ~0u) {
    backend.destroyBuffer(m_params);
    m_params = ~0u;
  }
  m_pipeline = ~0u;  // the pipeline dies with PipelineStore; nothing owned
}

}  // namespace renderer

KENSHIN_NS_END
