#include "kenshin/renderer/MaterialDecodePass.h"

#include <cstring>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/manager/ShaderManager.h"
#include "kenshin/renderer/VirtualGeometryScene.h"

KENSHIN_NS_BEGIN

namespace renderer {

namespace {
// The 128-byte push block - decode_material.slang's DecodePush, packed
// identically (glm raw).
struct DecodePushC {
  glm::mat4 viewProj;
  glm::vec4 camFlags;
  glm::vec4 lightDir;
  glm::vec4 kdScale;
  glm::vec4 viewport;
};
static_assert(sizeof(DecodePushC) == 128);
}  // namespace

bool MaterialDecodePass::init(backend::IRenderBackend& backend,
                              manager::ShaderManager& shaders) {
  std::vector<u32> vs, fs;
  std::string log;
  if (!shaders.compileFile("decode_material.slang", vs, log, "vsMain") ||
      !shaders.compileFile("decode_material.slang", fs, log, "fsMain")) {
    KENSHIN_ERROR("[decode] shader compile failed: {}", log);
    return false;
  }
  backend::IRenderBackend::FullscreenPipelineInfo info;
  info.vsSpirv = vs.data();
  info.vsBytes = vs.size() * sizeof(u32);
  info.fsSpirv = fs.data();
  info.fsBytes = fs.size() * sizeof(u32);
  m_pipeline = backend.createFullscreenPipeline(info);
  if (m_pipeline == backend::IRenderBackend::kInvalidPipelineHandle) {
    KENSHIN_ERROR("[decode] fullscreen pipeline creation failed");
    return false;
  }
  return true;
}

void MaterialDecodePass::updateVis(backend::IRenderBackend& backend,
                                   std::vector<u64> vis) {
  if (m_visBuffer != ~0u) backend.destroyBuffer(m_visBuffer);
  backend::IRenderBackend::BufferDesc desc;
  desc.size = vis.size() * sizeof(u64);
  desc.usage = backend::IRenderBackend::BufferStorage;
  desc.initialData = vis.data();
  desc.debugName = "visbuffer";
  m_visBuffer = backend.createBuffer(desc);
  m_visCount = vis.size();
}

void MaterialDecodePass::draw(backend::IRenderBackend& backend,
                              const glm::mat4& viewProj,
                              const VirtualGeometryScene& scene,
                              const DecodeParams& params, u32 width,
                              u32 height) {
  if (m_pipeline == ~0u || m_visBuffer == ~0u || width == 0 || height == 0)
    return;
  // The vis is produced between frames at that frame's viewport size; a
  // resize transition frame would read out of bounds - skip it (the
  // chain re-runs at the new size right after this frame).
  if (m_visCount != usize(width) * height) return;
  DecodePushC push;
  push.viewProj = viewProj;
  push.camFlags = glm::vec4(params.camPos, params.checkerboard ? 1.0f : 0.0f);
  // w carries the instance count the shader clamps payload indices with.
  push.lightDir =
      glm::vec4(params.lightDir, static_cast<f32>(scene.instanceCount()));
  push.kdScale = glm::vec4(params.kd, 8.0f);
  // zw carries the pool counts the shader clamps payload indices with.
  push.viewport = glm::vec4(static_cast<f32>(width), static_cast<f32>(height),
                            static_cast<f32>(scene.clusterCount()),
                            static_cast<f32>(scene.indexCount()));

  const u32 handles[7] = {
      m_visBuffer,
      scene.clusterBuffer(),
      scene.instanceBuffer(),
      scene.positionBuffer(),
      scene.normalBuffer(),
      scene.uvBuffer(),
      scene.indexBuffer(),
  };
  backend.drawFullscreen(m_pipeline, handles, 7, &push, sizeof(push));
}

void MaterialDecodePass::onShutdown(backend::IRenderBackend& backend) {
  if (m_visBuffer != ~0u) {
    backend.destroyBuffer(m_visBuffer);
    m_visBuffer = ~0u;
  }
  m_pipeline = ~0u;
}

}  // namespace renderer

KENSHIN_NS_END
