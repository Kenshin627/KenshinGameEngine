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

void MaterialDecodePass::draw(backend::IRenderBackend& backend,
                              const glm::mat4& viewProj,
                              const VirtualGeometryScene& scene,
                              const DecodeParams& params, u32 width, u32 height,
                              u32 visHandle) {
  if (m_pipeline == ~0u || width == 0 || height == 0) return;
  if (visHandle == backend::IRenderBackend::kInvalidBufferHandle) return;
  // The decode and its VisBuffer are same-frame now; the size guard
  // remains as a resize-transition backstop.
  if (backend::IRenderBackend* b = &backend;
      b->bufferSize(visHandle) != usize(width) * height * sizeof(u64))
    return;
  DecodePushC push;
  push.viewProj = viewProj;
  push.camFlags =
      glm::vec4(params.camPos, static_cast<f32>(static_cast<int>(params.view)));
  // w carries the instance count the shader clamps payload indices with.
  push.lightDir =
      glm::vec4(params.lightDir, static_cast<f32>(scene.instanceCount()));
  // w carries the max LOD level (mode 1 ramp denominator).
  push.kdScale = glm::vec4(params.kd, static_cast<f32>(scene.maxLodLevel()));
  // zw carry the table counts the shader clamps payload indices with.
  push.viewport = glm::vec4(static_cast<f32>(width), static_cast<f32>(height),
                            static_cast<f32>(scene.clusterCount()),
                            static_cast<f32>(scene.groupCount()));

  const u32 handles[8] = {
      visHandle,
      scene.clusterBuffer(),
      scene.instanceBuffer(),
      scene.positionBuffer(),
      scene.normalBuffer(),
      scene.uvBuffer(),
      scene.indexBuffer(),
      scene.groupBuffer(),  // S4.6 LOD-level view
  };
  backend.drawFullscreen(m_pipeline, handles, 8, &push, sizeof(push));
}

void MaterialDecodePass::onShutdown(backend::IRenderBackend& backend) {
  (void)backend;
  m_pipeline = ~0u;  // the pipeline dies with PipelineStore; nothing owned
}

}  // namespace renderer

KENSHIN_NS_END
