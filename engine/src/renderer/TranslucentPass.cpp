#include "kenshin/renderer/TranslucentPass.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/manager/ShaderManager.h"

KENSHIN_NS_BEGIN

namespace renderer {

namespace {

bool readExact(FILE* f, void* dst, usize bytes) {
  return std::fread(dst, 1, bytes, f) == bytes;
}

}  // namespace

bool TranslucentPass::init(backend::IRenderBackend& backend,
                           manager::ShaderManager& shaders,
                           const std::string& path) {
  // ---- the .tmesh (kgeo_builder's translucent side file) -------------
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) return false;
  u32 head[3] = {};
  std::vector<f32> mats, verts;
  u32 vcount = 0;
  bool ok = readExact(f, head, sizeof(head)) && head[0] == 0x314D544Bu &&
            head[1] == 2u && head[2] > 0u && head[2] <= 4096u;
  if (ok) {
    mats.resize(usize(head[2]) * 8);
    ok = readExact(f, mats.data(), mats.size() * sizeof(f32)) &&
         readExact(f, &vcount, sizeof(vcount)) && vcount > 0 &&
         vcount % 3 == 0 && vcount <= (1u << 24);
  }
  if (ok) {
    verts.resize(usize(vcount) * 12);
    ok = readExact(f, verts.data(), verts.size() * sizeof(f32));
  }
  std::fclose(f);
  if (!ok) {
    KENSHIN_ERROR("[translucent] {}: invalid .tmesh", path);
    return false;
  }
  // Clamp every material id into the table (file data drives a shader
  // index - never trust it).
  for (u32 v = 0; v < vcount; ++v) {
    f32& id = verts[usize(v) * 12 + 3];
    if (!(id >= 0.0f) || id > f32(head[2] - 1)) id = 0.0f;
  }

  // ---- pipeline --------------------------------------------------------
  std::vector<u32> vs, fs;
  std::string log;
  if (!shaders.compileFile("translucent.slang", vs, log, "vsMain") ||
      !shaders.compileFile("translucent.slang", fs, log, "fsMain")) {
    KENSHIN_ERROR("[translucent] shader compile failed: {}", log);
    return false;
  }
  backend::IRenderBackend::FullscreenPipelineInfo info;
  info.vsSpirv = vs.data();
  info.vsBytes = vs.size() * sizeof(u32);
  info.fsSpirv = fs.data();
  info.fsBytes = fs.size() * sizeof(u32);
  m_pipeline = backend.createTranslucentPipeline(info);
  if (m_pipeline == backend::IRenderBackend::kInvalidPipelineHandle) {
    KENSHIN_ERROR("[translucent] pipeline creation failed");
    m_pipeline = ~0u;
    return false;
  }

  // ---- buffers ---------------------------------------------------------
  // Shader layouts: verts = (pos.xyz, matId), (nrm.xyz, 0), (uv, 0, 0) -
  // the file's 12 floats per vertex verbatim; materials = (kd.rgb, alpha),
  // (rough, metal, 0, 0) - the file's 8 floats per record verbatim.
  backend::IRenderBackend::BufferDesc desc;
  desc.usage = backend::IRenderBackend::BufferStorage;
  desc.size = verts.size() * sizeof(f32);
  desc.initialData = verts.data();
  desc.debugName = "translucent-verts";
  m_verts = backend.createBuffer(desc);
  desc.size = mats.size() * sizeof(f32);
  desc.initialData = mats.data();
  desc.debugName = "translucent-materials";
  m_materials = backend.createBuffer(desc);
  desc.usage = backend::IRenderBackend::BufferStorage |
               backend::IRenderBackend::BufferHostVisible;
  desc.initialData = nullptr;
  desc.size = usize(kMaxInstances) * sizeof(glm::mat4);
  desc.debugName = "translucent-instances";
  m_instances = backend.createBuffer(desc);
  desc.size = usize(kMaxLights) * sizeof(GpuLightC);
  desc.debugName = "translucent-lights";
  m_lights = backend.createBuffer(desc);
  desc.size = 11 * sizeof(glm::vec4);
  desc.debugName = "translucent-env";
  m_env = backend.createBuffer(desc);
  const u32 invalid = backend::IRenderBackend::kInvalidBufferHandle;
  if (m_verts == invalid || m_materials == invalid || m_instances == invalid ||
      m_lights == invalid || m_env == invalid) {
    KENSHIN_ERROR("[translucent] buffer creation failed");
    return false;
  }
  m_vertexCount = vcount;
  KENSHIN_INFO("[translucent] {}: {} triangles, {} materials", path, vcount / 3,
               head[2]);
  return true;
}

void TranslucentPass::draw(backend::IRenderBackend& backend,
                           const TranslucentFrame& frame, u32 width,
                           u32 height) {
  if (!ready() || width == 0 || height == 0 || frame.instances.empty() ||
      frame.visBuffer == ~0u)
    return;
  const u32 instanceCount =
      static_cast<u32>(std::min<usize>(frame.instances.size(), kMaxInstances));
  std::memcpy(backend.mappedPtr(m_instances), frame.instances.data(),
              usize(instanceCount) * sizeof(glm::mat4));
  const u32 lightCount =
      static_cast<u32>(std::min<usize>(frame.lights.size(), kMaxLights));
  if (lightCount > 0)
    std::memcpy(backend.mappedPtr(m_lights), frame.lights.data(),
                usize(lightCount) * sizeof(GpuLightC));
  auto* env = static_cast<glm::vec4*>(backend.mappedPtr(m_env));
  for (u32 k = 0; k < 11; ++k) env[k] = glm::vec4(0.0f);
  if (frame.envSH)
    for (u32 k = 0; k < 9; ++k) env[2 + k] = (*frame.envSH)[k];

  struct PushC {
    glm::vec4 vp[4];
    glm::vec4 camPosCount;
    glm::vec4 viewport;
    glm::vec4 envParams;
  } push;
  for (int c = 0; c < 4; ++c) push.vp[c] = frame.viewProj[c];
  push.camPosCount = glm::vec4(frame.camPos, f32(lightCount));
  push.viewport = glm::vec4(f32(width), f32(height), frame.nearZ, frame.farZ);
  const bool sceneTable = frame.sceneMaterials != ~0u;
  push.envParams = glm::vec4(frame.envMaxLod, frame.envIntensity,
                             sceneTable ? 1.0f : 0.0f, 0.0f);

  // Per-slot snapshots of the per-frame host blocks (the LightingPass /
  // TAA lesson: the next frame's CPU write must not tear this frame's
  // in-flight read).
  // No scene table (v2 assets): bind the .tmesh records in its slot -
  // the shader gates every read on push.envParams.z.
  const u32 handles[7] = {m_verts,
                          backend.snapshotHostBuffer(m_instances),
                          m_materials,
                          backend.snapshotHostBuffer(m_lights),
                          backend.snapshotHostBuffer(m_env),
                          frame.visBuffer,
                          sceneTable ? frame.sceneMaterials : m_materials};
  backend.drawTranslucent(m_pipeline, handles, 7, frame.envTexture,
                          frame.albedoTexture, &push, sizeof(push),
                          m_vertexCount, instanceCount, width, height);
}

void TranslucentPass::onShutdown(backend::IRenderBackend& backend) {
  for (u32* h : {&m_verts, &m_materials, &m_instances, &m_lights, &m_env}) {
    if (*h != ~0u) backend.destroyBuffer(*h);
    *h = ~0u;
  }
  m_pipeline = ~0u;  // dies with PipelineStore
  m_vertexCount = 0;
}

}  // namespace renderer

KENSHIN_NS_END
