#pragma once

// ── SceneRenderer — one frame of the virtual-geometry scene ─────────
// The frame orchestration that used to live in the demo's main():
//   main VisBuffer chain (cull -> SW/HW raster, two-pass HZB occlusion)
//   -> imposter injection -> directional shadow clipmap (3 light-view
//   chains + decodes, page-cached by key + dirty tiles) -> GBuffer decode
//   (or a debug view) -> deferred lighting -> translucency -> TAA ->
//   auto exposure -> bloom -> tonemap (+ environment background).
// The caller owns the frame scope: Renderer::beginFrame / render() /
// Renderer::endFrame / afterEndFrame(). Everything the frame reads that
// a user can change is in SceneRenderSettings; what it measures comes
// back in SceneRenderStats.

#include <span>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "kenshin/Common.h"
#include "kenshin/renderer/BloomPass.h"
#include "kenshin/renderer/ClusterCullPass.h"
#include "kenshin/renderer/Environment.h"
#include "kenshin/renderer/ExposurePass.h"
#include "kenshin/renderer/ImposterPass.h"
#include "kenshin/renderer/InstanceCullPass.h"
#include "kenshin/renderer/LightingPass.h"
#include "kenshin/renderer/MaterialDecodePass.h"
#include "kenshin/renderer/TaaPass.h"
#include "kenshin/renderer/TonemapPass.h"
#include "kenshin/renderer/TranslucentPass.h"
#include "kenshin/renderer/VisBufferChain.h"

KENSHIN_NS_BEGIN

namespace backend {
class IRenderBackend;
}
namespace manager {
class ShaderManager;
}
namespace scene {
class Camera;
}

namespace renderer {

class VirtualGeometryScene;
class StreamingManager;

/// Every frame knob (panel / CLI / game settings), already resolved.
struct SceneRenderSettings {
  int viewMode = 0;  // DecodeView: 0 Final (deferred), 1..6 debug views
  // ---- geometry chain
  f32 pixelThreshold = 1.0f;  // LOD screen-space error (px)
  f32 hwThresholdPx = 1.0f;   // SW/HW routing edge threshold
  u32 hwMode = 1;             // main chain: 0 SW, 1 routed, 2 all HW
  u32 shadowHwMode = 1;       // shadow chains (same encoding)
  u32 cullMode = 1;           // 0 linear, 1 level-wise BVH, 2 persistent
  u32 rasterMode = 1;         // SW raster: 0 baseline walk, 1 row-span
  bool rasterProbe = false;   // coverage histogram ([rasthist])
  bool hwPixelStats = false;  // count HW pixel writes (costly atomic)
  bool occlusion = true;
  bool imposters = false;     // route tiny extra-mesh instances to imposters
  // ---- lighting
  glm::vec3 lightDir{0.35f, 0.8f, 0.45f};  // directional, TO light
  f32 ambient = 0.15f;                     // hemisphere fallback (no probe)
  glm::vec3 skyColor{0.6f};
  glm::vec3 groundColor{0.3f};
  f32 envIntensity = 0.7f;
  f32 emissiveStrength = 1.0f;
  bool emissivePassThrough = true;
  // ---- shadows
  bool shadows = true;
  int shadowLevels = 3;  // clipmap levels (1..3)
  glm::vec3 shadowCenter{0.0f};
  // ---- post
  bool taa = true;
  f32 taaSharpen = 0.3f;
  f32 exposure = 1.5f;  // manual exposure (auto off) / TAA weight fallback
  bool autoExposure = true;
  ExposureSettings exposureSettings;
  bool exposureForceTarget = false;  // jump to the target (headless runs)
  bool bloom = true;
  f32 bloomStrength = 0.08f;
  bool sky = true;  // environment behind the model
  f32 skyBlur = 0.0f;
  int tonemapMode = 2;  // TonemapPass::Operator
  glm::vec4 clearColor{0.0f, 0.0f, 0.0f, 1.0f};
};

/// What the last frame measured.
struct SceneRenderStats {
  /// True when the chain counters below were refreshed this frame (the
  /// non-blocking mapped snapshot can be unavailable).
  bool chainStatsValid = false;
  u32 workItems = 0, visibleClusters = 0, tooCoarse = 0, srcTooCoarse = 0;
  u32 pxWrites = 0, bfCulls = 0, nearClips = 0, occCulled = 0, revived = 0;
  u32 cutTris = 0, hwClusters = 0, visitedNodes = 0, prunedNodes = 0;
  u32 pendingSkips = 0, missPages = 0;
  f32 coveragePct = 0.0f;
  bool exposureValid = false;
  f32 autoScale = 0.0f, autoTargetScale = 0.0f, autoAvgLum = 0.0f;
};

/// The scene content the renderer draws (built once by the caller).
struct SceneRendererDesc {
  VirtualGeometryScene* scene = nullptr;
  StreamingManager* streamer = nullptr;  // v3 paged assets, else null
  std::vector<CullGpuInstance> instances;
  std::vector<GpuMeshRangePair> meshRanges;
  /// Per instance: the world bounds before any parking (imposter sizing).
  std::vector<glm::vec4> instanceBounds;
  /// Per mesh: imposter atlas planes (empty = none) + its single cluster.
  std::vector<std::vector<u32>> imposterDepth, imposterTri;
  std::vector<u32> imposterRootCluster;
  DecodeParams decodeParams;  // base (kd etc.); camera/view set per frame
  std::string translucentMeshPath;  // the .tmesh side file, empty = none
};

class SceneRenderer {
 public:
  bool setup(backend::IRenderBackend& backend,
             manager::ShaderManager& shaders, SceneRendererDesc desc);
  void setEnvironment(const EnvironmentProbe& probe) { m_env = probe; }
  const EnvironmentProbe& environment() const { return m_env; }

  struct FrameInputs {
    const scene::Camera* camera = nullptr;
    f32 fovYDeg = 45.0f;
    u32 width = 0, height = 0;  // viewport pixels
    u32 frameIndex = 0;         // TAA jitter phase + history continuity
    f32 dt = 0.0f;              // seconds (auto exposure adaptation)
    /// The light table (directional first: the shadow-casting key).
    std::span<const GpuLightC> lights;
  };
  /// Record the scene into the current frame (between beginFrame and
  /// endFrame). No-op until the backend can create the pipelines.
  void render(const FrameInputs& in, const SceneRenderSettings& s);
  /// After Renderer::endFrame: the streaming pump (touched pages, LRU,
  /// finished IO -> uploads).
  void afterEndFrame();
  /// The periodic diagnostics ([vgstats] [chainms] [rasthist]
  /// [shadowstats]).
  void logStats();
  void shutdown();

  const SceneRenderStats& stats() const { return m_stats; }
  bool chainReady() const { return m_chainReady; }

 private:
  void lazyInit();
  void renderShadows(const SceneRenderSettings& s, const ChainInputs& ci,
                     const scene::Camera& camera,
                     LightingPass::ShadowLevel (&levels)[3], u32& levelCount,
                     glm::vec3& origin, bool& shadowsOn);
  void injectImposters(const ChainInputs& ci, const scene::Camera& camera,
                       const glm::mat4& viewProjJ);

  backend::IRenderBackend* m_backend = nullptr;
  manager::ShaderManager* m_shaders = nullptr;
  SceneRendererDesc m_desc;
  EnvironmentProbe m_env;
  SceneRenderStats m_stats;

  VisBufferChain m_chain;
  VisBufferChain m_shadowChains[3];  // clipmap levels (extent 12/24/48 m)
  MaterialDecodePass m_decode;
  LightingPass m_lighting;
  TranslucentPass m_translucent;
  ImposterPass m_imposter;
  TaaPass m_taa;
  ExposurePass m_exposure;
  BloomPass m_bloom;
  TonemapPass m_tonemap;
  bool m_chainReady = false, m_shadowChainReady = false;
  bool m_decodeReady = false, m_lightingReady = false;
  bool m_tonemapReady = false, m_taaReady = false;
  bool m_bloomTried = false, m_exposureTried = false;
  bool m_translucentTried = false, m_imposterReady = false;
  std::vector<u32> m_meshAtlasBase;  // per mesh -> imposter atlas word

  // TAA history continuity: valid only if the resolve ran on the
  // previous rendered frame (toggles / debug views leave it stale).
  glm::mat4 m_taaPrevViewProj{1.0f};  // UNJITTERED
  bool m_taaPrevValid = false;
  int m_taaWarmup = 0;
  bool m_taaRanLastFrame = false;
  u32 m_lastRenderedFrame = 0xFFFFFFFFu;

  // Shadow content cache (UE FClipmapCacheKey analogue) + Stage C tiles.
  struct ShadowCacheKey {
    glm::vec3 L{0.0f, 0.0f, -2.0f};  // impossible direction = dirty
    f32 threshold = -1.0f;
    u32 levels = 0, residentPages = 0xFFFFFFFFu;
    u32 cullMode = 0xFFFFFFFFu, hwMode = 0xFFFFFFFFu;
    bool valid = false;
  } m_shadowKey;
  u32 m_shadowDirtyTiles[3] = {0u, 0u, 0u};
  u32 m_shadowChainStats[3][16] = {};

  u32 m_probeHist[8] = {};
  bool m_probeValid = false;
};

}  // namespace renderer

KENSHIN_NS_END
