#include "kenshin/renderer/SceneRenderer.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/geometry/PagePacker.h"
#include "kenshin/manager/ShaderManager.h"
#include "kenshin/renderer/StreamingManager.h"
#include "kenshin/renderer/VirtualGeometryScene.h"
#include "kenshin/scene/Camera.h"

KENSHIN_NS_BEGIN

namespace renderer {

namespace {

/// TAA jitter: Halton(2,3) - the standard low-discrepancy 8-phase
/// subpixel sequence.
glm::vec2 halton23(int i) {
  auto radical = [](int a, int base) {
    float r = 0.0f, f = 1.0f;
    while (a > 0) {
      f /= float(base);
      r += f * float(a % base);
      a /= base;
    }
    return r;
  };
  return {radical(i + 1, 2), radical(i + 1, 3)};
}

/// Stage C: which 512px tiles of a 2048^2 clipmap level the main view's
/// frustum touches. UE marks pages from the receiver mask (GBuffer
/// depth); this CPU v1 projects the frustum corners into the level's
/// light clip, clamps to the box, takes the convex hull, and tests the
/// 4x4 tile centers against it (inflated by half a tile so partial
/// overlaps count) - conservative, zero plumbing.
u32 shadowTileMask(const glm::mat4& lightVP, const glm::mat4& invViewProj) {
  glm::vec2 pts[8];
  for (int i = 0; i < 8; ++i) {
    const glm::vec4 ndc((i & 1) ? 1.0f : -1.0f, (i & 2) ? 1.0f : -1.0f,
                        (i & 4) ? 1.0f : 0.0f, 1.0f);
    const glm::vec4 clip = lightVP * (invViewProj * ndc);
    pts[i] = glm::clamp(glm::vec2(clip.x, clip.y) * 0.5f + 0.5f,
                        glm::vec2(0.0f), glm::vec2(1.0f));
  }
  // Convex hull (monotone chain, CCW).
  std::sort(pts, pts + 8, [](const glm::vec2& a, const glm::vec2& b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
  });
  const auto cross = [](const glm::vec2& o, const glm::vec2& a,
                        const glm::vec2& b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  };
  glm::vec2 hull[8];
  int n = 0;
  for (int i = 0; i < 8; ++i) {
    while (n >= 2 && cross(hull[n - 2], hull[n - 1], pts[i]) <= 0.0f) --n;
    hull[n++] = pts[i];
  }
  const int lower = n + 1;
  for (int i = 6; i >= 0; --i) {
    while (n >= lower && cross(hull[n - 2], hull[n - 1], pts[i]) <= 0.0f) --n;
    hull[n++] = pts[i];
  }
  if (n > 0) --n;  // the last point repeats the first
  u32 mask = 0;
  if (n < 3) {
    // Degenerate projection: fall back to the point bbox.
    glm::vec2 mn(1.0f), mx(0.0f);
    for (const auto& p : pts) {
      mn = glm::min(mn, p);
      mx = glm::max(mx, p);
    }
    if (mn.x > mx.x) return 0;
    for (u32 ty = u32(mn.y * 4.0f); ty <= u32(mx.y * 4.0f) && ty < 4; ++ty)
      for (u32 tx = u32(mn.x * 4.0f); tx <= u32(mx.x * 4.0f) && tx < 4; ++tx)
        mask |= 1u << (ty * 4u + tx);
    return mask;
  }
  // Tile centers vs the hull; the -0.18 margin (half a tile diagonal in
  // cross units) admits centers whose TILE still overlaps the hull.
  for (u32 ty = 0; ty < 4; ++ty)
    for (u32 tx = 0; tx < 4; ++tx) {
      const glm::vec2 c((tx + 0.5f) * 0.25f, (ty + 0.5f) * 0.25f);
      bool inside = true;
      for (int e = 0; e < n && inside; ++e)
        inside = cross(hull[e], hull[(e + 1) % n], c) >= -0.18f;
      if (inside) mask |= 1u << (ty * 4u + tx);
    }
  return mask;
}

DecodePagedInputs pagedDecodeInputs(const VirtualGeometryScene& scene) {
  DecodePagedInputs pd{};
  pd.slotPool = scene.pageSlotPool();
  pd.clusterPage = scene.clusterPageBuffer();
  pd.pageTable = scene.pageTableBuffer();
  pd.pageHeader = scene.pageHeaderBuffer();
  pd.slotU32Total = scene.pageSlotCount() * (geometry::kPageSlotSize / 4);
  pd.clusterMeta = scene.clusterMetaBuffer();
  return pd;
}

}  // namespace

bool SceneRenderer::setup(backend::IRenderBackend& backend,
                          manager::ShaderManager& shaders,
                          SceneRendererDesc desc) {
  if (desc.scene == nullptr) {
    KENSHIN_ERROR("[scene-renderer] setup without a VirtualGeometryScene");
    return false;
  }
  m_backend = &backend;
  m_shaders = &shaders;
  m_desc = std::move(desc);
  return true;
}

void SceneRenderer::lazyInit() {
  backend::IRenderBackend& be = *m_backend;
  VirtualGeometryScene& scene = *m_desc.scene;
  // The chain (pools ready; vis buffers start 1x1, the first execute
  // resizes).
  if (!m_chainReady) {
    std::vector<u32> meshHierRoots;
    for (const auto& mesh : scene.meshes())
      meshHierRoots.push_back(mesh.hierarchyRoot);
    // Paged assets feed the slot pool + tables; the v2 pools stay empty
    // (the chain binds the pool at 1/2 itself).
    ChainPagedInputs pagedInputs;
    ChainPagedInputs* pagedPtr = nullptr;
    if (scene.paged()) {
      pagedInputs.slotPool = scene.pageSlotPool();
      pagedInputs.clusterPage = scene.clusterPageBuffer();
      pagedInputs.pageTable = scene.pageTableBuffer();
      pagedInputs.pageHeader = scene.pageHeaderBuffer();
      pagedInputs.pageCount = scene.pageCount();
      pagedInputs.slotU32Total =
          scene.pageSlotCount() * (geometry::kPageSlotSize / 4);
      pagedInputs.clusterMeta = scene.clusterMetaBuffer();
      pagedPtr = &pagedInputs;
    }
    m_chainReady = m_chain.init(
        be, *m_shaders, scene.positions4(), scene.indices(),
        scene.clusterData(), scene.groupData(), scene.hierarchyData(),
        scene.groupRanges(), meshHierRoots, m_desc.instances,
        m_desc.meshRanges,
        // Pair-list capacity: min(the theoretical worst case, 4M pairs) -
        // the cull writers saturate and the consumers clamp at this cap.
        u32(std::min<u64>(u64(m_desc.instances.size()) * scene.clusterCount(),
                          u64(4) << 20)),
        scene.clusterAabbs(), pagedPtr, 1, 1);
    // Three clipmap levels (extent doubles per level; each chain the
    // same 2048 buffer - LOD rides worldPerPixel). Once-only guard.
    if (!m_shadowChainReady) {
      m_shadowChainReady = true;
      // Receiver-only instances (castShadows == 0, the ground) get a ZERO
      // transform in the shadow chains: degenerate bounds frustum-cull
      // them in EVERY cull mode, the index space stays aligned with the
      // scene's decode buffer, and no pairs exist for the decode to
      // touch. A ground plane in its own map self-receives at minified
      // sampling and occludes nothing above it in a directional light.
      std::vector<CullGpuInstance> shadowInsts = m_desc.instances;
      for (auto& si2 : shadowInsts)
        if (si2.pad[1] == 0u) {
          // Park the bounds sphere FAR outside every frustum: the
          // zero-transform alone leaves the instance at the origin
          // (inside the light box) and its CLUSTER bounds still pass the
          // cluster-level tests.
          si2.model = glm::mat4(0.0f);
          si2.bounds = glm::vec4(1e9f, 1e9f, 1e9f, 0.0f);
        }
      for (int lv = 0; lv < 3; ++lv)
        m_shadowChainReady =
            m_shadowChains[lv].init(
                be, *m_shaders, scene.positions4(), scene.indices(),
                scene.clusterData(), scene.groupData(), scene.hierarchyData(),
                scene.groupRanges(), meshHierRoots, shadowInsts,
                m_desc.meshRanges, m_chain.maxPairs(), scene.clusterAabbs(),
                pagedPtr, 2048, 2048) &&
            m_shadowChainReady;
    }
    // The timing pool is per-slot shared with the main chain - the shadow
    // chains must not touch it (see setTimingEnabled).
    for (int lv = 0; lv < 3; ++lv) m_shadowChains[lv].setTimingEnabled(false);
    if (m_chainReady)
      KENSHIN_INFO("[vgchain] in-frame chain ready{}",
                   pagedPtr ? " (paged)" : "");
  }
  if (!m_decodeReady) m_decodeReady = m_decode.init(be, *m_shaders);
  if (!m_lightingReady) m_lightingReady = m_lighting.init(be, *m_shaders);
  if (!m_tonemapReady) m_tonemapReady = m_tonemap.init(be, *m_shaders);
  if (!m_taaReady) m_taaReady = m_taa.init(be, *m_shaders);
  if (!m_bloomTried) {
    m_bloomTried = true;
    m_bloom.init(be, *m_shaders);
  }
  if (!m_exposureTried) {
    m_exposureTried = true;
    m_exposure.init(be, *m_shaders);
  }
}

void SceneRenderer::injectImposters(const ChainInputs& ci,
                                    const scene::Camera& camera,
                                    const glm::mat4& viewProjJ) {
  backend::IRenderBackend& be = *m_backend;
  if (!m_imposterReady) {
    m_imposterReady = m_imposter.init(be, *m_shaders);
    m_meshAtlasBase.assign(m_desc.scene->meshes().size(),
                           ImposterPass::kNoAtlas);
    for (u32 m = 0; m < m_desc.imposterDepth.size(); ++m) {
      if (m_desc.imposterDepth[m].empty()) continue;
      const u32 base = m_imposter.uploadAtlas(
          be, m_desc.imposterDepth[m].data(), m_desc.imposterTri[m].data(),
          u32(m_desc.imposterDepth[m].size()));
      m_meshAtlasBase[m] = base;
      KENSHIN_INFO("[imposter] mesh {}: atlas @{} (cluster {})", m, base,
                   m < m_desc.imposterRootCluster.size()
                       ? m_desc.imposterRootCluster[m]
                       : 0xFFFFFFFFu);
    }
  }
  if (!m_imposterReady) return;
  ImposterQuad quads[64];
  glm::uvec2 pairs[256];
  u32 quadCount = 0, pairCount = 0;
  const auto& instances = m_desc.instances;
  const auto& realBounds = m_desc.instanceBounds;
  for (u32 ii = 0; ii < instances.size() && quadCount < 64 && pairCount < 255;
       ++ii) {
    const auto& inst = instances[ii];
    if (inst.meshId >= m_meshAtlasBase.size()) continue;
    if (m_meshAtlasBase[inst.meshId] == ImposterPass::kNoAtlas) continue;
    // Mirror the cull's tiny-skip EXACTLY.
    const float thresholdPx = ci.imposterThresholdPx;
    if (thresholdPx <= 0.0f || inst.pad[1] != 2u) continue;
    if (ii >= realBounds.size()) continue;
    const glm::vec3 c(realBounds[ii]);
    const float r = realBounds[ii].w;
    const float d = glm::max(glm::distance(c, camera.position()), 1e-3f);
    if (r / d / ci.worldPerPixel >= thresholdPx) continue;
    const glm::vec4 clip = viewProjJ * glm::vec4(c, 1.0f);
    if (clip.w <= 0.0f) continue;
    const glm::vec2 ndc = glm::vec2(clip.x, clip.y) / clip.w;
    const float cx = (ndc.x * 0.5f + 0.5f) * float(ci.width);
    const float cy = (ndc.y * 0.5f + 0.5f) * float(ci.height);
    const int R = std::max(6, int(r / d / ci.worldPerPixel));
    if (R > int(ci.width)) continue;
    auto& q = quads[quadCount];
    q.rect = glm::ivec4(int(cx) - R, int(cy) - R, int(cx) + R, int(cy) + R);
    q.centerAtlas = glm::vec4(c, float(m_meshAtlasBase[inst.meshId]));
    q.halfPair = glm::vec4(r, r, r, float(pairCount));
    q.pad = glm::vec4(0.0f);
    ++quadCount;
    pairs[pairCount++] = glm::uvec2(
        ii, inst.meshId < m_desc.imposterRootCluster.size()
                ? m_desc.imposterRootCluster[inst.meshId]
                : 0xFFFFFFFFu);
  }
  if (quadCount > 0) {
    m_imposter.setPairs(be, pairs, pairCount);
    // The JITTERED matrix: the injected depth competes in the same
    // VisBuffer atomic as the (jittered) chain raster.
    m_imposter.draw(be, m_chain.visBufferCurrent(be), viewProjJ,
                    camera.position(), ci.width, ci.height, quads, quadCount);
  }
}

void SceneRenderer::renderShadows(const SceneRenderSettings& s,
                                  const ChainInputs& ci,
                                  const scene::Camera& camera,
                                  LightingPass::ShadowLevel (&levels)[3],
                                  u32& levelCount, glm::vec3& origin,
                                  bool& shadowsOn) {
  backend::IRenderBackend& be = *m_backend;
  VirtualGeometryScene& scene = *m_desc.scene;
  // The clipmap rig - three levels, extent 12/24/48m, each a light-view
  // chain + decode. worldPerPixel scales with extent (UE's texel-size
  // LOD calibration); sampling picks the level by log2 distance with
  // coarse fallback.
  shadowsOn = s.shadows && m_shadowChainReady && m_decodeReady;
  const bool shadowPhase = s.viewMode == 0 && m_lightingReady &&
                           m_shadowChainReady && m_decodeReady;
  DecodePagedInputs spd{};
  if (scene.paged()) spd = pagedDecodeInputs(scene);
  levelCount = 0;
  origin = glm::vec3(0.0f);
  if (shadowsOn) {
    const glm::vec3 L = glm::normalize(s.lightDir);
    const glm::vec3 center = s.shadowCenter;
    constexpr float kExtent0 = 12.0f;
    constexpr float kDepth = 40.0f;
    // Z-up world: the light view's up rides the Z axis (a horizontal up
    // sits near-parallel to L for low lights).
    const glm::vec3 up =
        std::abs(L.z) > 0.95f ? glm::vec3(1, 0, 0) : glm::vec3(0, 0, 1);
    levelCount = u32(s.shadowLevels);
    origin = center;
    // UE FClipmapCacheKey: light direction + level count, plus our
    // primitive-visibility analogues (streaming residency) and the LOD
    // threshold. A key change dirties every TILE of every level; each
    // frame then renders only dirty tiles the main view needs.
    const u32 residentNow =
        m_desc.streamer ? m_desc.streamer->residentPages() : 0xFFFFFFFFu;
    const bool keyChanged =
        !m_shadowKey.valid || glm::distance(L, m_shadowKey.L) > 1e-4f ||
        s.pixelThreshold != m_shadowKey.threshold ||
        levelCount != m_shadowKey.levels ||
        residentNow != m_shadowKey.residentPages ||
        ci.cullMode != m_shadowKey.cullMode || ci.hwMode != m_shadowKey.hwMode;
    if (keyChanged) {
      m_shadowDirtyTiles[0] = m_shadowDirtyTiles[1] = m_shadowDirtyTiles[2] =
          0xFFFFu;
      m_shadowKey.L = L;
      m_shadowKey.threshold = s.pixelThreshold;
      m_shadowKey.levels = levelCount;
      m_shadowKey.residentPages = residentNow;
      m_shadowKey.cullMode = ci.cullMode;
      m_shadowKey.hwMode = ci.hwMode;
      m_shadowKey.valid = true;
    }
    const glm::mat4 invMainVP = glm::inverse(camera.viewProjMatrix());
    for (u32 lv = 0; lv < levelCount; ++lv) {
      const float extent = kExtent0 * float(1 << lv);
      // The light eye rides the LIGHT'S side of the scene (+L): from the
      // far side the single-sided ground is backface-culled out of the
      // map AND every occluder above it is farther than the ground.
      const glm::mat4 lightView =
          glm::lookAtRH(center + L * (kDepth * 0.5f), center, up);
      glm::mat4 lightProj =
          glm::orthoRH_ZO(-extent, extent, -extent, extent, 0.1f, kDepth);
      lightProj[1][1] *= -1.0f;  // the engine's Y-flip convention
      levels[lv].viewProj = lightProj * lightView;
      levels[lv].texelWorld = (2.0f * extent) / 2048.0f;

      // Stage C: dirty tiles the main view needs. The bounding rect of
      // the worked tiles scissors BOTH the chain raster and the decode
      // (LOAD-op) - the rest of the map keeps its cached content.
      const u32 work = m_shadowDirtyTiles[lv] &
                       shadowTileMask(levels[lv].viewProj, invMainVP);
      if (work == 0u) continue;
      u32 tx0 = 4, ty0 = 4, tx1 = 0, ty1 = 0;
      for (u32 ty = 0; ty < 4; ++ty)
        for (u32 tx = 0; tx < 4; ++tx)
          if (work & (1u << (ty * 4u + tx))) {
            tx0 = std::min(tx0, tx);
            ty0 = std::min(ty0, ty);
            tx1 = std::max(tx1, tx);
            ty1 = std::max(ty1, ty);
          }
      constexpr u32 kTilePx = 2048 / 4;
      const u32 rect[4] = {tx0 * kTilePx, ty0 * kTilePx,
                           tx1 * kTilePx + kTilePx - 1,
                           ty1 * kTilePx + kTilePx - 1};

      ChainInputs si{};
      si.view = lightView;
      si.proj = lightProj;
      si.nearZ = 0.1f;
      // The EYE, not the look-at center: the routing guard and the LOD
      // distances measure from camPos.
      si.camPos = center + L * (kDepth * 0.5f);
      si.scissorRect[0] = rect[0];
      si.scissorRect[1] = rect[1];
      si.scissorRect[2] = rect[2];
      si.scissorRect[3] = rect[3];
      si.pixelThreshold = s.pixelThreshold * 2.0f;  // D11.5 shadow LOD bias
      si.worldPerPixel = (2.0f * extent) / 2048.0f;
      si.width = 2048;
      si.height = 2048;
      extractFrustumPlanes(levels[lv].viewProj, si.planes);
      si.occlusionEnabled = false;  // HZB is main-view; shadows skip
      // UE-faithful routing (NaniteCullRaster HardwareThenSoftware for
      // shadow maps): giant occluder triangles go to HW.
      si.hwMode = s.shadowHwMode;
      si.cullNonCasters = 1u;  // belt: the instances are parked too
      si.rasterMode = 1;
      m_shadowChains[lv].execute(be, si);

      DecodeParams sp{};
      sp.camPos = center;
      sp.lightDir = s.lightDir;
      sp.view = DecodeView::Final;
      if (scene.paged()) {
        m_decode.drawShadowPaged(
            be, levels[lv].viewProj, scene, sp, 2048, 2048,
            m_shadowChains[lv].visBufferCurrent(be), spd,
            m_shadowChains[lv].pairsCurrent(be),
            m_shadowChains[lv].hwPairsCurrent(be),
            m_shadowChains[lv].postPairsCurrent(be),
            m_shadowChains[lv].maxPairs(), lv, true, rect);
      } else {
        m_decode.drawShadow(be, levels[lv].viewProj, scene, sp, 2048, 2048,
                            m_shadowChains[lv].visBufferCurrent(be),
                            m_shadowChains[lv].pairsCurrent(be),
                            m_shadowChains[lv].hwPairsCurrent(be),
                            m_shadowChains[lv].postPairsCurrent(be),
                            m_shadowChains[lv].maxPairs(), lv, true, rect);
      }
      m_shadowDirtyTiles[lv] &= ~work;
    }
  } else if (shadowPhase) {
    // Disabled this frame: still touch a target so lighting's sampling
    // reads a fully-lit map (clear-to-one path). The clear destroys the
    // cached content - re-render on toggle-on.
    m_shadowKey.valid = false;
    m_decode.drawShadow(be, glm::mat4(1.0f), scene, {}, 16, 16, 0, 0, 0, 0, 0,
                        0, false);
  }
}

void SceneRenderer::render(const FrameInputs& in,
                           const SceneRenderSettings& s) {
  if (m_backend == nullptr || in.camera == nullptr || in.width <= 1 ||
      in.height <= 1)
    return;
  backend::IRenderBackend& be = *m_backend;
  VirtualGeometryScene& scene = *m_desc.scene;
  const scene::Camera& camera = *in.camera;
  m_stats.chainStatsValid = false;
  // The history is only valid if the resolve ran on the PREVIOUS
  // rendered frame.
  const bool taaHistoryLive = m_taaRanLastFrame &&
                              m_lastRenderedFrame != 0xFFFFFFFFu &&
                              m_lastRenderedFrame + 1 == in.frameIndex;
  bool taaRanThisFrame = false;

  lazyInit();
  if (m_chainReady && m_decodeReady) {
    // Same camera, same frame: the chain's inputs and the decode's
    // matrices cannot disagree.
    ChainInputs ci;
    ci.view = camera.viewMatrix();
    // TAA: subpixel NDC jitter (8-phase Halton(2,3)) rides the
    // projection's [2][0]/[2][1]; the resolve unprojects with the
    // UNJITTERED pair. Jitter ONLY when this frame's resolve will undo it
    // (the Final view): debug views decode straight to the screen.
    const bool taaOn =
        s.taa && s.viewMode == 0 && m_taaReady && m_lightingReady;
    glm::mat4 projJ = camera.projMatrix();
    glm::vec2 taaJitterUV(0.0f);
    if (taaOn) {
      const glm::vec2 h = halton23(int(in.frameIndex) & 7);
      const glm::vec2 jn((h.x * 2.0f - 1.0f) / f32(in.width),
                         (h.y * 2.0f - 1.0f) / f32(in.height));
      projJ[2][0] += jn.x;
      projJ[2][1] += jn.y;
      taaJitterUV = jn * 0.5f;  // NDC -> UV for the resolve
    }
    const glm::mat4 viewProjJ = projJ * camera.viewMatrix();
    ci.proj = projJ;
    ci.nearZ = camera.nearPlane();
    ci.camPos = camera.position();
    ci.pixelThreshold = s.pixelThreshold;
    ci.worldPerPixel = 2.0f * std::tan(glm::radians(in.fovYDeg) * 0.5f) /
                       static_cast<f32>(in.height);
    ci.width = in.width;
    ci.height = in.height;
    extractFrustumPlanes(viewProjJ, ci.planes);
    ci.imposterThresholdPx = s.imposters ? 4096.0f : 0.0f;
    ci.occlusionEnabled = s.occlusion;
    ci.hwMode = s.hwMode;
    ci.hwThresholdPx = s.hwThresholdPx;
    ci.cullMode = s.cullMode;
    ci.rasterMode = s.rasterMode;
    ci.rasterProbe = s.rasterProbe ? 1u : 0u;
    ci.hwPixelStats = s.hwPixelStats;
    // Stats: NON-BLOCKING mapped snapshot taken BEFORE this frame's
    // execute - the current slot still holds the frame-before-last's
    // final values, which beginFrame's pacing wait proved complete
    // (reading after execute raced the GPU: torn counters).
    m_probeValid = s.rasterProbe && m_chain.readRasterProbe(be, m_probeHist);
    {
      u32 cs[16];
      if (m_chain.readStatsMapped(be, cs)) {
        m_stats.visibleClusters = cs[2];
        m_stats.cutTris = cs[11];
        m_stats.pxWrites = cs[5] + cs[10];  // main + post raster
        m_stats.occCulled = cs[8];
        m_stats.revived = cs[9];
        m_stats.hwClusters = cs[12];
        m_stats.bfCulls = cs[6];
        m_stats.nearClips = cs[7];
        m_stats.workItems = cs[1];
        m_stats.tooCoarse = cs[3];
        m_stats.srcTooCoarse = cs[4];
        m_stats.visitedNodes = cs[13];
        m_stats.prunedNodes = cs[14];
        m_stats.pendingSkips = cs[15];
        m_stats.coveragePct = 100.0f * static_cast<f32>(cs[5] + cs[10]) /
                              (ci.width * ci.height);
        m_stats.chainStatsValid = true;
      }
    }

    m_chain.execute(be, ci);
    injectImposters(ci, camera, viewProjJ);

    LightingPass::ShadowLevel shadowLevels[3];
    u32 shadowLevelCount = 0;
    glm::vec3 shadowOrigin(0.0f);
    bool shadowsOn = false;
    renderShadows(s, ci, camera, shadowLevels, shadowLevelCount, shadowOrigin,
                  shadowsOn);

    DecodeParams decodeParams = m_desc.decodeParams;
    decodeParams.camPos = camera.position();
    decodeParams.view = static_cast<DecodeView>(s.viewMode);
    // View mode 0 (Final) runs the deferred pair - the decode writes the
    // GBuffer, the lighting pass shades it. Debug modes (1..6) keep the
    // direct decode (no GBuffer involvement).
    const bool finalView = s.viewMode == 0;
    if (scene.paged()) {
      const DecodePagedInputs pd = pagedDecodeInputs(scene);
      if (finalView) {
        m_decode.drawGBufferPaged(
            be, viewProjJ, scene, decodeParams, ci.width, ci.height,
            m_chain.visBufferCurrent(be), pd, m_chain.pairsCurrent(be),
            m_chain.hwPairsCurrent(be), m_chain.postPairsCurrent(be),
            m_chain.maxPairs(), m_desc.albedoTexture,
            m_imposterReady ? m_imposter.pairsBuffer() : 0);
      } else {
        m_decode.drawPaged(be, viewProjJ, scene, decodeParams, ci.width,
                           ci.height, m_chain.visBufferCurrent(be), pd,
                           m_chain.pairsCurrent(be),
                           m_chain.hwPairsCurrent(be),
                           m_chain.postPairsCurrent(be), m_chain.maxPairs());
      }
    } else {
      if (finalView) {
        m_decode.drawGBuffer(be, viewProjJ, scene, decodeParams, ci.width,
                             ci.height, m_chain.visBufferCurrent(be),
                             m_chain.pairsCurrent(be),
                             m_chain.hwPairsCurrent(be),
                             m_chain.postPairsCurrent(be), m_chain.maxPairs(),
                             m_desc.albedoTexture);
      } else {
        m_decode.draw(be, viewProjJ, scene, decodeParams, ci.width, ci.height,
                      m_chain.visBufferCurrent(be), m_chain.pairsCurrent(be),
                      m_chain.hwPairsCurrent(be),
                      m_chain.postPairsCurrent(be), m_chain.maxPairs());
      }
    }
    if (finalView && m_lightingReady) {
      const auto* envSH = m_env.shValid ? &m_env.sh : nullptr;
      m_lighting.draw(
          be, glm::inverse(viewProjJ), camera.position(), in.lights,
          scene.paged() ? scene.materialTableBuffer()
                        : backend::IRenderBackend::kInvalidBufferHandle,
          scene.paged() ? scene.materialCount() : 0u, s.clearColor, s.ambient,
          s.skyColor, s.groundColor, shadowLevels,
          shadowsOn && s.viewMode == 0 ? shadowLevelCount : 0u, 12.0f,
          shadowOrigin, ci.width, ci.height, m_env.texture, m_env.maxLod,
          envSH, s.envIntensity, s.emissiveStrength);
      // Translucency (BLEND materials) - after the opaque lighting, BEFORE
      // the TAA resolve, with the same jittered matrix.
      if (!m_translucentTried) {
        m_translucentTried = true;
        if (!m_desc.translucentMeshPath.empty() &&
            std::filesystem::exists(m_desc.translucentMeshPath))
          m_translucent.init(be, *m_shaders, m_desc.translucentMeshPath);
      }
      if (m_translucent.ready()) {
        // The primary asset's instances (mesh 0) carry the .tmesh.
        std::vector<glm::mat4> tInst;
        for (const auto& inst : m_desc.instances)
          if (inst.meshId == 0u) tInst.push_back(inst.model);
        TranslucentFrame tfr;
        tfr.viewProj = viewProjJ;
        tfr.camPos = camera.position();
        tfr.nearZ = camera.nearPlane();
        tfr.farZ = camera.farPlane();
        tfr.lights = in.lights;
        tfr.envSH = envSH;
        tfr.envIntensity = s.envIntensity;
        tfr.emissiveStrength = s.emissiveStrength;
        tfr.emissivePassThrough = s.emissivePassThrough;
        tfr.envTexture = m_env.texture;
        tfr.envMaxLod = m_env.maxLod;
        tfr.visBuffer = m_chain.visBufferCurrent(be);
        if (scene.paged()) {
          tfr.sceneMaterials = scene.materialTableBuffer();
          tfr.albedoTexture = m_desc.albedoTexture;
        }
        tfr.instances = tInst;
        m_translucent.draw(be, tfr, ci.width, ci.height);
      }
      // TAA resolve (skip the first frames after (re)validation: the
      // history is garbage until the ping-pong fills).
      bool resolvedByTaa = false;
      if (m_taaReady && taaOn) {
        if (!taaHistoryLive) m_taaWarmup = 0;  // stale history: reseed
        TaaFrame tf;
        tf.invViewProj = glm::inverse(camera.viewProjMatrix());
        tf.prevViewProj = m_taaPrevViewProj;
        tf.camPos = camera.position();
        tf.blend = m_taaPrevValid && m_taaWarmup > 1 ? 0.95f : 0.0f;
        tf.jitter = taaJitterUV;
        tf.sharpen = s.taaSharpen;
        // The HDR weight's exposure: the adapted scale (a frame or two
        // behind) under auto exposure.
        tf.exposure = s.autoExposure && m_stats.autoScale > 0.0f
                          ? m_stats.autoScale
                          : s.exposure;
        m_taa.draw(be, tf, ci.width, ci.height);
        resolvedByTaa = true;
        taaRanThisFrame = true;
      }
      m_taaPrevViewProj = camera.viewProjMatrix();
      m_taaPrevValid = true;
      ++m_taaWarmup;
      // Auto exposure over the resolved HDR (before bloom/tonemap).
      bool autoExposed = false;
      if (s.autoExposure && m_exposure.ready()) {
        ExposureView ev;
        ev.invViewProj = glm::inverse(camera.viewProjMatrix());
        ev.envTexture = m_env.texture;
        ev.envMip = std::max(m_env.maxLod - 5.0f, 0.0f);  // 1K -> 32x16
        ev.envIntensity = s.envIntensity;
        autoExposed =
            m_exposure.draw(be, s.exposureSettings, ev, resolvedByTaa,
                            ci.width, ci.height, in.dt, s.exposureForceTarget);
        float st[4];
        if (m_exposure.readState(be, st)) {
          m_stats.autoScale = st[0];
          m_stats.autoTargetScale = st[1];
          m_stats.autoAvgLum = st[2];
          m_stats.exposureValid = true;
        }
      }
      // Bloom over the resolved HDR (Final view only).
      const bool bloomed = s.bloom && s.viewMode == 0 &&
                           s.bloomStrength > 0.0f &&
                           m_bloom.draw(be, resolvedByTaa, ci.width, ci.height);
      TonemapPass::Sky skyView;
      skyView.invViewProj = glm::inverse(camera.viewProjMatrix());
      skyView.envTexture = m_env.texture;
      skyView.mip = std::min(s.skyBlur, m_env.maxLod);
      skyView.intensity = s.envIntensity;
      if (m_tonemapReady)
        m_tonemap.draw(be, static_cast<TonemapPass::Operator>(s.tonemapMode),
                       s.exposure, s.clearColor, ci.width, ci.height,
                       resolvedByTaa, resolvedByTaa ? s.taaSharpen : 0.0f,
                       bloomed ? s.bloomStrength : 0.0f,
                       1.0f / static_cast<float>(BloomPass::kLevels),
                       autoExposed ? m_exposure.stateBuffer() : ~0u,
                       s.sky ? &skyView : nullptr);
    }
  }
  m_taaRanLastFrame = taaRanThisFrame;
  m_lastRenderedFrame = in.frameIndex;
}

void SceneRenderer::afterEndFrame() {
  // The streaming pump rides between endFrame and the next execute
  // (readTouchedPages' pacing contract).
  if (m_desc.streamer && m_chainReady) {
    m_desc.streamer->onFrame(m_chain);
    m_stats.missPages = m_desc.streamer->missedPages();
  }
}

void SceneRenderer::logStats() {
  if (!m_chainReady || m_backend == nullptr) return;
  const SceneRenderStats& st = m_stats;
  KENSHIN_INFO(
      "[vgstats] work {} clusters {} tris {} writes {} occ {} revived {} hw "
      "{} tooCoarse {} srcTooCoarse {} visited {} pruned {} pend {} miss {} "
      "gpuMs {}",
      st.workItems, st.visibleClusters, st.cutTris, st.pxWrites, st.occCulled,
      st.revived, st.hwClusters, st.tooCoarse, st.srcTooCoarse,
      st.visitedNodes, st.prunedNodes, st.pendingSkips, st.missPages,
      m_backend->gpuTimings().frameMs);
  // Chain segment GPU times (execution only - scheduling latency sits
  // outside timestamps).
  const auto& gt = m_backend->gpuTimings();
  KENSHIN_INFO(
      "[chainms] cull {:.3f} sw {:.3f} hzb1 {:.3f} post {:.3f} hzb2 {:.3f} "
      "tail {:.3f}{}",
      gt.chainMs[0], gt.chainMs[1], gt.chainMs[2], gt.chainMs[3],
      gt.chainMs[4], gt.chainMs[5], gt.chainValid ? "" : " (unavailable)");
  // Coverage histogram (--raster-probe): triangles by covered-pixel
  // bucket 0 / 1 / 2-4 / 5-16 / 17-64 / >64, the pixel sum and the
  // attempted-triangle count (main + post).
  if (m_probeValid)
    KENSHIN_INFO(
        "[rasthist] b0 {} b1 {} b2_4 {} b5_16 {} b17_64 {} b64p {} pix {} "
        "tris {} bfcull {} nearclip {}",
        m_probeHist[0], m_probeHist[1], m_probeHist[2], m_probeHist[3],
        m_probeHist[4], m_probeHist[5], m_probeHist[6], m_probeHist[7],
        st.bfCulls, st.nearClips);
  // Shadow clipmap content stats: per level the cut triangles + pixel
  // writes of the LAST render (stale while cached - the cache working).
  if (m_shadowChainReady) {
    for (u32 lv = 0; lv < 3; ++lv)
      if (!m_shadowChains[lv].readStatsMapped(*m_backend,
                                              m_shadowChainStats[lv]))
        break;
    KENSHIN_INFO(
        "[shadowstats] key {} dirty {:02x}/{:02x}/{:02x} | lv0 tris {} px {} "
        "hw {} pend {} | lv1 tris {} px {} pend {} | lv2 tris {} px {} pend "
        "{}",
        m_shadowKey.valid ? 1 : 0, m_shadowDirtyTiles[0] & 0xFFFFu,
        m_shadowDirtyTiles[1] & 0xFFFFu, m_shadowDirtyTiles[2] & 0xFFFFu,
        m_shadowChainStats[0][11],
        m_shadowChainStats[0][5] + m_shadowChainStats[0][10],
        m_shadowChainStats[0][12], m_shadowChainStats[0][15],
        m_shadowChainStats[1][11],
        m_shadowChainStats[1][5] + m_shadowChainStats[1][10],
        m_shadowChainStats[1][15], m_shadowChainStats[2][11],
        m_shadowChainStats[2][5] + m_shadowChainStats[2][10],
        m_shadowChainStats[2][15]);
  }
}

void SceneRenderer::shutdown() {
  if (m_backend == nullptr) return;
  backend::IRenderBackend& be = *m_backend;
  m_decode.onShutdown(be);
  if (m_lightingReady) m_lighting.onShutdown(be);
  if (m_tonemapReady) m_tonemap.onShutdown(be);
  m_exposure.onShutdown(be);
  if (m_taaReady) m_taa.onShutdown(be);
  m_translucent.onShutdown(be);
  if (m_imposterReady) m_imposter.onShutdown(be);
  if (m_chainReady) m_chain.onShutdown(be);
  if (m_shadowChainReady)
    for (auto& sc : m_shadowChains) sc.onShutdown(be);
  m_backend = nullptr;
}

}  // namespace renderer

KENSHIN_NS_END
