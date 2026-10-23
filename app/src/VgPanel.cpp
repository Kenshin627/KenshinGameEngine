#include "VgPanel.h"

#include "kenshin/renderer/StreamingManager.h"

#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include <imgui.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace demo {

void drawVgPanel(VgDebugState& vgDebug, KgeoPreview& kgeoPreview,
                 const kenshin::renderer::StreamingManager* streamer) {
  demo::drawLodPanel(kgeoPreview);
  {
    ImGui::Begin("Virtual Geometry");
    const char* names[] = {"final",     "LOD level", "error heat",
                           "triangles", "checker",   "clusters",
                           "route path"};
    ImGui::SetNextItemWidth(150.0f);
    ImGui::Combo("view", &vgDebug.viewMode, names, 7);
    ImGui::Separator();
    ImGui::Text(
        "clusters %u  tris %u  writes %u  cover %.1f%%\n"
        "occ-culled %u  revived %u  hw %u",
        vgDebug.visibleClusters, vgDebug.cutTris, vgDebug.pxWrites,
        vgDebug.coveragePct, vgDebug.occCulled, vgDebug.revived,
        vgDebug.hwClusters);
    ImGui::Checkbox("occlusion", &vgDebug.occlusion);
    ImGui::Checkbox("BVH8 cull", &vgDebug.hierCull);
    ImGui::SameLine();
    ImGui::Checkbox("count HW px", &vgDebug.hwPxStats);
    ImGui::SliderFloat("pixel error", &vgDebug.pixelThreshold, 0.05f,
                       32.0f, "%.2f px", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("HW route >=", &vgDebug.hwThresholdPx, 1.0f,
                       128.0f, "%.0f px", ImGuiSliderFlags_Logarithmic);
    if (vgDebug.viewMode == 0) {
      // M8.1 lighting rig (only meaningful in the deferred final
      // view). Static table - deterministic for the regression.
      ImGui::Separator();
      ImGui::Text("lighting (deferred)");
      ImGui::SliderFloat3("dir light", &vgDebug.lightDir.x, -1.0f, 1.0f,
                          "%.2f");
      ImGui::SliderFloat("intensity", &vgDebug.lightIntensity, 0.0f,
                         10.0f, "%.2f");
      ImGui::SliderFloat("points", &vgDebug.pointIntensity, 0.0f, 30.0f,
                         "%.2f");
      ImGui::Checkbox("shadows", &vgDebug.shadows);
      ImGui::SameLine();
      ImGui::Checkbox("TAA", &vgDebug.taa);
      ImGui::Checkbox("imposter", &vgDebug.imposterDebug);
      if (vgDebug.taa)
        ImGui::SliderFloat("sharpen", &vgDebug.taaSharpen, 0.0f, 1.0f);
      ImGui::SliderInt("shadow levels", &vgDebug.shadowLevels, 1, 3);
      // Three items: the default (2, UE Filmic) was unselectable
      // and rendered as a blank combo.
      ImGui::Combo("tonemap", &vgDebug.tonemapMode,
                   "Reinhard\0ACES\0Filmic (UE)\0");
      ImGui::SliderFloat("env intensity", &vgDebug.envIntensity, 0.0f,
                         4.0f, "%.2f");
      ImGui::ColorEdit3("sky (no probe)", &vgDebug.skyColor.x);
      ImGui::ColorEdit3("ground (no probe)", &vgDebug.groundColor.x);
      ImGui::SliderFloat("ambient (no probe)", &vgDebug.ambient, 0.0f,
                         0.5f, "%.3f");
      ImGui::Checkbox("auto exposure", &vgDebug.autoExposure);
      if (vgDebug.autoExposure) {
        auto& es = vgDebug.exposureSettings;
        ImGui::SliderFloat("exposure bias (EV)", &es.biasEv, -4.0f,
                           4.0f, "%.2f");
        ImGui::SliderFloat("adapt up (EV/s)", &es.speedUp, 0.1f, 10.0f,
                           "%.1f");
        ImGui::SliderFloat("adapt down (EV/s)", &es.speedDown, 0.1f,
                           10.0f, "%.1f");
        ImGui::Text("scale %.3f  avg lum %.4f", vgDebug.autoScale,
                    vgDebug.autoAvgLum);
      } else {
        ImGui::SliderFloat("exposure", &vgDebug.exposure, 0.1f, 4.0f,
                           "%.2f");
      }
      ImGui::Checkbox("environment background", &vgDebug.sky);
      if (vgDebug.sky)
        ImGui::SliderFloat("background blur", &vgDebug.skyBlur, 0.0f,
                           8.0f, "%.1f");
      ImGui::Checkbox("bloom", &vgDebug.bloom);
      if (vgDebug.bloom)
        ImGui::SliderFloat("bloom strength", &vgDebug.bloomStrength,
                           0.0f, 0.3f, "%.3f");
      ImGui::SliderFloat("emissive strength", &vgDebug.emissiveStrength,
                         0.0f, 50.0f, "%.2f",
                         ImGuiSliderFlags_Logarithmic);
      ImGui::Checkbox("emissive through translucency",
                      &vgDebug.emissivePassThrough);
    }
    ImGui::PlotLines("##clusters", vgDebug.clusterHist.data(),
                     int(vgDebug.clusterHist.size()), 0, nullptr, 0.0f,
                     1200.0f, ImVec2(240, 40));
    ImGui::PlotLines("##tris", vgDebug.triHist.data(),
                     int(vgDebug.triHist.size()), 0, nullptr, 0.0f,
                     200000.0f, ImVec2(240, 40));
    ImGui::PlotLines("##coverage", vgDebug.coverHist.data(),
                     int(vgDebug.coverHist.size()), 0, nullptr, 0.0f,
                     100.0f, ImVec2(240, 40));
    ImGui::PlotLines("##occ", vgDebug.occHist.data(),
                     int(vgDebug.occHist.size()), 0, nullptr, 0.0f,
                     1500.0f, ImVec2(240, 40));
    ImGui::PlotLines("##revive", vgDebug.reviveHist.data(),
                     int(vgDebug.reviveHist.size()), 0, nullptr, 0.0f,
                     1500.0f, ImVec2(240, 40));
    ImGui::PlotLines("##hw", vgDebug.hwHist.data(),
                     int(vgDebug.hwHist.size()), 0, nullptr, 0.0f,
                     1500.0f, ImVec2(240, 40));
    if (ImGui::CollapsingHeader("Traversal (M5)")) {
      ImGui::Text("visited %u  pruned %u", vgDebug.visitedNodes,
                  vgDebug.prunedNodes);
      ImGui::PlotLines("##visited", vgDebug.visitedHist.data(),
                       int(vgDebug.visitedHist.size()), 0, nullptr, 0.0f,
                       20000.0f, ImVec2(240, 40));
      ImGui::PlotLines("##pruned", vgDebug.prunedHist.data(),
                       int(vgDebug.prunedHist.size()), 0, nullptr, 0.0f,
                       20000.0f, ImVec2(240, 40));
    }
    if (streamer != nullptr && ImGui::CollapsingHeader("Streaming (M6)")) {
      const double mb = 1024.0 * 1024.0;
      ImGui::Text(
          "resident %.1f MB (%u pages)  pending skips %u\n"
          "missed pages %u  io queue %u",
          double(streamer->residentBytes()) / mb,
          streamer->residentPages(), vgDebug.pendingSkips,
          vgDebug.missPages, streamer->pendingRequests());
      ImGui::PlotLines("##pending", vgDebug.pendingHist.data(),
                       int(vgDebug.pendingHist.size()), 0, nullptr, 0.0f,
                       2000.0f, ImVec2(240, 40));
      ImGui::PlotLines("##miss", vgDebug.missHist.data(),
                       int(vgDebug.missHist.size()), 0, nullptr, 0.0f,
                       500.0f, ImVec2(240, 40));
    }
    ImGui::End();
  }
}

kenshin::renderer::SceneRenderSettings resolveSettings(
    const VgDebugState& vgDebug, const Args& args) {
  kenshin::renderer::SceneRenderSettings s;
  s.viewMode = vgDebug.viewMode;
  s.pixelThreshold = vgDebug.pixelThreshold;
  s.hwThresholdPx = vgDebug.hwThresholdPx;  // CLI seeds, panel drives
  // HW/SW routing is the default (UE's shape); --no-hw keeps pure SW
  // for A/B, --force-hw routes everything to HW (--route-hw: the
  // default's alias). The shadow chains resolve the same flags with
  // --no-hw taking precedence.
  s.hwMode = args.forceHw ? 2u : (args.noHw ? 0u : 1u);
  s.shadowHwMode = args.noHw ? 0u : (args.forceHw ? 2u : 1u);
  s.cullMode =
      !vgDebug.hierCull || args.cullLinear ? 0u : (args.cullPersistent ? 2u : 1u);
  // Mode 1 (exact row-span clamp + stat aggregation) is the measured
  // default; --raster-base keeps the baseline walk for A/B.
  s.rasterMode = args.rasterBase ? 0u : 1u;
  s.rasterProbe = args.rasterProbe;
  s.hwPixelStats = args.hwPxStats || vgDebug.hwPxStats;
  s.occlusion = vgDebug.occlusion && !args.noOcclusion;
  s.imposters = args.imposterDebug || vgDebug.imposterDebug;
  s.lightDir = vgDebug.lightDir;
  s.ambient = vgDebug.ambient;
  s.skyColor = vgDebug.skyColor;
  s.groundColor = vgDebug.groundColor;
  s.envIntensity = vgDebug.envIntensity;
  s.emissiveStrength = vgDebug.emissiveStrength;
  s.emissivePassThrough = vgDebug.emissivePassThrough;
  s.shadows = vgDebug.shadows && !args.noShadows;
  s.shadowLevels = vgDebug.shadowLevels;
  // The clipmap centers on the --cam target (camera-following is
  // roadmap step 1.4).
  s.shadowCenter = args.hasCam ? glm::vec3(args.camTarget[0], args.camTarget[1],
                                           args.camTarget[2])
                               : glm::vec3(0.0f);
  s.taa = (vgDebug.taa || args.forceTaa) && !args.noTaa;
  s.taaSharpen = vgDebug.taaSharpen;
  s.exposure = vgDebug.exposure;
  s.autoExposure = vgDebug.autoExposure;
  s.exposureSettings = vgDebug.exposureSettings;
  // Headless runs jump to the exposure target every frame: screenshots
  // stay deterministic whatever the frame timing.
  s.exposureForceTarget = args.frames >= 0;
  s.bloom = vgDebug.bloom;
  s.bloomStrength = vgDebug.bloomStrength;
  s.sky = vgDebug.sky;
  s.skyBlur = vgDebug.skyBlur;
  s.tonemapMode = vgDebug.tonemapMode;
  s.clearColor = glm::vec4(kClearColor[0], kClearColor[1], kClearColor[2], 1.0f);
  return s;
}

void absorbStats(VgDebugState& vgDebug,
                 const kenshin::renderer::SceneRenderStats& st) {
  if (st.exposureValid) {
    vgDebug.autoScale = st.autoScale;
    vgDebug.autoAvgLum = st.autoAvgLum;
  }
  vgDebug.missPages = st.missPages;
  if (!st.chainStatsValid) return;
  vgDebug.visibleClusters = st.visibleClusters;
  vgDebug.cutTris = st.cutTris;
  vgDebug.pxWrites = st.pxWrites;
  vgDebug.occCulled = st.occCulled;
  vgDebug.revived = st.revived;
  vgDebug.hwClusters = st.hwClusters;
  vgDebug.bfCulls = st.bfCulls;
  vgDebug.nearClips = st.nearClips;
  vgDebug.workItems = st.workItems;
  vgDebug.tooCoarse = st.tooCoarse;
  vgDebug.srcTooCoarse = st.srcTooCoarse;
  vgDebug.visitedNodes = st.visitedNodes;
  vgDebug.prunedNodes = st.prunedNodes;
  vgDebug.pendingSkips = st.pendingSkips;
  vgDebug.coveragePct = st.coveragePct;
  vgDebug.pushHistory();
}

}  // namespace demo
