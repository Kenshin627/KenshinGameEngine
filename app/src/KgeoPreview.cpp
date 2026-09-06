#include "KgeoPreview.h"

#pragma warning(push, 0)
#include <imgui.h>
#pragma warning(pop)

namespace demo {

bool loadKgeo(const std::string& path, KgeoPreview& preview, std::string& log) {
  return kenshin::geometry::readKGeo(path, preview.dag, log);
}

void drawLodPanel(KgeoPreview& preview) {
  ImGui::Begin("LOD Data");
  ImGui::Text("source: %u tris", preview.dag.sourceTriangleCount);
  ImGui::Text("clusters: %zu  groups: %zu", preview.dag.clusters.size(),
              preview.dag.groups.size());
  ImGui::End();
}

}  // namespace demo
