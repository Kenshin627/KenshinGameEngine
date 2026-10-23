#include "KgeoPreview.h"

#ifdef _MSC_VER

#pragma warning(push, 0)

#endif
#include <imgui.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
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
