#pragma once

// ── .kgeo loader (S3.4; M4.4 retired the CPU-cut render half) ───
// Loads the cluster DAG - the data source for the GPU virtual
// geometry path (VG scene pools + the chain). The per-frame CPU cut
// selection / mesh rebuild that this preview carried until M4.4 is
// deleted (the chain's readStats feed the panels now); the engine's
// selectLodCut stays as the LOD semantic ground truth for tests.

#include <string>

#include "kenshin/Common.h"
#include "kenshin/geometry/ClusterDag.h"
#include "kenshin/geometry/KGeoFile.h"

namespace demo {

struct KgeoPreview {
  kenshin::geometry::ClusterDag dag;
};

/// Read a .kgeo file into the preview.
bool loadKgeo(const std::string& path, KgeoPreview& preview, std::string& log);

/// ImGui panel with the source-mesh stats (static; the live cut
/// numbers live in the Virtual Geometry panel).
void drawLodPanel(KgeoPreview& preview);

}  // namespace demo
