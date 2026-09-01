#pragma once

// ── .kgeo LOD preview (S3.4) ─────────────────────────────────────
// Load the cluster DAG, re-select the cut each frame from the
// pixel-error threshold + camera, and rebuild one concatenated mesh
// drawn through the baseline path. This is the pre-VisBuffer preview;
// M2's GPU runtime replaces it.

#include "kenshin/Common.h"
#include "kenshin/assets/MeshData.h"
#include "kenshin/geometry/KGeoFile.h"
#include "kenshin/geometry/LodCut.h"
#include "kenshin/manager/MeshPool.h"
#include "kenshin/scene/Camera.h"
#include "kenshin/scene/CameraController.h"

#include <string>
#include <vector>

namespace demo
{

struct KgeoPreview
{
    kenshin::geometry::ClusterDag dag;
    kenshin::assets::MeshHandle meshHandle = kenshin::assets::kInvalidMeshHandle;
    std::vector<kenshin::u32> cut;
    kenshin::geometry::CutStats lastStats;
    float pixelThreshold = 1.0f;
    bool lodColors = true;   // false = shaded
    bool needRebuild = true;
    std::vector<float> history; // triangle count over threshold changes
};

/// Read a .kgeo file into the preview (cut selection happens per frame).
bool loadKgeo(const std::string& path, KgeoPreview& preview, std::string& log);

/// Per-frame: re-select the cut when the threshold/camera moved, rebuild
/// the concatenated mesh. viewportHeight feeds the pixel-error metric.
void updateCut(KgeoPreview& preview, kenshin::manager::MeshPool& pool,
               const kenshin::scene::Camera& camera,
               const kenshin::scene::CameraController& controller,
               kenshin::u32 viewportHeight);

/// ImGui panel for the preview (threshold slider, cut stats, history).
void drawLodPanel(KgeoPreview& preview);

} // namespace demo
