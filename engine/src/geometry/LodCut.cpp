#include "kenshin/geometry/LodCut.h"

#include <algorithm>
#include <cmath>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/constants.hpp>

KENSHIN_NS_BEGIN

namespace geometry
{

    namespace
    {
        /// Convert a screen-space pixel error into an object-space world error
        /// at a given distance (same math the GPU cull will use):
        ///   worldError = pixels * (2 * tan(fov/2) / viewHeight) * distance
        f32 worldErrorAt(f32 pixelError, u32 viewHeight, f32 fovYDeg, f32 distance)
        {
            const f32 worldPerPixel =
                2.0f * std::tan(glm::radians(fovYDeg) * 0.5f) /
                static_cast<f32>(viewHeight);
            return pixelError * worldPerPixel * distance;
        }
    } // namespace

    void selectLodCut(const ClusterDag& dag, f32 pixelThreshold, u32 viewHeight,
                      f32 fovYDeg, const float cameraPos[3],
                      std::vector<u32>& outClusters)
    {
        outClusters.clear();

        // Camera-relative world threshold per group (the Phase 4 GPU rule
        // shape): worldErrorAt(distance(groupBounds, camera)). Groups near
        // the camera demand finer LOD, far groups coarsen earlier.
        const glm::vec3 cam(cameraPos[0], cameraPos[1], cameraPos[2]);
        std::vector<f32> thresholdOf(dag.groups.size());
        for (usize g = 0; g < dag.groups.size(); ++g)
        {
            const DagGroup& group = dag.groups[g];
            const glm::vec3 center(group.boundsSphere);
            const f32 distance = std::max(glm::distance(center, cam) -
                                              group.boundsSphere.w,
                                          0.1f);
            thresholdOf[g] = worldErrorAt(pixelThreshold, viewHeight, fovYDeg,
                                          distance);
        }

        // Render rule (clusterlod semantics, from its header):
        //   1. own group's simplifyError > T  (coarsening further would be
        //      too lossy; FLT_MAX on terminal groups always passes)
        //   2. every refined (finer parent) group's simplifyError <= T
        //      (the coarsening that produced this level was acceptable)
        // clusterlod's error merging already makes simplifyError monotonic
        // along a chain, so exactly one level crosses the threshold ->
        // crack-free cut by construction.
        for (u32 c = 0; c < dag.clusters.size(); ++c)
        {
            const DagCluster& cluster = dag.clusters[c];
            const DagGroup& group = dag.groups[cluster.group];

            // Per-CLUSTER threshold (own bounds distance) for BOTH
            // conditions - mixing different groups' distance-scaled
            // thresholds breaks the crossing logic and leaves holes on
            // paths where no level fires.
            const f32 worldThreshold = thresholdOf[cluster.group];
            if (!(group.simplifyError > worldThreshold))
                continue;
            bool parentsAcceptable = true;
            for (u32 parentGroup : group.parents)
                if (dag.groups[parentGroup].simplifyError > worldThreshold)
                {
                    parentsAcceptable = false;
                    break;
                }
            if (parentsAcceptable)
                outClusters.push_back(c);
        }

        // Fallback: pathological thresholds never select - render roots.
        if (outClusters.empty())
            outClusters = dag.roots;
    }
    CutStats summarizeCut(const ClusterDag& dag, const std::vector<u32>& clusters)
    {
        CutStats stats;
        stats.clusterCount = static_cast<u32>(clusters.size());
        stats.minLevel = 0x7FFFFFFF;
        for (u32 c : clusters)
        {
            stats.triangleCount += dag.triangleCount(c);
            const i32 level = dag.groups[dag.clusters[c].group].lodLevel;
            stats.minLevel = std::min(stats.minLevel, level);
            stats.maxLevel = std::max(stats.maxLevel, level);
        }
        if (clusters.empty())
            stats.minLevel = 0;
        return stats;
    }

} // namespace geometry

KENSHIN_NS_END
