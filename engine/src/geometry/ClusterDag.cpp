#include "kenshin/geometry/ClusterDag.h"

#include "kenshin/geometry/ClusterLodBuilder.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <queue>

KENSHIN_NS_BEGIN

namespace geometry
{

    namespace
    {
        glm::vec4 unionSphere(const glm::vec4& a, const glm::vec4& b)
        {
            if (b.w < 0.0f)
                return a;
            if (a.w < 0.0f)
                return b;
            const glm::vec3 pa(a), pb(b);
            const f32 d = glm::distance(pa, pb);
            if (d + b.w <= a.w)
                return a; // b inside a
            if (d + a.w <= b.w)
                return b; // a inside b
            const f32 t = (d + a.w - b.w) * 0.5f;
            const glm::vec3 dir = d > 0.0f ? (pb - pa) / d : glm::vec3(0.0f);
            const glm::vec3 center = pa + dir * t;
            return glm::vec4(center, (a.w + b.w + d) * 0.5f);
        }
    } // namespace

    bool assembleClusterDag(const assets::MeshData& mesh,
                            const std::vector<ClusterGroup>& builtGroups,
                            ClusterDag& dag, DagBuildStats& stats, std::string& log)
    {
        if (builtGroups.empty())
        {
            log = "no groups provided";
            return false;
        }

        dag = {};
        dag.sourceVertexCount = mesh.vertexCount();
        dag.sourceTriangleCount = mesh.triangleCount();

        // Pass 1: clusters + groups; geometry into cluster-local pools with
        // per-cluster vertex dedup.
        dag.groups.resize(builtGroups.size());
        for (u32 g = 0; g < builtGroups.size(); ++g)
        {
            const ClusterGroup& src = builtGroups[g];
            DagGroup& dst = dag.groups[g];
            dst.lodLevel = src.depth;
            for (int k = 0; k < 4; ++k)
                dst.boundsSphere[k] = src.simplifiedCenter[k];
            for (usize ci = 0; ci < src.clusters.size(); ++ci)
            {
                const Cluster& cluster = src.clusters[ci];
                const u32 clusterIndex = static_cast<u32>(dag.clusters.size());

                DagCluster node;
                node.group = g;
                for (int k = 0; k < 4; ++k)
                    node.boundsSphere[k] = cluster.boundsCenter[k];
                node.error = cluster.error;
                node.vertexOffset = static_cast<u32>(dag.positions.size() / 3);
                node.indexOffset = static_cast<u32>(dag.indices.size());

                std::map<u32, u32> localOf; // source vertex -> local index
                for (u32 index : cluster.indices)
                {
                    auto it = localOf.find(index);
                    if (it == localOf.end())
                    {
                        it = localOf.emplace(index, static_cast<u32>(localOf.size())).first;
                        const usize v = index;
                        for (int k = 0; k < 3; ++k)
                            dag.positions.push_back(mesh.positions[v * 3 + k]);
                        if (mesh.normals.size() == mesh.positions.size())
                            for (int k = 0; k < 3; ++k)
                                dag.normals.push_back(mesh.normals[v * 3 + k]);
                        if (mesh.uvs.size() / 2 == mesh.vertexCount())
                        {
                            dag.uvs.push_back(mesh.uvs[v * 2]);
                            dag.uvs.push_back(mesh.uvs[v * 2 + 1]);
                        }
                        else
                        {
                            dag.uvs.push_back(0.0f);
                            dag.uvs.push_back(0.0f);
                        }
                        node.vertexCount += 1;
                    }
                    dag.indices.push_back(it->second);
                }
                node.indexCount = static_cast<u32>(cluster.indices.size());
                dag.clusters.push_back(std::move(node));
                dst.clusters.push_back(clusterIndex);
            }
        }

        // Pass 2: DAG edges. Cluster::parentGroupId holds the group id OUR
        // callback returned (= index in builtGroups). Per the clusterlod
        // contract every cluster of a child group descends from the whole
        // parent group, so link parent group's every cluster -> each child
        // (Nanite "all siblings connect to all parents").
        for (u32 g = 0; g < builtGroups.size(); ++g)
        {
            const ClusterGroup& src = builtGroups[g];
            for (usize ci = 0; ci < src.clusters.size(); ++ci)
            {
                const Cluster& childMeta = src.clusters[ci];
                if (childMeta.parentGroupId < 0)
                    continue;
                const u32 childCluster = dag.groups[g].clusters[ci];
                const u32 parentGroup = static_cast<u32>(childMeta.parentGroupId);
                for (u32 parentCluster : dag.groups[parentGroup].clusters)
                    dag.clusters[parentCluster].children.push_back(childCluster);
                dag.groups[g].parents.push_back(parentGroup);
            }
        }
        for (auto& group : dag.groups)
        {
            std::sort(group.parents.begin(), group.parents.end());
            group.parents.erase(std::unique(group.parents.begin(), group.parents.end()),
                                group.parents.end());
        }

        // Roots: terminal clusters (nobody refines them further).
        std::vector<bool> isRefined(dag.clusters.size(), false);
        for (const auto& cluster : dag.clusters)
            for (u32 child : cluster.children)
                isRefined[child] = true;
        for (u32 c = 0; c < dag.clusters.size(); ++c)
            if (!isRefined[c])
            {
                dag.clusters[c].isRoot = true;
                dag.roots.push_back(c);
            }

        // Pass 3: group decision data (union bounds + max member error).
        for (auto& group : dag.groups)
        {
            glm::vec4 bounds(0.0f, 0.0f, 0.0f, -1.0f); // invalid seed
            f32 maxErr = 0.0f;
            for (u32 c : group.clusters)
            {
                bounds = unionSphere(bounds, dag.clusters[c].boundsSphere);
                maxErr = std::max(maxErr, dag.clusters[c].error);
            }
            group.boundsSphere = bounds;
            group.maxError = maxErr;
        }

        // Pass 4: monotonic error propagation, deepest groups first (build
        // order is non-decreasing depth, so reverse iterate):
        //   parentError(c) = max(error(c), parentError(children), group.maxError)
        // Members of a group all end with the same parentError (shared
        // decision data => identical parallel LOD decisions).
        for (usize gi = dag.groups.size(); gi-- > 0;)
        {
            DagGroup& group = dag.groups[gi];
            f32 subtreeError = group.maxError;
            for (u32 c : group.clusters)
            {
                f32 childMax = dag.clusters[c].error;
                for (u32 child : dag.clusters[c].children)
                    childMax = std::max(childMax, dag.clusters[child].parentError);
                subtreeError = std::max(subtreeError, childMax);
            }
            group.maxError = std::max(group.maxError, subtreeError);
            for (u32 c : group.clusters)
            {
                const f32 raised = std::max(dag.clusters[c].error, group.maxError);
                if (raised > dag.clusters[c].parentError)
                    ++stats.monotonicFixes;
                dag.clusters[c].parentError = raised;
            }
        }

        stats.clusterCount = static_cast<u32>(dag.clusters.size());
        stats.groupCount = static_cast<u32>(dag.groups.size());
        stats.rootCount = static_cast<u32>(dag.roots.size());
        for (const auto& group : dag.groups)
            stats.maxLevel = std::max(stats.maxLevel,
                                      static_cast<u32>(group.lodLevel + 1));
        return verifyDagInvariants(dag, log);
    }

    bool verifyDagInvariants(const ClusterDag& dag, std::string& log)
    {
        // 1) Acyclic (Kahn over the cluster graph).
        {
            std::vector<u32> indegree(dag.clusters.size(), 0);
            for (const auto& cluster : dag.clusters)
                for (u32 child : cluster.children)
                    ++indegree[child];
            std::queue<u32> queue;
            for (u32 c = 0; c < dag.clusters.size(); ++c)
                if (indegree[c] == 0)
                    queue.push(c);
            u32 visited = 0;
            while (!queue.empty())
            {
                const u32 c = queue.front();
                queue.pop();
                ++visited;
                for (u32 child : dag.clusters[c].children)
                    if (--indegree[child] == 0)
                        queue.push(child);
            }
            if (visited != dag.clusters.size())
            {
                log = "cycle detected: visited " + std::to_string(visited) + " of " +
                      std::to_string(dag.clusters.size()) + " clusters";
                return false;
            }
        }

        // 2) Monotonic error along every parent->child edge.
        for (const auto& cluster : dag.clusters)
        {
            if (cluster.parentError + 1e-6f < cluster.error)
            {
                log = "parentError < own error";
                return false;
            }
            for (u32 child : cluster.children)
                if (cluster.parentError + 1e-6f < dag.clusters[child].parentError)
                {
                    log = "parentError(parent) < parentError(child)";
                    return false;
                }
        }

        // 3) Same-group clusters share the decision data.
        for (const auto& group : dag.groups)
        {
            if (group.clusters.empty())
                continue;
            const DagCluster& first = dag.clusters[group.clusters.front()];
            for (u32 c : group.clusters)
                if (std::fabs(dag.clusters[c].parentError - first.parentError) > 1e-6f)
                {
                    log = "group members disagree on parentError";
                    return false;
                }
        }

        // 4) Pools consistent with the cluster slices.
        if (dag.positions.size() % 3 != 0 || dag.indices.size() % 3 != 0 ||
            (!dag.normals.empty() && dag.normals.size() != dag.positions.size()) ||
            dag.uvs.size() / 2 != dag.positions.size() / 3)
        {
            log = "pool size mismatch";
            return false;
        }
        for (const auto& cluster : dag.clusters)
        {
            if (cluster.vertexOffset + cluster.vertexCount > dag.positions.size() / 3 ||
                cluster.indexOffset + cluster.indexCount > dag.indices.size())
            {
                log = "cluster slice out of pool bounds";
                return false;
            }
            for (usize i = cluster.indexOffset;
                 i < cluster.indexOffset + cluster.indexCount; ++i)
                if (dag.indices[i] >= cluster.vertexCount)
                {
                    log = "cluster-local index out of vertex range";
                    return false;
                }
        }
        return true;
    }

} // namespace geometry

KENSHIN_NS_END
