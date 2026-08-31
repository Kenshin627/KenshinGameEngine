#include "kenshin/geometry/KGeoFile.h"

#include "kenshin/core/Log.h"

#include <cstdio>
#include <cstring>
#include <vector>

KENSHIN_NS_BEGIN

namespace geometry
{

    namespace
    {
        constexpr u32 kMagic = 0x314F474B; // "KGO1"
        constexpr u32 kVersion = 1;

#pragma pack(push, 4)
        struct FileHeader
        {
            u32 magic;
            u32 version;
            u32 groupCount;
            u32 clusterCount;
            u32 childEdgeCount;
            u32 rootCount;
            u32 poolVertexCount;
            u32 poolIndexCount;
            u32 hasNormals;
            u32 hasUvs;
            u32 sourceVertexCount;
            u32 sourceTriangleCount;
        };

        struct GroupDisk
        {
            i32 lodLevel;
            f32 maxError;      // monotonic subtree error (GPU rule data)
            f32 simplifyError; // clusterlod semantics: error of coarsening FROM
                               // this group; FLT_MAX on terminal groups
            f32 bounds[4]; // xyz + radius
            u32 memberOffset;
            u32 memberCount;
            u32 parentOffset;
            u32 parentCount;
        };

        struct ClusterDisk
        {
            f32 bounds[4];
            f32 error;
            f32 parentError;
            u32 vertexOffset;
            u32 vertexCount;
            u32 indexOffset;
            u32 indexCount;
            u32 group;
            u32 childOffset;
            u32 childCount;
            u32 isRoot;
            u32 pad;
        };
#pragma pack(pop)

        /// Span arrays are packed end-inclusively: offsets[i] = start of
        /// span i, count in the parallel field; offset[0] = 0.
        template <typename T>
        bool writeVector(std::FILE* file, const std::vector<T>& data)
        {
            return data.empty() ||
                   std::fwrite(data.data(), sizeof(T), data.size(), file) == data.size();
        }

        template <typename T>
        bool readVector(std::FILE* file, std::vector<T>& data, usize count)
        {
            data.resize(count);
            return count == 0 ||
                   std::fread(data.data(), sizeof(T), count, file) == count;
        }
    } // namespace

    bool writeKGeo(const std::string& path, const ClusterDag& dag, std::string& log)
    {
        std::FILE* file = nullptr;
        if (fopen_s(&file, path.c_str(), "wb") != 0 || !file)
        {
            log = "cannot open for write: " + path;
            return false;
        }

        // Pack member/parent/child spans.
        std::vector<u32> memberIndices, parentIndices, childIndices;
        std::vector<GroupDisk> groups(dag.groups.size());
        for (u32 g = 0; g < dag.groups.size(); ++g)
        {
            const DagGroup& src = dag.groups[g];
            groups[g] = GroupDisk {
                .lodLevel = src.lodLevel,
                .maxError = src.maxError,
                .simplifyError = src.simplifyError,
                .bounds = {src.boundsSphere.x, src.boundsSphere.y,
                           src.boundsSphere.z, src.boundsSphere.w},
                .memberOffset = static_cast<u32>(memberIndices.size()),
                .memberCount = static_cast<u32>(src.clusters.size()),
                .parentOffset = static_cast<u32>(parentIndices.size()),
                .parentCount = static_cast<u32>(src.parents.size()),
            };
            memberIndices.insert(memberIndices.end(), src.clusters.begin(),
                                 src.clusters.end());
            parentIndices.insert(parentIndices.end(), src.parents.begin(),
                                 src.parents.end());
        }

        std::vector<ClusterDisk> clusters(dag.clusters.size());
        for (u32 c = 0; c < dag.clusters.size(); ++c)
        {
            const DagCluster& src = dag.clusters[c];
            clusters[c] = ClusterDisk {
                .bounds = {src.boundsSphere.x, src.boundsSphere.y,
                           src.boundsSphere.z, src.boundsSphere.w},
                .error = src.error,
                .parentError = src.parentError,
                .vertexOffset = src.vertexOffset,
                .vertexCount = src.vertexCount,
                .indexOffset = src.indexOffset,
                .indexCount = src.indexCount,
                .group = src.group,
                .childOffset = static_cast<u32>(childIndices.size()),
                .childCount = static_cast<u32>(src.children.size()),
                .isRoot = src.isRoot ? 1u : 0u,
                .pad = 0,
            };
            childIndices.insert(childIndices.end(), src.children.begin(),
                                src.children.end());
        }

        const FileHeader header {
            .magic = kMagic,
            .version = kVersion,
            .groupCount = static_cast<u32>(dag.groups.size()),
            .clusterCount = static_cast<u32>(dag.clusters.size()),
            .childEdgeCount = static_cast<u32>(childIndices.size()),
            .rootCount = static_cast<u32>(dag.roots.size()),
            .poolVertexCount = static_cast<u32>(dag.positions.size() / 3),
            .poolIndexCount = static_cast<u32>(dag.indices.size()),
            .hasNormals = dag.normals.empty() ? 0u : 1u,
            .hasUvs = dag.uvs.size() / 2 == dag.positions.size() / 3 ? 1u : 0u,
            .sourceVertexCount = dag.sourceVertexCount,
            .sourceTriangleCount = dag.sourceTriangleCount,
        };

        const bool ok = std::fwrite(&header, sizeof(header), 1, file) == 1 &&
                        writeVector(file, groups) &&
                        writeVector(file, clusters) &&
                        writeVector(file, childIndices) &&
                        writeVector(file, memberIndices) &&
                        writeVector(file, parentIndices) &&
                        writeVector(file, dag.roots) &&
                        writeVector(file, dag.positions) &&
                        (header.hasNormals == 0 || writeVector(file, dag.normals)) &&
                        (header.hasUvs == 0 || writeVector(file, dag.uvs)) &&
                        writeVector(file, dag.indices);
        std::fclose(file);
        if (!ok)
            log = "write failed (disk full?)";
        return ok;
    }

    bool readKGeo(const std::string& path, ClusterDag& dag, std::string& log)
    {
        std::FILE* file = nullptr;
        if (fopen_s(&file, path.c_str(), "rb") != 0 || !file)
        {
            log = "cannot open for read: " + path;
            return false;
        }

        FileHeader header {};
        if (std::fread(&header, sizeof(header), 1, file) != 1 || header.magic != kMagic ||
            header.version != kVersion)
        {
            log = "bad magic/version";
            std::fclose(file);
            return false;
        }

        std::vector<GroupDisk> groups;
        std::vector<ClusterDisk> clusters;
        std::vector<u32> childIndices, memberIndices, parentIndices;
        if (!readVector(file, groups, header.groupCount) ||
            !readVector(file, clusters, header.clusterCount) ||
            !readVector(file, childIndices, header.childEdgeCount))
        {
            log = "truncated (groups/clusters/children)";
            std::fclose(file);
            return false;
        }

        usize memberTotal = 0, parentTotal = 0;
        for (const auto& g : groups)
        {
            memberTotal += g.memberCount;
            parentTotal += g.parentCount;
        }
        if (!readVector(file, memberIndices, memberTotal) ||
            !readVector(file, parentIndices, parentTotal) ||
            !readVector(file, dag.roots, header.rootCount) ||
            !readVector(file, dag.positions, usize(header.poolVertexCount) * 3))
        {
            log = "truncated (members/parents/roots/positions)";
            std::fclose(file);
            return false;
        }
        if (header.hasNormals &&
            !readVector(file, dag.normals, usize(header.poolVertexCount) * 3))
        {
            log = "truncated (normals)";
            std::fclose(file);
            return false;
        }
        if (header.hasUvs &&
            !readVector(file, dag.uvs, usize(header.poolVertexCount) * 2))
        {
            log = "truncated (uvs)";
            std::fclose(file);
            return false;
        }
        if (!readVector(file, dag.indices, header.poolIndexCount))
        {
            log = "truncated (indices)";
            std::fclose(file);
            return false;
        }
        std::fclose(file);

        // Rehydrate engine structures.
        dag.groups.resize(groups.size());
        for (u32 g = 0; g < groups.size(); ++g)
        {
            DagGroup& dst = dag.groups[g];
            dst.lodLevel = groups[g].lodLevel;
            dst.maxError = groups[g].maxError;
            dst.simplifyError = groups[g].simplifyError;
            dst.boundsSphere = glm::vec4(groups[g].bounds[0], groups[g].bounds[1],
                                         groups[g].bounds[2], groups[g].bounds[3]);
            dst.clusters.assign(memberIndices.begin() + groups[g].memberOffset,
                                memberIndices.begin() + groups[g].memberOffset +
                                    groups[g].memberCount);
            dst.parents.assign(parentIndices.begin() + groups[g].parentOffset,
                               parentIndices.begin() + groups[g].parentOffset +
                                   groups[g].parentCount);
        }
        dag.clusters.resize(clusters.size());
        for (u32 c = 0; c < clusters.size(); ++c)
        {
            DagCluster& dst = dag.clusters[c];
            const ClusterDisk& src = clusters[c];
            dst.boundsSphere = glm::vec4(src.bounds[0], src.bounds[1], src.bounds[2],
                                         src.bounds[3]);
            dst.error = src.error;
            dst.parentError = src.parentError;
            dst.vertexOffset = src.vertexOffset;
            dst.vertexCount = src.vertexCount;
            dst.indexOffset = src.indexOffset;
            dst.indexCount = src.indexCount;
            dst.group = src.group;
            dst.children.assign(childIndices.begin() + src.childOffset,
                                childIndices.begin() + src.childOffset + src.childCount);
            dst.isRoot = src.isRoot != 0;
        }
        dag.sourceVertexCount = header.sourceVertexCount;
        dag.sourceTriangleCount = header.sourceTriangleCount;

        return verifyDagInvariants(dag, log);
    }

} // namespace geometry

KENSHIN_NS_END
