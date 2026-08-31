#include "kenshin/assets/ObjLoader.h"

#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <sstream>
#include <tuple>

KENSHIN_NS_BEGIN

namespace assets
{

    namespace
    {
        /// Split an OBJ face vertex reference ("v", "v/vt", "v//vn", "v/vt/vn")
        /// into 1-based indices; negatives are relative. 0 = not present.
        bool parseFaceRef(const std::string& token, const i64 rawCounts[3], i64 out[3])
        {
            i64 parts[3] = {0, 0, 0};
            int part = 0;
            size_t start = 0;
            for (size_t i = 0; i <= token.size(); ++i)
            {
                if (i == token.size() || token[i] == '/')
                {
                    if (part >= 3)
                        return false;
                    if (i > start)
                        parts[part] = std::stoll(token.substr(start, i - start));
                    ++part;
                    start = i + 1;
                }
            }
            for (int c = 0; c < 3; ++c)
            {
                i64 value = parts[c];
                if (value == 0)
                {
                    out[c] = 0;
                }
                else if (value < 0)
                {
                    out[c] = rawCounts[c] + value + 1; // relative to end
                    if (out[c] < 1)
                        return false;
                }
                else
                {
                    out[c] = value;
                }
            }
            return true;
        }

        void parseMtl(const std::string& path, MeshData::Material& material)
        {
            std::FILE* file = nullptr;
            if (fopen_s(&file, path.c_str(), "r") != 0 || !file)
                return;
            char line[512];
            while (std::fgets(line, sizeof(line), file))
            {
                float x = 0, y = 0, z = 0;
                char buf[256] = "";
                if (sscanf_s(line, "Ka %f %f %f", &x, &y, &z) == 3)
                    material.ka = {x, y, z};
                else if (sscanf_s(line, "Kd %f %f %f", &x, &y, &z) == 3)
                    material.kd = {x, y, z};
                else if (sscanf_s(line, "map_Kd %255s", buf,
                                  static_cast<unsigned>(sizeof(buf))) == 1)
                {
                    material.diffuseTexture = buf;
                }
            }
            std::fclose(file);
        }
    } // namespace

    bool loadObj(const std::string& path, MeshData& out, std::string& log)
    {
        std::FILE* file = nullptr;
        if (fopen_s(&file, path.c_str(), "r") != 0 || !file)
        {
            log = "cannot open file: " + path;
            return false;
        }

        std::vector<f32> rawPositions; // 3 per v
        std::vector<f32> rawNormals;   // 3 per vn
        std::vector<f32> rawUvs;       // 2 per vt

        // Deduplicated output assembly: (vIndex, vtIndex, vnIndex) -> out index.
        std::map<std::tuple<i64, i64, i64>, u32> vertexMap;

        MeshData mesh;
        u32 faceCount = 0;
        char line[1024];
        while (std::fgets(line, sizeof(line), file))
        {
            if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
                continue;

            float x = 0, y = 0, z = 0;
            if (sscanf_s(line, "v %f %f %f", &x, &y, &z) == 3)
            {
                rawPositions.insert(rawPositions.end(), {x, y, z});
            }
            else if (sscanf_s(line, "vn %f %f %f", &x, &y, &z) == 3)
            {
                rawNormals.insert(rawNormals.end(), {x, y, z});
            }
            else if (sscanf_s(line, "vt %f %f", &x, &y) == 2)
            {
                rawUvs.insert(rawUvs.end(), {x, y});
            }
            else if (line[0] == 'f' && (line[1] == ' ' || line[1] == '\t'))
            {
                std::istringstream stream(line + 2);
                std::vector<std::string> refs;
                std::string token;
                while (stream >> token)
                    refs.push_back(token);
                if (refs.size() < 3)
                    continue;
                faceCount++;

                const i64 rawCounts[3] = {
                    static_cast<i64>(rawPositions.size() / 3),
                    static_cast<i64>(rawUvs.size() / 2),
                    static_cast<i64>(rawNormals.size() / 3),
                };

                // Fan triangulation: (0, i-1, i).
                i64 first[3] = {0, 0, 0};
                i64 prev[3] = {0, 0, 0};
                bool haveFirst = false;
                for (size_t i = 0; i < refs.size(); ++i)
                {
                    i64 ref[3] = {0, 0, 0};
                    if (!parseFaceRef(refs[i], rawCounts, ref))
                    {
                        std::snprintf(nullptr, 0, "%s", ""); // no-op
                        log = "bad face reference: " + refs[i];
                        std::fclose(file);
                        return false;
                    }
                    if (i == 0)
                    {
                        std::memcpy(first, ref, sizeof(first));
                        haveFirst = true;
                        continue;
                    }
                    if (i == 1)
                    {
                        std::memcpy(prev, ref, sizeof(prev));
                        continue;
                    }

                    i64 tri[3][3];
                    std::memcpy(tri[0], first, sizeof(first));
                    std::memcpy(tri[1], prev, sizeof(prev));
                    std::memcpy(tri[2], ref, sizeof(ref));

                    for (int corner = 0; corner < 3; ++corner)
                    {
                        const auto key = std::make_tuple(tri[corner][0],
                                                         tri[corner][1],
                                                         tri[corner][2]);
                        auto it = vertexMap.find(key);
                        if (it != vertexMap.end())
                        {
                            mesh.indices.push_back(it->second);
                            continue;
                        }

                        const i64 vi = tri[corner][0] - 1;
                        if (vi < 0 || static_cast<usize>(vi) * 3 + 2 >= rawPositions.size())
                        {
                            log = "position index out of range";
                            std::fclose(file);
                            return false;
                        }
                        const u32 index = mesh.vertexCount();
                        mesh.positions.insert(mesh.positions.end(),
                                              {rawPositions[vi * 3 + 0],
                                               rawPositions[vi * 3 + 1],
                                               rawPositions[vi * 3 + 2]});

                        const i64 ti = tri[corner][1];
                        if (ti > 0 && static_cast<usize>(ti - 1) * 2 + 1 < rawUvs.size())
                            mesh.uvs.insert(mesh.uvs.end(),
                                            {rawUvs[(ti - 1) * 2 + 0],
                                             rawUvs[(ti - 1) * 2 + 1]});
                        else
                            mesh.uvs.insert(mesh.uvs.end(), {0.0f, 0.0f});

                        const i64 ni = tri[corner][2];
                        if (ni > 0 && static_cast<usize>(ni - 1) * 3 + 2 < rawNormals.size())
                            mesh.normals.insert(mesh.normals.end(),
                                                {rawNormals[(ni - 1) * 3 + 0],
                                                 rawNormals[(ni - 1) * 3 + 1],
                                                 rawNormals[(ni - 1) * 3 + 2]});
                        else
                            mesh.normals.insert(mesh.normals.end(), {0.0f, 1.0f, 0.0f});

                        vertexMap[key] = index;
                        mesh.indices.push_back(index);
                    }
                    std::memcpy(prev, ref, sizeof(prev));
                }
                (void)haveFirst;
            }
            else if (line[0] == 'm' && std::strncmp(line, "mtllib", 6) == 0)
            {
                char buf[256] = "";
                if (sscanf_s(line, "mtllib %255s", buf,
                             static_cast<unsigned>(sizeof(buf))) == 1)
                {
                    const auto dir = std::filesystem::path(path).parent_path();
                    parseMtl((dir / buf).string(), mesh.material);
                }
            }
        }
        std::fclose(file);

        if (mesh.indices.empty())
        {
            log = "no faces found";
            return false;
        }

        out = std::move(mesh);
        log.clear();
        return true;
    }

} // namespace assets

KENSHIN_NS_END
