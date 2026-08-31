import re

p = 'app/src/main.cpp'
s = open(p, encoding='utf-8').read()

# includes
old = '#include "kenshin/manager/ShaderManager.h"'
new = '''#include "kenshin/assets/ObjLoader.h"
#include "kenshin/assets/ProceduralMesh.h"
#include "kenshin/manager/ShaderManager.h"'''
assert old in s
s = s.replace(old, new)

# args: --mesh
old = """        std::string cam;                             // parsed by Camera (S1.5)
        std::string screenshot;"""
new = """        std::string cam;                             // parsed by Camera (S1.5)
        std::string mesh = "cube";                   // cube | sphere | knot | <path.obj>
        std::string screenshot;"""
assert old in s
s = s.replace(old, new)

old = """            else if (std::strcmp(argv[i], "--cam") == 0 && i + 1 < argc)
                args.cam = argv[++i];"""
new = old + """
            else if (std::strcmp(argv[i], "--mesh") == 0 && i + 1 < argc)
                args.mesh = argv[++i];"""
assert old in s
s = s.replace(old, new)

# mesh load helper before drawUi namespace
anchor = """namespace
{
    struct UiState"""
helper = """namespace
{
    /// Load the demo mesh (--mesh cube|sphere|knot|<path.obj>).
    bool loadDemoMesh(const std::string& name, kenshin::assets::MeshData& out,
                      std::string& log)
    {
        if (name == "cube")
        {
            // Procedural cube (8 verts, CCW) with per-face normals.
            out.positions = {
                -0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  0.5f, 0.5f, -0.5f, -0.5f, 0.5f, -0.5f,
                -0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f, 0.5f,  0.5f, -0.5f, 0.5f,  0.5f,
            };
            out.normals.assign(24 * 3, 0.0f);
            out.uvs.assign(24 * 2, 0.0f);
            const int quads[6][4] = {
                {0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4},
                {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7},
            };
            const float normals[6][3] = {
                {0, 0, -1}, {0, 0, 1}, {0, -1, 0}, {1, 0, 0}, {0, 1, 0}, {-1, 0, 0},
            };
            kenshin::u32 vertex = 8;
            for (int q = 0; q < 6; ++q)
            {
                for (int c = 0; c < 4; ++c)
                {
                    const int src = quads[q][c];
                    for (int k = 0; k < 3; ++k)
                        out.positions.push_back(out.positions[src * 3 + k]);
                    for (int k = 0; k < 3; ++k)
                        out.normals.push_back(normals[q][k]);
                    out.uvs.push_back(0.0f);
                    out.uvs.push_back(0.0f);
                    ++vertex;
                }
                // Two CCW triangles from the fan (v0, v1, v2) and (v0, v2, v3).
                const kenshin::u32 base = vertex - 4;
                out.indices.insert(out.indices.end(), {base, base + 1, base + 2,
                                                       base, base + 2, base + 3});
            }
            return true;
        }
        if (name == "sphere")
        {
            out = kenshin::assets::createUvSphere(64, 32, 1.0f);
            return true;
        }
        if (name == "knot")
        {
            out = kenshin::assets::createTorusKnot(2, 3, 256, 32, 1.0f, 0.32f);
            return true;
        }
        return kenshin::assets::loadObj(name, out, log);
    }
} // namespace

namespace
{
    struct UiState"""
assert anchor in s
s = s.replace(anchor, helper, 1)

open(p, 'w', encoding='utf-8', newline='\n').write(s)
print("app part1 ok")
