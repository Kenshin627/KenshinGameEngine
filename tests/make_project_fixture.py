"""Writes tests/data/project: the asset-database project the render_project*
ctests open (roadmap 1.3d). Deterministic; rerun after changing it and
commit the output.

  project.kproject            startup scene assets/scenes/main.kscene
  assets/models/spheres.gltf  (+ .bin) mesh/0 a UV sphere (material 0: checker.png),
                              mesh/1 a ground grid (material 1: factors)
  assets/models/checker.png
  assets/textures/swatch.png  red
  assets/materials/painted.kmat  base colour = swatch.png
  ../project_edits/           the hot-reload tests' edits: swatch_green.png,
                              checker_blue.png (a model dependency),
                              painted_blue.kmat, main_moved.kscene
  assets/scenes/main.kscene   ground, sphere A (its own material),
                              sphere B (painted.kmat override)

Every asset has a committed .meta with a fixed Guid, so the scene can
name the model's sub-assets by their derived (RFC 4122 v5) Guids.
"""
import json, math, os, struct, sys, uuid, zlib

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
PROJ = os.path.join(ROOT, 'project')

G_MODEL = uuid.UUID('5d0c6a43-8f1e-4b7a-9c2d-3e4f5a6b7c80')
G_CHECKER = uuid.UUID('5d0c6a43-8f1e-4b7a-9c2d-3e4f5a6b7c81')
G_SWATCH = uuid.UUID('5d0c6a43-8f1e-4b7a-9c2d-3e4f5a6b7c82')
G_PAINTED = uuid.UUID('5d0c6a43-8f1e-4b7a-9c2d-3e4f5a6b7c83')
G_SCENE = uuid.UUID('5d0c6a43-8f1e-4b7a-9c2d-3e4f5a6b7c84')


def sub(asset, key):
    return str(uuid.uuid5(asset, key))


def write(rel, data):
    path = os.path.normpath(os.path.join(PROJ, rel))  # '../' reaches tests/data
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = 'wb' if isinstance(data, bytes) else 'w'
    with open(path, mode, **({} if mode == 'wb' else {'encoding': 'utf-8', 'newline': '\n'})) as f:
        f.write(data)


def png(w, h, pixel):
    raw = b''.join(b'\0' + b''.join(bytes(pixel(x, y)) for x in range(w)) for y in range(h))
    def chunk(t, d):
        return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) +
            chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))


def dumps(doc):
    return json.dumps(doc, indent=2, ensure_ascii=False) + '\n'


def meta(rel, guid, type_, settings):
    write(rel + '.meta', dumps({'format': 'kenshin.meta', 'version': 1, 'guid': str(guid),
                                'type': type_, 'settings': settings}))


TEX = {'color_space': 'srgb', 'normal_map': False, 'mipmaps': True, 'max_size': 0}

# ---- geometry (glTF: Y-up, metres)
def sphere(r, seg, rings):
    pos, nrm, uv, idx = [], [], [], []
    for j in range(rings + 1):
        v = j / rings
        th = v * math.pi
        for i in range(seg + 1):
            u = i / seg
            ph = u * 2 * math.pi
            n = (math.sin(th) * math.cos(ph), math.cos(th), -math.sin(th) * math.sin(ph))
            pos.append(tuple(r * c for c in n))
            nrm.append(n)
            uv.append((u * 2, v))
    for j in range(rings):
        for i in range(seg):
            a = j * (seg + 1) + i
            b = a + seg + 1
            idx += [a, b, a + 1, a + 1, b, b + 1]
    return pos, nrm, uv, idx


def grid(size, n):
    pos, nrm, uv, idx = [], [], [], []
    for j in range(n + 1):
        for i in range(n + 1):
            x, z = (i / n - 0.5) * size, (j / n - 0.5) * size
            pos.append((x, 0.0, z))
            nrm.append((0.0, 1.0, 0.0))
            uv.append((i / n * 4, j / n * 4))
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            b = a + n + 1
            idx += [a, b, a + 1, a + 1, b, b + 1]
    return pos, nrm, uv, idx


def gltf():
    blob = bytearray()
    views, accessors, meshes = [], [], []

    def add(data, fmt, comps, ctype, target, minmax=False):
        while len(blob) % 4:
            blob.append(0)
        off = len(blob)
        for item in data:
            blob.extend(struct.pack('<' + fmt * comps, *(item if comps > 1 else (item,))))
        views.append({'buffer': 0, 'byteOffset': off, 'byteLength': len(blob) - off, 'target': target})
        acc = {'bufferView': len(views) - 1, 'componentType': ctype, 'count': len(data),
               'type': {1: 'SCALAR', 2: 'VEC2', 3: 'VEC3'}[comps]}
        if minmax:
            acc['min'] = [min(p[k] for p in data) for k in range(3)]
            acc['max'] = [max(p[k] for p in data) for k in range(3)]
        accessors.append(acc)
        return len(accessors) - 1

    for name, (pos, nrm, uv, idx), mat in (('Sphere', sphere(0.5, 48, 24), 0),
                                           ('Ground', grid(5.0, 16), 1)):
        p = add(pos, 'f', 3, 5126, 34962, True)
        n = add(nrm, 'f', 3, 5126, 34962)
        t = add(uv, 'f', 2, 5126, 34962)
        i = add(idx, 'H', 1, 5123, 34963)
        meshes.append({'name': name, 'primitives': [
            {'attributes': {'POSITION': p, 'NORMAL': n, 'TEXCOORD_0': t}, 'indices': i, 'material': mat}]})
    write('assets/models/spheres.bin', bytes(blob))
    return dumps({
        'asset': {'version': '2.0', 'generator': 'KENSHIN make_project_fixture.py'},
        'scene': 0,
        'scenes': [{'nodes': [0, 1]}],
        'nodes': [{'name': 'Sphere', 'mesh': 0, 'translation': [0.0, 0.5, 0.0]},
                  {'name': 'Ground', 'mesh': 1}],
        'meshes': meshes,
        'materials': [
            {'name': 'Checker', 'pbrMetallicRoughness': {
                'baseColorTexture': {'index': 0}, 'metallicFactor': 0.0, 'roughnessFactor': 0.45}},
            {'name': 'Ground', 'pbrMetallicRoughness': {
                'baseColorFactor': [0.55, 0.55, 0.58, 1.0], 'metallicFactor': 0.0,
                'roughnessFactor': 0.85}}],
        'textures': [{'source': 0, 'sampler': 0}],
        'samplers': [{'magFilter': 9729, 'minFilter': 9987, 'wrapS': 10497, 'wrapT': 10497}],
        'images': [{'uri': 'checker.png'}],
        'buffers': [{'byteLength': len(blob), 'uri': 'spheres.bin'}],
        'bufferViews': views,
        'accessors': accessors,
    })


def entity(guid, name, pos, mesh, material=None, shadows=True):
    return {'guid': guid, 'name': name, 'parent': None, 'components': {
        'transform': {'position': pos, 'rotation': [0.0, 0.0, 0.0, 1.0], 'scale': [1.0, 1.0, 1.0]},
        'mesh_renderer': {'mesh': mesh, 'material': material, 'visible': True, 'cast_shadows': shadows}}}


def main():
    write('project.kproject', dumps({'format': 'kenshin.project', 'version': 1, 'name': 'ProjectFixture',
                                     'startup_scene': 'assets/scenes/main.kscene'}))
    write('assets/models/spheres.gltf', gltf())
    meta('assets/models/spheres.gltf', G_MODEL, 'model',
         {'split_translucent': True, 'condition_layers': True, 'double_sided': True, 'compress_pages': True})
    write('assets/models/checker.png',
          png(64, 64, lambda x, y: (235, 235, 235) if (x // 8 + y // 8) % 2 == 0 else (40, 44, 52)))
    meta('assets/models/checker.png', G_CHECKER, 'texture', TEX)
    write('assets/textures/swatch.png', png(16, 16, lambda x, y: (200, 40, 30)))
    meta('assets/textures/swatch.png', G_SWATCH, 'texture', TEX)
    # The hot-reload edits (tests/data/project_edits).
    write('../project_edits/swatch_green.png', png(16, 16, lambda x, y: (40, 180, 60)))
    write('../project_edits/checker_blue.png',
          png(64, 64, lambda x, y: (60, 110, 230) if (x // 16 + y // 16) % 2 == 0 else (250, 220, 60)))
    write('assets/materials/painted.kmat', dumps({
        'format': 'kenshin.material', 'version': 1, 'base_color': [1.0, 1.0, 1.0, 1.0], 'metallic': 0.0,
        'roughness': 0.35, 'emissive': [0.0, 0.0, 0.0], 'alpha_mode': 'opaque', 'alpha_cutoff': 0.5,
        'double_sided': False, 'normal_scale': 1.0, 'occlusion_strength': 1.0,
        'textures': {'base_color': {'texture': str(G_SWATCH), 'wrap_u': 'repeat', 'wrap_v': 'repeat',
                                    'filter': 'linear'}}}))
    meta('assets/materials/painted.kmat', G_PAINTED, 'material', {})
    sphere_mesh, ground_mesh = sub(G_MODEL, 'mesh/0'), sub(G_MODEL, 'mesh/1')
    write('assets/scenes/main.kscene', dumps({'format': 'kenshin.scene', 'version': 1, 'entities': [
        entity('5d0c6a43-8f1e-4b7a-9c2d-3e4f5a6b7d01', 'Ground', [0.0, 0.0, 0.0], ground_mesh, None, False),
        entity('5d0c6a43-8f1e-4b7a-9c2d-3e4f5a6b7d02', 'Sphere A', [-0.8, 0.0, 0.5], sphere_mesh),
        entity('5d0c6a43-8f1e-4b7a-9c2d-3e4f5a6b7d03', 'Sphere B', [0.8, 0.0, 0.5], sphere_mesh,
               str(G_PAINTED)),
    ]}))
    meta('assets/scenes/main.kscene', G_SCENE, 'scene', {})
    # painted.kmat tinted blue and shinier (a material-only edit).
    write('../project_edits/painted_blue.kmat', dumps({
        'format': 'kenshin.material', 'version': 1, 'base_color': [0.35, 0.55, 1.0, 1.0], 'metallic': 0.0,
        'roughness': 0.2, 'emissive': [0.0, 0.0, 0.0], 'alpha_mode': 'opaque', 'alpha_cutoff': 0.5,
        'double_sided': False, 'normal_scale': 1.0, 'occlusion_strength': 1.0,
        'textures': {'base_color': {'texture': str(G_SWATCH), 'wrap_u': 'repeat', 'wrap_v': 'repeat',
                                    'filter': 'linear'}}}))
    # Sphere B moved up and back to its own (checker) material.
    write('../project_edits/main_moved.kscene', dumps({'format': 'kenshin.scene', 'version': 1, 'entities': [
        entity('5d0c6a43-8f1e-4b7a-9c2d-3e4f5a6b7d01', 'Ground', [0.0, 0.0, 0.0], ground_mesh, None, False),
        entity('5d0c6a43-8f1e-4b7a-9c2d-3e4f5a6b7d02', 'Sphere A', [-0.8, 0.0, 0.5], sphere_mesh),
        entity('5d0c6a43-8f1e-4b7a-9c2d-3e4f5a6b7d03', 'Sphere B', [0.8, 0.4, 1.1], sphere_mesh),
    ]}))
    print('written', PROJ)


if __name__ == '__main__':
    sys.exit(main())
