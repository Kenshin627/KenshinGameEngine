#!/usr/bin/env python3
"""Roadmap 1.3e: the Khronos glTF sample gate.

Imports the curated glTF-Sample-Assets subset (tools/gltf_samples.sparse,
fetched by tools/fetch_gltf_samples.py) through the asset database and
renders every model headless with KENSHINDemo's project mode, front view
(the Khronos screenshots' side). Writes, under --out:

  project/            a throwaway project: assets/<Model>/ = Models/<Model>/glTF
  renders/<Model>.bmp our frame; logs/<Model>.log the run's log
  contact_sheet.png   per model: ours | the Khronos reference screenshot
  summary.md          per model: import notes, warnings, validation errors

The sheet is compared by eye; each model is then classed in the roadmap.
Our renders, downscaled to 320 px wide, are the regression references
(tests/data/khronos/<Model>.png):
  --update-refs  writes them (after reviewing the sheet)
  --check        compares against them; exit 1 on a failed run, a
                 validation error or a differing render (the khronos_gate
                 ctest, registered when KENSHIN_GLTF_SAMPLES is set)

Usage: python tools/khronos_gate.py [--samples DIR] [--out DIR] [--demo EXE]
                                    [--models A,B] [--frame yaw,pitch,scale]
                                    [--update-refs | --check]
Needs Pillow for the sheet and the references (the renders do not).
"""
import argparse, json, os, re, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace(os.sep, '/')
BUILD = ROOT + '/build/windows-msvc-release'

ap = argparse.ArgumentParser()
ap.add_argument('--samples', default=os.environ.get('KENSHIN_GLTF_SAMPLES', 'C:/code/glTF-Sample-Assets'))
ap.add_argument('--out', default=BUILD + '/khronos')
ap.add_argument('--demo', default=BUILD + '/app/Release/KENSHINDemo.exe')
ap.add_argument('--models', default='')
ap.add_argument('--frame', default='', help='one view for every model (default: VIEWS)')
ap.add_argument('--frames', type=int, default=16)
ap.add_argument('--refs', default=ROOT + '/tests/data/khronos')
mode = ap.add_mutually_exclusive_group()
mode.add_argument('--update-refs', action='store_true')
mode.add_argument('--check', action='store_true')
args = ap.parse_args()


def models_from_patterns():
    names = []
    with open(ROOT + '/tools/gltf_samples.sparse', encoding='utf-8') as f:
        for line in f:
            m = re.match(r'/Models/([^/]+)/glTF/$', line.strip())
            if m:
                names.append(m.group(1))
    return names


MODELS = [m for m in args.models.split(',') if m] or models_from_patterns()

# Per model, the view of its Khronos screenshot: --frame yaw,pitch,scale
# auto-frames the bounds (yaw 180 = the glTF front, +Z there / -Y here;
# scale x the bounding radius), --cam yaw,pitch,dist,tx,ty,tz is fixed.
FRONT = ['--frame', '180,5,1.6']
PLATE = ['--frame', '180,0,1.25']  # flat test boards facing the viewer
THREE_QUARTER = ['--frame', '145,25,2.0']
VIEWS = {
    'Box': THREE_QUARTER, 'BoxTextured': THREE_QUARTER, 'OrientationTest': THREE_QUARTER,
    'MultiUVTest': THREE_QUARTER,
    'Duck': ['--frame', '0,5,1.6'],
    'ABeautifulGame': ['--frame', '200,25,0.9'],
    'Sponza': ['--cam', '90,5,1,0,0,2'],  # inside, down the atrium
    'MetalRoughSpheres': PLATE, 'NormalTangentTest': PLATE, 'NormalTangentMirrorTest': PLATE,
    'TextureCoordinateTest': PLATE, 'TextureSettingsTest': PLATE, 'VertexColorTest': PLATE,
    'TextureTransformTest': PLATE, 'AlphaBlendModeTest': PLATE, 'SimpleSparseAccessor': PLATE,
}
OUT = args.out.replace(os.sep, '/')
PROJ = OUT + '/project'
for d in (PROJ + '/assets', OUT + '/renders', OUT + '/logs'):
    os.makedirs(d, exist_ok=True)
if not os.path.exists(PROJ + '/project.kproject'):
    with open(PROJ + '/project.kproject', 'w', encoding='utf-8', newline='\n') as f:
        json.dump({'format': 'kenshin.project', 'version': 1, 'name': 'KhronosGate',
                   'startup_scene': ''}, f, indent=2)
        f.write('\n')

# ---- sources: copy each model's plain glTF folder (once; the database
# writes .meta files next to its sources, never into the sample checkout)
for m in MODELS:
    src = f'{args.samples}/Models/{m}/glTF'
    dst = f'{PROJ}/assets/{m}'
    if not os.path.isdir(src):
        sys.exit(f'khronos_gate: {src} missing (run tools/fetch_gltf_samples.py)')
    if not os.path.isdir(dst):
        shutil.copytree(src, dst)


def guid_of(rel):
    with open(f'{PROJ}/{rel}.meta', encoding='utf-8') as f:
        return json.load(f)['guid']


# ---- render
rows = []
for i, m in enumerate(MODELS):
    scene = f'assets/{m}/{m}.gltf'
    shot = f'{OUT}/renders/{m}.bmp'
    if os.path.exists(shot):
        os.remove(shot)
    ini = os.path.join(os.path.dirname(args.demo), 'imgui.ini')
    if os.path.exists(ini):
        os.remove(ini)
    view = ['--frame', args.frame] if args.frame else VIEWS.get(m, FRONT)
    cmd = [args.demo, '--project', PROJ, '--scene', scene, '--frames', str(args.frames),
           '--size', '1280x720', *view, '--watch-interval', '-1', '--screenshot', shot]
    # The first run imports the whole project (minutes); later ones are current.
    r = subprocess.run(cmd, cwd=OUT, capture_output=True, text=True, errors='replace', timeout=3600)
    log = r.stdout + r.stderr
    with open(f'{OUT}/logs/{m}.log', 'w', encoding='utf-8') as f:
        f.write(log)
    notes = []
    try:
        with open(f'{PROJ}/.kenshin/cache/{guid_of(scene)}/import.json', encoding='utf-8') as f:
            notes = json.load(f).get('notes', [])
    except (OSError, KeyError, ValueError):
        notes = ['(no import record)']
    # Every run opens the whole project: another model's failed import
    # is that model's problem, not this one's.
    warns = sorted({re.sub(r'^\[[^]]*\] ', '', l) for l in log.splitlines()
                    if ('[warning]' in l or '[error]' in l) and 'validation' not in l and
                    not ('import failed: assets/' in l and f'assets/{m}/' not in l)})
    validation = sum(1 for l in log.splitlines() if '[vulkan][validation]' in l)
    built = re.search(r'\[world\] built: ([^\n]*)', log)
    rows.append({'model': m, 'rc': r.returncode, 'shot': os.path.exists(shot), 'notes': notes,
                 'warnings': warns, 'validation': validation,
                 'built': built.group(1) if built else '-'})
    print(f'[{i + 1}/{len(MODELS)}] {m}: rc {r.returncode}, {validation} validation, '
          f'{len(warns)} warnings, {len(notes)} notes', flush=True)

# ---- summary
with open(f'{OUT}/summary.md', 'w', encoding='utf-8', newline='\n') as f:
    f.write('# Khronos sample gate\n\n')
    for row in rows:
        f.write(f"## {row['model']}\n\n* run: rc {row['rc']}, screenshot {'yes' if row['shot'] else 'NO'}, "
                f"validation messages {row['validation']}\n* scene: {row['built']}\n")
        for n in row['notes']:
            f.write(f'* note: {n}\n')
        for w in row['warnings']:
            f.write(f'* log: {w}\n')
        f.write('\n')

# ---- contact sheet: ours | reference, one row per model
try:
    from PIL import Image, ImageDraw
except ImportError:
    print('khronos_gate: Pillow missing - no contact sheet or references (pip install pillow)')
    sys.exit(1 if (args.check or args.update_refs) else 0)
H = 300
cells = []
for row in rows:
    m = row['model']
    ours = Image.open(f'{OUT}/renders/{m}.bmp').convert('RGB') if row['shot'] else Image.new('RGB', (400, H))
    shots = f'{args.samples}/Models/{m}/screenshot'
    ref_name = next((n for n in ('screenshot.png', 'screenshot.jpg') if os.path.exists(f'{shots}/{n}')), None)
    ref = Image.open(f'{shots}/{ref_name}').convert('RGB') if ref_name else Image.new('RGB', (H, H))
    scale = lambda im: im.resize((max(1, round(im.width * H / im.height)), H))
    cells.append((m, scale(ours), scale(ref)))
W = max(a.width + b.width for _, a, b in cells) + 20
sheet = Image.new('RGB', (W, len(cells) * (H + 24)), (32, 32, 36))
draw = ImageDraw.Draw(sheet)
for k, (m, a, b) in enumerate(cells):
    y = k * (H + 24)
    draw.text((6, y + 5), f'{m}   (left: KENSHIN, right: Khronos reference)', fill=(230, 230, 230))
    sheet.paste(a, (0, y + 24))
    sheet.paste(b, (a.width + 20, y + 24))
sheet.save(f'{OUT}/contact_sheet.png')
print(f'written {OUT}/contact_sheet.png and summary.md')


# ---- regression references: 320 px wide, box-filtered (sub-pixel run-
# to-run noise - Sponza's depth ties - averages out)
def thumb(path):
    im = Image.open(path).convert('RGB')
    return im.resize((320, max(1, round(320 * im.height / im.width))), Image.BOX)


failed = []
if args.update_refs or args.check:
    os.makedirs(args.refs, exist_ok=True)
for row in rows:
    m = row['model']
    ref = f'{args.refs}/{m}.png'
    if args.update_refs:
        if row['shot']:
            thumb(f'{OUT}/renders/{m}.bmp').save(ref)
        continue
    if not args.check:
        continue
    if row['rc'] != 0 or not row['shot'] or row['validation'] > 0:
        failed.append(f"{m}: run rc {row['rc']}, screenshot {row['shot']}, "
                      f"{row['validation']} validation messages")
        continue
    if not os.path.exists(ref):
        failed.append(f'{m}: no reference {ref}')
        continue
    a, b = thumb(f'{OUT}/renders/{m}.bmp'), Image.open(ref).convert('RGB')
    if a.size != b.size:
        failed.append(f'{m}: size {a.size} vs reference {b.size}')
        continue
    pa, pb = a.load(), b.load()
    off = sum(1 for y in range(a.height) for x in range(a.width)
              if max(abs(c - d) for c, d in zip(pa[x, y], pb[x, y])) > 4)
    budget = a.width * a.height // 200  # 0.5%
    print(f'{m}: {off} of {a.width * a.height} thumbnail pixels differ (budget {budget})')
    if off > budget:
        failed.append(f'{m}: {off} pixels differ from the reference (budget {budget})')
if args.update_refs:
    print(f'references written to {args.refs}')
if failed:
    print('khronos_gate FAILED:\n  ' + '\n  '.join(failed))
    sys.exit(1)
if args.check:
    print(f'khronos_gate: {len(rows)} models match their references')
