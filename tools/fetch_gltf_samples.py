#!/usr/bin/env python3
"""Roadmap 1.3e: fetch the curated KhronosGroup/glTF-Sample-Assets subset.

A partial clone (blobs on demand) with a sparse checkout of the patterns
in tools/gltf_samples.sparse, at the pinned commit below: only each
model's plain glTF variant, its reference screenshot and its license
files are downloaded (212.8 MB at the pinned commit). The checkout lives
outside the repository; tools/khronos_gate.py reads it.

Usage: python tools/fetch_gltf_samples.py [DIR]   (default C:/code/glTF-Sample-Assets,
       or $KENSHIN_GLTF_SAMPLES)
Rerunning updates an existing checkout to the pinned commit and patterns.
"""
import os, subprocess, sys

REPO = 'https://github.com/KhronosGroup/glTF-Sample-Assets.git'
COMMIT = '8f492390ddc7a2ae221a1369ef560589cf8c70fb'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERNS = os.path.join(ROOT, 'tools', 'gltf_samples.sparse')
DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('KENSHIN_GLTF_SAMPLES',
                                                           'C:/code/glTF-Sample-Assets')


def git(*a, **kw):
    print('git', ' '.join(a), flush=True)
    subprocess.run(['git', *a], check=True, **kw)


if not os.path.isdir(os.path.join(DIR, '.git')):
    git('clone', '--filter=blob:none', '--no-checkout', REPO, DIR)
with open(PATTERNS, 'rb') as f:  # stdin, not a shell redirect: works on every shell
    git('-C', DIR, 'sparse-checkout', 'set', '--no-cone', '--stdin', input=f.read())
git('-C', DIR, 'fetch', 'origin', COMMIT)
git('-C', DIR, 'checkout', '--detach', COMMIT)

models = sorted(os.listdir(os.path.join(DIR, 'Models')))
print(f'{len(models)} models in {DIR}/Models')
