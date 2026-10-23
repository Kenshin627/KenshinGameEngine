#!/usr/bin/env python3
"""GCC -fsyntax-only pass over every project source (portability gate).

Applies GCC's C++20 front end to code MSVC accepts: catches MSVC-only
extensions, missing includes MSVC tolerates, two-phase lookup and
template issues - on a Windows machine with no Linux build available.
POSIX-only #else branches are not reached when the GCC is MinGW (it
defines _WIN32); a real Linux CI build covers those.

Usage: python tools/gcc_syntax_check.py [--gxx PATH] [--vcpkg-include DIR]
  defaults: g++ from $KENSHIN_GXX / PATH / Strawberry Perl's MinGW;
  vcpkg headers from build/windows-msvc-release/vcpkg_installed.
Exit 1 on any compile error.
"""
import argparse, glob, os, shutil, subprocess, sys, concurrent.futures as cf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace(os.sep, '/')
ap = argparse.ArgumentParser()
ap.add_argument('--gxx', default=os.environ.get('KENSHIN_GXX') or shutil.which('g++')
                or r'C:/Strawberry/c/bin/g++.exe')
ap.add_argument('--vcpkg-include', default=ROOT +
                '/build/windows-msvc-release/vcpkg_installed/x64-windows/include')
args = ap.parse_args()
GXX = args.gxx
VCPKG_INC = args.vcpkg_include
if not os.path.exists(GXX) and not shutil.which(GXX):
    sys.exit(f'gcc_syntax_check: g++ not found ({GXX}); pass --gxx')
if not os.path.isdir(VCPKG_INC):
    sys.exit(f'gcc_syntax_check: vcpkg include dir missing: {VCPKG_INC}')
INC = ['engine/include', 'engine/src', 'engine/src/backend/vulkan',
       'ThirdParty', 'ThirdParty/imgui-backends', 'app/src']
DEFS = ['VK_NO_PROTOTYPES', 'IMGUI_IMPL_VULKAN_USE_VOLK',
        'VMA_DYNAMIC_VULKAN_FUNCTIONS=0', 'SPIRV_REFLECT_USE_SYSTEM_SPIRV_H',
        f'KENSHIN_SOURCE_DIR="{ROOT}"', 'KENSHIN_GIT_SHA="gcc"',
        'KENSHIN_VERSION_STRING="0"']

srcs = [p for d in ('engine/src', 'app/src', 'tools', 'tests')
        for p in glob.glob(ROOT + '/' + d + '/**/*.cpp', recursive=True)]

def check(p):
    cmd = [GXX, '-std=c++20', '-fsyntax-only', '-Wall', '-Wextra',
           '-Wno-unknown-pragmas', '-Wno-missing-field-initializers']
    cmd += ['-I' + ROOT + '/' + i for i in INC] + ['-isystem', VCPKG_INC]
    cmd += ['-D' + d for d in DEFS] + [p]
    r = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
    return p, r.returncode, r.stderr

fails = 0
warn_files = 0
with cf.ThreadPoolExecutor(8) as ex:
    for p, rc, err in ex.map(check, srcs):
        rel = os.path.relpath(p, ROOT)
        errors = [l for l in err.splitlines() if ': error:' in l]
        warns = [l for l in err.splitlines() if ': warning:' in l and 'vcpkg_installed' not in l]
        if rc != 0:
            fails += 1
            print('FAIL', rel)
            for l in errors[:6]: print('   ', l[:240])
        elif warns:
            warn_files += 1
            print('warn', rel, len(warns))
            for l in warns[:3]: print('   ', l[:240])
print(f'gcc syntax: {len(srcs)} files, {fails} failing, {warn_files} with warnings')
sys.exit(1 if fails else 0)
