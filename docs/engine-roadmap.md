# KENSHIN → production game engine: gap analysis and roadmap

Goal: a simple, production-grade 3D game engine for Windows, macOS and
Linux (Ubuntu 24.04+), built on this repository. An editor authors
games; an exporter produces a distributable player; every editor action
is drivable by an AI agent without human intervention.

This document is the single source of truth for the plan. Each step lists
its **test & observation** gate; a step is done only when its gate passes
and the result is recorded here.

---

## 1. What exists today (survey, 2026-10-23)

| Area | State |
|---|---|
| Build | CMake + vcpkg (glm, spdlog, catch2, volk, VMA, imgui, spirv-reflect); Slang vendored as **Windows DLLs only**; MSVC-only flags/presets |
| Platform | **Win32 only**: hand-written `core/Window` (HWND), Win32 VK codes in `core/Input`, `imgui_impl_win32`, `VK_USE_PLATFORM_WIN32_KHR`, SEH crash filter, `Sleep`/`IsIconic` |
| RHI | Vulkan 1.4 backend (sync2, dynamic rendering, timeline semaphores, async compute, dedicated transfer queue), validation-clean |
| Geometry | Nanite-style virtual geometry: offline `kgeo_builder` (meshopt clusterization + DAG), paged/quantized `.kgeo`, streaming, GPU culling (HZB two-pass), SW + HW raster into a **64-bit atomic VisBuffer**, imposters |
| Shading | Deferred: GBuffer (normal+depth / albedo+RM / emissive), GGX PBR, normal maps, AO maps, emissive, translucent forward pass, directional VSM-style clipmap shadows (3 levels, PCF), HDRI IBL (SH diffuse + mip-chain specular, sun extraction), TAA, bloom, histogram auto exposure, UE filmic tonemap, environment background |
| Assets | glTF/GLB loader (hardened vs hostile files); materials MR/normal/AO/emissive/BLEND; textures re-encoded as **24-bit BMP**, one shared texture array |
| App | `app/src/main.cpp` (2350 lines): demo scene assembly, pass orchestration, debug panel, ~60 research CLI flags |
| Tests | 76 Catch2 cases (CPU/GPU differential tests), 4 GPU render regressions (`bmpdiff`), Windows only, no CI |

## 2. Missing modules

1. **Cross-platform platform layer**: window, input, gamepad, file dialogs, paths, crash handling for macOS/Linux; per-OS Slang binaries; MoltenVK on macOS.
2. **ECS + scene model**: entities, transform hierarchy, components, prefabs (EnTT was already chosen as the ECS for "later").
3. **Serialization**: scenes, prefabs, project settings, asset metadata (versioned, diff-friendly text).
4. **Asset system**: project folder, asset database with GUIDs, in-editor glTF import (the builder is a separate exe today), cooked cache, hot reload.
5. **Physics** (none).
6. **Scripting** (none).
7. **Audio** (none).
8. **Editor**: hierarchy, inspector, asset browser, gizmos, play mode, undo/redo, project management. Today only a debug ImGui overlay exists.
9. **AI-agent control surface**: none.
10. **Runtime player + export/packaging**: none.
11. **GTAO/SSAO**: none (AO maps only).
12. **CI** for three OSes: none.

## 3. Existing areas that are not production-grade

| # | Problem | Why it matters |
|---|---|---|
| W1 | Renderer orchestration lives in the demo `main.cpp`; hard-coded demo lights (2 point lights), camera, ground | A game needs `renderer.render(scene, camera)`, not a 2350-line demo |
| W2 | TAA reprojects with **camera motion only** | Every moving object (physics, scripts) ghosts |
| W3 | Shadow page cache keyed on light/LOD/streaming only | Moving casters leave stale shadows |
| W4 | Shadow clipmap centered on a **fixed CLI camera target**, not the camera | Shadows vanish as soon as the player walks away |
| W5 | Lights: one directional + 2 constant, unshadowed point lights | No light entities, no spot lights, no N-light support |
| W6 | Textures: 24-bit BMP re-encode, **no alpha**, all textures must share dimensions, box-filter mips, no compression | Real assets fail to load (seen: 32×32 vs 64×64 rejection); alpha-cutout foliage impossible |
| W7 | glTF: alpha MASK ignored; no second UV, vertex colors, `KHR_texture_transform`, skinning/animation | Many Poly Haven / Sketchfab assets render wrong |
| W8 | Specular IBL prefilter = blit mip chain (box), not GGX-convolved | Rough metals reflect the wrong lobe |
| W9 | Soft shadows = 3×3 PCF + ramp (fixed-width penumbra) | Requirement says soft shadow maps (contact-hardening) |
| W10 | Translucency only for mesh 0 of the loaded asset | Translucent objects in a scene don't render |
| W11 | VisBuffer requires `shaderBufferInt64Atomics` | Not available on MoltenVK for Apple M1/Intel Macs: renderer would not run |
| W12 | Offline-only asset building; `.kgeo` texture paths relative to output folder | No editor import workflow |
| W13 | Tests are Windows-only, render regressions cover one procedural knot | Not "fully tested" across platforms/features |

## 4. Technology decisions (proposed)

| Need | Choice | Reason |
|---|---|---|
| Window/input/gamepad/dialogs | **SDL3** | One API for all 3 OSes, Vulkan surface creation, gamepads, native file dialogs; zlib license |
| ECS | **EnTT** | Header-only, fast, proven (Minecraft); already the agreed future ECS |
| Serialization | **nlohmann/json** | Text scenes/prefabs are diff-able and trivially agent-editable |
| Physics | **Jolt Physics** | Production-grade (Horizon Forbidden West), deterministic option, MIT, CMake-native, cross-platform |
| Scripting | **Lua 5.4 + sol2** | Small, embeddable, sandboxable; sol2 gives safe typed bindings |
| Audio | **miniaudio** | As specified; single header, null backend for headless tests |
| Gizmos | **ImGuizmo** | Standard ImGui gizmo (translate/rotate/scale, snapping) |
| Texture compression | **KTX2 + Basis Universal** (later step) | Cross-platform GPU-compressed textures for export |

All of the above are vcpkg ports, so they come in through the existing manifest.

## 5. Architecture

```
engine/  (static lib, kenshin_engine)   core, platform(SDL3), rhi(Vulkan), renderer,
                                        ecs, scene, assets, physics, script, audio
editor/  (exe, kenshin_editor)          ImGui editor + command system + agent API server
player/  (exe, kenshin_player)          runtime only: loads a cooked project pack
tools/   kgeo_builder (kept as CLI over the same import library), bmpdiff, kenshin CLI
```

* **One reflection description per component** (fields, types, ranges) drives
  serialization, the inspector, the Lua bindings and the agent API, so the
  four never drift apart.
* **Every editor mutation is a Command** (do/undo). The UI, undo stack and
  agent API all go through the same commands.
* **Agent API**: JSON-RPC 2.0 over localhost TCP, plus a `kenshin` CLI that
  wraps it, plus a `--headless` editor mode (offscreen rendering, no window).
  Because scenes, prefabs and scripts are text files, an agent can also edit
  files directly and call `project.reload`.
* **Export** = cook assets → pack → copy `kenshin_player` + runtime libraries
  (+ `.app` bundle with MoltenVK on macOS).

## 6. Plan

Each step ends with a commit, its test gate passing, and an entry in §7.

### Phase 0 — Foundation
- **0.1 Platform layer on SDL3** (window, events, input codes, DPI, gamepad,
  dialogs; imgui SDL3 backend; Vulkan surface via SDL).
  *Gate*: all 4 render regressions byte-identical; input unit tests; app
  smoke with validation (resize/minimize/close).
- **0.2 Portable build**: GCC/Clang/AppleClang presets; per-OS Slang;
  per-OS crash handler; MoltenVK portability enumeration.
  *Gate*: clean builds on all three OSes (needs a Linux/macOS path, §8).
- **0.3 CI**: build + unit tests on 3 OSes; GPU tests on Linux via lavapipe.
  *Gate*: green matrix.
- **0.4 `SceneRenderer` facade** moves pass orchestration out of `main.cpp`
  (W1); the demo becomes a thin client.
  *Gate*: render regressions byte-identical; `main.cpp` < 500 lines.

### Phase 1 — Core runtime
- **1.1 ECS (EnTT)**: Transform hierarchy, Name, GUID, MeshRenderer, Light,
  Camera; component reflection.
  *Gate*: unit tests (hierarchy math, reparenting, destruction).
- **1.2 Serialization**: scenes/prefabs/project (versioned JSON, GUID refs).
  *Gate*: round-trip byte-identical; golden files; schema migration test.
- **1.3 Asset system**: project layout, asset DB + `.meta`, in-process glTF
  import (builder as a library), cooked cache, hot reload; texture rewrite
  (W6: PNG/JPEG/KTX2, alpha, any size, sRGB/linear, proper mips).
  *Gate*: import the Khronos glTF sample set (render each, compare to
  references); asset DB unit tests.
- **1.4 Renderer ↔ ECS**: dynamic transforms, per-object motion vectors
  (W2), shadow invalidation for moving casters (W3), camera-following
  clipmap (W4), N directional/point/spot lights (W5), translucency for all
  meshes (W10), alpha MASK (W7), built-in primitives.
  *Gate*: new regression scenes; moving-object TAA ghosting metric;
  moving-caster shadow test.
- **1.5 Input actions + game loop** (fixed-step simulation, variable render,
  interpolation).
  *Gate*: unit tests; frame-pacing measurement.

### Phase 2 — Rendering requirements
- **2.1 GTAO** (UE/XeGTAO port, applied to indirect light). *Gate*: AO
  golden images, per-pass GPU timing budget.
- **2.2 Soft shadows**: PCSS/contact hardening on the clipmap; spot-light
  shadows. *Gate*: penumbra-width test vs. analytic reference.
- **2.3 IBL done properly** (W8): GGX importance-sampled prefilter, BRDF LUT,
  Poly Haven browser/downloader in the editor. *Gate*: furnace test;
  roughness sweep vs. reference.
- **2.4 Non-int64 fallback** (W11): HW raster + depth-tested VisBuffer when
  64-bit atomics are absent. *Gate*: forced fallback on Windows vs. main
  path image diff.

### Phase 3 — Physics (Jolt)
- RigidBody (static/kinematic/dynamic), colliders (box/sphere/capsule/
  convex/mesh), materials, fixed step, interpolation, contacts/triggers,
  raycasts. *Gate*: deterministic simulation tests (drop, stack, restitution,
  triggers); editor-authored scene round-trip.

### Phase 4 — Scripting (Lua 5.4 + sol2)
- Script components (`onStart/onUpdate/onFixedUpdate/onCollision/onDestroy`),
  entity/component API, spawn/destroy (deferred), prefab instantiate, input,
  physics, audio, time; sandbox; errors reported with file:line, never crash;
  hot reload. *Gate*: headless Lua behavior tests.

### Phase 5 — Audio (miniaudio)
- AudioSource/Listener components, 2D/3D, streaming music, buses.
  *Gate*: null-backend tests (decode, mixing, spatial gains).

### Phase 6 — Editor
- Docked UI: hierarchy, inspector (reflection-driven), asset browser,
  viewport + ImGuizmo, play/pause/stop with scene snapshot, undo/redo,
  console, project create/open. *Gate*: command-level tests; UI smoke via
  the agent API; screenshots.

### Phase 7 — AI-agent control
- JSON-RPC API + `kenshin` CLI: project, assets, entities, components,
  scripts, play N seconds, screenshot, logs/state queries, export;
  `api.describe` self-documentation. *Gate*: end-to-end test that builds a
  scene purely through the API and asserts via queries and screenshots.

### Phase 8 — Player + export
- `kenshin_player` (no editor code linked), project pack, export command,
  macOS `.app`. *Gate*: export a sample, run the exported player headless
  N frames, compare screenshot.

### Phase 9 — Acceptance
- Build **Tetris (3D)** and a physics sample **only through the agent API**;
  export and run both. *Gate*: playable builds on all three OSes.

## 7. Results log

(Filled in per step: date, commit, gate evidence.)

### 0.1 Platform layer on SDL3 — done 2026-10-23 (Windows-verified)

* SDL3 3.4.8 (vcpkg, `vulkan` feature) replaces the Win32 window, Win32
  key codes, `imgui_impl_win32` and `VK_KHR_win32_surface`. The window
  hands the backend its instance extensions + a surface through
  Vulkan-free signatures; the backend owns the surface. ImGui's platform
  half moved out of the RHI into the overlay. New: `core::Key` (USB HID
  usage IDs = SDL scancodes, stable names), `platform::executableDir` /
  `setEnvironmentVariable`, `std::chrono` timer, portability-enumeration
  instance flag (MoltenVK).
* **Bug found by the gate**: minimizing quit the app - SDL updates its
  minimized flag only while pumping, so a frame can acquire on a 0x0
  surface. `beginFrame` now returns `FrameStatus {Ready, Skipped, Error}`.
* Gate evidence: 4/4 render regressions **byte-identical**; 7 new tests
  (key-name round trip, compile-time HID == SDL scancode checks, Input
  state machine, injected SDL events on a hidden window, minimize/restore
  with the real backend - mutation-tested: removing the fix reproduces
  `acquire failed (-1000001004)` and fails the test); interactive smoke
  with validation (resize, 3x minimize/restore, OS close): 0 errors,
  exit 0. ctest 5/5, suite 83 cases.
* Known issue (pre-existing, for 0.3): three `[p0]` tests read an
  untracked 163 MB `out/bigknot.kgeo` relative to the working directory;
  they pass only from the repo root on this machine. They must generate
  their fixture before CI.
* Remaining Win32-only code: the SEH crash handlers (`app`, `tests`),
  guarded by `_WIN32` until 0.2.

### 0.2 Portable build — done 2026-10-23 (Windows-verified; Linux/macOS unbuilt)

* **Slang from vcpkg** (`shader-slang` 2026.7.1, release binaries for all
  three OSes) replaces the vendored Windows-only 2026.3.1; only
  `slang.dll` + `slang-compiler.dll` are deployed (slang-llvm alone is
  105 MB and never loaded for SPIR-V). `ThirdParty/slang` deleted.
* **Compiler-neutral CMake**: `kenshin_target_defaults` (MSVC `/W4
  /permissive- /utf-8`, GCC/Clang `-Wall -Wextra`, `-rdynamic` for
  readable POSIX backtraces, Release PDBs via `/Z7 /DEBUG /OPT:REF,ICF`);
  `KENSHIN_NO_WARNINGS` genex for vendored sources; Threads linked;
  vcpkg located before `project()` (no hard-coded `C:/vcpkg`); presets
  for linux-gcc/clang and macos-arm64 (host-gated).
* **Portable code**: `util::openFile` (UTF-8 paths; `_wfopen` on Windows -
  plain `fopen` loses non-ASCII paths there) replaces 17 `fopen_s`;
  `sscanf_s` -> `std::sscanf`; `KENSHIN_DEBUG_BREAK`; MSVC pragmas
  guarded; `platform::installCrashHandler` (SEH + `StackWalk64` from the
  faulting context / POSIX signals + `backtrace` on an alternate stack);
  `VK_KHR_portability_subset` enabled when exposed (spec requirement).
* **Bugs found by the gates**: GCC rejected `Renderer(Config = {})`
  (nested-class default argument - Linux/macOS would not have compiled)
  and a `long long(x)` cast; the crash handler printed no frames
  (`CaptureStackBackTrace` inside the filter) and no symbols (Release had
  no PDBs).
* Gate evidence: clean full MSVC build **0 warnings** (was ~20); GCC 13
  `-fsyntax-only -Wall -Wextra` over all 94 sources: 0 errors, 0 warnings
  (`tools/gcc_syntax_check.py`); include-case check 487 includes, 0
  mismatches (`tools/check_include_case.py`, negative-tested); crash test
  (`crash_handler` ctest: report names `kenshinCrashProbeFault`);
  render regressions byte-identical on the new Slang; ctest 7/7;
  interactive smoke with validation 0 errors.
* **Not verified** (no Linux/macOS machine): actual GCC/Clang/AppleClang
  builds and links, the POSIX crash branch, MoltenVK behavior.


## 8. Decisions (2026-10-23)

1. **Dependencies approved** as proposed in section 4 (added per phase).
2. **Verification: Windows only for now.** All code stays portable (no new
   OS-specific code outside `platform/`), but Linux/macOS builds are
   unverified until a WSL/CI/hardware path exists. Each step's gate notes
   "Windows-verified" explicitly.

## 9. Original open items

1. **Linux/macOS verification**: this workstation has no WSL, Docker or git
   remote. Options: install WSL2 Ubuntu 24.04 (Linux build + lavapipe GPU
   tests locally), a GitHub repository with Actions CI (Linux + macOS
   runners), and/or access to a real Mac (MoltenVK GPU behavior cannot be
   verified in CI VMs).
2. **Dependency downloads**: SDL3, EnTT, nlohmann-json, Jolt, Lua, sol2,
   miniaudio, ImGuizmo via vcpkg; Slang Linux/macOS binaries; Khronos glTF
   sample models for tests.
