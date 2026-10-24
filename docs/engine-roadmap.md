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
- **1.3 Asset system** - five sub-steps, each committed with its own
  gate (detail plan below, "1.3 plan").
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

### 1.3 plan (approved 2026-10-23)

Survey of what 1.3 replaces: glTF/OBJ import lives in the
`kgeo_builder` executable (material composition, ORM packing, glTF
images re-encoded as **BMP24** - alpha lost); at runtime every material
texture is a layer of **one RGBA8 Texture2DArray**, so all textures must
share one size (the gallery's 32x32 checker is rejected); mips are a
GPU linear blit, so sRGB data is filtered in the wrong space (dark
mips); sRGB is decoded in the shader; samplers have no anisotropy and
ignore glTF wrap modes; a glTF file becomes ONE merged mesh with node
transforms baked in.

**Project layout**
```
MyGame/
  project.kproject
  assets/                 sources + one .meta sidecar per asset (in VCS)
    models/helmet.glb
    models/helmet.glb.meta
    textures/brick.png
    textures/brick.png.meta
  .kenshin/cache/         cooked artifacts, disposable (not in VCS)
```
A `.meta` (JSON, versioned like scenes) holds the asset's Guid, its
importer settings and, for container assets, the stable Guids of
its sub-assets. Moving or renaming a file together with its `.meta`
keeps every reference; a cooked artifact is keyed by (source hash,
settings hash, importer version), so stale caches rebuild themselves.

**1.3a Texture pipeline (W6)**
* Import: PNG/JPEG/TGA/BMP/HDR via stb (alpha kept, any size,
  16-bit sources kept as 16-bit), plus our own reader for
  *uncompressed* KTX2.
* CPU mip chain filtered in linear light (sRGB decoded, then
  re-encoded); normal maps renormalized per mip.
* Cooked format: KTX2 (uncompressed RGBA8/RG8/R8/RGBA16F, full mips),
  with a small writer/reader of our own.
* GPU: one image per texture in its real format (`*_SRGB` where
  colour, so the hardware decodes), uploaded with its mips. A
  **bindless** texture table (Vulkan 1.2 descriptor indexing:
  runtime-sized `Texture2D[]`, partially bound, non-uniform
  indexing) and a small sampler table (wrap modes x filtering, 16x
  anisotropy). Materials hold bindless indices; shaders drop layer
  indexing and the manual sRGB decode.
* Gate: unit tests (mip maths, sRGB round trip, KTX2 round trip,
  odd sizes); the gallery's 32x32 checker loads. The render
  references **will change** (correct mips plus anisotropy). The
  old-vs-new image diff is measured and reviewed, then the
  references are regenerated.

**1.3b Importer as a library**
* The OBJ/glTF -> cluster-DAG cook moves from `tools/kgeo_builder`
  into `engine/assets/import/`; `kgeo_builder` becomes a thin CLI over
  it.
* glTF import keeps the scene graph: one cooked mesh per glTF mesh
  (primitives = material slots), materials and textures as separate
  assets, and the node hierarchy as a **prefab** (1.2 format)
  holding MeshRenderer entities.
* Gate: `.kgeo` output byte-identical to the old builder for every
  test asset (merged mode); glTF import unit tests (hierarchy,
  transforms, slots, sampler wrap modes, colour space per texture
  role).

**1.3c Asset database**
* `AssetDatabase::open(project)`: scans `assets/` and creates missing
  `.meta` files.
* Repairs: an orphaned `.meta` is reported; a duplicated Guid (a
  copied `.meta`) is re-issued to the copy, with a warning.
* Imports by type: texture, model (glTF/GLB/OBJ -> meshes + materials
  + textures + prefab), material (`.kmat` JSON: PBR factors, texture
  Guids, alpha mode/cutoff, double-sided), scene, prefab.
* Sub-asset Guids are derived deterministically (parent Guid + sub
  path), so a re-import keeps references.
* Cache: staleness check, rebuild, garbage collection.
* Gate: unit tests on temp projects (add / move / rename / delete /
  copy-with-meta / corrupt meta / settings change / importer-version
  bump / deleted cache); a re-import is deterministic (same Guids,
  byte-identical artifacts).

**1.3d Loading + hot reload**
* Runtime loads by Guid with reference counting: textures go to
  bindless slots, meshes to the VirtualGeometryScene, materials to
  the material table.
* A minimal *static* World -> renderer bridge (instances from
  MeshRenderer + WorldTransform, built at load), so a project scene
  or prefab renders. 1.4 makes it dynamic.
* Hot reload: a portable polling watcher (mtime + size, 4 Hz, cheap
  for thousands of files) -> re-import -> swap the GPU resource at a
  frame boundary.
* Demo: `--project <dir> --scene <path>`.
* Gate: headless hot-reload test (rewrite a texture on disk -> new
  pixels within a bounded number of frames, no validation errors); a
  project scene renders; the old `--kgeo` path is unchanged (render
  regressions).

**1.3e Khronos sample gate**
* `tools/fetch_gltf_samples.py` downloads a curated subset of
  KhronosGroup/glTF-Sample-Assets, pinned to a commit, into an
  untracked folder.
  - Core: Box, BoxTextured, Duck, Avocado, WaterBottle, BoomBox,
    DamagedHelmet, FlightHelmet, Lantern, AntiqueCamera, Sponza,
    ABeautifulGame.
  - Feature tests: MetalRoughSpheres, NormalTangentTest,
    NormalTangentMirrorTest, TextureCoordinateTest,
    TextureSettingsTest, OrientationTest, AlphaBlendModeTest,
    MultiUVTest, VertexColorTest, TextureTransformTest,
    SimpleSparseAccessor.
  - Roughly 250 MB; the exact size is reported before downloading.
* Each model is imported and rendered headless. A contact sheet is
  compared by eye with Khronos' reference screenshots.
* Each model is classed as correct, or as a known gap tied to its
  W-item. Expected gaps: MASK, UV1, vertex colours and texture
  transform (W7, step 1.4).
* Our renders are then kept as downscaled regression references.

**Out of 1.3 (by design)**
* GPU texture compression (BC7/BC5/ASTC via Basis Universal) and
  Basis-compressed KTX2 sources (libktx) belong with export, Phase 8.
* Asynchronous background import comes with the editor, Phase 6.
* Per-slot material overrides on MeshRenderer arrive with the
  inspector.
* Skinning and animation are W7, later.

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

### 0.4 SceneRenderer — done 2026-10-23 (Windows-verified)

* **Engine**: `renderer::SceneRenderer` owns the frame (main + 3 shadow
  VisBuffer chains, decode, shadow clipmap cache/dirty tiles, imposters,
  lighting, translucency, TAA history, exposure, bloom, tonemap + sky);
  `SceneRenderSettings` (every knob, already resolved) in,
  `SceneRenderStats` out. `renderer::Environment` (HDRI load + sun
  extraction + SH, procedural studio probe), `util::f32ToHalf`.
* **App**: `main.cpp` 2335 -> 366 lines; `AppArgs` (CLI), `KgeoScene`
  (asset/instance assembly), `VgPanel` (UI model, panel, settings
  resolution, stats readback). The engine never sees the command line.
* **Bugs fixed on the way**: imposter staging arrays were sized by the
  gallery count but indexed by mesh id (out-of-bounds write with
  `--ground` + an imposter-capable `--mesh`); every debug view (`--view
  1..6`) left the decode's binding 18 unwritten (validation error per
  frame) - `drawFullscreen` now clamps to the reflected binding count and
  reports callers that pass too few handles; shadow chains are now shut
  down.
* Gate evidence: 4/4 render regressions byte-identical; **old-vs-new A/B**
  (pre-refactor commit built in a worktree) over 24 cases - 17
  byte-identical, `gallery_imposter` differs by the imposter fix, the rest
  match the OLD binary's own run-to-run noise (repeat runs of both);
  headless `[vgstats]`/`[shadowstats]` identical old vs new (3 scenes);
  debug views 0 errors (were 10), pixels unchanged; GCC check 100 files
  clean; MSVC 0 warnings; interactive smoke 0 errors; ctest 7/7.
* Known issues logged for later: several scenes render
  nondeterministically run-to-run (VisBuffer atomic depth ties - the knot
  tests carry a 64 px budget for it); the shadow chains' stats readback
  returns garbage (`[shadowstats]`, diagnostics only - rendering is
  verified); the gallery's `checker_albedo.bmp` (32x32) is rejected by
  the shared texture array (W6, step 1.3).

### 1.1 ECS — done 2026-10-23 (Windows-verified)

* **Engine** (`kenshin/ecs/`, EnTT 3.16): `ecs::World` wraps
  `entt::registry` and owns its invariants. Every entity carries Name,
  Identity (Guid), Transform (local TRS), Hierarchy and WorldTransform;
  MeshRenderer, Light and Camera are optional. Hierarchy is intrusive
  ordered sibling lists (roots are one more list). `setParent` keeps
  world or local pose, takes an insertion point and rejects cycles and
  bad siblings without changing anything. `destroy` (subtree, now) plus
  `queueDestroy`/`flushDestroyQueue` (mid-iteration). Lookup by Guid
  and by name. `updateTransforms` is an iterative pre-order pass (no
  recursion limit); `computeWorldMatrix`/`setWorldMatrix` are exact.
  Structural components (Hierarchy, Identity, WorldTransform) are only
  handed out const - views and `get` included - so callers cannot
  corrupt the links. `core::Guid`: random RFC 4122 v4, canonical text
  form, strict parsing.
* **Conventions**: RH Z-up meters; local +X right, +Y forward, +Z up
  (cameras/lights point along +Y). `decomposeTransform`: exact for TRS,
  mirror -> negative X scale, shear dropped, singular -> refused.
* **Reflection** (`ecs::ComponentRegistry`): per component a stable
  snake_case name and fields reached through member pointers (no
  `offsetof`), with type, range, enum labels, color hint and
  description. Type-erased has/get/add/remove. `setField` validates
  type, finiteness, range, enum membership and normalizes quaternions,
  rejecting with an agent-readable message and changing nothing.
  Registration errors abort. Game components register through the same
  builder.
* Gate evidence: 23 test cases / 133,735 assertions in Release and
  Debug (EnTT and engine asserts live): hierarchy math, sibling
  ordering, reparent (both modes, singular parent), cycle rejection,
  subtree destruction and relinking, stale-handle safety, queued
  destruction from inside a view, a 3000-step random
  create/reparent/destroy run checked link-by-link against a reference
  model, a 20,000-deep chain, Guid format/parse, and reflection
  read/write of every field. **Mutation test**: 10 deliberate engine
  bugs (link counts, last-child repair, cycle check, keep-world,
  compose order, Guid index, field wiring, range check, mirror fold,
  sibling check) are all killed. GCC check 105 files clean (found a
  latent MinGW `SIGTRAP` break in `Assert.h`, fixed); MSVC 0 warnings;
  ctest 7/7.
* Performance (`[bench]`, 100k entities, random tree): create 28 ms,
  `updateTransforms` 5-6 ms steady (10k: ~0.5 ms), destroy-all 50-65 ms.
  Building a deep chain was quadratic (new child seeded from
  `computeWorldMatrix`, 7.6 s for 20k) - now seeded from the parent's
  cache, 8 ms. For 1.4: sort pools in hierarchy order and skip
  unchanged subtrees (the change tracking motion vectors and shadow
  invalidation need anyway).

### 1.2 Serialization — done 2026-10-23 (Windows-verified)

* **Format** (`scene/SceneSerializer.h`, nlohmann/json 3.12): text
  JSON with a `{"format", "version"}` header. A scene is a flat entity
  list in hierarchy pre-order (guid, name, parent guid, components),
  and components are written field by field through the 1.1
  reflection: vectors are arrays, quaternions `[x, y, z, w]`, enums
  their labels, Guids strings or null.
* **Canonical output** (`serialization::writeJson`): insertion-ordered
  keys, 2-space indent, short scalar arrays on one line, `
` endings,
  and floats in their shortest exact form (`0.95`, not
  `0.949999988079071`). A double-rounding guard falls back to the exact
  double. `.gitattributes` pins `*.kscene/.kprefab/.kproject` to LF.
  `setField` leaves already-unit quaternions bit-exact, so re-saves
  don't drift.
* **Strict, all-or-nothing loading**. Every one of these is an error
  that names the entity and field, with nothing changed: unknown keys,
  components or fields (the error lists the valid ones); wrong types;
  out-of-range values; bad enum labels or guids; duplicate guids;
  dangling, self or cyclic parents; duplicate JSON keys; nesting over
  128; overflowing numbers. Scenes load into a staging World that is
  moved in only on success; prefab instantiation rolls back. Hand-
  written files may list entities in any order and omit guids, fields
  or components.
* **Migration**: `SchemaMigrator` chains per-version steps. It rejects
  wrong formats, newer versions, gaps and step failures (including
  exceptions) with a reason. Version 1 is the first format, so the
  built-in chains are empty.
* **Prefabs**: the same document with one root; each instance gets
  fresh Guids under any parent. **Project**: `project.kproject` (name,
  startup scene; paths must stay inside the project).
  `util::writeFileAtomic` (temp + flush/fsync + rename), `readFile`,
  `pathToUtf8`.
* Gate evidence: 16 test cases. Byte-identical round-trip of the
  golden scene, of 40 random worlds (every field type, control
  characters and CJK in strings) and of the project file. Golden files
  for scene, prefab, project and a migrated legacy v0 scene (hand-
  written v0 input + a test migration chain). 300k random float bit
  patterns round-trip exactly. 32 invalid-document cases each assert
  the message and an untouched world. UTF-8 (CJK) paths, overwrite, no
  temp file left behind. **Mutation test**: 16 deliberate bugs, all
  killed. One early survivor was a wrongly written mutation (`&&`
  binding tighter than `?:`); another exposed an untested
  large-object duplicate-key path, now covered. Debug build passes.
  GCC check 109 files clean; ctest 7/7.
* **Found on the way**: nlohmann throws `out_of_range` (not
  `parse_error`) for numbers like `1e999` - the loader would have let
  it escape; all its exceptions are now caught. A `version` beyond i32
  would have narrowed silently. Its callback parser was 8x slower than
  the plain one: now a SAX validation pass plus the plain parse. 10k
  entities (4.4 MB): load 400 -> ~120 ms, save ~95 ms.

### 1.3a Texture pipeline — done 2026-10-23 (Windows-verified, RTX 5060 Ti)

* **CPU side** (`assets/`):
  - `Image`: stb decode of PNG/JPEG/TGA/BMP/HDR, alpha kept, 16-bit
    and float kept.
  - `cookTexture`: full mip chain for any size. Each mip texel
    averages exactly the source area it covers; colour is filtered
    in linear light and alpha-weighted; normal maps are renormalized;
    `maxSize` limit; mip 0 is copied verbatim.
  - KTX 2.0 writer and strict reader: uncompressed RGBA8 UNORM/SRGB,
    RGBA16 UNORM, RGBA16F; Basic DFD, stable KTXwriter tag.
  - `util::halfToF32`.
* **GPU side**:
  - Each texture is its own image in its real format (`*_SRGB` for
    colour: the hardware decodes) with the CPU mips uploaded as they
    are.
  - `BindlessTable` (set 1): 16384-slot partially-bound
    update-after-bind `Texture2D[]` plus 18 immutable samplers (3x3
    wrap modes x linear/nearest, 16x anisotropy). Slot 0 is white;
    freed slots point back at white.
  - Descriptor indexing and anisotropy are queried and enabled;
    initialization fails with a clear message without them.
  - Materials store 32-bit refs (slot | sampler << 24). The decode
    pass samples with `SampleGrad` on its analytic UV derivatives, so
    each texture's own mip and the anisotropic footprint come from
    the hardware. The translucent pass uses the same table.
  - Set-0 layouts now honour set numbers; `imposterPairs` moved to
    binding 17, so set 0 is dense and the debug-view filler handle
    is gone.
  - `TextureManager` rewritten (decode -> cook -> upload,
    deduplicated by path + settings).
* **Fixed (W6)**: textures of any size; the gallery's 32x32 checker,
  rejected before, now textures its cube. Mips are filtered in
  linear light. glTF wrap modes are representable; the importer
  wires them in 1.3b.
* Gate evidence:
  - CPU: 14 test cases. Every half value round-trips; all 256 sRGB
    codes round-trip. PNG bytes come from an in-test encoder (RGBA,
    RGB, grey+alpha, 16-bit, multi-block). Mip chains for 8 sizes,
    including odd ones. Hand-computed expectations: linear-light
    average 188 (not 128), no colour bleed from transparent texels,
    area weights at 3->1 and 5->2, unit-length normal mips. Size
    limit; 16-bit and HDR range; KTX2 round trip in all 4 formats
    with spec layout checks; 12 malformed-KTX2 cases.
  - GPU: 4 test cases through `bindless_probe.slang`, which calls
    the shaders' own `sampleTextureGrad`. Every mip level of a 13x6
    texture lands where uploaded; hardware sRGB decode matches the
    cooker's curve; RGBA16F above 1 and RGBA16 come back; slot 0 is
    white; all wrap/filter modes behave; slots are reused and freed
    slots read white; bad descriptions are rejected.
  - **Mutation test**: 13 deliberate bugs (linear-light filter,
    alpha weighting, renormalize, area weights, KTX2 DFD check and
    level order, upload mip offsets, sRGB VkFormat, mirror sampler,
    release-to-white, shader sampler index, shader slot index, half
    denormals) are all killed. The first run exposed a real gap: the
    probe had its own sampling code, so the shaders' function was
    untested. It now goes through `sampleTextureGrad`.
  - Renders: the 3 untextured references stay byte-identical (set 1
    bound but unused changes nothing). The albedo knot differs by
    67k px, max delta 69.
  - Old-vs-new A/B (previous commit built in a worktree), 7 textured
    scenes: no errors. The knot grazing view shows linear-light
    blending at the green/yellow repeat seam; the close view is
    crisper (anisotropy); vulture and F-35 match visually; the
    gallery cube now has its texture. After that review,
    `render_kgeo_albedo_ref.bmp` was regenerated; 3 reruns pass.
  - Validation layer: 0 errors (knot, vulture: GBuffer +
    translucent). GCC check 115 files clean. Debug build: 57 cases
    (ECS + serialization + texture + bindless) pass. ctest 7/7.
* Costs and notes:
  - Loading sources now cooks mips on the CPU, about 0.6 s per 4096^2
    texture (vulture: 5 textures). The 1.3c cooked cache (KTX2)
    removes this.
  - The SPIR-V cache keys on the shader file's own mtime: imported
    modules (`bindless.slang`) need dependency tracking once 1.3d
    hot-reloads shaders.
  - `KGeoFile.cpp` uses MSVC-only `_ftelli64`/`_fseeki64`, which
    MinGW also accepts but Linux does not; fix with the importer move
    (1.3b).
  - The KTX2 files are checked by our own reader only; no `ktx
    validate` is available here.

### 1.3b Importer as a library — done 2026-10-24 (Windows-verified)

* **Engine `importer/`**:
  - `MeshConditioning` (recentre, translucent split + `.tmesh`, BLEND
    sink, layer separation, double-sided back faces), moved verbatim
    from the builder.
  - `MeshCooker` (cluster DAG, imposter bake, v2/v7/v8/v9 `.kgeo`,
    census).
  - `KgeoMaterials` (the self-contained `.kgeo` material table with
    BMP side files).
  - `kgeo_builder` is now a ~200-line CLI over them (1098 lines
    before). Dropped: the finished M7.2 index-span probe (logs only).
* **Scene-graph glTF** (`assets/GltfScene.h`, `importer/GltfModel.h`):
  - The default scene's node tree, with local transforms conjugated
    into Z-up (R M R^-1).
  - One mesh per glTF mesh: mesh-local, instanced by several nodes,
    material slots.
  - Full materials: every texture slot with its sampler (wrap modes,
    nearest), UV set and factors, alpha mode, emissive strength.
  - Images kept as encoded bytes.
  - `importGltfModel` turns these into sub-assets with stable keys
    (`mesh/N`, `material/N`, `texture/N.srgb|linear|normal`) and
    conditions each mesh, with thresholds relative to its own size.
    `modelPrefab` emits the 1.2 prefab (Transform + MeshRenderer),
    with Guids from a caller-supplied key -> Guid function (1.3c).
  - Unsupported content is reported in `notes`: other UV sets,
    KHR_texture_transform, vertex colours, skins, animations, cameras,
    lights.
  - Unsupported *required* extensions (Draco, meshopt, ...) now fail
    the load, as the spec demands.
* **Loader defects fixed**:
  - Percent-encoded URIs (`my%20image.png`) were validated decoded but
    opened raw, so the file was never found.
  - The scheme check only caught `:` at index 0/1 (`http:`/`file:`
    passed).
  - JSON `\uXXXX` escapes were kept verbatim; they are now decoded to
    UTF-8, including surrogate pairs.
  - Trailing data after the root JSON value was silently ignored;
    whitespace/NUL padding is still accepted.
  - The GLB chunk cursor could step past the buffer end on a hostile
    length.
  - `KGeoFile` used MSVC-only `_ftelli64`/`_fseeki64`; now
    `util::tell64/seek64` (`ftello`/`fseeko` on POSIX).
* **Deliberate change**: BMP side files keep their native size. The
  array-era box resample to the first image's size is gone; it only
  remains for AO -> ORM composition. The vulture visor MR map was
  being upsampled 1024 -> 4096 (16x memory); the render differs by
  3.3k px (max delta 12) against bit-exact noise.
* Gate evidence:
  - The builder is deterministic (two baseline runs identical).
  - A 17-case matrix: procedural knot/sphere with every flag
    (albedo, material overrides, v2, v7, imposter), 4 OBJ, 2 GLB, the
    vulture (default, `--opaque-blend`, `--no-double-sided`) and the
    F-35. Every `.kgeo` and `.tmesh` is byte-identical to the original
    builder, after each refactor step and at the end. Side files are
    identical too until the deliberate no-resample change, which
    alters exactly one file.
  - Error exit codes match.
  - 7 glTF import test cases on a generated reference model:
    hierarchy, names, `\u` decoding, conjugated transforms,
    non-uniform scale axis, instancing, slots, computed normals,
    roles/samplers/UV sets, encoded image bytes via a `%20` URI, notes;
    sub-asset keys, translucent split, deterministic re-import;
    hostile files (required extension, bad indices, cycle, depth 300,
    `../`, `file:`, `http:`, `C:`, trailing data).
  - **The prefab's world matrices x mesh-local vertices reproduce
    the flattening loader's world-space vertices**: exactly on the
    reference model, and on the real vulture (worst 4.6e-5 of a
    177-unit extent) and F-35 (2.5e-7 of 0.44).
  - **Mutation test**: 14 deliberate bugs (conjugation, normal frame,
    slot dedup, wrap mapping, URI decoding, required extensions, cycle
    guard, trailing data, surrogates, scheme check, data-role colour
    space, translucent split, prefab mesh ref, prefab transform) are
    all killed.
  - ctest 7/7; Debug: 72 cases pass; GCC check 120 files clean; demo
    renders the rebuilt vulture with 0 errors.
* Found for later:
  - The F-35 samples its MR and normal maps through TEXCOORD_1/2; the
    renderer uses set 0, so those maps shade wrong today (W7, step
    1.4).
  - Scene import conditions each mesh separately: the builder's
    cross-mesh layer separation (merged world space) has no
    equivalent for instanced meshes. Watch for it in the 1.3e review.

### Fix — helmet flicker with shadows off (2026-10-24, Windows-verified)

* **Symptom**: with shadows off, the helmet flickered.
* **Measurement**: the earlier per-frame-capture metric hid it,
  because capturing every frame serializes CPU and GPU. Measured the
  way a monitor sees it instead: free-running at ~575 fps, every 17th
  frame captured, temporal luma sigma over 41 captures.
  - Whole-helmet single-frame flashes (mean 91 -> 96-98).
  - 87,584 px with sigma > 5 (153,203 at fixed exposure); 469 with
    shadows on. The count varied run to run.
  - `--no-taa`: 0 (no race in the frames themselves); `--no-async`:
    unchanged.
* **Cause**: `MaterialDecodePass` kept its per-draw params (paged
  flag, material count, pair capacity) in ONE host-visible buffer,
  written at record time.
  - The shadows-off "clear the shadow map" decode, recorded every
    frame, wrote paged = 0 / materialCount = 0 while the previous
    frame's main decode could still be executing. That decode shaded
    the helmet unpaged and material-less for one frame.
  - Within a frame, the shadow-level decodes also read the main
    decode's values.
  - With shadows on, the cached shadow map rarely redraws, so the
    stray write was rare.
* **Fix**:
  - `IRenderBackend::transientHostBuffer(data, bytes)`: per-draw
    host-visible constants, one buffer per call, pooled per frame slot
    and rewound after that slot's completion wait.
    `snapshotHostBuffer` remains the once-per-frame variant.
  - The decode pass allocates its params per draw.
  - The per-frame imposter pair list is snapshotted where the decode
    consumes it (the same cross-frame race while the camera moves).
  - Audit: lighting, TAA, translucent, imposter params and the
    VisBuffer chain already used per-slot copies.
* After: 532-536 px with sigma > 5 (609 at fixed exposure), helmet
  brightness range 0.01. The rest is ordinary TAA edge jitter
  (shadows on: 466).
* Gate: GPU test (two dispatches in one frame each read their own
  transient data; same-slot frames reuse their buffers). The
  "shared buffer" and "no recycling" mutations are both killed.
  ctest 7/7; Debug suites pass; GCC check clean; 0 validation errors
  in the live runs.
* Also fixed on the way (separate commit): **specular anti-aliasing
  for normal maps** (Toksvig / vMF).
  - 1.3a's correct per-texture mips and anisotropy resolve more
    normal detail, which sparkled under unshadowed direct light: 646
    vs 257 px with sigma > 5 at matched exposure, from consecutive
    frames.
  - The cooker now stores each normal-map texel's average-normal
    length in alpha (1 - sqrt(1 - r)); the decode pass widens GGX
    roughness by 2/kappa.
  - Back to 265 px. Mip 0 / close-ups are unchanged; the ribbed strap
    shows a smooth sheen instead of sparkle. Cooker tests cover the
    encoding (4 mutations killed). The shader effect is verified by
    measurement only (no automated GPU test).
* Open: geometric specular AA (screen-space normal derivatives) for
  fine geometry, phase 2.

### 1.3c Asset database — done 2026-10-24 (Windows-verified)

* **`assets::AssetDatabase`** over a project folder: `project.kproject`,
  `assets/` (sources + `.meta` sidecars, in VCS), and
  `.kenshin/cache/<guid>/` (artifacts + `import.json`, disposable).
* **Types**:
  - texture (png/jpg/tga/bmp/hdr/ktx2 -> cooked KTX2);
  - model (gltf/glb/obj -> per-mesh `.kgeo` with a per-slot factor
    table, `.tmesh` for BLEND parts, per-texture KTX2 by colour role,
    per-material `.kmat`, and the prefab);
  - material (`.kmat` -> canonical);
  - scene / prefab (validated against the component registry,
    copied).
* **Identity**:
  - `.meta` = guid + type + importer settings (strict, defaults
    filled; a texture name heuristic for new assets: normal / linear
    data).
  - Sub-asset Guids are RFC 4122 v5 (SHA-1) of (asset Guid, key), so
    re-imports and lost caches reproduce them. New `core::Sha1` and
    `core::Guid::fromName`.
* **Repairs on scan**:
  - Missing `.meta` -> created.
  - An orphaned `.meta` whose last import matches a new file's content
    is adopted (a move that left the `.meta` behind keeps its Guid).
  - A duplicated Guid (copied `.meta`) is re-issued to the copy; the
    asset its import record names keeps it.
  - A corrupt `.meta` is reported, its asset skipped, the file never
    overwritten.
  - Orphans are reported and kept.
* **Staleness**: type, importer version (plus a test salt), settings
  hash, and every source and dependency (the glTF's `.bin`/images,
  the OBJ's MTL and `map_Kd`). Size + mtime is checked first, SHA-1
  only when those changed; a moved asset is checked at its new path.
  Imports build in `<guid>.tmp` and replace the old artifacts only on
  success. `collectGarbage` removes caches of vanished assets and
  temp leftovers.
* **Material format** (`assets/Material.h`): factors, alpha mode/
  cutoff, double-sided, normal scale, occlusion strength, five
  texture slots (Guid + wrap U/V + filter); canonical, strict,
  versioned.
* `GltfScene::externalFiles`: every file the loader read (dependency
  tracking).
* Gate evidence:
  - 15 test cases on temp projects: SHA-1 FIPS vectors (incl. 1M 'a'
    fed in uneven chunks), the RFC 4122 v5 vector.
  - New sources -> metas -> import -> a new session has nothing to
    do.
  - Staleness: same-bytes rewrite stays current, new content, settings
    (validated, written to `.meta`), importer version.
  - **A deleted cache re-imports byte-identically** (all cache files
    including records, with the generated glTF reference model).
  - Moves with and without the `.meta`; a copied `.meta`; three kinds
    of corrupt `.meta` left untouched; orphans + GC.
  - Model sub-assets: 9 keys with derived Guids; the prefab's mesh
    refs and the materials' texture refs resolve; the `.bin` is a
    dependency.
  - OBJ: mesh/material/texture/prefab; the MTL is a dependency.
  - Validation (bad `.kmat`, unknown component, no registry); a
    failed re-import keeps the good artifacts; `.kmat` round trip +
    6 strict-load errors; missing project file.
  - The real Vulture imports and resolves through the database.
  - **Mutation test**: 12 deliberate bugs (move adoption, guid
    re-issue, corrupt-meta protection, settings / importer version /
    content hash in staleness, moved-asset path, dependency stamps,
    derived Guids, GC keeping live caches, failed import keeping
    artifacts, the texture name heuristic) are all killed.
  - ctest 7/7; Debug 56 cases; GCC check 125 files clean.
* Notes:
  - The Vulture import takes 14.5 s: every 4096^2 texture is
    CPU-cooked, and the model's images are cooked twice (as model
    sub-assets and as standalone files in `textures/`).
    Parallel/background import is planned with the editor (Phase 6).
  - OBJ is imported as-is (no up-axis conversion), like `kgeo_builder`.

### 1.3d Loading by Guid + hot reload — done 2026-10-24 (Windows-verified)

* **`manager::AssetManager`**: database artifacts by asset or sub-asset
  Guid, with reference counts.
  - Textures become GPU textures in the bindless table (KTX2).
  - Materials are parsed `.kmat`s and hold their textures.
  - Meshes are the cooked DAG tables; their pages stream from the
    cache file.
  - The last release frees the resource.
* **Hot reload**, between frames, in two phases:
  - `core::FileWatcher` polls size + mtime of `assets/` (4 Hz; one
    code path on every platform). A burst of changes is acted on at
    the first quiet poll.
  - `planReload` rescans the database and lists the stale assets plus
    the resident meshes they own. The scene stops streaming from those
    files first: the streamer reopens a mesh file by path for every
    page, so after the re-import it would read the new file against
    the old tables, and a read in flight blocks the folder swap on
    Windows.
  - `applyReload` re-imports, then reloads only resident resources
    whose cooked bytes changed. Import records now carry each
    artifact's SHA-1, so a re-import that reproduces a sub-asset
    leaves it alone.
  - Textures are replaced **in place**: new
    `IRenderBackend::replaceTexture` keeps the handle and the bindless
    slot (device drain), so material records never change.
  - Material edits rewrite their records (a fresh material table, the
    old one retired after the frames in flight). Mesh and scene edits
    rebuild the scene.
  - A failed re-import keeps the loaded version.
* **`renderer::WorldRenderer`**, the static World -> renderer bridge:
  - Every visible MeshRenderer becomes an instance (hierarchy
    pre-order, exact world matrices) in its own VirtualGeometryScene +
    StreamingManager + SceneRenderer.
  - Slot materials come from the import record. New: mesh artifacts
    list their slot materials; model importer v2.
  - `MeshRenderer.material` overrides every slot of one instance,
    through a per-instance material base (new `VgInstance::materialBase`
    and `addMaterials`).
  - A rebuild acquires the new references before releasing the old
    ones, so shared resources stay resident.
  - `scene::loadWorldAsset` loads a scene, a prefab or a model's prefab.
* **Asset database**: an import now renames the old cache folder
  aside before moving the new one in. Before, a file held open by any
  reader (on Windows) made it delete half of the old artifacts and
  then fail; now it fails cleanly with the old import intact.
* **Demo**: `--project <dir> [--scene <path>]` (default: the project's
  startup scene), `--watch-interval <s>` (0 = every frame) and
  `--touch <frame> <src> <dst>` (the tests' scripted edit).
* **Fixture**: `tests/data/project`, 73 KB, generated by
  `tests/make_project_fixture.py`.
  - A glTF with a textured sphere and a ground grid.
  - A standalone texture and a material overriding one sphere.
  - A scene and fixed-Guid metas.
  - The edits are in `tests/data/project_edits`.
* Gate evidence:
  - **`render_project`**: the project scene renders, byte-identical
    to the committed reference across reruns.
  - **Four hot-reload ctests.** Each edits one file at frame 8 (16
    frames, TAA off) and must end on the same image as a cold start of
    the edited project. All are byte-identical, with no validation
    errors.
    - Texture: replaced in place, slot unchanged.
    - Material: one record rewritten.
    - Model dependency: the streamed files close, two assets
      re-import, the model's texture swaps in place, its meshes come
      out byte-identical so they are not reloaded, and the scene
      rebuilds.
    - Scene document: the World reloads.
    The swap lands two frames after the edit with per-frame polling.
  - **Unit and GPU tests** (7 new cases, 1 extended):
    - Watcher: adds, edits, removals, same-size saves, a missing root.
    - Texture reference counts and the bindless slot freed on the
      last release.
    - In-place reload probed on the GPU: same reference, new texels.
    - A same-bytes save does not re-import; a broken file keeps the old
      texels.
    - Settling and the poll interval.
    - Material texture swaps, and a material edit loading nothing but
      the material.
    - The WorldRenderer's instances, records and override; skipped
      invisible / missing meshes; reference hand-over on rebuild (no
      reloads); a model settings change closing files, re-importing
      and rebuilding.
    - Asset DB: mesh slot lists, artifact SHA-1s matching the files
      and surviving a new session, and the locked-file swap (the
      Windows branch is exercised).
  - The `--kgeo` render regressions (4) pass unchanged.
  - **A pre-existing intermittent, found and measured.**
    `render_kgeo_nohw` occasionally renders a second state: exactly
    89 px off, max delta 1 (sky and model), bbox [7,137]-[778,695]. It
    exceeds the 64 px budget, so the test fails then.
    - The log is identical, including exposure to 5 digits; it is not
      an off-by-one-frame capture.
    - Interleaved A/B on the same `.kgeo`, parent commit (cf92d33) vs
      this step: 4/120 vs 1/120 runs, the same signature. The parent
      has it too.
    - Cause found and fixed afterwards: see "Fix - TAA exposure read
      on the CPU" below.
  - **Mutation test**: 17 deliberate bugs are all killed.
    - Slot rewrite on replace, in-place reload, unchanged artifacts
      left alone, settling, resident meshes in the plan.
    - The last release freeing, shared textures kept across a material
      reload, the rename-aside swap.
    - Artifact SHA-1s and slot materials recorded, the override,
      acquire-before-release on rebuild, record filtering.
    - The per-instance material base, the material table re-upload,
      same-size saves, scene document reload.
    Of the 17, 13 are killed by unit tests and 4 only by the project
    ctests. The resident-meshes mutation survives the model ctest, whose
    meshes re-cook identically with all pages resident; the GPU unit
    test kills it.
  - ctest 12/12 (two full runs failed on the intermittent above).
    Debug: 80 cases pass; the Debug demo's model reload
    with validation logs no validation messages. GCC check 131 files
    clean; it caught `near` as a MinGW `<windows.h>` macro in a test
    helper.
* Not drawn yet (logged per build):
  - Translucent sub-meshes (BLEND) and alpha-mask cutoffs: 1.4 (W10,
    W7).
  - Occlusion maps not packed into the metallic-roughness map, and
    emissive maps: the Khronos gate (1.3e) will show which samples need
    them.
* Notes:
  - Shader hot reload is not part of this step. The SPIR-V cache still
    needs dependency tracking for imported modules (1.3a note).
  - A material or mesh change rebuilds or re-uploads whole tables;
    incremental GPU scene updates come with the dynamic bridge (1.4).
  - glTF `data:` buffers are still unsupported (the fixture uses a
    `.bin`).


### Fix — TAA exposure read on the CPU (2026-10-25, Windows-verified)

* **Symptom**: `render_kgeo_nohw` intermittently rendered a second
  state (89 px, max delta 1), in about 2% of runs (A/B: 4/120 before
  1.3d, 1/120 after).
* **Diagnosis**:
  - The streaming timeline was identical in failing runs (logged per
    frame), which ruled streaming out.
  - The value that moved was TAA's HDR weight exposure. SceneRenderer
    took it from `ExposurePass::readState`, a CPU read of the
    host-mapped state that the GPU's adapt pass writes. Which frame's
    value it saw depended on how far the GPU had got.
  - Caught in a failing run: frame 4 used frame 3's stale 1.40257716
    instead of 1.36214256. The resolve carried the difference into
    its history.
* **Fix**:
  - The resolve reads the adapted scale from the exposure state on
    the GPU (`taa.slang` binding 4). By queue order it is always the
    previous frame's value.
  - A fragment->compute barrier orders the read before that frame's
    adapt overwrite.
  - The CPU readback now only feeds the panel and stats. Manual
    exposure, and the first frames before the state is valid, use the
    CPU value as before.
* Gate evidence:
  - 150/150 runs byte-identical (the same command line).
  - The TAA-on references shift: knot 103 px, max delta 1; albedo
    367 px, max delta 4. Routed and `--no-hw` remain byte-identical to
    each other. Both references were regenerated; 3 reruns pass.
  - Reverting to the CPU read would fail `render_kgeo`: the old
    code's usual image is the previous reference, 103 px from the new
    one.
  - ctest 12/12; validation 0 with auto and manual exposure; GCC check
    131 files clean.

### 1.3e Khronos sample gate — done 2026-10-25 (Windows-verified)

* **Samples**: the 23 curated models of KhronosGroup/glTF-Sample-Assets
  at commit 8f49239: per model, the plain glTF variant, its reference
  screenshot and license (334 files, 212.7 MB, verified file by file).
  - `tools/gltf_samples.sparse` (sparse-checkout patterns) and
    `tools/fetch_gltf_samples.py` (partial clone + sparse checkout +
    pin).
  - The checkout lives outside the repository (here
    `C:/code/glTF-Sample-Assets`, downloaded by the user).
* **`tools/khronos_gate.py`**:
  - Copies each model's glTF folder into a throwaway project, imports
    it through the asset database, and renders it headless through
    the 1.3d project mode.
  - Each model gets a view matching its Khronos screenshot: front,
    3/4, face-on test boards, Sponza from inside. New demo flag
    `--frame yaw,pitch,scale` auto-frames the bounds from an angle.
  - Writes a contact sheet (ours | reference) and a summary of import
    notes, warnings and validation messages per model.
  - `--update-refs` / `--check` keep our renders, downscaled to 320 px,
    as regression references (`tests/data/khronos`, 23 PNGs, 1 MB).
    The `khronos_gate` ctest registers when `KENSHIN_GLTF_SAMPLES`
    names a checkout.
* **Defects the gate found, all fixed**:
  - **Emissive maps were ignored**. glTF emissive = factor x map, and
    these files rely on the map to mask a (1,1,1) factor, so
    WaterBottle, BoomBox, DamagedHelmet, Lantern and MultiUVTest
    rendered solid white. Material records grow to 16 floats (a 4th
    float4: the emissive map reference); `decode_material` and
    `translucent` sample it. The `--kgeo` regressions are byte-identical.
  - **One-primitive meshes skipped every conditioning pass** (a 1.3b
    regression). MeshData's "empty triMaterial = all slot 0" shorthand
    made the double-sided, translucent-split and layer passes return
    early. TextureSettingsTest's double-sided row showed a cross and
    TextureCoordinateTest lost its back plane; BLEND on a single
    primitive stayed opaque. The importer now materialises the slot
    list (model importer v3).
  - **Sparse accessors were rejected**; now supported: base values (or
    zeros without a bufferView), substitutions, bounds checks
    (SimpleSparseAccessor).
  - **`byteStride` was ignored**, a latent bug: interleaved vertex
    data would have been read as garbage. No sample interleaves (all
    434 strided accessors are tightly packed), so a unit test pins it.
  - **A model as `--scene` failed** (a 1.3d bug): a model's own Guid
    names no artifact, so `worldDocumentOf` returned before its
    "model -> prefab" branch.
  - **A material-less mesh slot recorded a Guid naming nothing**, and
    every build logged a failed load. It is now a null Guid.
  - **Tiny props were unframeable**: the camera's 0.1 m zoom floor and
    1 cm near plane left BoomBox (2 cm) as a speck. CameraController
    now owns a distance range (defaults = the old effective 0.1-100),
    and project framing scales it and the near plane with the bounds.
* **Classification** (contact sheet by eye against the references):
  - Correct: Box, BoxTextured, Duck, Avocado, WaterBottle, BoomBox,
    DamagedHelmet (front view), FlightHelmet, Lantern, AntiqueCamera,
    MetalRoughSpheres, NormalTangentTest, NormalTangentMirrorTest,
    TextureCoordinateTest, TextureSettingsTest (every row passes),
    OrientationTest, SimpleSparseAccessor.
  - Correct apart from known gaps:
    - Sponza: alpha-mask foliage renders as cards (W7), and the scene
      is brighter (auto exposure, studio light).
    - ABeautifulGame: KHR_materials_transmission / volume ignored.
    - DamagedHelmet: a separate occlusion map is ignored.
  - Known gaps, rendering as expected for a viewer without the
    feature (W7, step 1.4):
    - AlphaBlendModeTest (MASK cutoff, BLEND);
    - MultiUVTest (shows the file's own "multiple UVs not supported"
      fallback texture);
    - VertexColorTest (its crosses);
    - TextureTransformTest (KHR_texture_transform).
* Gate evidence:
  - All 23 models import and render with 0 validation messages and
    no warnings beyond the known gaps above.
  - Same binary twice: 22/23 renders byte-identical. Sponza differs
    by about 250 px with max delta 3 (depth ties in a dense interior);
    after downscaling, 1 thumbnail pixel against a budget of 379.
    `--check` passes 23/23.
  - New tests:
    - one-primitive conditioning (double-sided back faces, BLEND
      split, untouched plain);
    - interleaved and sparse accessors against dense twins, plus four
      hostile variants;
    - scene / model / prefab documents;
    - the null slot of a material-less mesh.
    The conditioning and documents tests fail without their fixes.
  - **Mutation test**: 8 deliberate bugs are all killed.
    - Emissive map sampling, the emissive ref in the record and the
      zoom floor fail the gate on exactly the affected models (the 5
      emissive ones; Avocado + BoomBox).
    - One-primitive conditioning, byteStride, sparse values and the
      null slot fail unit tests. Importer mutations do not reach the
      gate's import cache.
    - The model document fix fails both.
  - ctest 13/13 including `khronos_gate` (147 s from an empty cache).
    Debug: 30 cases (glTF / asset db / hot reload / watcher) pass. GCC
    check 131 files clean.


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
