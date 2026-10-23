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
