// KENSHINDemo entry (S1.3: swapchain + frame loop + screenshot pipeline).
// Usage:
//   KENSHINDemo                                  windowed loop (ESC quits)
//   KENSHINDemo --frames N                       auto-exit after N frames
//   KENSHINDemo --no-validation                 disable the validation layer (dev default: ON)
//   KENSHINDemo --size WxH                       client size (default 1600x900)
//   KENSHINDemo --cam yaw,pitch,dist[,x,y,z]     camera params (consumed from S1.5 on)
//   KENSHINDemo --screenshot path.bmp            capture the last rendered frame
//   KENSHINDemo --selftest-window                window event-chain selftest (no GPU)
//   KENSHINDemo --selftest-shader                Slang compile + uint64 atomic compute selftest

#include "kenshin/Version.h"
#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Input.h"
#include "kenshin/core/Log.h"
#include "kenshin/core/Window.h"
#include "kenshin/assets/ObjLoader.h"
#include "kenshin/geometry/KGeoFile.h"
#include "kenshin/geometry/LodCut.h"
#include "kenshin/assets/ProceduralMesh.h"
#include "kenshin/manager/ShaderManager.h"
#include "kenshin/scene/Camera.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "kenshin/scene/CameraController.h"
#include "kenshin/core/FrameStats.h"
#include "kenshin/core/Timer.h"
#include "kenshin/util/BmpWriter.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#pragma warning(push, 0)
#include <imgui.h>
#include <imgui_impl_win32.h> // after Windows.h: backend expects HWND declared
#pragma warning(pop)

// imgui_impl_win32.h intentionally wraps the WndProcHandler declaration in
// '#if 0' (to avoid dragging <windows.h> into the header). Per its comment,
// forward declare it here — Windows.h is already included above.
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg,
                                                             WPARAM wParam, LPARAM lParam);

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>

namespace
{
    struct Args
    {
        int  frames = -1;                            // -1 = run until the user closes
        int  width = 1600;
        int  height = 900;
        bool selftest = false;
        bool selftestShader = false;
        bool validation = true;   // dev default ON; --no-validation to disable
        int resizeAt = -1;                           // frame index to program-resize (S2.3)
        bool depthProbe = false;                     // two-cube occlusion diagnostic
        std::string kgeo;                            // .kgeo virtual-geometry preview (S3.4)
        float lodThreshold = 1.0f;                   // initial LOD pixel-error threshold
        std::string cam;                             // parsed by Camera (S1.5)
        std::string mesh = "cube";                   // cube | sphere | knot | <path.obj>
        std::string screenshot;
        std::string screenshotUi;                    // full window incl. ImGui
    };

    Args parseArgs(int argc, char** argv)
    {
        Args args;
        for (int i = 1; i < argc; ++i)
        {
            if (std::strcmp(argv[i], "--selftest-window") == 0)
                args.selftest = true;
            else if (std::strcmp(argv[i], "--selftest-shader") == 0)
                args.selftestShader = true;
            else if (std::strcmp(argv[i], "--resize-at") == 0 && i + 1 < argc)
                args.resizeAt = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "--validation") == 0)
                args.validation = true;
            else if (std::strcmp(argv[i], "--no-validation") == 0)
                args.validation = false;
            else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc)
                args.frames = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc)
                sscanf_s(argv[++i], "%dx%d", &args.width, &args.height);
            else if (std::strcmp(argv[i], "--cam") == 0 && i + 1 < argc)
                args.cam = argv[++i];
            else if (std::strcmp(argv[i], "--mesh") == 0 && i + 1 < argc)
                args.mesh = argv[++i];
            else if (std::strcmp(argv[i], "--kgeo") == 0 && i + 1 < argc)
                args.kgeo = argv[++i];
            else if (std::strcmp(argv[i], "--depth-probe") == 0)
                args.depthProbe = true;
            else if (std::strcmp(argv[i], "--lod-threshold") == 0 && i + 1 < argc)
                args.lodThreshold = std::atof(argv[++i]);
            else if (std::strcmp(argv[i], "--screenshot") == 0 && i + 1 < argc)
                args.screenshot = argv[++i];
            else if (std::strcmp(argv[i], "--screenshot-ui") == 0 && i + 1 < argc)
                args.screenshotUi = argv[++i];
        }
        if (args.width <= 0 || args.height <= 0)
        {
            args.width = 1600;
            args.height = 900;
        }
        return args;
    }

    /// Deterministic clear color for a frame index: exact k/255 values so the
    /// float32 -> UNORM round trip reproduces the same byte in every run.
    void clearColorForFrame(int frame, kenshin::f32 out[4])
    {
        const int r = (frame * 37 + 11) & 0xFF;
        const int g = (frame * 61 + 3) & 0xFF;
        const int b = (frame * 97 + 200) & 0xFF;
        out[0] = static_cast<kenshin::f32>(r) / 255.0f;
        out[1] = static_cast<kenshin::f32>(g) / 255.0f;
        out[2] = static_cast<kenshin::f32>(b) / 255.0f;
        out[3] = 1.0f;
    }

    void selftestWindow()
    {
        kenshin::core::Window window;
        if (!window.create({"KENSHIN S1.1 selftest", 1600, 900}))
        {
            KENSHIN_ERROR("selftest: window create FAILED");
            std::exit(2);
        }
        KENSHIN_INFO("selftest: step1 window-created ok");

        // Programmatic resize -> WM_SIZE should surface via pumpEvents.
        // (SetWindowPos takes the outer rect; convert a 1024x768 client rect first.)
        const HWND hwnd = static_cast<HWND>(window.nativeHandle());
        RECT clientRect {0, 0, 1024, 768};
        AdjustWindowRect(&clientRect, WS_OVERLAPPEDWINDOW, FALSE);
        SetWindowPos(hwnd, nullptr, 100, 100,
                     clientRect.right - clientRect.left,
                     clientRect.bottom - clientRect.top,
                     SWP_NOZORDER | SWP_NOACTIVATE);
        window.pumpEvents();
        if (window.width() == 1024 && window.height() == 768)
            KENSHIN_INFO("selftest: step2 resize 1024x768 ok");
        else
        {
            KENSHIN_ERROR("selftest: step2 resize FAILED (got {}x{})",
                          window.width(), window.height());
            std::exit(2);
        }

        // Inject a key event (synthesized message) -> Input state
        SendMessageW(hwnd, WM_KEYDOWN, 'A', 0);
        SendMessageW(hwnd, WM_KEYUP, 'A', 0);
        if (kenshin::core::Input::get().keyReleased('A'))
            KENSHIN_INFO("selftest: step3 key event ok");
        else
        {
            KENSHIN_ERROR("selftest: step3 key event FAILED");
            std::exit(2);
        }

        window.requestClose();
        window.pumpEvents();
        KENSHIN_INFO("selftest: PASSED");
    }
} // namespace



namespace
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
    // ---- .kgeo LOD preview (S3.4) -----------------------------------------
    struct KgeoPreview
    {
        kenshin::geometry::ClusterDag dag;
        kenshin::u32 meshHandle = kenshin::backend::IRenderBackend::kInvalidMeshHandle;
        std::vector<kenshin::u32> cut;
        kenshin::geometry::CutStats lastStats;
        float pixelThreshold = 1.0f;
        bool lodColors = true;   // false = shaded
        bool needRebuild = true;
        int framesSinceChange = 0;
        std::vector<float> history; // triangle count over threshold changes
    };

    bool loadKgeoPreview(const std::string& path, KgeoPreview& preview,
                         kenshin::backend::IRenderBackend* backend, std::string& log)
    {
        if (!kenshin::geometry::readKGeo(path, preview.dag, log))
            return false;
        preview.needRebuild = true;
        (void)backend;
        return true;
    }

    /// Rebuild the drawable mesh from the current cut (baseline path):
    /// concatenate the cut clusters' local vertices/indices into one buffer.
    void rebuildCutMesh(KgeoPreview& preview,
                        kenshin::backend::IRenderBackend* backend)
    {
        const auto& dag = preview.dag;
        kenshin::assets::MeshData mesh;
        for (kenshin::u32 c : preview.cut)
        {
            const auto& cluster = dag.clusters[c];
            const kenshin::u32 base = mesh.vertexCount();
            for (kenshin::u32 v = 0; v < cluster.vertexCount; ++v)
            {
                const kenshin::u32 src = cluster.vertexOffset + v;
                for (int k = 0; k < 3; ++k)
                    mesh.positions.push_back(dag.positions[src * 3 + k]);
                for (int k = 0; k < 3; ++k)
                    mesh.normals.push_back(dag.normals[src * 3 + k]);
                mesh.uvs.push_back(dag.uvs[src * 2]);
                mesh.uvs.push_back(dag.uvs[src * 2 + 1]);
            }
            for (kenshin::u32 i = 0; i < cluster.indexCount; ++i)
                mesh.indices.push_back(base + dag.indices[cluster.indexOffset + i]);
        }
        if (preview.meshHandle != kenshin::backend::IRenderBackend::kInvalidMeshHandle)
            backend->destroyMeshBuffer(preview.meshHandle);
        preview.meshHandle = kenshin::backend::IRenderBackend::kInvalidMeshHandle;
        if (mesh.indices.empty())
            return;
        kenshin::backend::IRenderBackend::MeshUpload upload;
        upload.positions = mesh.positions.data();
        upload.normals = mesh.normals.data();
        upload.uvs = mesh.uvs.data();
        upload.vertexCount = mesh.vertexCount();
        upload.indices = mesh.indices.data();
        upload.indexCount = static_cast<kenshin::u32>(mesh.indices.size());
        preview.meshHandle = backend->createMeshBuffer(upload);
    }
} // namespace

namespace
{
    struct UiState
    {
        ImVec2 viewportSize {1280.0f, 720.0f};
        bool viewportHovered = false;
        int controllerIndex = 0; // 0 orbit, 1 first person
    };

    void drawUi(UiState& ui, void* viewportTexture,
                kenshin::core::FrameStats<240>& stats,
                kenshin::scene::Camera& camera,
                kenshin::scene::ICameraController*& controller,
                kenshin::scene::OrbitCameraController& orbit,
                kenshin::scene::FirstPersonCameraController& firstPerson,
                KgeoPreview* uiKgeo)
    {
        // Full-window dockspace host: every panel (viewport included) can be
        // dragged out, docked, split; layout persists across sessions.
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(),
                                     ImGuiDockNodeFlags_PassthruCentralNode);

        // ---- Viewport panel --------------------------------------------
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowPos(ImVec2(60.0f, 60.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(640.0f, 480.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Viewport");
        const ImVec2 avail = ImGui::GetContentRegionAvail();
        if (avail.x >= 1.0f && avail.y >= 1.0f)
            ui.viewportSize = avail;
        if (viewportTexture && avail.x >= 1.0f && avail.y >= 1.0f)
            ImGui::Image(ImTextureRef(viewportTexture), avail);
        ui.viewportHovered = ImGui::IsItemHovered();
        ImGui::End();
        ImGui::PopStyleVar();

        // ---- Frame pacing stats ----------------------------------------
        ImGui::SetNextWindowPos(ImVec2(740.0f, 60.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(340.0f, 220.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Frame Stats");
        ImGui::Text("FPS %.1f", stats.fps());
        ImGui::Text("avg  %.3f ms", stats.averageMs());
        ImGui::Text("std  %.3f ms", stats.stddevMs());
        ImGui::Text("max  %.3f ms", stats.maxMs());
        ImGui::Text("samples %zu", stats.count());
        ImGui::Separator();
        static float intervals[240] = {};
        kenshin::usize count = 0;
        stats.copyTo(intervals, count);
        if (count > 1)
            ImGui::PlotLines("##intervals", intervals, static_cast<int>(count), 0,
                             nullptr, 0.0f, 50.0f, ImVec2(0, 60));
        ImGui::End();

        // ---- Camera panel ----------------------------------------------
        ImGui::SetNextWindowPos(ImVec2(60.0f, 580.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(380.0f, 300.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Camera");
        const char* items[] = { orbit.name(), firstPerson.name() };
        if (ImGui::Combo("controller", &ui.controllerIndex, items, 2))
            controller = ui.controllerIndex == 0
                       ? static_cast<kenshin::scene::ICameraController*>(&orbit)
                       : static_cast<kenshin::scene::ICameraController*>(&firstPerson);
        ImGui::SliderFloat("yaw", &camera.yawDeg, -180.0f, 180.0f, "%.1f deg");
        ImGui::SliderFloat("pitch", &camera.pitchDeg, -89.0f, 89.0f, "%.1f deg");
        ImGui::SliderFloat("fov", &camera.fovYDeg, 20.0f, 120.0f, "%.0f deg");
        if (ui.controllerIndex == 0)
        {
            ImGui::SliderFloat("distance", &orbit.distance, 0.1f, 200.0f, "%.2f m");
            ImGui::DragFloat3("target", &orbit.target.x, 0.05f);
        }
        else
        {
            ImGui::SliderFloat("speed", &firstPerson.moveSpeed, 0.5f, 50.0f, "%.1f m/s");
        }
        const glm::vec3 pos = camera.position;
        ImGui::Text("pos (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
        ImGui::Text("view orthonormal: %s",
                    camera.isOrthonormal() ? "yes" : "NO");
        ImGui::Text("(LMB drag rotates, wheel zooms, WASD/QE)");
        ImGui::End();

        if (uiKgeo)
        {
            ImGui::Begin("LOD Preview");
            ImGui::SliderFloat("pixel error", &uiKgeo->pixelThreshold, 0.05f, 32.0f,
                               "%.2f px", ImGuiSliderFlags_Logarithmic);
            ImGui::Checkbox("level coloring (shading off)", &uiKgeo->lodColors);
            const auto& st = uiKgeo->lastStats;
            ImGui::Text("cut: %u clusters / %u tris", st.clusterCount,
                        st.triangleCount);
            ImGui::Text("levels %d..%d", st.minLevel, st.maxLevel);
            ImGui::Text("source: %u tris", uiKgeo->dag.sourceTriangleCount);
            if (!uiKgeo->history.empty() &&
                ImGui::BeginChild("history", ImVec2(0, 80), true))
            {
                ImGui::PlotLines("##tris", uiKgeo->history.data(),
                                 static_cast<int>(uiKgeo->history.size()), 0,
                                 nullptr, 0.0f,
                                 static_cast<float>(uiKgeo->dag.sourceTriangleCount),
                                 ImVec2(0, 60));
                ImGui::EndChild();
            }
            ImGui::End();
        }
    }
} // namespace

namespace
{
    /// Locate engine/shaders: relative to the exe (build/<preset>/app/<cfg>),
    /// falling back to the working directory.
    std::string findShaderRoot()
    {
        wchar_t exePath[MAX_PATH] {};
        GetModuleFileNameW(nullptr, exePath, MAX_PATH);
        auto dir = std::filesystem::path(exePath).parent_path();
        const std::filesystem::path candidates[] = {
            dir / ".." / ".." / ".." / ".." / "engine" / "shaders",
            std::filesystem::path("engine/shaders"),
        };
        for (const auto& candidate : candidates)
        {
            std::error_code ec;
            if (std::filesystem::exists(candidate / "test_atomic.slang", ec))
                return std::filesystem::weakly_canonical(candidate, ec).string();
        }
        return "engine/shaders";
    }

    /// Point the loader at a validation layer shipped next to the exe.
    void setupValidationLayerPath()
    {
        wchar_t exePath[MAX_PATH] {};
        GetModuleFileNameW(nullptr, exePath, MAX_PATH);
        const auto exeDir = std::filesystem::path(exePath).parent_path();
        if (std::filesystem::exists(exeDir / "VkLayer_khronos_validation.json"))
        {
            SetEnvironmentVariableW(L"VK_LAYER_PATH", exeDir.c_str());
            KENSHIN_INFO("validation layer path -> {}", exeDir.string());
        }
    }

    /// S1.4 smoke: compile test_atomic.slang, run it, verify the uint64
    /// atomic path (the VisBuffer primitive). Exits 0 on success.
    int selftestShader(bool validation)
    {
        kenshin::manager::ShaderManager shaders;
        if (!shaders.init(findShaderRoot()))
        {
            KENSHIN_ERROR("selftest-shader: ShaderManager init FAILED");
            return 2;
        }
        KENSHIN_INFO("selftest-shader: step1 shader manager ok");

        // Error-path: a deliberately broken source must fail with diagnostics.
        {
            std::vector<kenshin::u32> bogusSpirv;
            std::string errLog;
            const std::string brokenSource =
                "[shader(\"compute\")]\n"
                "[numthreads(1,1,1)]\n"
                "void main(uint tid : SV_DispatchThreadID)\n"
                "{\n"
                "    totally_undefined_function_12345();\n"
                "}\n";
            if (shaders.compileSource(brokenSource, "broken.slang", bogusSpirv, errLog))
            {
                KENSHIN_ERROR("selftest-shader: step2 ERROR-PATH FAILED (broken shader compiled)");
                return 2;
            }
            const bool hasError = errLog.find("error") != std::string::npos;
            KENSHIN_INFO("selftest-shader: step2 error-path ok (diagnostics: {})",
                         errLog);
            if (!hasError)
            {
                KENSHIN_ERROR("selftest-shader: step2 diagnostics lack an 'error' marker");
                return 2;
            }
        }

        // Compile the smoke shader.
        std::vector<kenshin::u32> spirv;
        std::string log;
        if (!shaders.compileFile("test_atomic.slang", spirv, log))
        {
            KENSHIN_ERROR("selftest-shader: step3 compile FAILED: {}", log);
            return 2;
        }
        KENSHIN_INFO("selftest-shader: step3 compile ok ({} SPIR-V words)", spirv.size());

        // Device (no window) + run.
        if (validation)
            setupValidationLayerPath();
        auto backend = kenshin::backend::createVulkanBackend();
        kenshin::backend::BackendInitParams params;
        if (!backend->initialize(params))
        {
            KENSHIN_ERROR("selftest-shader: backend init FAILED");
            return 2;
        }

        const kenshin::u32 input[4] = {1, 2, 3, 4};
        kenshin::u64 atomicInit = 0;
        kenshin::u32 outputInit = 0;
        kenshin::backend::IRenderBackend::Job job;
        job.spirv = spirv.data();
        job.spirvBytes = spirv.size() * sizeof(kenshin::u32);
        job.entryPoint = "main";
        job.groupCountX = 1;
        job.buffers[0] = { input, sizeof(input), false };              // in
        job.buffers[1] = { &atomicInit, sizeof(atomicInit), true };    // u64 atomic out
        job.buffers[2] = { &outputInit, sizeof(outputInit), true };    // reduce out

        std::vector<kenshin::u8> outs[kenshin::backend::IRenderBackend::Job::kMaxBuffers];
        if (!backend->runComputeOneshot(job, outs))
        {
            KENSHIN_ERROR("selftest-shader: step4 compute FAILED");
            backend->shutdown();
            return 2;
        }
        const kenshin::u64 atomicMax = *reinterpret_cast<const kenshin::u64*>(outs[1].data());
        const kenshin::u32 reduce = *reinterpret_cast<const kenshin::u32*>(outs[2].data());
        backend->shutdown();

        KENSHIN_INFO("selftest-shader: step4 compute ok: atomicMax(u64)={}, reduce={}",
                     atomicMax, reduce);
        if (atomicMax != 16 || reduce != 10)
        {
            KENSHIN_ERROR("selftest-shader: FAILED (expected atomicMax=16, reduce=10)");
            return 2;
        }
        KENSHIN_INFO("selftest-shader: PASSED (uint64 atomic path verified)");
        return 0;
    }
} // namespace

int main(int argc, char** argv)
{
    const Args args = parseArgs(argc, argv);

    kenshin::log::init();
    KENSHIN_INFO("{}", kenshin::getBanner());

    if (args.selftest)
    {
        selftestWindow();
        kenshin::log::shutdown();
        return 0;
    }

    if (args.selftestShader)
    {
        const int code = selftestShader(args.validation);
        kenshin::log::shutdown();
        return code;
    }

    kenshin::scene::Camera camera;
    kenshin::scene::OrbitCameraController orbit;
    kenshin::scene::FirstPersonCameraController firstPerson;
    kenshin::scene::ICameraController* controller = &orbit;
    if (!args.cam.empty())
    {
        float yaw = 35.0f, pitch = 20.0f, dist = 8.0f, tx = 0.0f, ty = 0.5f, tz = 0.0f;
        if (sscanf_s(args.cam.c_str(), "%f,%f,%f,%f,%f,%f",
                     &yaw, &pitch, &dist, &tx, &ty, &tz) >= 3)
        {
            camera.yawDeg = yaw;
            camera.pitchDeg = pitch;
            orbit.distance = dist;
            orbit.target = {tx, ty, tz};
            KENSHIN_INFO("camera params applied (orbit yaw={} pitch={} dist={})", yaw, pitch, dist);
        }
        else
            KENSHIN_WARN("could not parse --cam '{}' (want yaw,pitch,dist[,tx,ty,tz])", args.cam);
    }

    kenshin::core::Window window;
    if (!window.create({"KENSHIN Demo", static_cast<kenshin::u32>(args.width),
                        static_cast<kenshin::u32>(args.height)}))
    {
        KENSHIN_ERROR("window create failed");
        kenshin::log::shutdown();
        return 1;
    }
    window.onResize = [](kenshin::u32 w, kenshin::u32 h) { KENSHIN_INFO("resize {}x{}", w, h); };
    window.onClose  = []() { KENSHIN_INFO("closing (user)"); };

    // Vulkan device init (S1.2): capability report + int64 atomics gate.
    // Dev default: validation ON. The layer ships next to the exe (vcpkg);
    // point the loader at it before instance creation.
    if (args.validation)
        setupValidationLayerPath();

    auto backend = kenshin::backend::createVulkanBackend();
    kenshin::backend::BackendInitParams backendParams;
    backendParams.nativeWindowHandle = window.nativeHandle();
    backendParams.enableValidation = args.validation;
    if (!backend->initialize(backendParams))
    {
        KENSHIN_ERROR("vulkan backend init failed");
        backend->shutdown();
        kenshin::log::shutdown();
        return 1;
    }
    if (!backend->caps().enabledShaderBufferInt64Atomics)
    {
        // HARD GATE: the 64-bit atomic VisBuffer depends on this feature.
        KENSHIN_ERROR("GATE FAILED: shaderBufferInt64Atomics not supported — "
                      "stop and decide the 32-bit packed fallback (see staged-plan risk table)");
        backend->shutdown();
        kenshin::log::shutdown();
        return 2;
    }
    KENSHIN_INFO("[vulkan] gate shaderBufferInt64Atomics: OK");

    // ImGui: docking workspace over the whole window; the viewport render is a
    // dockable panel (draggable out, dockable anywhere).
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    // Persist the layout next to the exe (default is the CWD, which is
    // whatever directory the app was launched from).
    {
        wchar_t exePath[MAX_PATH] {};
        GetModuleFileNameW(nullptr, exePath, MAX_PATH);
        static std::string iniPath =
            (std::filesystem::path(exePath).parent_path() / "imgui.ini").string();
        io.IniFilename = iniPath.c_str();
    }
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::StyleColorsDark();
    window.onRawMessage = [](void* hwnd, unsigned msg, kenshin::usize wparam, kenshin::isize lparam) {
        return ImGui_ImplWin32_WndProcHandler(static_cast<HWND>(hwnd), msg,
                                              static_cast<WPARAM>(wparam),
                                              static_cast<LPARAM>(lparam)) != 0;
    };
    if (!backend->initImGuiBackend(window.nativeHandle(), window.width(), window.height()))
    {
        KENSHIN_ERROR("ImGui backend init failed");
        backend->shutdown();
        ImGui::DestroyContext();
        kenshin::log::shutdown();
        return 1;
    }

    // ---- baseline mesh path (S2.2) --------------------------------------
    kenshin::assets::MeshData demoMesh;
    {
        std::string meshLog;
        if (!loadDemoMesh(args.mesh, demoMesh, meshLog))
        {
            KENSHIN_ERROR("mesh load failed: {}", meshLog);
            backend->shutdown();
            ImGui::DestroyContext();
            kenshin::log::shutdown();
            return 1;
        }
    }

    kenshin::manager::ShaderManager shaders;
    kenshin::u32 meshHandle = kenshin::backend::IRenderBackend::kInvalidMeshHandle;
    glm::vec3 meshKd(0.8f, 0.7f, 0.35f);
    if (shaders.init(findShaderRoot()))
    {
        std::vector<kenshin::u32> vsSpirv, fsSpirv;
        std::string shaderLog;
        if (shaders.compileFile("baseline.slang", vsSpirv, shaderLog, "vsMain") &&
            shaders.compileFile("baseline.slang", fsSpirv, shaderLog, "fsMain"))
        {
            backend->stageBaselineShaders(vsSpirv.data(), vsSpirv.size() * 4,
                                          fsSpirv.data(), fsSpirv.size() * 4);
            kenshin::backend::IRenderBackend::MeshUpload upload;
            upload.positions = demoMesh.positions.data();
            upload.normals = demoMesh.normals.data();
            upload.uvs = demoMesh.uvs.data();
            upload.vertexCount = demoMesh.vertexCount();
            upload.indices = demoMesh.indices.data();
            upload.indexCount = static_cast<kenshin::u32>(demoMesh.indices.size());
            meshHandle = backend->createMeshBuffer(upload);
            KENSHIN_INFO("[baseline] mesh '{}' uploaded: {} verts / {} tris (handle {})",
                         args.mesh, demoMesh.vertexCount(), demoMesh.triangleCount(),
                         meshHandle);
            meshKd = glm::vec3(demoMesh.material.kd);
        }
        else
        {
            KENSHIN_WARN("[baseline] shader compile failed; viewport shows only the "
                         "clear color. Log: {}", shaderLog);
        }
    }

    // ---- .kgeo LOD preview (S3.4) ---------------------------------------
    KgeoPreview kgeoPreview;
    bool kgeoMode = false;
    if (!args.kgeo.empty())
    {
        std::string kgeoLog;
        if (loadKgeoPreview(args.kgeo, kgeoPreview, backend.get(), kgeoLog))
        {
            kgeoMode = true;
            kgeoPreview.pixelThreshold = args.lodThreshold;
            KENSHIN_INFO("[kgeo] loaded: {} clusters / {} groups / {} levels",
                         kgeoPreview.dag.clusters.size(),
                         kgeoPreview.dag.groups.size(),
                         [&] { kenshin::i32 m = 0;
                               for (auto& g : kgeoPreview.dag.groups)
                                   m = std::max(m, g.lodLevel);
                               return m + 1; }());
        }
        else
            KENSHIN_WARN("[kgeo] load failed: {} (continuing with --mesh)", kgeoLog);
    }

    UiState ui;
    ui.viewportSize = ImVec2(static_cast<float>(window.width()),
                             static_cast<float>(window.height()));
    kenshin::core::FrameStats<240> stats;
    kenshin::Timer frameTimer;
    bool cameraChecked = false;

    int frame = 0;
    while (window.pumpEvents())
    {
        auto& input = kenshin::core::Input::get();
        if (input.keyPressed(VK_ESCAPE))
        {
            KENSHIN_INFO("ESC pressed -> close");
            window.requestClose();
        }
        input.nextFrame();

        // Programmatic mid-run resize (S2.3 verification): shrink then grow;
        // the swapchain must rebuild and rendering must continue cleanly.
        if (args.resizeAt >= 0 && frame == args.resizeAt)
        {
            const HWND hwnd = static_cast<HWND>(window.nativeHandle());
            for (const int pass : {0, 1})
            {
                RECT r {0, 0, pass == 0 ? 640 : 960, pass == 0 ? 480 : 720};
                AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
                SetWindowPos(hwnd, nullptr, 60, 60,
                             r.right - r.left, r.bottom - r.top,
                             SWP_NOZORDER | SWP_NOACTIVATE);
                window.pumpEvents();
            }
            KENSHIN_INFO("[resize-test] resized at frame {}", frame);
        }

        const kenshin::f32 dt = static_cast<kenshin::f32>(frameTimer.elapsedSeconds());
        frameTimer.reset();

        const bool isLastFrame = args.frames >= 0 && frame + 1 >= args.frames;

        kenshin::f32 color[4];
        clearColorForFrame(frame, color);
        if (isLastFrame)
        {
            const int r = static_cast<int>(color[0] * 255.0f + 0.5f);
            const int g = static_cast<int>(color[1] * 255.0f + 0.5f);
            const int b = static_cast<int>(color[2] * 255.0f + 0.5f);
            KENSHIN_INFO("frame {} expected color = ({},{},{})", frame, r, g, b);
        }

        kenshin::backend::FrameBeginInfo beginInfo;
        beginInfo.width = static_cast<kenshin::u32>(ui.viewportSize.x);
        beginInfo.height = static_cast<kenshin::u32>(ui.viewportSize.y);
        beginInfo.clearColor[0] = color[0];
        beginInfo.clearColor[1] = color[1];
        beginInfo.clearColor[2] = color[2];
        beginInfo.clearColor[3] = color[3];
        if (!backend->beginFrame(beginInfo))
        {
            KENSHIN_ERROR("beginFrame failed");
            break;
        }

        drawUi(ui, backend->viewportImGuiTexture(), stats, camera, controller,
               orbit, firstPerson, kgeoMode ? &kgeoPreview : nullptr);

        // Controller input (only while the viewport item is active).
        kenshin::scene::ControllerInput controllerInput;
        controllerInput.deltaTime = dt > 0.0f ? dt : 1.0f / 120.0f;
        controllerInput.rotateActive =
            ui.viewportHovered && ImGui::IsMouseDragging(ImGuiMouseButton_Left);
        if (ui.viewportHovered)
        {
            controllerInput.mouseDeltaX = io.MouseDelta.x;
            controllerInput.mouseDeltaY = io.MouseDelta.y;
            controllerInput.wheel = io.MouseWheel;
        }
        controllerInput.moveF = ImGui::IsKeyDown(ImGuiKey_W);
        controllerInput.moveB = ImGui::IsKeyDown(ImGuiKey_S);
        controllerInput.moveL = ImGui::IsKeyDown(ImGuiKey_A);
        controllerInput.moveR = ImGui::IsKeyDown(ImGuiKey_D);
        controllerInput.moveUp = ImGui::IsKeyDown(ImGuiKey_E);
        controllerInput.moveDown = ImGui::IsKeyDown(ImGuiKey_Q);
        controllerInput.fast = io.KeyShift;
        controller->update(camera, controllerInput);

        if (!cameraChecked && camera.isOrthonormal())
        {
            KENSHIN_INFO("[camera] view basis orthonormal: OK");
            cameraChecked = true;
        }

        // .kgeo LOD preview: re-select the cut when the threshold moves,
        // rebuild the concatenated mesh, then draw through the same path.
        if (kgeoMode)
        {
            std::vector<kenshin::u32> cut;
            const kenshin::f32 camPos[3] = {camera.position.x, camera.position.y,
                                            camera.position.z};
            kenshin::geometry::selectLodCut(kgeoPreview.dag, kgeoPreview.pixelThreshold,
                                            static_cast<kenshin::u32>(ui.viewportSize.y),
                                            camera.fovYDeg, camPos, cut);
            if (cut != kgeoPreview.cut)
            {
                kgeoPreview.cut = std::move(cut);
                kgeoPreview.needRebuild = true;
                kgeoPreview.lastStats =
                    kenshin::geometry::summarizeCut(kgeoPreview.dag, kgeoPreview.cut);
                kgeoPreview.history.push_back(
                    static_cast<float>(kgeoPreview.lastStats.triangleCount));
                if (kgeoPreview.history.size() > 240)
                    kgeoPreview.history.erase(kgeoPreview.history.begin());
                KENSHIN_INFO("[kgeo] cut: {} clusters / {} tris (levels {}..{}) "
                             "at {:.2f}px",
                             kgeoPreview.lastStats.clusterCount,
                             kgeoPreview.lastStats.triangleCount,
                             kgeoPreview.lastStats.minLevel,
                             kgeoPreview.lastStats.maxLevel,
                             kgeoPreview.pixelThreshold);
            }
            if (kgeoPreview.needRebuild)
            {
                rebuildCutMesh(kgeoPreview, backend.get());
                kgeoPreview.needRebuild = false;
            }
        }

        // Baseline draw (S2.2): the mesh renders into the viewport target
        // between beginFrame and endFrame.
        kenshin::u32 activeMesh =
            kgeoMode ? kgeoPreview.meshHandle : meshHandle;
        // Depth diagnostic (--depth-probe): far red cube drawn AFTER a near
        // blue cube; the near one must stay visible (LESS+clear1+ZO).
        static kenshin::u32 probeNear = kenshin::backend::IRenderBackend::kInvalidMeshHandle;
        static kenshin::u32 probeFar = kenshin::backend::IRenderBackend::kInvalidMeshHandle;
        if (args.depthProbe && probeNear == kenshin::backend::IRenderBackend::kInvalidMeshHandle)
        {
            kenshin::assets::MeshData m;
            std::string probeLog;
            loadDemoMesh("cube", m, probeLog);
            kenshin::backend::IRenderBackend::MeshUpload up;
            up.positions = m.positions.data();
            up.normals = m.normals.data();
            up.uvs = m.uvs.data();
            up.vertexCount = m.vertexCount();
            up.indices = m.indices.data();
            up.indexCount = static_cast<kenshin::u32>(m.indices.size());
            probeNear = backend->createMeshBuffer(up);
            probeFar = backend->createMeshBuffer(up);
        }
        if (args.depthProbe)
        {
            activeMesh = kenshin::backend::IRenderBackend::kInvalidMeshHandle;
            for (int pass = 0; pass < 2; ++pass)
            {
                // near cube (z=+0.5 toward camera), far cube (z=-0.5)
                const bool nearPass = pass == 0;
                kenshin::scene::Camera probeCam = camera;
                probeCam.position = nearPass
                    ? glm::vec3(0.0f, 0.0f, 3.0f)
                    : glm::vec3(0.0f, 0.0f, 3.0f);
                // draw both with an identity-ish view from +Z looking at origin
                probeCam.yawDeg = -90.0f;  // forward = (sin(-90),0,-cos(-90)) = (-1,0,0)?? no
                // simpler: build viewProj manually: camera at (0,0,dist) looking -Z
                const float dist = 4.0f;
                glm::vec3 eye(0.0f, 0.0f, dist);
                glm::vec3 fwd(0.0f, 0.0f, -1.0f);
                glm::mat4 viewM = glm::lookAtRH(eye, eye + fwd, glm::vec3(0, 1, 0));
                glm::mat4 projM = camera.projection(
                    ui.viewportSize.x / glm::max(ui.viewportSize.y, 1.0f));
                const float zOff = nearPass ? 0.75f : -0.75f;
                glm::mat4 modelM = glm::translate(glm::mat4(1.0f), glm::vec3(nearPass ? -0.35f : 0.35f, 0.0f, zOff));
                glm::mat4 mvp = glm::transpose(projM * viewM * modelM);
                kenshin::backend::IRenderBackend::MeshDraw draw;
                draw.mesh = nearPass ? probeNear : probeFar;
                std::memcpy(draw.viewProj, &mvp, sizeof(draw.viewProj));
                const kenshin::f32 kd[4] = {nearPass ? 0.1f : 0.9f, 0.2f,
                                            nearPass ? 0.9f : 0.1f, 1.0f};
                std::memcpy(draw.kd, kd, sizeof(draw.kd));
                const kenshin::f32 light[4] = {0.0f, 0.0f, 1.0f, 0.0f};
                std::memcpy(draw.lightDir, light, sizeof(draw.lightDir));
                const kenshin::f32 cp[4] = {0.0f, 0.0f, dist, 0.0f};
                std::memcpy(draw.camPos, cp, sizeof(draw.camPos));
                backend->drawMesh(draw);
            }
        }
        if (activeMesh != kenshin::backend::IRenderBackend::kInvalidMeshHandle &&
            ui.viewportSize.x > 1.0f && ui.viewportSize.y > 1.0f)
        {
            const kenshin::f32 aspect = ui.viewportSize.x / ui.viewportSize.y;
            // Slang push-constant matrices read the 64 bytes row-major;
            // GLM stores column-major, so upload the transpose.
            const glm::mat4 viewProj = glm::transpose(camera.projection(aspect) * camera.view());
            kenshin::backend::IRenderBackend::MeshDraw draw;
            draw.mesh = activeMesh;
            std::memcpy(draw.viewProj, &viewProj, sizeof(draw.viewProj));
            kenshin::f32 kdArr[4] = {meshKd.r, meshKd.g, meshKd.b, 1.0f};
            if (kgeoMode)
            {
                kdArr[0] = 0.75f; kdArr[1] = 0.75f; kdArr[2] = 0.78f; kdArr[3] = 1.0f;
            }
            std::memcpy(draw.kd, kdArr, sizeof(draw.kd));
            // Studio light from upper-left-front; camera position feeds the
            // Blinn-Phong specular half-vector.
            const kenshin::f32 light[4] = {0.35f, 0.8f, 0.45f, 0.0f};
            std::memcpy(draw.lightDir, light, sizeof(draw.lightDir));
            const kenshin::f32 camArr[4] = {camera.position.x, camera.position.y,
                                            camera.position.z, 0.0f};
            std::memcpy(draw.camPos, camArr, sizeof(draw.camPos));
            backend->drawMesh(draw);
        }

        // Full-frame capture must be flagged before endFrame records it.
        const bool exiting = isLastFrame || !window.isOpen();
        if (!args.screenshotUi.empty() && exiting)
            backend->requestFullFrameCapture();

        backend->endFrame();
        stats.push(dt * 1000.0f);
        if (frame > 0 && frame % 60 == 0)
            KENSHIN_INFO("[pacing] avg={:.3f}ms std={:.3f}ms max={:.3f}ms fps={:.1f}",
                         stats.averageMs(), stats.stddevMs(), stats.maxMs(), stats.fps());

        if (!args.screenshot.empty() && exiting)
        {
            std::vector<kenshin::u8> rgba;
            kenshin::u32 w = 0, h = 0;
            if (const auto parent = std::filesystem::path(args.screenshot).parent_path();
                !parent.empty())
                std::filesystem::create_directories(parent);
            if (backend->readbackFrame(rgba, w, h) &&
                kenshin::util::writeBmp24(args.screenshot, rgba, w, h))
                KENSHIN_INFO("screenshot written: {} ({}x{})", args.screenshot, w, h);
            else
                KENSHIN_ERROR("screenshot FAILED");
        }
        if (!args.screenshotUi.empty() && exiting)
        {
            std::vector<kenshin::u8> rgba;
            kenshin::u32 w = 0, h = 0;
            if (const auto parent = std::filesystem::path(args.screenshotUi).parent_path();
                !parent.empty())
                std::filesystem::create_directories(parent);
            if (backend->readbackFullFrame(rgba, w, h) &&
                kenshin::util::writeBmp24(args.screenshotUi, rgba, w, h))
                KENSHIN_INFO("ui screenshot written: {} ({}x{})", args.screenshotUi, w, h);
            else
                KENSHIN_ERROR("ui screenshot FAILED");
        }

        ++frame;
        if (isLastFrame)
        {
            KENSHIN_INFO("reached --frames {} -> close", args.frames);
            window.requestClose();
        }
    }

    KENSHIN_INFO("main loop exited after {} frames", frame);
    backend->shutdown();
    ImGui::DestroyContext();
    kenshin::log::shutdown();
    return 0;
}
