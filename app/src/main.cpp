// KENSHINDemo entry: parse args, build the demo scene, boot the engine
// (subsystem registry), drive the frame loop. The app talks to the
// Renderer / MeshPool / DebugOverlay services only — the render backend
// stays engine-internal.
// Usage:
//   KENSHINDemo                                  windowed loop (ESC quits)
//   KENSHINDemo --frames N                       auto-exit after N frames
//   KENSHINDemo --no-validation                 disable the validation layer (dev default: ON)
//   KENSHINDemo --size WxH                       client size (default 1600x900)
//   KENSHINDemo --screenshot path.bmp            capture the last rendered frame

#include "DemoScene.h"
#include "KgeoPreview.h"

#include "kenshin/DebugOverlay.h"
#include "kenshin/Engine.h"
#include "kenshin/Version.h"
#include "kenshin/core/FrameStats.h"
#include "kenshin/core/Input.h"
#include "kenshin/core/Log.h"
#include "kenshin/core/Timer.h"
#include "kenshin/core/Window.h"
#include "kenshin/manager/MeshPool.h"
#include "kenshin/manager/ShaderManager.h"
#include "kenshin/renderer/Renderer.h"
#include "kenshin/scene/Camera.h"
#include "kenshin/scene/CameraController.h"
#include "kenshin/util/BmpWriter.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace
{
    struct Args
    {
        int  frames = -1;                            // -1 = run until the user closes
        int  width = 1600;
        int  height = 900;
        bool validation = true;   // dev default ON; --no-validation to disable
        std::string kgeo;                            // .kgeo virtual-geometry preview (S3.4)
        float lodThreshold = 1.0f;                   // initial LOD pixel-error threshold
        std::string mesh = "all";                    // all (full scene) | cube | sphere |
                                                    // torus | cylinder | cone | knot | <path.obj>
        std::string screenshot;
        std::string screenshotUi;                    // full window incl. ImGui
    };

    Args parseArgs(int argc, char** argv)
    {
        Args args;
        for (int i = 1; i < argc; ++i)
        {
            if (std::strcmp(argv[i], "--no-validation") == 0)
                args.validation = false;
            else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc)
                args.frames = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc)
                sscanf_s(argv[++i], "%dx%d", &args.width, &args.height);
            else if (std::strcmp(argv[i], "--mesh") == 0 && i + 1 < argc)
                args.mesh = argv[++i];
            else if (std::strcmp(argv[i], "--kgeo") == 0 && i + 1 < argc)
                args.kgeo = argv[++i];
            else if (std::strcmp(argv[i], "--lod-threshold") == 0 && i + 1 < argc)
                args.lodThreshold = static_cast<float>(std::atof(argv[++i]));
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

    /// Deterministic clear color: constant dark slate (exact k/255 values
    /// keep the float32 -> UNORM round trip byte-exact across runs; the
    /// last-frame log above is what screenshot verification asserts).
    constexpr kenshin::f32 kClearColor[4] = {14.0f / 255.0f, 16.0f / 255.0f,
                                             22.0f / 255.0f, 1.0f};

    /// Studio light from upper-left-front (TO light, world space) — scene
    /// state, passed to the renderer as pure data each frame.
    constexpr kenshin::f32 kStudioLight[3] = {0.35f, 0.8f, 0.45f};

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
} // namespace

int main(int argc, char** argv)
{
    const Args args = parseArgs(argc, argv);

    kenshin::log::init();
    KENSHIN_INFO("{}", kenshin::getBanner());

    // Engine facade: global subsystem registry (type-indexed get<T>()).
    kenshin::Engine& engine = kenshin::Engine::get();

    // Per-scene state stays local (cameras are view state, not services).
    kenshin::scene::Camera camera;
    kenshin::scene::CameraController controller;

    auto& window = engine.registerSubsystem<kenshin::core::Window>();
    if (!window.create({"KENSHIN Demo", static_cast<kenshin::u32>(args.width),
                        static_cast<kenshin::u32>(args.height)}))
    {
        KENSHIN_ERROR("window create failed");
        kenshin::log::shutdown();
        return 1;
    }
    window.onResize = [](kenshin::u32 w, kenshin::u32 h) { KENSHIN_INFO("resize {}x{}", w, h); };
    window.onClose  = []() { KENSHIN_INFO("closing (user)"); };

    // Shader service must know the shader root before the renderer's
    // pipeline compiles baseline.slang.
    auto& shaders = engine.registerSubsystem<kenshin::manager::ShaderManager>();
    shaders.init(findShaderRoot());

    // Renderer subsystem: creates + registers the backend (engine-internal),
    // enforces the int64-atomics gate, compiles and stages the baseline
    // pipeline. MeshPool: GPU mesh lifetime. DebugOverlay: debug UI
    // (resolves backend + window from the registry when it registers).
    auto& renderer = engine.registerSubsystem<kenshin::renderer::Renderer>(
        kenshin::renderer::Renderer::Config{args.validation});
    auto& meshPool = engine.registerSubsystem<kenshin::manager::MeshPool>();
    auto& overlay = engine.registerSubsystem<kenshin::DebugOverlay>();

    // ---- demo scene (S2.2): one instance of every shape on the X axis;
    // --mesh <name> picks one, --kgeo switches to the LOD preview instead.
    auto scene = args.kgeo.empty() ? demo::buildScene(args.mesh)
                                   : std::vector<demo::SceneShape>{};
    demo::uploadScene(scene, meshPool);

    // ---- .kgeo LOD preview (S3.4) ---------------------------------------
    demo::KgeoPreview kgeoPreview;
    bool kgeoMode = false;
    if (!args.kgeo.empty())
    {
        std::string kgeoLog;
        if (demo::loadKgeo(args.kgeo, kgeoPreview, kgeoLog))
        {
            kgeoMode = true;
            kgeoPreview.pixelThreshold = args.lodThreshold;
            overlay.setExtraPanel([&kgeoPreview] { demo::drawLodPanel(kgeoPreview); });
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

    kenshin::core::FrameStats<240> stats;
    kenshin::Timer frameTimer;
    bool cameraChecked = false;
    std::vector<kenshin::renderer::DrawItem> items;
    const auto writeScreenshot = [&renderer](const char* what, const std::string& path,
                                             bool fullFrame) {
        std::vector<kenshin::u8> rgba;
        kenshin::u32 w = 0, h = 0;
        if (const auto parent = std::filesystem::path(path).parent_path(); !parent.empty())
            std::filesystem::create_directories(parent);
        const bool read = fullFrame ? renderer.readbackFrame(rgba, w, h)
                                    : renderer.readbackViewport(rgba, w, h);
        if (read && kenshin::util::writeBmp24(path, rgba, w, h))
            KENSHIN_INFO("{} written: {} ({}x{})", what, path, w, h);
        else
            KENSHIN_ERROR("{} FAILED", what);
    };

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

        const kenshin::f32 dt = static_cast<kenshin::f32>(frameTimer.elapsedSeconds());
        frameTimer.reset();
        engine.beginFrame(dt);

        const bool isLastFrame = args.frames >= 0 && frame + 1 >= args.frames;

        if (isLastFrame)
        {
            const int r = static_cast<int>(kClearColor[0] * 255.0f + 0.5f);
            const int g = static_cast<int>(kClearColor[1] * 255.0f + 0.5f);
            const int b = static_cast<int>(kClearColor[2] * 255.0f + 0.5f);
            KENSHIN_INFO("frame {} expected color = ({},{},{})", frame, r, g, b);
        }

        const glm::vec2 vp = overlay.viewportSize();
        if (!renderer.beginFrame(static_cast<kenshin::u32>(vp.x),
                                 static_cast<kenshin::u32>(vp.y), kClearColor))
        {
            KENSHIN_ERROR("beginFrame failed");
            break;
        }

        overlay.draw(renderer.viewportTexture(), stats, camera, controller);

        controller.handleInput(camera, overlay.controllerInput(dt));
        controller.updateMatrices(camera, vp.x / glm::max(vp.y, 1.0f));

        if (!cameraChecked &&
            glm::abs(glm::dot(camera.forward(), camera.up())) < 1e-4f)
        {
            KENSHIN_INFO("[camera] view basis orthonormal: OK");
            cameraChecked = true;
        }

        if (kgeoMode)
            demo::updateCut(kgeoPreview, meshPool, camera, controller,
                            static_cast<kenshin::u32>(vp.y));

        // Assemble this frame's draw stream (pure data; the renderer layer
        // consumes it between beginFrame and endFrame).
        const bool drawable = vp.x > 1.0f && vp.y > 1.0f;
        items.clear();
        if (!kgeoMode && drawable && !scene.empty())
        {
            for (const auto& shape : scene)
                items.push_back({shape.handle,
                                 glm::translate(glm::mat4(1.0f), shape.offset),
                                 shape.kd});
        }
        else if (kgeoMode && drawable &&
                 kgeoPreview.meshHandle != kenshin::assets::kInvalidMeshHandle)
        {
            items.push_back({kgeoPreview.meshHandle, glm::mat4(1.0f),
                             glm::vec3(0.75f, 0.75f, 0.78f)});
        }

        const kenshin::renderer::FrameView view{
            camera.viewProjMatrix(), camera.position(),
            glm::vec3(kStudioLight[0], kStudioLight[1], kStudioLight[2])};
        renderer.submit(view, items);

        // Full-frame capture must be flagged before endFrame records it.
        const bool exiting = isLastFrame || !window.isOpen();
        if (!args.screenshotUi.empty() && exiting)
            renderer.requestFrameCapture();

        renderer.endFrame();
        engine.endFrame();
        stats.push(dt * 1000.0f);
        if (frame > 0 && frame % 60 == 0)
            KENSHIN_INFO("[pacing] avg={:.3f}ms std={:.3f}ms max={:.3f}ms fps={:.1f}",
                         stats.averageMs(), stats.stddevMs(), stats.maxMs(), stats.fps());

        if (exiting)
        {
            if (!args.screenshot.empty())
                writeScreenshot("screenshot", args.screenshot, false);
            if (!args.screenshotUi.empty())
                writeScreenshot("ui screenshot", args.screenshotUi, true);
        }

        ++frame;
        if (isLastFrame)
        {
            KENSHIN_INFO("reached --frames {} -> close", args.frames);
            window.requestClose();
        }
    }

    KENSHIN_INFO("main loop exited after {} frames", frame);
    // Reverse registration order: overlay context, mesh pool (all GPU
    // meshes), renderer pipeline, shader manager, backend, window.
    engine.shutdown();
    kenshin::log::shutdown();
    return 0;
}
