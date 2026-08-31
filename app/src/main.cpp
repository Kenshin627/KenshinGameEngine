// KEngineDemo entry (S1.1: window + input).
// Usage:
//   KEngineDemo                    normal windowed loop (ESC quits)
//   KEngineDemo --frames N         auto-exit after N pumped frames
//   KEngineDemo --selftest-window  selftest: create -> programmatic resize -> close,
//                                  asserts the event chain via log output

#include "ke/Version.h"
#include "ke/core/Input.h"
#include "ke/core/Log.h"
#include "ke/core/Window.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#include <cstring>
#include <string>

namespace
{
    struct Args
    {
        int  frames = -1;      // -1 = run until the user closes
        bool selftest = false;
    };

    Args parseArgs(int argc, char** argv)
    {
        Args args;
        for (int i = 1; i < argc; ++i)
        {
            if (std::strcmp(argv[i], "--selftest-window") == 0)
                args.selftest = true;
            else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc)
                args.frames = std::atoi(argv[++i]);
        }
        return args;
    }

    void selftestWindow()
    {
        ke::core::Window window;
        if (!window.create({"KEngine S1.1 selftest", 1600, 900}))
        {
            KE_ERROR("selftest: window create FAILED");
            std::exit(2);
        }
        KE_INFO("selftest: step1 window-created ok");

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
            KE_INFO("selftest: step2 resize 1024x768 ok");
        else
        {
            KE_ERROR("selftest: step2 resize FAILED (got {}x{})",
                     window.width(), window.height());
            std::exit(2);
        }

        // Inject a key event (synthesized message) -> Input state
        SendMessageW(hwnd, WM_KEYDOWN, 'A', 0);
        SendMessageW(hwnd, WM_KEYUP, 'A', 0);
        if (ke::core::Input::get().keyReleased('A'))
            KE_INFO("selftest: step3 key event ok");
        else
        {
            KE_ERROR("selftest: step3 key event FAILED");
            std::exit(2);
        }

        window.requestClose();
        window.pumpEvents();
        KE_INFO("selftest: PASSED");
    }
} // namespace

int main(int argc, char** argv)
{
    const Args args = parseArgs(argc, argv);

    ke::log::init();
    KE_INFO("{}", ke::getBanner());

    if (args.selftest)
    {
        selftestWindow();
        ke::log::shutdown();
        return 0;
    }

    ke::core::Window window;
    if (!window.create({"KEngine Demo", 1600, 900}))
    {
        KE_ERROR("window create failed");
        ke::log::shutdown();
        return 1;
    }
    window.onResize = [](ke::u32 w, ke::u32 h) { KE_INFO("resize {}x{}", w, h); };
    window.onClose  = []() { KE_INFO("closing (user)"); };

    int frame = 0;
    while (window.pumpEvents())
    {
        auto& input = ke::core::Input::get();
        if (input.keyPressed(VK_ESCAPE))
        {
            KE_INFO("ESC pressed -> close");
            window.requestClose();
        }
        input.nextFrame();

        ++frame;
        if (args.frames >= 0 && frame >= args.frames)
        {
            KE_INFO("reached --frames {} -> close", args.frames);
            window.requestClose();
        }
    }

    KE_INFO("main loop exited after {} frames", frame);
    ke::log::shutdown();
    return 0;
}
