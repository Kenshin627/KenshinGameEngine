// Platform input (roadmap step 0.1): core::Key naming, the Input state
// machine, and the SDL3 Window's event -> Input path driven by injected
// events on a hidden window (no user interaction, CI-safe).

#include <SDL3/SDL.h>

#include <catch2/catch_test_macros.hpp>
#include <string>

#include "kenshin/core/Input.h"
#include "kenshin/core/Keys.h"
#include "kenshin/core/Window.h"

using kenshin::core::Input;
using kenshin::core::Key;
using kenshin::core::keyFromName;
using kenshin::core::keyName;
using kenshin::core::MouseButton;
using kenshin::core::Window;
using kenshin::core::WindowDesc;

// The Window casts SDL scancodes straight to Key: the two enumerations
// must agree value-for-value (both are USB HID usage IDs).
static_assert(static_cast<int>(Key::A) == SDL_SCANCODE_A);
static_assert(static_cast<int>(Key::Z) == SDL_SCANCODE_Z);
static_assert(static_cast<int>(Key::Num1) == SDL_SCANCODE_1);
static_assert(static_cast<int>(Key::Num0) == SDL_SCANCODE_0);
static_assert(static_cast<int>(Key::Enter) == SDL_SCANCODE_RETURN);
static_assert(static_cast<int>(Key::Escape) == SDL_SCANCODE_ESCAPE);
static_assert(static_cast<int>(Key::Space) == SDL_SCANCODE_SPACE);
static_assert(static_cast<int>(Key::Grave) == SDL_SCANCODE_GRAVE);
static_assert(static_cast<int>(Key::F1) == SDL_SCANCODE_F1);
static_assert(static_cast<int>(Key::F12) == SDL_SCANCODE_F12);
static_assert(static_cast<int>(Key::PageDown) == SDL_SCANCODE_PAGEDOWN);
static_assert(static_cast<int>(Key::Up) == SDL_SCANCODE_UP);
static_assert(static_cast<int>(Key::KpEnter) == SDL_SCANCODE_KP_ENTER);
static_assert(static_cast<int>(Key::Kp0) == SDL_SCANCODE_KP_0);
static_assert(static_cast<int>(Key::KpPeriod) == SDL_SCANCODE_KP_PERIOD);
static_assert(static_cast<int>(Key::LeftCtrl) == SDL_SCANCODE_LCTRL);
static_assert(static_cast<int>(Key::RightSuper) == SDL_SCANCODE_RGUI);
static_assert(kenshin::core::kKeyCount == SDL_SCANCODE_COUNT);

TEST_CASE("keys: names round-trip and are unique", "[input]") {
  int named = 0;
  for (int code = 0; code < static_cast<int>(kenshin::core::kKeyCount);
       ++code) {
    const Key key = static_cast<Key>(code);
    const std::string name(keyName(key));
    if (name == "unknown") continue;
    ++named;
    INFO("code " << code << " name " << name);
    CHECK(keyFromName(name) == key);
  }
  CHECK(named >= 100);  // letters, digits, F-keys, navigation, keypad, mods
  CHECK(keyName(Key::LeftShift) == "left_shift");
  CHECK(keyFromName("space") == Key::Space);
  CHECK(keyFromName("no_such_key") == Key::Unknown);
  CHECK(keyFromName("W") == Key::Unknown);  // names are exact, lower-case
}

TEST_CASE("input: pressed/down/released state machine", "[input]") {
  Input in;
  CHECK_FALSE(in.keyDown(Key::W));

  in.onKey(Key::W, true);
  CHECK(in.keyDown(Key::W));
  CHECK(in.keyPressed(Key::W));
  CHECK_FALSE(in.keyReleased(Key::W));

  in.nextFrame();  // held: still down, no longer "pressed"
  CHECK(in.keyDown(Key::W));
  CHECK_FALSE(in.keyPressed(Key::W));

  in.onKey(Key::W, true);  // a duplicate down is not a new press
  CHECK_FALSE(in.keyPressed(Key::W));

  in.onKey(Key::W, false);
  CHECK_FALSE(in.keyDown(Key::W));
  CHECK(in.keyReleased(Key::W));
  in.nextFrame();
  CHECK_FALSE(in.keyReleased(Key::W));

  in.onKey(Key::A, false);  // up without down: no spurious release
  CHECK_FALSE(in.keyReleased(Key::A));

  // Out-of-range codes are ignored, never index out of bounds.
  in.onKey(static_cast<Key>(9999), true);
  CHECK_FALSE(in.keyDown(static_cast<Key>(9999)));
}

TEST_CASE("input: press and release inside one frame both register",
          "[input]") {
  Input in;
  in.onKey(Key::Space, true);
  in.onKey(Key::Space, false);
  CHECK(in.keyPressed(Key::Space));
  CHECK(in.keyReleased(Key::Space));
  CHECK_FALSE(in.keyDown(Key::Space));
}

TEST_CASE("input: mouse buttons, motion deltas, wheel", "[input]") {
  Input in;
  in.onMouseMove(10, 20);  // first sample seeds, no delta
  CHECK(in.mouseDeltaX() == 0);
  in.onMouseMove(15, 18);
  in.onMouseMove(20, 25);
  CHECK(in.mouseX() == 20);
  CHECK(in.mouseDeltaX() == 10);
  CHECK(in.mouseDeltaY() == 5);

  in.onMouseButton(MouseButton::Right, true);
  CHECK(in.mouseDown(MouseButton::Right));
  CHECK(in.mousePressed(MouseButton::Right));
  in.onWheel(1.0f);
  in.onWheel(0.5f);
  CHECK(in.wheelDelta() == 1.5f);

  in.nextFrame();
  CHECK(in.mouseDeltaX() == 0);
  CHECK(in.wheelDelta() == 0.0f);
  CHECK_FALSE(in.mousePressed(MouseButton::Right));
  CHECK(in.mouseDown(MouseButton::Right));
  in.onMouseButton(MouseButton::Right, false);
  CHECK(in.mouseReleased(MouseButton::Right));
}

TEST_CASE("input: releaseAll lifts every held key and button", "[input]") {
  Input in;
  in.onKey(Key::W, true);
  in.onKey(Key::LeftShift, true);
  in.onMouseButton(MouseButton::Left, true);
  in.nextFrame();
  in.releaseAll();
  CHECK_FALSE(in.keyDown(Key::W));
  CHECK(in.keyReleased(Key::W));
  CHECK(in.keyReleased(Key::LeftShift));
  CHECK(in.mouseReleased(MouseButton::Left));
  CHECK_FALSE(in.keyReleased(Key::A));  // was never held
}

namespace {

void pushKey(SDL_WindowID id, SDL_Scancode sc, bool down, bool repeat) {
  SDL_Event e{};
  e.type = down ? SDL_EVENT_KEY_DOWN : SDL_EVENT_KEY_UP;
  e.key.windowID = id;
  e.key.scancode = sc;
  e.key.down = down;
  e.key.repeat = repeat;
  REQUIRE(SDL_PushEvent(&e));
}

}  // namespace

TEST_CASE("window: SDL events reach Input through a hidden window",
          "[input][window]") {
  Window window;
  WindowDesc desc;
  desc.title = "input_tests";
  desc.width = 320;
  desc.height = 200;
  desc.hidden = true;
  REQUIRE(window.create(desc));
  CHECK(window.width() > 0);
  CHECK(window.height() > 0);
  const SDL_WindowID id =
      SDL_GetWindowID(static_cast<SDL_Window*>(window.nativeHandle()));
  Input& in = Input::get();
  in.releaseAll();
  in.nextFrame();

  int keyCallbacks = 0;
  window.onKey = [&](Key, bool) { ++keyCallbacks; };

  pushKey(id, SDL_SCANCODE_W, true, false);
  pushKey(id, SDL_SCANCODE_W, true, true);  // auto-repeat: filtered
  REQUIRE(window.pumpEvents());
  CHECK(in.keyDown(Key::W));
  CHECK(in.keyPressed(Key::W));
  CHECK(keyCallbacks == 1);
  in.nextFrame();

  // Focus loss releases held keys (their key-up goes to another window).
  SDL_Event focus{};
  focus.type = SDL_EVENT_WINDOW_FOCUS_LOST;
  focus.window.windowID = id;
  REQUIRE(SDL_PushEvent(&focus));
  REQUIRE(window.pumpEvents());
  CHECK_FALSE(in.keyDown(Key::W));
  CHECK(in.keyReleased(Key::W));
  in.nextFrame();

  SDL_Event btn{};
  btn.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
  btn.button.windowID = id;
  btn.button.button = SDL_BUTTON_MIDDLE;
  btn.button.down = true;
  REQUIRE(SDL_PushEvent(&btn));
  SDL_Event wheel{};
  wheel.type = SDL_EVENT_MOUSE_WHEEL;
  wheel.wheel.windowID = id;
  wheel.wheel.y = 2.0f;
  wheel.wheel.direction = SDL_MOUSEWHEEL_NORMAL;
  REQUIRE(SDL_PushEvent(&wheel));
  REQUIRE(window.pumpEvents());
  CHECK(in.mousePressed(MouseButton::Middle));
  CHECK(in.wheelDelta() == 2.0f);
  in.releaseAll();
  in.nextFrame();

  // A close request ends the pump and fires onClose exactly once.
  int closes = 0;
  window.onClose = [&] { ++closes; };
  SDL_Event close{};
  close.type = SDL_EVENT_WINDOW_CLOSE_REQUESTED;
  close.window.windowID = id;
  REQUIRE(SDL_PushEvent(&close));
  CHECK_FALSE(window.pumpEvents());
  window.requestClose();  // idempotent
  CHECK(closes == 1);
  window.destroy();
  CHECK_FALSE(window.isOpen());
}

// ---- presenting window lifecycle (the 0.1 minimize regression) -------
// SDL updates its minimized flag only while pumping events, so a frame
// can reach vkAcquireNextImageKHR after the window collapsed: beginFrame
// must report Skipped (not Error - the demo used to quit on minimize).

#include "kenshin/backend/RenderBackend.h"

TEST_CASE("window: minimize skips frames, restore resumes", "[window][gpu]") {
  namespace backend = kenshin::backend;
  Window window;
  WindowDesc desc;
  desc.title = "minimize_test";
  desc.width = 320;
  desc.height = 240;
  REQUIRE(window.create(desc));
  auto rhi = backend::createVulkanBackend();
  backend::BackendInitParams params;
  params.window = &window;
  REQUIRE(rhi->initialize(params));

  const auto frame = [&](int& ready, int& skipped) {
    window.pumpEvents();
    backend::FrameBeginInfo fi;
    fi.width = window.width() ? window.width() : 1;
    fi.height = window.height() ? window.height() : 1;
    const backend::FrameStatus st = rhi->beginFrame(fi);
    REQUIRE(st != backend::FrameStatus::Error);
    if (st == backend::FrameStatus::Ready) {
      rhi->endFrame();
      ++ready;
    } else {
      ++skipped;
    }
  };
  int ready = 0, skipped = 0;
  for (int i = 0; i < 3; ++i) frame(ready, skipped);
  CHECK(ready == 3);

  SDL_Window* sw = static_cast<SDL_Window*>(window.nativeHandle());
  SDL_MinimizeWindow(sw);
  SDL_SyncWindow(sw);
  ready = skipped = 0;
  // Never pre-check hasNoDrawableArea here: the backend alone must cope.
  for (int i = 0; i < 20; ++i) frame(ready, skipped);
  INFO("minimized: ready " << ready << " skipped " << skipped);
  CHECK(window.hasNoDrawableArea());
  CHECK(skipped > 0);  // the deferred-recreate path really ran

  SDL_RestoreWindow(sw);
  SDL_SyncWindow(sw);
  ready = skipped = 0;
  for (int i = 0; i < 20 && ready < 3; ++i) frame(ready, skipped);
  CHECK(ready >= 3);  // presenting again after restore

  rhi->shutdown();
  window.destroy();
}
