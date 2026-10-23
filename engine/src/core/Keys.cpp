#include "kenshin/core/Keys.h"

KENSHIN_NS_BEGIN

namespace core {

namespace {

struct KeyNameEntry {
  Key key;
  std::string_view name;
};

// One row per named key; keyName/keyFromName both scan it (the table
// is small and these are not per-pixel paths).
constexpr KeyNameEntry kKeyNames[] = {
    {Key::A, "a"},
    {Key::B, "b"},
    {Key::C, "c"},
    {Key::D, "d"},
    {Key::E, "e"},
    {Key::F, "f"},
    {Key::G, "g"},
    {Key::H, "h"},
    {Key::I, "i"},
    {Key::J, "j"},
    {Key::K, "k"},
    {Key::L, "l"},
    {Key::M, "m"},
    {Key::N, "n"},
    {Key::O, "o"},
    {Key::P, "p"},
    {Key::Q, "q"},
    {Key::R, "r"},
    {Key::S, "s"},
    {Key::T, "t"},
    {Key::U, "u"},
    {Key::V, "v"},
    {Key::W, "w"},
    {Key::X, "x"},
    {Key::Y, "y"},
    {Key::Z, "z"},
    {Key::Num1, "num1"},
    {Key::Num2, "num2"},
    {Key::Num3, "num3"},
    {Key::Num4, "num4"},
    {Key::Num5, "num5"},
    {Key::Num6, "num6"},
    {Key::Num7, "num7"},
    {Key::Num8, "num8"},
    {Key::Num9, "num9"},
    {Key::Num0, "num0"},
    {Key::Enter, "enter"},
    {Key::Escape, "escape"},
    {Key::Backspace, "backspace"},
    {Key::Tab, "tab"},
    {Key::Space, "space"},
    {Key::Minus, "minus"},
    {Key::Equals, "equals"},
    {Key::LeftBracket, "left_bracket"},
    {Key::RightBracket, "right_bracket"},
    {Key::Backslash, "backslash"},
    {Key::Semicolon, "semicolon"},
    {Key::Apostrophe, "apostrophe"},
    {Key::Grave, "grave"},
    {Key::Comma, "comma"},
    {Key::Period, "period"},
    {Key::Slash, "slash"},
    {Key::CapsLock, "caps_lock"},
    {Key::F1, "f1"},
    {Key::F2, "f2"},
    {Key::F3, "f3"},
    {Key::F4, "f4"},
    {Key::F5, "f5"},
    {Key::F6, "f6"},
    {Key::F7, "f7"},
    {Key::F8, "f8"},
    {Key::F9, "f9"},
    {Key::F10, "f10"},
    {Key::F11, "f11"},
    {Key::F12, "f12"},
    {Key::PrintScreen, "print_screen"},
    {Key::ScrollLock, "scroll_lock"},
    {Key::Pause, "pause"},
    {Key::Insert, "insert"},
    {Key::Home, "home"},
    {Key::PageUp, "page_up"},
    {Key::Delete, "delete"},
    {Key::End, "end"},
    {Key::PageDown, "page_down"},
    {Key::Right, "right"},
    {Key::Left, "left"},
    {Key::Down, "down"},
    {Key::Up, "up"},
    {Key::NumLock, "num_lock"},
    {Key::KpDivide, "kp_divide"},
    {Key::KpMultiply, "kp_multiply"},
    {Key::KpMinus, "kp_minus"},
    {Key::KpPlus, "kp_plus"},
    {Key::KpEnter, "kp_enter"},
    {Key::Kp1, "kp1"},
    {Key::Kp2, "kp2"},
    {Key::Kp3, "kp3"},
    {Key::Kp4, "kp4"},
    {Key::Kp5, "kp5"},
    {Key::Kp6, "kp6"},
    {Key::Kp7, "kp7"},
    {Key::Kp8, "kp8"},
    {Key::Kp9, "kp9"},
    {Key::Kp0, "kp0"},
    {Key::KpPeriod, "kp_period"},
    {Key::LeftCtrl, "left_ctrl"},
    {Key::LeftShift, "left_shift"},
    {Key::LeftAlt, "left_alt"},
    {Key::LeftSuper, "left_super"},
    {Key::RightCtrl, "right_ctrl"},
    {Key::RightShift, "right_shift"},
    {Key::RightAlt, "right_alt"},
    {Key::RightSuper, "right_super"},
};

}  // namespace

std::string_view keyName(Key key) {
  for (const auto& e : kKeyNames)
    if (e.key == key) return e.name;
  return "unknown";
}

Key keyFromName(std::string_view name) {
  for (const auto& e : kKeyNames)
    if (e.name == name) return e.key;
  return Key::Unknown;
}

}  // namespace core

KENSHIN_NS_END
