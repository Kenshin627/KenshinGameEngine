#pragma once

// ── Keys — portable physical key codes ──────────────────────────────
// Values are USB HID keyboard usage IDs (HID Usage Tables, page 0x07),
// which is exactly what SDL3's SDL_Scancode uses - the platform layer
// casts, no translation table. Physical keys, not characters: Key::W is
// the key in the W position on every layout (WASD stays WASD on AZERTY).
// Text input is a separate stream (the editor's ImGui handles it).
//
// keyName / keyFromName give each key one stable lower-case name ("w",
// "space", "left_shift", "f5") for scripts, input-action files and the
// agent API.

#include <string_view>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace core {

enum class Key : u16 {
  Unknown = 0,
  A = 4, B, C, D, E, F, G, H, I, J, K, L, M,
  N, O, P, Q, R, S, T, U, V, W, X, Y, Z,  // Z = 29
  Num1 = 30, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9, Num0,  // 39
  Enter = 40,
  Escape = 41,
  Backspace = 42,
  Tab = 43,
  Space = 44,
  Minus = 45,
  Equals = 46,
  LeftBracket = 47,
  RightBracket = 48,
  Backslash = 49,
  Semicolon = 51,
  Apostrophe = 52,
  Grave = 53,
  Comma = 54,
  Period = 55,
  Slash = 56,
  CapsLock = 57,
  F1 = 58, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,  // F12 = 69
  PrintScreen = 70,
  ScrollLock = 71,
  Pause = 72,
  Insert = 73,
  Home = 74,
  PageUp = 75,
  Delete = 76,
  End = 77,
  PageDown = 78,
  Right = 79,
  Left = 80,
  Down = 81,
  Up = 82,
  NumLock = 83,
  KpDivide = 84,
  KpMultiply = 85,
  KpMinus = 86,
  KpPlus = 87,
  KpEnter = 88,
  Kp1 = 89, Kp2, Kp3, Kp4, Kp5, Kp6, Kp7, Kp8, Kp9, Kp0,  // Kp0 = 98
  KpPeriod = 99,
  LeftCtrl = 224,
  LeftShift = 225,
  LeftAlt = 226,
  LeftSuper = 227,
  RightCtrl = 228,
  RightShift = 229,
  RightAlt = 230,
  RightSuper = 231,
};

/// Size of key-state tables (SDL_SCANCODE_COUNT).
constexpr u32 kKeyCount = 512;

enum class MouseButton : u8 { Left = 0, Middle = 1, Right = 2, X1 = 3, X2 = 4 };
constexpr u32 kMouseButtonCount = 5;

/// Stable lower-case name ("a", "num1", "space", "left_shift"); "unknown"
/// for codes without a name.
std::string_view keyName(Key key);
/// Inverse of keyName (exact match); Key::Unknown when not found.
Key keyFromName(std::string_view name);

}  // namespace core

KENSHIN_NS_END
