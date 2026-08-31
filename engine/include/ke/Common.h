#pragma once

// ── KEngine 公共基础 ────────────────────────────────────────────
// 命名空间宏与基础标量类型别名（参照 renderEngine::Common.h 的角色）。

#include <cstddef>
#include <cstdint>

#define KE_NS_BEGIN namespace ke {
#define KE_NS_END   }
#define KE_NS_USING using namespace ke;

KE_NS_BEGIN

using i8  = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using f32 = float;
using f64 = double;

using usize = size_t;

KE_NS_END
