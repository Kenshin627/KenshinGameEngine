#pragma once

// ── Frame pacing statistics (S1.5) ───────────────────────────────
// Ring buffer of frame intervals with avg/stddev/max — per the project
// convention, we track pacing variance, not just average FPS.

#include <cmath>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace core {

template <usize kCapacity = 240>
class FrameStats {
 public:
  void push(f32 intervalMs) {
    m_intervals[m_head] = intervalMs;
    m_head = (m_head + 1) % kCapacity;
    m_count = m_count < kCapacity ? m_count + 1 : kCapacity;
    m_dirty = true;
  }

  f32 averageMs() const {
    refresh();
    return m_avg;
  }
  f32 stddevMs() const {
    refresh();
    return m_std;
  }
  f32 maxMs() const {
    refresh();
    return m_max;
  }
  f32 fps() const {
    refresh();
    return m_avg > 0.0f ? 1000.0f / m_avg : 0.0f;
  }
  usize count() const { return m_count; }

  /// Raw samples (oldest first) for plotting.
  void copyTo(f32* out, usize& outCount) const {
    outCount = m_count;
    const usize start = m_count < kCapacity ? kCapacity - m_count : m_head;
    for (usize i = 0; i < m_count; ++i)
      out[i] = m_intervals[(start + i) % kCapacity];
  }

 private:
  void refresh() const {
    if (!m_dirty) return;
    m_dirty = false;
    if (m_count == 0) {
      m_avg = m_std = m_max = 0.0f;
      return;
    }
    f32 sum = 0.0f, maxV = 0.0f;
    for (usize i = 0; i < m_count; ++i) {
      const f32 v = m_intervals[i];
      sum += v;
      if (v > maxV) maxV = v;
    }
    m_avg = sum / static_cast<f32>(m_count);
    f32 var = 0.0f;
    for (usize i = 0; i < m_count; ++i) {
      const f32 d = m_intervals[i] - m_avg;
      var += d * d;
    }
    m_std = std::sqrt(var / static_cast<f32>(m_count));
    m_max = maxV;
  }

  f32 m_intervals[kCapacity] = {};
  usize m_head = 0;
  usize m_count = 0;
  mutable bool m_dirty = false;
  mutable f32 m_avg = 0.0f, m_std = 0.0f, m_max = 0.0f;
};

}  // namespace core

KENSHIN_NS_END
