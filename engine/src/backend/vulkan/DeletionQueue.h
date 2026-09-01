#pragma once

// ── DeletionQueue — per-frame-slot deferred destruction ──────────
// vkguide pattern: destroy requests land on the CURRENT frame slot's
// queue; a slot is only reused after its fence confirms that frame
// completed, so flushing at reuse destroys objects that are provably
// no longer in flight. Fixes the destroy-while-in-use hazard (e.g.
// kgeo rebuilding its mesh while the previous frame still draws it).

#include <functional>
#include <utility>
#include <vector>

#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

namespace backend {

class DeletionQueue {
 public:
  void push(std::function<void()>&& destroy) {
    m_entries.push_back(std::move(destroy));
  }
  bool empty() const { return m_entries.empty(); }
  usize size() const { return m_entries.size(); }

  /// Run and drop all pending destroy callbacks. Call when the owning
  /// slot's fence has confirmed completion (frame reuse), or at shutdown
  /// after a device idle.
  void flush() {
    for (auto& destroy : m_entries) destroy();
    m_entries.clear();
  }

 private:
  std::vector<std::function<void()>> m_entries;
};

}  // namespace backend

KENSHIN_NS_END
