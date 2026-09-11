#pragma once

// ── StreamingManager — M6.3: page residency driver ────────────────
// nanite.md D14 (self-designed): the touched-page bitmap the chain's
// page_touch pass produces is the feedback signal - touched pages
// refresh their LRU stamp, touched-but-absent pages become IO requests.
// Pages stream asset-file -> (IO thread, host bytes) -> transfer-queue
// slot upload (the M6.2 batch API) -> pageTable flip AFTER the upload
// timeline wait, so a frame only ever sees a COMPLETE page.
//
// Budget: resident page bytes <= budgetBytes. Eviction is LRU over
// non-pinned pages; each asset's page 0 is pinned (the coarsest pages
// ride first in every .kgeo v3 - D14.3's always-resident safety net,
// guaranteeing a coarse image even at 0 budget).
//
// Threading: the FIRST std::thread in the engine. The IO thread owns
// nothing but its KGeoPageReader handles, host byte buffers and two
// mutex-guarded queues; every Vulkan call stays on the main thread.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "kenshin/Common.h"
#include "kenshin/geometry/KGeoFile.h"
#include "kenshin/renderer/VisBufferChain.h"

KENSHIN_NS_BEGIN

namespace backend {
class IRenderBackend;
}
namespace renderer {
class VirtualGeometryScene;
}

KENSHIN_NS_END

KENSHIN_NS_BEGIN

namespace renderer {

class StreamingManager {
 public:
  /// Wire up against an ALREADY-FLUSHED paged scene. budgetBytes = 0
  /// means unlimited (everything streams in and stays).
  bool init(backend::IRenderBackend& backend, VirtualGeometryScene& scene,
            usize budgetBytes);
  void shutdown();

  /// Register the asset FILE backing mesh `meshId` (its page range in
  /// the global page space comes from the scene's mesh table). Call
  /// after init, before the first onFrame.
  bool addAsset(u32 meshId, const std::string& kgeoPath);

  /// Per-frame pump (main thread, between endFrame and the next
  /// execute - readTouchedPages' contract). Reads the touched bitmap,
  /// drives LRU + requests, drains finished IO into upload batches.
  void onFrame(VisBufferChain& chain);

  /// Prefetch `page` at the next onFrame (root-page warmup etc).
  void request(u32 page) {
    if (page < m_state.size() && m_state[page].pin) return;
    if (page < m_state.size()) m_state[page].wanted = true;
  }

  usize residentBytes() const { return usize(m_residentPages) * m_slotBytes; }
  u32 residentPages() const { return m_residentPages; }
  u32 pendingRequests() const;
  /// Pages touched this frame whose data was absent (the miss counter).
  u32 missedPages() const { return m_missedPages; }

 private:
  struct PageState {
    u32 slot = 0xFFFFFFFFu;      // valid when resident
    u64 lastTouch = 0;           // LRU stamp (frame counter)
    bool resident = false;
    bool wanted = false;         // request latched (touch or prefetch)
    bool inflight = false;       // IO thread owns it right now
    bool pin = false;            // never evict (page 0 of an asset)
  };

  /// Upload-batch page cap = the backend staging arena's (64 x 128KB).
  static constexpr usize kPagesPerBatch = 64;

  void ensureIoThread();
  void ioLoop();
  bool evictOne();
  void uploadReady();

  backend::IRenderBackend* m_backend = nullptr;
  VirtualGeometryScene* m_scene = nullptr;
  usize m_budgetBytes = 0;  // 0 = unlimited
  u32 m_pageCount = 0;
  u32 m_slotCount = 0;  // the pool's slot count (budget-capped)
  u32 m_slotBytes = 0;
  u32 m_residentPages = 0;
  u32 m_missedPages = 0;
  u64 m_frameCounter = 0;
  std::vector<PageState> m_state;
  std::vector<u32> m_freeSlots;  // LIFO
  u32* m_pageTable = nullptr;    // the scene's mapped table

  // Per-asset page range in the global page space + the source file.
  struct Asset {
    u32 pageBegin = 0;
    u32 pageEnd = 0;
    std::string path;
  };
  std::vector<Asset> m_assets;  // stable once the IO thread starts

  // ---- IO thread plumbing ----------------------------------------------
  // The queues hold GLOBAL page ids; the byte payloads live in m_ready
  // and are owned by whichever side holds the entry. The thread starts
  // at the FIRST onFrame (after every addAsset - no asset-list lock).
  std::thread m_ioThread;
  std::atomic<bool> m_ioRunning{false};
  mutable std::mutex m_ioMutex;
  std::condition_variable m_ioCv;
  std::vector<u32> m_ioQueue;  // pending reads (page ids)
  struct Ready {
    u32 page = 0;
    std::vector<u8> bytes;
  };
  std::vector<Ready> m_ready;  // finished reads awaiting upload
};

}  // namespace renderer

KENSHIN_NS_END
