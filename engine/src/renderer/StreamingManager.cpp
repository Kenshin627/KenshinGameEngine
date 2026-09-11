#include "kenshin/renderer/StreamingManager.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include "kenshin/backend/RenderBackend.h"
#include "kenshin/core/Log.h"
#include "kenshin/geometry/PagePacker.h"
#include "kenshin/renderer/VirtualGeometryScene.h"

KENSHIN_NS_BEGIN

namespace renderer {

bool StreamingManager::init(backend::IRenderBackend& backend,
                            VirtualGeometryScene& scene, usize budgetBytes) {
  if (!scene.paged()) {
    KENSHIN_ERROR("[stream] scene is not paged");
    return false;
  }
  m_backend = &backend;
  m_scene = &scene;
  m_budgetBytes = budgetBytes;
  m_pageCount = scene.pageCount();
  m_slotBytes = geometry::kPageSlotSize;
  m_slotCount = scene.pageSlotCount();
  m_pageTable = scene.pageTableMapped();
  if (m_pageTable == nullptr || m_pageCount == 0 || m_slotCount == 0) {
    KENSHIN_ERROR("[stream] scene page tables not ready (flushPools first)");
    return false;
  }
  m_state.assign(m_pageCount, PageState{});
  m_freeSlots.reserve(m_slotCount);
  for (u32 s = m_slotCount; s-- > 0;) m_freeSlots.push_back(s);

  KENSHIN_INFO("[stream] ready: {} pages / {} slots ({} MB pool), budget {}",
               m_pageCount, m_slotCount,
               double(usize(m_slotCount) * m_slotBytes) / (1024.0 * 1024.0),
               m_budgetBytes == 0
                   ? std::string("unlimited")
                   : std::to_string(m_budgetBytes / (1024 * 1024)) + " MB");
  return true;
}

bool StreamingManager::addAsset(u32 meshId, const std::string& kgeoPath) {
  if (m_backend == nullptr || meshId >= m_scene->meshes().size())
    return false;
  const VgMeshRange& mesh = m_scene->meshes()[meshId];
  Asset asset;
  asset.pageBegin = mesh.pageBegin;
  asset.pageEnd = mesh.pageEnd;
  asset.path = kgeoPath;
  m_assets.push_back(asset);
  // Page 0 of every asset is the coarsest (the packer's coarse-first
  // order): pin it - D14.3's always-resident safety net.
  if (mesh.pageBegin < m_state.size()) {
    m_state[mesh.pageBegin].pin = true;
    m_state[mesh.pageBegin].wanted = true;
  }
  return true;
}

u32 StreamingManager::pendingRequests() const {
  std::lock_guard<std::mutex> lock(m_ioMutex);
  return static_cast<u32>(m_ioQueue.size());
}

void StreamingManager::shutdown() {
  if (m_ioRunning.exchange(false)) {
    m_ioCv.notify_all();
    if (m_ioThread.joinable()) m_ioThread.join();
  }
  std::lock_guard<std::mutex> lock(m_ioMutex);
  m_ready.clear();
  m_ioQueue.clear();
  m_backend = nullptr;
  m_scene = nullptr;
  m_pageTable = nullptr;
}

// ---- IO thread ------------------------------------------------------------
// Owns NOTHING but the per-asset readers, its host buffers and the
// queues (queue access under m_ioMutex). The thread starts at the
// FIRST onFrame - after every addAsset, so the asset list is stable
// and needs no lock. KGeoPageReader v1 reopens the FILE* per page; the
// thread-local reader table just avoids re-parsing the page directory
// every read.

void StreamingManager::ioLoop() {
  std::vector<geometry::KGeoPageReader> readers(m_assets.size());
  std::vector<bool> opened(m_assets.size(), false);
  while (m_ioRunning.load()) {
    u32 page = 0xFFFFFFFFu;
    {
      std::unique_lock<std::mutex> lock(m_ioMutex);
      m_ioCv.wait(lock,
                  [this] { return !m_ioQueue.empty() || !m_ioRunning.load(); });
      if (!m_ioRunning.load() && m_ioQueue.empty()) return;
      page = m_ioQueue.back();
      m_ioQueue.pop_back();
    }
    usize assetIdx = m_assets.size();
    for (usize a = 0; a < m_assets.size(); ++a)
      if (page >= m_assets[a].pageBegin && page < m_assets[a].pageEnd) {
        assetIdx = a;
        break;
      }
    if (assetIdx == m_assets.size()) continue;

    Ready ready;
    ready.page = page;
    std::string log;
    if (!opened[assetIdx] &&
        !readers[assetIdx].open(m_assets[assetIdx].path, log)) {
      KENSHIN_ERROR("[stream] asset open failed ({}): {}",
                    m_assets[assetIdx].path, log);
      continue;
    }
    opened[assetIdx] = true;
    if (!readers[assetIdx].readPage(page - m_assets[assetIdx].pageBegin,
                                    ready.bytes, log)) {
      KENSHIN_ERROR("[stream] page {} read failed: {}", page, log);
      continue;
    }
    {
      std::lock_guard<std::mutex> lock(m_ioMutex);
      m_ready.push_back(std::move(ready));
    }
  }
}

void StreamingManager::ensureIoThread() {
  if (m_ioThread.joinable()) return;
  m_ioRunning = true;
  m_ioThread = std::thread(&StreamingManager::ioLoop, this);
}

void StreamingManager::onFrame(VisBufferChain& chain) {
  ++m_frameCounter;
  ensureIoThread();

  // 1) Feedback: the touched bitmap drives LRU stamps + misses.
  std::vector<u32> touched;
  if (chain.readTouchedPages(*m_backend, touched)) {
    m_missedPages = 0;
    for (u32 w = 0; w < touched.size(); ++w) {
      u32 bits = touched[w];
      for (u32 b = 0; bits != 0 && b < 32; ++b) {
        if ((bits & (1u << b)) == 0) continue;
        bits &= ~(1u << b);
        const u32 page = w * 32u + b;
        if (page >= m_pageCount) continue;
        PageState& st = m_state[page];
        if (st.resident) {
          st.lastTouch = m_frameCounter;
        } else {
          if (!st.wanted) ++m_missedPages;
          st.wanted = true;
        }
      }
    }
  }

  // 2) Drain finished IO into transfer-queue upload batches (throttled
  // to the M6.2 arena: 64 pages of 128KB per batch), page-id order so
  // coarse pages ride first under pressure.
  uploadReady();

  // 3) Push new requests to the IO thread (deduped by `wanted`).
  std::vector<u32> fresh;
  for (u32 p = 0; p < m_pageCount; ++p) {
    const PageState& st = m_state[p];
    if (st.wanted && !st.resident && !st.inflight) fresh.push_back(p);
  }
  if (!fresh.empty()) {
    std::lock_guard<std::mutex> lock(m_ioMutex);
    for (const u32 p : fresh) m_state[p].inflight = true;
    m_ioQueue.insert(m_ioQueue.end(), fresh.begin(), fresh.end());
    m_ioCv.notify_one();
  }
}

void StreamingManager::uploadReady() {
  std::vector<Ready> batch;
  {
    std::lock_guard<std::mutex> lock(m_ioMutex);
    if (m_ready.empty()) return;
    std::sort(m_ready.begin(), m_ready.end(),
              [](const Ready& a, const Ready& b) { return a.page < b.page; });
    const usize take = std::min<usize>(m_ready.size(), kPagesPerBatch);
    batch.assign(std::make_move_iterator(m_ready.begin()),
                 std::make_move_iterator(m_ready.begin() + take));
    m_ready.erase(m_ready.begin(), m_ready.begin() + take);
  }

  // Budget: evict LRU non-pinned pages until the post-batch resident
  // total fits. The flip-to-absent happens BEFORE the upload (frames
  // in flight either read the old entry - the data stays intact until
  // the upload's built-in frame-timeline wait drains them - or the new
  // one and skip).
  for (const auto& r : batch) {
    while (m_budgetBytes != 0 && !m_state[r.page].resident &&
           (usize(m_residentPages) + 1) * m_slotBytes > m_budgetBytes) {
      if (!evictOne()) break;
    }
  }

  u32 uploaded = 0;
  if (m_backend->beginPageUploads(m_scene->pageSlotPool())) {
    for (auto& r : batch) {
      PageState& st = m_state[r.page];
      if (st.resident) continue;  // raced a prefetch: drop the stale copy
      if (m_freeSlots.empty()) break;  // pool full: next frame retries
      const u32 slot = m_freeSlots.back();
      if (!m_backend->copyPageToSlot(r.bytes.data(), slot,
                                     static_cast<u32>(r.bytes.size()))) {
        break;  // arena full: the rest waits for the next onFrame
      }
      m_freeSlots.pop_back();
      st.slot = slot;
      st.lastTouch = m_frameCounter;
      ++uploaded;
    }
    const u64 value = m_backend->submitPageUploads();
    // The pageTable flips ONLY after the batch completed - a frame
    // never sees a half-written page. (v1 host-waits here; a paced
    // deferred flip is the tuning knob if the wait shows up in frame
    // time.)
    if (value != 0) m_backend->waitPageUploads(value);
  }
  if (uploaded > 0) {
    for (auto& r : batch) {
      PageState& st = m_state[r.page];
      if (st.slot != 0xFFFFFFFFu && !st.resident) {
        st.resident = true;
        st.wanted = false;
        st.inflight = false;
        m_pageTable[r.page] = st.slot;
        ++m_residentPages;
      }
    }
  }
  for (auto& r : batch) {
    PageState& st = m_state[r.page];
    if (!st.resident) st.inflight = false;  // retry next frame
  }
}

bool StreamingManager::evictOne() {
  u64 best = ~0ull;
  u32 victim = 0xFFFFFFFFu;
  for (u32 p = 0; p < m_pageCount; ++p) {
    const PageState& st = m_state[p];
    if (st.resident && !st.pin && st.lastTouch < best) {
      best = st.lastTouch;
      victim = p;
    }
  }
  if (victim == 0xFFFFFFFFu) return false;
  PageState& st = m_state[victim];
  m_pageTable[victim] = 0xFFFFFFFFu;  // absent FIRST (see uploadReady)
  m_freeSlots.push_back(st.slot);
  st.resident = false;
  st.slot = 0xFFFFFFFFu;
  --m_residentPages;
  return true;
}

}  // namespace renderer

KENSHIN_NS_END
