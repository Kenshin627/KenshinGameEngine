// M6.2: page streaming upload surface - slot pool + transfer-queue
// batches (timeline-paced) vs the graphics-queue fallback. Byte-exact
// readback is the acceptance (same discipline as the chain tests).

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cstring>
#include <vector>

#include "kenshin/backend/RenderBackend.h"

namespace {

using namespace kenshin;

/// Distinct per-slot byte pattern: slot id repeated as u32.
std::vector<u8> slotPattern(u32 slot, usize bytes, u32 salt) {
  std::vector<u8> data(bytes);
  for (usize i = 0; i + 4 <= bytes; i += 4) {
    const u32 v = (slot * 131071u + salt) ^ static_cast<u32>(i);
    std::memcpy(data.data() + i, &v, 4);
  }
  return data;
}

/// Upload `slots` into the pool via batches of `perBatch` copies each;
/// returns the LAST signaled value (every batch waited). Exercises the
/// cross-batch timeline ordering (batch N+1's begin waits batch N).
u64 uploadAll(backend::IRenderBackend& be, u32 pool, const u8* const* datas,
              const u32* slots, u32 count, u32 slotBytes, u32 perBatch) {
  u64 last = 0;
  for (u32 base = 0; base < count; base += perBatch) {
    const u32 n = std::min(perBatch, count - base);
    REQUIRE(be.beginPageUploads(pool));
    for (u32 i = 0; i < n; ++i)
      REQUIRE(be.copyPageToSlot(datas[base + i], slots[base + i], slotBytes));
    const u64 value = be.submitPageUploads();
    REQUIRE(value > last);  // monotone timeline
    last = value;
  }
  REQUIRE(be.waitPageUploads(last));
  return last;
}

}  // namespace

TEST_CASE("page slot pool uploads byte-exact (transfer queue)",
          "[backend][pages]") {
  auto be = backend::createVulkanBackend();
  backend::BackendInitParams params;
  REQUIRE(be->initialize(params));
  INFO("transfer queue in use: " << be->pageUploadsOnTransferQueue());

  constexpr u32 kSlots = 12, kSlotBytes = 16384;
  const u32 pool = be->createPageSlotPool(kSlots, kSlotBytes);
  REQUIRE(pool != backend::IRenderBackend::kInvalidBufferHandle);
  REQUIRE(be->bufferSize(pool) == usize(kSlots) * kSlotBytes);

  std::vector<std::vector<u8>> data;
  std::vector<const u8*> ptrs;
  std::vector<u32> slots;
  for (u32 s = 0; s < kSlots; ++s) {
    data.push_back(slotPattern(s, kSlotBytes, 7u));
    ptrs.push_back(data.back().data());
    slots.push_back(s);
  }
  // 12 pages in batches of 5: exercises arena rollover + timeline
  // ordering across three submissions.
  uploadAll(*be, pool, ptrs.data(), slots.data(), kSlots, kSlotBytes, 5);

  std::vector<u8> readback;
  REQUIRE(be->readbackBuffer(pool, 0, usize(kSlots) * kSlotBytes, readback));
  for (u32 s = 0; s < kSlots; ++s)
    REQUIRE(std::memcmp(readback.data() + usize(s) * kSlotBytes,
                        data[s].data(), kSlotBytes) == 0);
}

TEST_CASE("page uploads graphics-queue fallback parity", "[backend][pages]") {
  auto be = backend::createVulkanBackend();
  backend::BackendInitParams params;
  REQUIRE(be->initialize(params));
  be->setTransferQueueEnabled(false);
  REQUIRE_FALSE(be->pageUploadsOnTransferQueue());

  constexpr u32 kSlots = 6, kSlotBytes = 8192;
  const u32 pool = be->createPageSlotPool(kSlots, kSlotBytes);
  REQUIRE(pool != backend::IRenderBackend::kInvalidBufferHandle);

  std::vector<std::vector<u8>> data;
  std::vector<const u8*> ptrs;
  std::vector<u32> slots;
  for (u32 s = 0; s < kSlots; ++s) {
    data.push_back(slotPattern(s, kSlotBytes, 99u));
    ptrs.push_back(data.back().data());
    slots.push_back(s);
  }
  uploadAll(*be, pool, ptrs.data(), slots.data(), kSlots, kSlotBytes, 3);

  std::vector<u8> readback;
  REQUIRE(be->readbackBuffer(pool, 0, usize(kSlots) * kSlotBytes, readback));
  for (u32 s = 0; s < kSlots; ++s)
    REQUIRE(std::memcmp(readback.data() + usize(s) * kSlotBytes,
                        data[s].data(), kSlotBytes) == 0);

  // Slot REUSE on the same pool: overwrite slot 2 with new content,
  // verify only that slot changed.
  const auto replacement = slotPattern(2, kSlotBytes, 1234u);
  REQUIRE(be->beginPageUploads(pool));
  REQUIRE(be->copyPageToSlot(replacement.data(), 2, kSlotBytes));
  const u64 value = be->submitPageUploads();
  REQUIRE(value > 0);
  REQUIRE(be->waitPageUploads(value));
  REQUIRE(be->readbackBuffer(pool, 0, usize(kSlots) * kSlotBytes, readback));
  for (u32 s = 0; s < kSlots; ++s) {
    const auto& expect = s == 2 ? replacement : data[s];
    REQUIRE(std::memcmp(readback.data() + usize(s) * kSlotBytes,
                        expect.data(), kSlotBytes) == 0);
  }
}

TEST_CASE("page upload partial slots and empty batches", "[backend][pages]") {
  auto be = backend::createVulkanBackend();
  backend::BackendInitParams params;
  REQUIRE(be->initialize(params));

  constexpr u32 kSlots = 4, kSlotBytes = 4096;
  const u32 pool = be->createPageSlotPool(kSlots, kSlotBytes);
  REQUIRE(pool != backend::IRenderBackend::kInvalidBufferHandle);

  // Fill slots 1 and 3 fully (the baseline for the partial overwrite).
  const auto full = slotPattern(1, kSlotBytes, 3u);
  REQUIRE(be->beginPageUploads(pool));
  REQUIRE(be->copyPageToSlot(full.data(), 1, kSlotBytes));
  const auto full3 = slotPattern(3, kSlotBytes, 4u);
  REQUIRE(be->copyPageToSlot(full3.data(), 3, kSlotBytes));
  REQUIRE(be->waitPageUploads(be->submitPageUploads()));

  // Partial copy: 100 bytes land at slot 3's start; the REST of slot 3
  // keeps its previous contents and slot 1 stays untouched.
  const auto partial = slotPattern(3, 100, 55u);
  REQUIRE(be->beginPageUploads(pool));
  REQUIRE(be->copyPageToSlot(partial.data(), 3, 100));
  const u64 value = be->submitPageUploads();
  REQUIRE(value > 0);
  REQUIRE(be->waitPageUploads(value));

  std::vector<u8> readback;
  REQUIRE(be->readbackBuffer(pool, 0, usize(kSlots) * kSlotBytes, readback));
  REQUIRE(std::memcmp(readback.data() + usize(3) * kSlotBytes, partial.data(),
                      100) == 0);
  REQUIRE(std::memcmp(readback.data() + usize(3) * kSlotBytes + 100,
                      full3.data() + 100, kSlotBytes - 100) == 0);
  REQUIRE(std::memcmp(readback.data() + kSlotBytes, full.data(),
                      kSlotBytes) == 0);

  // Empty batch: closes cleanly, signals nothing (0), next batch works.
  REQUIRE(be->beginPageUploads(pool));
  REQUIRE(be->submitPageUploads() == 0);
  REQUIRE(be->beginPageUploads(pool));
  REQUIRE(be->copyPageToSlot(full.data(), 0, kSlotBytes));
  REQUIRE(be->waitPageUploads(be->submitPageUploads()));

  // Invalid copies rejected without corrupting the batch.
  REQUIRE(be->beginPageUploads(pool));
  REQUIRE_FALSE(be->copyPageToSlot(full.data(), 0, kSlotBytes + 1));  // > slot
  REQUIRE_FALSE(be->copyPageToSlot(full.data(), kSlots, 4));          // OOR slot
  REQUIRE(be->copyPageToSlot(full.data(), 0, kSlotBytes));
  REQUIRE(be->waitPageUploads(be->submitPageUploads()));
  REQUIRE_FALSE(be->beginPageUploads(
      backend::IRenderBackend::kInvalidBufferHandle));

  // A second beginPageUploads while one is open must fail (batch state
  // is single).
  REQUIRE(be->beginPageUploads(pool));
  REQUIRE_FALSE(be->beginPageUploads(pool));
  be->submitPageUploads();
}

TEST_CASE("page uploads latch queue and pool geometry per batch",
          "[backend][pages]") {
  auto be = backend::createVulkanBackend();
  backend::BackendInitParams params;
  REQUIRE(be->initialize(params));

  // Two pools with DIFFERENT slot geometry; a non-pool buffer must be
  // rejected as a batch target.
  constexpr u32 kSlots = 4;
  const u32 poolA = be->createPageSlotPool(kSlots, 8192);
  const u32 poolB = be->createPageSlotPool(kSlots, 4096);
  REQUIRE(poolA != backend::IRenderBackend::kInvalidBufferHandle);
  REQUIRE(poolB != backend::IRenderBackend::kInvalidBufferHandle);
  const u32 plain = be->createBuffer({.size = 4096});
  REQUIRE(plain != backend::IRenderBackend::kInvalidBufferHandle);
  REQUIRE_FALSE(be->beginPageUploads(plain));

  // Pool B batch: 4096B slots; bytes > 4096 must reject even though
  // pool A (created earlier, larger) once set the global geometry.
  const auto dataA = slotPattern(0, 8192, 1u);
  const auto dataB = slotPattern(0, 4096, 2u);
  REQUIRE(be->beginPageUploads(poolB));
  REQUIRE_FALSE(be->copyPageToSlot(dataA.data(), 0, 8192));  // B's slots are 4K
  REQUIRE(be->copyPageToSlot(dataB.data(), 1, 4096));
  // Mid-batch queue flip must NOT reroute the recording: the batch
  // continues on its begun command buffer and submits cleanly.
  be->setTransferQueueEnabled(false);
  REQUIRE(be->copyPageToSlot(dataB.data(), 2, 4096));
  REQUIRE(be->waitPageUploads(be->submitPageUploads()));
  be->setTransferQueueEnabled(true);

  // Pool A batch afterwards: 8192B stride lands at A's offsets.
  REQUIRE(be->beginPageUploads(poolA));
  REQUIRE(be->copyPageToSlot(dataA.data(), 1, 8192));
  REQUIRE(be->waitPageUploads(be->submitPageUploads()));

  std::vector<u8> rb;
  REQUIRE(be->readbackBuffer(poolB, 0, kSlots * 4096, rb));
  REQUIRE(std::memcmp(rb.data() + 4096, dataB.data(), 4096) == 0);
  REQUIRE(std::memcmp(rb.data() + 2 * 4096, dataB.data(), 4096) == 0);
  REQUIRE(be->readbackBuffer(poolA, 8192, 8192, rb));
  REQUIRE(std::memcmp(rb.data(), dataA.data(), 8192) == 0);
}
