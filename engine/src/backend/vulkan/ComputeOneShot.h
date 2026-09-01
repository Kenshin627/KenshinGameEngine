#pragma once

// ── ComputeOneShot — self-contained compute job (S1.4) ───────────
// Uploads buffers, dispatches once, waits, copies marked buffers back.
// Headless (windowless device is fine) - this is the V1 differential-test
// channel for M2's compute shaders. Per-frame in-frame compute arrives
// with M2 in CommandContext.

#include <volk.h>

#include <vma/vk_mem_alloc.h>

#include <vector>

#include "kenshin/Common.h"
#include "kenshin/backend/RenderBackend.h"

KENSHIN_NS_BEGIN

namespace backend {

struct VulkanDevice;

/// Runs one Job on the graphics queue. `pool` must be a command pool whose
/// buffers are not concurrently recorded (frame slot 0's is the norm).
/// For each buffer with readBack, outputs[i] receives its bytes.
bool runComputeOneshotJob(const VulkanDevice& device,
                          VkCommandPool commandPool,
                          const IRenderBackend::Job& job,
                          std::vector<u8> outputs[IRenderBackend::Job::kMaxBuffers]);

}  // namespace backend

KENSHIN_NS_END
