#pragma once

// ── BaselineForwardPipeline ──────────────────────────────────────
// The S2.2 forward path: one Blinn-Phong pass over the draw stream.
// Stateless like every pipeline strategy — no per-frame members beyond
// its backend pipeline handle, so one instance serves every frame.

#include "kenshin/renderer/Renderer.h"

KENSHIN_NS_BEGIN

namespace renderer {

class BaselineForwardPipeline final : public IRenderPipeline {
 public:
  bool init(backend::IRenderBackend* backend,
            manager::ShaderManager& shaders) override;
  void shutdown(backend::IRenderBackend* backend) override;
  void execute(const FrameContext& ctx) override;

 private:
  /// The backend pipeline is created lazily on first execute(): its
  /// color-attachment format comes from the swapchain, which only
  /// exists after the first beginFrame.
  void ensurePipeline(backend::IRenderBackend* backend);

  std::vector<u32> m_vsSpirv, m_fsSpirv;
  u32 m_pipeline = 0xFFFFFFFFu;  // backend pipeline handle (kInvalid)
};

}  // namespace renderer

KENSHIN_NS_END
