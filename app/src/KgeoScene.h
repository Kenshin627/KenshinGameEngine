#pragma once

// The demo's virtual-geometry scene from the command line: the primary
// .kgeo, an optional --ground plane, the --mesh gallery, their material
// textures, the instance layout (--wall / --grid / --multi / default),
// imposter atlases and the page streamer. The result feeds
// renderer::SceneRenderer.

#include <memory>

#include "AppArgs.h"
#include "KgeoPreview.h"
#include "kenshin/renderer/SceneRenderer.h"
#include "kenshin/renderer/StreamingManager.h"

namespace kenshin {
class Engine;
namespace manager {
class TextureManager;
}
}  // namespace kenshin

namespace demo {

struct KgeoScene {
  KgeoPreview preview;  // the primary asset's dag (LOD panel, framing)
  kenshin::renderer::SceneRendererDesc desc;
  std::unique_ptr<kenshin::renderer::StreamingManager> streamer;
};

enum class KgeoSceneResult { Ok, LoadFailed, Fatal };

/// Load and register everything. LoadFailed: the primary .kgeo did not
/// load (the app continues without the virtual path); Fatal: a rejected
/// or broken asset (the app exits non-zero).
KgeoSceneResult buildKgeoScene(const Args& args, kenshin::Engine& engine,
                               kenshin::manager::TextureManager& textures,
                               KgeoScene& out);

}  // namespace demo
