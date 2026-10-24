#pragma once

// The demo's project mode (roadmap 1.3d): --project <dir> [--scene
// <path>] opens the asset database, imports what is stale, loads the
// scene document into an ecs::World and renders it through
// renderer::WorldRenderer. Between frames, syncProjectScene() runs the
// hot-reload step: settled source edits re-import, textures swap in
// place, materials rewrite their records, mesh and scene edits rebuild.

#include <string>

#include "AppArgs.h"
#include "kenshin/assets/AssetDatabase.h"
#include "kenshin/ecs/Reflection.h"
#include "kenshin/ecs/World.h"
#include "kenshin/manager/AssetManager.h"
#include "kenshin/renderer/WorldRenderer.h"

namespace kenshin {
class Engine;
}

namespace demo {

struct ProjectScene {
  kenshin::ecs::ComponentRegistry registry;
  kenshin::assets::AssetDatabase database;
  kenshin::manager::AssetManager assets;
  kenshin::ecs::World world;
  kenshin::renderer::WorldRenderer renderer;
  kenshin::core::Guid document;  // the --scene asset
  std::string documentSha1;      // of the artifact the World came from
  kenshin::u32 reloads = 0;      // applied hot reloads (logged)
};

/// False on a fatal problem (logged): no project, no scene, nothing
/// renderable.
bool openProjectScene(const Args& args, kenshin::Engine& engine, ProjectScene& out);
/// The frame-boundary hot-reload step (after endFrame).
void syncProjectScene(ProjectScene& ps, kenshin::f32 dt);
void closeProjectScene(ProjectScene& ps);

}  // namespace demo
