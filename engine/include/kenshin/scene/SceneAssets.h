#pragma once

// ── Scene documents from the asset database (roadmap 1.3d) ──────
// The World an asset Guid names:
//   scene  -> its entities (the cooked .kscene)
//   prefab -> one instance (fresh entity Guids) as the only root
//   model  -> one instance of the model's prefab sub-asset
// On failure `world` is untouched.

#include <string>

#include "kenshin/assets/AssetDatabase.h"
#include "kenshin/core/Guid.h"
#include "kenshin/ecs/Reflection.h"
#include "kenshin/ecs/World.h"

KENSHIN_NS_BEGIN
namespace scene {

bool loadWorldAsset(const assets::AssetDatabase& database, const core::Guid& asset,
                    const ecs::ComponentRegistry& registry, ecs::World& world,
                    std::string* error);

/// The artifact Guid loadWorldAsset reads for `asset` (the model's
/// prefab sub-asset for a model); null if none.
core::Guid worldDocumentOf(const assets::AssetDatabase& database, const core::Guid& asset);

}  // namespace scene
KENSHIN_NS_END
