#pragma once

// ── Demo scene ───────────────────────────────────────────────────
// Shape table + layout (--mesh all spreads one instance of every shape
// on the X axis; a single name picks one; anything else is an OBJ path)
// plus mesh loading and GPU upload.

#include <glm/glm.hpp>
#include <string>
#include <vector>

#include "kenshin/Common.h"
#include "kenshin/assets/MeshData.h"
#include "kenshin/manager/MeshPool.h"

namespace demo {

struct SceneShape {
  const char* name;
  kenshin::assets::MeshHandle handle = kenshin::assets::kInvalidMeshHandle;
  glm::vec3 offset;
  glm::vec3 kd;
};

/// Build the shape list for a --mesh argument. A name that matches no
/// demo shape is kept verbatim and later treated as an OBJ path.
std::vector<SceneShape> buildScene(const std::string& meshArg);

/// Load one demo shape by name; unknown names fall back to loadObj.
bool loadMesh(const std::string& name, kenshin::assets::MeshData& out,
              std::string& log);

/// Load + upload every scene shape (handles filled in). Returns true if
/// at least one shape made it to the GPU.
bool uploadScene(std::vector<SceneShape>& scene,
                 kenshin::manager::MeshPool& pool);

}  // namespace demo
