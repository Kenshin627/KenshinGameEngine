#pragma once

// ── Camera (S1.5) ────────────────────────────────────────────────
// Pure pose + projection: position and yaw/pitch orientation. Controllers
// (orbit / first-person) own their own input mapping and write the pose.
// Conventions: right-handed, Y-up, meters; Vulkan projection with depth
// [0,1] and flipped clip-space Y (see Camera.cpp).

#include "kenshin/Common.h"

#include <glm/glm.hpp>

KENSHIN_NS_BEGIN

namespace scene
{

    class Camera
    {
    public:
        // Pose (owned by controllers)
        glm::vec3 position {0.0f, 2.0f, 8.0f};
        f32 yawDeg = -90.0f;   // -90 looks down -Z in RH Y-up
        f32 pitchDeg = -10.0f;

        // Perspective parameters
        f32 fovYDeg = 60.0f;
        f32 nearZ = 0.01f;
        f32 farZ = 1000.0f;

        glm::vec3 forward() const;
        glm::vec3 right() const;

        /// Right-handed view matrix (look along forward()).
        glm::mat4 view() const;

        /// Vulkan perspective: depth [0,1], Y flipped for clip space.
        glm::mat4 projection(f32 aspect) const;

        /// True when the view basis is orthonormal (sanity check, S1.5 gate).
        bool isOrthonormal(f32 epsilon = 1e-4f) const;
    };

} // namespace scene

KENSHIN_NS_END
