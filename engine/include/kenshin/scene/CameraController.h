#pragma once

// ── CameraController (drives the internal Camera) ───────────────
// The ONLY public entry point for camera motion (Camera keeps its state
// private; this class is its friend). Orbit-look + screen-plane pan +
// geometric zoom, all expressed through Camera's spherical state.
//
// Angles in this header are DEGREES (UI-facing); Camera works in radians.

#include "kenshin/Common.h"
#include "kenshin/scene/Camera.h"

#include <glm/glm.hpp>

KENSHIN_NS_BEGIN

namespace scene
{

    /// Input snapshot handed to the controller each frame (filled by the app
    /// from ImGui queries so inputs only apply when the viewport is active).
    struct ControllerInput
    {
        f32 mouseDeltaX = 0.0f;
        f32 mouseDeltaY = 0.0f;
        f32 wheel = 0.0f;          // notches
        bool rotateActive = false;  // drag accepted
        bool moveF = false, moveB = false, moveL = false, moveR = false;
        bool moveUp = false, moveDown = false;
        bool fast = false;         // Shift
        f32 deltaTime = 0.0f;
    };

    class CameraController
    {
    public:
        // --- tuning ---------------------------------------------------------
        f32 rotateSpeedDegPerPx = 0.30f;
        f32 panSpeedPerUnit    = 0.0025f;  // screen fraction per px/unit
        f32 moveSpeed          = 6.0f;     // m/s (WASD target pan)
        f32 fastMultiplier    = 3.0f;

        f32 fovYDeg = 60.0f;   // re-applied to the camera each frame
        f32 nearPlane = 0.01f;
        f32 farPlane  = 1000.0f;

        /// Initial placement from the CLI/defaults. Angles in degrees:
        /// yaw = azimuth around +Z, pitch = elevation above the X/Y plane.
        void setup(f32 yawDeg, f32 pitchDeg, f32 distance,
                   const glm::vec3& target, f32 aspect);

        /// Apply frame input; updates the camera state.
        void handleInput(Camera& camera, const ControllerInput& input);

        /// Push perspective (fov/aspect/near/far) and rebuild matrices.
        /// Call once per frame after handleInput, before rendering.
        void updateMatrices(Camera& camera, f32 aspect);

        // --- state accessors (for UI / debug) --------------------------------
        f32 yawDeg() const { return m_yawDeg; }
        f32 pitchDeg() const { return m_pitchDeg; }
        f32 distance() const { return m_distance; }
        const glm::vec3& target() const { return m_target; }

    private:
        void applyTo(Camera& camera) const;

        f32 m_yawDeg = 35.0f;
        f32 m_pitchDeg = 20.0f;
        f32 m_distance = 8.0f;
        glm::vec3 m_target{0.0f, 0.0f, 0.0f};
    };

} // namespace scene

KENSHIN_NS_END
