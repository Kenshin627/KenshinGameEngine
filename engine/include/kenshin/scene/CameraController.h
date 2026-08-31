#pragma once

// ── Camera controllers (S1.5) ────────────────────────────────────
// Two flavors over one Camera pose: orbit (turntable: drag rotates around a
// target, wheel dollies, WASD/QE pans the target) and first-person (drag
// looks around, WASD flies, Q/E or Space/Ctrl moves vertically).

#include "kenshin/scene/Camera.h"

#include <glm/glm.hpp>

KENSHIN_NS_BEGIN

namespace scene
{

    /// Input snapshot handed to controllers each frame (filled by the app
    /// from ImGui queries so inputs only apply when the viewport is active).
    struct ControllerInput
    {
        f32 mouseDeltaX = 0.0f;
        f32 mouseDeltaY = 0.0f;
        f32 wheel = 0.0f;          // notches
        bool rotateActive = false; // drag accepted
        bool moveF = false, moveB = false, moveL = false, moveR = false;
        bool moveUp = false, moveDown = false;
        bool fast = false;            // Shift
        f32 deltaTime = 0.0f;
    };

    class ICameraController
    {
    public:
        virtual ~ICameraController() = default;

        /// Apply input; writes the resulting pose into `camera`.
        virtual void update(Camera& camera, const ControllerInput& input) = 0;

        virtual const char* name() const = 0;
    };

    /// Turntable orbit around a target point.
    class OrbitCameraController final : public ICameraController
    {
    public:
        glm::vec3 target {0.0f, 0.5f, 0.0f};
        f32 distance = 8.0f;
        f32 rotateSpeedDegPerPx = 0.4f;
        f32 panSpeedPerUnit = 0.0025f;
        f32 dollyFactorPerNotch = 0.9f;

        void update(Camera& camera, const ControllerInput& input) override;
        const char* name() const override { return "Orbit"; }

    private:
        void applyPose(Camera& camera) const;
    };

    /// Free-flying first person (WASD + mouse look).
    class FirstPersonCameraController final : public ICameraController
    {
    public:
        f32 lookSpeedDegPerPx = 0.25f;
        f32 moveSpeed = 4.0f;          // meters per second
        f32 fastMultiplier = 3.0f;     // with Shift

        void update(Camera& camera, const ControllerInput& input) override;
        const char* name() const override { return "FirstPerson"; }
    };

} // namespace scene

KENSHIN_NS_END
