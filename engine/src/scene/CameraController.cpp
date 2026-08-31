#include "kenshin/scene/CameraController.h"

#include <cmath>

KENSHIN_NS_BEGIN

namespace scene
{

    void OrbitCameraController::applyPose(Camera& camera) const
    {
        // Orbit state (yaw/pitch/distance/target) lives in the controller;
        // derived position is written into the camera pose each update.
        const f32 yaw = glm::radians(camera.yawDeg);
        const f32 pitch = glm::radians(camera.pitchDeg);
        const f32 cp = std::cos(pitch);
        camera.position =
            target + distance * glm::vec3(cp * std::sin(yaw), std::sin(pitch), cp * std::cos(yaw));
    }

    void OrbitCameraController::update(Camera& camera, const ControllerInput& input)
    {
        if (input.rotateActive)
        {
            camera.yawDeg -= input.mouseDeltaX * rotateSpeedDegPerPx;
            camera.pitchDeg += input.mouseDeltaY * rotateSpeedDegPerPx;
            camera.pitchDeg = glm::clamp(camera.pitchDeg, -89.0f, 89.0f);
        }

        if (input.wheel != 0.0f)
            distance *= std::pow(dollyFactorPerNotch, input.wheel);

        // Pan the target in the camera's screen plane (W/S depth, A/D lateral,
        // Q/E vertical).
        if (input.moveF || input.moveB || input.moveL || input.moveR ||
            input.moveUp || input.moveDown)
        {
            const f32 scale = panSpeedPerUnit * distance * 100.0f * input.deltaTime;
            const glm::vec3 fwd = camera.forward();
            const glm::vec3 right = camera.right();
            glm::vec3 move(0.0f);
            if (input.moveF) move += fwd;
            if (input.moveB) move -= fwd;
            if (input.moveR) move += right;
            if (input.moveL) move -= right;
            if (input.moveUp) move += glm::vec3(0.0f, 1.0f, 0.0f);
            if (input.moveDown) move -= glm::vec3(0.0f, 1.0f, 0.0f);
            target += move * scale;
        }

        applyPose(camera);
    }

    void FirstPersonCameraController::update(Camera& camera, const ControllerInput& input)
    {
        if (input.rotateActive)
        {
            camera.yawDeg -= input.mouseDeltaX * lookSpeedDegPerPx;
            camera.pitchDeg += input.mouseDeltaY * lookSpeedDegPerPx;
            camera.pitchDeg = glm::clamp(camera.pitchDeg, -89.0f, 89.0f);
        }

        glm::vec3 move(0.0f);
        const glm::vec3 fwd = camera.forward();
        const glm::vec3 right = camera.right();
        if (input.moveF) move += fwd;
        if (input.moveB) move -= fwd;
        if (input.moveR) move += right;
        if (input.moveL) move -= right;
        if (input.moveUp) move += glm::vec3(0.0f, 1.0f, 0.0f);
        if (input.moveDown) move -= glm::vec3(0.0f, 1.0f, 0.0f);

        const f32 speed = moveSpeed * (input.fast ? fastMultiplier : 1.0f);
        if (glm::dot(move, move) > 0.0f)
            camera.position += glm::normalize(move) * speed * input.deltaTime;
    }

} // namespace scene

KENSHIN_NS_END
