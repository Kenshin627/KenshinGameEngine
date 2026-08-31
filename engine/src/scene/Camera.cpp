#include "kenshin/scene/Camera.h"

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

KENSHIN_NS_BEGIN

namespace scene
{

    glm::vec3 Camera::forward() const
    {
        const f32 yaw = glm::radians(yawDeg);
        const f32 pitch = glm::radians(pitchDeg);
        const f32 cp = std::cos(pitch);
        return glm::normalize(glm::vec3(cp * std::sin(yaw), std::sin(pitch), -cp * std::cos(yaw)));
    }

    glm::vec3 Camera::right() const
    {
        return glm::normalize(glm::cross(forward(), glm::vec3(0.0f, 1.0f, 0.0f)));
    }

    glm::mat4 Camera::view() const
    {
        return glm::lookAtRH(position, position + forward(), glm::vec3(0.0f, 1.0f, 0.0f));
    }

    glm::mat4 Camera::projection(f32 aspect) const
    {
        glm::mat4 proj = glm::perspectiveRH_ZO(glm::radians(fovYDeg), aspect, nearZ, farZ);
        proj[1][1] *= -1.0f; // Vulkan clip-space Y points down
        return proj;
    }

    bool Camera::isOrthonormal(f32 epsilon) const
    {
        const glm::mat4 v = view();
        const glm::vec3 x = glm::vec3(v[0]);
        const glm::vec3 y = glm::vec3(v[1]);
        const glm::vec3 z = glm::vec3(v[2]);
        const auto nearOne = [epsilon](f32 v0) { return std::fabs(v0 - 1.0f) < epsilon; };
        const auto nearZero = [epsilon](f32 v0) { return std::fabs(v0) < epsilon; };
        if (!nearOne(glm::length(x)) || !nearOne(glm::length(y)) || !nearOne(glm::length(z)))
            return false;
        if (!nearZero(glm::dot(x, y)) || !nearZero(glm::dot(x, z)) || !nearZero(glm::dot(y, z)))
            return false;
        return true;
    }

} // namespace scene

KENSHIN_NS_END
