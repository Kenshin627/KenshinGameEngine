#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "kenshin/Common.h"

KENSHIN_NS_BEGIN

// ---------------------------------------------------------------------------
// Ray — a world-space ray (origin + unit-ish direction).
// The single shared ray type for the whole engine (camera picking, gizmo
// hit-tests, ray-plane math).  Do not define a second copy anywhere.
// ---------------------------------------------------------------------------
struct Ray 
{
    glm::vec3 origin;
    glm::vec3 dir;
};

/// Ray-plane intersection. Returns the ray parameter t at which the ray
/// crosses the plane, or -1.0f when the ray is (near-)parallel to the plane
/// or the hit lies behind the ray's origin (t < 0).
inline float rayPlaneT(const Ray& ray, const glm::vec3& planePoint, const glm::vec3& planeNormal)
{
	float denom = glm::dot(ray.dir, planeNormal);
    if (std::abs(denom) < 0.000001f) return -1.0f;
	float t = glm::dot(planeNormal, planePoint - ray.origin) / denom;
    return t >= 0.0f ? t : -1.0f;
}

KENSHIN_NS_END
