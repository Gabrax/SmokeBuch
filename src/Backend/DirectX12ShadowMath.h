#pragma once

#include <array>
#include <cstdint>
#include <glm/gtc/matrix_transform.hpp>

namespace DX12Shadow
{
inline glm::mat4 PointViewProjection(const glm::vec3& lightPosition,
                                    uint32_t face, float nearPlane, float farPlane)
{
  static constexpr std::array<glm::vec3, 6> directions = {
    glm::vec3(1, 0, 0), glm::vec3(-1, 0, 0),
    glm::vec3(0, 1, 0), glm::vec3(0, -1, 0),
    glm::vec3(0, 0, 1), glm::vec3(0, 0, -1)
  };
  static constexpr std::array<glm::vec3, 6> upDirections = {
    glm::vec3(0, 1, 0), glm::vec3(0, 1, 0),
    glm::vec3(0, 0, -1), glm::vec3(0, 0, 1),
    glm::vec3(0, 1, 0), glm::vec3(0, 1, 0)
  };
  // TextureCube uses these face bases with a left-handed view. A right-handed
  // lookAt mirrors each face horizontally relative to the sampler. Keep both
  // handedness and the D3D [0,1] depth range explicit, independent of GLM flags.
  return glm::perspectiveLH_ZO(glm::radians(90.0f), 1.0f, nearPlane, farPlane)
    * glm::lookAtLH(lightPosition, lightPosition + directions[face], upDirections[face]);
}
}
