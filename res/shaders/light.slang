#type VERTEX
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoords;

out vec2 TexCoords;

void main()
{
  TexCoords = aTexCoords;
  gl_Position = vec4(aPos, 1.0);
}

#type FRAGMENT
#version 460 core
#extension GL_ARB_bindless_texture : require

#define GABGL_FLOAT3 vec3
#define GABGL_SATURATE(value) clamp(value, 0.0, 1.0)
#include "common/material_lighting.slang"

in vec2 TexCoords;

layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 BrightColor;

layout(binding = 1) uniform sampler2D gPosition;
layout(binding = 2) uniform sampler2D gNormal;
layout(binding = 3) uniform sampler2D gAlbedoSpec;
layout(binding = 4) uniform sampler2D u_DirectShadow;
layout(binding = 5) uniform sampler3D u_OffsetTexture;
layout(binding = 6) uniform samplerCubeArray u_OmniShadow;

layout(std140, binding = 0) uniform Camera
{
  mat4 ViewProjection;
  mat4 OrtoProjection;
  mat4 NonRotViewProjection;
  vec3 CameraPos;
};

layout(std140, binding = 2) uniform DirectShadowData
{
  vec2 windowSize;
  vec2 offsetSize_filterSize;
  vec2 randomRadius;
};

layout(std430, binding = 0) buffer LightPositions    { vec4 positions[]; };
layout(std430, binding = 1) buffer LightRotations    { vec4 rotations[]; };
layout(std430, binding = 2) buffer LightsQuantity    { int numLights;    };
layout(std430, binding = 3) buffer LightColors       { vec4 colors[];    };
layout(std430, binding = 4) buffer LightTypes        { int lightTypes[]; };
layout(std430, binding = 21) readonly buffer TileLightGrid
{
  uvec2 tileLightGrid[];
};
layout(std430, binding = 22) readonly buffer TileLightIndices
{
  uint tileLightIndices[];
};
layout(std430, binding = 23) readonly buffer PointShadowSlots
{
  int pointShadowSlots[];
};

uniform mat4 u_DirectShadowViewProj;
uniform bool u_DirectShadowsEnabled;
uniform bool u_PointShadowsEnabled;
uniform bool u_TiledLightingEnabled;
uniform int u_TiledDebugMode;
uniform float u_BloomThreshold = 1.0;

const int TILE_SIZE = 16;
const float LOCAL_LIGHT_RADIUS = 20.0;

const float gamma = 1.2;

float gammaCorrection(float value) {
  return pow(value, 1.0 / gamma);
}

vec3 gammaCorrection(vec3 value) {
  return pow(value, vec3(1.0 / gamma));
}

// ACES tone mapping
vec3 toneMappingACES(vec3 color) {
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;
  return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

float calculateDirectShadow(vec4 fragPosLightSpace, vec3 lightDir, vec3 normal) {
  if (!u_DirectShadowsEnabled || fragPosLightSpace.w <= 0.0)
    return 1.0;

  vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
  vec3 shadowCoords = projCoords * 0.5 + 0.5;
  if (shadowCoords.z <= 0.0 || shadowCoords.z >= 1.0 ||
      any(lessThan(shadowCoords.xy, vec2(0.0))) || any(greaterThan(shadowCoords.xy, vec2(1.0))))
    return 1.0;

  float texelSizeX = 1.0 / windowSize.x;
  float texelSizeY = 1.0 / windowSize.y;
  vec2 texelSize = vec2(texelSizeX, texelSizeY);

  // Anchor the Poisson pattern to the shadow map rather than to screen
  // pixels. gl_FragCoord changes whenever the camera moves and made a fixed
  // world-space point select a different kernel every frame.
  ivec2 patternSize = max(ivec2(offsetSize_filterSize.xx), ivec2(1));
  ivec2 shadowTexel = ivec2(floor(shadowCoords.xy * windowSize));
  ivec3 OffsetCoord = ivec3(0, shadowTexel % patternSize);

  float diffuseFactor = max(dot(normalize(normal), normalize(lightDir)), 0.0);
  float bias = max(0.0015 * (1.0 - diffuseFactor), 0.00015);

  float sum = 0.0;
  int sampleCount = int(offsetSize_filterSize.y * offsetSize_filterSize.y);
  int samplesDiv2 = sampleCount / 2;

  for (int i = 0; i < samplesDiv2; ++i) {
    OffsetCoord.x = i;
    vec4 offsets = texelFetch(u_OffsetTexture, OffsetCoord, 0) * randomRadius.x;

    vec2 offsets1 = shadowCoords.xy + offsets.rg * texelSize;
    float depth1 = texture(u_DirectShadow, offsets1).r;
    sum += (depth1 + bias < shadowCoords.z) ? 0.15 : 1.0;

    vec2 offsets2 = shadowCoords.xy + offsets.ba * texelSize;
    float depth2 = texture(u_DirectShadow, offsets2).r;
    sum += (depth2 + bias < shadowCoords.z) ? 0.15 : 1.0;
  }

  return sum / float(sampleCount);
}

float calculatePointShadow(vec3 fragPos, vec3 lightPos, vec3 normal, int shadowSlot) {
  if (!u_PointShadowsEnabled || shadowSlot < 0)
    return 1.0;

  vec3 fragToLight = fragPos - lightPos;
  float currentDepth = length(fragToLight);
  if (currentDepth >= 20.0)
    return 1.0;
  vec3 direction = normalize(fragToLight);

  float bias = max(0.03 * (1.0 - dot(normalize(normal), -direction)), 0.003);
  float shadowResolution = float(textureSize(u_OmniShadow, 0).x);
  float filterRadius = (2.0 / max(shadowResolution, 1.0)) *
    (1.0 + 2.0 * currentDepth / 20.0);
  // A tetrahedral kernel covers the cubemap direction with half the texture
  // fetches of the previous eight-corner kernel.
  const vec3 sampleOffsets[4] = vec3[4](
    vec3( 1.0,  1.0,  1.0), vec3(-1.0, -1.0,  1.0),
    vec3(-1.0,  1.0, -1.0), vec3( 1.0, -1.0, -1.0));
  float occludedSamples = 0.0;
  for (int sampleIndex = 0; sampleIndex < 4; ++sampleIndex)
  {
    float closestDepth = texture(u_OmniShadow,
      vec4(direction + sampleOffsets[sampleIndex] * filterRadius,
           float(shadowSlot))).r;
    occludedSamples += currentDepth - bias > closestDepth ? 1.0 : 0.0;
  }
  return mix(1.0, 0.15, occludedSamples * 0.25);
}

vec3 tileHeatmap(float value)
{
  value = clamp(value, 0.0, 1.0);
  if (value <= 0.0)
    return vec3(0.01);
  if (value < 0.25)
    return mix(vec3(0.02, 0.08, 0.8), vec3(0.0, 0.9, 1.0), value * 4.0);
  if (value < 0.5)
    return mix(vec3(0.0, 0.9, 1.0), vec3(0.1, 1.0, 0.1), (value - 0.25) * 4.0);
  if (value < 0.75)
    return mix(vec3(0.1, 1.0, 0.1), vec3(1.0, 0.9, 0.0), (value - 0.5) * 4.0);
  return mix(vec3(1.0, 0.9, 0.0), vec3(1.0, 0.02, 0.0), (value - 0.75) * 4.0);
}

vec3 calculateDirectionalLight(vec3 lightDir, vec3 normal, vec3 viewDir, vec3 surfaceColor, vec4 fragPosLightSpace, vec3 ambientCol, vec3 specularCol, vec3 lightColor, float shininess) {
  lightDir = normalize(-lightDir);
  float shadow = calculateDirectShadow(fragPosLightSpace, lightDir, normal);
  return GabglEvaluateLight(lightColor, surfaceColor, specularCol, normal,
    viewDir, lightDir, shininess, 1.0, 1.0, shadow);
}

vec3 calculatePointLight(vec3 lightPos, vec3 fragPos, vec3 normal, vec3 viewDir, vec3 surfaceColor, vec3 ambientCol, vec3 specularCol, int shadowSlot, vec3 lightColor, float shininess) {
  vec3 lightDir = normalize(lightPos - fragPos);
  float distance = length(lightPos - fragPos);
  float attenuation = GabglDistanceAttenuation(distance);
  attenuation *= 1.0 - smoothstep(LOCAL_LIGHT_RADIUS * 0.8,
    LOCAL_LIGHT_RADIUS, distance);
  float shadow = calculatePointShadow(fragPos, lightPos, normal, shadowSlot);
  return GabglEvaluateLight(lightColor, surfaceColor, specularCol, normal,
    viewDir, lightDir, shininess, attenuation, 1.0, shadow);
}

vec3 calculateSpotlight(vec3 lightPos, vec3 lightDir, vec3 fragPos, vec3 normal, vec3 viewDir, vec3 surfaceColor, vec3 ambientCol, vec3 specularCol, vec3 lightColor, float shininess) {
  vec3 lightToFrag = normalize(fragPos - lightPos);
  float intensity = GabglSpotIntensity(lightToFrag, lightDir);
  float distance = length(lightPos - fragPos);
  float attenuation = GabglDistanceAttenuation(distance);
  attenuation *= 1.0 - smoothstep(LOCAL_LIGHT_RADIUS * 0.8,
    LOCAL_LIGHT_RADIUS, distance);
  return GabglEvaluateLight(lightColor, surfaceColor, specularCol, normal,
    viewDir, -lightToFrag, shininess, attenuation, intensity, 1.0);
}

void main()
{
  vec4 PositionData = texture(gPosition, TexCoords);
  if (PositionData.a <= 0.0)
  {
    FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    BrightColor = vec4(0.0);
    return;
  }
  vec3 FragPos = PositionData.rgb;
  vec3 Normal = normalize(texture(gNormal, TexCoords).rgb);
  vec4 AlbedoSpec = texture(gAlbedoSpec, TexCoords);
  vec3 Color = AlbedoSpec.rgb;
  vec3 viewDir = normalize(CameraPos - FragPos);
  vec4 fragPosLightSpace = u_DirectShadowViewProj * vec4(FragPos, 1.0);
  vec3 specularCol = vec3(AlbedoSpec.a);

  float shininess = GabglMaterialShininess(AlbedoSpec.a);

  vec3 result = vec3(0.0);

  uvec2 tileEntry = uvec2(0u, uint(numLights));
  if (u_TiledLightingEnabled)
  {
    const ivec2 framebufferSize = textureSize(gPosition, 0);
    const int tileCountX = (framebufferSize.x + TILE_SIZE - 1) / TILE_SIZE;
    const ivec2 tile = ivec2(gl_FragCoord.xy) / TILE_SIZE;
    tileEntry = tileLightGrid[tile.y * tileCountX + tile.x];
  }

  for (uint lightOffset = 0u; lightOffset < tileEntry.y; ++lightOffset)
  {
    const int i = u_TiledLightingEnabled
      ? int(tileLightIndices[tileEntry.x + lightOffset])
      : int(lightOffset);
    vec3 position = positions[i].xyz;
    vec3 rotation = rotations[i].xyz;
    vec3 lightColor = colors[i].rgb;
    vec3 ambient = lightColor * 0.1;

    int type = lightTypes[i];

    if (type == 0) {
      result += calculateDirectionalLight(rotation, Normal, viewDir, Color, fragPosLightSpace, ambient, specularCol, lightColor, shininess);
    } else if (type == 1) {
      result += calculatePointLight(position, FragPos, Normal, viewDir, Color, ambient,
        specularCol, pointShadowSlots[i], lightColor, shininess);
    } else if (type == 2) {
      result += calculateSpotlight(position, rotation, FragPos, Normal, viewDir, Color, ambient, specularCol, lightColor, shininess);
    }
  }

  result *= max(PositionData.a, 1.0);

  if (u_TiledLightingEnabled && u_TiledDebugMode != 0)
  {
    const float density = numLights > 0
      ? clamp(float(tileEntry.y) / float(numLights), 0.0, 1.0)
      : 0.0;
    vec3 debugColor = tileHeatmap(density);
    const vec2 tilePixel = mod(gl_FragCoord.xy, float(TILE_SIZE));
    const bool border = tilePixel.x < 1.0 || tilePixel.y < 1.0;
    if (border) debugColor = vec3(0.02);
    result = u_TiledDebugMode == 1
      ? mix(result, debugColor, border ? 0.85 : 0.55)
      : debugColor;
  }

  float brightness = dot(result, vec3(0.2126, 0.7152, 0.0722));
  BrightColor = brightness > max(u_BloomThreshold, 0.0)
    ? vec4(result, 1.0) : vec4(0.0, 0.0, 0.0, 1.0);
  FragColor = vec4(result, 1.0);
}


