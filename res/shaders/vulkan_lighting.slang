#api VULKAN

[[vk::binding(0, 0)]] Texture2D<float4> GBufferAlbedo;
[[vk::binding(1, 0)]] Texture2D<float4> GBufferNormal;
[[vk::binding(2, 0)]] Texture2D<float4> GBufferPosition;
[[vk::binding(3, 0)]] SamplerState GBufferSampler;

struct SceneFrameData
{
  float4x4 LightViewProjection;
  float4x4 InverseViewProjection;
  float4 CameraPosition;
  float4 LightPositions[128];
  float4 LightDirections[128];
  float4 LightColors[128];
  float4 Params;
  float4 Effects;
  float4 ShadowInfo;
  float4 PointShadowIndices;
  float4 TileInfo;
  float4 ViewportInfo;
};

[[vk::binding(4, 0)]] ConstantBuffer<SceneFrameData> Frame;
[[vk::binding(5, 0)]] Texture2D<float> ShadowTexture;
[[vk::binding(6, 0)]] SamplerState ShadowSampler;
[[vk::binding(7, 0)]] TextureCube<float4> SkyboxTexture;
[[vk::binding(8, 0)]] TextureCube<float> PointShadow0;
[[vk::binding(9, 0)]] TextureCube<float> PointShadow1;
[[vk::binding(10, 0)]] TextureCube<float> PointShadow2;
[[vk::binding(11, 0)]] TextureCube<float> PointShadow3;
[[vk::binding(12, 0)]] SamplerState PointShadowSampler;
[[vk::binding(13, 0)]] StructuredBuffer<uint2> TileGrid;
[[vk::binding(14, 0)]] StructuredBuffer<uint> TileIndices;
[[vk::binding(15, 0)]] StructuredBuffer<float2> ShadowOffsets;

#define GABGL_FLOAT3 float3
#define GABGL_SATURATE(value) saturate(value)
#include "common/material_lighting.slang"

struct VertexOutput
{
  float4 Position : SV_Position;
  float2 TexCoords : TEXCOORD;
};

VertexOutput VSMain(float2 position : POSITION)
{
  VertexOutput output;
  output.Position = float4(position, 0.0f, 1.0f);
  output.TexCoords = position * 0.5f + 0.5f;
  return output;
}

float DirectionalShadow(float3 worldPosition, float3 normal, float3 toLight)
{
  if (Frame.Params.y < 0.5f) return 1.0f;
  const float4 shadowPosition = mul(Frame.LightViewProjection, float4(worldPosition, 1.0f));
  const float3 projected = shadowPosition.xyz / max(shadowPosition.w, 0.00001f);
  const float2 uv = projected.xy * float2(0.5f, -0.5f) + 0.5f;
  if (any(uv < 0.0f) || any(uv > 1.0f) || projected.z < 0.0f || projected.z > 1.0f)
    return 1.0f;
  const float bias = max(0.0015f * (1.0f - saturate(dot(normal, toLight))), 0.00015f);
  float visibility = 0.0f;
  const uint filterSize = clamp((uint)Frame.TileInfo.w, 2u, 4u);
  const uint sampleCount = filterSize * filterSize;
  uint shadowWidth, shadowHeight;
  ShadowTexture.GetDimensions(shadowWidth, shadowHeight);
  const uint2 pattern = uint2(floor(
    float2(uv.x, 1.0f - uv.y) * float2(shadowWidth, shadowHeight))) % 16u;
  const uint offsetBase = (pattern.y * 16u + pattern.x) * sampleCount;
  const float radius = 0.5f + 0.5f * (float)filterSize;
  for (uint sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
  {
    const float2 offset = ShadowOffsets[offsetBase + sampleIndex] *
                          radius * float2(1.0f, -1.0f) * Frame.ShadowInfo.xx;
    const float depth = ShadowTexture.SampleLevel(ShadowSampler, uv + offset, 0).r;
    visibility += projected.z - bias <= depth ? 1.0f : 0.15f;
  }
  return visibility / max((float)sampleCount, 1.0f);
}

float PointShadowDepth(int slot, float3 direction)
{
  if (slot == 0) return PointShadow0.Sample(PointShadowSampler, direction);
  if (slot == 1) return PointShadow1.Sample(PointShadowSampler, direction);
  if (slot == 2) return PointShadow2.Sample(PointShadowSampler, direction);
  return PointShadow3.Sample(PointShadowSampler, direction);
}

float PointShadow(int lightIndex, float3 worldPosition, float3 normal, float3 toLight)
{
  if (Frame.ShadowInfo.z < 0.5f) return 1.0f;
  int slot = -1;
  for (int candidate = 0; candidate < 4; ++candidate)
    if ((int)Frame.PointShadowIndices[candidate] == lightIndex) slot = candidate;
  if (slot < 0) return 1.0f;
  const float3 offset = worldPosition - Frame.LightPositions[lightIndex].xyz;
  const float distanceToLight = length(offset);
  if (distanceToLight >= Frame.ShadowInfo.w || distanceToLight <= 0.0001f) return 1.0f;
  const float3 direction = offset / distanceToLight;
  const float bias = max(0.03f * (1.0f - dot(normal, toLight)), 0.003f);
  uint shadowWidth, shadowHeight;
  PointShadow0.GetDimensions(shadowWidth, shadowHeight);
  const float filterRadius = (2.0f / max((float)shadowWidth, 1.0f)) *
    (1.0f + 2.0f * distanceToLight / Frame.ShadowInfo.w);
  const float3 offsets[4] = {
    float3(1, 1, 1), float3(-1, -1, 1),
    float3(-1, 1, -1), float3(1, -1, -1)};
  float occluded = 0.0f;
  for (uint sampleIndex = 0; sampleIndex < 4; ++sampleIndex)
    occluded += distanceToLight - bias > PointShadowDepth(
      slot, direction + offsets[sampleIndex] * filterRadius) ? 1.0f : 0.0f;
  return lerp(1.0f, 0.15f, occluded * 0.25f);
}

float4 PSMain(VertexOutput input) : SV_Target
{
  const float4 albedoSample = GBufferAlbedo.Sample(GBufferSampler, input.TexCoords);
  if (albedoSample.a < 0.5f)
    return float4(0.0f, 0.0f, 0.0f, 1.0f);
  const float4 normalSpecular = GBufferNormal.Sample(GBufferSampler, input.TexCoords);
  const float4 positionData = GBufferPosition.Sample(GBufferSampler, input.TexCoords);
  const float3 worldPosition = positionData.xyz;
  const float3 normal = normalize(normalSpecular.xyz);
  const float3 viewDirection = normalize(Frame.CameraPosition.xyz - worldPosition);
  const float3 specularColor = normalSpecular.www;
  const float shininess = GabglMaterialShininess(normalSpecular.w);
  float3 color = 0.0f;
  const int lightCount = min((int)Frame.Params.x, 128);
  uint tileOffset = 0;
  int tileLightCount = lightCount;
  if (Frame.TileInfo.y > 0.5f)
  {
    const uint2 pixel = uint2(input.Position.xy);
    const uint tileIndex = (pixel.y / 16) * (uint)Frame.TileInfo.x + pixel.x / 16;
    const uint2 tile = TileGrid[tileIndex];
    tileOffset = tile.x;
    tileLightCount = min((int)tile.y, 128);
  }
  for (int tileLight = 0; tileLight < tileLightCount; ++tileLight)
  {
    const int index = Frame.TileInfo.y > 0.5f
      ? (int)TileIndices[tileOffset + tileLight] : tileLight;
    const int type = (int)Frame.LightPositions[index].w;
    float3 toLight;
    float attenuation = 1.0f;
    float intensity = 1.0f;
    if (type == 0)
      toLight = normalize(-Frame.LightDirections[index].xyz);
    else
    {
      const float3 offset = Frame.LightPositions[index].xyz - worldPosition;
      const float distanceToLight = max(length(offset), 0.0001f);
      toLight = offset / distanceToLight;
      attenuation = GabglDistanceAttenuation(distanceToLight) *
        (1.0f - smoothstep(16.0f, 20.0f, distanceToLight));
      if (type == 2)
        intensity = GabglSpotIntensity(-toLight, Frame.LightDirections[index].xyz);
    }
    const float shadow = type == 0
      ? DirectionalShadow(worldPosition, normal, toLight)
      : (type == 1 ? PointShadow(index, worldPosition, normal, toLight) : 1.0f);
    color += GabglEvaluateLight(
      Frame.LightColors[index].rgb, albedoSample.rgb, specularColor,
      normal, viewDirection, toLight, shininess,
      attenuation, intensity, shadow);
  }
  color *= max(positionData.a, 1.0f);
  if (Frame.TileInfo.z > 0.5f)
  {
    const float occupancy = (float)tileLightCount / max((float)lightCount, 1.0f);
    float3 heat = saturate(float3(
      1.5f - abs(4.0f * occupancy - 3.0f),
      1.5f - abs(4.0f * occupancy - 2.0f),
      1.5f - abs(4.0f * occupancy - 1.0f)));
    const bool border = ((uint)input.Position.x % 16 == 0) ||
                        ((uint)input.Position.y % 16 == 0);
    if (border) heat = 0.02f;
    color = Frame.TileInfo.z < 1.5f
      ? lerp(color, heat, border ? 0.85f : 0.55f) : heat;
  }
  return float4(color, 1.0f);
}
