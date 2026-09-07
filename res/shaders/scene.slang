#api OPENGL
#type VERTEX
#version 460 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec3 tangent;
layout (location = 4) in vec3 bitangent;
layout (location = 5) in ivec4 boneIds; 
layout (location = 6) in vec4 weights;

layout(std140, binding = 0) uniform Camera
{
  mat4 ViewProjection;
	mat4 OrtoProjection;
	mat4 NonRotViewProjection;
  vec3 CameraPos;
};

layout(std140, binding = 1) uniform Resolution
{
  vec2 resolution;
};

layout(std430, binding = 5) buffer ModelTransforms    { mat4 transforms[];     };
layout(std430, binding = 6) buffer MeshToTransformMap { int meshToTransform[]; };

out VS_OUT
{
  vec3 FragPos;
  vec3 Normal;
  vec2 TexCoords;
  mat3 TBN;
  flat uint DrawID;
} vs_out;

const int MAX_BONES = 100;
const int MAX_BONE_INFLUENCE = 4;
layout(std430, binding = 9) buffer FinalBoneMatrices  { mat4 boneMatrices[];    };
layout(std430, binding = 10) buffer ModelIsAnimated   { int modelIsAnimated[];  };
layout(std430, binding = 13) readonly buffer InstanceTransforms { mat4 instanceTransforms[]; };
layout(std430, binding = 14) readonly buffer DrawRemap { uint compactDrawToOriginal[]; };

#define GABGL_MATRIX4 mat4
#define GABGL_INT4 ivec4
#define GABGL_FLOAT4 vec4
#define GABGL_BONE_MATRIX(index) boneMatrices[index]
#include "common/skinning.slang"

uniform bool u_PS1Enabled;
uniform float u_PS1VirtualHeight;
uniform bool u_PreviewMode;
uniform int u_PreviewDrawID;
uniform mat4 u_PreviewModel;
uniform bool u_GPUDriven;

void main()
{
  uint drawID = u_PreviewMode ? uint(u_PreviewDrawID)
    : (u_GPUDriven ? compactDrawToOriginal[gl_DrawID] : uint(gl_DrawID));
  vs_out.DrawID = drawID;
  int transformIndex = meshToTransform[drawID];
  bool isAnimated = !u_PreviewMode && (modelIsAnimated[transformIndex] == 1);
  mat4 modelMat = u_PreviewMode
    ? u_PreviewModel
    : instanceTransforms[gl_BaseInstance + gl_InstanceID];
  mat4 skinMatrix = GabglSkinMatrix(
    boneIds, weights, transformIndex * MAX_BONES, isAnimated);

  mat4 localToWorld = modelMat * skinMatrix;
  vec4 worldPos = localToWorld * vec4(aPos, 1.0);
  mat3 linearTransform = mat3(localToWorld);
  // The cofactor matrix is proportional to inverse-transpose but avoids a
  // full matrix inverse for every vertex. Preserve orientation for mirrored
  // transforms; normalization removes the remaining determinant magnitude.
  mat3 normalTransform = mat3(
    cross(linearTransform[1], linearTransform[2]),
    cross(linearTransform[2], linearTransform[0]),
    cross(linearTransform[0], linearTransform[1]));
  normalTransform *= determinant(linearTransform) < 0.0 ? -1.0 : 1.0;

  vec3 N = normalize(normalTransform * aNormal);
  vec3 transformedTangent = linearTransform * tangent;
  vec3 T = length(transformedTangent) > 0.00001
    ? normalize(transformedTangent - N * dot(N, transformedTangent))
    : normalize(abs(N.y) < 0.999 ? cross(vec3(0.0, 1.0, 0.0), N) : cross(vec3(1.0, 0.0, 0.0), N));
  float handedness = dot(cross(aNormal, tangent), bitangent) < 0.0 ? -1.0 : 1.0;
  vec3 B = normalize(cross(N, T)) * handedness;

  vs_out.FragPos = worldPos.xyz;
  vs_out.Normal = N;
  vs_out.TBN = mat3(T, B, N);
  vs_out.TexCoords = aTexCoords;

  vec4 clipPosition = ViewProjection * worldPos;
  vec2 outputResolution = max(resolution, vec2(1.0));
  float virtualHeight = max(u_PS1VirtualHeight, 1.0);
  vec2 snapResolution = vec2(
    max(floor(virtualHeight * outputResolution.x / outputResolution.y + 0.5), 1.0),
    virtualHeight);
  float safeW = abs(clipPosition.w) > 0.00001 ? clipPosition.w : 0.00001;
  if (u_PS1Enabled)
  {
    vec2 ndc = clipPosition.xy / safeW;
    vec2 snappedNdc = (floor((ndc * 0.5 + 0.5) * snapResolution + 0.5) / snapResolution) * 2.0 - 1.0;
    clipPosition.xy = snappedNdc * clipPosition.w;
  }
  gl_Position = clipPosition;
}

#type FRAGMENT
#version 460 core
#extension GL_ARB_bindless_texture : require

#define GABGL_FLOAT3 vec3
#define GABGL_SATURATE(value) clamp(value, 0.0, 1.0)
#include "common/material_lighting.slang"

layout(location = 0) out vec4 gPosition;
layout(location = 1) out vec3 gNormal;
layout(location = 2) out vec4 gAlbedoSpec;

layout(std430, binding = 7) buffer MeshTextures      { sampler2D meshTextures[]; };
layout(std430, binding = 8) buffer MeshTextureRanges { uvec2 meshTextureRanges[]; };

layout(std430, binding = 11) buffer NormalMapFlags    { int normalMapFlags[];   };
layout(std430, binding = 12) buffer SpecularMapFlags  { int specularMapFlags[]; };

uniform bool u_PreviewMode;
uniform float u_PreviewBrightness;

struct Material
{
  sampler2D diffuse;
  sampler2D normalMap;  
  sampler2D specular;    
  float shininess;
};

in VS_OUT
{
  vec3 FragPos;
  vec3 Normal;
  vec2 TexCoords;
  mat3 TBN;
  flat uint DrawID;
} fs_in;

void main()
{
  Material material;
  uvec2 texRange = meshTextureRanges[fs_in.DrawID];
  material.diffuse = meshTextures[texRange.x];
  material.normalMap = meshTextures[texRange.x + 1];
  material.specular = meshTextures[texRange.x + 2];
  material.shininess = 32.0;

  gPosition = vec4(fs_in.FragPos,
    u_PreviewMode ? max(u_PreviewBrightness, 1.0) : 1.0);

  vec3 normal;
  if (normalMapFlags[fs_in.DrawID] == 1)
  {
    normal = GabglApplyTangentNormal(
      normalize(fs_in.Normal), fs_in.TBN[0], fs_in.TBN[1],
      texture(material.normalMap, fs_in.TexCoords).rgb);
  }
  else
  {
    normal = normalize(fs_in.Normal);
  }
  gNormal = normal;

  vec3 albedo = texture(material.diffuse, fs_in.TexCoords).rgb;
  gAlbedoSpec.rgb = albedo;

  float specular = 0.0;
  if (specularMapFlags[fs_in.DrawID] == 1)
  {
      specular = texture(material.specular, fs_in.TexCoords).r;
  }
  gAlbedoSpec.a = specular;
}

#api DX12
#include "common/dx12_scene.slang"

cbuffer ShadowFrame : register(b1)
{
  float4x4 ShadowViewProjection;
  float4 ShadowLightPosition;
};

#define GABGL_MATRIX4 float4x4
#define GABGL_INT4 int4
#define GABGL_FLOAT4 float4
#define GABGL_BONE_MATRIX(index) Bones[index]
#include "common/skinning.slang"

#include "common/dx12_instance.slang"
StructuredBuffer<InstanceTransform> InstanceTransforms : register(t5);
StructuredBuffer<uint2> TileGrid : register(t6);
StructuredBuffer<uint> TileIndices : register(t7);
StructuredBuffer<float2> ShadowOffsets : register(t8);
Texture2D MaterialTextures[2048] : register(t0, space1);
Texture2D DiffuseTexture : register(t0);
Texture2D NormalTexture : register(t1);
Texture2D SpecularTexture : register(t2);
Texture2D ShadowTexture : register(t3);
TextureCubeArray PointShadowTexture : register(t4);
SamplerState MaterialSampler : register(s0);
SamplerState ShadowSampler : register(s1);

#define GABGL_FLOAT3 float3
#define GABGL_SATURATE(value) saturate(value)
#include "common/material_lighting.slang"

struct VSInput
{
  float3 position : POSITION;
  float3 normal : NORMAL;
  float2 uv : TEXCOORD;
  float3 tangent : TANGENT;
  float3 bitangent : BITANGENT;
  int4 boneIds : BONEIDS;
  float4 weights : BONEWEIGHTS;
};

struct PSInput
{
  float4 position : SV_POSITION;
  float3 worldPosition : POSITION0;
  float3 normal : NORMAL;
  float3 tangent : TANGENT;
  float3 bitangent : BITANGENT;
  float2 uv : TEXCOORD;
  float4 shadowPosition : TEXCOORD1;
};

PSInput VSMain(VSInput input, uint instanceID : SV_InstanceID)
{
  const float4x4 skin = GabglSkinMatrix(
    input.boneIds, input.weights, 0, CameraPosition.w > 0.5f);
  // SV_InstanceID excludes DrawIndexedInstanced's StartInstanceLocation.
  const float4x4 instanceModel = InstanceMatrix(InstanceTransforms[MaterialIndices.w + instanceID]);
  const float4 worldPosition = mul(instanceModel, mul(skin, float4(input.position, 1.0f)));
  const float3x3 linearTransform = mul((float3x3)instanceModel, (float3x3)skin);
  // Cofactor matrix is inverse-transpose up to a scalar determinant. The
  // following normalization removes that scalar and works on Shader Model 5.1,
  // where a matrix inverse intrinsic is unavailable.
  const float3x3 normalTransform = float3x3(
    cross(linearTransform[1], linearTransform[2]),
    cross(linearTransform[2], linearTransform[0]),
    cross(linearTransform[0], linearTransform[1]));
  const float orientation = determinant(linearTransform) < 0.0f ? -1.0f : 1.0f;
  const float3 normal = normalize(mul(normalTransform, input.normal) * orientation);
  float3 tangent = mul(linearTransform, input.tangent);
  tangent = length(tangent) > 0.00001f ? normalize(tangent - normal * dot(normal, tangent))
    : normalize(abs(normal.y) < 0.999f ? cross(float3(0, 1, 0), normal) : cross(float3(1, 0, 0), normal));
  const float handedness = dot(cross(input.normal, input.tangent), input.bitangent) < 0.0f ? -1.0f : 1.0f;
  PSInput output;
  float4 clipPosition = mul(ViewProjection, worldPosition);
  const float2 outputResolution = max(Resolution.xy, float2(1.0f, 1.0f));
  const float virtualHeight = max(Resolution.z, 1.0f);
  const float2 snapResolution = float2(
    max(floor(virtualHeight * outputResolution.x / outputResolution.y + 0.5f), 1.0f),
    virtualHeight);
  const float safeW = abs(clipPosition.w) > 0.00001f ? clipPosition.w : 0.00001f;
  if (Resolution.w > 0.5f)
  {
    const float2 ndc = clipPosition.xy / safeW;
    const float2 snappedNdc =
      (floor((ndc * 0.5f + 0.5f) * snapResolution + 0.5f) / snapResolution) * 2.0f - 1.0f;
    clipPosition.xy = snappedNdc * clipPosition.w;
  }
  output.position = clipPosition;
  output.worldPosition = worldPosition.xyz;
  output.normal = normal;
  output.tangent = tangent;
  output.bitangent = normalize(cross(normal, tangent)) * handedness;
  output.uv = input.uv;
  output.shadowPosition = mul(LightViewProjection, worldPosition);
  return output;
}

PSInput ShadowVS(VSInput input, uint instanceID : SV_InstanceID)
{
  PSInput output = VSMain(input, instanceID);
  output.position = mul(ShadowViewProjection, float4(output.worldPosition, 1));
  return output;
}

float PointShadowPS(PSInput input) : SV_TARGET
{
  return length(input.worldPosition - ShadowLightPosition.xyz);
}

float ShadowFactor(float4 shadowPosition, float3 normal, float3 lightDirection)
{
  if (LightCount.y == 0) return 1.0f;
  if (shadowPosition.w <= 0.0f) return 1.0f;
  const float3 ndc = shadowPosition.xyz / shadowPosition.w;
  const float2 uv = ndc.xy * float2(0.5f, -0.5f) + 0.5f;
  if (ndc.z <= 0.0f || ndc.z >= 1.0f || any(uv < 0.0f) || any(uv > 1.0f)) return 1.0f;
  uint width, height;
  ShadowTexture.GetDimensions(width, height);
  const float2 texel = 1.0f / float2(width, height);
  const float bias = max(0.0015f * (1.0f - saturate(dot(normal, -lightDirection))), 0.00015f);
  float visibility = 0.0f;
  // Anchor to shadow texels. Flip the pattern's Y coordinate with the DX viewport.
  const uint2 pattern = uint2(floor(float2(uv.x, 1.0f - uv.y) * float2(width, height))) % 16;
  const uint samples = uint(ShadowFilter.x);
  const uint offsetBase = (pattern.y * 16 + pattern.x) * samples;
  for (uint i = 0; i < samples; ++i)
  {
    const float2 offset = ShadowOffsets[offsetBase + i] * ShadowFilter.y * float2(1, -1);
    const float depth = ShadowTexture.SampleLevel(ShadowSampler, uv + offset * texel, 0).r;
    visibility += ndc.z - bias <= depth ? 1.0f : 0.15f;
  }
  return visibility / max(float(samples), 1.0f);
}

float PointShadowFactor(float3 worldPosition, float3 normal, float3 lightPosition,
                        float shadowSlotPlusOne)
{
  if (LightCount.y == 0 || shadowSlotPlusOne < 0.5f) return 1.0f;
  const float3 fragToLight = worldPosition - lightPosition;
  const float currentDepth = length(fragToLight);
  if (currentDepth >= 20.0f || currentDepth <= 0.0001f) return 1.0f;
  const float3 direction = fragToLight / currentDepth;
  const float bias = max(0.03f * (1.0f - dot(normal, -direction)), 0.003f);
  uint width, height, elements;
  PointShadowTexture.GetDimensions(width, height, elements);
  const float filterRadius = (2.0f / max(float(width), 1.0f)) * (1.0f + 2.0f * currentDepth / 20.0f);
  const float3 offsets[4] = {float3(1, 1, 1), float3(-1, -1, 1), float3(-1, 1, -1), float3(1, -1, -1)};
  float occluded = 0;
  for (uint i = 0; i < 4; ++i)
  {
    const float depth = PointShadowTexture.SampleLevel(ShadowSampler,
      float4(direction + offsets[i] * filterRadius, shadowSlotPlusOne - 1.0f), 0).r;
    occluded += currentDepth - bias > depth ? 1.0f : 0.0f;
  }
  return lerp(1.0f, 0.15f, occluded * 0.25f);
}

struct GBufferOutput
{
  float4 position : SV_TARGET0;
  float4 normal : SV_TARGET1;
  float4 albedoSpec : SV_TARGET2;
};

GBufferOutput GBufferPS(PSInput input)
{
  const float4 albedoSample = MaterialTextures[MaterialIndices.x].Sample(MaterialSampler, input.uv);
  float3 normal = normalize(input.normal);
  if (MaterialFlags.x > 0.5f)
  {
    normal = GabglApplyTangentNormal(
      normal, input.tangent, input.bitangent,
      MaterialTextures[MaterialIndices.y].Sample(MaterialSampler, input.uv).xyz);
  }
  const float specularStrength = MaterialFlags.y > 0.5f
    ? MaterialTextures[MaterialIndices.z].Sample(MaterialSampler, input.uv).r : 0.0f;

  GBufferOutput output;
  output.position = float4(input.worldPosition, max(MaterialFlags.z, 1.0f));
  output.normal = float4(normal, 0.0f);
  output.albedoSpec = float4(albedoSample.rgb, specularStrength);
  return output;
}

struct LightingInput
{
  float4 position : SV_POSITION;
  float2 uv : TEXCOORD;
};

LightingInput LightingVS(uint vertexID : SV_VertexID)
{
  LightingInput output;
  const float2 position = float2((vertexID << 1) & 2, vertexID & 2);
  output.uv = position;
  output.position = float4(
    position * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f);
  return output;
}

float4 LightingPS(LightingInput input) : SV_TARGET
{
  const float4 positionData = DiffuseTexture.Sample(ShadowSampler, input.uv);
  if (positionData.a <= 0.0f)
    return float4(0.0f, 0.0f, 0.0f, 1.0f);
  const float3 worldPosition = positionData.xyz;
  const float3 normal = normalize(NormalTexture.Sample(ShadowSampler, input.uv).xyz);
  const float4 albedoSpec = SpecularTexture.Sample(ShadowSampler, input.uv);
  const float3 viewDirection = normalize(CameraPosition.xyz - worldPosition);
  const float shininess = GabglMaterialShininess(albedoSpec.a);
  const float4 shadowPosition = mul(LightViewProjection, float4(worldPosition, 1.0f));
  float3 lighting = 0.0f;
  const uint2 tile = uint2(input.position.xy) / 16;
  const uint2 tileEntry = TileGrid[tile.y * LightCount.z + tile.x];
  for (uint lightOffset = 0; lightOffset < tileEntry.y; ++lightOffset)
  {
    const uint i = TileIndices[tileEntry.x + lightOffset];
    const int type = (int)(LightTypes[i].x + 0.5f);
    const float3 lightColor = LightColors[i].rgb;
    float3 toLight;
    float attenuation = 1.0f;
    float intensity = 1.0f;
    float shadow = 1.0f;
    if (type == 0)
    {
      const float3 direction = normalize(LightDirections[i].xyz);
      toLight = -direction;
      shadow = ShadowFactor(shadowPosition, normal, direction);
      lighting += GabglEvaluateLight(
        lightColor, albedoSpec.rgb, albedoSpec.aaa, normal, viewDirection,
        toLight, shininess, 1.0f, 1.0f, shadow);
    }
    else if (type == 1)
    {
      const float3 offset = LightPositions[i].xyz - worldPosition;
      const float distanceToLight = max(length(offset), 0.001f);
      toLight = offset / distanceToLight;
      attenuation = GabglDistanceAttenuation(distanceToLight) * (1.0f - smoothstep(16.0f, 20.0f, distanceToLight));
      shadow = PointShadowFactor(worldPosition, normal, LightPositions[i].xyz, LightTypes[i].y);
      lighting += GabglEvaluateLight(
        lightColor, albedoSpec.rgb, albedoSpec.aaa, normal, viewDirection,
        toLight, shininess, attenuation, 1.0f, shadow);
    }
    else
    {
      const float3 offset = LightPositions[i].xyz - worldPosition;
      const float distanceToLight = max(length(offset), 0.001f);
      toLight = offset / distanceToLight;
      attenuation = GabglDistanceAttenuation(distanceToLight) * (1.0f - smoothstep(16.0f, 20.0f, distanceToLight));
      intensity = GabglSpotIntensity(-toLight, LightDirections[i].xyz);
      lighting += GabglEvaluateLight(
        lightColor, albedoSpec.rgb, albedoSpec.aaa, normal, viewDirection,
        toLight, shininess, attenuation, intensity, 1.0f);
    }
  }
  lighting *= max(positionData.a, 1.0f);
  if (LightCount.w != 0)
  {
    const float density = LightCount.x > 0 ? saturate(float(tileEntry.y) / LightCount.x) : 0.0f;
    float3 heat = saturate(float3(1.5f - abs(4.0f * density - 3.0f),
      1.5f - abs(4.0f * density - 2.0f), 1.5f - abs(4.0f * density - 1.0f)));
    const bool border = (uint(input.position.x) % 16 == 0) || (uint(input.position.y) % 16 == 0);
    if (border) heat = 0.02f;
    lighting = LightCount.w == 1 ? lerp(lighting, heat, border ? 0.85f : 0.55f) : heat;
  }
  return float4(lighting, 1.0f);
}
