#api OPENGL
#type COMPUTE
#version 460 core

struct DrawCullData
{
  vec4 localSphere;
  uvec4 metadata;
};

layout(std430, binding = 14) readonly buffer SourceCommands
{
  uint sourceCommandWords[];
};
layout(std430, binding = 15) readonly buffer CullData
{
  DrawCullData drawCullData[];
};
layout(std430, binding = 16) readonly buffer SourceTransforms
{
  mat4 sourceTransforms[];
};
layout(std430, binding = 17) writeonly buffer VisibleTransforms
{
  mat4 visibleTransforms[];
};
layout(std430, binding = 18) writeonly buffer OutputCommands
{
  uint outputCommandWords[];
};
layout(std430, binding = 19) writeonly buffer DrawRemap
{
  uint outputDrawIDs[];
};
layout(std430, binding = 20) buffer OutputCount
{
  uint outputDrawCount;
};

uniform mat4 u_ViewProjection;
uniform bool u_UseHiZ;
uniform int u_HiZMaxMip;
uniform vec2 u_HiZResolution;
uniform sampler2D u_HiZTexture;

vec4 MatrixRow(mat4 matrixValue, int row)
{
  return vec4(matrixValue[0][row], matrixValue[1][row],
              matrixValue[2][row], matrixValue[3][row]);
}

bool SphereInFrustum(vec3 center, float radius)
{
  vec4 row0 = MatrixRow(u_ViewProjection, 0);
  vec4 row1 = MatrixRow(u_ViewProjection, 1);
  vec4 row2 = MatrixRow(u_ViewProjection, 2);
  vec4 row3 = MatrixRow(u_ViewProjection, 3);
  vec4 planes[6] = vec4[6](
    row3 + row0, row3 - row0,
    row3 + row1, row3 - row1,
    row3 + row2, row3 - row2);
  for (int planeIndex = 0; planeIndex < 6; ++planeIndex)
  {
    float normalLength = length(planes[planeIndex].xyz);
    if (normalLength > 0.00001 &&
        dot(planes[planeIndex].xyz, center) + planes[planeIndex].w < -radius * normalLength)
      return false;
  }
  return true;
}

bool SphereOccluded(vec3 center, float radius)
{
  vec2 minimumNdc = vec2(1.0);
  vec2 maximumNdc = vec2(-1.0);
  float nearestDepth = 1.0;
  for (uint corner = 0; corner < 8; ++corner)
  {
    vec3 signValue = vec3(
      (corner & 1u) == 0u ? -1.0 : 1.0,
      (corner & 2u) == 0u ? -1.0 : 1.0,
      (corner & 4u) == 0u ? -1.0 : 1.0);
    vec4 clip = u_ViewProjection * vec4(center + signValue * radius, 1.0);
    if (clip.w <= 0.00001)
      return false;
    vec3 ndc = clip.xyz / clip.w;
    minimumNdc = min(minimumNdc, ndc.xy);
    maximumNdc = max(maximumNdc, ndc.xy);
    nearestDepth = min(nearestDepth, ndc.z * 0.5 + 0.5);
  }

  vec2 minimumUV = clamp(minimumNdc * 0.5 + 0.5, vec2(0.0), vec2(1.0));
  vec2 maximumUV = clamp(maximumNdc * 0.5 + 0.5, vec2(0.0), vec2(1.0));
  vec2 extentPixels = max((maximumUV - minimumUV) * u_HiZResolution, vec2(1.0));
  if (nearestDepth <= 0.0 || max(extentPixels.x, extentPixels.y) < 2.0)
    return false;

  float mip = clamp(floor(log2(max(extentPixels.x, extentPixels.y))),
                    0.0, float(u_HiZMaxMip));
  float maximumDepth = max(
    max(textureLod(u_HiZTexture, minimumUV, mip).r,
        textureLod(u_HiZTexture, vec2(maximumUV.x, minimumUV.y), mip).r),
    max(textureLod(u_HiZTexture, vec2(minimumUV.x, maximumUV.y), mip).r,
        textureLod(u_HiZTexture, maximumUV, mip).r));
  return nearestDepth > maximumDepth + 0.0025;
}

[numthreads(64, 1, 1)]
void main()
{
  uint sourceDrawID = gl_GlobalInvocationID.x;
  if (sourceDrawID >= uint(sourceCommandWords.length()) / 5u)
    return;

  // DrawElementsIndirectCommand is exactly five tightly packed 32-bit words
  // in OpenGL. A struct array may acquire a target-specific std430 stride
  // when emitted by Slang, so address the command buffer explicitly.
  uint sourceWord = sourceDrawID * 5u;
  uint indexCount = sourceCommandWords[sourceWord + 0u];
  uint sourceInstanceCount = sourceCommandWords[sourceWord + 1u];
  uint firstIndex = sourceCommandWords[sourceWord + 2u];
  uint baseVertexBits = sourceCommandWords[sourceWord + 3u];
  uint sourceBaseInstance = sourceCommandWords[sourceWord + 4u];
  if (sourceInstanceCount == 0u)
    return;

  DrawCullData cullData = drawCullData[sourceDrawID];
  uint visibleCount = 0u;
  for (uint instance = 0u; instance < sourceInstanceCount; ++instance)
  {
    mat4 transform = sourceTransforms[sourceBaseInstance + instance];
    vec3 center = (transform * vec4(cullData.localSphere.xyz, 1.0)).xyz;
    float radiusScale = max(
      max(length(transform[0].xyz), length(transform[1].xyz)),
      length(transform[2].xyz));
    float radius = cullData.localSphere.w * radiusScale;
    if (!SphereInFrustum(center, radius))
      continue;
    if (u_UseHiZ && cullData.metadata.z != 0u && SphereOccluded(center, radius))
      continue;
    visibleTransforms[cullData.metadata.x + visibleCount] = transform;
    ++visibleCount;
  }

  if (visibleCount == 0u)
    return;

  uint compactDrawID = atomicAdd(outputDrawCount, 1u);
  uint outputWord = compactDrawID * 5u;
  outputCommandWords[outputWord + 0u] = indexCount;
  outputCommandWords[outputWord + 1u] = visibleCount;
  outputCommandWords[outputWord + 2u] = firstIndex;
  outputCommandWords[outputWord + 3u] = baseVertexBits;
  outputCommandWords[outputWord + 4u] = cullData.metadata.x;
  outputDrawIDs[compactDrawID] = cullData.metadata.y;
}

#api DX12
#include "common/dx12_instance.slang"
cbuffer CullConstants : register(b0)
{
  float4x4 CullViewProjection;
  uint4 CullParameters; // draw count, use Hi-Z, width, height
  float4 SnapMargin;
};
struct DrawCullData { float4 sphere; uint4 instances; };
ByteAddressBuffer SourceCommands : register(t0);
StructuredBuffer<InstanceTransform> SourceTransforms : register(t1);
StructuredBuffer<DrawCullData> CullData : register(t2);
Texture2D<float> HiZ : register(t3);
RWByteAddressBuffer OutputCommands : register(u0);
RWStructuredBuffer<InstanceTransform> VisibleTransforms : register(u1);
RWByteAddressBuffer OutputCount : register(u2);

bool SphereVisible(float3 center, float radius)
{
  const float4 r0 = CullViewProjection[0], r1 = CullViewProjection[1];
  const float4 r2 = CullViewProjection[2], r3 = CullViewProjection[3];
  const float4 planes[6] = {r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2};
  for (uint i = 0; i < 6; ++i)
    if (dot(planes[i], float4(center, 1)) < -radius * length(planes[i].xyz)) return false;
  return true;
}

bool SphereOccluded(float3 center, float radius)
{
  float2 minUV = 1, maxUV = 0;
  float nearestDepth = 1;
  for (uint corner = 0; corner < 8; ++corner)
  {
    const float3 signValue = float3((corner & 1) != 0 ? 1 : -1,
      (corner & 2) != 0 ? 1 : -1, (corner & 4) != 0 ? 1 : -1);
    const float4 clip = mul(CullViewProjection, float4(center + signValue * radius, 1));
    if (clip.w <= 0.00001f || clip.z <= 0) return false;
    const float3 ndc = clip.xyz / clip.w;
    const float2 uv = ndc.xy * float2(0.5f, -0.5f) + 0.5f;
    minUV = min(minUV, uv);
    maxUV = max(maxUV, uv);
    nearestDepth = min(nearestDepth, ndc.z);
  }
  minUV = saturate(minUV - SnapMargin.xy);
  maxUV = saturate(maxUV + SnapMargin.xy);
  const float2 extent = (maxUV - minUV) * float2(CullParameters.zw);
  if (max(extent.x, extent.y) < 2) return false;
  uint width, height, levels;
  HiZ.GetDimensions(0, width, height, levels);
  const uint mip = min(uint(ceil(log2(max(max(extent.x, extent.y), 1.0f)))), levels - 1);
  HiZ.GetDimensions(mip, width, height, levels);
  const uint2 lo = min(uint2(minUV * float2(width, height)), uint2(width - 1, height - 1));
  const uint2 hi = min(uint2(maxUV * float2(width, height)), uint2(width - 1, height - 1));
  float maximumDepth = 0;
  // Cover every overlapping texel, including boundaries of non-power-of-two mips.
  for (uint y = lo.y; y <= hi.y; ++y)
    for (uint x = lo.x; x <= hi.x; ++x)
      maximumDepth = max(maximumDepth, HiZ.Load(int3(x, y, mip)));
  return nearestDepth > maximumDepth + 0.0025f;
}

[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
  if (id.x >= CullParameters.x) return;
  const DrawCullData data = CullData[id.x];
  if (CullParameters.y == 2 && (data.instances.w & 4) != 0) return;
  uint visibleCount = 0;
  for (uint i = 0; i < data.instances.y; ++i)
  {
    const InstanceTransform instanceTransform = SourceTransforms[data.instances.x + i];
    const float4x4 transform = InstanceMatrix(instanceTransform);
    const float3 center = mul(transform, float4(data.sphere.xyz, 1)).xyz;
    const float3 scales = float3(
      length(float3(transform[0][0], transform[1][0], transform[2][0])),
      length(float3(transform[0][1], transform[1][1], transform[2][1])),
      length(float3(transform[0][2], transform[1][2], transform[2][2])));
    const float radius = data.sphere.w * max(max(scales.x, scales.y), scales.z);
    if (!SphereVisible(center, radius)) continue;
    if (CullParameters.y == 1 && (data.instances.w & 1) != 0 && SphereOccluded(center, radius)) continue;
    VisibleTransforms[data.instances.z + visibleCount++] = instanceTransform;
  }
  if (visibleCount == 0) return;
  uint destination;
  if ((data.instances.w & 2) != 0) OutputCount.InterlockedAdd(4, visibleCount, destination);
  OutputCount.InterlockedAdd(0, 1, destination);
  for (uint word = 0; word < 16; ++word)
    OutputCommands.Store(destination * 64 + word * 4, SourceCommands.Load(id.x * 64 + word * 4));
  OutputCommands.Store(destination * 64 + 44, visibleCount);
}
