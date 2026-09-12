#api VULKAN

struct CullInput
{
  float4 LocalSphere;
  uint4 Metadata;
};

struct DrawIndexedIndirectCommand
{
  uint IndexCount;
  uint InstanceCount;
  uint FirstIndex;
  int VertexOffset;
  uint FirstInstance;
};

struct CullPushConstants
{
  float4x4 ViewProjection;
  uint4 Params;
};

[[vk::push_constant]] ConstantBuffer<CullPushConstants> Cull;
[[vk::binding(0, 0)]] StructuredBuffer<float4x4> SourceTransforms;
[[vk::binding(1, 0)]] StructuredBuffer<CullInput> Inputs;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4x4> VisibleTransforms;
[[vk::binding(3, 0)]] RWStructuredBuffer<DrawIndexedIndirectCommand> Commands;
[[vk::binding(4, 0)]] Texture2D<float> HiZTexture;
[[vk::binding(5, 0)]] SamplerState HiZSampler;

float4 MatrixRow(float4x4 matrixValue, int row)
{
  return float4(matrixValue[0][row], matrixValue[1][row],
                matrixValue[2][row], matrixValue[3][row]);
}

bool SphereVisible(float3 center, float radius)
{
  const float4 row0 = MatrixRow(Cull.ViewProjection, 0);
  const float4 row1 = MatrixRow(Cull.ViewProjection, 1);
  const float4 row2 = MatrixRow(Cull.ViewProjection, 2);
  const float4 row3 = MatrixRow(Cull.ViewProjection, 3);
  const float4 planes[6] = {
    row3 + row0, row3 - row0, row3 + row1,
    row3 - row1, row3 + row2, row3 - row2};
  for (int planeIndex = 0; planeIndex < 6; ++planeIndex)
    if (dot(planes[planeIndex], float4(center, 1.0f)) <
        -radius * length(planes[planeIndex].xyz)) return false;
  return true;
}

bool SphereOccluded(float3 center, float radius)
{
  float2 minimumUV = 1.0f;
  float2 maximumUV = 0.0f;
  float nearestDepth = 1.0f;
  for (uint corner = 0; corner < 8; ++corner)
  {
    const float3 signValue = float3(
      (corner & 1) != 0 ? 1.0f : -1.0f,
      (corner & 2) != 0 ? 1.0f : -1.0f,
      (corner & 4) != 0 ? 1.0f : -1.0f);
    const float4 clip = mul(Cull.ViewProjection,
                            float4(center + signValue * radius, 1.0f));
    if (clip.w <= 0.00001f) return false;
    const float3 ndc = clip.xyz / clip.w;
    const float2 uv = ndc.xy * 0.5f + 0.5f;
    minimumUV = min(minimumUV, uv);
    maximumUV = max(maximumUV, uv);
    nearestDepth = min(nearestDepth, ndc.z);
  }
  minimumUV = saturate(minimumUV - 0.002f);
  maximumUV = saturate(maximumUV + 0.002f);
  const float2 extent = (maximumUV - minimumUV) * float2(Cull.Params.zw);
  if (nearestDepth <= 0.0f || max(extent.x, extent.y) < 2.0f) return false;
  const float mip = clamp(ceil(log2(max(max(extent.x, extent.y), 1.0f))),
                          0.0f, 15.0f);
  const float maximumDepth = max(
    max(HiZTexture.SampleLevel(HiZSampler, minimumUV, mip),
        HiZTexture.SampleLevel(HiZSampler, float2(maximumUV.x, minimumUV.y), mip)),
    max(HiZTexture.SampleLevel(HiZSampler, float2(minimumUV.x, maximumUV.y), mip),
        HiZTexture.SampleLevel(HiZSampler, maximumUV, mip)));
  return nearestDepth > maximumDepth + 0.0025f;
}

[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
  const uint drawIndex = id.x;
  if (drawIndex >= Cull.Params.x) return;
  const CullInput input = Inputs[drawIndex];
  const bool preview = (input.Metadata.w & 0x80000000u) != 0;
  uint visibleCount = 0;
  for (uint instance = 0; instance < input.Metadata.y; ++instance)
  {
    const float4x4 transform = SourceTransforms[input.Metadata.x + instance];
    const float3 center = mul(transform, float4(input.LocalSphere.xyz, 1.0f)).xyz;
    const float radiusScale = max(max(length(transform[0].xyz), length(transform[1].xyz)),
                                  length(transform[2].xyz));
    const float radius = input.LocalSphere.w * radiusScale;
    if (!SphereVisible(center, radius)) continue;
    if (Cull.Params.y != 0 && !preview && SphereOccluded(center, radius)) continue;
    VisibleTransforms[input.Metadata.z + visibleCount] = transform;
    ++visibleCount;
  }
  DrawIndexedIndirectCommand command;
  command.IndexCount = input.Metadata.w & 0x7fffffffu;
  command.InstanceCount = visibleCount;
  command.FirstIndex = 0;
  command.VertexOffset = 0;
  // SV_InstanceID is local to the draw on HLSL-style shader paths. The compacted
  // transform base is passed explicitly in the draw push constants.
  command.FirstInstance = 0;
  Commands[drawIndex] = command;
}
