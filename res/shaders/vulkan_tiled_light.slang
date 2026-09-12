#api VULKAN

struct SceneFrameData
{
  float4x4 LightViewProjection;
  float4x4 InverseViewProjection;
  float4 CameraPosition;
  float4 LightPositions[32];
  float4 LightDirections[32];
  float4 LightColors[32];
  float4 Params;
  float4 Effects;
  float4 ShadowInfo;
  float4 PointShadowIndices;
  float4 TileInfo;
  float4 ViewportInfo;
};

[[vk::binding(0, 0)]] Texture2D<float4> PositionTexture;
[[vk::binding(1, 0)]] SamplerState PositionSampler;
[[vk::binding(2, 0)]] ConstantBuffer<SceneFrameData> Frame;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint2> TileGrid;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> TileIndices;

groupshared float4 MinimumPositions[256];
groupshared float3 MaximumPositions[256];
groupshared uint SharedLightCount;
groupshared uint SharedLightIndices[32];

[numthreads(16, 16, 1)]
void CSMain(uint3 dispatchID : SV_DispatchThreadID,
            uint3 groupID : SV_GroupID,
            uint lane : SV_GroupIndex)
{
  uint width;
  uint height;
  PositionTexture.GetDimensions(width, height);
  const bool inside = dispatchID.x < width && dispatchID.y < height;
  const float4 position = inside
    ? PositionTexture.Load(int3(dispatchID.xy, 0)) : float4(0.0f);
  MinimumPositions[lane] = position.w > 0.0f
    ? float4(position.xyz, 1.0f)
    : float4(3.402823e38f, 3.402823e38f, 3.402823e38f, 0.0f);
  MaximumPositions[lane] = position.w > 0.0f
    ? position.xyz : float3(-3.402823e38f, -3.402823e38f, -3.402823e38f);
  if (lane == 0) SharedLightCount = 0;
  GroupMemoryBarrierWithGroupSync();

  for (uint stride = 128; stride > 0; stride >>= 1)
  {
    if (lane < stride)
    {
      MinimumPositions[lane].xyz = min(MinimumPositions[lane].xyz,
                                        MinimumPositions[lane + stride].xyz);
      MinimumPositions[lane].w = max(MinimumPositions[lane].w,
                                      MinimumPositions[lane + stride].w);
      MaximumPositions[lane] = max(MaximumPositions[lane],
                                    MaximumPositions[lane + stride]);
    }
    GroupMemoryBarrierWithGroupSync();
  }

  const uint lightCount = min((uint)Frame.Params.x, 32u);
  if (MinimumPositions[0].w > 0.0f && lane < lightCount)
  {
    const int type = (int)Frame.LightPositions[lane].w;
    const float3 closest = clamp(Frame.LightPositions[lane].xyz,
                                  MinimumPositions[0].xyz,
                                  MaximumPositions[0]);
    const float3 offset = Frame.LightPositions[lane].xyz - closest;
    if (type == 0 || dot(offset, offset) <= 400.0f)
    {
      uint slot;
      InterlockedAdd(SharedLightCount, 1, slot);
      if (slot < 32) SharedLightIndices[slot] = lane;
    }
  }
  GroupMemoryBarrierWithGroupSync();

  const uint tileIndex = groupID.y * (uint)Frame.TileInfo.x + groupID.x;
  const uint storedCount = min(SharedLightCount, 32u);
  if (lane < storedCount)
    TileIndices[tileIndex * 32 + lane] = SharedLightIndices[lane];
  if (lane == 0)
    TileGrid[tileIndex] = uint2(tileIndex * 32, storedCount);
}
