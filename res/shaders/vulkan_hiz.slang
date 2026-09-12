#api VULKAN

[[vk::binding(0, 0)]] Texture2D<float> SourceTexture;
[[vk::binding(1, 0)]] SamplerState SourceSampler;
[[vk::binding(2, 0)]] RWTexture2D<float> Destination;

struct HiZPushConstants
{
  uint4 Params; // width, height, source mip, copy depth
};

[[vk::push_constant]] ConstantBuffer<HiZPushConstants> HiZ;

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
  if (id.x >= HiZ.Params.x || id.y >= HiZ.Params.y) return;
  if (HiZ.Params.w != 0)
  {
    Destination[id.xy] = SourceTexture.Load(int3(id.xy, 0));
    return;
  }
  uint sourceWidth;
  uint sourceHeight;
  uint sourceLevels;
  SourceTexture.GetDimensions(HiZ.Params.z, sourceWidth, sourceHeight, sourceLevels);
  const uint2 sourceBase = id.xy * 2;
  float maximumDepth = 0.0f;
  for (uint y = 0; y < 2; ++y)
    for (uint x = 0; x < 2; ++x)
      maximumDepth = max(maximumDepth, SourceTexture.Load(int3(
        min(sourceBase + uint2(x, y), uint2(sourceWidth - 1, sourceHeight - 1)),
        HiZ.Params.z)));
  Destination[id.xy] = maximumDepth;
}
