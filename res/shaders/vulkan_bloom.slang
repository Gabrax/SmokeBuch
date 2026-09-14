#api VULKAN

[[vk::binding(0, 0)]] Texture2D<float4> SourceTexture;
[[vk::binding(1, 0)]] SamplerState LinearSampler;
[[vk::binding(2, 0)]] RWTexture2D<float4> TargetTexture;

struct BloomPushConstants
{
  float4 Params; // source width, source height, mip level, threshold/radius
};

[[vk::push_constant]] ConstantBuffer<BloomPushConstants> Bloom;

float3 ExtractTexel(float3 color)
{
  const float brightness = dot(color, float3(0.2126f, 0.7152f, 0.0722f));
  return brightness > max(Bloom.Params.w, 0.0f) ? color : 0.0f;
}

float3 SampleSource(float2 uv)
{
  if (Bloom.Params.z >= 0.5f)
    return SourceTexture.SampleLevel(LinearSampler, uv, 0).rgb;

  // GL writes a thresholded lighting MRT and DX12 performs a full-resolution
  // extraction pass before the linear downsample. Reconstruct the same
  // threshold-before-filtering result without allocating another Vulkan image.
  const int2 dimensions = max(int2(Bloom.Params.xy), int2(1, 1));
  const float2 texelPosition = uv * float2(dimensions) - 0.5f;
  const int2 base = int2(floor(texelPosition));
  const float2 weight = frac(texelPosition);
  const int2 maximum = dimensions - 1;
  const int2 p00 = clamp(base, int2(0, 0), maximum);
  const int2 p10 = clamp(base + int2(1, 0), int2(0, 0), maximum);
  const int2 p01 = clamp(base + int2(0, 1), int2(0, 0), maximum);
  const int2 p11 = clamp(base + int2(1, 1), int2(0, 0), maximum);
  const float3 c00 = ExtractTexel(SourceTexture.Load(int3(p00, 0)).rgb);
  const float3 c10 = ExtractTexel(SourceTexture.Load(int3(p10, 0)).rgb);
  const float3 c01 = ExtractTexel(SourceTexture.Load(int3(p01, 0)).rgb);
  const float3 c11 = ExtractTexel(SourceTexture.Load(int3(p11, 0)).rgb);
  return lerp(lerp(c00, c10, weight.x), lerp(c01, c11, weight.x), weight.y);
}

float KarisAverage(float3 color)
{
  return 1.0f / (1.0f + dot(
    pow(max(color, 0.0f), 1.0f / 2.2f),
    float3(0.299f, 0.587f, 0.114f)) * 0.25f);
}

[numthreads(8, 8, 1)]
void DownsampleCS(uint3 id : SV_DispatchThreadID)
{
  uint width, height;
  TargetTexture.GetDimensions(width, height);
  if (id.x >= width || id.y >= height) return;
  const float2 uv = (float2(id.xy) + 0.5f) / float2(width, height);
  const float2 texel = 1.0f / max(Bloom.Params.xy, 1.0f);
  const float3 a = SampleSource(uv + texel * float2(-2,  2));
  const float3 b = SampleSource(uv + texel * float2( 0,  2));
  const float3 c = SampleSource(uv + texel * float2( 2,  2));
  const float3 d = SampleSource(uv + texel * float2(-2,  0));
  const float3 e = SampleSource(uv);
  const float3 f = SampleSource(uv + texel * float2( 2,  0));
  const float3 g = SampleSource(uv + texel * float2(-2, -2));
  const float3 h = SampleSource(uv + texel * float2( 0, -2));
  const float3 i = SampleSource(uv + texel * float2( 2, -2));
  const float3 j = SampleSource(uv + texel * float2(-1,  1));
  const float3 k = SampleSource(uv + texel * float2( 1,  1));
  const float3 l = SampleSource(uv + texel * float2(-1, -1));
  const float3 m = SampleSource(uv + texel * float2( 1, -1));
  float3 result;
  if (Bloom.Params.z < 0.5f)
  {
    float3 groups[5] = {
      (a+b+d+e)*0.03125f, (b+c+e+f)*0.03125f,
      (d+e+g+h)*0.03125f, (e+f+h+i)*0.03125f,
      (j+k+l+m)*0.125f};
    result = 0.0f;
    for (uint group = 0; group < 5; ++group)
      result += groups[group] * KarisAverage(groups[group]);
    result = max(result, 0.0001f);
  }
  else
    result = e*0.125f + (a+c+g+i)*0.03125f +
             (b+d+f+h)*0.0625f + (j+k+l+m)*0.125f;
  TargetTexture[id.xy] = float4(result, 1.0f);
}

[numthreads(8, 8, 1)]
void UpsampleCS(uint3 id : SV_DispatchThreadID)
{
  uint width, height;
  TargetTexture.GetDimensions(width, height);
  if (id.x >= width || id.y >= height) return;
  const float2 uv = (float2(id.xy) + 0.5f) / float2(width, height);
  const float radius = Bloom.Params.w;
  float3 result = 0.0f;
  for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x)
      result += SourceTexture.SampleLevel(
        LinearSampler, uv + float2(x, y) * radius, 0).rgb *
        float((x == 0 ? 2 : 1) * (y == 0 ? 2 : 1));
  TargetTexture[id.xy] = float4(
    TargetTexture[id.xy].rgb + result / 16.0f, 1.0f);
}
