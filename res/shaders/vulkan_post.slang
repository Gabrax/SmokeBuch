#api VULKAN

[[vk::binding(0, 0)]] Texture2D<float4> SceneTexture;
[[vk::binding(1, 0)]] SamplerState LinearSampler;
[[vk::binding(2, 0)]] Texture2D<float4> BloomPyramid;

struct PostPushConstants
{
  float4 Params;
  float4 Effects;
  float4 Resolution;
};

[[vk::push_constant]] ConstantBuffer<PostPushConstants> Post;

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

float3 BrightPass(float3 color, float threshold)
{
  const float brightness = max(max(color.r, color.g), color.b);
  return color * saturate((brightness - threshold) / max(brightness, 0.0001f));
}

float4 PSMain(VertexOutput input) : SV_Target
{
  float2 uv = input.TexCoords;
  if (Post.Effects.x > 0.5f)
  {
    const float virtualHeight = max(Post.Effects.z, 1.0f);
    const float virtualWidth = virtualHeight * Post.Resolution.x / max(Post.Resolution.y, 1.0f);
    uv = (floor(uv * float2(virtualWidth, virtualHeight)) + 0.5f) /
         float2(virtualWidth, virtualHeight);
  }
  float3 color = SceneTexture.Sample(LinearSampler, uv).rgb;
  const int passCount = clamp((int)Post.Effects.w, 0, 6);
  if (passCount > 0 && Post.Params.x > 0.0f)
  {
    float3 bloom = 0.0f;
    float weight = 0.0f;
    for (int level = 0; level < 6; ++level)
    {
      if (level >= passCount) break;
      const float sampleWeight = 1.0f / (1.0f + (float)level * 0.35f);
      bloom += BrightPass(BloomPyramid.SampleLevel(
        LinearSampler, uv, (float)level).rgb, Post.Params.y) * sampleWeight;
      weight += sampleWeight;
    }
    color += bloom / max(weight, 0.0001f) * Post.Params.x * Post.Params.z;
  }
  color = color / (1.0f + color);
  if (Post.Effects.x > 0.5f)
  {
    const float levels = max(Post.Effects.y, 2.0f);
    color = floor(saturate(color) * levels + 0.5f) / levels;
  }
  color = pow(max(color, 0.0f), 1.0f / max(Post.Params.w, 0.001f));
  return float4(color, 1.0f);
}
