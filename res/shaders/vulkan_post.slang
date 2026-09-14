#api VULKAN

[[vk::binding(0, 0)]] Texture2D<float4> SceneTexture;
[[vk::binding(1, 0)]] SamplerState LinearSampler;

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

#define GABGL_FLOAT2 float2
#define GABGL_FLOAT3 float3
#define GABGL_INT2 int2
#define GABGL_SATURATE(value) saturate(value)
#include "common/ps1_postprocess.slang"

VertexOutput VSMain(float2 position : POSITION)
{
  VertexOutput output;
  output.Position = float4(position, 0.0f, 1.0f);
  output.TexCoords = position * 0.5f + 0.5f;
  return output;
}

float4 PSMain(VertexOutput input) : SV_Target
{
  float2 uv = input.TexCoords;
  float2 virtualPixel = 0.0f;
  if (Post.Effects.x > 0.5f)
    uv = GabglPS1SampleUV(
      input.TexCoords, Post.Resolution.xy, Post.Effects.z, virtualPixel);
  float3 color = SceneTexture.Sample(LinearSampler, uv).rgb;
  if (Post.Effects.x > 0.5f)
    color = GabglPS1Quantize(color, virtualPixel, Post.Effects.y);
  return float4(color, 1.0f);
}
