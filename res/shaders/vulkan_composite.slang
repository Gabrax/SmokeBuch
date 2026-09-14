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

float3 ToneMappingACES(float3 color)
{
  const float a = 2.51f;
  const float b = 0.03f;
  const float c = 2.43f;
  const float d = 0.59f;
  const float e = 0.14f;
  return saturate((color * (a * color + b)) /
                  (color * (c * color + d) + e));
}

float4 PSMain(VertexOutput input) : SV_Target
{
  float3 color = SceneTexture.Sample(LinearSampler, input.TexCoords).rgb;
  if (Post.Effects.w > 0.5f && Post.Params.x > 0.0f)
    color += BloomPyramid.SampleLevel(
      LinearSampler, input.TexCoords, 0.0f).rgb * Post.Params.x;
  color = ToneMappingACES(color * Post.Params.z);
  color = pow(max(color, 0.0f), 1.0f / max(Post.Params.w, 0.001f));
  return float4(color, 1.0f);
}
