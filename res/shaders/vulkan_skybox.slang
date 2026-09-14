#api VULKAN

[[vk::binding(0, 0)]] Texture2D<float4> GBufferAlbedo;
[[vk::binding(3, 0)]] SamplerState LinearSampler;

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
[[vk::binding(7, 0)]] TextureCube<float4> SkyboxTexture;

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

float4 PSMain(VertexOutput input) : SV_Target
{
  if (Frame.ShadowInfo.y < 0.5f ||
      GBufferAlbedo.Load(int3(uint2(input.Position.xy), 0)).a >= 0.5f)
    discard;
  const float2 clip = input.TexCoords * 2.0f - 1.0f;
  const float4 farPoint = mul(
    Frame.InverseViewProjection, float4(clip, 1.0f, 1.0f));
  const float3 direction = normalize(
    farPoint.xyz / max(abs(farPoint.w), 0.00001f) - Frame.CameraPosition.xyz);
  return float4(SkyboxTexture.Sample(LinearSampler, direction).rgb * 0.65f, 1.0f);
}
