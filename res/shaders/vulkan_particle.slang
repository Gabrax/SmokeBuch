#api VULKAN

struct ParticlePushConstants
{
  float4x4 ViewProjection;
};

[[vk::push_constant]] ConstantBuffer<ParticlePushConstants> Scene;

struct VertexInput
{
  float3 Position : POSITION;
  float4 Color : COLOR;
  float2 LocalPosition : TEXCOORD;
  float IsSquare : STYLE;
};

struct VertexOutput
{
  float4 Position : SV_Position;
  float4 Color : COLOR;
  float2 LocalPosition : TEXCOORD;
  nointerpolation float IsSquare : STYLE;
};

VertexOutput VSMain(VertexInput input)
{
  VertexOutput output;
  output.Position = mul(Scene.ViewProjection, float4(input.Position, 1.0f));
  output.Color = input.Color;
  output.LocalPosition = input.LocalPosition;
  output.IsSquare = input.IsSquare;
  return output;
}

float4 PSMain(VertexOutput input) : SV_Target
{
  const float edge = input.IsSquare > 0.5f
    ? 1.0f : 1.0f - smoothstep(0.15f, 1.0f, length(input.LocalPosition));
  const float4 color = float4(input.Color.rgb, input.Color.a * edge);
  if (color.a < 0.01f) discard;
  return color;
}
