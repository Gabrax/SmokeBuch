#api VULKAN

struct DebugPushConstants
{
  float4x4 ViewProjection;
};

[[vk::push_constant]] ConstantBuffer<DebugPushConstants> Scene;

struct VertexInput
{
  float3 Position : POSITION;
  float4 Color : COLOR;
};

struct VertexOutput
{
  float4 Position : SV_Position;
  float4 Color : COLOR;
};

VertexOutput VSMain(VertexInput input)
{
  VertexOutput output;
  output.Position = mul(Scene.ViewProjection, float4(input.Position, 1.0f));
  output.Color = input.Color;
  return output;
}

float4 PSMain(VertexOutput input) : SV_Target
{
  return input.Color;
}
