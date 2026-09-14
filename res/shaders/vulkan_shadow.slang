#api VULKAN

struct ScenePushConstants
{
  float4x4 ModelViewProjection;
  float4 DrawParams;
  float4 ShadowLightPosition;
};

[[vk::push_constant]] ConstantBuffer<ScenePushConstants> Scene;
[[vk::binding(4, 0)]] StructuredBuffer<float4x4> BoneMatrices;
[[vk::binding(8, 0)]] StructuredBuffer<float4x4> VisibleTransforms;

#define GABGL_MATRIX4 float4x4
#define GABGL_INT4 int4
#define GABGL_FLOAT4 float4
#define GABGL_BONE_MATRIX(index) BoneMatrices[index]
#include "common/skinning.slang"

struct VertexInput
{
  float3 Position : POSITION;
  int4 BoneIDs : BONEIDS;
  float4 Weights : BONEWEIGHTS;
  uint InstanceID : SV_InstanceID;
};

struct VertexOutput
{
  float4 Position : SV_Position;
  float3 WorldPosition : TEXCOORD0;
};

VertexOutput VSMain(VertexInput input)
{
  const float4x4 skin = GabglSkinMatrix(input.BoneIDs, input.Weights, 0, true);
  const float4 world = mul(VisibleTransforms[(uint)Scene.DrawParams.y + input.InstanceID],
                           mul(skin, float4(input.Position, 1.0f)));
  VertexOutput output;
  output.Position = mul(Scene.ModelViewProjection, world);
  output.WorldPosition = world.xyz;
  return output;
}

float PointPSMain(VertexOutput input) : SV_Target
{
  return length(input.WorldPosition - Scene.ShadowLightPosition.xyz);
}
