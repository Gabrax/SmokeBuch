#api VULKAN

struct ScenePushConstants
{
  float4x4 ModelViewProjection;
  float4 DrawParams;
};

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

[[vk::push_constant]] ConstantBuffer<ScenePushConstants> Scene;
[[vk::binding(4, 0)]] StructuredBuffer<float4x4> BoneMatrices;
[[vk::binding(5, 0)]] ConstantBuffer<SceneFrameData> Frame;
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

float4 VSMain(VertexInput input) : SV_Position
{
  const float4x4 model = VisibleTransforms[(uint)Scene.DrawParams.y + input.InstanceID];
  const float4x4 skin = GabglSkinMatrix(input.BoneIDs, input.Weights, 0, true);
  float4 position = mul(Scene.ModelViewProjection,
                        mul(model, mul(skin, float4(input.Position, 1.0f))));
  if (Frame.Params.z > 0.5f && position.w > 0.00001f)
  {
    const float2 outputResolution = max(Frame.ViewportInfo.xy, float2(1.0f));
    const float virtualHeight = max(Frame.ViewportInfo.w, 1.0f);
    const float2 virtualResolution = float2(
      max(floor(virtualHeight * outputResolution.x / outputResolution.y + 0.5f), 1.0f),
      virtualHeight);
    float2 ndc = position.xy / position.w;
    ndc = (floor((ndc * 0.5f + 0.5f) * virtualResolution + 0.5f) /
           virtualResolution) * 2.0f - 1.0f;
    position.xy = ndc * position.w;
  }
  return position;
}
