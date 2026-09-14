#api VULKAN

struct ScenePushConstants
{
  float4x4 ModelViewProjection;
  float4 DrawParams;
};

[[vk::push_constant]] ConstantBuffer<ScenePushConstants> Scene;
[[vk::binding(0, 0)]] Texture2D<float4> DiffuseTexture;
[[vk::binding(1, 0)]] SamplerState MaterialSampler;
[[vk::binding(2, 0)]] Texture2D<float4> NormalTexture;
[[vk::binding(3, 0)]] Texture2D<float4> SpecularTexture;
[[vk::binding(4, 0)]] StructuredBuffer<float4x4> BoneMatrices;

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

[[vk::binding(5, 0)]] ConstantBuffer<SceneFrameData> Frame;
[[vk::binding(6, 0)]] Texture2D<float> ShadowTexture;
[[vk::binding(7, 0)]] SamplerState ShadowSampler;
[[vk::binding(8, 0)]] StructuredBuffer<float4x4> VisibleTransforms;

#define GABGL_MATRIX4 float4x4
#define GABGL_INT4 int4
#define GABGL_FLOAT4 float4
#define GABGL_BONE_MATRIX(index) BoneMatrices[index]
#include "common/skinning.slang"

#define GABGL_FLOAT3 float3
#define GABGL_SATURATE(value) saturate(value)
#include "common/material_lighting.slang"

struct VertexInput
{
  float3 Position : POSITION;
  float3 Normal : NORMAL;
  float2 TexCoords : TEXCOORD;
  float3 Tangent : TANGENT;
  float3 Bitangent : BITANGENT;
  int4 BoneIDs : BONEIDS;
  float4 Weights : BONEWEIGHTS;
  uint InstanceID : SV_InstanceID;
};

struct VertexOutput
{
  float4 Position : SV_Position;
  float3 Normal : NORMAL;
  float2 TexCoords : TEXCOORD;
  float3 Tangent : TANGENT;
  float3 Bitangent : BITANGENT;
  float3 WorldPosition : POSITION0;
  nointerpolation float Brightness : COLOR1;
};

VertexOutput VSMain(VertexInput input)
{
  VertexOutput output;
  const float4x4 model = VisibleTransforms[(uint)Scene.DrawParams.y + input.InstanceID];
  const float4x4 skin = GabglSkinMatrix(input.BoneIDs, input.Weights, 0, true);
  const float3x3 modelSkin = mul((float3x3)model, (float3x3)skin);
  output.Position = mul(Scene.ModelViewProjection,
                        mul(model, mul(skin, float4(input.Position, 1.0f))));
  if (Frame.Params.z > 0.5f && output.Position.w > 0.00001f)
  {
    const float2 outputResolution = max(Frame.ViewportInfo.xy, float2(1.0f));
    const float virtualHeight = max(Frame.ViewportInfo.w, 1.0f);
    const float2 virtualResolution = float2(
      max(floor(virtualHeight * outputResolution.x / outputResolution.y + 0.5f), 1.0f),
      virtualHeight);
    float2 ndc = output.Position.xy / output.Position.w;
    ndc = (floor((ndc * 0.5f + 0.5f) * virtualResolution + 0.5f) /
           virtualResolution) * 2.0f - 1.0f;
    output.Position.xy = ndc * output.Position.w;
  }
  const float3x3 normalTransform = float3x3(
    cross(modelSkin[1], modelSkin[2]),
    cross(modelSkin[2], modelSkin[0]),
    cross(modelSkin[0], modelSkin[1]));
  const float orientation = determinant(modelSkin) < 0.0f ? -1.0f : 1.0f;
  const float3 normal = normalize(mul(normalTransform, input.Normal) * orientation);
  float3 tangent = mul(modelSkin, input.Tangent);
  tangent = length(tangent) > 0.00001f
    ? normalize(tangent - normal * dot(normal, tangent))
    : normalize(abs(normal.y) < 0.999f
      ? cross(float3(0.0f, 1.0f, 0.0f), normal)
      : cross(float3(1.0f, 0.0f, 0.0f), normal));
  const float3 transformedBitangent = mul(modelSkin, input.Bitangent);
  const float handedness = dot(cross(normal, tangent), transformedBitangent) < 0.0f ? -1.0f : 1.0f;
  output.Normal = normal;
  output.TexCoords = input.TexCoords;
  output.Tangent = tangent;
  output.Bitangent = normalize(cross(normal, tangent)) * handedness;
  output.WorldPosition = mul(model, mul(skin, float4(input.Position, 1.0f))).xyz;
  output.Brightness = max(Scene.DrawParams.x, 0.0f);
  return output;
}

struct GeometryOutput
{
  float4 Albedo : SV_Target0;
  float4 NormalSpecular : SV_Target1;
  float4 WorldPosition : SV_Target2;
};

GeometryOutput PSMain(VertexOutput input)
{
  const float3 normal = GabglApplyTangentNormal(
    normalize(input.Normal), normalize(input.Tangent), normalize(input.Bitangent),
    NormalTexture.Sample(MaterialSampler, input.TexCoords).xyz);
  const float4 baseColor = DiffuseTexture.Sample(MaterialSampler, input.TexCoords);
  const float3 specularColor = SpecularTexture.Sample(MaterialSampler, input.TexCoords).rgb;
  GeometryOutput output;
  output.Albedo = float4(baseColor.rgb, 1.0f);
  output.NormalSpecular = float4(normal, specularColor.r);
  output.WorldPosition = float4(input.WorldPosition, max(input.Brightness, 1.0f));
  return output;
}
