#define MAX_BONES 100
#define MAX_LIGHTS 128

cbuffer SceneConstants : register(b0)
{
  float4x4 ViewProjection;
  float4x4 Model;
  float4x4 LightViewProjection;
  float4x4 Bones[MAX_BONES];
  float4 CameraPosition;
  float4 MaterialFlags;
  float4 LightPositions[MAX_LIGHTS];
  float4 LightDirections[MAX_LIGHTS];
  float4 LightColors[MAX_LIGHTS];
  float4 LightTypes[MAX_LIGHTS];
  uint4 LightCount;
  float4 Resolution;
  uint4 MaterialIndices; // xyz: material textures, w: compacted instance base
  float4 ShadowFilter;
};

