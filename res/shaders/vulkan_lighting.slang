#api VULKAN

[[vk::binding(0, 0)]] Texture2D<float4> GBufferAlbedo;
[[vk::binding(1, 0)]] Texture2D<float4> GBufferNormal;
[[vk::binding(2, 0)]] Texture2D<float4> GBufferPosition;
[[vk::binding(3, 0)]] SamplerState GBufferSampler;

struct SceneFrameData
{
  float4x4 LightViewProjection;
  float4x4 InverseViewProjection;
  float4 CameraPosition;
  float4 LightPositions[32];
  float4 LightDirections[32];
  float4 LightColors[32];
  float4 Params;
  float4 Effects;
  float4 ShadowInfo;
  float4 PointShadowIndices;
  float4 TileInfo;
  float4 ViewportInfo;
};

[[vk::binding(4, 0)]] ConstantBuffer<SceneFrameData> Frame;
[[vk::binding(5, 0)]] Texture2D<float> ShadowTexture;
[[vk::binding(6, 0)]] SamplerState ShadowSampler;
[[vk::binding(7, 0)]] TextureCube<float4> SkyboxTexture;
[[vk::binding(8, 0)]] TextureCube<float> PointShadow0;
[[vk::binding(9, 0)]] TextureCube<float> PointShadow1;
[[vk::binding(10, 0)]] TextureCube<float> PointShadow2;
[[vk::binding(11, 0)]] TextureCube<float> PointShadow3;
[[vk::binding(12, 0)]] SamplerState PointShadowSampler;
[[vk::binding(13, 0)]] StructuredBuffer<uint2> TileGrid;
[[vk::binding(14, 0)]] StructuredBuffer<uint> TileIndices;

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

float DirectionalShadow(float3 worldPosition, float3 normal, float3 toLight)
{
  if (Frame.Params.y < 0.5f) return 1.0f;
  const float4 shadowPosition = mul(Frame.LightViewProjection, float4(worldPosition, 1.0f));
  const float3 projected = shadowPosition.xyz / max(shadowPosition.w, 0.00001f);
  const float2 uv = projected.xy * 0.5f + 0.5f;
  if (any(uv < 0.0f) || any(uv > 1.0f) || projected.z < 0.0f || projected.z > 1.0f)
    return 1.0f;
  const float bias = max(0.0004f * (1.0f - dot(normal, toLight)), 0.00008f);
  float visibility = 0.0f;
  for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x)
      visibility += projected.z - bias <= ShadowTexture.Sample(
        ShadowSampler, uv + float2(x, y) * Frame.ShadowInfo.xx) ? 1.0f : 0.2f;
  return visibility / 9.0f;
}

float PointShadowDepth(int slot, float3 direction)
{
  if (slot == 0) return PointShadow0.Sample(PointShadowSampler, direction);
  if (slot == 1) return PointShadow1.Sample(PointShadowSampler, direction);
  if (slot == 2) return PointShadow2.Sample(PointShadowSampler, direction);
  return PointShadow3.Sample(PointShadowSampler, direction);
}

float PointShadow(int lightIndex, float3 worldPosition, float3 normal, float3 toLight)
{
  if (Frame.ShadowInfo.z < 0.5f) return 1.0f;
  int slot = -1;
  for (int candidate = 0; candidate < 4; ++candidate)
    if ((int)Frame.PointShadowIndices[candidate] == lightIndex) slot = candidate;
  if (slot < 0) return 1.0f;
  const float3 offset = worldPosition - Frame.LightPositions[lightIndex].xyz;
  const float distanceToLight = length(offset);
  const float majorDistance = max(max(abs(offset.x), abs(offset.y)), abs(offset.z));
  if (distanceToLight >= Frame.ShadowInfo.w || majorDistance <= 0.0001f) return 1.0f;
  const float nearPlane = 0.1f;
  const float farPlane = Frame.ShadowInfo.w;
  const float referenceDepth = farPlane / (farPlane - nearPlane) -
    farPlane * nearPlane / ((farPlane - nearPlane) * majorDistance);
  const float storedDepth = PointShadowDepth(slot, offset);
  const float bias = max(0.0025f * (1.0f - dot(normal, toLight)), 0.0008f);
  return referenceDepth - bias <= storedDepth ? 1.0f : 0.2f;
}

float4 PSMain(VertexOutput input) : SV_Target
{
  const float4 albedoSample = GBufferAlbedo.Sample(GBufferSampler, input.TexCoords);
  if (albedoSample.a < 0.5f)
  {
    if (Frame.ShadowInfo.y < 0.5f) return 0.0f;
    const float2 clip = input.TexCoords * 2.0f - 1.0f;
    const float4 farPoint = mul(Frame.InverseViewProjection, float4(clip, 1.0f, 1.0f));
    const float3 direction = normalize(farPoint.xyz / max(abs(farPoint.w), 0.00001f)
                                       - Frame.CameraPosition.xyz);
    return float4(SkyboxTexture.Sample(GBufferSampler, direction).rgb, 1.0f);
  }
  const float4 normalSpecular = GBufferNormal.Sample(GBufferSampler, input.TexCoords);
  const float3 worldPosition = GBufferPosition.Sample(GBufferSampler, input.TexCoords).xyz;
  const float3 normal = normalize(normalSpecular.xyz);
  const float3 viewDirection = normalize(Frame.CameraPosition.xyz - worldPosition);
  const float3 specularColor = normalSpecular.www;
  float3 color = albedoSample.rgb * 0.08f;
  const int lightCount = min((int)Frame.Params.x, 32);
  uint tileOffset = 0;
  int tileLightCount = lightCount;
  if (Frame.TileInfo.y > 0.5f)
  {
    const uint2 pixel = uint2(input.Position.xy);
    const uint tileIndex = (pixel.y / 16) * (uint)Frame.TileInfo.x + pixel.x / 16;
    const uint2 tile = TileGrid[tileIndex];
    tileOffset = tile.x;
    tileLightCount = min((int)tile.y, 32);
  }
  for (int tileLight = 0; tileLight < tileLightCount; ++tileLight)
  {
    const int index = Frame.TileInfo.y > 0.5f
      ? (int)TileIndices[tileOffset + tileLight] : tileLight;
    const int type = (int)Frame.LightPositions[index].w;
    float3 toLight;
    float attenuation = 1.0f;
    float intensity = 1.0f;
    if (type == 0)
      toLight = normalize(-Frame.LightDirections[index].xyz);
    else
    {
      const float3 offset = Frame.LightPositions[index].xyz - worldPosition;
      const float distanceToLight = max(length(offset), 0.0001f);
      toLight = offset / distanceToLight;
      attenuation = 1.0f / (1.0f + 0.09f * distanceToLight +
                           0.032f * distanceToLight * distanceToLight);
      if (type == 2)
      {
        const float theta = dot(normalize(-toLight), normalize(Frame.LightDirections[index].xyz));
        intensity = saturate((theta - cos(0.3054326191f)) /
                             max(cos(0.2181661565f) - cos(0.3054326191f), 0.0001f));
      }
    }
    const float shadow = type == 0
      ? DirectionalShadow(worldPosition, normal, toLight)
      : (type == 1 ? PointShadow(index, worldPosition, normal, toLight) : 1.0f);
    const float diffuse = max(dot(normal, toLight), 0.0f);
    const float specular = pow(max(dot(viewDirection, reflect(-toLight, normal)), 0.0f),
                               8.0f + 120.0f * saturate(normalSpecular.w));
    color += Frame.LightColors[index].rgb *
      (albedoSample.rgb * diffuse + specularColor * specular) *
      attenuation * intensity * shadow;
  }
  if (Frame.TileInfo.z > 0.5f)
  {
    const float occupancy = (float)tileLightCount / max((float)lightCount, 1.0f);
    const float3 heat = float3(saturate(occupancy * 2.0f),
                               saturate(1.0f - abs(occupancy * 2.0f - 1.0f)),
                               saturate(1.0f - occupancy * 2.0f));
    color = Frame.TileInfo.z > 1.5f ? heat : lerp(color, heat, 0.35f);
  }
  return float4(color, 1.0f);
}
