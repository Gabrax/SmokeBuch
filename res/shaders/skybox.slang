#api OPENGL
#type VERTEX
#version 450 core
layout (location = 0) in vec3 aPos;

out vec3 TexCoords;

layout(std140, binding = 0) uniform Camera
{
  mat4 ViewProjection;
	mat4 OrtoProjection;
	mat4 NonRotViewProjection;
	vec3 CameraPos;
};

void main()
{
    TexCoords = aPos;
    vec4 pos = NonRotViewProjection * vec4(aPos, 1.0);
    gl_Position = pos.xyww;
}  

#type FRAGMENT
#version 330 core
layout (location = 0) out vec4 FragColor;

in vec3 TexCoords;

uniform samplerCube skybox;

vec3 toneMappingACES(vec3 color)
{
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;

  return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

void main()
{
    vec3 skyTexture = texture(skybox, TexCoords).rgb;

    FragColor = vec4(skyTexture * 0.65, 1.0);
}

#api DX12
cbuffer SkyboxConstants : register(b0) { float4x4 ViewProjection; };
TextureCube SkyboxTexture : register(t0);
SamplerState SkyboxSampler : register(s0);
struct PSInput { float4 position : SV_POSITION; float3 direction : TEXCOORD; };
PSInput VSMain(float3 position : POSITION)
{
  PSInput output;
  float4 clip = mul(ViewProjection, float4(position, 1.0f));
  output.position = clip.xyww;
  output.direction = position;
  return output;
}
float4 PSMain(PSInput input) : SV_TARGET
{
  return float4(SkyboxTexture.Sample(SkyboxSampler, input.direction).rgb * 0.65f, 1.0f);
}
