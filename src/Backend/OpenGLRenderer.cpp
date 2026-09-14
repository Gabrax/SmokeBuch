#include "OpenGLRenderer.h"

#include "Buffer.h"
#include "Camera.h"
#include "ModelManager.h"
#include "ParticleRenderer.h"
#include "RenderBackend.h"
#include "Shader.h"
#include "Texture.h"
#include "AudioManager.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <cstring>
#include <stb_image.h>
#include <imgui.h>
#include <imgui_internal.h>
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <gabdebug.h>
#include <gabdebug_gpu_opengl.h>
//#include "ImGuizmo.h"
#include "glm/ext/scalar_constants.hpp"
#include "glm/trigonometric.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <glm/fwd.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>
#include "SceneManager.h"
#include "Settings.h"
#include "Timer.hpp"
#include "Window.h"
#include "NativeFileDialog.h"

void MessageCallback(unsigned source,unsigned type,unsigned id,unsigned severity,int length,const char* message,const void* userParam)
{
	switch (severity)
	{
		case GL_DEBUG_SEVERITY_HIGH:         gablog_log(LOG_ERROR, __FILE__, __LINE__, "%s", message); return;
		case GL_DEBUG_SEVERITY_MEDIUM:       gablog_log(LOG_ERROR, __FILE__, __LINE__, "%s", message); return;
		case GL_DEBUG_SEVERITY_LOW:          gablog_log(LOG_WARN, __FILE__, __LINE__, "%s", message); return;
		case GL_DEBUG_SEVERITY_NOTIFICATION: gablog_log(LOG_TRACE, __FILE__, __LINE__, "%s", message); return;
	}

	gablog_log(LOG_ASSERT, __FILE__, __LINE__, "Unknown severity level!");
	gabdebug_break();
}

static void DrawProfilerTree(const GABProfileNode* node, const uint32_t depth = 0)
{
  for (const GABProfileNode* current = node; current; current = current->nextSibling)
  {
    const bool gpuBound = current->gpuTime > current->cpuTime;
    const ImVec4 color = gpuBound
      ? ImVec4(1.0f, 0.4f, 0.4f, 1.0f)
      : ImVec4(0.4f, 1.0f, 0.4f, 1.0f);
    ImGui::TextColored(color, "%*s%s: CPU %.3f ms | GPU %.3f ms",
      static_cast<int>(depth * 2), "", current->name,
      current->cpuTime, current->gpuTime);
    DrawProfilerTree(current->firstChild, depth + 1);
  }
}

struct QuadVertex
{
	glm::vec3 Position;
	glm::vec4 Color;
	glm::vec2 TexCoord;
	float TexIndex;
	float TilingFactor;

	int EntityID;
};

struct LineVertex
{
	glm::vec3 Position;
	glm::vec4 Color;

	int EntityID;
};

struct CameraData
{
	glm::mat4 ViewProjection;
	glm::mat4 OrtoProjection;
	glm::mat4 NonRotViewProjection;
	glm::vec3 CameraPos;
};

struct DrawElementsIndirectCommand
{
	GLuint count;         // Number of indices
	GLuint instanceCount; // This is for instancing
	GLuint firstIndex;    // Offset into the index buffer
	GLint baseVertex;    // Base vertex for this draw
	GLuint baseInstance;  // You can use this to index per-object data
};
static_assert(sizeof(DrawElementsIndirectCommand) == sizeof(uint32_t) * 5);

struct alignas(16) DrawCullData
{
  glm::vec4 LocalSphere{0.0f};
  glm::uvec4 Metadata{0u};
};
static_assert(sizeof(DrawCullData) == 32);

struct Debug2DCommand
{
	enum class Type { Quad, Text } type = Type::Quad;
	glm::vec2 position = glm::vec2(0.0f);
	glm::vec2 size = glm::vec2(0.0f);
	glm::vec4 color = glm::vec4(1.0f);
	float rotation = 0.0f;
	float textSize = 0.35f;
	bool outline = false;
	std::string text;
};

struct OpenGLRendererData
{
	int m_GizmoType;
	uint64_t m_SelectedEntityID = 0;
	uint64_t m_SelectedLightID = 0;
	int selectedSceneIndex = -1;
	bool m_ViewportFocused = false, m_ViewportHovered = false;
	glm::vec2 m_ViewportSize = { 0.0f, 0.0f };
	glm::vec2 m_ViewportBounds[2];
	bool m_BlockEvents = true;

	static constexpr uint32_t MaxQuads = 20000;
	static constexpr uint32_t MaxVertices = MaxQuads * 4;
	static constexpr uint32_t MaxIndices = MaxQuads * 6;
	static constexpr uint32_t MaxTextureSlots = 32; // TODO: RenderCaps

	std::shared_ptr<Texture> WhiteTexture;

	std::shared_ptr<VertexArray> QuadVertexArray;
	std::shared_ptr<VertexBuffer> QuadVertexBuffer;
	uint32_t QuadIndexCount = 0;
	QuadVertex* QuadVertexBufferBase = nullptr;
	QuadVertex* QuadVertexBufferPtr = nullptr;
	glm::vec4 QuadVertexPositions[4];
	static constexpr size_t quadVertexCount = 4;

	std::shared_ptr<VertexArray> LineVertexArray;
	std::shared_ptr<VertexBuffer> LineVertexBuffer;
	uint32_t LineVertexCount = 0;
	LineVertex* LineVertexBufferBase = nullptr;
	LineVertex* LineVertexBufferPtr = nullptr;
	float LineWidth = 2.0f;

	static constexpr glm::vec3 quadPositions[4] =
	{
		{ 0.0f, 0.0f, 0.0f },
		{ 1.0f, 0.0f, 0.0f },
		{ 1.0f, 1.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f }
	};

	static constexpr glm::vec2 tex3DCoords[4] =
	{
		{ 0.0f, 1.0f },
		{ 1.0f, 1.0f },
		{ 1.0f, 0.0f },
		{ 0.0f, 0.0f }
	};

	std::array<std::shared_ptr<Texture>, MaxTextureSlots> TextureSlots;
	uint32_t TextureSlotIndex = 1; // 0 = white texture

	static constexpr glm::vec2 tex2DCoords[4] = 
	{
		{ 0.0f, 0.0f },
		{ 1.0f, 0.0f },
		{ 1.0f, 1.0f },
		{ 0.0f, 1.0f }
	};
	static constexpr float tilingFactor = 1.0f;

	struct Shaders
	{
		std::shared_ptr<Shader> QuadShader;
		std::shared_ptr<Shader> CircleShader;
		std::shared_ptr<Shader> LineShader;
		std::shared_ptr<Shader> FramebufferShader;
		std::shared_ptr<Shader> skyboxShader;
		std::shared_ptr<Shader> GeometryShader;
		std::shared_ptr<Shader> LightShader;
		std::shared_ptr<Shader> DownSampleShader;
		std::shared_ptr<Shader> UpSampleShader;
		std::shared_ptr<Shader> BloomResultShader;
		std::shared_ptr<Shader> OmniDirectShadowShader;
		std::shared_ptr<Shader> DirectShadowShader;
		std::shared_ptr<Shader> PhysicsDebugShader;
		std::shared_ptr<Shader> ParticleShader;
		std::shared_ptr<Shader> GPUCullShader;
		std::shared_ptr<Shader> HiZBuildShader;
		std::shared_ptr<Shader> ZPrepassShader;
		std::shared_ptr<Shader> TiledLightCullShader;
	} s_Shaders;

  CameraData m_CameraBuffer;
  std::shared_ptr<UniformBuffer> m_CameraUniformBuffer;
  std::shared_ptr<UniformBuffer> m_ResolutionUniformBuffer;
  std::shared_ptr<StorageBuffer> m_LightPositionBuffer;
  std::shared_ptr<StorageBuffer> m_LightDirectionBuffer;
  std::shared_ptr<StorageBuffer> m_LightCountBuffer;
  std::shared_ptr<StorageBuffer> m_LightColorBuffer;
  std::shared_ptr<StorageBuffer> m_LightTypeBuffer;
  std::shared_ptr<StorageBuffer> m_PointShadowSlotBuffer;
  uint32_t m_LightCapacity = 32;

  GLuint m_TileLightGridBuffer = 0;
  GLuint m_TileLightIndexBuffer = 0;
  size_t m_TileBufferCapacity = 0;
  bool m_TiledLightingSupported = false;
  uint32_t m_TileDebugTileCount = 0;
  uint32_t m_TileDebugActiveTileCount = 0;
  uint32_t m_TileDebugMinLights = 0;
  uint32_t m_TileDebugMaxLights = 0;
  float m_TileDebugAverageLights = 0.0f;

  struct PointShadowCacheEntry
  {
    bool Valid = false;
    uint32_t LightIndex = std::numeric_limits<uint32_t>::max();
    glm::vec3 LightPosition{0.0f};
    uint64_t CasterHash = 0;
  };
  std::array<PointShadowCacheEntry, 4> m_PointShadowCache{};
  uint32_t m_PointShadowFacesRendered = 0;
  uint32_t m_PointShadowFacesCached = 0;

  std::unordered_map<std::string, std::shared_ptr<Texture>> skyboxes;

  std::shared_ptr<FrameBuffer> m_ResultBuffer;
  std::shared_ptr<FrameBuffer> m_PostProcessBuffer;
  std::shared_ptr<BloomBuffer> m_BloomBuffer;
  std::shared_ptr<OmniDirectShadowBuffer> m_OmniDirectShadowBuffer;
  std::shared_ptr<DirectShadowBuffer> m_DirectShadowBuffer;
  std::shared_ptr<GeometryBuffer> m_GeometryBuffer;

  std::vector<DrawElementsIndirectCommand> m_DrawCommands;
  std::vector<DrawElementsIndirectCommand> m_CulledDrawCommands;
  std::vector<glm::mat4> m_VisibleInstanceTransforms;
  std::vector<RenderModelPreview> m_ModelPreviews;
  std::unordered_map<std::string, std::vector<size_t>> m_ModelDrawCommandIndices;
  uint32_t m_DrawIndexOffset = 0;
  uint32_t m_DrawVertexOffset = 0;
  uint32_t m_cmdBufer = 0;
  uint32_t m_CulledCmdBuffer = 0;
  size_t m_cmdBufferSize = 0;
  GLuint m_GPUVisibleTransforms = 0;
  GLuint m_GPUDrawRemap = 0;
  GLuint m_GPUDrawCount = 0;
  GLuint m_CullDataRingBuffer = 0;
  uint8_t* m_CullDataMapped = nullptr;
  size_t m_CullDataSegmentStride = 0;
  size_t m_CullDataCapacity = 0;
  size_t m_CullDataCurrentOffset = 0;
  bool m_CullDataPrepared = false;
  uint32_t m_CullRingIndex = 2;
  std::array<GLsync, 3> m_CullRingFences{};
  size_t m_GPUVisibleTransformCapacity = 0;
  size_t m_GPUCommandCapacity = 0;
  size_t m_CurrentVisibleTransformCapacity = 0;
  GLuint m_HiZTexture = 0;
  uint32_t m_HiZWidth = 0;
  uint32_t m_HiZHeight = 0;
  uint32_t m_HiZMipCount = 0;
  bool m_GPUDrivenSupported = false;
  uint32_t m_GPUCullLocalSizeX = 1;
  uint32_t m_HiZLocalSizeX = 1;
  uint32_t m_HiZLocalSizeY = 1;
  bool m_HiZValid = false;
  GLuint m_FullscreenQuadVAO = 0;
  GLuint m_FullscreenQuadVBO = 0;
  GLuint m_FramebufferQuadVAO = 0;
  GLuint m_FramebufferQuadVBO = 0;
  GLuint m_SkyboxVAO = 0;
  GLuint m_SkyboxVBO = 0;
  GLuint m_ParticleVAO = 0;
  GLuint m_ParticleQuadVBO = 0;
  GLuint m_ParticleInstanceBuffer = 0;
  std::vector<Debug2DCommand> m_Debug2DCommands;
  uint32_t m_AppliedShadowQuality = std::numeric_limits<uint32_t>::max();
  static constexpr bool DirectionalShadowsEnabled = true;
  static constexpr bool HiZOcclusionEnabled = true;
  static constexpr uint32_t MaxShadowedPointLights = 4;
  static constexpr uint32_t TileSize = 16;
  static constexpr uint32_t MaxLightsPerTile = 128;
  static constexpr float PointShadowRadius = 20.0f;

	enum class SceneState { Edit = 0, Play = 1 } m_SceneState;

  bool Is3D = false;

} s_Data;

static void DestroyGPUDrivenResources();
static void BuildHiZPyramid();
static void DispatchGPUCull(const glm::mat4& viewProjection, bool useHiZ);
static void DrawCompactedCommands();
static bool PrepareGPUCullData();
static bool DispatchTiledLightCulling();
static void UploadPointShadowSlots(const std::vector<int32_t>& slots);
static void UpdateShadowFaceCulling(const glm::mat4& viewProjection);

struct PointShadowCasterState
{
  uint64_t Hash = 1469598103934665603ull;
  bool HasAnimatedCaster = false;
};

static void HashShadowBytes(uint64_t& hash, const void* data, size_t size)
{
  constexpr uint64_t prime = 1099511628211ull;
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (size_t index = 0; index < size; ++index)
  {
    hash ^= bytes[index];
    hash *= prime;
  }
}

static PointShadowCasterState GetPointShadowCasterState(const glm::vec3& lightPosition)
{
  PointShadowCasterState state;
  for (const std::string& modelName : ModelManager::GetModelNames())
  {
    const auto model = ModelManager::GetModel(modelName);
    if (!model || !model->m_IsRendered) continue;

    bool modelAdded = false;
    for (const glm::mat4& transform : model->m_InstanceTransforms)
    {
      const WorldBoundingSphere sphere = CalculateWorldBoundingSphere(*model, transform);
      const float range = OpenGLRendererData::PointShadowRadius + sphere.radius;
      const glm::vec3 offset = sphere.center - lightPosition;
      if (glm::dot(offset, offset) > range * range) continue;

      if (!modelAdded)
      {
        HashShadowBytes(state.Hash, modelName.data(), modelName.size());
        const float boundsRadius = model->GetBoundsRadius();
        HashShadowBytes(state.Hash, &boundsRadius, sizeof(boundsRadius));
        modelAdded = true;
      }
      HashShadowBytes(state.Hash, glm::value_ptr(transform), sizeof(glm::mat4));
      state.HasAnimatedCaster |= model->IsAnimated();
    }
  }
  return state;
}

static bool HasLight(const LightType type)
{
  return std::ranges::any_of(RenderBackend::Lights(), [type](const RenderLight& light)
  {
    return light.Type == type;
  });
}

static glm::vec3 GetDirectionalLightDirection()
{
  const auto& lights = RenderBackend::Lights();
  const auto light = std::ranges::find_if(lights, [](const RenderLight& candidate)
  {
    return candidate.Type == LightType::DIRECT;
  });
  return light == lights.end() ? glm::vec3(-1.0f, -2.0f, -1.0f) : light->Direction;
}

static void DrawWireSphere(const glm::vec3& center, float radius, const glm::vec4& color, int segments = 24)
{
  if (radius <= 0.0f || segments < 3) return;

  for (int i = 0; i < segments; ++i)
  {
    const float angle0 = glm::two_pi<float>() * static_cast<float>(i) / static_cast<float>(segments);
    const float angle1 = glm::two_pi<float>() * static_cast<float>(i + 1) / static_cast<float>(segments);
    const float c0 = std::cos(angle0), s0 = std::sin(angle0);
    const float c1 = std::cos(angle1), s1 = std::sin(angle1);
    OpenGLRenderer::DrawLine(center + glm::vec3(c0, s0, 0.0f) * radius,
      center + glm::vec3(c1, s1, 0.0f) * radius, color);
    OpenGLRenderer::DrawLine(center + glm::vec3(c0, 0.0f, s0) * radius,
      center + glm::vec3(c1, 0.0f, s1) * radius, color);
    OpenGLRenderer::DrawLine(center + glm::vec3(0.0f, c0, s0) * radius,
      center + glm::vec3(0.0f, c1, s1) * radius, color);
  }
}

static void DrawWireCapsule(const glm::vec3& center, float radius, float height,
  const glm::vec3& upDirection, const glm::vec4& color, int segments = 24)
{
  if (radius <= 0.0f || height < 0.0f || segments < 4) return;

  const glm::vec3 up = glm::length(upDirection) > 0.0001f
    ? glm::normalize(upDirection)
    : glm::vec3(0.0f, 1.0f, 0.0f);
  const glm::vec3 fallbackAxis = std::abs(glm::dot(up, glm::vec3(0.0f, 1.0f, 0.0f))) > 0.98f
    ? glm::vec3(1.0f, 0.0f, 0.0f)
    : glm::vec3(0.0f, 1.0f, 0.0f);
  const glm::vec3 right = glm::normalize(glm::cross(up, fallbackAxis));
  const glm::vec3 forward = glm::normalize(glm::cross(right, up));
  const glm::vec3 topCenter = center + up * (height * 0.5f);
  const glm::vec3 bottomCenter = center - up * (height * 0.5f);

  for (int i = 0; i < segments; ++i)
  {
    const float angle0 = glm::two_pi<float>() * static_cast<float>(i) / static_cast<float>(segments);
    const float angle1 = glm::two_pi<float>() * static_cast<float>(i + 1) / static_cast<float>(segments);
    const glm::vec3 radial0 = right * std::cos(angle0) + forward * std::sin(angle0);
    const glm::vec3 radial1 = right * std::cos(angle1) + forward * std::sin(angle1);

    OpenGLRenderer::DrawLine(topCenter + radial0 * radius, topCenter + radial1 * radius, color);
    OpenGLRenderer::DrawLine(bottomCenter + radial0 * radius, bottomCenter + radial1 * radius, color);
  }

  constexpr int meridians = 8;
  const int arcSegments = std::max(4, segments / 4);
  for (int meridian = 0; meridian < meridians; ++meridian)
  {
    const float angle = glm::two_pi<float>() * static_cast<float>(meridian) / static_cast<float>(meridians);
    const glm::vec3 radial = right * std::cos(angle) + forward * std::sin(angle);
    OpenGLRenderer::DrawLine(bottomCenter + radial * radius, topCenter + radial * radius, color);

    for (int arc = 0; arc < arcSegments; ++arc)
    {
      const float arc0 = glm::half_pi<float>() * static_cast<float>(arc) / static_cast<float>(arcSegments);
      const float arc1 = glm::half_pi<float>() * static_cast<float>(arc + 1) / static_cast<float>(arcSegments);
      OpenGLRenderer::DrawLine(
        topCenter + (radial * std::cos(arc0) + up * std::sin(arc0)) * radius,
        topCenter + (radial * std::cos(arc1) + up * std::sin(arc1)) * radius,
        color);
      OpenGLRenderer::DrawLine(
        bottomCenter + (radial * std::cos(arc0) - up * std::sin(arc0)) * radius,
        bottomCenter + (radial * std::cos(arc1) - up * std::sin(arc1)) * radius,
        color);
    }
  }
}

static bool WorldToScreen(const glm::vec3& worldPosition, glm::vec2& screenPosition)
{
  const glm::vec4 clip = Camera::GetViewProjection() * glm::vec4(worldPosition, 1.0f);
  if (clip.w <= 0.001f)
    return false;

  const glm::vec3 ndc = glm::vec3(clip) / clip.w;
  if (ndc.x < -1.0f || ndc.x > 1.0f || ndc.y < -1.0f || ndc.y > 1.0f ||
      ndc.z < -1.0f || ndc.z > 1.0f)
    return false;

  screenPosition = {
    (ndc.x * 0.5f + 0.5f) * static_cast<float>(Window::GetWidth()),
    (ndc.y * 0.5f + 0.5f) * static_cast<float>(Window::GetHeight())
  };
  return true;
}

static float GetResolutionUIScale()
{
  const float widthScale = static_cast<float>(Window::GetWidth()) / 1280.0f;
  const float heightScale = static_cast<float>(Window::GetHeight()) / 720.0f;
  return std::max(0.25f, std::min(widthScale, heightScale));
}

static void DrawInteractionLabels()
{
  glm::vec3 playerPosition;
  if (!SceneManager::GetPlayerPosition(playerPosition)) return;

  const Font* font = FontManager::GetFont("dpcomic");
  const float uiScale = GetResolutionUIScale();
  const uint64_t focusedEntity = SceneManager::GetFocusedEntityID();
  for (const SceneEntity& entity : SceneManager::GetEntities())
  {
    if (!entity.active || !entity.interactable || entity.player)
      continue;

    const float distance = glm::length(entity.transform.GetPosition() - playerPosition);
    if (distance > entity.interactionRange)
      continue;

    glm::vec2 screenPosition;
    if (!WorldToScreen(entity.transform.GetPosition() + glm::vec3(0.0f, entity.labelHeight, 0.0f),
        screenPosition))
      continue;

    const float fadeStart = entity.interactionRange * 0.7f;
    const float fadeLength = std::max(entity.interactionRange - fadeStart, 0.001f);
    const float alpha = 1.0f - glm::clamp((distance - fadeStart) / fadeLength, 0.0f, 1.0f);
    const bool focused = entity.id == focusedEntity;
    const std::string& displayName = entity.itemName.empty() ? entity.name : entity.itemName;
    OpenGLRenderer::DrawText(font, displayName, screenPosition, (focused ? 0.48f : 0.4f) * uiScale,
      focused ? glm::vec4(1.0f, 0.88f, 0.38f, alpha) : glm::vec4(1.0f, 1.0f, 1.0f, alpha));
    if (focused)
    {
      OpenGLRenderer::DrawText(font, entity.pickable ? "E / X - PICK UP" : "E / X - INTERACT",
        screenPosition - glm::vec2(0.0f, 27.0f * uiScale), 0.27f * uiScale,
        glm::vec4(0.9f, 0.9f, 0.9f, alpha));
    }
  }
}

static void InitializeEditorUI()
{
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
	ImGui::StyleColorsDark();

	ImGuiStyle& style = ImGui::GetStyle();
	if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		style.WindowRounding = 0.0f;
		style.Colors[ImGuiCol_WindowBg].w = 1.0f;
	}

	auto* window = reinterpret_cast<GLFWwindow*>(Window::GetWindowPtr());
	if (RenderBackend::Capabilities().OpenGLContext)
		ImGui_ImplGlfw_InitForOpenGL(window, true);
	else
		ImGui_ImplGlfw_InitForOther(window, true);

	if (!RenderBackend::Get().InitializeImGuiRenderer())
		gablog_log(LOG_ERROR, __FILE__, __LINE__, "Could not initialize ImGui for the %s backend", RenderBackend::Get().GetName());
}

static void ShutdownEditorUI()
{
	RenderBackend::Get().ShutdownImGuiRenderer();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void OpenGLRenderer::LoadShaders()
{
	Shader::Create(s_Data.s_Shaders.QuadShader, "../res/shaders/ui.slang");
	Shader::Create(s_Data.s_Shaders.CircleShader, "../res/shaders/batch_circle.slang");
	Shader::Create(s_Data.s_Shaders.LineShader, "../res/shaders/debug_lines.slang");
	Shader::Create(s_Data.s_Shaders.FramebufferShader, "../res/shaders/postprocess.slang");
	Shader::Create(s_Data.s_Shaders.skyboxShader, "../res/shaders/skybox.slang");
	Shader::Create(s_Data.s_Shaders.GeometryShader, "../res/shaders/scene.slang");
	Shader::Create(s_Data.s_Shaders.LightShader, "../res/shaders/light.slang");
	Shader::Create(s_Data.s_Shaders.DownSampleShader, "../res/shaders/bloom_downsample.slang");
	Shader::Create(s_Data.s_Shaders.UpSampleShader, "../res/shaders/bloom_upsample.slang");
	Shader::Create(s_Data.s_Shaders.BloomResultShader, "../res/shaders/bloom_final.slang");
	Shader::Create(s_Data.s_Shaders.OmniDirectShadowShader, "../res/shaders/point_shadow.slang");
	Shader::Create(s_Data.s_Shaders.DirectShadowShader, "../res/shaders/shadow.slang");
	Shader::Create(s_Data.s_Shaders.PhysicsDebugShader, "../res/shaders/physics_debug.slang");
	Shader::Create(s_Data.s_Shaders.ParticleShader, "../res/shaders/particle.slang");
	if (s_Data.m_GPUDrivenSupported)
	{
		Shader::Create(s_Data.s_Shaders.GPUCullShader, "../res/shaders/gpu_cull.slang");
		Shader::Create(s_Data.s_Shaders.HiZBuildShader, "../res/shaders/hiz_build.slang");
		Shader::Create(s_Data.s_Shaders.ZPrepassShader, "../res/shaders/geometry_z_prepass.slang");
	}
	if (s_Data.m_TiledLightingSupported)
		Shader::Create(s_Data.s_Shaders.TiledLightCullShader,
			"../res/shaders/tiled_light_cull.slang");
}

void OpenGLRenderer::SetLights(const std::vector<RenderLight>& lights)
{
  if (!s_Data.m_LightCountBuffer) return;

  if (lights.size() > s_Data.m_LightCapacity)
  {
    while (lights.size() > s_Data.m_LightCapacity) s_Data.m_LightCapacity *= 2;
    s_Data.m_LightPositionBuffer = StorageBuffer::Create(sizeof(glm::vec4) * s_Data.m_LightCapacity, 0);
    s_Data.m_LightDirectionBuffer = StorageBuffer::Create(sizeof(glm::vec4) * s_Data.m_LightCapacity, 1);
    s_Data.m_LightCountBuffer = StorageBuffer::Create(sizeof(uint32_t), 2);
    s_Data.m_LightColorBuffer = StorageBuffer::Create(sizeof(glm::vec4) * s_Data.m_LightCapacity, 3);
    s_Data.m_LightTypeBuffer = StorageBuffer::Create(sizeof(uint32_t) * s_Data.m_LightCapacity, 4);
    s_Data.m_PointShadowSlotBuffer = StorageBuffer::Create(
      sizeof(int32_t) * s_Data.m_LightCapacity, 23);
  }

  const uint32_t count = static_cast<uint32_t>(lights.size());
  s_Data.m_LightCountBuffer->SetData(sizeof(count), &count);
  if (lights.empty())
  {
    return;
  }

  std::vector<glm::vec4> positions(count);
  std::vector<glm::vec4> directions(count);
  std::vector<glm::vec4> colors(count);
  std::vector<uint32_t> types(count);
  for (size_t i = 0; i < lights.size(); ++i)
  {
    positions[i] = glm::vec4(lights[i].Position, 0.0f);
    directions[i] = glm::vec4(lights[i].Direction, 0.0f);
    colors[i] = glm::vec4(lights[i].Color, 1.0f);
    types[i] = static_cast<uint32_t>(lights[i].Type);
  }
  s_Data.m_LightPositionBuffer->SetData(positions.size() * sizeof(glm::vec4), positions.data());
  s_Data.m_LightDirectionBuffer->SetData(directions.size() * sizeof(glm::vec4), directions.data());
  s_Data.m_LightColorBuffer->SetData(colors.size() * sizeof(glm::vec4), colors.data());
  s_Data.m_LightTypeBuffer->SetData(types.size() * sizeof(uint32_t), types.data());
  UploadPointShadowSlots(std::vector<int32_t>(count, -1));
}

void OpenGLRenderer::Init()
{
  if (RenderBackend::Capabilities().NativeSceneRenderer)
  {
    const glm::vec2 resolution = {Window::GetWidth(), Window::GetHeight()};
    Camera::Init(45.0f, resolution.x / resolution.y, 0.01f, 2000.0f);
    Camera::SetViewportSize(resolution.x, resolution.y);
    Camera::SetMode(CameraMode::PLAYER);
    Window::SetCursorVisible(false);
    RenderBackend::Get().InitializeSceneRenderer();
    s_Data.m_SceneState = OpenGLRendererData::SceneState::Play;

    InitializeEditorUI();
    return;
  }

#ifndef NDEBUG
	glEnable(GL_DEBUG_OUTPUT);
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageCallback(MessageCallback, nullptr);

	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, NULL, GL_FALSE);
#endif

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LINE_SMOOTH);

	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

	s_Data.QuadVertexArray = VertexArray::Create();
	s_Data.QuadVertexBuffer = VertexBuffer::Create(s_Data.MaxVertices * sizeof(QuadVertex));
	s_Data.QuadVertexBuffer->SetLayout({
		{ ShaderDataType::Float3, "a_Position"     },
		{ ShaderDataType::Float4, "a_Color"        },
		{ ShaderDataType::Float2, "a_TexCoord"     },
		{ ShaderDataType::Float,  "a_TexIndex"     },
		{ ShaderDataType::Float,  "a_TilingFactor" },
		{ ShaderDataType::Int,    "a_EntityID"     }
		});
	s_Data.QuadVertexArray->AddVertexBuffer(s_Data.QuadVertexBuffer);
	s_Data.QuadVertexBufferBase = new QuadVertex[s_Data.MaxVertices];

	uint32_t* quadIndices = new uint32_t[s_Data.MaxIndices];

	uint32_t offset = 0;
	for (uint32_t i = 0; i < s_Data.MaxIndices; i += 6)
	{
		quadIndices[i + 0] = offset + 0;
		quadIndices[i + 1] = offset + 1;
		quadIndices[i + 2] = offset + 2;

		quadIndices[i + 3] = offset + 2;
		quadIndices[i + 4] = offset + 3;
		quadIndices[i + 5] = offset + 0;

		offset += 4;
	}

	std::shared_ptr<IndexBuffer> quadIB = IndexBuffer::Create(quadIndices, s_Data.MaxIndices);
	s_Data.QuadVertexArray->SetIndexBuffer(quadIB);
	delete[] quadIndices;

	s_Data.LineVertexArray = VertexArray::Create();
	s_Data.LineVertexBuffer = VertexBuffer::Create(s_Data.MaxVertices * sizeof(LineVertex));
	s_Data.LineVertexBuffer->SetLayout({
		{ ShaderDataType::Float3, "a_Position" },
		{ ShaderDataType::Float4, "a_Color"    },
		{ ShaderDataType::Int,    "a_EntityID" }
		});
	s_Data.LineVertexArray->AddVertexBuffer(s_Data.LineVertexBuffer);
	s_Data.LineVertexBufferBase = new LineVertex[s_Data.MaxVertices];

	s_Data.WhiteTexture = Texture::Create(TextureSpecification());
	uint32_t whiteTextureData = 0xffffffff;
	s_Data.WhiteTexture->SetData(&whiteTextureData, sizeof(uint32_t));

	int32_t samplers[s_Data.MaxTextureSlots];
	for (uint32_t i = 0; i < s_Data.MaxTextureSlots; i++) samplers[i] = i;

	s_Data.TextureSlots[0] = s_Data.WhiteTexture;

	s_Data.QuadVertexPositions[0] = { -0.5f, -0.5f, 0.0f, 1.0f };
	s_Data.QuadVertexPositions[1] = { 0.5f, -0.5f, 0.0f, 1.0f };
	s_Data.QuadVertexPositions[2] = { 0.5f,  0.5f, 0.0f, 1.0f };
	s_Data.QuadVertexPositions[3] = { -0.5f,  0.5f, 0.0f, 1.0f };

	std::vector<uint32_t> indices;
	indices.reserve(s_Data.MaxIndices);

  s_Data.m_GPUDrivenSupported = GLAD_GL_VERSION_4_6 &&
		glDispatchCompute != nullptr && glMultiDrawElementsIndirectCount != nullptr;
	s_Data.m_TiledLightingSupported = GLAD_GL_VERSION_4_6 && glDispatchCompute != nullptr;
	LoadShaders();
	if (s_Data.m_GPUDrivenSupported)
	{
		GLint localSize[3] = {1, 1, 1};
		glGetProgramiv(s_Data.s_Shaders.GPUCullShader->GetID(),
			GL_COMPUTE_WORK_GROUP_SIZE, localSize);
		s_Data.m_GPUCullLocalSizeX = static_cast<uint32_t>(std::max(localSize[0], 1));
		glGetProgramiv(s_Data.s_Shaders.HiZBuildShader->GetID(),
			GL_COMPUTE_WORK_GROUP_SIZE, localSize);
		s_Data.m_HiZLocalSizeX = static_cast<uint32_t>(std::max(localSize[0], 1));
		s_Data.m_HiZLocalSizeY = static_cast<uint32_t>(std::max(localSize[1], 1));
	}

	constexpr glm::vec2 particleQuadVertices[4] = {
		{-0.5f, -0.5f}, {0.5f, -0.5f}, {-0.5f, 0.5f}, {0.5f, 0.5f}};
	glCreateVertexArrays(1, &s_Data.m_ParticleVAO);
	glCreateBuffers(1, &s_Data.m_ParticleQuadVBO);
	glCreateBuffers(1, &s_Data.m_ParticleInstanceBuffer);
	glNamedBufferData(s_Data.m_ParticleQuadVBO, sizeof(particleQuadVertices),
		particleQuadVertices, GL_STATIC_DRAW);
	glNamedBufferData(s_Data.m_ParticleInstanceBuffer,
		static_cast<GLsizeiptr>(640 * sizeof(ParticleRenderInstance)), nullptr, GL_DYNAMIC_DRAW);
	glVertexArrayVertexBuffer(s_Data.m_ParticleVAO, 0, s_Data.m_ParticleQuadVBO, 0,
		sizeof(glm::vec2));
	glEnableVertexArrayAttrib(s_Data.m_ParticleVAO, 0);
	glVertexArrayAttribFormat(s_Data.m_ParticleVAO, 0, 2, GL_FLOAT, GL_FALSE, 0);
	glVertexArrayAttribBinding(s_Data.m_ParticleVAO, 0, 0);
	glVertexArrayVertexBuffer(s_Data.m_ParticleVAO, 1, s_Data.m_ParticleInstanceBuffer, 0,
		sizeof(ParticleRenderInstance));
	glVertexArrayBindingDivisor(s_Data.m_ParticleVAO, 1, 1);
	const std::array<GLint, 5> particleComponents = {4, 4, 1, 4, 4};
	const std::array<GLuint, 5> particleOffsets = {
		static_cast<GLuint>(offsetof(ParticleRenderInstance, PositionAndSize)),
		static_cast<GLuint>(offsetof(ParticleRenderInstance, Color)),
		static_cast<GLuint>(offsetof(ParticleRenderInstance, Rotation)),
		static_cast<GLuint>(offsetof(ParticleRenderInstance, RightAndStyle)),
		static_cast<GLuint>(offsetof(ParticleRenderInstance, Up))};
	for (GLuint attribute = 0; attribute < particleComponents.size(); ++attribute)
	{
		const GLuint location = attribute + 1;
		glEnableVertexArrayAttrib(s_Data.m_ParticleVAO, location);
		glVertexArrayAttribFormat(s_Data.m_ParticleVAO, location,
			particleComponents[attribute], GL_FLOAT, GL_FALSE, particleOffsets[attribute]);
		glVertexArrayAttribBinding(s_Data.m_ParticleVAO, location, 1);
	}

	glm::vec2 resolution = { Window::GetWidth(), Window::GetHeight() };

	FramebufferSpecification fbSpec;
	fbSpec.Attachments = { FramebufferTextureFormat::RGBA8, FramebufferTextureFormat::RED_INTEGER };
	fbSpec.Width = resolution.x;
	fbSpec.Height = resolution.y;
	s_Data.m_ResultBuffer = FrameBuffer::Create(fbSpec);

	FramebufferSpecification postProcessSpec;
	postProcessSpec.Attachments = { FramebufferTextureFormat::RGBA8 };
	postProcessSpec.Width = resolution.x;
	postProcessSpec.Height = resolution.y;
	postProcessSpec.NearestFiltering = true;
	s_Data.m_PostProcessBuffer = FrameBuffer::Create(postProcessSpec);

	s_Data.m_GeometryBuffer = GeometryBuffer::Create(resolution.x, resolution.y);
	// Forward rendering consumes the same depth produced by the geometry pass.
	// Sharing the attachment avoids a full-resolution depth blit every frame.
	s_Data.m_ResultBuffer->AttachExternalDepthTexture(
		s_Data.m_GeometryBuffer->GetDepthAttachmentRendererID());
	s_Data.m_BloomBuffer = BloomBuffer::Create(s_Data.s_Shaders.DownSampleShader, s_Data.s_Shaders.UpSampleShader, s_Data.s_Shaders.BloomResultShader);
	ApplyGraphicsSettings();

	s_Data.m_CameraUniformBuffer = UniformBuffer::Create(sizeof(CameraData), 0);
	Camera::Init(45.0f, (float)resolution.x / (float)resolution.y, 0.01f, 2000.0f);
	Camera::SetViewportSize((float)resolution.x, (float)resolution.y);

	// std140 rounds a uniform block containing a vec2 up to a 16-byte block.
	s_Data.m_ResolutionUniformBuffer = UniformBuffer::Create(sizeof(glm::vec4), 1);
	s_Data.m_ResolutionUniformBuffer->SetData(&resolution, sizeof(glm::vec2));
	s_Data.m_LightPositionBuffer = StorageBuffer::Create(sizeof(glm::vec4) * s_Data.m_LightCapacity, 0);
	s_Data.m_LightDirectionBuffer = StorageBuffer::Create(sizeof(glm::vec4) * s_Data.m_LightCapacity, 1);
	s_Data.m_LightCountBuffer = StorageBuffer::Create(sizeof(uint32_t), 2);
	s_Data.m_LightColorBuffer = StorageBuffer::Create(sizeof(glm::vec4) * s_Data.m_LightCapacity, 3);
	s_Data.m_LightTypeBuffer = StorageBuffer::Create(sizeof(uint32_t) * s_Data.m_LightCapacity, 4);
	s_Data.m_PointShadowSlotBuffer = StorageBuffer::Create(
		sizeof(int32_t) * s_Data.m_LightCapacity, 23);
	SetLights(RenderBackend::Lights());

	s_Data.m_SceneState = OpenGLRendererData::SceneState::Play;
	Camera::SetMode(CameraMode::PLAYER);
	Window::SetCursorVisible(false);

	InitializeEditorUI();
	SetLineWidth(4.0f);
	//s_Data.m_GizmoType = ImGuizmo::OPERATION::TRANSLATE;

	GABOpenGLConfig profilerConfig{};
	profilerConfig.gl.genQueries = reinterpret_cast<GABGLGenQueriesFn>(glGenQueries);
	profilerConfig.gl.deleteQueries = reinterpret_cast<GABGLDeleteQueriesFn>(glDeleteQueries);
	profilerConfig.gl.queryCounter = reinterpret_cast<GABGLQueryCounterFn>(glQueryCounter);
	profilerConfig.gl.getQueryObjectiv =
		reinterpret_cast<GABGLGetQueryObjectivFn>(glGetQueryObjectiv);
	profilerConfig.gl.getQueryObjectui64v =
		reinterpret_cast<GABGLGetQueryObjectui64vFn>(glGetQueryObjectui64v);
	profilerConfig.frameLatency = 3;
	profilerConfig.maxThreads = 1;
	profilerConfig.maxScopes = 256;
	profilerConfig.maxOccurrences = 4;
	if (!gab_gpu_opengl_install(&profilerConfig))
		gablog_log(LOG_WARN, __FILE__, __LINE__, "GABDEBUG OpenGL GPU profiler could not be installed; CPU profiling remains active");
}

void OpenGLRenderer::Shutdown()
{
  if (RenderBackend::Capabilities().NativeSceneRenderer)
  {
    ShutdownEditorUI();
    RenderBackend::Get().ShutdownSceneRenderer();
    return;
  }

	gabprofiler_end_frame();
	gabprofiler_unregister_thread();
	gabprofiler_shutdown_gpu_backend();
	DestroyGPUDrivenResources();
	ResetModelDrawCommands();

	ShutdownEditorUI();

	delete[] s_Data.QuadVertexBufferBase;
	s_Data.QuadVertexBufferBase = nullptr;
	s_Data.QuadVertexBufferPtr = nullptr;
	delete[] s_Data.LineVertexBufferBase;
	s_Data.LineVertexBufferBase = nullptr;
	s_Data.LineVertexBufferPtr = nullptr;

	if (s_Data.m_FullscreenQuadVBO) glDeleteBuffers(1, &s_Data.m_FullscreenQuadVBO);
	if (s_Data.m_FullscreenQuadVAO) glDeleteVertexArrays(1, &s_Data.m_FullscreenQuadVAO);
	if (s_Data.m_FramebufferQuadVBO) glDeleteBuffers(1, &s_Data.m_FramebufferQuadVBO);
	if (s_Data.m_FramebufferQuadVAO) glDeleteVertexArrays(1, &s_Data.m_FramebufferQuadVAO);
	if (s_Data.m_SkyboxVBO) glDeleteBuffers(1, &s_Data.m_SkyboxVBO);
	if (s_Data.m_SkyboxVAO) glDeleteVertexArrays(1, &s_Data.m_SkyboxVAO);
	if (s_Data.m_ParticleInstanceBuffer) glDeleteBuffers(1, &s_Data.m_ParticleInstanceBuffer);
	if (s_Data.m_ParticleQuadVBO) glDeleteBuffers(1, &s_Data.m_ParticleQuadVBO);
	if (s_Data.m_ParticleVAO) glDeleteVertexArrays(1, &s_Data.m_ParticleVAO);
	if (s_Data.m_TileLightGridBuffer) glDeleteBuffers(1, &s_Data.m_TileLightGridBuffer);
	if (s_Data.m_TileLightIndexBuffer) glDeleteBuffers(1, &s_Data.m_TileLightIndexBuffer);
	s_Data.m_FullscreenQuadVAO = s_Data.m_FullscreenQuadVBO = 0;
	s_Data.m_FramebufferQuadVAO = s_Data.m_FramebufferQuadVBO = 0;
	s_Data.m_SkyboxVAO = s_Data.m_SkyboxVBO = 0;
	s_Data.m_ParticleVAO = s_Data.m_ParticleQuadVBO = s_Data.m_ParticleInstanceBuffer = 0;
	s_Data.m_TileLightGridBuffer = s_Data.m_TileLightIndexBuffer = 0;
	s_Data.m_TileBufferCapacity = 0;

	s_Data.m_ResultBuffer.reset();
	s_Data.m_PostProcessBuffer.reset();
	s_Data.m_GeometryBuffer.reset();
	s_Data.m_BloomBuffer.reset();
	s_Data.m_OmniDirectShadowBuffer.reset();
	s_Data.m_DirectShadowBuffer.reset();
	s_Data.m_CameraUniformBuffer.reset();
	s_Data.m_ResolutionUniformBuffer.reset();
	s_Data.m_LightPositionBuffer.reset();
	s_Data.m_LightDirectionBuffer.reset();
	s_Data.m_LightCountBuffer.reset();
	s_Data.m_LightColorBuffer.reset();
	s_Data.m_LightTypeBuffer.reset();
	s_Data.m_PointShadowSlotBuffer.reset();
	s_Data.QuadVertexArray.reset();
	s_Data.QuadVertexBuffer.reset();
	s_Data.LineVertexArray.reset();
	s_Data.LineVertexBuffer.reset();
	s_Data.skyboxes.clear();
	s_Data.TextureSlots.fill(nullptr);
	s_Data.WhiteTexture.reset();
	s_Data.s_Shaders = {};
}

void OpenGLRenderer::DrawScene(DeltaTime& dt, const std::function<void()>& scene_logic, bool advanceSimulation)
{
  const RenderEffectSettings effects = RenderBackend::GetEffectSettings();
  const bool renderForEditor = s_Data.m_SceneState == OpenGLRendererData::SceneState::Edit;

  ApplyGraphicsSettings();

  const bool shadowsEnabled = effects.ShadowQuality != GraphicsQuality::Off;

  glDisable(GL_DITHER);
  glDisable(GL_BLEND);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_FRONT);
  glFrontFace(GL_CW);

  if (advanceSimulation) scene_logic();

  // The editor reads the completed tree above; beginning the new frame here
  // lets GABDEBUG recycle its nodes only after that direct read is finished.
  gabprofiler_begin_frame();

  if (advanceSimulation)
  {
    GABProfilerScope profileScope = gabprofiler_begin("MISC UPDATE PASS");

    ModelManager::UpdateControllers(dt);
    PhysX::Simulate(dt);
    ModelManager::UpdateTransforms(dt);
    Camera::OnUpdate(dt);
    AudioManager::SetListenerLocation(Camera::GetPosition());
    AudioManager::SetListenerOrientation(Camera::GetForwardDirection(), Camera::GetUpDirection());
    AudioManager::UpdateAllMusic();
    gabprofiler_end(&profileScope);
  }
  ModelManager::BindAllInstanceTransforms();
  s_Data.m_CullDataPrepared = false;
  const bool shadowGPUCull = PrepareGPUCullData();
  if(OpenGLRendererData::DirectionalShadowsEnabled &&
      shadowsEnabled && HasLight(LightType::DIRECT))
  {
    GABProfilerScope profileScope = gabprofiler_begin("DIRECT SHADOW PASS");

    s_Data.m_DirectShadowBuffer->Bind();
    float max = std::numeric_limits<float>::max();
  	glClearColor(max, max, max, max);
    glClear(GL_DEPTH_BUFFER_BIT);

    const glm::vec3 shadowFocus = Camera::GetPosition() + Camera::GetForwardDirection() * 45.0f;
    s_Data.m_DirectShadowBuffer->UpdateShadowView(GetDirectionalLightDirection(), shadowFocus);
    const glm::mat4 lightViewProjection = s_Data.m_DirectShadowBuffer->GetShadowViewProj();
    if (shadowGPUCull)
      DispatchGPUCull(lightViewProjection, false);
    s_Data.s_Shaders.DirectShadowShader->Bind();
    s_Data.s_Shaders.DirectShadowShader->SetMat4("u_LightSpaceMatrix", lightViewProjection);
    s_Data.s_Shaders.DirectShadowShader->SetBool("u_GPUDriven", shadowGPUCull);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(2.0f, 4.0f);
    glBindVertexArray(ModelManager::GetModelsVAO());
    if (shadowGPUCull)
      DrawCompactedCommands();
    else
    {
      glBindBuffer(GL_DRAW_INDIRECT_BUFFER, s_Data.m_cmdBufer);
      glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, nullptr,
        static_cast<GLsizei>(s_Data.m_DrawCommands.size()), 0);
      glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
    }
    glBindVertexArray(0);
    glDisable(GL_POLYGON_OFFSET_FILL);

    s_Data.s_Shaders.DirectShadowShader->UnBind();
    s_Data.m_DirectShadowBuffer->UnBind();
    gabprofiler_end(&profileScope);
  }
  if (shadowsEnabled && HasLight(LightType::POINT))
  {
    GABProfilerScope profileScope = gabprofiler_begin("OMNI SHADOW PASS");

    s_Data.m_PointShadowFacesRendered = 0;
    s_Data.m_PointShadowFacesCached = 0;

    struct ShadowCandidate
    {
      float DistanceSquared;
      uint32_t LightIndex;
    };

    const auto& lights = RenderBackend::Lights();
    std::vector<ShadowCandidate> candidates;
    candidates.reserve(lights.size());
    const glm::vec3 cameraPosition = Camera::GetPosition();
    const RenderFrustum cameraFrustum(Camera::GetViewProjection());
    for (size_t lightIndex = 0; lightIndex < lights.size(); ++lightIndex)
    {
      if (lights[lightIndex].Type != LightType::POINT) continue;
      if (!cameraFrustum.IntersectsSphere(
            lights[lightIndex].Position, OpenGLRendererData::PointShadowRadius))
        continue;
      const glm::vec3 toLight = lights[lightIndex].Position - cameraPosition;
      candidates.push_back({glm::dot(toLight, toLight),
        static_cast<uint32_t>(lightIndex)});
    }
    std::ranges::sort(candidates, [](const ShadowCandidate& lhs,
                                    const ShadowCandidate& rhs)
    {
      return lhs.DistanceSquared < rhs.DistanceSquared;
    });
    if (candidates.size() > OpenGLRendererData::MaxShadowedPointLights)
      candidates.resize(OpenGLRendererData::MaxShadowedPointLights);

    std::vector<int32_t> pointShadowSlots(lights.size(), -1);
    std::vector<size_t> candidateSlots(candidates.size(),
      s_Data.m_PointShadowCache.size());
    std::array<bool, OpenGLRendererData::MaxShadowedPointLights> usedSlots{};
    for (size_t candidateIndex = 0; candidateIndex < candidates.size(); ++candidateIndex)
    {
      for (size_t slot = 0; slot < s_Data.m_PointShadowCache.size(); ++slot)
      {
        const auto& cache = s_Data.m_PointShadowCache[slot];
        if (!usedSlots[slot] && cache.Valid &&
            cache.LightIndex == candidates[candidateIndex].LightIndex)
        {
          candidateSlots[candidateIndex] = slot;
          usedSlots[slot] = true;
          break;
        }
      }
    }
    for (size_t candidateIndex = 0; candidateIndex < candidates.size(); ++candidateIndex)
    {
      if (candidateSlots[candidateIndex] < s_Data.m_PointShadowCache.size()) continue;
      const auto freeSlot = std::ranges::find(usedSlots, false);
      if (freeSlot == usedSlots.end())
      {
        gablog_log(LOG_ASSERT, __FILE__, __LINE__, "No free point-shadow cache slot");
        gabdebug_break();
      }
      const size_t slot = static_cast<size_t>(std::distance(usedSlots.begin(), freeSlot));
      candidateSlots[candidateIndex] = slot;
      usedSlots[slot] = true;
    }

    s_Data.m_OmniDirectShadowBuffer->Bind();
    glBindVertexArray(ModelManager::GetModelsVAO());
    const auto& directions = s_Data.m_OmniDirectShadowBuffer->GetFaceDirections();
    for (size_t candidateIndex = 0; candidateIndex < candidates.size(); ++candidateIndex)
    {
      const ShadowCandidate& candidate = candidates[candidateIndex];
      const size_t shadowSlot = candidateSlots[candidateIndex];
      const uint32_t lightIndex = candidate.LightIndex;
      const glm::vec3& light = lights[lightIndex].Position;
      pointShadowSlots[lightIndex] = static_cast<int32_t>(shadowSlot);
      const PointShadowCasterState casterState = GetPointShadowCasterState(light);
      auto& cache = s_Data.m_PointShadowCache[shadowSlot];
      const glm::vec3 lightDelta = cache.LightPosition - light;
      const bool lightMoved = glm::dot(lightDelta, lightDelta) > 0.000001f;
      const bool renderShadow = !cache.Valid || cache.LightIndex != lightIndex ||
        lightMoved || cache.CasterHash != casterState.Hash ||
        casterState.HasAnimatedCaster;
      if (!renderShadow)
      {
        s_Data.m_PointShadowFacesCached += static_cast<uint32_t>(directions.size());
        continue;
      }

      for (size_t face = 0; face < directions.size(); ++face)
      {
        s_Data.m_OmniDirectShadowBuffer->BindCubemapFaceForWriting(
          static_cast<uint32_t>(shadowSlot), static_cast<uint32_t>(face));
        glClearColor(OpenGLRendererData::PointShadowRadius,
          OpenGLRendererData::PointShadowRadius,
          OpenGLRendererData::PointShadowRadius,
          OpenGLRendererData::PointShadowRadius);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        const glm::mat4 view = glm::lookAt(light,
          light + directions[face].Target, directions[face].Up);
        const glm::mat4 lightViewProjection =
          s_Data.m_OmniDirectShadowBuffer->GetShadowProj() * view;
        if (shadowGPUCull)
          DispatchGPUCull(lightViewProjection, false);
        else
          UpdateShadowFaceCulling(lightViewProjection);
        s_Data.s_Shaders.OmniDirectShadowShader->Bind();
        s_Data.s_Shaders.OmniDirectShadowShader->SetBool(
          "u_GPUDriven", shadowGPUCull);
        s_Data.s_Shaders.OmniDirectShadowShader->SetVec3(
          "gLightWorldPos", light);
        s_Data.s_Shaders.OmniDirectShadowShader->SetMat4(
          "u_LightViewProjection", lightViewProjection);
        if (shadowGPUCull)
          DrawCompactedCommands();
        else
        {
          ModelManager::BindVisibleInstanceTransforms();
          glBindBuffer(GL_DRAW_INDIRECT_BUFFER, s_Data.m_CulledCmdBuffer);
          glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, nullptr,
            static_cast<GLsizei>(s_Data.m_CulledDrawCommands.size()), 0);
          glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
        }
        ++s_Data.m_PointShadowFacesRendered;
      }
      cache.Valid = true;
      cache.LightIndex = lightIndex;
      cache.LightPosition = light;
      cache.CasterHash = casterState.Hash;
    }
    for (size_t shadowSlot = 0; shadowSlot < s_Data.m_PointShadowCache.size(); ++shadowSlot)
      if (!usedSlots[shadowSlot]) s_Data.m_PointShadowCache[shadowSlot].Valid = false;
    glBindVertexArray(0);
    s_Data.s_Shaders.OmniDirectShadowShader->UnBind();
    s_Data.m_OmniDirectShadowBuffer->UnBind();
    UploadPointShadowSlots(pointShadowSlots);
    gabprofiler_end(&profileScope);
  }
  else
  {
    s_Data.m_PointShadowFacesRendered = 0;
    s_Data.m_PointShadowFacesCached = 0;
    UploadPointShadowSlots(std::vector<int32_t>(RenderBackend::Lights().size(), -1));
  }

  UpdateModelFrustumCulling();
  BeginScene();

  const bool useDepthPrepass = s_Data.m_GPUDrivenSupported &&
    OpenGLRendererData::HiZOcclusionEnabled && !s_Data.m_DrawCommands.empty();
  if (useDepthPrepass)
  {
    GABProfilerScope profileScope = gabprofiler_begin("GPU DEPTH PREPASS + HIZ");
    s_Data.m_GeometryBuffer->Bind();
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(0.0f, 1.0f);
    glClear(GL_DEPTH_BUFFER_BIT);
    s_Data.s_Shaders.ZPrepassShader->Bind();
    s_Data.s_Shaders.ZPrepassShader->SetBool("u_PS1Enabled", effects.PS1Enabled);
    s_Data.s_Shaders.ZPrepassShader->SetFloat("u_PS1VirtualHeight", effects.PS1VirtualHeight);
    glBindVertexArray(ModelManager::GetModelsVAO());
    DrawCompactedCommands();
    glBindVertexArray(0);
    s_Data.s_Shaders.ZPrepassShader->UnBind();
    glDisable(GL_POLYGON_OFFSET_FILL);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    s_Data.m_GeometryBuffer->UnBind();

    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT |
                    GL_COMMAND_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    BuildHiZPyramid();
    DispatchGPUCull(Camera::GetViewProjection(), true);
    gabprofiler_end(&profileScope);
  }

  {
    GABProfilerScope profileScope = gabprofiler_begin("GEOMETRY PASS");

    s_Data.m_GeometryBuffer->Bind();

	glDepthFunc(useDepthPrepass ? GL_LEQUAL : GL_LESS);
	glDepthMask(useDepthPrepass ? GL_FALSE : GL_TRUE);
  	glClearColor(0, 0, 0, 0);
	glClear(useDepthPrepass ? GL_COLOR_BUFFER_BIT
	                        : GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    s_Data.s_Shaders.GeometryShader->Bind();
    s_Data.s_Shaders.GeometryShader->SetFloat("u_PS1VirtualHeight", effects.PS1VirtualHeight);
    s_Data.s_Shaders.GeometryShader->SetBool("u_PS1Enabled", effects.PS1Enabled);
    s_Data.s_Shaders.GeometryShader->SetBool("u_PreviewMode", false);
    s_Data.s_Shaders.GeometryShader->SetBool("u_GPUDriven", s_Data.m_GPUDrivenSupported);
    glBindVertexArray(ModelManager::GetModelsVAO());
    if (s_Data.m_GPUDrivenSupported)
      DrawCompactedCommands();
    else
    {
      ModelManager::BindVisibleInstanceTransforms();
      glBindBuffer(GL_DRAW_INDIRECT_BUFFER, s_Data.m_CulledCmdBuffer);
      glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, NULL,
        s_Data.m_CulledDrawCommands.size(), 0);
      glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
    }
    glDepthMask(GL_TRUE);

    if (!s_Data.m_ModelPreviews.empty())
    {
      s_Data.s_Shaders.GeometryShader->SetBool("u_PreviewMode", true);
      for (const RenderModelPreview& preview : s_Data.m_ModelPreviews)
      {
        const auto model = ModelManager::GetModel(preview.ModelName);
        const auto commandIndices = s_Data.m_ModelDrawCommandIndices.find(preview.ModelName);
        if (!model || commandIndices == s_Data.m_ModelDrawCommandIndices.end() ||
            model->GetPhysXMeshType() == MeshType::CONVEXMESH)
          continue;

        s_Data.s_Shaders.GeometryShader->SetMat4("u_PreviewModel", preview.Transform);
        s_Data.s_Shaders.GeometryShader->SetFloat("u_PreviewBrightness", preview.Brightness);
        for (const size_t commandIndex : commandIndices->second)
        {
          if (commandIndex >= s_Data.m_DrawCommands.size()) continue;
          const DrawElementsIndirectCommand& command = s_Data.m_DrawCommands[commandIndex];
          s_Data.s_Shaders.GeometryShader->SetInt(
            "u_PreviewDrawID", static_cast<int>(commandIndex));
          glDrawElementsBaseVertex(
            GL_TRIANGLES, static_cast<GLsizei>(command.count), GL_UNSIGNED_INT,
            reinterpret_cast<const void*>(static_cast<uintptr_t>(command.firstIndex) * sizeof(GLuint)),
            command.baseVertex);
        }
      }
      s_Data.s_Shaders.GeometryShader->SetBool("u_PreviewMode", false);
    }

    glBindVertexArray(0);
    EndScene();
    s_Data.s_Shaders.GeometryShader->UnBind();

    s_Data.m_GeometryBuffer->UnBind();
    if (s_Data.m_GPUDrivenSupported)
    {
      GLsync& fence = s_Data.m_CullRingFences[s_Data.m_CullRingIndex];
      if (fence) glDeleteSync(fence);
      fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    }
    ModelManager::EndGPUFrame();
    gabprofiler_end(&profileScope);
  }
  bool tiledLighting = false;
  {
    GABProfilerScope profileScope = gabprofiler_begin("TILED LIGHT CULL");
    tiledLighting = DispatchTiledLightCulling();
    gabprofiler_end(&profileScope);
  }

  {
    GABProfilerScope profileScope = gabprofiler_begin("LIGHT PASS");

    s_Data.m_BloomBuffer->Bind();
	glDisable(GL_DEPTH_TEST);
  	glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    s_Data.m_GeometryBuffer->BindPositionTextureForReading(GL_TEXTURE1);
    s_Data.m_GeometryBuffer->BindNormalTextureForReading(GL_TEXTURE2);
    s_Data.m_GeometryBuffer->BindAlbedoTextureForReading(GL_TEXTURE3);
    s_Data.m_DirectShadowBuffer->BindShadowTextureForReading(GL_TEXTURE4);
    s_Data.m_DirectShadowBuffer->BindOffsetTextureForReading(GL_TEXTURE5);
    s_Data.m_OmniDirectShadowBuffer->BindShadowTextureForReading(GL_TEXTURE6);

    s_Data.s_Shaders.LightShader->Bind();
    s_Data.s_Shaders.LightShader->SetInt("gPosition", 1);
    s_Data.s_Shaders.LightShader->SetInt("gNormal", 2);
    s_Data.s_Shaders.LightShader->SetInt("gAlbedoSpec", 3);
    s_Data.s_Shaders.LightShader->SetInt("u_DirectShadow", 4);
    s_Data.s_Shaders.LightShader->SetInt("u_OffsetTexture", 5);
    s_Data.s_Shaders.LightShader->SetInt("u_OmniShadow", 6);
    s_Data.s_Shaders.LightShader->SetBool(
      "u_DirectShadowsEnabled",
      shadowsEnabled && OpenGLRendererData::DirectionalShadowsEnabled);
    s_Data.s_Shaders.LightShader->SetBool("u_PointShadowsEnabled", shadowsEnabled);
    s_Data.s_Shaders.LightShader->SetBool("u_TiledLightingEnabled", tiledLighting);
    s_Data.s_Shaders.LightShader->SetInt("u_TiledDebugMode",
      tiledLighting ? static_cast<int>(RenderBackend::DebugSettings().TiledLightingMode) : 0);
    s_Data.s_Shaders.LightShader->SetFloat("u_BloomThreshold", effects.BloomThreshold);
    s_Data.s_Shaders.LightShader->SetMat4("u_DirectShadowViewProj", s_Data.m_DirectShadowBuffer->GetShadowViewProj());

    DrawFullscreenQuad();

    s_Data.s_Shaders.LightShader->UnBind();
    s_Data.m_BloomBuffer->UnBind();
    gabprofiler_end(&profileScope);
  }
  {
    GABProfilerScope profileScope = gabprofiler_begin("BLOOM PASS");

    if (effects.BloomQuality != GraphicsQuality::Off)
    {
      s_Data.m_BloomBuffer->RenderBloomTexture(effects.BloomFilterRadius, effects.BloomPassCount());
    }

    s_Data.m_ResultBuffer->ClearAttachment(1, -1);
    s_Data.m_BloomBuffer->CompositeTo(s_Data.m_ResultBuffer,
      effects.BloomQuality != GraphicsQuality::Off,
      effects.BloomExposure, effects.BloomStrength, effects.Gamma);
    gabprofiler_end(&profileScope);
  }
  {
    GABProfilerScope profileScope = gabprofiler_begin("FORWARD PASS");

    s_Data.m_ResultBuffer->Bind();
    s_Data.m_ResultBuffer->SetDrawBuffer(0);

    DrawSkybox("night");
    DrawPhysicsDebug();
    DrawDebugVisualizations();
    ParticleRenderer::UpdateAndRender(dt);
    DrawDebug2D();

    s_Data.m_ResultBuffer->SetDrawBuffers();
    s_Data.m_ResultBuffer->UnBind();
    gabprofiler_end(&profileScope);
  }
  {
    GABProfilerScope profileScope = gabprofiler_begin("SCENE RESULT PASS");

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    if (s_Data.m_SceneState == OpenGLRendererData::SceneState::Edit)
    {
      s_Data.m_PostProcessBuffer->Bind();
      DrawFramebuffer(s_Data.m_ResultBuffer->GetColorAttachmentRendererID(), true);
      s_Data.m_PostProcessBuffer->UnBind();
      DrawEditorFrameBuffer(s_Data.m_PostProcessBuffer->GetColorAttachmentRendererID());
    }
    else
    {
      // The game view does not need an intermediate texture for ImGui. Apply
      // the postprocess directly to the swapchain framebuffer.
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glViewport(0, 0, Window::GetWidth(), Window::GetHeight());
      DrawFramebuffer(s_Data.m_ResultBuffer->GetColorAttachmentRendererID(), true);
    }
    gabprofiler_end(&profileScope);
  }
  {
    GABProfilerScope profileScope = gabprofiler_begin("UI PASS");

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, Window::GetWidth(), Window::GetHeight());
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (advanceSimulation)
    {
      const float uiScale = GetResolutionUIScale();
      BeginScene();
      if (s_Data.m_SceneState == OpenGLRendererData::SceneState::Play) DrawInteractionLabels();
      DrawText(FontManager::GetFont("dpcomic"), "FPS: " + std::to_string(dt.GetFPS()),
        glm::vec2(100.0f, 50.0f) * uiScale, 0.5f * uiScale, glm::vec4(1.0f));
      EndScene();
    }
    gabprofiler_end(&profileScope);
  }
  gabprofiler_end_frame();
}

bool OpenGLRenderer::IsRenderingEditor()
{
  return s_Data.m_SceneState == OpenGLRendererData::SceneState::Edit;
}

void OpenGLRenderer::DrawNativeSceneOverlay(DeltaTime& dt, bool advanceSimulation, bool renderForEditor)
{
  if (renderForEditor)
  {
    DrawEditorFrameBuffer(RenderBackend::Get().GetEditorTextureID());
  }
  else if (advanceSimulation)
  {
    const float uiScale = GetResolutionUIScale();
    BeginScene();
    DrawInteractionLabels();
    DrawText(nullptr, "FPS: " + std::to_string(dt.GetFPS()),
      glm::vec2(100.0f, 50.0f) * uiScale, 0.5f * uiScale, glm::vec4(1.0f));
    EndScene();
  }
}

void OpenGLRenderer::DrawLoadingScreen()
{
  PrepareScreenUI(glm::vec4(0.008f, 0.012f, 0.025f, 1.0f));

  const float time = static_cast<float>(glfwGetTime());
  const int dotCount = static_cast<int>(time * 2.5f) % 4;
  std::string label = "LOADING";
  label.append(static_cast<size_t>(dotCount), '.');
  const float pulse = 0.72f + std::sin(time * 3.0f) * 0.18f;
  const float width = static_cast<float>(Window::GetWidth());
  const float height = static_cast<float>(Window::GetHeight());
  const float uiScale = GetResolutionUIScale();

  BeginScene();
  DrawText(RenderBackend::Capabilities().OpenGLContext ? FontManager::GetFont("dpcomic") : nullptr, label,
    glm::vec2(width * 0.5f, height * 0.5f), 0.82f * uiScale,
    glm::vec4(0.72f, 0.86f, 1.0f, pulse));
  DrawQuad(
    glm::vec2(width * 0.5f + std::sin(time * 1.8f) * 55.0f * uiScale, height * 0.44f),
    glm::vec2(68.0f, 3.0f) * uiScale, 0.0f,
    glm::vec4(0.28f, 0.58f, 0.92f, 0.65f));
  EndScene();
}

void OpenGLRenderer::DrawScreenOverlay(float opacity, const glm::vec3& color)
{
  opacity = std::clamp(opacity, 0.0f, 1.0f);
  if (opacity <= 0.0f) return;

  PrepareScreenUI({}, false);

  BeginScene();
  DrawQuad(
    glm::vec2(Window::GetWidth() * 0.5f, Window::GetHeight() * 0.5f),
    glm::vec2(Window::GetWidth(), Window::GetHeight()),
    0.0f,
    glm::vec4(color, opacity));
  EndScene();
}

void OpenGLRenderer::PrepareScreenUI(const glm::vec4& clearColor, bool clear)
{
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glViewport(0, 0, Window::GetWidth(), Window::GetHeight());
  if (clear)
  {
    glClearColor(clearColor.r, clearColor.g, clearColor.b, clearColor.a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  }
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void OpenGLRenderer::SwitchRenderState()
{
  if (s_Data.m_SceneState == OpenGLRendererData::SceneState::Edit)
  {
    Camera::SetMode(CameraMode::PLAYER);
    Window::SetCursorVisible(false);
    s_Data.m_SceneState = OpenGLRendererData::SceneState::Play;
  }
  else if (s_Data.m_SceneState == OpenGLRendererData::SceneState::Play)
  {
    Camera::SetMode(CameraMode::ORBITAL);
    Window::SetCursorVisible(true);
    s_Data.m_SceneState = OpenGLRendererData::SceneState::Edit;
  }
}

void OpenGLRenderer::SetFullscreen(const std::string& sound, bool windowed)
{
  Window::SetFullscreen(windowed);

  auto width = Window::GetWidth();
  auto height = Window::GetHeight();

  glm::vec2 newResolution = { width, height };
  if (RenderBackend::Capabilities().NativeSceneRenderer)
  {
    Camera::SetViewportSize(width, height);
    AudioManager::PlaySound(sound);
    return;
  }
  s_Data.m_ResolutionUniformBuffer->SetData(&newResolution, sizeof(glm::vec2));

  s_Data.m_GeometryBuffer->Resize(width, height);
  s_Data.m_BloomBuffer->Resize(width, height);
  s_Data.m_ResultBuffer->Resize(width, height);
  s_Data.m_PostProcessBuffer->Resize(width, height);
  s_Data.m_ResultBuffer->AttachExternalDepthTexture(
    s_Data.m_GeometryBuffer->GetDepthAttachmentRendererID());
  Camera::SetViewportSize(width, height);
  AudioManager::PlaySound(sound);
}

void OpenGLRenderer::ApplyDisplaySettings()
{
  Window::SetWindowMode(
    Settings::GetWindowMode(),
    Settings::GetWindowWidth(),
    Settings::GetWindowHeight());
  Window::SetVSync(Settings::GetVSync());

  const uint32_t width = Window::GetWidth();
  const uint32_t height = Window::GetHeight();
  if (RenderBackend::Capabilities().NativeSceneRenderer)
  {
    Camera::SetViewportSize(width, height);
    return;
  }
  const glm::vec2 resolution = {width, height};
  s_Data.m_ResolutionUniformBuffer->SetData(&resolution, sizeof(glm::vec2));
  s_Data.m_GeometryBuffer->Resize(width, height);
  s_Data.m_ResultBuffer->Resize(width, height);
  s_Data.m_PostProcessBuffer->Resize(width, height);
  s_Data.m_BloomBuffer->Resize(width, height);
  s_Data.m_ResultBuffer->AttachExternalDepthTexture(
    s_Data.m_GeometryBuffer->GetDepthAttachmentRendererID());
  Camera::SetViewportSize(width, height);
}

void OpenGLRenderer::ApplyGraphicsSettings()
{
  if (!RenderBackend::Capabilities().OpenGLContext) return;

  const RenderEffectSettings effects = RenderBackend::GetEffectSettings();
  const GraphicsQuality quality = effects.ShadowQuality;
  const auto qualityValue = static_cast<uint32_t>(quality);
  if (qualityValue == s_Data.m_AppliedShadowQuality &&
      s_Data.m_DirectShadowBuffer && s_Data.m_OmniDirectShadowBuffer)
    return;

  const uint32_t directResolution = effects.DirectionalShadowResolution();
  const uint32_t omniResolution = effects.PointShadowResolution();
  float filterSize = 2.0f;
  float randomRadius = 1.5f;

  if (quality == GraphicsQuality::Medium)
  {
    filterSize = 3.0f;
    randomRadius = 2.0f;
  }
  else if (quality == GraphicsQuality::High)
  {
    filterSize = 4.0f;
    randomRadius = 2.5f;
  }
  s_Data.m_DirectShadowBuffer = DirectShadowBuffer::Create(
    directResolution, directResolution, 16.0f, filterSize, randomRadius);
  s_Data.m_OmniDirectShadowBuffer = OmniDirectShadowBuffer::Create(
    omniResolution, omniResolution, OpenGLRendererData::MaxShadowedPointLights);
  s_Data.m_PointShadowCache = {};
  s_Data.m_AppliedShadowQuality = qualityValue;
}

void OpenGLRenderer::DrawPhysicsDebug()
{
  if (!RenderBackend::DebugSettings().Physics) return;
  if (RenderBackend::Get().DrawPhysicsDebug()) return;
  if (!s_Data.s_Shaders.PhysicsDebugShader) return;

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);
  glDepthMask(GL_FALSE);
  glDisable(GL_CULL_FACE);
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  glLineWidth(2.0f);

  s_Data.s_Shaders.PhysicsDebugShader->Bind();
  s_Data.s_Shaders.PhysicsDebugShader->SetVec4("u_Color", glm::vec4(0.15f, 1.0f, 0.35f, 1.0f));
  ModelManager::BindAllInstanceTransforms();
  glBindVertexArray(ModelManager::GetModelsVAO());

  for (const auto& [modelName, commandIndices] : s_Data.m_ModelDrawCommandIndices)
  {
    const auto model = ModelManager::GetModel(modelName);
    if (!model || model->GetPhysXMeshType() != MeshType::CONVEXMESH) continue;

    const auto instanceCount = static_cast<GLsizei>(model->m_InstanceTransforms.size());
    for (const size_t commandIndex : commandIndices)
    {
      const auto& command = s_Data.m_DrawCommands[commandIndex];
      glDrawElementsInstancedBaseVertexBaseInstance(
        GL_TRIANGLES,
        static_cast<GLsizei>(command.count),
        GL_UNSIGNED_INT,
        reinterpret_cast<const void*>(static_cast<uintptr_t>(command.firstIndex) * sizeof(GLuint)),
        instanceCount,
        command.baseVertex,
        model->m_InstanceBase);
    }
  }

  glBindVertexArray(0);
  s_Data.s_Shaders.PhysicsDebugShader->UnBind();

  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  glEnable(GL_CULL_FACE);
  glDepthMask(GL_TRUE);
  glDepthFunc(GL_LESS);
  glLineWidth(s_Data.LineWidth);
}

void OpenGLRenderer::DrawDebugVisualizations()
{
  const RenderDebugSettings& debug = RenderBackend::DebugSettings();
  if (!debug.Physics && !debug.Lights && !debug.CullingBounds) return;

  const bool nativeDebugLines = RenderBackend::Get().BeginDebugLines();
  if (!nativeDebugLines)
  {
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_FALSE);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    BeginScene();
  }

  if (debug.Physics)
  {
    constexpr glm::vec4 controllerColor(0.15f, 0.8f, 1.0f, 0.9f);
    for (const std::string& modelName : ModelManager::GetModelNames())
    {
      const auto model = ModelManager::GetModel(modelName);
      PxController* controller = model ? model->GetController() : nullptr;
      if (!controller || controller->getType() != PxControllerShapeType::eCAPSULE)
        continue;

      auto* capsuleController = static_cast<PxCapsuleController*>(controller);
      const PxExtendedVec3 position = controller->getPosition();
      const PxVec3 upDirection = controller->getUpDirection();
      DrawWireCapsule(
        glm::vec3(
          static_cast<float>(position.x),
          static_cast<float>(position.y),
          static_cast<float>(position.z)),
        capsuleController->getRadius(),
        capsuleController->getHeight(),
        glm::vec3(upDirection.x, upDirection.y, upDirection.z),
        controllerColor);
    }
  }

  if (debug.CullingBounds)
  {
    const RenderFrustum frustum(Camera::GetViewProjection());
    for (const std::string& modelName : ModelManager::GetModelNames())
    {
      const auto model = ModelManager::GetModel(modelName);
      if (!model || !model->m_IsRendered) continue;

      for (const glm::mat4& transform : model->m_InstanceTransforms)
      {
        const WorldBoundingSphere sphere = CalculateWorldBoundingSphere(*model, transform);
        const bool visible = frustum.IntersectsSphere(sphere.center, sphere.radius);
        DrawWireSphere(sphere.center, sphere.radius,
          visible ? glm::vec4(0.15f, 1.0f, 0.3f, 0.8f) : glm::vec4(1.0f, 0.2f, 0.15f, 0.8f));
      }
    }
  }

  if (debug.Lights)
  {
    for (const SceneLight& light : SceneManager::GetLights())
    {
      const glm::vec4 color(light.color, 0.9f);
      if (light.type == LightType::DIRECT)
      {
        const glm::vec3 origin = Camera::GetPosition() + Camera::GetForwardDirection() * 8.0f;
        const glm::vec3 direction = glm::length(light.rotation) > 0.0001f
          ? glm::normalize(light.rotation)
          : glm::vec3(0.0f, -1.0f, 0.0f);
        DrawWireSphere(origin, 0.5f, color);
        DrawLine(origin, origin + direction * 12.0f, color);
      }
      else if (light.type == LightType::POINT)
      {
		DrawWireSphere(light.position, 0.5f, color);
		DrawWireSphere(light.position, OpenGLRendererData::PointShadowRadius, glm::vec4(light.color, 0.2f), 32);
		DrawLine(light.position - glm::vec3(1.0f, 0.0f, 0.0f), light.position + glm::vec3(1.0f, 0.0f, 0.0f), color);
		DrawLine(light.position - glm::vec3(0.0f, 1.0f, 0.0f), light.position + glm::vec3(0.0f, 1.0f, 0.0f), color);
		DrawLine(light.position - glm::vec3(0.0f, 0.0f, 1.0f), light.position + glm::vec3(0.0f, 0.0f, 1.0f), color);
      }
      else
      {
        const glm::vec3 direction = glm::length(light.rotation) > 0.0001f
          ? glm::normalize(light.rotation)
          : glm::vec3(0.0f, -1.0f, 0.0f);
        const glm::vec3 fallbackUp = std::abs(glm::dot(direction, glm::vec3(0.0f, 1.0f, 0.0f))) > 0.98f
          ? glm::vec3(1.0f, 0.0f, 0.0f)
          : glm::vec3(0.0f, 1.0f, 0.0f);
        const glm::vec3 right = glm::normalize(glm::cross(direction, fallbackUp));
        const glm::vec3 up = glm::normalize(glm::cross(right, direction));
        const glm::vec3 end = light.position + direction * 8.0f;
        constexpr int segments = 20;
        for (int i = 0; i < segments; ++i)
        {
	        constexpr float coneRadius = 3.0f;
	        const float angle0 = glm::two_pi<float>() * static_cast<float>(i) / static_cast<float>(segments);
			const float angle1 = glm::two_pi<float>() * static_cast<float>(i + 1) / static_cast<float>(segments);
			const glm::vec3 p0 = end + (right * std::cos(angle0) + up * std::sin(angle0)) * coneRadius;
			const glm::vec3 p1 = end + (right * std::cos(angle1) + up * std::sin(angle1)) * coneRadius;
			DrawLine(p0, p1, color);
			if (i % 5 == 0) DrawLine(light.position, p0, color);
        }
        DrawWireSphere(light.position, 0.4f, color);
        DrawLine(light.position, end, color);
      }
    }
  }

  if (nativeDebugLines)
    RenderBackend::Get().EndDebugLines();
  else
  {
    EndScene();
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);
    glEnable(GL_CULL_FACE);
    glDisable(GL_BLEND);
  }
}

static bool BrowseForModelFile(char* destination, size_t destinationSize)
{
  if (!destination || destinationSize == 0) return false;
  const std::string selectedPath = NativeFileDialog::OpenModelFile();
  if (selectedPath.empty()) return false;
  const size_t copyLength = std::min(destinationSize - 1, selectedPath.size());
  std::memcpy(destination, selectedPath.data(), copyLength);
  destination[copyLength] = '\0';
  return true;
}

void OpenGLRenderer::DebugDrawQuad2D(const glm::vec2& position, const glm::vec2& size,
  const glm::vec4& color, float rotation, bool outline)
{
  Debug2DCommand command;
  command.type = Debug2DCommand::Type::Quad;
  command.position = position;
  command.size = glm::max(size, glm::vec2(0.0f));
  command.color = color;
  command.rotation = rotation;
  command.outline = outline;
  s_Data.m_Debug2DCommands.push_back(std::move(command));
}

void OpenGLRenderer::DebugDrawText2D(const std::string& text, const glm::vec2& position,
  float size, const glm::vec4& color)
{
  if (text.empty()) return;
  Debug2DCommand command;
  command.type = Debug2DCommand::Type::Text;
  command.position = position;
  command.color = color;
  command.textSize = std::max(size, 0.01f);
  command.text = text;
  s_Data.m_Debug2DCommands.push_back(std::move(command));
}

void OpenGLRenderer::DrawDebug2D()
{
  if (!RenderBackend::DebugSettings().Debug2D)
  {
    s_Data.m_Debug2DCommands.clear();
    return;
  }

  const bool openGL = RenderBackend::Capabilities().OpenGLContext;
  GLboolean blendWasEnabled = GL_FALSE;
  GLboolean depthWasEnabled = GL_FALSE;
  GLboolean cullWasEnabled = GL_FALSE;
  GLboolean previousDepthMask = GL_TRUE;
  GLint previousDepthFunc = GL_LESS;
  GLint previousBlendSrcRGB = GL_ONE, previousBlendDstRGB = GL_ZERO;
  GLint previousBlendSrcAlpha = GL_ONE, previousBlendDstAlpha = GL_ZERO;
  if (openGL)
  {
    blendWasEnabled = glIsEnabled(GL_BLEND);
    depthWasEnabled = glIsEnabled(GL_DEPTH_TEST);
    cullWasEnabled = glIsEnabled(GL_CULL_FACE);
    glGetBooleanv(GL_DEPTH_WRITEMASK, &previousDepthMask);
    glGetIntegerv(GL_DEPTH_FUNC, &previousDepthFunc);
    glGetIntegerv(GL_BLEND_SRC_RGB, &previousBlendSrcRGB);
    glGetIntegerv(GL_BLEND_DST_RGB, &previousBlendDstRGB);
    glGetIntegerv(GL_BLEND_SRC_ALPHA, &previousBlendSrcAlpha);
    glGetIntegerv(GL_BLEND_DST_ALPHA, &previousBlendDstAlpha);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  }
  BeginScene();
  Set3D(false);

  const auto drawOutline = [](const glm::vec2& position, const glm::vec2& size,
    float rotation, const glm::vec4& color)
  {
    constexpr float thickness = 2.0f;
    const float radians = glm::radians(rotation);
    const glm::vec2 right(std::cos(radians), std::sin(radians));
    const glm::vec2 up(-std::sin(radians), std::cos(radians));
    const glm::vec2 halfSize = size * 0.5f;
    DrawQuad(position + up * halfSize.y, glm::vec2(size.x, thickness), rotation, color);
    DrawQuad(position - up * halfSize.y, glm::vec2(size.x, thickness), rotation, color);
    DrawQuad(position + right * halfSize.x, glm::vec2(thickness, size.y), rotation, color);
    DrawQuad(position - right * halfSize.x, glm::vec2(thickness, size.y), rotation, color);
  };

  // A small built-in marker makes it immediately clear that the layer is active.
  DrawQuad(glm::vec2(141.0f, 37.0f), glm::vec2(250.0f, 42.0f), 0.0f,
    glm::vec4(0.02f, 0.04f, 0.07f, 0.78f));
  drawOutline(glm::vec2(141.0f, 37.0f), glm::vec2(250.0f, 42.0f), 0.0f,
    glm::vec4(0.2f, 0.85f, 1.0f, 0.9f));
  DrawText(openGL ? FontManager::GetFont("dpcomic") : nullptr, "2D DEBUG RENDER",
    glm::vec2(141.0f, 37.0f), 0.28f, glm::vec4(0.45f, 0.92f, 1.0f, 1.0f));

  for (const auto& command : s_Data.m_Debug2DCommands)
  {
    if (command.type == Debug2DCommand::Type::Text)
    {
      DrawText(openGL ? FontManager::GetFont("dpcomic") : nullptr, command.text, command.position,
        command.textSize, command.color);
    }
    else if (command.outline)
    {
      drawOutline(command.position, command.size, command.rotation, command.color);
    }
    else
    {
      DrawQuad(command.position, command.size, command.rotation, command.color);
    }
  }

  EndScene();
  s_Data.m_Debug2DCommands.clear();
  if (openGL)
  {
    glDepthMask(previousDepthMask);
    glDepthFunc(previousDepthFunc);
    glBlendFuncSeparate(previousBlendSrcRGB, previousBlendDstRGB,
      previousBlendSrcAlpha, previousBlendDstAlpha);
    if (blendWasEnabled) glEnable(GL_BLEND); else glDisable(GL_BLEND);
    if (depthWasEnabled) glEnable(GL_DEPTH_TEST); else glDisable(GL_DEPTH_TEST);
    if (cullWasEnabled) glEnable(GL_CULL_FACE); else glDisable(GL_CULL_FACE);
  }
}

void OpenGLRenderer::BeginScene()
{
	s_Data.m_CameraBuffer.ViewProjection = Camera::GetViewProjection();
	s_Data.m_CameraBuffer.OrtoProjection = Camera::GetOrtoProjection();
	s_Data.m_CameraBuffer.NonRotViewProjection = Camera::GetNonRotationViewProjection();
	s_Data.m_CameraBuffer.CameraPos = Camera::GetPosition();
	s_Data.m_CameraUniformBuffer->SetData(&s_Data.m_CameraBuffer, sizeof(CameraData));

	StartBatch();
}

void OpenGLRenderer::EndScene()
{
	Flush();
}

void OpenGLRenderer::Flush()
{
	if (s_Data.QuadIndexCount)
	{
		auto dataSize = static_cast<uint32_t>(reinterpret_cast<uint8_t *>(s_Data.QuadVertexBufferPtr) - reinterpret_cast<uint8_t *>(s_Data.QuadVertexBufferBase));
		s_Data.QuadVertexBuffer->SetData(s_Data.QuadVertexBufferBase, dataSize);

		for (uint32_t i = 0; i < s_Data.TextureSlotIndex; i++) s_Data.TextureSlots[i]->Bind(i);

		s_Data.s_Shaders.QuadShader->Bind();
		s_Data.s_Shaders.QuadShader->SetBool("u_Is3D", s_Data.Is3D);
		DrawIndexed(s_Data.QuadVertexArray, s_Data.QuadIndexCount);
	}
	if (s_Data.LineVertexCount)
	{
		auto dataSize = static_cast<uint32_t>(reinterpret_cast<uint8_t *>(s_Data.LineVertexBufferPtr) - reinterpret_cast<uint8_t *>(s_Data.LineVertexBufferBase));
		s_Data.LineVertexBuffer->SetData(s_Data.LineVertexBufferBase, dataSize);

		s_Data.s_Shaders.LineShader->Bind();
		SetLineWidth(s_Data.LineWidth);
		DrawLines(s_Data.LineVertexArray, s_Data.LineVertexCount);
	}
}

void OpenGLRenderer::StartBatch()
{
	s_Data.QuadIndexCount = 0;
	s_Data.QuadVertexBufferPtr = s_Data.QuadVertexBufferBase;
	s_Data.TextureSlotIndex = 1;

	s_Data.LineVertexCount = 0;
	s_Data.LineVertexBufferPtr = s_Data.LineVertexBufferBase;
}

void OpenGLRenderer::NextBatch()
{
	Flush();
	StartBatch();
}

void OpenGLRenderer::DrawLine(const glm::vec3& p0, const glm::vec3& p1, const glm::vec4& color, int entityID)
{
	if (s_Data.LineVertexCount + 2 > OpenGLRendererData::MaxVertices) NextBatch();

	s_Data.LineVertexBufferPtr->Position = p0;
	s_Data.LineVertexBufferPtr->Color = color;
	s_Data.LineVertexBufferPtr->EntityID = entityID;
	s_Data.LineVertexBufferPtr++;

	s_Data.LineVertexBufferPtr->Position = p1;
	s_Data.LineVertexBufferPtr->Color = color;
	s_Data.LineVertexBufferPtr->EntityID = entityID;
	s_Data.LineVertexBufferPtr++;

	s_Data.LineVertexCount += 2;
}

void OpenGLRenderer::DrawQuad(const glm::vec3& position, const glm::vec3& size, const glm::vec3& rotation, const glm::vec4& color)
{
  glm::mat4 transform = glm::translate(glm::mat4(1.0f), position)
    * glm::rotate(glm::mat4(1.0f), glm::radians(rotation.x), glm::vec3(1, 0, 0))
    * glm::rotate(glm::mat4(1.0f), glm::radians(rotation.y), glm::vec3(0, 1, 0))
    * glm::rotate(glm::mat4(1.0f), glm::radians(rotation.z), glm::vec3(0, 0, 1))
    * glm::scale(glm::mat4(1.0f), size);

	DrawQuad(transform, color);
}

void OpenGLRenderer::DrawQuad(const glm::vec2& position, const glm::vec2& size, float rotation, const glm::vec4& color)
{
	glm::mat4 transform = glm::translate(glm::mat4(1.0f), glm::vec3(position, 0.0f))
		* glm::rotate(glm::mat4(1.0f), glm::radians(rotation), { 0.0f, 0.0f, 1.0f })
		* glm::scale(glm::mat4(1.0f), { size.x, size.y, 1.0f });

	DrawQuad(transform, color);
}

void OpenGLRenderer::DrawQuad(const glm::mat4& transform, const glm::vec4& color, int entityID)
{
	if (s_Data.QuadIndexCount >= OpenGLRendererData::MaxIndices) NextBatch();

  auto position = glm::vec3(transform[3]);

  float sizeX = glm::length(glm::vec3(transform[0]));
  float sizeY = glm::length(glm::vec3(transform[1]));

  glm::vec3 cameraRight = Camera::GetRightDirection();
  glm::vec3 cameraUp = Camera::GetUpDirection();

	for (size_t i = 0; i < OpenGLRendererData::quadVertexCount; i++)
	{
		constexpr float textureIndex = 0.0f;
		if(s_Data.Is3D)
		{
	      glm::vec3 worldPos =
	          position +
	          cameraRight * (s_Data.QuadVertexPositions[i].x * sizeX) +
	          cameraUp    * (s_Data.QuadVertexPositions[i].y * sizeY);

	      s_Data.QuadVertexBufferPtr->Position = worldPos;
		}
		else s_Data.QuadVertexBufferPtr->Position = transform * s_Data.QuadVertexPositions[i];

	    s_Data.QuadVertexBufferPtr->Color = color;
	    s_Data.QuadVertexBufferPtr->TexCoord = OpenGLRendererData::tex3DCoords[i];
	    s_Data.QuadVertexBufferPtr->TexIndex = textureIndex;
	    s_Data.QuadVertexBufferPtr->TilingFactor = OpenGLRendererData::tilingFactor;
	    s_Data.QuadVertexBufferPtr->EntityID = entityID;
	    s_Data.QuadVertexBufferPtr++;
	}

	s_Data.QuadIndexCount += 6;
}

void OpenGLRenderer::DrawQuad(const glm::vec3& position, const glm::vec3& size, const glm::vec3& rotation, const std::shared_ptr<Texture>& texture, const glm::vec4& tintColor)
{
  glm::mat4 transform = glm::translate(glm::mat4(1.0f), position)
    * glm::rotate(glm::mat4(1.0f), glm::radians(rotation.x), glm::vec3(1, 0, 0))
    * glm::rotate(glm::mat4(1.0f), glm::radians(rotation.y), glm::vec3(0, 1, 0))
    * glm::rotate(glm::mat4(1.0f), glm::radians(rotation.z), glm::vec3(0, 0, 1))
    * glm::scale(glm::mat4(1.0f), size);

	DrawQuad(transform, texture, tintColor, 1.0f);
}

void OpenGLRenderer::DrawQuad(const glm::vec2& position, const glm::vec2& size, float rotation, const std::shared_ptr<Texture>& texture, const glm::vec4& tintColor)
{
	glm::mat4 transform = glm::translate(glm::mat4(1.0f), glm::vec3(position, 0.0f))
		* glm::rotate(glm::mat4(1.0f), glm::radians(rotation), { 0.0f, 0.0f, 1.0f })
		* glm::scale(glm::mat4(1.0f), { size.x, size.y, 1.0f });

	DrawQuad(transform, texture, tintColor, 1.0f);
}

void OpenGLRenderer::DrawQuad(const glm::mat4& transform, const std::shared_ptr<Texture>& texture, const glm::vec4& tintColor, float tilingFactor, int entityID)
{
	if (s_Data.QuadIndexCount >= OpenGLRendererData::MaxIndices) NextBatch();

	float textureIndex = 0.0f;
	for (uint32_t i = 1; i < s_Data.TextureSlotIndex; i++)
	{
		if (*s_Data.TextureSlots[i] == *texture)
		{
			textureIndex = static_cast<float>(i);
			break;
		}
	}

	if (textureIndex == 0.0f)
	{
		if (s_Data.TextureSlotIndex >= OpenGLRendererData::MaxTextureSlots) NextBatch();

		textureIndex = static_cast<float>(s_Data.TextureSlotIndex);
		s_Data.TextureSlots[s_Data.TextureSlotIndex] = texture;
		s_Data.TextureSlotIndex++;
	}

	for (size_t i = 0; i < s_Data.quadVertexCount; i++)
	{
		s_Data.QuadVertexBufferPtr->Position = transform * s_Data.QuadVertexPositions[i];
		s_Data.QuadVertexBufferPtr->Color = tintColor;
		s_Data.QuadVertexBufferPtr->TexCoord = s_Data.tex2DCoords[i];
		s_Data.QuadVertexBufferPtr->TexIndex = textureIndex;
		s_Data.QuadVertexBufferPtr->TilingFactor = s_Data.tilingFactor;
		s_Data.QuadVertexBufferPtr->EntityID = entityID;
		s_Data.QuadVertexBufferPtr++;
	}

	s_Data.QuadIndexCount += 6;
}

void OpenGLRenderer::DrawQuadContour(const glm::vec3& position, const glm::vec2& size, const glm::vec4& color, int entityID)
{
	glm::vec3 p0 = glm::vec3(position.x - size.x * 0.5f, position.y - size.y * 0.5f, position.z);
	glm::vec3 p1 = glm::vec3(position.x + size.x * 0.5f, position.y - size.y * 0.5f, position.z);
	glm::vec3 p2 = glm::vec3(position.x + size.x * 0.5f, position.y + size.y * 0.5f, position.z);
	glm::vec3 p3 = glm::vec3(position.x - size.x * 0.5f, position.y + size.y * 0.5f, position.z);

	DrawLine(p0, p1, color, entityID);
	DrawLine(p1, p2, color, entityID);
	DrawLine(p2, p3, color, entityID);
	DrawLine(p3, p0, color, entityID);
}

void OpenGLRenderer::DrawQuadContour(const glm::vec2& position, const glm::vec2& size, float rotation, const glm::vec4& color, int entityID)
{
	glm::mat4 transform = glm::translate(glm::mat4(1.0f), glm::vec3(position,0.0f))
		* glm::rotate(glm::mat4(1.0f), glm::radians(rotation), { 0.0f, 0.0f, 1.0f })
		* glm::scale(glm::mat4(1.0f), { size.x, size.y, 1.0f });

	DrawQuadContour(transform, color);
}

void OpenGLRenderer::DrawQuadContour(const glm::mat4& transform, const glm::vec4& color, int entityID)
{
	glm::vec3 lineVertices[4];
	for (size_t i = 0; i < 4; i++) lineVertices[i] = transform * s_Data.QuadVertexPositions[i];

	DrawLine(lineVertices[0], lineVertices[1], color, entityID);
	DrawLine(lineVertices[1], lineVertices[2], color, entityID);
	DrawLine(lineVertices[2], lineVertices[3], color, entityID);
	DrawLine(lineVertices[3], lineVertices[0], color, entityID);
}

void OpenGLRenderer::DrawCube(const glm::vec3& position, const glm::vec3& size, const std::shared_ptr<Texture>& texture, const glm::vec4& tintColor, int entityID)
{
  Set3D(true);

  GLint prevFrontFace;
  glGetIntegerv(GL_FRONT_FACE, &prevFrontFace);

  glEnable(GL_CULL_FACE);
  glCullFace(GL_FRONT);
  glFrontFace(GL_CCW); // Only cubes use counter-clockwise winding

  glm::vec2 xy = { size.x, size.y };
	glm::vec2 yz = { size.z, size.y };
	glm::vec2 xz = { size.x, size.z };

	float halfX = size.x / 2.0f;
	float halfY = size.y / 2.0f;
	float halfZ = size.z / 2.0f;

	// FRONT (+Z)
	DrawQuad(
		glm::translate(glm::mat4(1.0f), position + glm::vec3(0.0f, 0.0f, +halfZ)) *
		glm::scale(glm::mat4(1.0f), glm::vec3(xy, 1.0f)),texture,tintColor);

	// BACK (-Z)
	DrawQuad(
		glm::translate(glm::mat4(1.0f), position + glm::vec3(0.0f, 0.0f, -halfZ)) *
		glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), { 0, 1, 0 }) *
		glm::scale(glm::mat4(1.0f), glm::vec3(xy, 1.0f)),texture,tintColor);

	// LEFT (-X)
	DrawQuad(
		glm::translate(glm::mat4(1.0f), position + glm::vec3(-halfX, 0.0f, 0.0f)) *
		glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), { 0, 1, 0 }) *
		glm::scale(glm::mat4(1.0f), glm::vec3(yz, 1.0f)),texture,tintColor);

	// RIGHT (+X)
	DrawQuad(
		glm::translate(glm::mat4(1.0f), position + glm::vec3(+halfX, 0.0f, 0.0f)) *
		glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), { 0, 1, 0 }) *
		glm::scale(glm::mat4(1.0f), glm::vec3(yz, 1.0f)),texture,tintColor);

	// TOP (+Y)
	DrawQuad(
		glm::translate(glm::mat4(1.0f), position + glm::vec3(0.0f, +halfY, 0.0f)) *
		glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), { 1, 0, 0 }) *
		glm::scale(glm::mat4(1.0f), glm::vec3(xz, 1.0f)),texture,tintColor);

	// BOTTOM (-Y)
	DrawQuad(
		glm::translate(glm::mat4(1.0f), position + glm::vec3(0.0f, -halfY, 0.0f)) *
		glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), { 1, 0, 0 }) *
		glm::scale(glm::mat4(1.0f), glm::vec3(xz, 1.0f)),texture,tintColor);

  glFrontFace(prevFrontFace);
  glDisable(GL_CULL_FACE);

  Set3D(false);
}

void OpenGLRenderer::DrawCube(const glm::vec3& position, const glm::vec3& size, const glm::vec4& color, int entityID)
{
	Set3D(true);

	GLint prevFrontFace;
	glGetIntegerv(GL_FRONT_FACE, &prevFrontFace);

	glEnable(GL_CULL_FACE);
	glCullFace(GL_FRONT);
	glFrontFace(GL_CCW); // Only cubes use counter-clockwise winding

	glm::vec2 xy = { size.x, size.y };
	glm::vec2 yz = { size.z, size.y };
	glm::vec2 xz = { size.x, size.z };

	float halfX = size.x / 2.0f;
	float halfY = size.y / 2.0f;
	float halfZ = size.z / 2.0f;

	// FRONT (+Z)
	DrawQuad(
		glm::translate(glm::mat4(1.0f), position + glm::vec3(0.0f, 0.0f, +halfZ)) *
		glm::scale(glm::mat4(1.0f), glm::vec3(xy, 1.0f)), color);

	// BACK (-Z)
	DrawQuad(
		glm::translate(glm::mat4(1.0f), position + glm::vec3(0.0f, 0.0f, -halfZ)) *
		glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), { 0, 1, 0 }) *
		glm::scale(glm::mat4(1.0f), glm::vec3(xy, 1.0f)), color);

	// LEFT (-X)
	DrawQuad(
		glm::translate(glm::mat4(1.0f), position + glm::vec3(-halfX, 0.0f, 0.0f)) *
		glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), { 0, 1, 0 }) *
		glm::scale(glm::mat4(1.0f), glm::vec3(yz, 1.0f)), color);

	// RIGHT (+X)
	DrawQuad(
		glm::translate(glm::mat4(1.0f), position + glm::vec3(+halfX, 0.0f, 0.0f)) *
		glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), { 0, 1, 0 }) *
		glm::scale(glm::mat4(1.0f), glm::vec3(yz, 1.0f)), color);

	// TOP (+Y)
	DrawQuad(
		glm::translate(glm::mat4(1.0f), position + glm::vec3(0.0f, +halfY, 0.0f)) *
		glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), { 1, 0, 0 }) *
		glm::scale(glm::mat4(1.0f), glm::vec3(xz, 1.0f)), color);

	// BOTTOM (-Y)
	DrawQuad(
		glm::translate(glm::mat4(1.0f), position + glm::vec3(0.0f, -halfY, 0.0f)) *
		glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), { 1, 0, 0 }) *
		glm::scale(glm::mat4(1.0f), glm::vec3(xz, 1.0f)), color);

  glFrontFace(prevFrontFace);
  glDisable(GL_CULL_FACE);

  Set3D(false);
}

void OpenGLRenderer::DrawCubeContour(const glm::vec3& position, const glm::vec3& size, const glm::vec4& color, int entityID)
{
  glm::vec3 half = size * 0.5f;

  // 8 cube vertices relative to center position
  glm::vec3 v0 = position + glm::vec3(-half.x, -half.y, -half.z); // left bottom back
  glm::vec3 v1 = position + glm::vec3( half.x, -half.y, -half.z); // right bottom back
  glm::vec3 v2 = position + glm::vec3( half.x,  half.y, -half.z); // right top back
  glm::vec3 v3 = position + glm::vec3(-half.x,  half.y, -half.z); // left top back

  glm::vec3 v4 = position + glm::vec3(-half.x, -half.y,  half.z); // left bottom front
  glm::vec3 v5 = position + glm::vec3( half.x, -half.y,  half.z); // right bottom front
  glm::vec3 v6 = position + glm::vec3( half.x,  half.y,  half.z); // right top front
  glm::vec3 v7 = position + glm::vec3(-half.x,  half.y,  half.z); // left top front

  // Bottom square
  DrawLine(v0, v1, color, entityID);
  DrawLine(v1, v2, color, entityID);
  DrawLine(v2, v3, color, entityID);
  DrawLine(v3, v0, color, entityID);

  // Top square
  DrawLine(v4, v5, color, entityID);
  DrawLine(v5, v6, color, entityID);
  DrawLine(v6, v7, color, entityID);
  DrawLine(v7, v4, color, entityID);

  // Vertical edges
  DrawLine(v0, v4, color, entityID);
  DrawLine(v1, v5, color, entityID);
  DrawLine(v2, v6, color, entityID);
  DrawLine(v3, v7, color, entityID);
}

void OpenGLRenderer::DrawFullscreenQuad()
{
  if (s_Data.m_FullscreenQuadVAO == 0)
  {
    float quadVertices[] = {
        // positions        // tex Coords
        -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
         1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
         1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
    };

    glCreateVertexArrays(1, &s_Data.m_FullscreenQuadVAO);
    glCreateBuffers(1, &s_Data.m_FullscreenQuadVBO);

    glNamedBufferData(s_Data.m_FullscreenQuadVBO, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glVertexArrayVertexBuffer(s_Data.m_FullscreenQuadVAO, 0, s_Data.m_FullscreenQuadVBO, 0, 5 * sizeof(float));

    glEnableVertexArrayAttrib(s_Data.m_FullscreenQuadVAO, 0);
    glVertexArrayAttribFormat(s_Data.m_FullscreenQuadVAO, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(s_Data.m_FullscreenQuadVAO, 0, 0);

    glEnableVertexArrayAttrib(s_Data.m_FullscreenQuadVAO, 1);
    glVertexArrayAttribFormat(s_Data.m_FullscreenQuadVAO, 1, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float));
    glVertexArrayAttribBinding(s_Data.m_FullscreenQuadVAO, 1, 0);
  }

  glBindVertexArray(s_Data.m_FullscreenQuadVAO);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glBindVertexArray(0);
}

void OpenGLRenderer::DrawFramebuffer(uint32_t textureID, bool applyPS1Effect)
{
  if (s_Data.m_FramebufferQuadVAO == 0)
  {
    float quadVertices[] = {
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    glCreateVertexArrays(1, &s_Data.m_FramebufferQuadVAO);

    glCreateBuffers(1, &s_Data.m_FramebufferQuadVBO);
    glNamedBufferData(s_Data.m_FramebufferQuadVBO, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glVertexArrayVertexBuffer(s_Data.m_FramebufferQuadVAO, 0, s_Data.m_FramebufferQuadVBO, 0, 4 * sizeof(float));

    glEnableVertexArrayAttrib(s_Data.m_FramebufferQuadVAO, 0);
    glVertexArrayAttribFormat(s_Data.m_FramebufferQuadVAO, 0, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(s_Data.m_FramebufferQuadVAO, 0, 0);

    glEnableVertexArrayAttrib(s_Data.m_FramebufferQuadVAO, 1);
    glVertexArrayAttribFormat(s_Data.m_FramebufferQuadVAO, 1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float));
    glVertexArrayAttribBinding(s_Data.m_FramebufferQuadVAO, 1, 0);
  }

  s_Data.s_Shaders.FramebufferShader->Bind();
  const RenderEffectSettings effects = RenderBackend::GetEffectSettings();
  s_Data.s_Shaders.FramebufferShader->SetInt("u_Texture", 0);
  s_Data.s_Shaders.FramebufferShader->SetBool("u_PS1Effect", applyPS1Effect && effects.PS1Enabled);
  s_Data.s_Shaders.FramebufferShader->SetFloat("u_PS1VirtualHeight", effects.PS1VirtualHeight);
  s_Data.s_Shaders.FramebufferShader->SetFloat("u_PS1ColorLevels", effects.PS1ColorLevels);

  glBindTextureUnit(0, textureID);

  glBindVertexArray(s_Data.m_FramebufferQuadVAO);
  glDrawArrays(GL_TRIANGLES, 0, 6);
  glBindVertexArray(0);
  s_Data.s_Shaders.FramebufferShader->UnBind();
}

void OpenGLRenderer::BakeSkyboxTextures(const std::string& name, const std::shared_ptr<Texture>& cubemap)
{
  Timer timer;

  auto channels = cubemap->GetChannels();
  auto width = cubemap->GetWidth();
  auto height = cubemap->GetHeight();
  auto& pixels = cubemap->GetPixels();

  GLuint rendererID = 0;
  glCreateTextures(GL_TEXTURE_CUBE_MAP, 1, &rendererID);
  if (rendererID == 0)
  {
    gablog_log(LOG_ASSERT, __FILE__, __LINE__, "Failed to create cube map texture!");
    gabdebug_break();
  }

  cubemap->SetRendererID(rendererID);

  GLenum internalFormat = (channels == 4) ? GL_RGBA8 : GL_RGB8;
  GLenum dataFormat     = (channels == 4) ? GL_RGBA  : GL_RGB;

  glTextureStorage2D(rendererID, 1, internalFormat, width, height);

  for (int i = 0; i < 6; ++i)
  {
      glTextureSubImage3D(
          rendererID,
          0,                // mip level
          0, 0, i,          // x, y, z offset — z=i for cube face
          width, height, 1, // width, height, depth (1 face)
          dataFormat,
          GL_UNSIGNED_BYTE,
          pixels[i]
      );

      stbi_image_free(pixels[i]);
      pixels[i] = nullptr;
  }

  glTextureParameteri(rendererID, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTextureParameteri(rendererID, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTextureParameteri(rendererID, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTextureParameteri(rendererID, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTextureParameteri(rendererID, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

  glGenerateTextureMipmap(rendererID);

  s_Data.skyboxes[name] = cubemap;

  gablog_log(LOG_WARN, __FILE__, __LINE__, "Skybox uploading took %.3f ms", timer.ElapsedMillis());
}

void OpenGLRenderer::DrawSkybox(const std::string& name)
{
  if (s_Data.m_SkyboxVAO == 0)
  {
    float skyboxVertices[] =
    {
        -1.0f,  1.0f, -1.0f,  -1.0f, -1.0f, -1.0f,   1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,   1.0f,  1.0f, -1.0f,  -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,  -1.0f, -1.0f, -1.0f,  -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,  -1.0f,  1.0f,  1.0f,  -1.0f, -1.0f,  1.0f,

         1.0f, -1.0f, -1.0f,   1.0f, -1.0f,  1.0f,   1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,   1.0f,  1.0f, -1.0f,   1.0f, -1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,  -1.0f,  1.0f,  1.0f,   1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,   1.0f, -1.0f,  1.0f,  -1.0f, -1.0f,  1.0f,

        -1.0f,  1.0f, -1.0f,   1.0f,  1.0f, -1.0f,   1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,  -1.0f,  1.0f,  1.0f,  -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f, -1.0f,  -1.0f, -1.0f,  1.0f,   1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,  -1.0f, -1.0f,  1.0f,   1.0f, -1.0f,  1.0f
    };

    glCreateVertexArrays(1, &s_Data.m_SkyboxVAO);
    glCreateBuffers(1, &s_Data.m_SkyboxVBO);
    glNamedBufferData(s_Data.m_SkyboxVBO, sizeof(skyboxVertices), skyboxVertices, GL_STATIC_DRAW);

    glVertexArrayVertexBuffer(s_Data.m_SkyboxVAO, 0, s_Data.m_SkyboxVBO, 0, 3 * sizeof(float));
    glEnableVertexArrayAttrib(s_Data.m_SkyboxVAO, 0);
    glVertexArrayAttribFormat(s_Data.m_SkyboxVAO, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(s_Data.m_SkyboxVAO, 0, 0);
  }

  glDepthFunc(GL_LEQUAL);
  glDepthMask(GL_FALSE);
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

  s_Data.s_Shaders.skyboxShader->Bind();

  auto it = s_Data.skyboxes.find(name);
  if (it != s_Data.skyboxes.end())
  {
    glBindTextureUnit(0, it->second->GetRendererID());
  }
  else
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "Skybox texture not found: %s", name.c_str());
    return;
  }

  glBindVertexArray(s_Data.m_SkyboxVAO);
  glDrawArrays(GL_TRIANGLES, 0, 36);
  glBindVertexArray(0);

  glDepthFunc(GL_LESS);
  glDepthMask(GL_TRUE);
}

void OpenGLRenderer::DrawText(const Font* font, const std::string& text, const glm::vec3& position, const glm::vec3& rotation, float size, const glm::vec4& color)
{
  Set3D(true);
  DrawText(font, text, position, rotation, size, color, -1);
  Set3D(false);
}

void OpenGLRenderer::DrawParticles(const std::vector<ParticleRenderInstance>& instances)
{
  if (instances.empty() || !s_Data.m_ParticleVAO ||
      !s_Data.m_ParticleInstanceBuffer || !s_Data.s_Shaders.ParticleShader)
    return;

  glNamedBufferSubData(s_Data.m_ParticleInstanceBuffer, 0,
    static_cast<GLsizeiptr>(instances.size() * sizeof(ParticleRenderInstance)),
    instances.data());
  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_FALSE);
  glDisable(GL_CULL_FACE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  s_Data.s_Shaders.ParticleShader->Bind();
  glBindVertexArray(s_Data.m_ParticleVAO);
  glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4,
    static_cast<GLsizei>(instances.size()));
  glBindVertexArray(0);
  s_Data.s_Shaders.ParticleShader->UnBind();
  glDisable(GL_BLEND);
  glEnable(GL_CULL_FACE);
  glDepthMask(GL_TRUE);
}

void OpenGLRenderer::DrawText(const Font* font, const std::string& text, const glm::vec2& position, float size, const glm::vec4& color)
{
  DrawText(font, text, glm::vec3(position, 0.0f), glm::vec3(0.0f), size, color, -1);
}

void OpenGLRenderer::DrawText(const Font* font, const std::string& text, const glm::vec3& position, const glm::vec3& rotation, float size, const glm::vec4& color, int entityID)
{
  if (!font || font->m_AtlasHandle == 0 || font->m_Characters.empty() || text.empty())
  {
      gablog_log(LOG_ERROR, __FILE__, __LINE__, "Font is nullptr, empty, or text is empty");
      return;
  }

  float textWidth = 0.0f;

  // First pass: calculate the horizontal extent. Vertical placement uses the
  // font-wide ascender/descender so every label shares the same baseline.
  for (char c : text)
  {
    auto it = font->m_Characters.find(c);
    if (it == font->m_Characters.end()) continue;

    const auto& ch = it->second;
    textWidth += (ch.Advance >> 6) * size;
  }

  const float ascender = font->m_Ascender > 0.0f ? font->m_Ascender : 48.0f;
  const float descender = std::max(font->m_Descender, 0.0f);
  const float baselineY = (descender - ascender) * size * 0.5f;
  glm::vec3 cursor(-textWidth * 0.5f, 0.0f, 0.0f);

  // Precompute global transform
  glm::mat4 baseTransform = glm::translate(glm::mat4(1.0f), position) *
                            glm::rotate(glm::mat4(1.0f), rotation.x, {1, 0, 0}) *
                            glm::rotate(glm::mat4(1.0f), rotation.y, {0, 1, 0}) *
                            glm::rotate(glm::mat4(1.0f), rotation.z, {0, 0, 1});

  for (char c : text)
  {
    auto it = font->m_Characters.find(c);
    if (it == font->m_Characters.end()) continue;

    const Character& glyph = it->second;
    const auto& Size = glyph.Size;
    const auto& Bearing = glyph.Bearing;
    const uint32_t Advance = glyph.Advance;

    if (s_Data.QuadIndexCount >= OpenGLRendererData::MaxIndices)
        NextBatch();

    float xpos = cursor.x + Bearing.x * size;
    float ypos = baselineY + (Bearing.y - Size.y) * size;
    float w = Size.x * size;
    float h = Size.y * size;

    glm::mat4 localTransform = glm::translate(glm::mat4(1.0f), glm::vec3(xpos, ypos, 0.0f)) *
                               glm::scale(glm::mat4(1.0f), glm::vec3(w, h, 1.0f));

    glm::mat4 transform = baseTransform * localTransform;

    float textureIndex = 0.0f;
    for (uint32_t slot = 1; slot < s_Data.TextureSlotIndex; ++slot)
    {
      if (s_Data.TextureSlots[slot] &&
          s_Data.TextureSlots[slot]->GetRendererID() == font->m_AtlasHandle)
      {
        textureIndex = static_cast<float>(slot);
        break;
      }
    }
    if (textureIndex == 0.0f)
    {
      if (s_Data.TextureSlotIndex >= OpenGLRendererData::MaxTextureSlots)
        NextBatch();

      textureIndex = static_cast<float>(s_Data.TextureSlotIndex);
      s_Data.TextureSlots[s_Data.TextureSlotIndex++] = Texture::WrapExisting(
        static_cast<uint32_t>(font->m_AtlasHandle));
    }

    const glm::vec2 glyphTexCoords[4] = {
      {glyph.UVTopLeft.x, glyph.UVBottomRight.y},
      glyph.UVBottomRight,
      {glyph.UVBottomRight.x, glyph.UVTopLeft.y},
      glyph.UVTopLeft};
    for (int i = 0; i < 4; i++) {
        s_Data.QuadVertexBufferPtr->Position = transform * glm::vec4(OpenGLRendererData::quadPositions[i], 1.0f);
        s_Data.QuadVertexBufferPtr->Color = color;
        s_Data.QuadVertexBufferPtr->TexCoord = glyphTexCoords[i];
        s_Data.QuadVertexBufferPtr->TexIndex = textureIndex;
        s_Data.QuadVertexBufferPtr->TilingFactor = 1.0f;
        s_Data.QuadVertexBufferPtr->EntityID = entityID;
        s_Data.QuadVertexBufferPtr++;
    }

    s_Data.QuadIndexCount += 6;
    cursor.x += (Advance >> 6) * size;
  }
}

void OpenGLRenderer::AddDrawCommand(const std::string& modelName, uint32_t verticesSize, uint32_t indicesSize)
{
  if (RenderBackend::Capabilities().NativeModelResources) return;

  DrawElementsIndirectCommand cmd =
  {
    .count = (indicesSize),
    .instanceCount = 1,
    .firstIndex = (s_Data.m_DrawIndexOffset),
    .baseVertex = static_cast<GLint>(s_Data.m_DrawVertexOffset),
    .baseInstance = 0,
  };

  s_Data.m_ModelDrawCommandIndices[modelName].push_back(s_Data.m_DrawCommands.size()); // store index
  s_Data.m_DrawCommands.push_back(cmd);

  s_Data.m_DrawIndexOffset += indicesSize;
  s_Data.m_DrawVertexOffset += verticesSize;
}

void OpenGLRenderer::RebuildDrawCommandsForModel(const std::shared_ptr<Model>& model, bool render)
{
  model->m_IsRendered = render;
  if (RenderBackend::Capabilities().NativeModelResources) return;
  UpdateDrawCommandInstances(model);
}

void OpenGLRenderer::UpdateModelFrustumCulling()
{
  if (PrepareGPUCullData())
  {
    DispatchGPUCull(Camera::GetViewProjection(), false);
    return;
  }

  if (s_Data.m_DrawCommands.empty() || s_Data.m_CulledCmdBuffer == 0)
  {
    RenderBackend::SetStatistics({});
    return;
  }

  const RenderFrustum frustum(Camera::GetViewProjection());
  auto& visibleTransforms = s_Data.m_VisibleInstanceTransforms;
  visibleTransforms.clear();
  s_Data.m_CulledDrawCommands = s_Data.m_DrawCommands;
  RenderStatistics statistics;

  for (const std::string& modelName : ModelManager::GetModelNames())
  {
    const auto model = ModelManager::GetModel(modelName);
    const auto commandIndices = s_Data.m_ModelDrawCommandIndices.find(modelName);
    if (!model || commandIndices == s_Data.m_ModelDrawCommandIndices.end())
      continue;

    const auto visibleBase = static_cast<GLuint>(visibleTransforms.size());
    GLuint visibleCount = 0;
    if (model->m_IsRendered)
    {
      statistics.RenderableInstances += static_cast<uint32_t>(model->m_InstanceTransforms.size());
      for (const glm::mat4& transform : model->m_InstanceTransforms)
      {
        const WorldBoundingSphere sphere = CalculateWorldBoundingSphere(*model, transform);
        if (!frustum.IntersectsSphere(sphere.center, sphere.radius))
          continue;

        visibleTransforms.push_back(transform);
        ++visibleCount;
      }
    }

    statistics.VisibleInstances += visibleCount;
    for (const size_t commandIndex : commandIndices->second)
    {
      auto& command = s_Data.m_CulledDrawCommands[commandIndex];
      command.instanceCount = visibleCount;
      command.baseInstance = visibleBase;
    }
  }

  ModelManager::UploadVisibleInstanceTransforms(visibleTransforms);
  glNamedBufferSubData(s_Data.m_CulledCmdBuffer, 0,
    static_cast<GLsizeiptr>(s_Data.m_CulledDrawCommands.size() * sizeof(DrawElementsIndirectCommand)),
    s_Data.m_CulledDrawCommands.data());
  RenderBackend::SetStatistics(statistics);
}

static void UpdateShadowFaceCulling(const glm::mat4& viewProjection)
{
  if (s_Data.m_DrawCommands.empty() || s_Data.m_CulledCmdBuffer == 0) return;

  const RenderFrustum frustum(viewProjection);
  auto& visibleTransforms = s_Data.m_VisibleInstanceTransforms;
  visibleTransforms.clear();
  s_Data.m_CulledDrawCommands = s_Data.m_DrawCommands;

  for (const std::string& modelName : ModelManager::GetModelNames())
  {
    const auto model = ModelManager::GetModel(modelName);
    const auto commandIndices = s_Data.m_ModelDrawCommandIndices.find(modelName);
    if (!model || commandIndices == s_Data.m_ModelDrawCommandIndices.end()) continue;

    const auto visibleBase = static_cast<GLuint>(visibleTransforms.size());
    GLuint visibleCount = 0;
    if (model->m_IsRendered)
    {
      for (const glm::mat4& transform : model->m_InstanceTransforms)
      {
        const WorldBoundingSphere sphere = CalculateWorldBoundingSphere(*model, transform);
        if (!frustum.IntersectsSphere(sphere.center, sphere.radius)) continue;
        visibleTransforms.push_back(transform);
        ++visibleCount;
      }
    }

    for (const size_t commandIndex : commandIndices->second)
    {
      auto& command = s_Data.m_CulledDrawCommands[commandIndex];
      command.instanceCount = visibleCount;
      command.baseInstance = visibleBase;
    }
  }

  ModelManager::UploadVisibleInstanceTransforms(visibleTransforms);
  glNamedBufferSubData(s_Data.m_CulledCmdBuffer, 0,
    static_cast<GLsizeiptr>(s_Data.m_CulledDrawCommands.size() *
      sizeof(DrawElementsIndirectCommand)), s_Data.m_CulledDrawCommands.data());
}

static void UploadPointShadowSlots(const std::vector<int32_t>& slots)
{
  if (!s_Data.m_PointShadowSlotBuffer || slots.empty()) return;
  s_Data.m_PointShadowSlotBuffer->SetData(
    slots.size() * sizeof(int32_t), slots.data());
}

static void EnsureTiledLightBuffers(uint32_t tileCount)
{
  const size_t requiredTiles = std::max<size_t>(tileCount, 1);
  if (s_Data.m_TileLightGridBuffer && s_Data.m_TileLightIndexBuffer &&
      requiredTiles <= s_Data.m_TileBufferCapacity)
    return;

  if (s_Data.m_TileLightGridBuffer)
    glDeleteBuffers(1, &s_Data.m_TileLightGridBuffer);
  if (s_Data.m_TileLightIndexBuffer)
    glDeleteBuffers(1, &s_Data.m_TileLightIndexBuffer);

  s_Data.m_TileBufferCapacity = std::max(
    requiredTiles, s_Data.m_TileBufferCapacity * 2);
  glCreateBuffers(1, &s_Data.m_TileLightGridBuffer);
  glNamedBufferData(s_Data.m_TileLightGridBuffer,
    static_cast<GLsizeiptr>(s_Data.m_TileBufferCapacity * sizeof(glm::uvec2)),
    nullptr, GL_DYNAMIC_DRAW);
  glCreateBuffers(1, &s_Data.m_TileLightIndexBuffer);
  glNamedBufferData(s_Data.m_TileLightIndexBuffer,
    static_cast<GLsizeiptr>(s_Data.m_TileBufferCapacity *
      OpenGLRendererData::MaxLightsPerTile * sizeof(uint32_t)),
    nullptr, GL_DYNAMIC_DRAW);
}

static bool DispatchTiledLightCulling()
{
  if (!s_Data.m_TiledLightingSupported ||
      !s_Data.s_Shaders.TiledLightCullShader ||
      !s_Data.m_GeometryBuffer || RenderBackend::Lights().empty())
  {
    s_Data.m_TileDebugTileCount = 0;
    s_Data.m_TileDebugActiveTileCount = 0;
    s_Data.m_TileDebugMinLights = 0;
    s_Data.m_TileDebugMaxLights = 0;
    s_Data.m_TileDebugAverageLights = 0.0f;
    return false;
  }

  const uint32_t width = std::max(Window::GetWidth(), 1u);
  const uint32_t height = std::max(Window::GetHeight(), 1u);
  const uint32_t tileCountX =
    (width + OpenGLRendererData::TileSize - 1u) / OpenGLRendererData::TileSize;
  const uint32_t tileCountY =
    (height + OpenGLRendererData::TileSize - 1u) / OpenGLRendererData::TileSize;
  EnsureTiledLightBuffers(tileCountX * tileCountY);

  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 21, s_Data.m_TileLightGridBuffer);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 22, s_Data.m_TileLightIndexBuffer);
  s_Data.m_GeometryBuffer->BindPositionTextureForReading(GL_TEXTURE1);
  glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

  const auto& shader = s_Data.s_Shaders.TiledLightCullShader;
  shader->Bind();
  shader->SetInt("gPosition", 1);
  shader->SetInt("u_TileCountX", static_cast<int>(tileCountX));
  glDispatchCompute(tileCountX, tileCountY, 1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  shader->UnBind();

  if (RenderBackend::DebugSettings().TiledLightingMode != 0)
  {
    const uint32_t tileCount = tileCountX * tileCountY;
    std::vector<glm::uvec2> grid(tileCount);
    glGetNamedBufferSubData(s_Data.m_TileLightGridBuffer, 0,
      static_cast<GLsizeiptr>(grid.size() * sizeof(glm::uvec2)), grid.data());
    uint64_t lightSum = 0;
    uint32_t activeTiles = 0;
    uint32_t minimumLights = std::numeric_limits<uint32_t>::max();
    uint32_t maximumLights = 0;
    for (const glm::uvec2& entry : grid)
    {
      if (entry.y == 0) continue;
      ++activeTiles;
      lightSum += entry.y;
      minimumLights = std::min(minimumLights, entry.y);
      maximumLights = std::max(maximumLights, entry.y);
    }
    s_Data.m_TileDebugTileCount = tileCount;
    s_Data.m_TileDebugActiveTileCount = activeTiles;
    s_Data.m_TileDebugMinLights = activeTiles > 0 ? minimumLights : 0;
    s_Data.m_TileDebugMaxLights = maximumLights;
    s_Data.m_TileDebugAverageLights = activeTiles > 0
      ? static_cast<float>(lightSum) / static_cast<float>(activeTiles)
      : 0.0f;
  }
  return true;
}

static void WaitForCullFence(GLsync& fence)
{
  if (!fence) return;
  GLenum result = glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 0);
  while (result == GL_TIMEOUT_EXPIRED)
    result = glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 1'000'000);
  glDeleteSync(fence);
  fence = nullptr;
}

static void DestroyGPUDrivenResources()
{
  for (GLsync& fence : s_Data.m_CullRingFences)
    WaitForCullFence(fence);
  if (s_Data.m_CullDataMapped && s_Data.m_CullDataRingBuffer)
    glUnmapNamedBuffer(s_Data.m_CullDataRingBuffer);
  const std::array<GLuint, 4> buffers = {
    s_Data.m_GPUVisibleTransforms, s_Data.m_GPUDrawRemap,
    s_Data.m_GPUDrawCount, s_Data.m_CullDataRingBuffer};
  glDeleteBuffers(static_cast<GLsizei>(buffers.size()), buffers.data());
  if (s_Data.m_HiZTexture) glDeleteTextures(1, &s_Data.m_HiZTexture);
  s_Data.m_GPUVisibleTransforms = 0;
  s_Data.m_GPUDrawRemap = 0;
  s_Data.m_GPUDrawCount = 0;
  s_Data.m_CullDataRingBuffer = 0;
  s_Data.m_CullDataMapped = nullptr;
  s_Data.m_CullDataSegmentStride = 0;
  s_Data.m_CullDataCapacity = 0;
  s_Data.m_CullDataCurrentOffset = 0;
  s_Data.m_CullDataPrepared = false;
  s_Data.m_GPUVisibleTransformCapacity = 0;
  s_Data.m_GPUCommandCapacity = 0;
  s_Data.m_HiZTexture = 0;
  s_Data.m_HiZWidth = s_Data.m_HiZHeight = s_Data.m_HiZMipCount = 0;
  s_Data.m_HiZValid = false;
}

static size_t AlignBufferOffset(size_t value, size_t alignment)
{
  return (value + alignment - 1) / alignment * alignment;
}

static void EnsureCullDataRing(size_t requiredBytes)
{
  requiredBytes = std::max(requiredBytes, sizeof(DrawCullData));
  if (s_Data.m_CullDataRingBuffer && requiredBytes <= s_Data.m_CullDataCapacity)
    return;

  for (GLsync& fence : s_Data.m_CullRingFences)
    WaitForCullFence(fence);
  if (s_Data.m_CullDataMapped && s_Data.m_CullDataRingBuffer)
    glUnmapNamedBuffer(s_Data.m_CullDataRingBuffer);
  if (s_Data.m_CullDataRingBuffer)
    glDeleteBuffers(1, &s_Data.m_CullDataRingBuffer);

  GLint alignment = 256;
  glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &alignment);
  s_Data.m_CullDataCapacity = std::max(requiredBytes, s_Data.m_CullDataCapacity * 2);
  s_Data.m_CullDataSegmentStride = AlignBufferOffset(
    s_Data.m_CullDataCapacity, static_cast<size_t>(std::max(alignment, 1)));
  const size_t totalBytes = s_Data.m_CullDataSegmentStride * s_Data.m_CullRingFences.size();
  constexpr GLbitfield storageFlags =
    GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
  glCreateBuffers(1, &s_Data.m_CullDataRingBuffer);
  glNamedBufferStorage(s_Data.m_CullDataRingBuffer,
    static_cast<GLsizeiptr>(totalBytes), nullptr, storageFlags);
  s_Data.m_CullDataMapped = static_cast<uint8_t*>(glMapNamedBufferRange(
    s_Data.m_CullDataRingBuffer, 0, static_cast<GLsizeiptr>(totalBytes), storageFlags));
  if (!s_Data.m_CullDataMapped)
    throw std::runtime_error("Failed to persistently map OpenGL cull-data ring");
}

static void EnsureGPUCullOutputBuffers(size_t transformCount, size_t commandCount)
{
  const size_t requiredTransforms = std::max<size_t>(transformCount, 1);
  const size_t requiredCommands = std::max<size_t>(commandCount, 1);
  if (s_Data.m_GPUVisibleTransforms && s_Data.m_GPUDrawRemap && s_Data.m_GPUDrawCount &&
      requiredTransforms <= s_Data.m_GPUVisibleTransformCapacity &&
      requiredCommands <= s_Data.m_GPUCommandCapacity)
    return;

  glFinish();
  if (s_Data.m_GPUVisibleTransforms) glDeleteBuffers(1, &s_Data.m_GPUVisibleTransforms);
  if (s_Data.m_GPUDrawRemap) glDeleteBuffers(1, &s_Data.m_GPUDrawRemap);
  if (s_Data.m_GPUDrawCount) glDeleteBuffers(1, &s_Data.m_GPUDrawCount);
  s_Data.m_GPUVisibleTransformCapacity = std::max(
    requiredTransforms, s_Data.m_GPUVisibleTransformCapacity * 2);
  s_Data.m_GPUCommandCapacity = std::max(
    requiredCommands, s_Data.m_GPUCommandCapacity * 2);

  glCreateBuffers(1, &s_Data.m_GPUVisibleTransforms);
  glNamedBufferStorage(s_Data.m_GPUVisibleTransforms,
    static_cast<GLsizeiptr>(s_Data.m_GPUVisibleTransformCapacity * sizeof(glm::mat4)),
    nullptr, 0);
  glCreateBuffers(1, &s_Data.m_GPUDrawRemap);
  glNamedBufferStorage(s_Data.m_GPUDrawRemap,
    static_cast<GLsizeiptr>(s_Data.m_GPUCommandCapacity * sizeof(uint32_t)), nullptr, 0);
  glCreateBuffers(1, &s_Data.m_GPUDrawCount);
  const uint32_t zero = 0;
  glNamedBufferStorage(s_Data.m_GPUDrawCount, sizeof(uint32_t), &zero,
    GL_DYNAMIC_STORAGE_BIT);
}

static void EnsureHiZTexture(uint32_t width, uint32_t height)
{
  width = std::max(width, 1u);
  height = std::max(height, 1u);
  if (s_Data.m_HiZTexture && width == s_Data.m_HiZWidth && height == s_Data.m_HiZHeight)
    return;
  if (s_Data.m_HiZTexture) glDeleteTextures(1, &s_Data.m_HiZTexture);
  s_Data.m_HiZWidth = width;
  s_Data.m_HiZHeight = height;
  s_Data.m_HiZMipCount = 1u + static_cast<uint32_t>(std::floor(
    std::log2(static_cast<float>(std::max(width, height)))));
  glCreateTextures(GL_TEXTURE_2D, 1, &s_Data.m_HiZTexture);
  glTextureStorage2D(s_Data.m_HiZTexture, static_cast<GLsizei>(s_Data.m_HiZMipCount),
    GL_R32F, static_cast<GLsizei>(width), static_cast<GLsizei>(height));
  glTextureParameteri(s_Data.m_HiZTexture, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
  glTextureParameteri(s_Data.m_HiZTexture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTextureParameteri(s_Data.m_HiZTexture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTextureParameteri(s_Data.m_HiZTexture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  s_Data.m_HiZValid = false;
}

static void BuildHiZPyramid()
{
  if (!s_Data.m_GPUDrivenSupported || !s_Data.s_Shaders.HiZBuildShader ||
      !s_Data.m_GeometryBuffer)
    return;
  EnsureHiZTexture(Window::GetWidth(), Window::GetHeight());
  auto& shader = s_Data.s_Shaders.HiZBuildShader;
  shader->Bind();
  shader->SetInt("u_Source", 15);
  for (uint32_t mip = 0; mip < s_Data.m_HiZMipCount; ++mip)
  {
    const uint32_t width = std::max(s_Data.m_HiZWidth >> mip, 1u);
    const uint32_t height = std::max(s_Data.m_HiZHeight >> mip, 1u);
    glBindTextureUnit(15, mip == 0
      ? s_Data.m_GeometryBuffer->GetDepthAttachmentRendererID()
      : s_Data.m_HiZTexture);
    glBindImageTexture(0, s_Data.m_HiZTexture, static_cast<GLint>(mip), GL_FALSE, 0,
      GL_WRITE_ONLY, GL_R32F);
    shader->SetBool("u_CopyDepth", mip == 0);
    shader->SetInt("u_SourceMip", mip == 0 ? 0 : static_cast<int>(mip - 1));
    shader->SetVec2("u_DestinationSize", static_cast<float>(width), static_cast<float>(height));
    glDispatchCompute(
      (width + s_Data.m_HiZLocalSizeX - 1u) / s_Data.m_HiZLocalSizeX,
      (height + s_Data.m_HiZLocalSizeY - 1u) / s_Data.m_HiZLocalSizeY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
  }
  shader->UnBind();
  s_Data.m_HiZValid = true;
}

static void DispatchGPUCull(const glm::mat4& viewProjection, bool useHiZ)
{
  const size_t drawCount = s_Data.m_DrawCommands.size();
  if (!s_Data.m_GPUDrivenSupported || drawCount == 0 ||
      !s_Data.s_Shaders.GPUCullShader)
    return;
  const uint32_t zero = 0;
  glClearNamedBufferData(s_Data.m_GPUDrawCount, GL_R32UI,
    GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, s_Data.m_cmdBufer);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 16,
    ModelManager::GetAllInstanceTransformsBuffer());
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 17, s_Data.m_GPUVisibleTransforms);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 18, s_Data.m_CulledCmdBuffer);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 19, s_Data.m_GPUDrawRemap);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 20, s_Data.m_GPUDrawCount);
  glBindTextureUnit(15, s_Data.m_HiZTexture);
  auto& shader = s_Data.s_Shaders.GPUCullShader;
  shader->Bind();
  shader->SetMat4("u_ViewProjection", viewProjection);
  shader->SetBool("u_UseHiZ", useHiZ && s_Data.m_HiZValid);
  shader->SetInt("u_HiZMaxMip", static_cast<int>(
    s_Data.m_HiZMipCount > 0 ? s_Data.m_HiZMipCount - 1 : 0));
  shader->SetVec2("u_HiZResolution",
    static_cast<float>(s_Data.m_HiZWidth), static_cast<float>(s_Data.m_HiZHeight));
  shader->SetInt("u_HiZTexture", 15);
  glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
  glDispatchCompute(static_cast<GLuint>(
    (drawCount + s_Data.m_GPUCullLocalSizeX - 1u) / s_Data.m_GPUCullLocalSizeX), 1, 1);
  glMemoryBarrier(GL_COMMAND_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
  shader->UnBind();
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, s_Data.m_GPUVisibleTransforms);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, s_Data.m_GPUDrawRemap);
}

static void DrawCompactedCommands()
{
  glBindBuffer(GL_DRAW_INDIRECT_BUFFER, s_Data.m_CulledCmdBuffer);
  glBindBuffer(GL_PARAMETER_BUFFER, s_Data.m_GPUDrawCount);
  glMultiDrawElementsIndirectCount(GL_TRIANGLES, GL_UNSIGNED_INT, nullptr, 0,
    static_cast<GLsizei>(s_Data.m_DrawCommands.size()), 0);
  glBindBuffer(GL_PARAMETER_BUFFER, 0);
  glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
}

static bool PrepareGPUCullData()
{
  if (!s_Data.m_GPUDrivenSupported || s_Data.m_DrawCommands.empty())
    return false;
  if (s_Data.m_CullDataPrepared)
    return true;
  std::vector<DrawCullData> cullData(s_Data.m_DrawCommands.size());
  size_t visibleTransformCapacity = 0;
  for (size_t draw = 0; draw < s_Data.m_DrawCommands.size(); ++draw)
  {
    cullData[draw].Metadata.x = static_cast<uint32_t>(visibleTransformCapacity);
    cullData[draw].Metadata.y = static_cast<uint32_t>(draw);
    visibleTransformCapacity += s_Data.m_DrawCommands[draw].instanceCount;
  }

  RenderStatistics statistics;
  for (const std::string& modelName : ModelManager::GetModelNames())
  {
    const auto model = ModelManager::GetModel(modelName);
    const auto commandIndices = s_Data.m_ModelDrawCommandIndices.find(modelName);
    if (!model || commandIndices == s_Data.m_ModelDrawCommandIndices.end()) continue;
    if (model->m_IsRendered)
      statistics.RenderableInstances += static_cast<uint32_t>(model->m_InstanceTransforms.size());
    const bool allowHiZ = !model->IsAnimated() &&
      model->GetPhysXMeshType() != MeshType::CONTROLLER;
    for (const size_t commandIndex : commandIndices->second)
    {
      if (commandIndex >= cullData.size()) continue;
      cullData[commandIndex].LocalSphere = glm::vec4(
        model->GetBoundsCenter(), model->GetBoundsRadius());
      cullData[commandIndex].Metadata.z = allowHiZ ? 1u : 0u;
    }
  }
  statistics.VisibleInstances = statistics.RenderableInstances;
  RenderBackend::SetStatistics(statistics);

  s_Data.m_CurrentVisibleTransformCapacity = visibleTransformCapacity;
  EnsureGPUCullOutputBuffers(visibleTransformCapacity, cullData.size());
  const size_t bytes = cullData.size() * sizeof(DrawCullData);
  EnsureCullDataRing(bytes);
  s_Data.m_CullRingIndex = (s_Data.m_CullRingIndex + 1u) %
    static_cast<uint32_t>(s_Data.m_CullRingFences.size());
  WaitForCullFence(s_Data.m_CullRingFences[s_Data.m_CullRingIndex]);
  s_Data.m_CullDataCurrentOffset =
    s_Data.m_CullRingIndex * s_Data.m_CullDataSegmentStride;
  std::memcpy(s_Data.m_CullDataMapped + s_Data.m_CullDataCurrentOffset,
    cullData.data(), bytes);
  glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 15, s_Data.m_CullDataRingBuffer,
    static_cast<GLintptr>(s_Data.m_CullDataCurrentOffset), static_cast<GLsizeiptr>(bytes));
  s_Data.m_CullDataPrepared = true;
  return true;
}

void OpenGLRenderer::UpdateDrawCommandInstances(const std::shared_ptr<Model>& model)
{
  if (RenderBackend::Capabilities().NativeModelResources) return;

  const auto commandIndices = s_Data.m_ModelDrawCommandIndices.find(model->m_Name);
  if (commandIndices == s_Data.m_ModelDrawCommandIndices.end()) return;

  const GLuint instanceCount = model->m_IsRendered
    ? static_cast<GLuint>(model->m_InstanceTransforms.size())
    : 0;

  for (const size_t commandIndex : commandIndices->second)
  {
    auto& command = s_Data.m_DrawCommands[commandIndex];
    command.instanceCount = instanceCount;
    command.baseInstance = model->m_InstanceBase;
  }

  const size_t requiredSize = s_Data.m_DrawCommands.size() * sizeof(DrawElementsIndirectCommand);
  if (s_Data.m_cmdBufer != 0 && requiredSize <= s_Data.m_cmdBufferSize)
  {
    glNamedBufferSubData(
      s_Data.m_cmdBufer,
      0,
      requiredSize,
      s_Data.m_DrawCommands.data());
  }
}

void OpenGLRenderer::SetModelPreviews(const std::vector<RenderModelPreview>& previews)
{
  s_Data.m_ModelPreviews = previews;
}

void OpenGLRenderer::InitDrawCommandBuffer()
{
  if (RenderBackend::Capabilities().NativeModelResources) return;

  if (s_Data.m_DrawCommands.empty()) return;

  if (s_Data.m_cmdBufer != 0) glDeleteBuffers(1, &s_Data.m_cmdBufer);
  if (s_Data.m_CulledCmdBuffer != 0) glDeleteBuffers(1, &s_Data.m_CulledCmdBuffer);

  s_Data.m_cmdBufferSize = s_Data.m_DrawCommands.size() * sizeof(DrawElementsIndirectCommand);
  s_Data.m_CulledDrawCommands = s_Data.m_DrawCommands;
  glCreateBuffers(1, &s_Data.m_cmdBufer);
  glNamedBufferStorage(s_Data.m_cmdBufer, s_Data.m_cmdBufferSize, s_Data.m_DrawCommands.data(), GL_DYNAMIC_STORAGE_BIT);
  glCreateBuffers(1, &s_Data.m_CulledCmdBuffer);
  glNamedBufferStorage(s_Data.m_CulledCmdBuffer, s_Data.m_cmdBufferSize, s_Data.m_CulledDrawCommands.data(), GL_DYNAMIC_STORAGE_BIT);
}

void OpenGLRenderer::ResetModelDrawCommands()
{
  if (RenderBackend::Capabilities().NativeModelResources)
  {
    RenderBackend::Get().ResetSceneResources();
    return;
  }

  if (s_Data.m_cmdBufer != 0) glDeleteBuffers(1, &s_Data.m_cmdBufer);
  if (s_Data.m_CulledCmdBuffer != 0) glDeleteBuffers(1, &s_Data.m_CulledCmdBuffer);

  s_Data.m_cmdBufer = 0;
  s_Data.m_CulledCmdBuffer = 0;
  s_Data.m_cmdBufferSize = 0;
  s_Data.m_DrawCommands.clear();
  s_Data.m_CulledDrawCommands.clear();
  s_Data.m_VisibleInstanceTransforms.clear();
  s_Data.m_ModelPreviews.clear();
  s_Data.m_ModelDrawCommandIndices.clear();
  s_Data.m_DrawIndexOffset = 0;
  s_Data.m_DrawVertexOffset = 0;
  s_Data.m_PointShadowCache = {};
  RenderBackend::SetStatistics({});
}

void OpenGLRenderer::DrawIndexed(const std::shared_ptr<VertexArray>& vertexArray, uint32_t indexCount)
{
	vertexArray->Bind();
	uint32_t count = indexCount ? indexCount : vertexArray->GetIndexBuffer()->GetCount();
	glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_INT, nullptr);
}

void OpenGLRenderer::DrawLines(const std::shared_ptr<VertexArray>& vertexArray, uint32_t vertexCount)
{
	vertexArray->Bind();
	glDrawArrays(GL_LINES, 0, vertexCount);
}

void OpenGLRenderer::SetLineWidth(float width)
{
	glLineWidth(width);
}

float OpenGLRenderer::GetLineWidth()
{
	return s_Data.LineWidth;
}

uint32_t OpenGLRenderer::GetActiveWidgetID()
{
	return GImGui->ActiveId;
}

void OpenGLRenderer::BlockEvents(bool block)
{
	s_Data.m_BlockEvents = block;
}

void OpenGLRenderer::Set3D(bool is3D)
{
  if (s_Data.Is3D == is3D) return;

  Flush();
  StartBatch();
  s_Data.Is3D = is3D;
}

void OpenGLRenderer::DrawEditorFrameBuffer(uint64_t framebufferTexture)
{
	const RenderBackendCapabilities& capabilities = RenderBackend::Capabilities();
	RenderBackend::Get().BeginImGuiFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	//ImGuizmo::BeginFrame();
	// Note: Switch this to true to enable dockspace
	static bool dockspaceOpen = true;
	static bool opt_fullscreen_persistant = true;
	bool opt_fullscreen = opt_fullscreen_persistant;
	static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_NoTabBar;

	// We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
	// because it would be confusing to have two docking targets within each others.
	ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking;
	if (opt_fullscreen)
	{
		ImGuiViewport* viewport = ImGui::GetMainViewport();
		ImGui::SetNextWindowPos(viewport->Pos);
		ImGui::SetNextWindowSize(viewport->Size);
		ImGui::SetNextWindowViewport(viewport->ID);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
		window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
		window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
	}

	// When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background and handle the pass-thru hole, so we ask Begin() to not render a background.
	if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
		window_flags |= ImGuiWindowFlags_NoBackground;

	// Important: note that we proceed even if Begin() returns false (aka window is collapsed).
	// This is because we want to keep our DockSpace() active. If a DockSpace() is inactive,
	// all active windows docked into it will lose their parent and become undocked.
	// We cannot preserve the docking relationship between an active window and an inactive docking, otherwise
	// any change of dockspace/settings would lead to windows being stuck in limbo and never being visible.
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
	ImGui::Begin("DockSpace Demo", &dockspaceOpen, window_flags);
	ImGui::PopStyleVar();
	if (opt_fullscreen) ImGui::PopStyleVar(2);

	// DockSpace
	ImGuiIO& io = ImGui::GetIO();
	ImGuiStyle& style = ImGui::GetStyle();
	float minWinSizeX = style.WindowMinSize.x;
	style.WindowMinSize.x = 370.0f;
	if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
	{
		ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
		ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
	}

	style.WindowMinSize.x = minWinSizeX;

	ImGui::Begin("Scene Hierarchy", nullptr, ImGuiWindowFlags_NoCollapse);

	SceneManager::SyncEditorEntityTransforms();
	const std::string activeSceneName = SceneManager::GetActiveSceneName();
	const auto sceneNames = SceneManager::GetAvailableSceneNames();
	if (ImGui::BeginCombo("Scene", activeSceneName.empty() ? "<none>" : activeSceneName.c_str()))
	{
		for (const auto& sceneName : sceneNames)
		{
			const bool selected = sceneName == activeSceneName;
			if (ImGui::Selectable(sceneName.c_str(), selected) && !selected)
			{
				s_Data.m_SelectedEntityID = 0;
				s_Data.m_SelectedLightID = 0;
				SceneManager::LoadScene(sceneName);
			}
			if (selected)
				ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}

	static const char* saveStatus = nullptr;
	if (ImGui::Button("Save Scene")) saveStatus = SceneManager::SaveActiveScene() ? "Scene saved" : "Save failed";
	ImGui::SameLine();
	if (ImGui::Button("Reload Scene") && !activeSceneName.empty())
	{
		s_Data.m_SelectedEntityID = 0;
		s_Data.m_SelectedLightID = 0;
		SceneManager::LoadScene(activeSceneName);
	}
	if (saveStatus) ImGui::TextUnformatted(saveStatus);

	ImGui::Separator();
	if (ImGui::CollapsingHeader("Import External Model"))
	{
		static char externalModelPath[1024]{};
		static bool externalModelAnimated = false;
		static float externalModelOptimizer = 1.0f;
		static float externalModelBoundsScale = 1.0f;
		static int externalModelCollision = 0;
		static std::string importStatus;

		ImGui::SetNextItemWidth(-88.0f);
		ImGui::InputText("##ExternalModelPath", externalModelPath, sizeof(externalModelPath));
		ImGui::SameLine();
		if (ImGui::Button("Browse...")) BrowseForModelFile(externalModelPath, sizeof(externalModelPath));
		ImGui::Checkbox("Animated / Controller", &externalModelAnimated);
		ImGui::DragFloat("Optimizer Strength", &externalModelOptimizer, 0.05f, 0.0f, 10.0f, "%.2f");
		ImGui::DragFloat("Initial Bounds Scale", &externalModelBoundsScale, 0.02f, 0.01f, 100.0f, "%.2f");
		ImGui::BeginDisabled(externalModelAnimated);
		const char* collisionTypes[] = { "None", "Triangle Mesh", "Convex Helper" };
		ImGui::Combo("Physics Mesh", &externalModelCollision, collisionTypes, IM_ARRAYSIZE(collisionTypes));
		ImGui::EndDisabled();

		ImGui::BeginDisabled(externalModelPath[0] == '\0' || activeSceneName.empty());
		if (ImGui::Button("Import and Reload"))
		{
			const MeshType meshType = externalModelAnimated ? MeshType::CONTROLLER
				: externalModelCollision == 1 ? MeshType::TRIANGLEMESH
				: externalModelCollision == 2 ? MeshType::CONVEXMESH
				: MeshType::NONE;
			if (SceneManager::ImportExternalModel(externalModelPath, externalModelAnimated,
				externalModelOptimizer, meshType, externalModelBoundsScale))
			{
				importStatus = "Model imported; reloading scene...";
				s_Data.m_SelectedEntityID = 0;
				s_Data.m_SelectedLightID = 0;
				SceneManager::LoadScene(activeSceneName);
			}
			else
			{
				importStatus = "Import failed (check path or duplicate model name)";
			}
		}
		ImGui::EndDisabled();
		if (!importStatus.empty()) ImGui::TextWrapped("%s", importStatus.c_str());
	}

	ImGui::SeparatorText("Add Loaded Model");
	static std::string modelToAdd;
	const auto& modelNames = ModelManager::GetModelNames();
	auto canAddModel = [](const std::string& name)
	{
		const auto model = ModelManager::GetModel(name);
		return model && model->GetPhysXMeshType() != MeshType::CONVEXMESH &&
			model->GetPhysXMeshType() != MeshType::CONTROLLER;
	};
	if ((modelToAdd.empty() || !canAddModel(modelToAdd)))
	{
		const auto firstAddable = std::ranges::find_if(modelNames, canAddModel);
		modelToAdd = firstAddable == modelNames.end() ? std::string() : *firstAddable;
	}
	ImGui::SetNextItemWidth(190.0f);
	if (ImGui::BeginCombo("##ModelToAdd", modelToAdd.empty() ? "<no model>" : modelToAdd.c_str()))
	{
		for (const auto& modelName : modelNames)
		{
			if (!canAddModel(modelName)) continue;
			const bool selected = modelName == modelToAdd;
			if (ImGui::Selectable(modelName.c_str(), selected)) modelToAdd = modelName;
			if (selected) ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}
	ImGui::SameLine();
	ImGui::BeginDisabled(modelToAdd.empty());
	if (ImGui::Button("Add Model"))
	{
		if (const uint64_t entityID = SceneManager::AddModelEntity(modelToAdd); entityID != 0)
		{
			s_Data.m_SelectedEntityID = entityID;
			s_Data.m_SelectedLightID = 0;
		}
	}
	ImGui::EndDisabled();
	if (ImGui::IsItemHovered()) ImGui::SetTooltip("Adds an instance 3 units in front of the editor camera");

	ImGui::SeparatorText("Models");
	uint64_t duplicateRequest = 0;
	uint64_t removeEntityRequest = 0;
	for (const auto& entity : SceneManager::GetEntities())
	{
		ImGui::PushID(static_cast<int>(entity.id));
		const bool selected = entity.id == s_Data.m_SelectedEntityID;
		if (ImGui::Selectable(entity.name.c_str(), selected))
		{
			s_Data.m_SelectedEntityID = entity.id;
			s_Data.m_SelectedLightID = 0;
		}

		if (ImGui::BeginPopupContextItem("EntityContext"))
		{
			if (entity.type != "controller" && ImGui::MenuItem("Duplicate / Instance"))
				duplicateRequest = entity.id;
			if (entity.type == "controller")
				ImGui::TextDisabled("Controllers cannot be instanced");
			ImGui::Separator();
			if (ImGui::MenuItem("Remove Entity")) removeEntityRequest = entity.id;
			ImGui::EndPopup();
		}
		ImGui::PopID();
	}

	if (duplicateRequest != 0)
	{
		const uint64_t duplicateID = SceneManager::DuplicateEntity(duplicateRequest);
		if (duplicateID != 0)
		{
			s_Data.m_SelectedEntityID = duplicateID;
			s_Data.m_SelectedLightID = 0;
		}
	}
	if (removeEntityRequest != 0 && SceneManager::RemoveEntity(removeEntityRequest))
	{
		if (s_Data.m_SelectedEntityID == removeEntityRequest) s_Data.m_SelectedEntityID = 0;
	}

	ImGui::SeparatorText("Lights");
	static int newLightType = static_cast<int>(LightType::POINT);
	const char* lightTypes[] = { "Directional", "Point", "Spot" };
	ImGui::SetNextItemWidth(150.0f);
	ImGui::Combo("##NewLightType", &newLightType, lightTypes, IM_ARRAYSIZE(lightTypes));
	ImGui::SameLine();
	const bool directLightAlreadyExists = newLightType == static_cast<int>(LightType::DIRECT) &&
		std::ranges::any_of(SceneManager::GetLights(), [](const SceneLight& light) { return light.type == LightType::DIRECT; });
	ImGui::BeginDisabled(directLightAlreadyExists);
	if (ImGui::Button("Add Light"))
	{
		if (const uint64_t lightID = SceneManager::AddLight(static_cast<LightType>(newLightType)); lightID != 0)
		{
			s_Data.m_SelectedEntityID = 0;
			s_Data.m_SelectedLightID = lightID;
		}
	}
	ImGui::EndDisabled();
	if (directLightAlreadyExists && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
		ImGui::SetTooltip("Only one directional light is allowed per scene");

	uint64_t removeLightRequest = 0;
	for (const auto& light : SceneManager::GetLights())
	{
		ImGui::PushID("Light");
		ImGui::PushID(static_cast<int>(light.id));
		if (const bool selected = light.id == s_Data.m_SelectedLightID; ImGui::Selectable(light.name.c_str(), selected))
		{
			s_Data.m_SelectedEntityID = 0;
			s_Data.m_SelectedLightID = light.id;
		}
		if (ImGui::BeginPopupContextItem("LightContext"))
		{
			if (ImGui::MenuItem("Remove"))
				removeLightRequest = light.id;
			ImGui::EndPopup();
		}
		ImGui::PopID();
		ImGui::PopID();
	}
	if (removeLightRequest != 0 && SceneManager::RemoveLight(removeLightRequest))
	{
		if (s_Data.m_SelectedLightID == removeLightRequest) s_Data.m_SelectedLightID = 0;
	}


	ImGui::End();

	ImGui::Begin("Components", nullptr, ImGuiWindowFlags_NoCollapse);

	ImGui::BeginDisabled(!capabilities.OpenGLContext);
	if (ImGui::Button("Reload Shaders")) LoadShaders();
	ImGui::EndDisabled();
	if (!capabilities.OpenGLContext && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
		ImGui::SetTooltip("Native backend shaders are compiled when the renderer starts");
	ImGui::SameLine();
	RenderDebugSettings& debug = RenderBackend::DebugSettings();
	ImGui::Checkbox("Physics Debug", &debug.Physics);
	ImGui::Checkbox("Light Debug", &debug.Lights);
	ImGui::SameLine();
	ImGui::Checkbox("Culling Bounds", &debug.CullingBounds);
	ImGui::SameLine();
	ImGui::Checkbox("2D Debug", &debug.Debug2D);
	ImGui::Checkbox("G-Buffer", &debug.GBuffer);
	ImGui::SameLine();
	static const char* tiledDebugModes[] = { "Tiles: Off", "Tiles: Overlay", "Tiles: Heatmap" };
	int tiledDebugMode = static_cast<int>(debug.TiledLightingMode);
	ImGui::SetNextItemWidth(145.0f);
	if (ImGui::Combo("##TiledDebug", &tiledDebugMode, tiledDebugModes,
		IM_ARRAYSIZE(tiledDebugModes)))
		debug.TiledLightingMode = static_cast<uint32_t>(tiledDebugMode);
	if (ImGui::IsItemHovered())
		ImGui::SetTooltip("Shows 16x16 tiles colored by the number of affecting lights");
	const RenderStatistics& statistics = RenderBackend::Statistics();
	ImGui::TextDisabled("Frustum culling: %u / %u model instances visible",
		statistics.VisibleInstances, statistics.RenderableInstances);
	ImGui::TextDisabled("Point-shadow faces: %u rendered, %u cached",
		s_Data.m_PointShadowFacesRendered, s_Data.m_PointShadowFacesCached);
	if (debug.TiledLightingMode != 0 && capabilities.OpenGLContext)
	{
		ImGui::TextDisabled("Tiled grid: %u / %u active, lights min %u max %u avg %.2f",
			s_Data.m_TileDebugActiveTileCount, s_Data.m_TileDebugTileCount,
			s_Data.m_TileDebugMinLights, s_Data.m_TileDebugMaxLights,
			s_Data.m_TileDebugAverageLights);
		ImGui::TextDisabled("Heatmap scale: black 0, blue low, green medium, red all lights");
	}

	if (SceneEntity* entity = SceneManager::FindEntity(s_Data.m_SelectedEntityID))
	{
		const uint64_t selectedEntityID = entity->id;
		ImGui::Separator();
		ImGui::Text("Entity: %s", entity->name.c_str());
		ImGui::Text("Model: %s", entity->model.c_str());
		ImGui::Text("Instance: %u", entity->instanceIndex);

		glm::vec3 position = entity->transform.GetPosition();
		glm::vec3 rotation = entity->transform.GetRotation();
		glm::vec3 scale = entity->transform.GetScale();
		bool transformChanged = false;
		transformChanged |= ImGui::DragFloat3("Position", glm::value_ptr(position), 0.1f);
		transformChanged |= ImGui::DragFloat3("Rotation", glm::value_ptr(rotation), 0.5f);
		transformChanged |= ImGui::DragFloat3("Scale", glm::value_ptr(scale), 0.05f);

		if (transformChanged) SceneManager::UpdateEntityTransform(entity->id, Transform(position, rotation, scale));

		if (const auto model = ModelManager::GetModel(entity->model))
		{
			float boundsScale = model->GetCullingBoundsScale();
			if (ImGui::DragFloat("Culling Bounds Scale", &boundsScale, 0.02f, 0.01f, 100.0f, "%.2f"))
				model->SetCullingBoundsScale(boundsScale);
			ImGui::TextDisabled("Effective radius: %.3f", model->GetBoundsRadius());
			ImGui::SameLine();
			if (ImGui::SmallButton("Reset Bounds")) model->SetCullingBoundsScale(1.0f);
		}

		if (entity->type == "controller")
		{
			float controllerRadius = entity->controllerRadius;
			float controllerHeight = entity->controllerHeight;
			bool controllerSizeChanged = false;
			controllerSizeChanged |= ImGui::DragFloat("Controller Radius", &controllerRadius, 0.02f, 0.01f, 100.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
			controllerSizeChanged |= ImGui::DragFloat("Controller Height", &controllerHeight, 0.02f, 0.01f, 100.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
			if (controllerSizeChanged)
				SceneManager::UpdateControllerSize(entity->id, controllerRadius, controllerHeight);
			ImGui::TextDisabled("Total capsule height: %.2f", controllerHeight + controllerRadius * 2.0f);
			ImGui::Checkbox("Player", &entity->player);
		}
		else
		{
			ImGui::Checkbox("Interactable", &entity->interactable);
			ImGui::Checkbox("Pickable", &entity->pickable);
			if (entity->pickable) entity->interactable = true;
		}

		if (entity->interactable)
		{
			char itemName[128]{};
			const size_t copyLength = std::min(entity->itemName.size(), sizeof(itemName) - 1);
			entity->itemName.copy(itemName, copyLength);
			if (ImGui::InputText("Item Name", itemName, sizeof(itemName))) entity->itemName = itemName;
			ImGui::DragFloat("Interaction Range", &entity->interactionRange, 0.1f, 0.1f, 25.0f);
			ImGui::DragFloat("Label Height", &entity->labelHeight, 0.1f, -10.0f, 25.0f);
		}

		if (entity->type != "controller" && ImGui::Button("Duplicate / Instance"))
		{
			const uint64_t duplicateID = SceneManager::DuplicateEntity(entity->id);
			if (duplicateID != 0) s_Data.m_SelectedEntityID = duplicateID;
		}
		if (ImGui::Button("Remove Entity"))
		{
			if (SceneManager::RemoveEntity(selectedEntityID)) s_Data.m_SelectedEntityID = 0;
		}
	}
	else if (SceneLight* light = SceneManager::FindLight(s_Data.m_SelectedLightID))
	{
		ImGui::Separator();
		const char* typeName = light->type == LightType::DIRECT ? "Directional"
			: light->type == LightType::SPOT ? "Spot" : "Point";
		ImGui::Text("Light: %s", light->name.c_str());
		ImGui::Text("Type: %s", typeName);

		glm::vec3 color = light->color;
		glm::vec3 position = light->position;
		glm::vec3 rotation = light->rotation;
		bool lightChanged = false;
		lightChanged |= ImGui::ColorEdit3("Color", glm::value_ptr(color), ImGuiColorEditFlags_Float);
		if (light->type != LightType::DIRECT) lightChanged |= ImGui::DragFloat3("Position", glm::value_ptr(position), 0.1f);
		if (light->type != LightType::POINT) lightChanged |= ImGui::DragFloat3("Direction", glm::value_ptr(rotation), 0.05f);

		if (lightChanged) SceneManager::UpdateLight(light->id, light->name, color, position, rotation);

		if (ImGui::Button("Remove Light"))
		{
			const uint64_t removedID = light->id;
			if (SceneManager::RemoveLight(removedID)) s_Data.m_SelectedLightID = 0;
		}
	}

  DrawProfilerTree(gabprofiler_get_root());

	ImGui::End();

	if (debug.GBuffer)
	{
		if (!capabilities.OpenGLContext || !s_Data.m_GeometryBuffer)
		{
			ImGui::Begin("G-Buffer", &debug.GBuffer);
			ImGui::TextDisabled("G-buffer preview is available in the OpenGL renderer.");
			ImGui::End();
		}
		else
		{
			ImGui::Begin("G-Buffer", &debug.GBuffer);
			ImGui::TextDisabled("Position.rgb = world position, .a = material brightness");
			ImGui::TextDisabled("Normal.rgb = world normal, Albedo.a = specular");
			const float width = std::max(ImGui::GetContentRegionAvail().x, 160.0f);
			const float aspect = std::max(static_cast<float>(Window::GetWidth()), 1.0f) /
				std::max(static_cast<float>(Window::GetHeight()), 1.0f);
			const ImVec2 previewSize(width, width / aspect);
			const ImVec2 previewUV0{0.0f, 1.0f};
			const ImVec2 previewUV1{1.0f, 0.0f};
			auto drawAttachment = [&](const char* label, GLuint texture)
			{
				ImGui::SeparatorText(label);
				ImGui::Image(reinterpret_cast<ImTextureID>(static_cast<uintptr_t>(texture)),
					previewSize, previewUV0, previewUV1);
			};
			drawAttachment("Position + brightness",
				s_Data.m_GeometryBuffer->GetPositionAttachmentRendererID());
			drawAttachment("Normal",
				s_Data.m_GeometryBuffer->GetNormalAttachmentRendererID());
			drawAttachment("Albedo + specular",
				s_Data.m_GeometryBuffer->GetAlbedoSpecAttachmentRendererID());
			drawAttachment("Depth",
				s_Data.m_GeometryBuffer->GetDepthAttachmentRendererID());
			ImGui::End();
		}
	}

	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 0, 0 });
	ImGui::Begin("Viewport", nullptr, NULL);
	auto viewportMinRegion = ImGui::GetWindowContentRegionMin();
	auto viewportMaxRegion = ImGui::GetWindowContentRegionMax();
	auto viewportOffset = ImGui::GetWindowPos();
	s_Data.m_ViewportBounds[0] = { viewportMinRegion.x + viewportOffset.x, viewportMinRegion.y + viewportOffset.y };
	s_Data.m_ViewportBounds[1] = { viewportMaxRegion.x + viewportOffset.x, viewportMaxRegion.y + viewportOffset.y };

	s_Data.m_ViewportFocused = ImGui::IsWindowFocused();
	s_Data.m_ViewportHovered = ImGui::IsWindowHovered();

	BlockEvents(!s_Data.m_ViewportHovered);
	ImVec2 viewportPanelSize = ImGui::GetContentRegionAvail();
	s_Data.m_ViewportSize = { viewportPanelSize.x, viewportPanelSize.y };

	const auto textureID = reinterpret_cast<ImTextureID>(static_cast<uintptr_t>(framebufferTexture));
	const ImVec2 uv0 = capabilities.FramebufferOriginBottomLeft ? ImVec2{0, 1} : ImVec2{0, 0};
	const ImVec2 uv1 = capabilities.FramebufferOriginBottomLeft ? ImVec2{1, 0} : ImVec2{1, 1};
	if (framebufferTexture != 0)
		ImGui::Image(textureID, ImVec2{ s_Data.m_ViewportSize.x, s_Data.m_ViewportSize.y }, uv0, uv1);
	else
	{
		ImGui::Dummy(ImVec2{ s_Data.m_ViewportSize.x, s_Data.m_ViewportSize.y });
		ImGui::SetCursorPos(ImVec2{16.0f, 36.0f});
		ImGui::TextDisabled("Scene target is not implemented by %s yet.", RenderBackend::Get().GetName());
	}

	ImGui::End();
	ImGui::PopStyleVar();

	ImGui::End();
	io.DisplaySize = ImVec2(static_cast<float>(Window::GetWidth()), static_cast<float>(Window::GetHeight()));

	ImGui::Render();
	RenderBackend::Get().RenderImGuiDrawData();

	if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		RenderBackend::Get().RenderImGuiPlatformWindows();
}

bool OpenGLRenderer::DecomposeTransform(const glm::mat4& transform, glm::vec3& translation, glm::vec3& rotation, glm::vec3& scale)
{
  // From glm::decompose in matrix_decompose.inl

  using namespace glm;
  using T = float;

  mat4 LocalMatrix(transform);

  // Normalize the matrix.
  if (epsilonEqual(LocalMatrix[3][3], static_cast<float>(0), epsilon<T>()))
    return false;

  // First, isolate perspective.  This is the messiest.
  if (
    epsilonNotEqual(LocalMatrix[0][3], static_cast<T>(0), epsilon<T>()) ||
    epsilonNotEqual(LocalMatrix[1][3], static_cast<T>(0), epsilon<T>()) ||
    epsilonNotEqual(LocalMatrix[2][3], static_cast<T>(0), epsilon<T>()))
  {
    // Clear the perspective partition
    LocalMatrix[0][3] = LocalMatrix[1][3] = LocalMatrix[2][3] = static_cast<T>(0);
    LocalMatrix[3][3] = static_cast<T>(1);
  }

  // Next take care of translation (easy).
  translation = vec3(LocalMatrix[3]);
  LocalMatrix[3] = vec4(0, 0, 0, LocalMatrix[3].w);

  vec3 Row[3], Pdum3;

  // Now get scale and shear.
  for (length_t i = 0; i < 3; ++i)
    for (length_t j = 0; j < 3; ++j)
      Row[i][j] = LocalMatrix[i][j];

  // Compute X scale factor and normalize first row.
  scale.x = length(Row[0]);
  Row[0] = detail::scale(Row[0], static_cast<T>(1));
  scale.y = length(Row[1]);
  Row[1] = detail::scale(Row[1], static_cast<T>(1));
  scale.z = length(Row[2]);
  Row[2] = detail::scale(Row[2], static_cast<T>(1));

  // At this point, the matrix (in rows[]) is orthonormal.
  // Check for a coordinate system flip.  If the determinant
  // is -1, then negate the matrix and the scaling factors.
#if 0
  Pdum3 = cross(Row[1], Row[2]); // v3Cross(row[1], row[2], Pdum3);
  if (dot(Row[0], Pdum3) < 0)
  {
    for (length_t i = 0; i < 3; i++)
    {
      scale[i] *= static_cast<T>(-1);
      Row[i] *= static_cast<T>(-1);
    }
  }
#endif

  rotation.y = asin(-Row[0][2]);
  if (cos(rotation.y) != 0) {
    rotation.x = atan2(Row[1][2], Row[2][2]);
    rotation.z = atan2(Row[0][1], Row[0][0]);
  }
  else {
    rotation.x = atan2(-Row[2][0], Row[1][1]);
    rotation.z = 0;
  }


  return true;
}

