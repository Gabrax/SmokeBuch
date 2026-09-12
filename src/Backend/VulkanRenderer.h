#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "RenderCommon.h"

struct DeltaTime;
struct Font;
struct Model;
struct ParticleRenderInstance;
struct Texture;

// Vulkan owns presentation and is deliberately isolated from the engine-facing
// renderer contract. It provides swapchain synchronization plus the native
// shadow, deferred scene, particle, post-process and UI passes.
struct VulkanRenderer
{
  static bool Init(uint32_t width, uint32_t height);
  static void Shutdown();
  static bool Resize(uint32_t width, uint32_t height);
  static bool BeginFrame();
  static bool EndFrame(bool vSync);

  static bool InitSceneRenderer();
  static void ShutdownSceneRenderer();
  static bool UploadModel(const std::shared_ptr<Model>& model);
  static bool UploadSkybox(const std::shared_ptr<Texture>& cubemap);
  static void ResetSceneResources();
  static void SetModelPreviews(const std::vector<RenderModelPreview>& previews);
  static void DrawScene(DeltaTime& dt, const std::function<void()>& sceneLogic,
                        bool advanceSimulation, bool renderForEditor,
                        const RenderEffectSettings& effects);
  static bool DrawParticles(const std::vector<ParticleRenderInstance>& instances);
  static bool BeginDebugLines();
  static bool DrawDebugLine(const glm::vec3& start, const glm::vec3& end,
                            const glm::vec4& color);
  static bool EndDebugLines();
  static bool DrawPhysicsDebug();
  static uint64_t CreateFontAtlas(const uint8_t* pixels, uint32_t width,
                                  uint32_t height);
  static void DestroyFontAtlas(uint64_t handle);

  static bool InitImGui();
  static void ShutdownImGui();
  static void BeginImGuiFrame();
  static void RenderImGuiDrawData();
  static void RenderImGuiPlatformWindows();
  [[nodiscard]] static uint64_t GetEditorTextureID();

  static bool BeginScreenUI();
  static bool DrawScreenQuad(const glm::mat4& transform, const glm::vec4& color);
  static bool DrawScreenText(const Font* font, const std::string& text,
                             const glm::vec2& position,
                             float size, const glm::vec4& color);
  static void Clear(const glm::vec4& color);
};
