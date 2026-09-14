#include "RenderBackendFactory.h"

#include "RenderBackend.h"
#include "VulkanRenderer.h"

namespace
{
  class VulkanRenderBackend final : public IRenderBackend
  {
  public:
    [[nodiscard]] GraphicsAPI GetAPI() const override { return GraphicsAPI::Vulkan; }
    [[nodiscard]] const char* GetName() const override { return "Vulkan"; }
    [[nodiscard]] const RenderBackendCapabilities& GetCapabilities() const override
    {
      static constexpr RenderBackendCapabilities capabilities{
        .NativePresentation = true,
        .NativeSceneRenderer = true,
        // Prevent legacy OpenGL upload calls while a Vulkan window has no GL context.
        .NativeModelResources = true,
        .NativeParticleRenderer = true,
        .NativeUIRenderer = true,
        .NativeDebugLines = true,
        .NativePhysicsDebug = true,
        .PointLightShadows = true,
        .TimestampProfiler = true
      };
      return capabilities;
    }

    bool InitializeDevice(void*, uint32_t width, uint32_t height) override
    {
      return VulkanRenderer::Init(width, height);
    }
    void ShutdownDevice() override { VulkanRenderer::Shutdown(); }
    bool BeginFrame(uint32_t width, uint32_t height) override
    {
      return VulkanRenderer::Resize(width, height) && VulkanRenderer::BeginFrame();
    }
    bool EndFrame(bool vSync) override { return VulkanRenderer::EndFrame(vSync); }

    bool InitializeSceneRenderer() override { return VulkanRenderer::InitSceneRenderer(); }
    void ShutdownSceneRenderer() override { VulkanRenderer::ShutdownSceneRenderer(); }
    void DrawScene(DeltaTime& dt, const std::function<void()>& sceneLogic,
                   bool advanceSimulation, bool renderForEditor,
                   const RenderEffectSettings& effects) override
    {
      VulkanRenderer::DrawScene(dt, sceneLogic, advanceSimulation, renderForEditor, effects);
    }

    bool UploadModel(const std::shared_ptr<Model>& model) override
    {
      return VulkanRenderer::UploadModel(model);
    }
    bool UploadSkybox(const std::shared_ptr<Texture>& cubemap) override
    {
      return VulkanRenderer::UploadSkybox(cubemap);
    }
    void ResetSceneResources() override { VulkanRenderer::ResetSceneResources(); }
    void RegisterModelDrawCommand(const std::string&, uint32_t, uint32_t) override {}
    void UpdateModelInstances(const std::shared_ptr<Model>&) override {}
    void SetModelRendered(const std::shared_ptr<Model>&, bool) override {}
    void SetModelPreviews(const std::vector<RenderModelPreview>& previews) override
    {
      VulkanRenderer::SetModelPreviews(previews);
    }
    void FinalizeModelUpload() override {}
    void ResetModelDrawCommands() override {}
    bool DrawParticles(const std::vector<ParticleRenderInstance>& instances) override
    {
      return VulkanRenderer::DrawParticles(instances);
    }
    uint64_t CreateFontAtlas(const uint8_t* pixels, uint32_t width, uint32_t height) override
    {
      return VulkanRenderer::CreateFontAtlas(pixels, width, height);
    }
    void DestroyFontAtlas(uint64_t handle) override
    {
      VulkanRenderer::DestroyFontAtlas(handle);
    }

    bool BeginUI() override { return VulkanRenderer::BeginScreenUI(); }
    void PrepareScreenUI(const glm::vec4& clearColor, bool clear) override
    {
      if (clear) VulkanRenderer::Clear(clearColor);
    }
    bool EndUI() override { return true; }
    bool DrawQuad(const glm::mat4& transform, const glm::vec4& color) override
    {
      return VulkanRenderer::DrawScreenQuad(transform, color);
    }
    bool DrawText(const Font* font, const std::string& text, const glm::vec2& position,
                  float size, const glm::vec4& color) override
    {
      return VulkanRenderer::DrawScreenText(font, text, position, size, color);
    }
    bool BeginDebugLines() override { return VulkanRenderer::BeginDebugLines(); }
    bool DrawDebugLine(const glm::vec3& start, const glm::vec3& end,
                       const glm::vec4& color) override
    {
      return VulkanRenderer::DrawDebugLine(start, end, color);
    }
    bool EndDebugLines() override { return VulkanRenderer::EndDebugLines(); }
    bool DrawPhysicsDebug() override { return VulkanRenderer::DrawPhysicsDebug(); }

    bool InitializeImGuiRenderer() override { return VulkanRenderer::InitImGui(); }
    void ShutdownImGuiRenderer() override { VulkanRenderer::ShutdownImGui(); }
    void BeginImGuiFrame() override { VulkanRenderer::BeginImGuiFrame(); }
    void RenderImGuiDrawData() override { VulkanRenderer::RenderImGuiDrawData(); }
    void RenderImGuiPlatformWindows() override { VulkanRenderer::RenderImGuiPlatformWindows(); }
    [[nodiscard]] uint64_t GetEditorTextureID() const override
    {
      return VulkanRenderer::GetEditorTextureID();
    }
    void OnLightsChanged(const std::vector<RenderLight>&) override {}
  };
}

std::unique_ptr<IRenderBackend> CreateVulkanRenderBackend()
{
#if defined(GABGL_ENABLE_VULKAN)
  return std::make_unique<VulkanRenderBackend>();
#else
  return {};
#endif
}
