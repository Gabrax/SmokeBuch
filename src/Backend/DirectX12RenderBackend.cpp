#include "RenderBackendFactory.h"

#include "DirectX12Renderer.h"
#include "RenderBackend.h"

#include <imgui.h>

namespace
{
  class DirectX12RenderBackend final : public IRenderBackend
  {
  public:
    [[nodiscard]] GraphicsAPI GetAPI() const override { return GraphicsAPI::DirectX12; }
    [[nodiscard]] const char* GetName() const override { return "DirectX 12"; }
    [[nodiscard]] const RenderBackendCapabilities& GetCapabilities() const override
    {
      static constexpr RenderBackendCapabilities capabilities{
        .NativePresentation = true,
        .NativeSceneRenderer = true,
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

    bool InitializeDevice(void* nativeWindow, uint32_t width, uint32_t height) override
    {
      return DirectX12Renderer::Init(nativeWindow, width, height);
    }
    void ShutdownDevice() override { DirectX12Renderer::Shutdown(); }
    bool BeginFrame(uint32_t width, uint32_t height) override
    {
      return DirectX12Renderer::Resize(width, height) && DirectX12Renderer::BeginFrame();
    }
    bool EndFrame(bool vSync) override { return DirectX12Renderer::EndFrame(vSync); }

    bool InitializeSceneRenderer() override { return DirectX12Renderer::InitSceneRenderer(); }
    void ShutdownSceneRenderer() override { DirectX12Renderer::ShutdownSceneRenderer(); }
    void DrawScene(DeltaTime& dt, const std::function<void()>& sceneLogic,
                   bool advanceSimulation, bool renderForEditor,
                   const RenderEffectSettings& effects) override
    {
      DirectX12Renderer::DrawScene(dt, sceneLogic, advanceSimulation, renderForEditor, effects);
    }

    bool UploadModel(const std::shared_ptr<Model>& model) override { return DirectX12Renderer::UploadModel(model); }
    bool UploadSkybox(const std::shared_ptr<Texture>& cubemap) override { return DirectX12Renderer::UploadSkybox(cubemap); }
    void ResetSceneResources() override { DirectX12Renderer::ResetSceneResources(); }
    void RegisterModelDrawCommand(const std::string&, uint32_t, uint32_t) override {}
    void UpdateModelInstances(const std::shared_ptr<Model>&) override {}
    void SetModelRendered(const std::shared_ptr<Model>&, bool) override {}
    void SetModelPreviews(const std::vector<RenderModelPreview>& previews) override
    {
      DirectX12Renderer::SetModelPreviews(previews);
    }
    void FinalizeModelUpload() override {}
    void ResetModelDrawCommands() override {}
    bool DrawParticles(const std::vector<ParticleRenderInstance>& instances) override
    {
      DirectX12Renderer::DrawParticles(instances);
      return true;
    }
    uint64_t CreateFontAtlas(const uint8_t* pixels, uint32_t width,
                             uint32_t height) override
    {
      return DirectX12Renderer::CreateFontAtlas(pixels, width, height);
    }
    void DestroyFontAtlas(uint64_t handle) override
    {
      DirectX12Renderer::DestroyFontAtlas(handle);
    }
    bool BeginUI() override { DirectX12Renderer::BeginScene(); return true; }
    void PrepareScreenUI(const glm::vec4& clearColor, bool clear) override
    {
      DirectX12Renderer::PrepareScreenUI(clearColor, clear);
    }
    bool EndUI() override { DirectX12Renderer::EndScene(); return true; }
    bool DrawQuad(const glm::mat4& transform, const glm::vec4& color) override
    {
      DirectX12Renderer::DrawQuad(transform, color);
      return true;
    }
    bool DrawText(const Font* font, const std::string& text, const glm::vec2& position,
                  float size, const glm::vec4& color) override
    {
      DirectX12Renderer::DrawText(font, text, position, size, color);
      return true;
    }
    bool BeginDebugLines() override { DirectX12Renderer::BeginDebugLines(); return true; }
    bool DrawDebugLine(const glm::vec3& start, const glm::vec3& end, const glm::vec4& color) override
    {
      DirectX12Renderer::DrawDebugLine(start, end, color);
      return true;
    }
    bool EndDebugLines() override { DirectX12Renderer::EndDebugLines(); return true; }
    bool DrawPhysicsDebug() override { DirectX12Renderer::DrawPhysicsDebug(); return true; }

    bool InitializeImGuiRenderer() override { return DirectX12Renderer::InitImGui(); }
    void ShutdownImGuiRenderer() override { DirectX12Renderer::ShutdownImGui(); }
    void BeginImGuiFrame() override { DirectX12Renderer::BeginImGuiFrame(); }
    void RenderImGuiDrawData() override { DirectX12Renderer::RenderImGuiDrawData(); }
    void RenderImGuiPlatformWindows() override
    {
      ImGui::UpdatePlatformWindows();
      ImGui::RenderPlatformWindowsDefault();
    }
    [[nodiscard]] uint64_t GetEditorTextureID() const override { return DirectX12Renderer::GetEditorTextureID(); }
    void OnLightsChanged(const std::vector<RenderLight>&) override {}
  };
}

std::unique_ptr<IRenderBackend> CreateDirectX12RenderBackend()
{
#if defined(GABGL_ENABLE_DX12) && defined(_WIN32)
  return std::make_unique<DirectX12RenderBackend>();
#else
  return {};
#endif
}
