// Opt-in integration test; requires a desktop and the repository's scene assets.
#include "AudioManager.h"
#include "Camera.h"
#include "FontManager.h"
#include "ModelManager.h"
#include "ParticleRenderer.h"
#include "PhysX.h"
#include "RenderBackend.h"
#include "RenderSystem.h"
#include "SceneManager.h"
#include "Settings.h"
#include "Window.h"

#include <gabdebug.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <string_view>

namespace
{
  bool HasGPUProfileTime(const GABProfileNode* node)
  {
    for (const GABProfileNode* current = node; current; current = current->nextSibling)
    {
      if (current->gpuTime > 0.0f || HasGPUProfileTime(current->firstChild))
        return true;
    }
    return false;
  }

  RenderEffectSettings effects;
  bool editor = false;
  bool menuStarted = false;
  struct SmokeScene final : Scene
  {
    SmokeScene() : Scene("game") {}
    void OnSceneStart() override {}
    void OnUpdate(DeltaTime& dt) override
    {
      RenderBackend::Get().DrawScene(dt, [] {}, true, editor, effects);
      RenderBackend::Get().BeginUI();
      RenderBackend::Get().DrawText(FontManager::GetFont("dpcomic"), "Vulkan UI atlas",
        glm::vec2(180.0f, 60.0f), 0.35f, glm::vec4(1.0f));
      RenderBackend::Get().EndUI();
    }
  };

  struct SmokeMenuScene final : Scene
  {
    SmokeMenuScene() : Scene("menu") {}
    void OnSceneStart() override { menuStarted = true; }
    void OnUpdate(DeltaTime&) override
    {
      RenderSystem::PrepareScreenUI(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
      RenderBackend::Get().BeginUI();
      RenderBackend::Get().DrawText(FontManager::GetFont("dpcomic"), "Menu transition",
        glm::vec2(640.0f, 360.0f), 0.35f, glm::vec4(1.0f));
      RenderBackend::Get().EndUI();
    }
  };
}

int main(int argc, char** argv)
{
  gablog_set_level(LOG_INFO);
  Settings::Init();
  GraphicsAPI api = GraphicsAPI::DirectX12;
  if (argc > 1 && std::string_view(argv[1]) == "--opengl") api = GraphicsAPI::OpenGL;
  else if (argc > 1 && std::string_view(argv[1]) == "--vulkan") api = GraphicsAPI::Vulkan;
  RenderBackend::Select(api);
  Window::Init("GABGL pipeline smoke", 1280, 720, api);
  auto& backend = RenderBackend::Get();
  if (!backend.InitializeDevice(Window::GetNativeHandle(), 1280, 720)) return 1;
  AudioManager::Init();
  PhysX::Init();
  RenderSystem::Initialize();
  if (!backend.InitializeSceneRenderer()) return 2;
  ModelManager::Init();
  SceneManager::RegisterScene("game", [] { return std::make_unique<SmokeScene>(); });
  SceneManager::RegisterScene("menu", [] { return std::make_unique<SmokeMenuScene>(); });
  SceneManager::LoadScene("game");
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(90);
  uint32_t renderedFrames = 0;
  uint32_t maximumVisible = 0;
  float maximumAspectError = 0.0f;
  bool windowModeTransitionsPassed = true;
  bool gpuProfilerPassed = false;
  bool menuRequested = false;
  bool passed = true;
  while (Window::IsRunning() && renderedFrames < 160 && std::chrono::steady_clock::now() < deadline)
  {
    Window::PollEvents();
    if (Window::IsMinimized()) continue;
    if (renderedFrames == 56)
    {
      Window::SetWindowMode(WindowMode::Borderless, 960, 720);
      windowModeTransitionsPassed &=
        glfwGetWindowAttrib(Window::GetWindowPtr(), GLFW_DECORATED) == GLFW_FALSE;
    }
    if (renderedFrames == 64)
    {
      Window::SetWindowMode(WindowMode::Windowed, 960, 720);
      windowModeTransitionsPassed &=
        glfwGetWindowMonitor(Window::GetWindowPtr()) == nullptr &&
        glfwGetWindowAttrib(Window::GetWindowPtr(), GLFW_DECORATED) == GLFW_TRUE;
    }
    if (renderedFrames == 112)
      Window::SetWindowMode(WindowMode::Windowed, 1280, 720);
    if (renderedFrames == 120 && !menuRequested)
    {
      SceneManager::LoadScene("menu");
      menuRequested = true;
    }
    if (!backend.BeginFrame(Window::GetWidth(), Window::GetHeight())) { passed = false; break; }
    const glm::mat4& projection = Camera::GetProjection();
    const float expectedAspect = static_cast<float>(Window::GetWidth()) /
      static_cast<float>(std::max(Window::GetHeight(), 1u));
    const float projectionAspect = std::abs(projection[1][1] / projection[0][0]);
    maximumAspectError = std::max(
      maximumAspectError, std::abs(projectionAspect - expectedAspect));
    DeltaTime dt(1.0f / 60.0f);
    const bool loaded = SceneManager::GetActiveScene() && !SceneManager::IsLoading();
    if (loaded)
    {
      effects.BloomQuality = static_cast<GraphicsQuality>((renderedFrames / 16) % 4);
      effects.ShadowQuality = static_cast<GraphicsQuality>((renderedFrames / 32) % 4);
      effects.PS1Enabled = (renderedFrames / 8) % 2 == 0;
      editor = renderedFrames >= 80 && renderedFrames < 112;
      if (renderedFrames == 24)
        backend.SetModelPreviews({{"pistol", glm::mat4(1.0f), 1.8f}});
      else if (renderedFrames == 40)
        backend.SetModelPreviews({});
      RenderBackend::DebugSettings().TiledLightingMode = (renderedFrames / 16) % 3;
      if (renderedFrames % 16 == 0)
        std::cout << "phase frame=" << renderedFrames << " bloom=" << int(effects.BloomQuality)
          << " shadows=" << int(effects.ShadowQuality) << " editor=" << editor << '\n';
      if (renderedFrames == 16)
        ParticleRenderer::EmitImpact(Camera::GetPosition() + Camera::GetForwardDirection() * 3.0f,
          glm::vec3(0, 1, 0), 20);
      ++renderedFrames;
    }
    SceneManager::Update(dt);
    gpuProfilerPassed |= HasGPUProfileTime(gabprofiler_get_root());
    if (editor && backend.GetAPI() == GraphicsAPI::Vulkan)
      passed &= backend.GetEditorTextureID() != 0;
    maximumVisible = std::max(maximumVisible, RenderBackend::Statistics().VisibleInstances);
    if (!backend.EndFrame(false)) { passed = false; break; }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  passed &= renderedFrames == 160 && maximumVisible > 0 &&
    maximumAspectError < 0.001f && windowModeTransitionsPassed && menuStarted &&
    SceneManager::GetActiveSceneName() == "menu" &&
    (!backend.GetCapabilities().TimestampProfiler || gpuProfilerPassed);
  std::cout << GraphicsAPIName(api) << " smoke " << (passed ? "PASS" : "FAIL")
    << ": frames=" << renderedFrames << " max-visible=" << maximumVisible
    << " max-aspect-error=" << maximumAspectError
    << " window-modes=" << (windowModeTransitionsPassed ? "PASS" : "FAIL")
    << " menu-return=" << (menuStarted ? "PASS" : "FAIL")
    << " gpu-profiler=" << (gpuProfilerPassed ? "PASS" : "FAIL") << '\n';
  SceneManager::Shutdown();
  AudioManager::Terminate();
  ModelManager::Shutdown();
  RenderBackend::ClearLights();
  RenderSystem::Shutdown();
  PhysX::Shutdown();
  backend.ShutdownDevice();
  Window::Terminate();
  return passed ? 0 : 3;
}
