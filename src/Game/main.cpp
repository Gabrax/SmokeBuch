#include "GameBootstrap.h"

#include "AudioManager.h"
#include "ModelManager.h"
#include "PhysX.h"
#include "RenderBackend.h"
#include "RenderSystem.h"
#include "SceneManager.h"
#include "Settings.h"
#include "Window.h"

#include <gabdebug.h>

#include <thread>
#include <chrono>
#include <string_view>

// Prevent accidentally selecting integrated GPU
extern "C" {
	__declspec(dllexport) unsigned __int32 AmdPowerXpressRequestHighPerformance = 0x1;
	__declspec(dllexport) unsigned __int32 NvOptimusEnablement = 0x1;
}

namespace
{
  GraphicsAPI ParseGraphicsAPI(int argc, char** argv)
  {
    GraphicsAPI api = Settings::GetGraphicsAPI();
    for (int i = 1; i < argc; ++i)
    {
      if (const std::string_view argument(argv[i]); argument == "--dx12" || argument == "--renderer=dx12")
        api = GraphicsAPI::DirectX12;
      else if (argument == "--vulkan" || argument == "--renderer=vulkan")
        api = GraphicsAPI::Vulkan;
      else if (argument == "--opengl" || argument == "--renderer=opengl")
        api = GraphicsAPI::OpenGL;
      else if (argument == "--renderer" && i + 1 < argc)
      {
        if (const std::string_view value(argv[++i]); value == "dx12" || value == "directx12") api = GraphicsAPI::DirectX12;
        else if (value == "vulkan" || value == "vk") api = GraphicsAPI::Vulkan;
        else if (value == "opengl") api = GraphicsAPI::OpenGL;
      }
    }
    return api;
  }

  void LimitFrameRate(const std::chrono::steady_clock::time_point frameStart)
  {
    if (const uint32_t fpsLimit = Settings::GetFPSLimit(); !Settings::GetVSync() && fpsLimit > 0)
    {
      const auto targetFrameTime = std::chrono::duration<double>(1.0 / static_cast<double>(fpsLimit));
      if (const auto elapsed = std::chrono::steady_clock::now() - frameStart; elapsed < targetFrameTime)
        std::this_thread::sleep_for(targetFrameTime - elapsed);
    }
  }
}

int main(int argc, char** argv)
{
  gablog_set_level(LOG_TRACE);
  gablog_set_color_mode(GAB_COLOR_AUTO);
  Settings::Init();
  const GraphicsAPI graphicsAPI = ParseGraphicsAPI(argc, argv);
  if (!RenderBackend::Select(graphicsAPI))
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "Unsupported graphics API");
    return 1;
  }
  gablog_log(LOG_INFO, __FILE__, __LINE__, "Selected graphics API: %s", GraphicsAPIName(graphicsAPI));

  const std::string windowTitle = graphicsAPI == GraphicsAPI::OpenGL
    ? "GABGL" : std::string("GABGL - ") + RenderBackend::Get().GetName();
  Window::Init(windowTitle, Settings::GetWindowWidth(), Settings::GetWindowHeight(), graphicsAPI);
  if (!RenderBackend::Get().InitializeDevice(
        Window::GetNativeHandle(), Window::GetWidth(), Window::GetHeight()))
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "Could not initialize the %s backend", RenderBackend::Get().GetName());
    if (graphicsAPI == GraphicsAPI::DirectX12)
      gablog_log(LOG_ERROR, __FILE__, __LINE__, "Configure with -DGABGL_ENABLE_DX12=ON to include DirectX 12");
    else if (graphicsAPI == GraphicsAPI::Vulkan)
      gablog_log(LOG_ERROR, __FILE__, __LINE__, "Configure with -DGABGL_ENABLE_VULKAN=ON and a Vulkan SDK");
    Window::Terminate();
    return 1;
  }

  AudioManager::Init();
  PhysX::Init();
  RenderSystem::Initialize();
  ModelManager::Init();
  Game::Register();

  Window::SetEventCallback([](Event& event)
  {
    if (auto* scene = SceneManager::GetActiveScene()) scene->OnEvent(event);
  });

  AudioManager::SetMusicVolume(Settings::GetMusicVolume());
  AudioManager::SetSFXVolume(Settings::GetSFXVolume());
  RenderSystem::ApplyDisplaySettings();

  SceneManager::LoadScene("menu");

  while (Window::IsRunning())
  {
    const auto frameStart = std::chrono::steady_clock::now();
    Window::PollEvents();

    if (Window::IsMinimized())
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(16));
      continue;
    }

    if (!RenderBackend::Get().BeginFrame(Window::GetWidth(), Window::GetHeight()))
    {
      Window::RequestClose();
      continue;
    }

    DeltaTime dt;
    SceneManager::Update(dt);

    if (!RenderBackend::Get().EndFrame(Settings::GetVSync()))
      Window::RequestClose();

    LimitFrameRate(frameStart);
  }

  SceneManager::Shutdown();
  AudioManager::Terminate();
  ModelManager::Shutdown();
  RenderBackend::ClearLights();
  RenderSystem::Shutdown();
  PhysX::Shutdown();
  RenderBackend::Get().ShutdownDevice();
  Window::Terminate();

  return 0;
}
