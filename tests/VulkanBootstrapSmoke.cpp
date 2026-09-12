#include "RenderBackend.h"
#include "Window.h"

#include <cstdint>
#include <iostream>

int main()
{
  constexpr uint32_t InitialWidth = 640;
  constexpr uint32_t InitialHeight = 360;
  if (!RenderBackend::Select(GraphicsAPI::Vulkan)) return 1;
  Window::Init("GABGL Vulkan bootstrap smoke", InitialWidth, InitialHeight, GraphicsAPI::Vulkan);
  auto& backend = RenderBackend::Get();
  if (!backend.InitializeDevice(nullptr, InitialWidth, InitialHeight))
  {
    Window::Terminate();
    return 2;
  }

  bool passed = true;
  for (uint32_t frame = 0; frame < 24 && Window::IsRunning(); ++frame)
  {
    Window::PollEvents();
    if (frame == 8) Window::SetResolution(641, 359);
    if (frame == 16) Window::SetResolution(InitialWidth, InitialHeight);
    if (!backend.BeginFrame(Window::GetWidth(), Window::GetHeight()) ||
        !backend.EndFrame(frame >= 12))
    {
      passed = false;
      break;
    }
  }
  backend.ShutdownDevice();
  Window::Terminate();
  std::cout << "Vulkan bootstrap smoke " << (passed ? "PASS" : "FAIL") << '\n';
  return passed ? 0 : 3;
}
