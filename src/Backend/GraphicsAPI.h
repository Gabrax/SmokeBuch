#pragma once

#include <cstdint>

enum class GraphicsAPI : uint32_t
{
  OpenGL = 0,
  DirectX12 = 1,
  Vulkan = 2
};

inline const char* GraphicsAPIName(GraphicsAPI api)
{
  switch (api)
  {
    case GraphicsAPI::Vulkan: return "Vulkan";
    case GraphicsAPI::DirectX12: return "DirectX 12";
    case GraphicsAPI::OpenGL:
    default: return "OpenGL";
  }
}

struct GraphicsAPIState
{
  static void Set(GraphicsAPI api) { s_ActiveAPI = api; }
  static GraphicsAPI Get() { return s_ActiveAPI; }
  static bool IsDirectX12() { return s_ActiveAPI == GraphicsAPI::DirectX12; }
  static bool IsVulkan() { return s_ActiveAPI == GraphicsAPI::Vulkan; }

private:
  static inline GraphicsAPI s_ActiveAPI = GraphicsAPI::OpenGL;
};
