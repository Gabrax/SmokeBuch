#include "RenderBackend.h"

#include "RenderBackendFactory.h"
#include "Settings.h"

#include <algorithm>
#include <stdexcept>

namespace
{
  std::unique_ptr<IRenderBackend> s_Backend;
  RenderDebugSettings s_DebugSettings;
  RenderStatistics s_Statistics;
  std::vector<RenderLight> s_Lights;

  void NotifyLightsChanged()
  {
    if (s_Backend) s_Backend->OnLightsChanged(s_Lights);
  }
}

bool RenderBackend::Select(const GraphicsAPI api)
{
  switch (api)
  {
    case GraphicsAPI::Vulkan: s_Backend = CreateVulkanRenderBackend(); break;
    case GraphicsAPI::DirectX12: s_Backend = CreateDirectX12RenderBackend(); break;
    case GraphicsAPI::OpenGL: s_Backend = CreateOpenGLRenderBackend(); break;
    default: return false;
  }
  if (!s_Backend) return false;
  GraphicsAPIState::Set(api);
  NotifyLightsChanged();
  return true;
}

IRenderBackend& RenderBackend::Get()
{
  if (!s_Backend) throw std::runtime_error("No graphics backend has been selected");
  return *s_Backend;
}

const RenderBackendCapabilities& RenderBackend::Capabilities() { return Get().GetCapabilities(); }

RenderEffectSettings RenderBackend::GetEffectSettings()
{
  RenderEffectSettings settings;
  settings.ShadowQuality = Settings::GetShadowQuality();
  settings.BloomQuality = Settings::GetBloomQuality();
  return settings;
}

RenderDebugSettings& RenderBackend::DebugSettings() { return s_DebugSettings; }
const RenderStatistics& RenderBackend::Statistics() { return s_Statistics; }
void RenderBackend::SetStatistics(const RenderStatistics& statistics) { s_Statistics = statistics; }

bool RenderBackend::AddLight(const RenderLight& light)
{
  if (light.Type == LightType::DIRECT &&
      std::ranges::any_of(s_Lights, [](const RenderLight& existing)
      {
        return existing.Type == LightType::DIRECT;
      }))
    return false;

  s_Lights.push_back(light);
  NotifyLightsChanged();
  return true;
}

bool RenderBackend::UpdateLight(const size_t index, const RenderLight& light)
{
  if (index >= s_Lights.size()) return false;
  if (light.Type == LightType::DIRECT)
  {
    for (size_t i = 0; i < s_Lights.size(); ++i)
      if (i != index && s_Lights[i].Type == LightType::DIRECT) return false;
  }
  s_Lights[index] = light;
  NotifyLightsChanged();
  return true;
}

bool RenderBackend::RemoveLight(const size_t index)
{
  if (index >= s_Lights.size()) return false;
  s_Lights.erase(s_Lights.begin() + static_cast<std::ptrdiff_t>(index));
  NotifyLightsChanged();
  return true;
}

void RenderBackend::ClearLights()
{
  s_Lights.clear();
  NotifyLightsChanged();
}

const std::vector<RenderLight>& RenderBackend::Lights() { return s_Lights; }
