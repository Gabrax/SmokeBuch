#pragma once

#include <memory>

class IRenderBackend;

std::unique_ptr<IRenderBackend> CreateOpenGLRenderBackend();
std::unique_ptr<IRenderBackend> CreateDirectX12RenderBackend();
std::unique_ptr<IRenderBackend> CreateVulkanRenderBackend();
