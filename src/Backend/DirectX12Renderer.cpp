#include "DirectX12Renderer.h"

#if defined(GABGL_ENABLE_DX12) && defined(_WIN32)

#include "DirectX12ShadowMath.h"

#include "AudioManager.h"
#include "Camera.h"
#include "DeltaTime.hpp"
#include "FontManager.h"
#include "ModelManager.h"
#include "ParticleRenderer.h"
#include "PhysX.h"
#include "RenderBackend.h"
#include "Shader.h"
#include "Settings.h"
#include "Texture.h"
#include "Window.h"

#include <gabdebug.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifdef APIENTRY
#undef APIENTRY
#endif
#include <Windows.h>
#ifdef DrawText
#undef DrawText
#endif
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <imgui.h>
#include "backends/imgui_impl_dx12.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>

using Microsoft::WRL::ComPtr;

namespace
{
  constexpr uint32_t FrameCount = 2;
  constexpr uint32_t MaxDescriptors = 2048;
  constexpr uint32_t MaxSceneLights = 128;
  constexpr uint32_t BloomMipCount = 6;
  constexpr uint32_t MaxHiZMips = 16;
  constexpr uint32_t MaxShadowedPointLights = 4;
  constexpr uint32_t PointShadowFaceCount = 6;
  constexpr float PointShadowRadius = 20.0f;
  constexpr uint64_t ConstantBufferBytes = 64ull * 1024ull * 1024ull;
  constexpr uint64_t UIVertexBufferBytes = 8ull * 1024ull * 1024ull;
  constexpr uint32_t SceneRTVIndex = FrameCount;
  constexpr uint32_t BloomARTVIndex = FrameCount + 1;
  constexpr uint32_t PostProcessRTVIndex = FrameCount + 3;
  constexpr uint32_t SceneResultRTVIndex = FrameCount + 4;
  constexpr uint32_t GPositionRTVIndex = FrameCount + 5;
  constexpr uint32_t GNormalRTVIndex = FrameCount + 6;
  constexpr uint32_t GAlbedoSpecRTVIndex = FrameCount + 7;
  constexpr uint32_t PointShadowRTVBaseIndex = FrameCount + 8;
  constexpr uint32_t BloomMipRTVBase = PointShadowRTVBaseIndex + MaxShadowedPointLights * PointShadowFaceCount;
  constexpr DXGI_FORMAT BackBufferFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
  constexpr DXGI_FORMAT SceneColorFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;
  constexpr DXGI_FORMAT DepthFormat = DXGI_FORMAT_D32_FLOAT;

  struct UIVertex
  {
    glm::vec2 Position;
    glm::vec4 Color;
    glm::vec2 UV;
  };

  struct UIBatch
  {
    uint32_t DescriptorIndex = 0;
    uint32_t FirstVertex = 0;
    uint32_t VertexCount = 0;
  };

  struct GPUTexture
  {
    ComPtr<ID3D12Resource> Resource;
    uint32_t DescriptorIndex = 0;
  };

  struct GPUMesh
  {
    ComPtr<ID3D12Resource> VertexBuffer;
    ComPtr<ID3D12Resource> IndexBuffer;
    D3D12_VERTEX_BUFFER_VIEW VertexView{};
    D3D12_INDEX_BUFFER_VIEW IndexView{};
    uint32_t IndexCount = 0;
    uint32_t DiffuseDescriptorIndex = 0;
    uint32_t NormalDescriptorIndex = 0;
    uint32_t SpecularDescriptorIndex = 0;
    bool HasNormalMap = false;
    bool HasSpecularMap = false;
  };

  struct alignas(256) SceneConstants
  {
    glm::mat4 ViewProjection{1.0f};
    glm::mat4 Model{1.0f};
    glm::mat4 LightViewProjection{1.0f};
    std::array<glm::mat4, MAX_BONES> Bones{};
    glm::vec4 CameraPosition{0.0f};
    glm::vec4 MaterialFlags{0.0f};
    std::array<glm::vec4, MaxSceneLights> LightPositions{};
    std::array<glm::vec4, MaxSceneLights> LightDirections{};
    std::array<glm::vec4, MaxSceneLights> LightColors{};
    std::array<glm::vec4, MaxSceneLights> LightTypes{};
    glm::uvec4 LightCount{0};
    glm::vec4 Resolution{1.0f};
    glm::uvec4 MaterialIndices{0};
    glm::vec4 ShadowFilter{0};
  };

  struct alignas(256) SkyboxConstants
  {
    glm::mat4 ViewProjection{1.0f};
  };

  struct ParticleVertex
  {
    glm::vec3 Position;
    glm::vec4 Color;
    glm::vec2 LocalPosition;
    float IsSquare = 0.0f;
  };

  struct DebugLineVertex
  {
    glm::vec3 Position;
    glm::vec4 Color;
  };

  struct alignas(256) PhysicsDebugConstants
  {
    glm::mat4 ViewProjection{1.0f};
    glm::mat4 Model{1.0f};
    glm::vec4 Color{0.15f, 1.0f, 0.35f, 1.0f};
  };

  // ExecuteIndirect reads these fields in signature order, with a 64-byte stride.
  struct IndirectDraw
  {
    D3D12_GPU_VIRTUAL_ADDRESS Constants;
    D3D12_VERTEX_BUFFER_VIEW Vertex;
    D3D12_INDEX_BUFFER_VIEW Index;
    D3D12_DRAW_INDEXED_ARGUMENTS Draw;
    uint32_t Padding = 0;
  };
  static_assert(sizeof(IndirectDraw) == 64);
  static_assert(offsetof(IndirectDraw, Draw) == 40);

  struct DrawCullData
  {
    glm::vec4 Sphere;
    glm::uvec4 Instances; // source start, count, output start, allow occlusion
  };

  struct PointShadowCacheEntry
  {
    uint64_t CasterHash = 0;
    glm::vec3 LightPosition{0};
    uint32_t LightIndex = 0;
    bool Valid = false;
  };

  struct GPUFrameData
  {
    ComPtr<ID3D12Resource> Commands, Transforms, Count, StatisticsReadback;
    uint64_t CommandBytes = 0, TransformBytes = 0;
    D3D12_GPU_VIRTUAL_ADDRESS SourceCommands = 0, SourceTransforms = 0, CullData = 0;
    uint32_t DrawCount = 0;
    uint32_t RenderableInstances = 0;
    bool Culled = false, StatisticsReady = false;
  };

  struct FrameUploadData
  {
    ComPtr<ID3D12Resource> Constants;
    uint8_t* ConstantsMapped = nullptr;
    uint64_t ConstantsOffset = 0;
    ComPtr<ID3D12Resource> UIVertices;
    uint8_t* UIVerticesMapped = nullptr;
    uint64_t UIVerticesOffset = 0;
  };

  struct DirectX12Data
  {
    ComPtr<IDXGIFactory6> Factory;
    ComPtr<IDXGIAdapter1> Adapter;
    ComPtr<ID3D12Device> Device;
    ComPtr<ID3D12InfoQueue> InfoQueue;
    ComPtr<ID3D12CommandQueue> CommandQueue;
    ComPtr<IDXGISwapChain3> SwapChain;
    ComPtr<ID3D12DescriptorHeap> RTVHeap;
    ComPtr<ID3D12DescriptorHeap> DSVHeap;
    ComPtr<ID3D12DescriptorHeap> SRVHeap;
    std::array<ComPtr<ID3D12Resource>, FrameCount> RenderTargets;
    ComPtr<ID3D12Resource> DepthBuffer;
    std::array<ComPtr<ID3D12CommandAllocator>, FrameCount> CommandAllocators;
    ComPtr<ID3D12GraphicsCommandList> CommandList;
    ComPtr<ID3D12CommandAllocator> UploadAllocator;
    ComPtr<ID3D12GraphicsCommandList> UploadCommandList;
    ComPtr<ID3D12Fence> Fence;
    std::array<uint64_t, FrameCount> FrameFenceValues{};
    uint64_t NextFenceValue = 0;
    HANDLE FenceEvent = nullptr;

    ComPtr<ID3D12RootSignature> SceneRootSignature;
    ComPtr<ID3D12PipelineState> ScenePipeline;
    ComPtr<ID3D12PipelineState> LightingPipeline;
    ComPtr<ID3D12PipelineState> DepthPrepassPipeline;
    ComPtr<ID3D12PipelineState> GPUShadowPipeline, GPUPointShadowPipeline;
    ComPtr<ID3D12RootSignature> CullRootSignature, HiZRootSignature, TileRootSignature;
    ComPtr<ID3D12PipelineState> CullPipeline, HiZPipeline, TilePipeline;
    ComPtr<ID3D12CommandSignature> DrawSignature;
    std::array<GPUFrameData, FrameCount> GPUFrames;
    GPUTexture HiZ;
    uint32_t DepthDescriptor = 0, HiZUAVBase = 0, HiZSRVBase = 0, HiZMipCount = 0;
    ComPtr<ID3D12Resource> TileGrid, TileIndices;
    ComPtr<ID3D12Resource> ShadowOffsets;
    uint32_t TileCountX = 0, TileCountY = 0;
    std::array<GPUTexture, BloomMipCount> BloomMips;
    ComPtr<ID3D12PipelineState> BloomDownsamplePipeline, BloomUpsamplePipeline;
    ComPtr<ID3D12RootSignature> SkyboxRootSignature;
    ComPtr<ID3D12PipelineState> SkyboxPipeline;
    ComPtr<ID3D12RootSignature> ParticleRootSignature;
    ComPtr<ID3D12PipelineState> ParticlePipeline;
    ComPtr<ID3D12RootSignature> DebugLineRootSignature;
    ComPtr<ID3D12PipelineState> DebugLinePipeline;
    ComPtr<ID3D12RootSignature> PhysicsDebugRootSignature;
    ComPtr<ID3D12PipelineState> PhysicsDebugPipeline;
    ComPtr<ID3D12RootSignature> PostRootSignature;
    ComPtr<ID3D12PipelineState> BloomExtractPipeline;
    ComPtr<ID3D12PipelineState> CompositePipeline;
    ComPtr<ID3D12PipelineState> PostProcessPipeline;
    ComPtr<ID3D12RootSignature> UIRootSignature;
    ComPtr<ID3D12PipelineState> UIPipeline;
    ComPtr<ID3D12PipelineState> SceneUIPipeline;
    std::array<FrameUploadData, FrameCount> FrameUploads;

    GPUTexture WhiteTexture;
    GPUTexture SkyboxTexture;
    GPUTexture SceneColor;
    GPUTexture SceneResult;
    GPUTexture GPosition;
    GPUTexture GNormal;
    GPUTexture GAlbedoSpec;
    GPUTexture BloomA;
    GPUTexture PostProcessColor;
    GPUTexture ShadowMap;
    GPUTexture PointShadowMap;
    ComPtr<ID3D12Resource> PointShadowDepth;
    ComPtr<ID3D12Resource> SkyboxVertexBuffer;
    D3D12_VERTEX_BUFFER_VIEW SkyboxVertexView{};
    glm::mat4 LightViewProjection{1.0f};
    std::unordered_map<uint64_t, GPUTexture> FontAtlases;
    std::unordered_map<const Texture*, GPUTexture> Textures;
    std::unordered_map<const Mesh*, GPUMesh> Meshes;
    std::vector<ComPtr<ID3D12Resource>> RetiredResources;
    std::vector<UIVertex> PendingUIVertices;
    std::vector<UIBatch> PendingUIBatches;
    std::vector<DebugLineVertex> PendingDebugLines;
    std::vector<uint32_t> PointShadowLightIndices;
    std::array<PointShadowCacheEntry, MaxShadowedPointLights> PointShadowCache;
    std::vector<RenderModelPreview> ModelPreviews;
    uint32_t NextDescriptor = 0;
    uint32_t ImGuiFontDescriptor = std::numeric_limits<uint32_t>::max();
    uint32_t ShadowMapSize = 0;
    uint32_t PointShadowMapSize = 0;

    D3D12_VIEWPORT Viewport{};
    D3D12_RECT ScissorRect{};
    uint32_t Width = 0;
    uint32_t Height = 0;
    uint32_t FrameIndex = 0;
    uint32_t RTVDescriptorSize = 0;
    uint32_t SRVDescriptorSize = 0;
    bool TearingSupported = false;
    bool FrameStarted = false;
    bool FrameFailed = false;
    bool ShadowMapReadable = false;
    bool PointShadowMapReadable = false;
    bool ImGuiInitialized = false;
    bool SceneRendererInitialized = false;
    bool UIToSceneColor = false;
    bool Initialized = false;
  } s_Data;

  [[noreturn]] void ThrowHRESULT(HRESULT result, const char* operation)
  {
    std::ostringstream message;
    message << operation << " failed (HRESULT 0x" << std::hex << std::uppercase
            << static_cast<uint32_t>(result) << ')';
    throw std::runtime_error(message.str());
  }

  void CheckHRESULT(HRESULT result, const char* operation)
  {
    if (FAILED(result)) ThrowHRESULT(result, operation);
  }

  std::string WideToUTF8(const wchar_t* value)
  {
    if (!value || *value == L'\0') return {};
    const int requiredSize = WideCharToMultiByte(
      CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    if (requiredSize <= 1) return {};
    std::string result(static_cast<size_t>(requiredSize), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value, -1, result.data(), requiredSize, nullptr, nullptr);
    result.resize(static_cast<size_t>(requiredSize - 1));
    return result;
  }

  uint64_t AlignUp(uint64_t value, uint64_t alignment)
  {
    return (value + alignment - 1) & ~(alignment - 1);
  }

  D3D12_HEAP_PROPERTIES HeapProperties(D3D12_HEAP_TYPE type)
  {
    D3D12_HEAP_PROPERTIES properties{};
    properties.Type = type;
    properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    properties.CreationNodeMask = 1;
    properties.VisibleNodeMask = 1;
    return properties;
  }

  D3D12_RESOURCE_DESC BufferDescription(uint64_t size)
  {
    D3D12_RESOURCE_DESC description{};
    description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    description.Width = size;
    description.Height = 1;
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.Format = DXGI_FORMAT_UNKNOWN;
    description.SampleDesc.Count = 1;
    description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return description;
  }

  ComPtr<ID3D12Resource> CreateUploadBuffer(uint64_t size, uint8_t** mappedData = nullptr)
  {
    ComPtr<ID3D12Resource> resource;
    const D3D12_HEAP_PROPERTIES heap = HeapProperties(D3D12_HEAP_TYPE_UPLOAD);
    const D3D12_RESOURCE_DESC description = BufferDescription(size);
    CheckHRESULT(s_Data.Device->CreateCommittedResource(
                   &heap, D3D12_HEAP_FLAG_NONE, &description,
                   D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                   IID_PPV_ARGS(&resource)),
                 "ID3D12Device::CreateCommittedResource (upload buffer)");
    if (mappedData)
    {
      const D3D12_RANGE noRead{0, 0};
      CheckHRESULT(resource->Map(0, &noRead, reinterpret_cast<void**>(mappedData)),
                   "ID3D12Resource::Map");
    }
    return resource;
  }

  D3D12_CPU_DESCRIPTOR_HANDLE GetRTVHandle(uint32_t index)
  {
    D3D12_CPU_DESCRIPTOR_HANDLE handle = s_Data.RTVHeap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<SIZE_T>(index) * s_Data.RTVDescriptorSize;
    return handle;
  }

  D3D12_CPU_DESCRIPTOR_HANDLE GetDSVHandle(uint32_t index)
  {
    D3D12_CPU_DESCRIPTOR_HANDLE handle = s_Data.DSVHeap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<SIZE_T>(index) * s_Data.Device->GetDescriptorHandleIncrementSize(
      D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    return handle;
  }

  D3D12_CPU_DESCRIPTOR_HANDLE GetSRVCPUHandle(uint32_t index)
  {
    if (!s_Data.SRVHeap || index >= s_Data.NextDescriptor)
      throw std::runtime_error("Invalid DX12 SRV descriptor index");
    D3D12_CPU_DESCRIPTOR_HANDLE handle = s_Data.SRVHeap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<SIZE_T>(index) * s_Data.SRVDescriptorSize;
    return handle;
  }

  D3D12_GPU_DESCRIPTOR_HANDLE GetSRVGPUHandle(uint32_t index)
  {
    if (!s_Data.SRVHeap || index >= s_Data.NextDescriptor)
      throw std::runtime_error("Invalid DX12 SRV descriptor index");
    D3D12_GPU_DESCRIPTOR_HANDLE handle = s_Data.SRVHeap->GetGPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<UINT64>(index) * s_Data.SRVDescriptorSize;
    return handle;
  }

  uint32_t AllocateDescriptor()
  {
    if (s_Data.NextDescriptor >= MaxDescriptors)
      throw std::runtime_error("DX12 SRV descriptor heap is full");
    return s_Data.NextDescriptor++;
  }

  void TransitionResource(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                          D3D12_RESOURCE_STATES after)
  {
    if (!resource || before == after) return;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    s_Data.CommandList->ResourceBarrier(1, &barrier);
  }

  GPUTexture CreateColorTarget(uint32_t width, uint32_t height, uint32_t rtvIndex,
                               uint32_t descriptorIndex,
                               DXGI_FORMAT format = SceneColorFormat)
  {
    GPUTexture target;
    target.DescriptorIndex = descriptorIndex;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE clear{};
    clear.Format = format;
    const D3D12_HEAP_PROPERTIES heap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    CheckHRESULT(s_Data.Device->CreateCommittedResource(
                   &heap, D3D12_HEAP_FLAG_NONE, &desc,
                   D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &clear,
                   IID_PPV_ARGS(&target.Resource)),
                 "Create DX12 color target");
    s_Data.Device->CreateRenderTargetView(target.Resource.Get(), nullptr, GetRTVHandle(rtvIndex));
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Format = format;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Texture2D.MipLevels = 1;
    s_Data.Device->CreateShaderResourceView(
      target.Resource.Get(), &srv, GetSRVCPUHandle(descriptorIndex));
    return target;
  }

  void UpdateViewport(uint32_t width, uint32_t height)
  {
    s_Data.Viewport = {0.0f, 0.0f, static_cast<float>(width),
                      static_cast<float>(height), 0.0f, 1.0f};
    s_Data.ScissorRect = {0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
  }

  void CreateRenderTargets()
  {
    for (uint32_t i = 0; i < FrameCount; ++i)
    {
      CheckHRESULT(s_Data.SwapChain->GetBuffer(i, IID_PPV_ARGS(&s_Data.RenderTargets[i])),
                   "IDXGISwapChain::GetBuffer");
      s_Data.Device->CreateRenderTargetView(s_Data.RenderTargets[i].Get(), nullptr, GetRTVHandle(i));
    }
  }

  void CreateDepthBuffer(uint32_t width, uint32_t height)
  {
    D3D12_RESOURCE_DESC depthDesc{};
    depthDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    depthDesc.Width = width;
    depthDesc.Height = height;
    depthDesc.DepthOrArraySize = 1;
    depthDesc.MipLevels = 1;
    depthDesc.Format = DXGI_FORMAT_R32_TYPELESS;
    depthDesc.SampleDesc.Count = 1;
    depthDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    depthDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

    D3D12_CLEAR_VALUE clearValue{};
    clearValue.Format = DepthFormat;
    clearValue.DepthStencil.Depth = 1.0f;
    clearValue.DepthStencil.Stencil = 0;
    const D3D12_HEAP_PROPERTIES heap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    CheckHRESULT(s_Data.Device->CreateCommittedResource(
                   &heap, D3D12_HEAP_FLAG_NONE, &depthDesc,
                   D3D12_RESOURCE_STATE_DEPTH_WRITE, &clearValue,
                   IID_PPV_ARGS(&s_Data.DepthBuffer)),
                 "ID3D12Device::CreateCommittedResource (depth buffer)");

    D3D12_DEPTH_STENCIL_VIEW_DESC viewDesc{};
    viewDesc.Format = DepthFormat;
    viewDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    s_Data.Device->CreateDepthStencilView(
      s_Data.DepthBuffer.Get(), &viewDesc, GetDSVHandle(0));
  }

  void CreateGPUResources(uint32_t width, uint32_t height);

  void CreatePostProcessResources(uint32_t width, uint32_t height)
  {
    if (s_Data.SceneColor.DescriptorIndex == 0)
    {
      s_Data.SceneColor.DescriptorIndex = AllocateDescriptor();
      s_Data.BloomA.DescriptorIndex = AllocateDescriptor();
      s_Data.PostProcessColor.DescriptorIndex = AllocateDescriptor();
      s_Data.SceneResult.DescriptorIndex = AllocateDescriptor();
      s_Data.GPosition.DescriptorIndex = AllocateDescriptor();
      s_Data.GNormal.DescriptorIndex = AllocateDescriptor();
      s_Data.GAlbedoSpec.DescriptorIndex = AllocateDescriptor();
    }
    const uint32_t sceneDescriptor = s_Data.SceneColor.DescriptorIndex;
    const uint32_t bloomADescriptor = s_Data.BloomA.DescriptorIndex;
    const uint32_t postProcessDescriptor = s_Data.PostProcessColor.DescriptorIndex;
    const uint32_t sceneResultDescriptor = s_Data.SceneResult.DescriptorIndex;
    const uint32_t gPositionDescriptor = s_Data.GPosition.DescriptorIndex;
    const uint32_t gNormalDescriptor = s_Data.GNormal.DescriptorIndex;
    const uint32_t gAlbedoSpecDescriptor = s_Data.GAlbedoSpec.DescriptorIndex;
    s_Data.SceneColor = CreateColorTarget(width, height, SceneRTVIndex, sceneDescriptor);
    s_Data.BloomA = CreateColorTarget(width, height, BloomARTVIndex, bloomADescriptor);
    for (uint32_t i = 0; i < BloomMipCount; ++i)
    {
      auto& mip = s_Data.BloomMips[i];
      if (!mip.DescriptorIndex) mip.DescriptorIndex = AllocateDescriptor();
      mip = CreateColorTarget(std::max(1u, width >> (i + 1)),
        std::max(1u, height >> (i + 1)), BloomMipRTVBase + i, mip.DescriptorIndex,
        DXGI_FORMAT_R11G11B10_FLOAT);
    }
    CreateGPUResources(width, height);
    s_Data.PostProcessColor = CreateColorTarget(
      width, height, PostProcessRTVIndex, postProcessDescriptor, BackBufferFormat);
    s_Data.SceneResult = CreateColorTarget(
      width, height, SceneResultRTVIndex, sceneResultDescriptor, SceneColorFormat);
    s_Data.GPosition = CreateColorTarget(
      width, height, GPositionRTVIndex, gPositionDescriptor, SceneColorFormat);
    s_Data.GNormal = CreateColorTarget(
      width, height, GNormalRTVIndex, gNormalDescriptor, SceneColorFormat);
    s_Data.GAlbedoSpec = CreateColorTarget(
      width, height, GAlbedoSpecRTVIndex, gAlbedoSpecDescriptor, BackBufferFormat);
  }

  void CreateShadowMap(uint32_t size)
  {
    if (s_Data.ShadowMap.DescriptorIndex == 0)
      s_Data.ShadowMap.DescriptorIndex = AllocateDescriptor();
    const uint32_t descriptorIndex = s_Data.ShadowMap.DescriptorIndex;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = size;
    desc.Height = size;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R32_TYPELESS;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_CLEAR_VALUE clear{};
    clear.Format = DepthFormat;
    clear.DepthStencil.Depth = 1.0f;
    const D3D12_HEAP_PROPERTIES heap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    CheckHRESULT(s_Data.Device->CreateCommittedResource(
                   &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                   &clear, IID_PPV_ARGS(&s_Data.ShadowMap.Resource)),
                 "Create DX12 shadow map");
    D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
    dsv.Format = DepthFormat;
    dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    s_Data.Device->CreateDepthStencilView(s_Data.ShadowMap.Resource.Get(), &dsv, GetDSVHandle(1));
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Format = DXGI_FORMAT_R32_FLOAT;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Texture2D.MipLevels = 1;
    s_Data.Device->CreateShaderResourceView(
      s_Data.ShadowMap.Resource.Get(), &srv, GetSRVCPUHandle(descriptorIndex));
    s_Data.ShadowMapSize = size;
    // Same stratified disk kernel and 16x16 shadow-texel pattern as GL.
    const uint32_t filterSize = size >= 4096 ? 4u : (size >= 2048 ? 3u : 2u);
    std::default_random_engine generator;
    std::uniform_real_distribution<float> jitter(-0.5f, 0.5f);
    std::vector<glm::vec2> offsets;
    offsets.reserve(16 * 16 * filterSize * filterSize);
    for (uint32_t tileY = 0; tileY < 16; ++tileY)
      for (uint32_t tileX = 0; tileX < 16; ++tileX)
        for (int v = int(filterSize) - 1; v >= 0; --v)
          for (uint32_t u = 0; u < filterSize; ++u)
          {
            const float x = (float(u) + 0.5f + jitter(generator)) / filterSize;
            const float y = (float(v) + 0.5f + jitter(generator)) / filterSize;
            offsets.emplace_back(std::sqrt(y) * std::cos(glm::two_pi<float>() * x),
                                 std::sqrt(y) * std::sin(glm::two_pi<float>() * x));
          }
    uint8_t* mapped = nullptr;
    s_Data.ShadowOffsets = CreateUploadBuffer(offsets.size() * sizeof(glm::vec2), &mapped);
    std::memcpy(mapped, offsets.data(), offsets.size() * sizeof(glm::vec2));
    s_Data.ShadowOffsets->Unmap(0, nullptr);
  }

  void CreatePointShadowMap(uint32_t size)
  {
    s_Data.PointShadowCache = {};
    if (s_Data.PointShadowMap.DescriptorIndex == 0)
      s_Data.PointShadowMap.DescriptorIndex = AllocateDescriptor();
    const uint32_t descriptorIndex = s_Data.PointShadowMap.DescriptorIndex;

    D3D12_RESOURCE_DESC textureDesc{};
    textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    textureDesc.Width = size;
    textureDesc.Height = size;
    textureDesc.DepthOrArraySize = static_cast<UINT16>(
      MaxShadowedPointLights * PointShadowFaceCount);
    textureDesc.MipLevels = 1;
    textureDesc.Format = DXGI_FORMAT_R32_FLOAT;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    textureDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE colorClear{};
    colorClear.Format = DXGI_FORMAT_R32_FLOAT;
    colorClear.Color[0] = PointShadowRadius;
    colorClear.Color[1] = PointShadowRadius;
    colorClear.Color[2] = PointShadowRadius;
    colorClear.Color[3] = PointShadowRadius;
    const D3D12_HEAP_PROPERTIES defaultHeap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    CheckHRESULT(s_Data.Device->CreateCommittedResource(
                   &defaultHeap, D3D12_HEAP_FLAG_NONE, &textureDesc,
                   D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &colorClear,
                   IID_PPV_ARGS(&s_Data.PointShadowMap.Resource)),
                 "Create DX12 point shadow cube array");

    for (uint32_t slice = 0;
         slice < MaxShadowedPointLights * PointShadowFaceCount; ++slice)
    {
      D3D12_RENDER_TARGET_VIEW_DESC rtv{};
      rtv.Format = DXGI_FORMAT_R32_FLOAT;
      rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
      rtv.Texture2DArray.MipSlice = 0;
      rtv.Texture2DArray.FirstArraySlice = slice;
      rtv.Texture2DArray.ArraySize = 1;
      s_Data.Device->CreateRenderTargetView(
        s_Data.PointShadowMap.Resource.Get(), &rtv,
        GetRTVHandle(PointShadowRTVBaseIndex + slice));
    }

    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Format = DXGI_FORMAT_R32_FLOAT;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
    srv.TextureCubeArray.MostDetailedMip = 0;
    srv.TextureCubeArray.MipLevels = 1;
    srv.TextureCubeArray.First2DArrayFace = 0;
    srv.TextureCubeArray.NumCubes = MaxShadowedPointLights;
    s_Data.Device->CreateShaderResourceView(
      s_Data.PointShadowMap.Resource.Get(), &srv, GetSRVCPUHandle(descriptorIndex));

    D3D12_RESOURCE_DESC depthDesc{};
    depthDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    depthDesc.Width = size;
    depthDesc.Height = size;
    depthDesc.DepthOrArraySize = 1;
    depthDesc.MipLevels = 1;
    depthDesc.Format = DXGI_FORMAT_R32_TYPELESS;
    depthDesc.SampleDesc.Count = 1;
    depthDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    depthDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_CLEAR_VALUE depthClear{};
    depthClear.Format = DepthFormat;
    depthClear.DepthStencil.Depth = 1.0f;
    CheckHRESULT(s_Data.Device->CreateCommittedResource(
                   &defaultHeap, D3D12_HEAP_FLAG_NONE, &depthDesc,
                   D3D12_RESOURCE_STATE_DEPTH_WRITE, &depthClear,
                   IID_PPV_ARGS(&s_Data.PointShadowDepth)),
                 "Create DX12 point shadow depth buffer");
    D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
    dsv.Format = DepthFormat;
    dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    s_Data.Device->CreateDepthStencilView(
      s_Data.PointShadowDepth.Get(), &dsv, GetDSVHandle(2));

    s_Data.PointShadowMapSize = size;
    s_Data.PointShadowMapReadable = true;
  }

  void ReleaseRenderTargets()
  {
    for (auto& renderTarget : s_Data.RenderTargets) renderTarget.Reset();
    s_Data.DepthBuffer.Reset();
  }

  void WaitForGPU()
  {
    if (!s_Data.CommandQueue || !s_Data.Fence || !s_Data.FenceEvent) return;
    const uint64_t fenceValue = ++s_Data.NextFenceValue;
    CheckHRESULT(s_Data.CommandQueue->Signal(s_Data.Fence.Get(), fenceValue),
                 "ID3D12CommandQueue::Signal");
    if (s_Data.Fence->GetCompletedValue() < fenceValue)
    {
      CheckHRESULT(s_Data.Fence->SetEventOnCompletion(fenceValue, s_Data.FenceEvent),
                   "ID3D12Fence::SetEventOnCompletion");
      WaitForSingleObject(s_Data.FenceEvent, INFINITE);
    }
  }

  void WaitForFrame(uint32_t frameIndex)
  {
    const uint64_t fenceValue = s_Data.FrameFenceValues[frameIndex];
    if (fenceValue == 0 || s_Data.Fence->GetCompletedValue() >= fenceValue) return;
    CheckHRESULT(s_Data.Fence->SetEventOnCompletion(fenceValue, s_Data.FenceEvent),
                 "ID3D12Fence::SetEventOnCompletion");
    WaitForSingleObject(s_Data.FenceEvent, INFINITE);
  }

  void LogD3D12Messages()
  {
    if (!s_Data.InfoQueue) return;
    const uint64_t messageCount = s_Data.InfoQueue->GetNumStoredMessagesAllowedByRetrievalFilter();
    for (uint64_t i = 0; i < messageCount; ++i)
    {
      SIZE_T size = 0;
      s_Data.InfoQueue->GetMessage(i, nullptr, &size);
      if (size == 0) continue;
      std::vector<uint8_t> storage(size);
      auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
      if (SUCCEEDED(s_Data.InfoQueue->GetMessage(i, message, &size)) &&
          message->Severity <= D3D12_MESSAGE_SEVERITY_WARNING)
      {
        if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR)
        {
          s_Data.FrameFailed = true;
          gablog_log(LOG_ERROR, __FILE__, __LINE__, "D3D12 validation: %s", message->pDescription);
        }
        else
          gablog_log(LOG_WARN, __FILE__, __LINE__, "D3D12 validation: %s", message->pDescription);
      }
    }
    s_Data.InfoQueue->ClearStoredMessages();
  }

  Shader::Bytecode CompileSlang(const char* path, const char* entryPoint, const char* target)
  {
    return Shader::CompileSlang(path, entryPoint, target);
  }

  ComPtr<ID3D12RootSignature> CreateRootSignature(const D3D12_ROOT_SIGNATURE_DESC& description)
  {
    ComPtr<ID3DBlob> serialized;
    ComPtr<ID3DBlob> errors;
    const HRESULT result = D3D12SerializeRootSignature(
      &description, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &errors);
    if (FAILED(result))
    {
      if (errors)
        gablog_log(LOG_ERROR, __FILE__, __LINE__, "DX12 root signature creation failed: %s",
                    static_cast<const char*>(errors->GetBufferPointer()));
      ThrowHRESULT(result, "D3D12SerializeRootSignature");
    }
    ComPtr<ID3D12RootSignature> rootSignature;
    CheckHRESULT(s_Data.Device->CreateRootSignature(
                   0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                   IID_PPV_ARGS(&rootSignature)),
                 "ID3D12Device::CreateRootSignature");
    return rootSignature;
  }

  D3D12_RASTERIZER_DESC DefaultRasterizer()
  {
    D3D12_RASTERIZER_DESC state{};
    state.FillMode = D3D12_FILL_MODE_SOLID;
    state.CullMode = D3D12_CULL_MODE_NONE;
    state.FrontCounterClockwise = FALSE;
    state.DepthBias = D3D12_DEFAULT_DEPTH_BIAS;
    state.DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
    state.SlopeScaledDepthBias = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;
    state.DepthClipEnable = TRUE;
    return state;
  }

  D3D12_BLEND_DESC DefaultBlendState()
  {
    D3D12_BLEND_DESC state{};
    state.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    return state;
  }

  void DrawDebugWireSphere(const glm::vec3& center, float radius, const glm::vec4& color,
                           int segments = 24)
  {
    if (radius <= 0.0f || segments < 3) return;
    for (int i = 0; i < segments; ++i)
    {
      const float angle0 = glm::two_pi<float>() * static_cast<float>(i) / static_cast<float>(segments);
      const float angle1 = glm::two_pi<float>() * static_cast<float>(i + 1) / static_cast<float>(segments);
      const float c0 = std::cos(angle0), s0 = std::sin(angle0);
      const float c1 = std::cos(angle1), s1 = std::sin(angle1);
      DirectX12Renderer::DrawDebugLine(center + glm::vec3(c0, s0, 0.0f) * radius,
                                       center + glm::vec3(c1, s1, 0.0f) * radius, color);
      DirectX12Renderer::DrawDebugLine(center + glm::vec3(c0, 0.0f, s0) * radius,
                                       center + glm::vec3(c1, 0.0f, s1) * radius, color);
      DirectX12Renderer::DrawDebugLine(center + glm::vec3(0.0f, c0, s0) * radius,
                                       center + glm::vec3(0.0f, c1, s1) * radius, color);
    }
  }

  void DrawDebugWireCapsule(const glm::vec3& center, float radius, float height,
                            const glm::vec3& upDirection, const glm::vec4& color, int segments = 24)
  {
    if (radius <= 0.0f || height < 0.0f || segments < 4) return;
    const glm::vec3 up = glm::length(upDirection) > 0.0001f
      ? glm::normalize(upDirection) : glm::vec3(0.0f, 1.0f, 0.0f);
    const glm::vec3 fallbackAxis = std::abs(glm::dot(up, glm::vec3(0.0f, 1.0f, 0.0f))) > 0.98f
      ? glm::vec3(1.0f, 0.0f, 0.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
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
      DirectX12Renderer::DrawDebugLine(topCenter + radial0 * radius, topCenter + radial1 * radius, color);
      DirectX12Renderer::DrawDebugLine(bottomCenter + radial0 * radius, bottomCenter + radial1 * radius, color);
    }
    constexpr int meridians = 8;
    const int arcSegments = std::max(4, segments / 4);
    for (int meridian = 0; meridian < meridians; ++meridian)
    {
      const float angle = glm::two_pi<float>() * static_cast<float>(meridian) / static_cast<float>(meridians);
      const glm::vec3 radial = right * std::cos(angle) + forward * std::sin(angle);
      DirectX12Renderer::DrawDebugLine(bottomCenter + radial * radius, topCenter + radial * radius, color);
      for (int arc = 0; arc < arcSegments; ++arc)
      {
        const float arc0 = glm::half_pi<float>() * static_cast<float>(arc) / static_cast<float>(arcSegments);
        const float arc1 = glm::half_pi<float>() * static_cast<float>(arc + 1) / static_cast<float>(arcSegments);
        DirectX12Renderer::DrawDebugLine(
          topCenter + (radial * std::cos(arc0) + up * std::sin(arc0)) * radius,
          topCenter + (radial * std::cos(arc1) + up * std::sin(arc1)) * radius, color);
        DirectX12Renderer::DrawDebugLine(
          bottomCenter + (radial * std::cos(arc0) - up * std::sin(arc0)) * radius,
          bottomCenter + (radial * std::cos(arc1) - up * std::sin(arc1)) * radius, color);
      }
    }
  }

  void DrawDebugVisualizations()
  {
    const RenderDebugSettings& debug = RenderBackend::DebugSettings();
    if (!debug.Physics && !debug.Lights && !debug.CullingBounds) return;

    DirectX12Renderer::BeginDebugLines();
    if (debug.Physics)
    {
      constexpr glm::vec4 controllerColor(0.15f, 0.8f, 1.0f, 0.9f);
      for (const std::string& modelName : ModelManager::GetModelNames())
      {
        const auto model = ModelManager::GetModel(modelName);
        PxController* controller = model ? model->GetController() : nullptr;
        if (!controller || controller->getType() != PxControllerShapeType::eCAPSULE) continue;
        auto* capsule = static_cast<PxCapsuleController*>(controller);
        const PxExtendedVec3 position = controller->getPosition();
        const PxVec3 up = controller->getUpDirection();
        DrawDebugWireCapsule(
          glm::vec3(static_cast<float>(position.x), static_cast<float>(position.y), static_cast<float>(position.z)),
          capsule->getRadius(), capsule->getHeight(), glm::vec3(up.x, up.y, up.z), controllerColor);
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
          DrawDebugWireSphere(sphere.center, sphere.radius,
                              visible ? glm::vec4(0.15f, 1.0f, 0.3f, 0.8f)
                                      : glm::vec4(1.0f, 0.2f, 0.15f, 0.8f));
        }
      }
    }

    if (debug.Lights)
    {
      for (const RenderLight& light : RenderBackend::Lights())
      {
        const glm::vec4 color(light.Color, 0.9f);
        if (light.Type == LightType::DIRECT)
        {
          const glm::vec3 origin = Camera::GetPosition() + Camera::GetForwardDirection() * 8.0f;
          const glm::vec3 direction = glm::length(light.Direction) > 0.0001f
            ? glm::normalize(light.Direction) : glm::vec3(0.0f, -1.0f, 0.0f);
          DrawDebugWireSphere(origin, 0.5f, color);
          DirectX12Renderer::DrawDebugLine(origin, origin + direction * 12.0f, color);
        }
        else if (light.Type == LightType::POINT)
        {
          DrawDebugWireSphere(light.Position, 0.5f, color);
          DrawDebugWireSphere(light.Position, PointShadowRadius, glm::vec4(light.Color, 0.2f), 32);
          DirectX12Renderer::DrawDebugLine(light.Position - glm::vec3(1.0f, 0.0f, 0.0f), light.Position + glm::vec3(1.0f, 0.0f, 0.0f), color);
          DirectX12Renderer::DrawDebugLine(light.Position - glm::vec3(0.0f, 1.0f, 0.0f), light.Position + glm::vec3(0.0f, 1.0f, 0.0f), color);
          DirectX12Renderer::DrawDebugLine(light.Position - glm::vec3(0.0f, 0.0f, 1.0f), light.Position + glm::vec3(0.0f, 0.0f, 1.0f), color);
        }
        else
        {
          const glm::vec3 direction = glm::length(light.Direction) > 0.0001f
            ? glm::normalize(light.Direction) : glm::vec3(0.0f, -1.0f, 0.0f);
          const glm::vec3 fallbackUp = std::abs(glm::dot(direction, glm::vec3(0.0f, 1.0f, 0.0f))) > 0.98f
            ? glm::vec3(1.0f, 0.0f, 0.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
          const glm::vec3 right = glm::normalize(glm::cross(direction, fallbackUp));
          const glm::vec3 up = glm::normalize(glm::cross(right, direction));
          const glm::vec3 end = light.Position + direction * 8.0f;
          constexpr int segments = 20;
          for (int i = 0; i < segments; ++i)
          {
            constexpr float coneRadius = 3.0f;
            const float angle0 = glm::two_pi<float>() * static_cast<float>(i) / static_cast<float>(segments);
            const float angle1 = glm::two_pi<float>() * static_cast<float>(i + 1) / static_cast<float>(segments);
            const glm::vec3 p0 = end + (right * std::cos(angle0) + up * std::sin(angle0)) * coneRadius;
            const glm::vec3 p1 = end + (right * std::cos(angle1) + up * std::sin(angle1)) * coneRadius;
            DirectX12Renderer::DrawDebugLine(p0, p1, color);
            if (i % 5 == 0) DirectX12Renderer::DrawDebugLine(light.Position, p0, color);
          }
          DrawDebugWireSphere(light.Position, 0.4f, color);
          DirectX12Renderer::DrawDebugLine(light.Position, end, color);
        }
      }
    }
    DirectX12Renderer::EndDebugLines();
  }

  void DrawDebug2D()
  {
    if (!RenderBackend::DebugSettings().Debug2D) return;
    const glm::mat4 transform = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(141.0f, 37.0f, 0.0f)),
                                           glm::vec3(250.0f, 42.0f, 1.0f));
    DirectX12Renderer::BeginScene();
    DirectX12Renderer::DrawQuad(transform, glm::vec4(0.02f, 0.04f, 0.07f, 0.78f));
    DirectX12Renderer::DrawText(FontManager::GetFont("dpcomic"), "2D DEBUG RENDER",
                                glm::vec2(141.0f, 37.0f), 0.28f,
                                glm::vec4(0.45f, 0.92f, 1.0f, 1.0f));
    DirectX12Renderer::EndScene();
  }

  void CreateScenePipeline()
  {
    static constexpr const char* SceneShader = "../res/shaders/scene.slang";

    const auto vertexShader = CompileSlang(SceneShader, "VSMain", "vs_5_1");
    const auto geometryPixelShader = CompileSlang(SceneShader, "GBufferPS", "ps_5_1");
    const auto lightingVertexShader = CompileSlang(SceneShader, "LightingVS", "vs_5_1");
    const auto lightingPixelShader = CompileSlang(SceneShader, "LightingPS", "ps_5_1");

    std::array<D3D12_DESCRIPTOR_RANGE, 5> textureRanges{};
    for (uint32_t i = 0; i < textureRanges.size(); ++i)
    {
      textureRanges[i].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
      textureRanges[i].NumDescriptors = 1;
      textureRanges[i].BaseShaderRegister = i;
      textureRanges[i].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    }

    std::array<D3D12_ROOT_PARAMETER, 12> parameters{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    parameters[0].Descriptor.ShaderRegister = 0;
    parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    for (uint32_t i = 0; i < textureRanges.size(); ++i)
    {
      parameters[i + 1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      parameters[i + 1].DescriptorTable.NumDescriptorRanges = 1;
      parameters[i + 1].DescriptorTable.pDescriptorRanges = &textureRanges[i];
      parameters[i + 1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    }

    parameters[6].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    parameters[6].Descriptor.ShaderRegister = 5;
    parameters[6].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    D3D12_DESCRIPTOR_RANGE materials{};
    materials.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    materials.NumDescriptors = MaxDescriptors;
    materials.RegisterSpace = 1;
    parameters[7].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[7].DescriptorTable = {1, &materials};
    parameters[7].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    for (uint32_t i = 8; i < 10; ++i)
    {
      parameters[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
      parameters[i].Descriptor.ShaderRegister = i - 2;
      parameters[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    }
    parameters[10].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    parameters[10].Descriptor.ShaderRegister = 1;
    parameters[11].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    parameters[11].Descriptor.ShaderRegister = 8;
    parameters[11].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    std::array<D3D12_STATIC_SAMPLER_DESC, 2> samplers{};
    samplers[0].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    samplers[0].AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samplers[0].AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samplers[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samplers[0].MaxLOD = D3D12_FLOAT32_MAX;
    samplers[0].ShaderRegister = 0;
    samplers[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    samplers[1].Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    samplers[1].AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplers[1].AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplers[1].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samplers[1].MaxLOD = D3D12_FLOAT32_MAX;
    samplers[1].ShaderRegister = 1;
    samplers[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_ROOT_SIGNATURE_DESC rootDesc{};
    rootDesc.NumParameters = static_cast<UINT>(parameters.size());
    rootDesc.pParameters = parameters.data();
    rootDesc.NumStaticSamplers = static_cast<UINT>(samplers.size());
    rootDesc.pStaticSamplers = samplers.data();
    rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    s_Data.SceneRootSignature = CreateRootSignature(rootDesc);

    static constexpr D3D12_INPUT_ELEMENT_DESC InputLayout[] = {
      {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, offsetof(Vertex, Position),
       D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, offsetof(Vertex, Normal),
       D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, offsetof(Vertex, TexCoords),
       D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, offsetof(Vertex, Tangent),
       D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"BITANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, offsetof(Vertex, Bitangent),
       D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"BONEIDS", 0, DXGI_FORMAT_R32G32B32A32_SINT, 0, offsetof(Vertex, m_BoneIDs),
       D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"BONEWEIGHTS", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, offsetof(Vertex, m_Weights),
       D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline{};
    pipeline.pRootSignature = s_Data.SceneRootSignature.Get();
    pipeline.VS = {vertexShader.GetBufferPointer(), vertexShader.GetBufferSize()};
    pipeline.PS = {geometryPixelShader.GetBufferPointer(), geometryPixelShader.GetBufferSize()};
    pipeline.BlendState = DefaultBlendState();
    pipeline.SampleMask = std::numeric_limits<UINT>::max();
    pipeline.RasterizerState = DefaultRasterizer();
    pipeline.DepthStencilState.DepthEnable = TRUE;
    pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    pipeline.DepthStencilState.StencilEnable = FALSE;
    pipeline.InputLayout = {InputLayout, static_cast<UINT>(std::size(InputLayout))};
    pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pipeline.NumRenderTargets = 3;
    pipeline.RTVFormats[0] = SceneColorFormat;
    pipeline.RTVFormats[1] = SceneColorFormat;
    pipeline.RTVFormats[2] = BackBufferFormat;
    pipeline.DSVFormat = DepthFormat;
    pipeline.SampleDesc.Count = 1;
    CheckHRESULT(s_Data.Device->CreateGraphicsPipelineState(
                   &pipeline, IID_PPV_ARGS(&s_Data.ScenePipeline)),
                 "ID3D12Device::CreateGraphicsPipelineState (scene)");

    pipeline.PS = {};
    pipeline.NumRenderTargets = 0;
    std::fill(std::begin(pipeline.RTVFormats), std::end(pipeline.RTVFormats), DXGI_FORMAT_UNKNOWN);
    pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    pipeline.RasterizerState.DepthBias = 1;
    CheckHRESULT(s_Data.Device->CreateGraphicsPipelineState(
      &pipeline, IID_PPV_ARGS(&s_Data.DepthPrepassPipeline)), "Create DX12 depth prepass");

    const auto shadowVS = CompileSlang(SceneShader, "ShadowVS", "vs_5_1");
    const auto pointPS = CompileSlang(SceneShader, "PointShadowPS", "ps_5_1");
    pipeline.VS = {shadowVS.GetBufferPointer(), shadowVS.GetBufferSize()};
    pipeline.RasterizerState.DepthBias = 0;
    CheckHRESULT(s_Data.Device->CreateGraphicsPipelineState(
      &pipeline, IID_PPV_ARGS(&s_Data.GPUShadowPipeline)), "Create DX12 GPU shadow pipeline");
    pipeline.NumRenderTargets = 1;
    pipeline.RTVFormats[0] = DXGI_FORMAT_R32_FLOAT;
    pipeline.PS = {pointPS.GetBufferPointer(), pointPS.GetBufferSize()};
    CheckHRESULT(s_Data.Device->CreateGraphicsPipelineState(
      &pipeline, IID_PPV_ARGS(&s_Data.GPUPointShadowPipeline)), "Create DX12 GPU point shadow pipeline");

    std::array<D3D12_INDIRECT_ARGUMENT_DESC, 4> arguments{};
    arguments[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW;
    arguments[0].ConstantBufferView.RootParameterIndex = 0;
    arguments[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW;
    arguments[1].VertexBuffer.Slot = 0;
    arguments[2].Type = D3D12_INDIRECT_ARGUMENT_TYPE_INDEX_BUFFER_VIEW;
    arguments[3].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
    D3D12_COMMAND_SIGNATURE_DESC signature{};
    signature.ByteStride = sizeof(IndirectDraw);
    signature.NumArgumentDescs = static_cast<UINT>(arguments.size());
    signature.pArgumentDescs = arguments.data();
    CheckHRESULT(s_Data.Device->CreateCommandSignature(&signature,
      s_Data.SceneRootSignature.Get(), IID_PPV_ARGS(&s_Data.DrawSignature)), "Create DX12 draw signature");

    D3D12_GRAPHICS_PIPELINE_STATE_DESC lightingPipeline{};
    lightingPipeline.pRootSignature = s_Data.SceneRootSignature.Get();
    lightingPipeline.VS = {
      lightingVertexShader.GetBufferPointer(), lightingVertexShader.GetBufferSize()};
    lightingPipeline.PS = {
      lightingPixelShader.GetBufferPointer(), lightingPixelShader.GetBufferSize()};
    lightingPipeline.BlendState = DefaultBlendState();
    lightingPipeline.SampleMask = std::numeric_limits<UINT>::max();
    lightingPipeline.RasterizerState = DefaultRasterizer();
    lightingPipeline.DepthStencilState.DepthEnable = FALSE;
    lightingPipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    lightingPipeline.NumRenderTargets = 1;
    lightingPipeline.RTVFormats[0] = SceneColorFormat;
    lightingPipeline.SampleDesc.Count = 1;
    CheckHRESULT(s_Data.Device->CreateGraphicsPipelineState(
                   &lightingPipeline, IID_PPV_ARGS(&s_Data.LightingPipeline)),
                 "ID3D12Device::CreateGraphicsPipelineState (lighting)");
  }

  void CreateSkyboxPipeline()
  {
    static constexpr const char* shaderSource = "../res/shaders/skybox.slang";
    const auto vertexShader = CompileSlang(shaderSource, "VSMain", "vs_5_1");
    const auto pixelShader = CompileSlang(shaderSource, "PSMain", "ps_5_1");
    D3D12_DESCRIPTOR_RANGE range{};
    range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    range.NumDescriptors = 1;
    range.BaseShaderRegister = 0;
    std::array<D3D12_ROOT_PARAMETER, 2> parameters{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    parameters[0].Descriptor.ShaderRegister = 0;
    parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[1].DescriptorTable = {1, &range};
    parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_STATIC_SAMPLER_DESC sampler{};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_ROOT_SIGNATURE_DESC rootDesc{};
    rootDesc.NumParameters = static_cast<UINT>(parameters.size());
    rootDesc.pParameters = parameters.data();
    rootDesc.NumStaticSamplers = 1;
    rootDesc.pStaticSamplers = &sampler;
    rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    s_Data.SkyboxRootSignature = CreateRootSignature(rootDesc);
    const D3D12_INPUT_ELEMENT_DESC input = {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0};
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline{};
    pipeline.pRootSignature = s_Data.SkyboxRootSignature.Get();
    pipeline.VS = {vertexShader.GetBufferPointer(), vertexShader.GetBufferSize()};
    pipeline.PS = {pixelShader.GetBufferPointer(), pixelShader.GetBufferSize()};
    pipeline.BlendState = DefaultBlendState();
    pipeline.SampleMask = std::numeric_limits<UINT>::max();
    pipeline.RasterizerState = DefaultRasterizer();
    pipeline.DepthStencilState.DepthEnable = TRUE;
    pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    pipeline.InputLayout = {&input, 1};
    pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pipeline.NumRenderTargets = 1;
    pipeline.RTVFormats[0] = SceneColorFormat;
    pipeline.DSVFormat = DepthFormat;
    pipeline.SampleDesc.Count = 1;
    CheckHRESULT(s_Data.Device->CreateGraphicsPipelineState(&pipeline, IID_PPV_ARGS(&s_Data.SkyboxPipeline)),
                 "Create DX12 skybox pipeline");

    static constexpr float vertices[] = {
      -1,1,-1, -1,-1,-1, 1,-1,-1, 1,-1,-1, 1,1,-1, -1,1,-1,
      -1,-1,1, -1,-1,-1, -1,1,-1, -1,1,-1, -1,1,1, -1,-1,1,
      1,-1,-1, 1,-1,1, 1,1,1, 1,1,1, 1,1,-1, 1,-1,-1,
      -1,-1,1, -1,1,1, 1,1,1, 1,1,1, 1,-1,1, -1,-1,1,
      -1,1,-1, 1,1,-1, 1,1,1, 1,1,1, -1,1,1, -1,1,-1,
      -1,-1,-1, -1,-1,1, 1,-1,-1, 1,-1,-1, -1,-1,1, 1,-1,1
    };
    s_Data.SkyboxVertexBuffer = CreateUploadBuffer(sizeof(vertices));
    void* mapped = nullptr;
    const D3D12_RANGE noRead{0, 0};
    CheckHRESULT(s_Data.SkyboxVertexBuffer->Map(0, &noRead, &mapped), "Map DX12 skybox vertices");
    std::memcpy(mapped, vertices, sizeof(vertices));
    s_Data.SkyboxVertexBuffer->Unmap(0, nullptr);
    s_Data.SkyboxVertexView = {s_Data.SkyboxVertexBuffer->GetGPUVirtualAddress(), sizeof(vertices), 3 * sizeof(float)};
  }

  void CreateParticlePipeline()
  {
    static constexpr const char* shaderSource = "../res/shaders/particle.slang";
    const auto vs = CompileSlang(shaderSource, "VSMain", "vs_5_1");
    const auto ps = CompileSlang(shaderSource, "PSMain", "ps_5_1");
    D3D12_ROOT_PARAMETER parameter{};
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    parameter.Descriptor.ShaderRegister = 0;
    D3D12_ROOT_SIGNATURE_DESC rootDesc{};
    rootDesc.NumParameters = 1; rootDesc.pParameters = &parameter;
    rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    s_Data.ParticleRootSignature = CreateRootSignature(rootDesc);
    static constexpr D3D12_INPUT_ELEMENT_DESC inputs[] = {
      {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(ParticleVertex,Position),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
      {"COLOR",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,offsetof(ParticleVertex,Color),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
      {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,offsetof(ParticleVertex,LocalPosition),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},
      {"STYLE",0,DXGI_FORMAT_R32_FLOAT,0,offsetof(ParticleVertex,IsSquare),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0}
    };
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline{};
    pipeline.pRootSignature=s_Data.ParticleRootSignature.Get(); pipeline.VS={vs.GetBufferPointer(),vs.GetBufferSize()}; pipeline.PS={ps.GetBufferPointer(),ps.GetBufferSize()};
    pipeline.BlendState=DefaultBlendState(); auto& blend=pipeline.BlendState.RenderTarget[0]; blend.BlendEnable=TRUE; blend.SrcBlend=D3D12_BLEND_SRC_ALPHA; blend.DestBlend=D3D12_BLEND_INV_SRC_ALPHA; blend.BlendOp=D3D12_BLEND_OP_ADD; blend.SrcBlendAlpha=D3D12_BLEND_ONE; blend.DestBlendAlpha=D3D12_BLEND_INV_SRC_ALPHA; blend.BlendOpAlpha=D3D12_BLEND_OP_ADD;
    pipeline.SampleMask=std::numeric_limits<UINT>::max(); pipeline.RasterizerState=DefaultRasterizer(); pipeline.DepthStencilState.DepthEnable=TRUE; pipeline.DepthStencilState.DepthWriteMask=D3D12_DEPTH_WRITE_MASK_ZERO; pipeline.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_LESS_EQUAL;
    pipeline.InputLayout={inputs,static_cast<UINT>(std::size(inputs))}; pipeline.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE; pipeline.NumRenderTargets=1; pipeline.RTVFormats[0]=SceneColorFormat; pipeline.DSVFormat=DepthFormat; pipeline.SampleDesc.Count=1;
    CheckHRESULT(s_Data.Device->CreateGraphicsPipelineState(&pipeline,IID_PPV_ARGS(&s_Data.ParticlePipeline)),"Create DX12 particle pipeline");
  }

  void CreateDebugLinePipeline()
  {
    static constexpr const char* shaderSource = "../res/shaders/debug_lines.slang";
    const auto vs=CompileSlang(shaderSource,"VSMain","vs_5_1"); const auto ps=CompileSlang(shaderSource,"PSMain","ps_5_1");
    D3D12_ROOT_PARAMETER parameter{}; parameter.ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV; parameter.Descriptor.ShaderRegister=0;
    D3D12_ROOT_SIGNATURE_DESC rootDesc{}; rootDesc.NumParameters=1; rootDesc.pParameters=&parameter; rootDesc.Flags=D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT; s_Data.DebugLineRootSignature=CreateRootSignature(rootDesc);
    static constexpr D3D12_INPUT_ELEMENT_DESC inputs[]={{"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,offsetof(DebugLineVertex,Position),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0},{"COLOR",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,offsetof(DebugLineVertex,Color),D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,0}};
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline{}; pipeline.pRootSignature=s_Data.DebugLineRootSignature.Get(); pipeline.VS={vs.GetBufferPointer(),vs.GetBufferSize()}; pipeline.PS={ps.GetBufferPointer(),ps.GetBufferSize()}; pipeline.BlendState=DefaultBlendState(); auto& blend=pipeline.BlendState.RenderTarget[0]; blend.BlendEnable=TRUE; blend.SrcBlend=D3D12_BLEND_SRC_ALPHA; blend.DestBlend=D3D12_BLEND_INV_SRC_ALPHA; blend.BlendOp=D3D12_BLEND_OP_ADD; blend.SrcBlendAlpha=D3D12_BLEND_ONE; blend.DestBlendAlpha=D3D12_BLEND_INV_SRC_ALPHA; blend.BlendOpAlpha=D3D12_BLEND_OP_ADD; pipeline.SampleMask=std::numeric_limits<UINT>::max(); pipeline.RasterizerState=DefaultRasterizer(); pipeline.DepthStencilState.DepthEnable=TRUE; pipeline.DepthStencilState.DepthWriteMask=D3D12_DEPTH_WRITE_MASK_ZERO; pipeline.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_LESS_EQUAL; pipeline.InputLayout={inputs,static_cast<UINT>(std::size(inputs))}; pipeline.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE; pipeline.NumRenderTargets=1; pipeline.RTVFormats[0]=SceneColorFormat; pipeline.DSVFormat=DepthFormat; pipeline.SampleDesc.Count=1;
    CheckHRESULT(s_Data.Device->CreateGraphicsPipelineState(&pipeline,IID_PPV_ARGS(&s_Data.DebugLinePipeline)),"Create DX12 debug line pipeline");
  }

  void CreatePhysicsDebugPipeline()
  {
    static constexpr const char* shaderSource = "../res/shaders/physics_debug.slang";
    const auto vertexShader = CompileSlang(shaderSource, "VSMain", "vs_5_1");
    const auto pixelShader = CompileSlang(shaderSource, "PSMain", "ps_5_1");

    D3D12_ROOT_PARAMETER parameter{};
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    parameter.Descriptor.ShaderRegister = 0;
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rootDesc{};
    rootDesc.NumParameters = 1;
    rootDesc.pParameters = &parameter;
    rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    s_Data.PhysicsDebugRootSignature = CreateRootSignature(rootDesc);

    static constexpr D3D12_INPUT_ELEMENT_DESC inputLayout[] = {
      {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, offsetof(Vertex, Position),
       D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}
    };
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline{};
    pipeline.pRootSignature = s_Data.PhysicsDebugRootSignature.Get();
    pipeline.VS = {vertexShader.GetBufferPointer(), vertexShader.GetBufferSize()};
    pipeline.PS = {pixelShader.GetBufferPointer(), pixelShader.GetBufferSize()};
    pipeline.BlendState = DefaultBlendState();
    auto& blend = pipeline.BlendState.RenderTarget[0];
    blend.BlendEnable = TRUE;
    blend.SrcBlend = D3D12_BLEND_SRC_ALPHA;
    blend.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
    blend.BlendOp = D3D12_BLEND_OP_ADD;
    blend.SrcBlendAlpha = D3D12_BLEND_ONE;
    blend.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
    blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    pipeline.SampleMask = std::numeric_limits<UINT>::max();
    pipeline.RasterizerState = DefaultRasterizer();
    pipeline.RasterizerState.FillMode = D3D12_FILL_MODE_WIREFRAME;
    pipeline.DepthStencilState.DepthEnable = TRUE;
    pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    pipeline.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    pipeline.InputLayout = {inputLayout, static_cast<UINT>(std::size(inputLayout))};
    pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pipeline.NumRenderTargets = 1;
    pipeline.RTVFormats[0] = SceneColorFormat;
    pipeline.DSVFormat = DepthFormat;
    pipeline.SampleDesc.Count = 1;
    CheckHRESULT(s_Data.Device->CreateGraphicsPipelineState(
                   &pipeline, IID_PPV_ARGS(&s_Data.PhysicsDebugPipeline)),
                 "Create DX12 physics debug pipeline");
  }

  void CreatePostProcessPipelines()
  {
    static constexpr const char* shaderSource = "../res/shaders/postprocess.slang";
    const auto vs=CompileSlang(shaderSource,"VSMain","vs_5_1"); const auto extract=CompileSlang(shaderSource,"Extract","ps_5_1"); const auto down=CompileSlang("../res/shaders/bloom_downsample.slang","PSMain","ps_5_1"); const auto up=CompileSlang("../res/shaders/bloom_upsample.slang","PSMain","ps_5_1"); const auto composite=CompileSlang(shaderSource,"Composite","ps_5_1"); const auto postProcess=CompileSlang(shaderSource,"PostProcess","ps_5_1");
    std::array<D3D12_DESCRIPTOR_RANGE,2> ranges{}; for(uint32_t i=0;i<2;++i){ranges[i].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV;ranges[i].NumDescriptors=1;ranges[i].BaseShaderRegister=i;}
    std::array<D3D12_ROOT_PARAMETER,3> parameters{}; parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; parameters[0].Constants.ShaderRegister=0; parameters[0].Constants.Num32BitValues=8; for(uint32_t i=0;i<2;++i){parameters[i+1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;parameters[i+1].DescriptorTable={1,&ranges[i]};parameters[i+1].ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;}
    D3D12_STATIC_SAMPLER_DESC sampler{}; sampler.Filter=D3D12_FILTER_MIN_MAG_MIP_LINEAR; sampler.AddressU=sampler.AddressV=sampler.AddressW=D3D12_TEXTURE_ADDRESS_MODE_CLAMP; sampler.MaxLOD=D3D12_FLOAT32_MAX; sampler.ShaderVisibility=D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_ROOT_SIGNATURE_DESC rootDesc{}; rootDesc.NumParameters=static_cast<UINT>(parameters.size()); rootDesc.pParameters=parameters.data(); rootDesc.NumStaticSamplers=1; rootDesc.pStaticSamplers=&sampler; s_Data.PostRootSignature=CreateRootSignature(rootDesc);
    auto createPipeline=[&](const Shader::Bytecode& pixelShader,DXGI_FORMAT format,ComPtr<ID3D12PipelineState>& result,const char* name, bool additive=false){D3D12_GRAPHICS_PIPELINE_STATE_DESC p{};p.pRootSignature=s_Data.PostRootSignature.Get();p.VS={vs.GetBufferPointer(),vs.GetBufferSize()};p.PS={pixelShader.GetBufferPointer(),pixelShader.GetBufferSize()};p.BlendState=DefaultBlendState();if(additive){auto& b=p.BlendState.RenderTarget[0];b.BlendEnable=TRUE;b.BlendOp=b.BlendOpAlpha=D3D12_BLEND_OP_ADD;b.SrcBlend=b.DestBlend=b.SrcBlendAlpha=b.DestBlendAlpha=D3D12_BLEND_ONE;}p.SampleMask=std::numeric_limits<UINT>::max();p.RasterizerState=DefaultRasterizer();p.DepthStencilState.DepthEnable=FALSE;p.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;p.NumRenderTargets=1;p.RTVFormats[0]=format;p.SampleDesc.Count=1;CheckHRESULT(s_Data.Device->CreateGraphicsPipelineState(&p,IID_PPV_ARGS(&result)),name);};
    createPipeline(extract,SceneColorFormat,s_Data.BloomExtractPipeline,"Create DX12 bloom extract pipeline"); createPipeline(down,DXGI_FORMAT_R11G11B10_FLOAT,s_Data.BloomDownsamplePipeline,"Create DX12 bloom downsample"); createPipeline(up,DXGI_FORMAT_R11G11B10_FLOAT,s_Data.BloomUpsamplePipeline,"Create DX12 bloom upsample",true); createPipeline(composite,SceneColorFormat,s_Data.CompositePipeline,"Create DX12 composite pipeline"); createPipeline(postProcess,BackBufferFormat,s_Data.PostProcessPipeline,"Create DX12 post-process pipeline");
  }

  void CreateUIPipeline()
  {
    static constexpr const char* UIShader = "../res/shaders/ui.slang";

    const auto vertexShader = CompileSlang(UIShader, "VSMain", "vs_5_1");
    const auto pixelShader = CompileSlang(UIShader, "PSMain", "ps_5_1");

    D3D12_DESCRIPTOR_RANGE textureRange{};
    textureRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    textureRange.NumDescriptors = 1;
    textureRange.BaseShaderRegister = 0;
    textureRange.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    std::array<D3D12_ROOT_PARAMETER, 2> parameters{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters[0].Constants.ShaderRegister = 0;
    parameters[0].Constants.Num32BitValues = 2;
    parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[1].DescriptorTable.NumDescriptorRanges = 1;
    parameters[1].DescriptorTable.pDescriptorRanges = &textureRange;
    parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_STATIC_SAMPLER_DESC sampler{};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_ROOT_SIGNATURE_DESC rootDesc{};
    rootDesc.NumParameters = static_cast<UINT>(parameters.size());
    rootDesc.pParameters = parameters.data();
    rootDesc.NumStaticSamplers = 1;
    rootDesc.pStaticSamplers = &sampler;
    rootDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    s_Data.UIRootSignature = CreateRootSignature(rootDesc);

    static constexpr D3D12_INPUT_ELEMENT_DESC InputLayout[] = {
      {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, offsetof(UIVertex, Position),
       D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, offsetof(UIVertex, Color),
       D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
      {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, offsetof(UIVertex, UV),
       D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline{};
    pipeline.pRootSignature = s_Data.UIRootSignature.Get();
    pipeline.VS = {vertexShader.GetBufferPointer(), vertexShader.GetBufferSize()};
    pipeline.PS = {pixelShader.GetBufferPointer(), pixelShader.GetBufferSize()};
    pipeline.BlendState = DefaultBlendState();
    auto& blend = pipeline.BlendState.RenderTarget[0];
    blend.BlendEnable = TRUE;
    blend.SrcBlend = D3D12_BLEND_SRC_ALPHA;
    blend.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
    blend.BlendOp = D3D12_BLEND_OP_ADD;
    blend.SrcBlendAlpha = D3D12_BLEND_ONE;
    blend.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
    blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    pipeline.SampleMask = std::numeric_limits<UINT>::max();
    pipeline.RasterizerState = DefaultRasterizer();
    pipeline.DepthStencilState.DepthEnable = FALSE;
    pipeline.DepthStencilState.StencilEnable = FALSE;
    pipeline.InputLayout = {InputLayout, static_cast<UINT>(std::size(InputLayout))};
    pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pipeline.NumRenderTargets = 1;
    pipeline.RTVFormats[0] = BackBufferFormat;
    pipeline.SampleDesc.Count = 1;
    CheckHRESULT(s_Data.Device->CreateGraphicsPipelineState(
                   &pipeline, IID_PPV_ARGS(&s_Data.UIPipeline)),
                 "ID3D12Device::CreateGraphicsPipelineState (UI)");
    pipeline.RTVFormats[0] = SceneColorFormat;
    CheckHRESULT(s_Data.Device->CreateGraphicsPipelineState(
                   &pipeline, IID_PPV_ARGS(&s_Data.SceneUIPipeline)),
                 "ID3D12Device::CreateGraphicsPipelineState (scene UI)");
  }

  GPUTexture UploadTextureBytes(const uint8_t* pixels, uint32_t width, uint32_t height,
                                DXGI_FORMAT format, uint32_t bytesPerPixel,
                                bool generateMips = false)
  {
    if (!pixels || width == 0 || height == 0)
      return s_Data.WhiteTexture;

    GPUTexture texture;
    texture.DescriptorIndex = AllocateDescriptor();

    D3D12_RESOURCE_DESC textureDesc{};
    textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    textureDesc.Width = width;
    textureDesc.Height = height;
    textureDesc.DepthOrArraySize = 1;
    uint16_t mipLevels = 1;
    if (generateMips)
      for (uint32_t mipWidth = width, mipHeight = height;
           mipWidth > 1 || mipHeight > 1;
           mipWidth = std::max(1u, mipWidth / 2u),
           mipHeight = std::max(1u, mipHeight / 2u))
        ++mipLevels;
    textureDesc.MipLevels = mipLevels;
    textureDesc.Format = format;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

    const D3D12_HEAP_PROPERTIES defaultHeap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    CheckHRESULT(s_Data.Device->CreateCommittedResource(
                   &defaultHeap, D3D12_HEAP_FLAG_NONE, &textureDesc,
                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                   IID_PPV_ARGS(&texture.Resource)),
                 "ID3D12Device::CreateCommittedResource (texture)");

    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> footprints(mipLevels);
    std::vector<UINT> rowCounts(mipLevels);
    std::vector<UINT64> rowSizes(mipLevels);
    UINT64 uploadSize = 0;
    s_Data.Device->GetCopyableFootprints(
      &textureDesc, 0, mipLevels, 0, footprints.data(), rowCounts.data(),
      rowSizes.data(), &uploadSize);
    ComPtr<ID3D12Resource> upload = CreateUploadBuffer(uploadSize);
    uint8_t* mapped = nullptr;
    const D3D12_RANGE noRead{0, 0};
    CheckHRESULT(upload->Map(0, &noRead, reinterpret_cast<void**>(&mapped)),
                 "ID3D12Resource::Map (texture upload)");
    std::vector<uint8_t> currentPixels(
      pixels, pixels + static_cast<size_t>(width) * height * bytesPerPixel);
    uint32_t mipWidth = width;
    uint32_t mipHeight = height;
    for (uint16_t mip = 0; mip < mipLevels; ++mip)
    {
      const uint64_t sourcePitch = static_cast<uint64_t>(mipWidth) * bytesPerPixel;
      if (rowCounts[mip] != mipHeight || rowSizes[mip] < sourcePitch)
        throw std::runtime_error("Invalid DX12 texture upload footprint");
      for (uint32_t row = 0; row < mipHeight; ++row)
        std::memcpy(
          mapped + footprints[mip].Offset +
            static_cast<uint64_t>(row) * footprints[mip].Footprint.RowPitch,
          currentPixels.data() + static_cast<uint64_t>(row) * sourcePitch,
          static_cast<size_t>(sourcePitch));

      if (mip + 1 < mipLevels)
      {
        const uint32_t nextWidth = std::max(1u, mipWidth / 2u);
        const uint32_t nextHeight = std::max(1u, mipHeight / 2u);
        std::vector<uint8_t> nextPixels(
          static_cast<size_t>(nextWidth) * nextHeight * bytesPerPixel);
        for (uint32_t y = 0; y < nextHeight; ++y)
          for (uint32_t x = 0; x < nextWidth; ++x)
            for (uint32_t channel = 0; channel < bytesPerPixel; ++channel)
            {
              uint32_t sum = 0;
              uint32_t samples = 0;
              for (uint32_t offsetY = 0; offsetY < 2; ++offsetY)
                for (uint32_t offsetX = 0; offsetX < 2; ++offsetX)
                {
                  const uint32_t sourceX = x * 2u + offsetX;
                  const uint32_t sourceY = y * 2u + offsetY;
                  if (sourceX >= mipWidth || sourceY >= mipHeight) continue;
                  sum += currentPixels[
                    (static_cast<size_t>(sourceY) * mipWidth + sourceX) *
                      bytesPerPixel + channel];
                  ++samples;
                }
              nextPixels[(static_cast<size_t>(y) * nextWidth + x) *
                bytesPerPixel + channel] = static_cast<uint8_t>(sum / samples);
            }
        currentPixels = std::move(nextPixels);
        mipWidth = nextWidth;
        mipHeight = nextHeight;
      }
    }
    upload->Unmap(0, nullptr);

    CheckHRESULT(s_Data.UploadAllocator->Reset(), "ID3D12CommandAllocator::Reset (upload)");
    CheckHRESULT(s_Data.UploadCommandList->Reset(s_Data.UploadAllocator.Get(), nullptr),
                 "ID3D12GraphicsCommandList::Reset (upload)");

    for (uint16_t mip = 0; mip < mipLevels; ++mip)
    {
      D3D12_TEXTURE_COPY_LOCATION destination{};
      destination.pResource = texture.Resource.Get();
      destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      destination.SubresourceIndex = mip;
      D3D12_TEXTURE_COPY_LOCATION source{};
      source.pResource = upload.Get();
      source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
      source.PlacedFootprint = footprints[mip];
      s_Data.UploadCommandList->CopyTextureRegion(
        &destination, 0, 0, 0, &source, nullptr);
    }

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = texture.Resource.Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    s_Data.UploadCommandList->ResourceBarrier(1, &barrier);
    CheckHRESULT(s_Data.UploadCommandList->Close(), "ID3D12GraphicsCommandList::Close (upload)");
    ID3D12CommandList* commandLists[] = {s_Data.UploadCommandList.Get()};
    s_Data.CommandQueue->ExecuteCommandLists(1, commandLists);
    WaitForGPU();

    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    view.Format = format;
    view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    view.Texture2D.MipLevels = mipLevels;
    s_Data.Device->CreateShaderResourceView(
      texture.Resource.Get(), &view, GetSRVCPUHandle(texture.DescriptorIndex));
    return texture;
  }

  GPUTexture UploadCubeTexture(const std::shared_ptr<Texture>& source)
  {
    if (!source || source->GetWidth() == 0 || source->GetHeight() == 0)
      return {};
    const uint32_t width = source->GetWidth();
    const uint32_t height = source->GetHeight();
    const uint32_t channels = static_cast<uint32_t>(source->GetChannels());
    auto& faces = source->GetPixels();
    for (const auto* face : faces) if (!face) return {};

    GPUTexture texture;
    texture.DescriptorIndex = AllocateDescriptor();
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 6;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    const D3D12_HEAP_PROPERTIES heap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
    CheckHRESULT(s_Data.Device->CreateCommittedResource(
                   &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST,
                   nullptr, IID_PPV_ARGS(&texture.Resource)),
                 "Create DX12 cubemap");

    std::array<D3D12_PLACED_SUBRESOURCE_FOOTPRINT, 6> footprints{};
    std::array<UINT, 6> rowCounts{};
    std::array<UINT64, 6> rowSizes{};
    UINT64 uploadSize = 0;
    s_Data.Device->GetCopyableFootprints(
      &desc, 0, 6, 0, footprints.data(), rowCounts.data(), rowSizes.data(), &uploadSize);
    ComPtr<ID3D12Resource> upload = CreateUploadBuffer(uploadSize);
    uint8_t* mapped = nullptr;
    const D3D12_RANGE noRead{0, 0};
    CheckHRESULT(upload->Map(0, &noRead, reinterpret_cast<void**>(&mapped)), "Map DX12 cubemap upload");
    std::vector<uint8_t> rgba(static_cast<size_t>(width) * height * 4);
    for (uint32_t faceIndex = 0; faceIndex < 6; ++faceIndex)
    {
      const uint8_t* face = faces[faceIndex];
      for (size_t pixel = 0; pixel < static_cast<size_t>(width) * height; ++pixel)
      {
        rgba[pixel * 4] = face[pixel * channels];
        rgba[pixel * 4 + 1] = channels > 1 ? face[pixel * channels + 1] : rgba[pixel * 4];
        rgba[pixel * 4 + 2] = channels > 2 ? face[pixel * channels + 2] : rgba[pixel * 4];
        rgba[pixel * 4 + 3] = channels > 3 ? face[pixel * channels + 3] : 255;
      }
      for (uint32_t row = 0; row < height; ++row)
        std::memcpy(mapped + footprints[faceIndex].Offset + static_cast<uint64_t>(row) * footprints[faceIndex].Footprint.RowPitch,
                    rgba.data() + static_cast<size_t>(row) * width * 4,
                    static_cast<size_t>(width) * 4);
    }
    upload->Unmap(0, nullptr);

    CheckHRESULT(s_Data.UploadAllocator->Reset(), "Reset DX12 cubemap allocator");
    CheckHRESULT(s_Data.UploadCommandList->Reset(s_Data.UploadAllocator.Get(), nullptr), "Reset DX12 cubemap list");
    for (uint32_t faceIndex = 0; faceIndex < 6; ++faceIndex)
    {
      D3D12_TEXTURE_COPY_LOCATION destination{};
      destination.pResource = texture.Resource.Get();
      destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      destination.SubresourceIndex = faceIndex;
      D3D12_TEXTURE_COPY_LOCATION sourceLocation{};
      sourceLocation.pResource = upload.Get();
      sourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
      sourceLocation.PlacedFootprint = footprints[faceIndex];
      s_Data.UploadCommandList->CopyTextureRegion(&destination, 0, 0, 0, &sourceLocation, nullptr);
    }
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = texture.Resource.Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    s_Data.UploadCommandList->ResourceBarrier(1, &barrier);
    CheckHRESULT(s_Data.UploadCommandList->Close(), "Close DX12 cubemap list");
    ID3D12CommandList* lists[] = {s_Data.UploadCommandList.Get()};
    s_Data.CommandQueue->ExecuteCommandLists(1, lists);
    WaitForGPU();

    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
    srv.TextureCube.MipLevels = 1;
    s_Data.Device->CreateShaderResourceView(texture.Resource.Get(), &srv, GetSRVCPUHandle(texture.DescriptorIndex));
    return texture;
  }

  uint32_t GetOrCreateModelTexture(const std::shared_ptr<Texture>& source)
  {
    if (!source) return s_Data.WhiteTexture.DescriptorIndex;
    if (const auto existing = s_Data.Textures.find(source.get()); existing != s_Data.Textures.end())
      return existing->second.DescriptorIndex;

    const uint32_t width = source->GetWidth();
    const uint32_t height = source->GetHeight();
    const uint8_t* raw = source->GetRawData();
    if (!raw || width == 0 || height == 0)
      return s_Data.WhiteTexture.DescriptorIndex;

    const uint32_t channelCount = static_cast<uint32_t>(
      std::clamp(source->GetChannels(), 1, 4));
    std::vector<uint8_t> rgba(static_cast<size_t>(width) * height * 4);

    // Texture owns one canonical row order used by every backend. Reversing
    // the rows only for Direct3D makes atlas UVs address a different material
    // region (especially visible on animated characters).
    for (size_t pixel = 0; pixel < static_cast<size_t>(width) * height; ++pixel)
    {
      const size_t sourcePixel = pixel * channelCount;
      const size_t destinationPixel = pixel * 4;
      const uint8_t r = raw[sourcePixel];
      rgba[destinationPixel + 0] = r;
      rgba[destinationPixel + 1] = channelCount > 1 ? raw[sourcePixel + 1] : r;
      rgba[destinationPixel + 2] = channelCount > 2 ? raw[sourcePixel + 2] : r;
      rgba[destinationPixel + 3] = channelCount > 3 ? raw[sourcePixel + 3] : 255;
    }

    GPUTexture texture = UploadTextureBytes(
      rgba.data(), width, height, DXGI_FORMAT_R8G8B8A8_UNORM, 4, true);
    const uint32_t descriptorIndex = texture.DescriptorIndex;
    s_Data.Textures.emplace(source.get(), std::move(texture));
    return descriptorIndex;
  }

  template <typename T>
  D3D12_GPU_VIRTUAL_ADDRESS UploadConstants(const T& constants)
  {
    auto& frame = s_Data.FrameUploads[s_Data.FrameIndex];
    const uint64_t offset = AlignUp(frame.ConstantsOffset, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
    if (offset + sizeof(T) > ConstantBufferBytes)
      throw std::runtime_error("DX12 per-frame constant buffer exhausted");
    std::memcpy(frame.ConstantsMapped + offset, &constants, sizeof(T));
    frame.ConstantsOffset = offset + sizeof(T);
    return frame.Constants->GetGPUVirtualAddress() + offset;
  }

  D3D12_GPU_VIRTUAL_ADDRESS UploadSceneConstants(const SceneConstants& constants)
  {
    return UploadConstants(constants);
  }

  glm::mat4 DirectXViewProjection()
  {
    glm::mat4 correction(1.0f);
    correction[2][2] = 0.5f;
    correction[3][2] = 0.5f;
    return correction * Camera::GetViewProjection();
  }

  SceneConstants BuildSceneConstants(const RenderEffectSettings& effects)
  {
    SceneConstants constants;
    constants.ViewProjection = DirectXViewProjection();
    constants.LightViewProjection = s_Data.LightViewProjection;
    constants.CameraPosition = glm::vec4(Camera::GetPosition(), 0.0f);
    constants.Bones.fill(glm::mat4(1.0f));
    const auto& lights = RenderBackend::Lights();
    const uint32_t lightCount = static_cast<uint32_t>(
      std::min(lights.size(), static_cast<size_t>(MaxSceneLights)));
    constants.LightCount.x = lightCount;
    constants.LightCount.y = effects.ShadowQuality == GraphicsQuality::Off ? 0u : 1u;
    constants.LightCount.z = s_Data.TileCountX;
    constants.LightCount.w = RenderBackend::DebugSettings().TiledLightingMode;
    const uint32_t filterSize = effects.ShadowQuality == GraphicsQuality::High ? 4u :
      (effects.ShadowQuality == GraphicsQuality::Medium ? 3u : 2u);
    constants.ShadowFilter = glm::vec4(float(filterSize * filterSize),
      0.5f + 0.5f * filterSize, 0, 0);
    constants.Resolution = glm::vec4(
      static_cast<float>(s_Data.Width), static_cast<float>(s_Data.Height),
      effects.PS1VirtualHeight, effects.PS1Enabled ? 1.0f : 0.0f);
    for (uint32_t i = 0; i < lightCount; ++i)
    {
      constants.LightPositions[i] = glm::vec4(lights[i].Position, 1.0f);
      constants.LightDirections[i] = glm::vec4(lights[i].Direction, 0.0f);
      constants.LightColors[i] = glm::vec4(lights[i].Color, 1.0f);
      constants.LightTypes[i].x = static_cast<float>(lights[i].Type);
    }
    for (uint32_t slot = 0;
         slot < static_cast<uint32_t>(s_Data.PointShadowLightIndices.size()); ++slot)
    {
      const uint32_t lightIndex = s_Data.PointShadowLightIndices[slot];
      if (lightIndex < lightCount)
        constants.LightTypes[lightIndex].y = static_cast<float>(slot + 1u);
    }
    return constants;
  }

  #include "DirectX12Pipeline.inl"

  glm::mat4 CalculateLightViewProjection()
  {
    glm::vec3 direction(-1.0f, -2.0f, -1.0f);
    for (const RenderLight& light : RenderBackend::Lights())
      if (light.Type == LightType::DIRECT) { direction = light.Direction; break; }
    if (glm::dot(direction, direction) < 0.0001f) direction = glm::vec3(-1.0f, -2.0f, -1.0f);
    direction = glm::normalize(direction);
    constexpr float orthoSize = 90.0f;
    glm::vec3 focus = Camera::GetPosition() + Camera::GetForwardDirection() * 45.0f;
    const glm::vec3 fallbackUp =
      std::abs(glm::dot(direction, glm::vec3(0.0f, 1.0f, 0.0f))) > 0.98f
        ? glm::vec3(0.0f, 0.0f, 1.0f)
        : glm::vec3(0.0f, 1.0f, 0.0f);
    const glm::vec3 right = glm::normalize(glm::cross(direction, fallbackUp));
    const glm::vec3 lightUp = glm::normalize(glm::cross(right, direction));
    const float worldUnitsPerTexel =
      (2.0f * orthoSize) / static_cast<float>(std::max(s_Data.ShadowMapSize, 1u));
    focus += right * (std::round(glm::dot(focus, right) / worldUnitsPerTexel)
      * worldUnitsPerTexel - glm::dot(focus, right));
    focus += lightUp * (std::round(glm::dot(focus, lightUp) / worldUnitsPerTexel)
      * worldUnitsPerTexel - glm::dot(focus, lightUp));
    const glm::mat4 view = glm::lookAt(focus - direction * 180.0f, focus, lightUp);
    const glm::mat4 projection = glm::ortho(
      -orthoSize, orthoSize, -orthoSize, orthoSize, 0.1f, 400.0f);
    glm::mat4 correction(1.0f);
    correction[2][2] = 0.5f;
    correction[3][2] = 0.5f;
    return correction * projection * view;
  }

  void DrawShadowMap(const RenderEffectSettings& effects)
  {
    if (!s_Data.ShadowMap.Resource) return;
    if (s_Data.ShadowMapReadable)
      TransitionResource(s_Data.ShadowMap.Resource.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                         D3D12_RESOURCE_STATE_DEPTH_WRITE);
    s_Data.ShadowMapReadable = false;
    const D3D12_CPU_DESCRIPTOR_HANDLE dsv = GetDSVHandle(1);
    s_Data.CommandList->OMSetRenderTargets(0, nullptr, FALSE, &dsv);
    s_Data.CommandList->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
    const D3D12_VIEWPORT viewport{0.0f, 0.0f, static_cast<float>(s_Data.ShadowMapSize), static_cast<float>(s_Data.ShadowMapSize), 0.0f, 1.0f};
    const D3D12_RECT scissor{0, 0, static_cast<LONG>(s_Data.ShadowMapSize), static_cast<LONG>(s_Data.ShadowMapSize)};
    s_Data.CommandList->RSSetViewports(1, &viewport);
    s_Data.CommandList->RSSetScissorRects(1, &scissor);
    s_Data.LightViewProjection = CalculateLightViewProjection();

    if (effects.ShadowQuality != GraphicsQuality::Off)
    {
      DispatchGPUCull(false, s_Data.LightViewProjection, effects, true);
      struct ShadowFrame { glm::mat4 ViewProjection; glm::vec4 LightPosition; };
      DrawGPUScene(s_Data.GPUShadowPipeline.Get(),
        UploadConstants(ShadowFrame{s_Data.LightViewProjection, glm::vec4(0)}));
    }
    TransitionResource(s_Data.ShadowMap.Resource.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE,
                       D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    s_Data.ShadowMapReadable = true;
  }

  void DrawPointShadowMaps(const RenderEffectSettings& effects)
  {
    s_Data.PointShadowLightIndices.clear();
    if (effects.ShadowQuality == GraphicsQuality::Off || !s_Data.PointShadowMap.Resource)
      return;

    struct Candidate
    {
      float DistanceSquared = 0.0f;
      uint32_t LightIndex = 0;
    };
    const auto& lights = RenderBackend::Lights();
    std::vector<Candidate> candidates;
    candidates.reserve(lights.size());
    const glm::vec3 cameraPosition = Camera::GetPosition();
    const RenderFrustum cameraFrustum(Camera::GetViewProjection());
    for (uint32_t i = 0; i < std::min(uint32_t(lights.size()), MaxSceneLights); ++i)
    {
      if (lights[i].Type != LightType::POINT) continue;
      if (!cameraFrustum.IntersectsSphere(lights[i].Position, PointShadowRadius)) continue;
      const glm::vec3 delta = lights[i].Position - cameraPosition;
      candidates.push_back({glm::dot(delta, delta), i});
    }
    std::ranges::sort(candidates, {}, &Candidate::DistanceSquared);
    const uint32_t shadowCount = static_cast<uint32_t>(std::min(
      candidates.size(), static_cast<size_t>(MaxShadowedPointLights)));
    if (shadowCount == 0) return;
    s_Data.PointShadowLightIndices.assign(MaxShadowedPointLights, UINT32_MAX);
    std::array<bool, MaxShadowedPointLights> assigned{};
    for (uint32_t i = 0; i < shadowCount; ++i)
      for (uint32_t slot = 0; slot < MaxShadowedPointLights; ++slot)
        if (!assigned[slot] && s_Data.PointShadowCache[slot].Valid &&
            s_Data.PointShadowCache[slot].LightIndex == candidates[i].LightIndex)
        {
          s_Data.PointShadowLightIndices[slot] = candidates[i].LightIndex;
          assigned[slot] = true;
          break;
        }
    for (uint32_t i = 0; i < shadowCount; ++i)
    {
      if (std::ranges::find(s_Data.PointShadowLightIndices, candidates[i].LightIndex) != s_Data.PointShadowLightIndices.end()) continue;
      const auto freeSlot = std::ranges::find(assigned, false);
      const auto slot = static_cast<size_t>(freeSlot - assigned.begin());
      assigned[slot] = true;
      s_Data.PointShadowLightIndices[slot] = candidates[i].LightIndex;
    }

    if (s_Data.PointShadowMapReadable)
      TransitionResource(s_Data.PointShadowMap.Resource.Get(),
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
    s_Data.PointShadowMapReadable = false;

    const D3D12_VIEWPORT viewport{0.0f, 0.0f,
      static_cast<float>(s_Data.PointShadowMapSize),
      static_cast<float>(s_Data.PointShadowMapSize), 0.0f, 1.0f};
    const D3D12_RECT scissor{0, 0, static_cast<LONG>(s_Data.PointShadowMapSize),
      static_cast<LONG>(s_Data.PointShadowMapSize)};
    const D3D12_CPU_DESCRIPTOR_HANDLE dsv = GetDSVHandle(2);
    s_Data.CommandList->RSSetViewports(1, &viewport);
    s_Data.CommandList->RSSetScissorRects(1, &scissor);
    static constexpr float clearDistance[] = {
      PointShadowRadius, PointShadowRadius, PointShadowRadius, PointShadowRadius};
    for (uint32_t slot = 0; slot < MaxShadowedPointLights; ++slot)
    {
      const uint32_t lightIndex = s_Data.PointShadowLightIndices[slot];
      if (lightIndex == UINT32_MAX) continue;
      const glm::vec3 lightPosition = lights[lightIndex].Position;
      const auto casterState = GetPointShadowCasterState(lightPosition);
      auto& cache = s_Data.PointShadowCache[slot];
      if (cache.Valid && cache.LightIndex == lightIndex &&
          cache.LightPosition == lightPosition && cache.CasterHash == casterState.Hash &&
          !casterState.Animated) continue;
      for (uint32_t face = 0; face < PointShadowFaceCount; ++face)
      {
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv = GetRTVHandle(
          PointShadowRTVBaseIndex + slot * PointShadowFaceCount + face);
        s_Data.CommandList->ClearRenderTargetView(rtv, clearDistance, 0, nullptr);
        s_Data.CommandList->ClearDepthStencilView(
          dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
        s_Data.CommandList->OMSetRenderTargets(1, &rtv, FALSE, &dsv);
        const glm::mat4 lightViewProjection = DX12Shadow::PointViewProjection(
          lightPosition, face, 0.1f, PointShadowRadius);
        DispatchGPUCull(false, lightViewProjection, effects, true);
        struct ShadowFrame { glm::mat4 ViewProjection; glm::vec4 LightPosition; };
        DrawGPUScene(s_Data.GPUPointShadowPipeline.Get(),
          UploadConstants(ShadowFrame{lightViewProjection, glm::vec4(lightPosition, 0)}));
      }
      cache = {casterState.Hash, lightPosition, lightIndex, true};
    }

    TransitionResource(s_Data.PointShadowMap.Resource.Get(),
      D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    s_Data.PointShadowMapReadable = true;
  }

  void BeginGeometryPass()
  {
    TransitionResource(s_Data.GPosition.Resource.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                       D3D12_RESOURCE_STATE_RENDER_TARGET);
    TransitionResource(s_Data.GNormal.Resource.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                       D3D12_RESOURCE_STATE_RENDER_TARGET);
    TransitionResource(s_Data.GAlbedoSpec.Resource.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                       D3D12_RESOURCE_STATE_RENDER_TARGET);
    const std::array<D3D12_CPU_DESCRIPTOR_HANDLE, 3> renderTargets = {
      GetRTVHandle(GPositionRTVIndex),
      GetRTVHandle(GNormalRTVIndex),
      GetRTVHandle(GAlbedoSpecRTVIndex)
    };
    const D3D12_CPU_DESCRIPTOR_HANDLE dsv = GetDSVHandle(0);
    static constexpr float clear[] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (const auto& renderTarget : renderTargets)
      s_Data.CommandList->ClearRenderTargetView(renderTarget, clear, 0, nullptr);
    s_Data.CommandList->OMSetRenderTargets(
      static_cast<UINT>(renderTargets.size()), renderTargets.data(), FALSE, &dsv);
    s_Data.CommandList->RSSetViewports(1, &s_Data.Viewport);
    s_Data.CommandList->RSSetScissorRects(1, &s_Data.ScissorRect);
  }

  void EndGeometryPass()
  {
    TransitionResource(s_Data.GPosition.Resource.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET,
                       D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    TransitionResource(s_Data.GNormal.Resource.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET,
                       D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    TransitionResource(s_Data.GAlbedoSpec.Resource.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET,
                       D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
  }

  void DrawLightingPass(const RenderEffectSettings& effects)
  {
    TransitionResource(s_Data.SceneColor.Resource.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                       D3D12_RESOURCE_STATE_RENDER_TARGET);
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = GetRTVHandle(SceneRTVIndex);
    static constexpr float clear[] = {0.0f, 0.0f, 0.0f, 0.0f};
    s_Data.CommandList->ClearRenderTargetView(rtv, clear, 0, nullptr);
    s_Data.CommandList->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    s_Data.CommandList->RSSetViewports(1, &s_Data.Viewport);
    s_Data.CommandList->RSSetScissorRects(1, &s_Data.ScissorRect);
    ID3D12DescriptorHeap* descriptorHeaps[] = {s_Data.SRVHeap.Get()};
    s_Data.CommandList->SetDescriptorHeaps(1, descriptorHeaps);
    s_Data.CommandList->SetPipelineState(s_Data.LightingPipeline.Get());
    s_Data.CommandList->SetGraphicsRootSignature(s_Data.SceneRootSignature.Get());
    const SceneConstants constants = BuildSceneConstants(effects);
    s_Data.CommandList->SetGraphicsRootConstantBufferView(0, UploadSceneConstants(constants));
    s_Data.CommandList->SetGraphicsRootShaderResourceView(8, s_Data.TileGrid->GetGPUVirtualAddress());
    s_Data.CommandList->SetGraphicsRootShaderResourceView(9, s_Data.TileIndices->GetGPUVirtualAddress());
    s_Data.CommandList->SetGraphicsRootShaderResourceView(11, s_Data.ShadowOffsets->GetGPUVirtualAddress());
    s_Data.CommandList->SetGraphicsRootDescriptorTable(
      1, GetSRVGPUHandle(s_Data.GPosition.DescriptorIndex));
    s_Data.CommandList->SetGraphicsRootDescriptorTable(
      2, GetSRVGPUHandle(s_Data.GNormal.DescriptorIndex));
    s_Data.CommandList->SetGraphicsRootDescriptorTable(
      3, GetSRVGPUHandle(s_Data.GAlbedoSpec.DescriptorIndex));
    s_Data.CommandList->SetGraphicsRootDescriptorTable(
      4, GetSRVGPUHandle(s_Data.ShadowMap.DescriptorIndex));
    s_Data.CommandList->SetGraphicsRootDescriptorTable(
      5, GetSRVGPUHandle(s_Data.PointShadowMap.DescriptorIndex));
    s_Data.CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    s_Data.CommandList->DrawInstanced(3, 1, 0, 0);

    // The deferred light pass does not use depth, but all following forward
    // draws (skybox, particles and debug geometry) share the geometry depth.
    const D3D12_CPU_DESCRIPTOR_HANDLE dsv = GetDSVHandle(0);
    s_Data.CommandList->OMSetRenderTargets(1, &rtv, FALSE, &dsv);
  }

  void DrawSkybox()
  {
    if (!s_Data.SkyboxTexture.Resource) return;
    ID3D12DescriptorHeap* heaps[] = {s_Data.SRVHeap.Get()};
    s_Data.CommandList->SetDescriptorHeaps(1, heaps);
    s_Data.CommandList->SetPipelineState(s_Data.SkyboxPipeline.Get());
    s_Data.CommandList->SetGraphicsRootSignature(s_Data.SkyboxRootSignature.Get());
    SkyboxConstants constants;
    glm::mat4 correction(1.0f);
    correction[2][2] = 0.5f;
    correction[3][2] = 0.5f;
    constants.ViewProjection = correction * Camera::GetProjection() * glm::mat4(glm::mat3(Camera::GetViewMatrix()));
    s_Data.CommandList->SetGraphicsRootConstantBufferView(0, UploadConstants(constants));
    s_Data.CommandList->SetGraphicsRootDescriptorTable(1, GetSRVGPUHandle(s_Data.SkyboxTexture.DescriptorIndex));
    s_Data.CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    s_Data.CommandList->IASetVertexBuffers(0, 1, &s_Data.SkyboxVertexView);
    s_Data.CommandList->DrawInstanced(36, 1, 0, 0);
  }

  void DrawFullscreenPass(ID3D12PipelineState* pipeline, uint32_t sourceDescriptor,
                          uint32_t bloomDescriptor, const glm::vec4& parameters,
                          const glm::vec4& effectParameters = glm::vec4(0.0f))
  {
    ID3D12DescriptorHeap* heaps[] = {s_Data.SRVHeap.Get()};
    s_Data.CommandList->SetDescriptorHeaps(1, heaps);
    s_Data.CommandList->SetPipelineState(pipeline);
    s_Data.CommandList->SetGraphicsRootSignature(s_Data.PostRootSignature.Get());
    const std::array<glm::vec4, 2> constants{parameters, effectParameters};
    s_Data.CommandList->SetGraphicsRoot32BitConstants(0, 8, constants.data(), 0);
    s_Data.CommandList->SetGraphicsRootDescriptorTable(1, GetSRVGPUHandle(sourceDescriptor));
    s_Data.CommandList->SetGraphicsRootDescriptorTable(2, GetSRVGPUHandle(bloomDescriptor));
    s_Data.CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    s_Data.CommandList->DrawInstanced(3, 1, 0, 0);
  }

  void CompositeLighting(const RenderEffectSettings& effects)
  {
    TransitionResource(s_Data.SceneColor.Resource.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET,
                       D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    const bool bloomEnabled = effects.BloomQuality != GraphicsQuality::Off;
    if (bloomEnabled)
    {
      // Extract at full resolution before filtering, as the GL lighting MRT does.
      TransitionResource(s_Data.BloomA.Resource.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                         D3D12_RESOURCE_STATE_RENDER_TARGET);
      auto rtv = GetRTVHandle(BloomARTVIndex);
      s_Data.CommandList->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
      s_Data.CommandList->RSSetViewports(1, &s_Data.Viewport);
      s_Data.CommandList->RSSetScissorRects(1, &s_Data.ScissorRect);
      DrawFullscreenPass(s_Data.BloomExtractPipeline.Get(), s_Data.SceneColor.DescriptorIndex,
        s_Data.WhiteTexture.DescriptorIndex, glm::vec4(0.0f), glm::vec4(effects.BloomThreshold, 0, 0, 0));
      TransitionResource(s_Data.BloomA.Resource.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET,
                         D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
      const uint32_t mipCount = std::clamp(effects.BloomPassCount(), 1u, BloomMipCount);
      auto bindMip = [&](uint32_t mip)
      {
        const uint32_t width = std::max(1u, s_Data.Width >> (mip + 1));
        const uint32_t height = std::max(1u, s_Data.Height >> (mip + 1));
        const D3D12_VIEWPORT viewport{0, 0, float(width), float(height), 0, 1};
        const D3D12_RECT scissor{0, 0, LONG(width), LONG(height)};
        rtv = GetRTVHandle(BloomMipRTVBase + mip);
        s_Data.CommandList->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        s_Data.CommandList->RSSetViewports(1, &viewport);
        s_Data.CommandList->RSSetScissorRects(1, &scissor);
      };
      for (uint32_t i = 0; i < mipCount; ++i)
      {
        auto& mip = s_Data.BloomMips[i];
        TransitionResource(mip.Resource.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
        bindMip(i);
        DrawFullscreenPass(s_Data.BloomDownsamplePipeline.Get(),
          i == 0 ? s_Data.BloomA.DescriptorIndex : s_Data.BloomMips[i - 1].DescriptorIndex,
          s_Data.WhiteTexture.DescriptorIndex,
          glm::vec4(float(std::max(1u, s_Data.Width >> i)), float(std::max(1u, s_Data.Height >> i)), float(i), 0));
        TransitionResource(mip.Resource.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
      }
      for (uint32_t i = mipCount - 1; i > 0; --i)
      {
        auto& target = s_Data.BloomMips[i - 1];
        TransitionResource(target.Resource.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
        bindMip(i - 1);
        DrawFullscreenPass(s_Data.BloomUpsamplePipeline.Get(), s_Data.BloomMips[i].DescriptorIndex,
          s_Data.WhiteTexture.DescriptorIndex, glm::vec4(effects.BloomFilterRadius, 0, 0, 0));
        TransitionResource(target.Resource.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
      }
    }

    TransitionResource(s_Data.SceneResult.Resource.Get(),
                       D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                       D3D12_RESOURCE_STATE_RENDER_TARGET);
    const D3D12_CPU_DESCRIPTOR_HANDLE sceneResult = GetRTVHandle(SceneResultRTVIndex);
    s_Data.CommandList->OMSetRenderTargets(1, &sceneResult, FALSE, nullptr);
    s_Data.CommandList->RSSetViewports(1, &s_Data.Viewport);
    s_Data.CommandList->RSSetScissorRects(1, &s_Data.ScissorRect);
    DrawFullscreenPass(s_Data.CompositePipeline.Get(), s_Data.SceneColor.DescriptorIndex,
                       bloomEnabled ? s_Data.BloomMips[0].DescriptorIndex : s_Data.WhiteTexture.DescriptorIndex,
                       glm::vec4(bloomEnabled ? effects.BloomStrength : 0.0f),
                       glm::vec4(effects.BloomExposure, 0.0f, 0.0f, effects.Gamma));

    // OpenGL composites lighting and bloom before its forward pass. Keep the
    // same depth buffer so skybox, particles and debug geometry behave alike.
    const D3D12_CPU_DESCRIPTOR_HANDLE dsv = GetDSVHandle(0);
    s_Data.CommandList->OMSetRenderTargets(1, &sceneResult, FALSE, &dsv);
  }

  void PostProcessScene(bool renderForEditor, const RenderEffectSettings& effects)
  {
    TransitionResource(s_Data.SceneResult.Resource.Get(),
                       D3D12_RESOURCE_STATE_RENDER_TARGET,
                       D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    const D3D12_CPU_DESCRIPTOR_HANDLE backBuffer = GetRTVHandle(s_Data.FrameIndex);
    D3D12_CPU_DESCRIPTOR_HANDLE outputTarget = backBuffer;
    if (renderForEditor)
    {
      TransitionResource(s_Data.PostProcessColor.Resource.Get(),
                         D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                         D3D12_RESOURCE_STATE_RENDER_TARGET);
      outputTarget = GetRTVHandle(PostProcessRTVIndex);
    }
    s_Data.CommandList->OMSetRenderTargets(1, &outputTarget, FALSE, nullptr);
    s_Data.CommandList->RSSetViewports(1, &s_Data.Viewport);
    s_Data.CommandList->RSSetScissorRects(1, &s_Data.ScissorRect);
    DrawFullscreenPass(s_Data.PostProcessPipeline.Get(), s_Data.SceneResult.DescriptorIndex,
                       s_Data.WhiteTexture.DescriptorIndex,
                       glm::vec4(0.0f, static_cast<float>(s_Data.Width),
                         static_cast<float>(s_Data.Height),
                         effects.PS1Enabled ? 1.0f : 0.0f),
                       glm::vec4(0.0f, effects.PS1VirtualHeight,
                         effects.PS1ColorLevels, 0.0f));
    if (renderForEditor)
    {
      TransitionResource(s_Data.PostProcessColor.Resource.Get(),
                         D3D12_RESOURCE_STATE_RENDER_TARGET,
                         D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    }
    const D3D12_CPU_DESCRIPTOR_HANDLE dsv = GetDSVHandle(0);
    s_Data.CommandList->OMSetRenderTargets(1, &backBuffer, FALSE, &dsv);
  }

  void AppendUIBatch(uint32_t descriptorIndex, uint32_t firstVertex, uint32_t vertexCount)
  {
    if (!s_Data.PendingUIBatches.empty() &&
        s_Data.PendingUIBatches.back().DescriptorIndex == descriptorIndex &&
        s_Data.PendingUIBatches.back().FirstVertex +
          s_Data.PendingUIBatches.back().VertexCount == firstVertex)
    {
      s_Data.PendingUIBatches.back().VertexCount += vertexCount;
    }
    else
    {
      s_Data.PendingUIBatches.push_back({descriptorIndex, firstVertex, vertexCount});
    }
  }

  void AppendUIQuad(const glm::vec2& bottomLeft, const glm::vec2& topRight,
                    const glm::vec4& color, const glm::vec2& uvTopLeft,
                    const glm::vec2& uvBottomRight, uint32_t descriptorIndex)
  {
    const uint32_t firstVertex = static_cast<uint32_t>(s_Data.PendingUIVertices.size());
    const UIVertex bottomLeftVertex{bottomLeft, color, {uvTopLeft.x, uvBottomRight.y}};
    const UIVertex bottomRightVertex{{topRight.x, bottomLeft.y}, color, uvBottomRight};
    const UIVertex topRightVertex{topRight, color, {uvBottomRight.x, uvTopLeft.y}};
    const UIVertex topLeftVertex{{bottomLeft.x, topRight.y}, color, uvTopLeft};
    s_Data.PendingUIVertices.insert(s_Data.PendingUIVertices.end(), {
      bottomLeftVertex, bottomRightVertex, topRightVertex,
      topRightVertex, topLeftVertex, bottomLeftVertex
    });
    AppendUIBatch(descriptorIndex, firstVertex, 6);
  }

  void RetireSceneResources()
  {
    s_Data.PointShadowCache = {};
    for (auto& frame : s_Data.GPUFrames) frame.StatisticsReady = false;
    for (auto& [key, mesh] : s_Data.Meshes)
    {
      if (mesh.VertexBuffer) s_Data.RetiredResources.push_back(std::move(mesh.VertexBuffer));
      if (mesh.IndexBuffer) s_Data.RetiredResources.push_back(std::move(mesh.IndexBuffer));
    }
    for (auto& [key, texture] : s_Data.Textures)
      if (texture.Resource) s_Data.RetiredResources.push_back(std::move(texture.Resource));
    s_Data.Meshes.clear();
    s_Data.Textures.clear();
  }

  void ResetState()
  {
    s_Data = {};
  }
}

bool DirectX12Renderer::Init(void* nativeWindow, uint32_t width, uint32_t height)
{
  if (s_Data.Initialized) return true;
  if (!nativeWindow || width == 0 || height == 0)
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "Cannot initialize DirectX 12 without a valid window and dimensions");
    return false;
  }

  try
  {
    UINT factoryFlags = 0;
    bool validation = GetEnvironmentVariableA("GABGL_DX12_VALIDATION", nullptr, 0) != 0;
#ifdef DEBUG
    validation = true;
#endif
    if (validation)
    {
      ComPtr<ID3D12Debug> debugController;
      if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
      {
        debugController->EnableDebugLayer();
        factoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
        gablog_log(LOG_INFO, __FILE__, __LINE__, "DirectX 12 validation enabled");
      }
      else
        gablog_log(LOG_WARN, __FILE__, __LINE__, "DirectX 12 debug layer is not available");
    }

    CheckHRESULT(CreateDXGIFactory2(factoryFlags, IID_PPV_ARGS(&s_Data.Factory)),
                 "CreateDXGIFactory2");
    ComPtr<IDXGIFactory5> factory5;
    if (SUCCEEDED(s_Data.Factory.As(&factory5)))
    {
      BOOL allowTearing = FALSE;
      if (SUCCEEDED(factory5->CheckFeatureSupport(
            DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing, sizeof(allowTearing))))
        s_Data.TearingSupported = allowTearing == TRUE;
    }

    for (uint32_t adapterIndex = 0;; ++adapterIndex)
    {
      ComPtr<IDXGIAdapter1> adapter;
      const HRESULT result = s_Data.Factory->EnumAdapterByGpuPreference(
        adapterIndex, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter));
      if (result == DXGI_ERROR_NOT_FOUND) break;
      CheckHRESULT(result, "IDXGIFactory6::EnumAdapterByGpuPreference");
      DXGI_ADAPTER_DESC1 description{};
      adapter->GetDesc1(&description);
      if ((description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) continue;
      if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                      IID_PPV_ARGS(&s_Data.Device))))
      {
        s_Data.Adapter = adapter;
        break;
      }
    }
    if (!s_Data.Device)
    {
      ComPtr<IDXGIAdapter> warp;
      CheckHRESULT(s_Data.Factory->EnumWarpAdapter(IID_PPV_ARGS(&warp)),
                   "IDXGIFactory::EnumWarpAdapter");
      CheckHRESULT(warp.As(&s_Data.Adapter), "Query WARP adapter");
      CheckHRESULT(D3D12CreateDevice(s_Data.Adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                     IID_PPV_ARGS(&s_Data.Device)),
                   "D3D12CreateDevice (WARP)");
      gablog_log(LOG_WARN, __FILE__, __LINE__, "No hardware DX12 adapter found; using WARP software renderer");
    }

    if (validation) s_Data.Device.As(&s_Data.InfoQueue);

    DXGI_ADAPTER_DESC1 selectedAdapter{};
    s_Data.Adapter->GetDesc1(&selectedAdapter);
    gablog_log(LOG_INFO, __FILE__, __LINE__, "DirectX 12 adapter: %s", WideToUTF8(selectedAdapter.Description).c_str());

    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    CheckHRESULT(s_Data.Device->CreateCommandQueue(
                   &queueDesc, IID_PPV_ARGS(&s_Data.CommandQueue)),
                 "ID3D12Device::CreateCommandQueue");

    DXGI_SWAP_CHAIN_DESC1 swapChainDesc{};
    swapChainDesc.Width = width;
    swapChainDesc.Height = height;
    swapChainDesc.Format = BackBufferFormat;
    swapChainDesc.SampleDesc.Count = 1;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.BufferCount = FrameCount;
    swapChainDesc.Scaling = DXGI_SCALING_STRETCH;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    if (s_Data.TearingSupported) swapChainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
    ComPtr<IDXGISwapChain1> swapChain;
    CheckHRESULT(s_Data.Factory->CreateSwapChainForHwnd(
                   s_Data.CommandQueue.Get(), static_cast<HWND>(nativeWindow),
                   &swapChainDesc, nullptr, nullptr, &swapChain),
                 "IDXGIFactory::CreateSwapChainForHwnd");
    CheckHRESULT(s_Data.Factory->MakeWindowAssociation(
                   static_cast<HWND>(nativeWindow), DXGI_MWA_NO_ALT_ENTER),
                 "IDXGIFactory::MakeWindowAssociation");
    CheckHRESULT(swapChain.As(&s_Data.SwapChain), "Query IDXGISwapChain3");
    s_Data.FrameIndex = s_Data.SwapChain->GetCurrentBackBufferIndex();

    D3D12_DESCRIPTOR_HEAP_DESC rtvDesc{};
    rtvDesc.NumDescriptors = BloomMipRTVBase + BloomMipCount;
    rtvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    CheckHRESULT(s_Data.Device->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&s_Data.RTVHeap)),
                 "ID3D12Device::CreateDescriptorHeap (RTV)");
    s_Data.RTVDescriptorSize = s_Data.Device->GetDescriptorHandleIncrementSize(
      D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    D3D12_DESCRIPTOR_HEAP_DESC dsvDesc{};
    dsvDesc.NumDescriptors = 3;
    dsvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    CheckHRESULT(s_Data.Device->CreateDescriptorHeap(&dsvDesc, IID_PPV_ARGS(&s_Data.DSVHeap)),
                 "ID3D12Device::CreateDescriptorHeap (DSV)");
    CreateRenderTargets();
    CreateDepthBuffer(width, height);

    for (auto& allocator : s_Data.CommandAllocators)
      CheckHRESULT(s_Data.Device->CreateCommandAllocator(
                     D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)),
                   "ID3D12Device::CreateCommandAllocator");
    CheckHRESULT(s_Data.Device->CreateCommandList(
                   0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                   s_Data.CommandAllocators[s_Data.FrameIndex].Get(), nullptr,
                   IID_PPV_ARGS(&s_Data.CommandList)),
                 "ID3D12Device::CreateCommandList");
    CheckHRESULT(s_Data.CommandList->Close(), "ID3D12GraphicsCommandList::Close");

    CheckHRESULT(s_Data.Device->CreateCommandAllocator(
                   D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&s_Data.UploadAllocator)),
                 "ID3D12Device::CreateCommandAllocator (upload)");
    CheckHRESULT(s_Data.Device->CreateCommandList(
                   0, D3D12_COMMAND_LIST_TYPE_DIRECT, s_Data.UploadAllocator.Get(), nullptr,
                   IID_PPV_ARGS(&s_Data.UploadCommandList)),
                 "ID3D12Device::CreateCommandList (upload)");
    CheckHRESULT(s_Data.UploadCommandList->Close(), "ID3D12GraphicsCommandList::Close (upload)");

    CheckHRESULT(s_Data.Device->CreateFence(
                   0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&s_Data.Fence)),
                 "ID3D12Device::CreateFence");
    s_Data.FenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!s_Data.FenceEvent) ThrowHRESULT(HRESULT_FROM_WIN32(GetLastError()), "CreateEvent");

    UpdateViewport(width, height);
    s_Data.Width = width;
    s_Data.Height = height;
    s_Data.Initialized = true;
    gablog_log(LOG_INFO, __FILE__, __LINE__, "DirectX 12 initialized (%ux%u, tearing: %s)",
               width, height, s_Data.TearingSupported ? "supported" : "unavailable");
    return true;
  }
  catch (const std::exception& error)
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "DirectX 12 initialization failed: %s", error.what());
    Shutdown();
    return false;
  }
}

bool DirectX12Renderer::InitSceneRenderer()
{
  if (!s_Data.Initialized) return false;
  if (s_Data.SceneRendererInitialized) return true;
  try
  {
    D3D12_DESCRIPTOR_HEAP_DESC srvDesc{};
    srvDesc.NumDescriptors = MaxDescriptors;
    srvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    CheckHRESULT(s_Data.Device->CreateDescriptorHeap(&srvDesc, IID_PPV_ARGS(&s_Data.SRVHeap)),
                 "ID3D12Device::CreateDescriptorHeap (SRV)");
    s_Data.SRVDescriptorSize = s_Data.Device->GetDescriptorHandleIncrementSize(
      D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    for (auto& frame : s_Data.FrameUploads)
    {
      frame.Constants = CreateUploadBuffer(ConstantBufferBytes, &frame.ConstantsMapped);
      frame.UIVertices = CreateUploadBuffer(UIVertexBufferBytes, &frame.UIVerticesMapped);
    }

    CreateScenePipeline();
    CreateComputePipelines();
    CreateSkyboxPipeline();
    CreateParticlePipeline();
    CreateDebugLinePipeline();
    CreatePhysicsDebugPipeline();
    CreatePostProcessPipelines();
    CreateUIPipeline();
    constexpr std::array<uint8_t, 4> whitePixel = {255, 255, 255, 255};
    s_Data.WhiteTexture = UploadTextureBytes(whitePixel.data(), 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, 4);
    CreatePostProcessResources(s_Data.Width, s_Data.Height);
    CreateShadowMap(RenderBackend::GetEffectSettings().DirectionalShadowResolution());
    CreatePointShadowMap(RenderBackend::GetEffectSettings().PointShadowResolution());
    s_Data.PendingUIVertices.reserve(32768);
    s_Data.PendingUIBatches.reserve(64);
    s_Data.PendingDebugLines.reserve(8192);
    s_Data.SceneRendererInitialized = true;
    gablog_log(LOG_INFO, __FILE__, __LINE__, "DirectX 12 scene renderer initialized");
    return true;
  }
  catch (const std::exception& error)
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "DirectX 12 scene renderer initialization failed: %s", error.what());
    return false;
  }
}

void DirectX12Renderer::ShutdownSceneRenderer()
{
  if (!s_Data.SceneRendererInitialized) return;
  ShutdownImGui();
  try { WaitForGPU(); } catch (...) {}
  RetireSceneResources();
  for (auto& frame : s_Data.FrameUploads)
  {
    if (frame.Constants && frame.ConstantsMapped) frame.Constants->Unmap(0, nullptr);
    if (frame.UIVertices && frame.UIVerticesMapped) frame.UIVertices->Unmap(0, nullptr);
    frame = {};
  }
  s_Data.RetiredResources.clear();
  s_Data.WhiteTexture = {};
  s_Data.FontAtlases.clear();
  s_Data.SkyboxTexture = {};
  s_Data.SceneColor = {};
  s_Data.SceneResult = {};
  s_Data.GPosition = {};
  s_Data.GNormal = {};
  s_Data.GAlbedoSpec = {};
  s_Data.BloomA = {};
  s_Data.BloomMips = {};
  s_Data.HiZ = {};
  s_Data.GPUFrames = {};
  s_Data.TileGrid.Reset();
  s_Data.TileIndices.Reset();
  s_Data.ShadowOffsets.Reset();
  s_Data.CullRootSignature.Reset();
  s_Data.HiZRootSignature.Reset();
  s_Data.TileRootSignature.Reset();
  s_Data.CullPipeline.Reset();
  s_Data.HiZPipeline.Reset();
  s_Data.TilePipeline.Reset();
  s_Data.DrawSignature.Reset();
  s_Data.DepthPrepassPipeline.Reset();
  s_Data.GPUShadowPipeline.Reset();
  s_Data.GPUPointShadowPipeline.Reset();
  s_Data.BloomDownsamplePipeline.Reset();
  s_Data.BloomUpsamplePipeline.Reset();
  s_Data.PostProcessColor = {};
  s_Data.ShadowMap = {};
  s_Data.PointShadowMap = {};
  s_Data.PointShadowDepth.Reset();
  s_Data.SkyboxVertexBuffer.Reset();
  s_Data.SceneRootSignature.Reset();
  s_Data.ScenePipeline.Reset();
  s_Data.LightingPipeline.Reset();
  s_Data.SkyboxRootSignature.Reset();
  s_Data.SkyboxPipeline.Reset();
  s_Data.ParticleRootSignature.Reset();
  s_Data.ParticlePipeline.Reset();
  s_Data.DebugLineRootSignature.Reset();
  s_Data.DebugLinePipeline.Reset();
  s_Data.PhysicsDebugRootSignature.Reset();
  s_Data.PhysicsDebugPipeline.Reset();
  s_Data.PostRootSignature.Reset();
  s_Data.BloomExtractPipeline.Reset();
  s_Data.CompositePipeline.Reset();
  s_Data.PostProcessPipeline.Reset();
  s_Data.UIRootSignature.Reset();
  s_Data.UIPipeline.Reset();
  s_Data.SceneUIPipeline.Reset();
  s_Data.SRVHeap.Reset();
  s_Data.PendingUIVertices.clear();
  s_Data.PendingUIBatches.clear();
  s_Data.PendingDebugLines.clear();
  s_Data.PointShadowLightIndices.clear();
  s_Data.ModelPreviews.clear();
  s_Data.SceneRendererInitialized = false;
}

void DirectX12Renderer::ResetSceneResources()
{
  s_Data.ModelPreviews.clear();
  if (s_Data.SceneRendererInitialized) RetireSceneResources();
}

void DirectX12Renderer::SetModelPreviews(const std::vector<RenderModelPreview>& previews)
{
  s_Data.ModelPreviews = previews;
}

bool DirectX12Renderer::UploadModel(const std::shared_ptr<Model>& model)
{
  if (!s_Data.SceneRendererInitialized || !model) return false;
  try
  {
    for (Mesh& mesh : model->GetMeshes())
    {
      if (mesh.m_Vertices.empty() || mesh.m_Indices.empty()) continue;
      GPUMesh gpuMesh;
      const uint64_t vertexBytes = mesh.m_Vertices.size() * sizeof(Vertex);
      const uint64_t indexBytes = mesh.m_Indices.size() * sizeof(uint32_t);
      gpuMesh.VertexBuffer = CreateUploadBuffer(vertexBytes);
      gpuMesh.IndexBuffer = CreateUploadBuffer(indexBytes);
      void* mapped = nullptr;
      const D3D12_RANGE noRead{0, 0};
      CheckHRESULT(gpuMesh.VertexBuffer->Map(0, &noRead, &mapped), "Map model vertex buffer");
      std::memcpy(mapped, mesh.m_Vertices.data(), static_cast<size_t>(vertexBytes));
      gpuMesh.VertexBuffer->Unmap(0, nullptr);
      CheckHRESULT(gpuMesh.IndexBuffer->Map(0, &noRead, &mapped), "Map model index buffer");
      std::memcpy(mapped, mesh.m_Indices.data(), static_cast<size_t>(indexBytes));
      gpuMesh.IndexBuffer->Unmap(0, nullptr);
      gpuMesh.VertexView = {gpuMesh.VertexBuffer->GetGPUVirtualAddress(),
                            static_cast<UINT>(vertexBytes), sizeof(Vertex)};
      gpuMesh.IndexView = {gpuMesh.IndexBuffer->GetGPUVirtualAddress(),
                           static_cast<UINT>(indexBytes), DXGI_FORMAT_R32_UINT};
      gpuMesh.IndexCount = static_cast<uint32_t>(mesh.m_Indices.size());
      gpuMesh.DiffuseDescriptorIndex = s_Data.WhiteTexture.DescriptorIndex;
      gpuMesh.NormalDescriptorIndex = s_Data.WhiteTexture.DescriptorIndex;
      gpuMesh.SpecularDescriptorIndex = s_Data.WhiteTexture.DescriptorIndex;
      if (mesh.m_DiffuseTexture)
        gpuMesh.DiffuseDescriptorIndex = GetOrCreateModelTexture(mesh.m_DiffuseTexture);
      if (mesh.m_NormalTexture)
      {
        gpuMesh.NormalDescriptorIndex = GetOrCreateModelTexture(mesh.m_NormalTexture);
        gpuMesh.HasNormalMap =
          gpuMesh.NormalDescriptorIndex != s_Data.WhiteTexture.DescriptorIndex;
      }
      if (mesh.m_SpecularTexture)
      {
        gpuMesh.SpecularDescriptorIndex = GetOrCreateModelTexture(mesh.m_SpecularTexture);
        gpuMesh.HasSpecularMap =
          gpuMesh.SpecularDescriptorIndex != s_Data.WhiteTexture.DescriptorIndex;
      }
      s_Data.Meshes[&mesh] = std::move(gpuMesh);
      s_Data.PointShadowCache = {};
    }
    return true;
  }
  catch (const std::exception& error)
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "DX12 model upload failed for '%s': %s", model->m_Name.c_str(), error.what());
    return false;
  }
}

bool DirectX12Renderer::UploadSkybox(const std::shared_ptr<Texture>& cubemap)
{
  if (!s_Data.SceneRendererInitialized || !cubemap) return false;
  try
  {
    if (s_Data.SkyboxTexture.Resource)
      s_Data.RetiredResources.push_back(std::move(s_Data.SkyboxTexture.Resource));
    s_Data.SkyboxTexture = UploadCubeTexture(cubemap);
    return s_Data.SkyboxTexture.Resource != nullptr;
  }
  catch (const std::exception& error)
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "DX12 skybox upload failed: %s", error.what());
    return false;
  }
}

bool DirectX12Renderer::InitImGui()
{
  if (!s_Data.SceneRendererInitialized || !s_Data.Device || !s_Data.SRVHeap ||
      ImGui::GetCurrentContext() == nullptr)
    return false;
  if (s_Data.ImGuiInitialized) return true;

  try
  {
    s_Data.ImGuiFontDescriptor = AllocateDescriptor();
    const bool initialized = ImGui_ImplDX12_Init(
      s_Data.Device.Get(), static_cast<int>(FrameCount), BackBufferFormat,
      s_Data.SRVHeap.Get(), GetSRVCPUHandle(s_Data.ImGuiFontDescriptor),
      GetSRVGPUHandle(s_Data.ImGuiFontDescriptor));
    if (!initialized)
    {
      gablog_log(LOG_ERROR, __FILE__, __LINE__, "Could not initialize the Dear ImGui DirectX 12 backend");
      return false;
    }
    s_Data.ImGuiInitialized = true;
    return true;
  }
  catch (const std::exception& error)
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "Dear ImGui DirectX 12 initialization failed: %s", error.what());
    return false;
  }
}

void DirectX12Renderer::ShutdownImGui()
{
  if (!s_Data.ImGuiInitialized) return;
  try { WaitForGPU(); } catch (...) {}
  ImGui_ImplDX12_Shutdown();
  s_Data.ImGuiInitialized = false;
  s_Data.ImGuiFontDescriptor = std::numeric_limits<uint32_t>::max();
}

void DirectX12Renderer::BeginImGuiFrame()
{
  if (s_Data.ImGuiInitialized) ImGui_ImplDX12_NewFrame();
}

void DirectX12Renderer::RenderImGuiDrawData()
{
  if (!s_Data.ImGuiInitialized || !s_Data.FrameStarted || !ImGui::GetDrawData()) return;
  const D3D12_CPU_DESCRIPTOR_HANDLE backBuffer = GetRTVHandle(s_Data.FrameIndex);
  s_Data.CommandList->OMSetRenderTargets(1, &backBuffer, FALSE, nullptr);
  s_Data.CommandList->RSSetViewports(1, &s_Data.Viewport);
  s_Data.CommandList->RSSetScissorRects(1, &s_Data.ScissorRect);
  ID3D12DescriptorHeap* heaps[] = {s_Data.SRVHeap.Get()};
  s_Data.CommandList->SetDescriptorHeaps(1, heaps);
  ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), s_Data.CommandList.Get());
}

uint64_t DirectX12Renderer::GetEditorTextureID()
{
  if (!s_Data.PostProcessColor.Resource || !s_Data.SRVHeap) return 0;
  return GetSRVGPUHandle(s_Data.PostProcessColor.DescriptorIndex).ptr;
}

void DirectX12Renderer::Shutdown()
{
  ShutdownSceneRenderer();
  try
  {
    if (s_Data.CommandQueue && s_Data.Fence && s_Data.FenceEvent) WaitForGPU();
  }
  catch (const std::exception& error)
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "DirectX 12 shutdown synchronization failed: %s", error.what());
  }
  if (s_Data.FenceEvent)
  {
    CloseHandle(s_Data.FenceEvent);
    s_Data.FenceEvent = nullptr;
  }
  ResetState();
}

bool DirectX12Renderer::Resize(uint32_t width, uint32_t height)
{
  if (!s_Data.Initialized || width == 0 || height == 0) return false;
  if (width == s_Data.Width && height == s_Data.Height) return true;
  if (s_Data.FrameStarted) return false;
  try
  {
    WaitForGPU();
    ReleaseRenderTargets();
    s_Data.FrameFenceValues.fill(0);
    const UINT flags = s_Data.TearingSupported
      ? static_cast<UINT>(DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING) : 0u;
    CheckHRESULT(s_Data.SwapChain->ResizeBuffers(
                   FrameCount, width, height, BackBufferFormat, flags),
                 "IDXGISwapChain::ResizeBuffers");
    s_Data.FrameIndex = s_Data.SwapChain->GetCurrentBackBufferIndex();
    CreateRenderTargets();
    CreateDepthBuffer(width, height);
    if (s_Data.SceneRendererInitialized)
      CreatePostProcessResources(width, height);
    UpdateViewport(width, height);
    s_Data.Width = width;
    s_Data.Height = height;
    return true;
  }
  catch (const std::exception& error)
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "DirectX 12 resize failed: %s", error.what());
    return false;
  }
}

bool DirectX12Renderer::BeginFrame()
{
  if (!s_Data.Initialized || s_Data.FrameStarted) return false;
  try
  {
    WaitForFrame(s_Data.FrameIndex);
    s_Data.FrameFailed = false;
    auto& allocator = s_Data.CommandAllocators[s_Data.FrameIndex];
    CheckHRESULT(allocator->Reset(), "ID3D12CommandAllocator::Reset");
    CheckHRESULT(s_Data.CommandList->Reset(allocator.Get(), nullptr),
                 "ID3D12GraphicsCommandList::Reset");
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = s_Data.RenderTargets[s_Data.FrameIndex].Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    s_Data.CommandList->ResourceBarrier(1, &barrier);

    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = GetRTVHandle(s_Data.FrameIndex);
    const D3D12_CPU_DESCRIPTOR_HANDLE dsv = s_Data.DSVHeap->GetCPUDescriptorHandleForHeapStart();
    static constexpr float clearColor[] = {0.008f, 0.012f, 0.025f, 1.0f};
    s_Data.CommandList->ClearRenderTargetView(rtv, clearColor, 0, nullptr);
    s_Data.CommandList->ClearDepthStencilView(
      dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
    s_Data.CommandList->OMSetRenderTargets(1, &rtv, FALSE, &dsv);
    s_Data.CommandList->RSSetViewports(1, &s_Data.Viewport);
    s_Data.CommandList->RSSetScissorRects(1, &s_Data.ScissorRect);

    auto& frame = s_Data.FrameUploads[s_Data.FrameIndex];
    frame.ConstantsOffset = 0;
    frame.UIVerticesOffset = 0;
    s_Data.PendingUIVertices.clear();
    s_Data.PendingUIBatches.clear();
    s_Data.FrameStarted = true;
    return true;
  }
  catch (const std::exception& error)
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "DirectX 12 frame start failed: %s", error.what());
    return false;
  }
}

bool DirectX12Renderer::EndFrame(bool vSync)
{
  if (!s_Data.Initialized || !s_Data.FrameStarted) return false;
  try
  {
    if (!s_Data.PendingUIVertices.empty()) EndScene();
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = s_Data.RenderTargets[s_Data.FrameIndex].Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    s_Data.CommandList->ResourceBarrier(1, &barrier);
    CheckHRESULT(s_Data.CommandList->Close(), "ID3D12GraphicsCommandList::Close");
    ID3D12CommandList* commandLists[] = {s_Data.CommandList.Get()};
    s_Data.CommandQueue->ExecuteCommandLists(1, commandLists);
    const UINT syncInterval = vSync ? 1u : 0u;
    const UINT presentFlags = !vSync && s_Data.TearingSupported
      ? static_cast<UINT>(DXGI_PRESENT_ALLOW_TEARING) : 0u;
    CheckHRESULT(s_Data.SwapChain->Present(syncInterval, presentFlags),
                 "IDXGISwapChain::Present");
    const uint64_t fenceValue = ++s_Data.NextFenceValue;
    CheckHRESULT(s_Data.CommandQueue->Signal(s_Data.Fence.Get(), fenceValue),
                 "ID3D12CommandQueue::Signal");
    s_Data.FrameFenceValues[s_Data.FrameIndex] = fenceValue;
    s_Data.FrameIndex = s_Data.SwapChain->GetCurrentBackBufferIndex();
    LogD3D12Messages();
    s_Data.FrameStarted = false;
    return !s_Data.FrameFailed;
  }
  catch (const std::exception& error)
  {
    s_Data.FrameStarted = false;
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "DirectX 12 presentation failed: %s", error.what());
    return false;
  }
}

void DirectX12Renderer::DrawScene(DeltaTime& dt, const std::function<void()>& sceneLogic,
                                  bool advanceSimulation, bool renderForEditor,
                                  const RenderEffectSettings& effects)
{
  if (!s_Data.FrameStarted || !s_Data.SceneRendererInitialized) return;
  if (advanceSimulation) sceneLogic();
  if (advanceSimulation)
  {
    ModelManager::UpdateControllers(dt);
    PhysX::Simulate(dt);
    ModelManager::UpdateTransforms(dt);
    Camera::OnUpdate(dt);
    AudioManager::SetListenerLocation(Camera::GetPosition());
    AudioManager::SetListenerOrientation(Camera::GetForwardDirection(), Camera::GetUpDirection());
    AudioManager::UpdateAllMusic();
  }
  try
  {
    const uint32_t shadowResolution = effects.DirectionalShadowResolution();
    const uint32_t pointShadowResolution = effects.PointShadowResolution();
    const bool recreateDirectionalShadow = shadowResolution != s_Data.ShadowMapSize;
    const bool recreatePointShadow = pointShadowResolution != s_Data.PointShadowMapSize;
    if (recreateDirectionalShadow || recreatePointShadow)
      WaitForGPU();
    if (recreateDirectionalShadow)
    {
      s_Data.ShadowMap.Resource.Reset();
      s_Data.ShadowMapReadable = false;
      CreateShadowMap(shadowResolution);
    }
    if (recreatePointShadow)
    {
      s_Data.PointShadowMap.Resource.Reset();
      s_Data.PointShadowDepth.Reset();
      s_Data.PointShadowMapReadable = false;
      CreatePointShadowMap(pointShadowResolution);
    }
    PrepareGPUScene(effects);
    DrawShadowMap(effects);
    DrawPointShadowMaps(effects);
    DispatchGPUCull(false, DirectXViewProjection(), effects);
    DrawDepthPrepass();
    BuildHiZ();
    DispatchGPUCull(true, DirectXViewProjection(), effects);
    BeginGeometryPass();
    DrawGPUScene(s_Data.ScenePipeline.Get());
    EndGeometryPass();
    DispatchTileLights(effects);
    DrawLightingPass(effects);
    CompositeLighting(effects);
    DrawSkybox();
    if (RenderBackend::DebugSettings().Physics) DrawPhysicsDebug();
    DrawDebugVisualizations();
    ParticleRenderer::UpdateAndRender(dt);
    s_Data.UIToSceneColor = true;
    DrawDebug2D();
    s_Data.UIToSceneColor = false;
    PostProcessScene(renderForEditor, effects);
  }
  catch (const std::exception& error)
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "DirectX 12 scene rendering failed: %s", error.what());
    s_Data.FrameFailed = true;
  }
}

void DirectX12Renderer::DrawParticles(const std::vector<ParticleRenderInstance>& instances)
{
  if (!s_Data.FrameStarted || instances.empty()) return;
  std::vector<ParticleVertex> vertices;
  vertices.reserve(instances.size() * 6);
  static constexpr std::array<glm::vec2, 6> corners = {
    glm::vec2(-0.5f,-0.5f), glm::vec2(0.5f,-0.5f), glm::vec2(0.5f,0.5f),
    glm::vec2(0.5f,0.5f), glm::vec2(-0.5f,0.5f), glm::vec2(-0.5f,-0.5f)
  };
  for (const ParticleRenderInstance& instance : instances)
  {
    const float cosine = std::cos(instance.Rotation);
    const float sine = std::sin(instance.Rotation);
    for (const glm::vec2 corner : corners)
    {
      const glm::vec2 rotated(cosine * corner.x - sine * corner.y,
                              sine * corner.x + cosine * corner.y);
      const glm::vec3 world = glm::vec3(instance.PositionAndSize)
        + glm::vec3(instance.RightAndStyle) * rotated.x * instance.PositionAndSize.w
        + glm::vec3(instance.Up) * rotated.y * instance.PositionAndSize.w;
      vertices.push_back({world, instance.Color, corner * 2.0f, instance.RightAndStyle.w});
    }
  }
  auto& frame = s_Data.FrameUploads[s_Data.FrameIndex];
  const uint64_t byteCount = vertices.size() * sizeof(ParticleVertex);
  const uint64_t offset = AlignUp(frame.UIVerticesOffset, 16);
  if (offset + byteCount > UIVertexBufferBytes) return;
  std::memcpy(frame.UIVerticesMapped + offset, vertices.data(), static_cast<size_t>(byteCount));
  frame.UIVerticesOffset = offset + byteCount;
  const D3D12_VERTEX_BUFFER_VIEW view{
    frame.UIVertices->GetGPUVirtualAddress() + offset,
    static_cast<UINT>(byteCount), sizeof(ParticleVertex)};
  SkyboxConstants constants;
  constants.ViewProjection = DirectXViewProjection();
  s_Data.CommandList->SetPipelineState(s_Data.ParticlePipeline.Get());
  s_Data.CommandList->SetGraphicsRootSignature(s_Data.ParticleRootSignature.Get());
  s_Data.CommandList->SetGraphicsRootConstantBufferView(0, UploadConstants(constants));
  s_Data.CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  s_Data.CommandList->IASetVertexBuffers(0, 1, &view);
  s_Data.CommandList->DrawInstanced(static_cast<UINT>(vertices.size()), 1, 0, 0);
}

uint64_t DirectX12Renderer::CreateFontAtlas(
  const uint8_t* pixels, uint32_t width, uint32_t height)
{
  if (!s_Data.SceneRendererInitialized || !pixels || width == 0 || height == 0)
    return 0;
  try
  {
    GPUTexture atlas = UploadTextureBytes(
      pixels, width, height, DXGI_FORMAT_R8_UNORM, 1);
    const uint64_t handle = atlas.DescriptorIndex;
    s_Data.FontAtlases.emplace(handle, std::move(atlas));
    return handle;
  }
  catch (const std::exception& error)
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "DX12 font atlas upload failed: %s", error.what());
    return 0;
  }
}

void DirectX12Renderer::DestroyFontAtlas(uint64_t handle)
{
  const auto atlas = s_Data.FontAtlases.find(handle);
  if (atlas == s_Data.FontAtlases.end()) return;
  try { WaitForGPU(); } catch (...) {}
  s_Data.FontAtlases.erase(atlas);
}

void DirectX12Renderer::BeginDebugLines()
{
  s_Data.PendingDebugLines.clear();
}

void DirectX12Renderer::DrawDebugLine(const glm::vec3& start, const glm::vec3& end,
                                      const glm::vec4& color)
{
  s_Data.PendingDebugLines.push_back({start, color});
  s_Data.PendingDebugLines.push_back({end, color});
}

void DirectX12Renderer::EndDebugLines()
{
  if (!s_Data.FrameStarted || s_Data.PendingDebugLines.empty()) return;
  auto& frame = s_Data.FrameUploads[s_Data.FrameIndex];
  const uint64_t byteCount = s_Data.PendingDebugLines.size() * sizeof(DebugLineVertex);
  const uint64_t offset = AlignUp(frame.UIVerticesOffset, 16);
  if (offset + byteCount > UIVertexBufferBytes) return;
  std::memcpy(frame.UIVerticesMapped + offset, s_Data.PendingDebugLines.data(), static_cast<size_t>(byteCount));
  frame.UIVerticesOffset = offset + byteCount;
  const D3D12_VERTEX_BUFFER_VIEW view{
    frame.UIVertices->GetGPUVirtualAddress() + offset,
    static_cast<UINT>(byteCount), sizeof(DebugLineVertex)};
  SkyboxConstants constants;
  constants.ViewProjection = DirectXViewProjection();
  s_Data.CommandList->SetPipelineState(s_Data.DebugLinePipeline.Get());
  s_Data.CommandList->SetGraphicsRootSignature(s_Data.DebugLineRootSignature.Get());
  s_Data.CommandList->SetGraphicsRootConstantBufferView(0, UploadConstants(constants));
  s_Data.CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_LINELIST);
  s_Data.CommandList->IASetVertexBuffers(0, 1, &view);
  s_Data.CommandList->DrawInstanced(static_cast<UINT>(s_Data.PendingDebugLines.size()), 1, 0, 0);
  s_Data.PendingDebugLines.clear();
}

void DirectX12Renderer::DrawPhysicsDebug()
{
  if (!s_Data.FrameStarted || !s_Data.PhysicsDebugPipeline) return;

  const D3D12_CPU_DESCRIPTOR_HANDLE rtv = GetRTVHandle(SceneResultRTVIndex);
  const D3D12_CPU_DESCRIPTOR_HANDLE dsv = GetDSVHandle(0);
  s_Data.CommandList->OMSetRenderTargets(1, &rtv, FALSE, &dsv);
  s_Data.CommandList->RSSetViewports(1, &s_Data.Viewport);
  s_Data.CommandList->RSSetScissorRects(1, &s_Data.ScissorRect);
  s_Data.CommandList->SetPipelineState(s_Data.PhysicsDebugPipeline.Get());
  s_Data.CommandList->SetGraphicsRootSignature(s_Data.PhysicsDebugRootSignature.Get());
  s_Data.CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

  PhysicsDebugConstants constants;
  constants.ViewProjection = DirectXViewProjection();
  for (const std::string& modelName : ModelManager::GetModelNames())
  {
    const auto model = ModelManager::GetModel(modelName);
    if (!model || model->GetPhysXMeshType() != MeshType::CONVEXMESH) continue;
    for (const glm::mat4& transform : model->m_InstanceTransforms)
    {
      constants.Model = transform;
      for (const Mesh& mesh : model->GetMeshes())
      {
        const auto found = s_Data.Meshes.find(&mesh);
        if (found == s_Data.Meshes.end()) continue;
        const GPUMesh& gpu = found->second;
        s_Data.CommandList->SetGraphicsRootConstantBufferView(0, UploadConstants(constants));
        s_Data.CommandList->IASetVertexBuffers(0, 1, &gpu.VertexView);
        s_Data.CommandList->IASetIndexBuffer(&gpu.IndexView);
        s_Data.CommandList->DrawIndexedInstanced(gpu.IndexCount, 1, 0, 0, 0);
      }
    }
  }
}

void DirectX12Renderer::BeginScene()
{
  s_Data.PendingUIVertices.clear();
  s_Data.PendingUIBatches.clear();
}

void DirectX12Renderer::PrepareScreenUI(const glm::vec4& clearColor, bool clear)
{
  if (!s_Data.FrameStarted) return;
  s_Data.UIToSceneColor = false;
  const D3D12_CPU_DESCRIPTOR_HANDLE rtv = GetRTVHandle(s_Data.FrameIndex);
  const D3D12_CPU_DESCRIPTOR_HANDLE dsv = GetDSVHandle(0);
  if (clear)
  {
    s_Data.CommandList->ClearRenderTargetView(rtv, &clearColor.x, 0, nullptr);
    s_Data.CommandList->ClearDepthStencilView(
      dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
  }
  s_Data.CommandList->OMSetRenderTargets(1, &rtv, FALSE, &dsv);
  s_Data.CommandList->RSSetViewports(1, &s_Data.Viewport);
  s_Data.CommandList->RSSetScissorRects(1, &s_Data.ScissorRect);
}

void DirectX12Renderer::EndScene()
{
  if (!s_Data.FrameStarted || s_Data.PendingUIVertices.empty()) return;
  try
  {
    auto& frame = s_Data.FrameUploads[s_Data.FrameIndex];
    const uint64_t byteCount = s_Data.PendingUIVertices.size() * sizeof(UIVertex);
    const uint64_t offset = AlignUp(frame.UIVerticesOffset, alignof(UIVertex));
    if (offset + byteCount > UIVertexBufferBytes)
      throw std::runtime_error("DX12 UI vertex buffer exhausted");
    std::memcpy(frame.UIVerticesMapped + offset, s_Data.PendingUIVertices.data(),
                static_cast<size_t>(byteCount));

    D3D12_VERTEX_BUFFER_VIEW view{};
    view.BufferLocation = frame.UIVertices->GetGPUVirtualAddress() + offset;
    view.SizeInBytes = static_cast<UINT>(byteCount);
    view.StrideInBytes = sizeof(UIVertex);
    frame.UIVerticesOffset = offset + byteCount;

    ID3D12DescriptorHeap* heaps[] = {s_Data.SRVHeap.Get()};
    s_Data.CommandList->SetDescriptorHeaps(1, heaps);
    if (s_Data.UIToSceneColor)
    {
      const D3D12_CPU_DESCRIPTOR_HANDLE sceneRTV = GetRTVHandle(SceneResultRTVIndex);
      const D3D12_CPU_DESCRIPTOR_HANDLE dsv = GetDSVHandle(0);
      s_Data.CommandList->OMSetRenderTargets(1, &sceneRTV, FALSE, &dsv);
    }
    s_Data.CommandList->SetPipelineState(
      s_Data.UIToSceneColor ? s_Data.SceneUIPipeline.Get() : s_Data.UIPipeline.Get());
    s_Data.CommandList->SetGraphicsRootSignature(s_Data.UIRootSignature.Get());
    const std::array<float, 2> screenSize = {
      static_cast<float>(s_Data.Width), static_cast<float>(s_Data.Height)};
    s_Data.CommandList->SetGraphicsRoot32BitConstants(0, 2, screenSize.data(), 0);
    s_Data.CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    s_Data.CommandList->IASetVertexBuffers(0, 1, &view);
    for (const UIBatch& batch : s_Data.PendingUIBatches)
    {
      s_Data.CommandList->SetGraphicsRootDescriptorTable(
        1, GetSRVGPUHandle(batch.DescriptorIndex));
      s_Data.CommandList->DrawInstanced(
        batch.VertexCount, 1, batch.FirstVertex, 0);
    }
    s_Data.PendingUIVertices.clear();
    s_Data.PendingUIBatches.clear();
  }
  catch (const std::exception& error)
  {
    s_Data.PendingUIVertices.clear();
    s_Data.PendingUIBatches.clear();
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "DirectX 12 UI rendering failed: %s", error.what());
  }
}

void DirectX12Renderer::DrawQuad(const glm::mat4& transform, const glm::vec4& color)
{
  static constexpr std::array<glm::vec4, 4> corners = {
    glm::vec4(-0.5f, -0.5f, 0.0f, 1.0f), glm::vec4(0.5f, -0.5f, 0.0f, 1.0f),
    glm::vec4(0.5f, 0.5f, 0.0f, 1.0f), glm::vec4(-0.5f, 0.5f, 0.0f, 1.0f)};
  std::array<glm::vec2, 4> positions{};
  for (size_t i = 0; i < corners.size(); ++i)
    positions[i] = glm::vec2(transform * corners[i]);
  const uint32_t firstVertex = static_cast<uint32_t>(s_Data.PendingUIVertices.size());
  constexpr glm::vec2 whiteUV(0.5f);
  const UIVertex vertices[] = {
    {positions[0], color, whiteUV}, {positions[1], color, whiteUV},
    {positions[2], color, whiteUV}, {positions[2], color, whiteUV},
    {positions[3], color, whiteUV}, {positions[0], color, whiteUV}
  };
  s_Data.PendingUIVertices.insert(s_Data.PendingUIVertices.end(),
                                  std::begin(vertices), std::end(vertices));
  AppendUIBatch(s_Data.WhiteTexture.DescriptorIndex, firstVertex, 6);
}

void DirectX12Renderer::DrawText(const Font* font, const std::string& text,
                                 const glm::vec2& position,
                                 float size, const glm::vec4& color)
{
  if (!font || font->m_AtlasHandle == 0 || font->m_Characters.empty() || text.empty())
    return;
  const auto atlas = s_Data.FontAtlases.find(font->m_AtlasHandle);
  if (atlas == s_Data.FontAtlases.end()) return;
  float textWidth = 0.0f;
  for (const unsigned char character : text)
    if (const auto glyph = font->m_Characters.find(static_cast<char>(character));
        glyph != font->m_Characters.end())
      textWidth += static_cast<float>(glyph->second.Advance >> 6) * size;

  const float baselineY = (font->m_Descender - font->m_Ascender) * size * 0.5f;
  float cursorX = -textWidth * 0.5f;
  for (const unsigned char character : text)
  {
    const auto glyphIt = font->m_Characters.find(static_cast<char>(character));
    if (glyphIt == font->m_Characters.end()) continue;
    const Character& glyph = glyphIt->second;
    const float x = position.x + cursorX + glyph.Bearing.x * size;
    const float y = position.y + baselineY + (glyph.Bearing.y - glyph.Size.y) * size;
    const float width = glyph.Size.x * size;
    const float height = glyph.Size.y * size;
    if (width > 0.0f && height > 0.0f)
      AppendUIQuad({x, y}, {x + width, y + height}, color,
                   glyph.UVTopLeft, glyph.UVBottomRight,
                   atlas->second.DescriptorIndex);
    cursorX += static_cast<float>(glyph.Advance >> 6) * size;
  }
}

bool DirectX12Renderer::IsInitialized()
{
  return s_Data.Initialized;
}

#else

bool DirectX12Renderer::Init(void*, uint32_t, uint32_t) { return false; }
void DirectX12Renderer::Shutdown() {}
bool DirectX12Renderer::Resize(uint32_t, uint32_t) { return false; }
bool DirectX12Renderer::BeginFrame() { return false; }
bool DirectX12Renderer::EndFrame(bool) { return false; }
bool DirectX12Renderer::IsInitialized() { return false; }
bool DirectX12Renderer::InitSceneRenderer() { return false; }
void DirectX12Renderer::ShutdownSceneRenderer() {}
void DirectX12Renderer::ResetSceneResources() {}
bool DirectX12Renderer::UploadModel(const std::shared_ptr<Model>&) { return false; }
bool DirectX12Renderer::UploadSkybox(const std::shared_ptr<Texture>&) { return false; }
void DirectX12Renderer::SetModelPreviews(const std::vector<RenderModelPreview>&) {}
bool DirectX12Renderer::InitImGui() { return false; }
void DirectX12Renderer::ShutdownImGui() {}
void DirectX12Renderer::BeginImGuiFrame() {}
void DirectX12Renderer::RenderImGuiDrawData() {}
uint64_t DirectX12Renderer::GetEditorTextureID() { return 0; }
void DirectX12Renderer::DrawScene(DeltaTime&, const std::function<void()>&, bool, bool,
                                  const RenderEffectSettings&) {}
void DirectX12Renderer::DrawParticles(const std::vector<ParticleRenderInstance>&) {}
uint64_t DirectX12Renderer::CreateFontAtlas(const uint8_t*, uint32_t, uint32_t) { return 0; }
void DirectX12Renderer::DestroyFontAtlas(uint64_t) {}
void DirectX12Renderer::BeginDebugLines() {}
void DirectX12Renderer::DrawDebugLine(const glm::vec3&, const glm::vec3&, const glm::vec4&) {}
void DirectX12Renderer::EndDebugLines() {}
void DirectX12Renderer::DrawPhysicsDebug() {}
void DirectX12Renderer::PrepareScreenUI(const glm::vec4&, bool) {}
void DirectX12Renderer::BeginScene() {}
void DirectX12Renderer::EndScene() {}
void DirectX12Renderer::DrawQuad(const glm::mat4&, const glm::vec4&) {}
void DirectX12Renderer::DrawText(const Font*, const std::string&, const glm::vec2&, float,
                                 const glm::vec4&) {}

#endif
