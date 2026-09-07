// Headless WARP regression: execute the real scene/shadow vertex shaders and
// read back world/clip positions. Visibility counts cannot detect wrong models.
#define NOMINMAX
#include <Windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include "Shader.h"
#include "DirectX12ShadowMath.h"
#include <glm/gtc/matrix_transform.hpp>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <source_location>

using Microsoft::WRL::ComPtr;

static void Check(HRESULT result, std::source_location location = std::source_location::current())
{
  if (FAILED(result)) throw std::runtime_error("D3D12 HRESULT " + std::to_string(result) + " at line " + std::to_string(location.line()));
}

static ComPtr<ID3D12Resource> Buffer(ID3D12Device* device, UINT64 bytes,
  D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state, const void* data = nullptr)
{
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = type;
  D3D12_RESOURCE_DESC desc{};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Width = bytes;
  desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  ComPtr<ID3D12Resource> resource;
  Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state,
    nullptr, IID_PPV_ARGS(&resource)));
  if (data)
  {
    void* mapped;
    Check(resource->Map(0, nullptr, &mapped));
    std::memcpy(mapped, data, size_t(bytes));
    resource->Unmap(0, nullptr);
  }
  return resource;
}

struct TestVertex
{
  glm::vec3 position{0.25f, -0.5f, 0.75f}, normal{0, 1, 0};
  glm::vec2 uv{0};
  glm::vec3 tangent{1, 0, 0}, bitangent{0, 0, 1};
  glm::ivec4 bones{0, -1, -1, -1};
  glm::vec4 weights{1, 0, 0, 0};
};

static void Run(ID3D12Device* device, const char* entry, bool animated)
{
  const auto shader = Shader::CompileSlang("res/shaders/scene.slang", entry, "vs_5_1");
  ComPtr<ID3D12ShaderReflection> reflection;
  Check(D3DReflect(shader.GetBufferPointer(), shader.GetBufferSize(), IID_PPV_ARGS(&reflection)));
  auto* cb = reflection->GetConstantBufferByName("SceneConstants");
  auto* cbType = cb->GetVariableByName("SceneConstants")->GetType();
  constexpr UINT ConstantStride = 16384;
  std::array<std::byte, ConstantStride * 3> constants{};
  auto set = [&](UINT draw, const char* name, const auto& value)
  {
    D3D12_SHADER_TYPE_DESC desc{};
    Check(cbType->GetMemberTypeByName(name)->GetDesc(&desc));
    if (desc.Offset + sizeof(value) > ConstantStride) throw std::runtime_error("CB overflow");
    std::memcpy(constants.data() + draw * ConstantStride + desc.Offset, &value, sizeof(value));
  };
  const auto view = glm::scale(glm::mat4(1), glm::vec3(0.1f, 0.2f, 0.3f));
  const auto bone = glm::translate(glm::mat4(1), glm::vec3(2, -1, 3));
  // Deliberately skip slots and reorder draws, as happens after GPU culling.
  const std::array<UINT, 3> bases{0, 3, 1}, counts{1, 2, 1};
  for (UINT draw = 0; draw < bases.size(); ++draw)
  {
    set(draw, "ViewProjection", view);
    set(draw, "LightViewProjection", view);
    set(draw, "Bones", bone);
    set(draw, "CameraPosition", glm::vec4(0, 0, 0, animated ? 1 : 0));
    set(draw, "MaterialIndices", glm::uvec4(0, 0, 0, bases[draw]));
    set(draw, "Resolution", glm::vec4(1280, 720, 240, 0));
  }
  std::array<glm::mat4, 5> transforms;
  for (UINT i = 0; i < transforms.size(); ++i)
  {
    auto m = glm::translate(glm::mat4(1), glm::vec3(10 * i + 1, -3.0f * i, 2 * i + 4));
    m = glm::rotate(m, 0.3f * (i + 1), glm::normalize(glm::vec3(1, 2, 3)));
    transforms[i] = glm::scale(m, glm::vec3(i + 1, 0.5f, 2));
  }
  auto cbResource = Buffer(device, sizeof(constants), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, constants.data());
  auto matrixResource = Buffer(device, sizeof(transforms), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, transforms.data());
  std::array<std::byte, 256> shadowConstants{};
  std::memcpy(shadowConstants.data(), &view, sizeof(view));
  auto shadowResource = Buffer(device, sizeof(shadowConstants), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, shadowConstants.data());
  TestVertex vertex;
  auto vb = Buffer(device, sizeof(vertex), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, &vertex);
  const UINT index = 0;
  auto ib = Buffer(device, sizeof(index), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, &index);

  D3D12_ROOT_PARAMETER params[3]{};
  params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
  params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
  params[1].Descriptor.ShaderRegister = 5;
  params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
  params[2].Descriptor.ShaderRegister = 1;
  D3D12_ROOT_SIGNATURE_DESC rootDesc{3, params, 0, nullptr,
    D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT | D3D12_ROOT_SIGNATURE_FLAG_ALLOW_STREAM_OUTPUT};
  ComPtr<ID3DBlob> serialized, errors;
  Check(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &errors));
  ComPtr<ID3D12RootSignature> root;
  Check(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&root)));
  const D3D12_INPUT_ELEMENT_DESC inputs[]{
    {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, offsetof(TestVertex, position)},
    {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, offsetof(TestVertex, normal)},
    {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, offsetof(TestVertex, uv)},
    {"TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, offsetof(TestVertex, tangent)},
    {"BITANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, offsetof(TestVertex, bitangent)},
    {"BONEIDS", 0, DXGI_FORMAT_R32G32B32A32_SINT, 0, offsetof(TestVertex, bones)},
    {"BONEWEIGHTS", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, offsetof(TestVertex, weights)}
  };
  const D3D12_SO_DECLARATION_ENTRY outputs[]{
    {0, "POSITION", 0, 0, 3, 0}, {0, "SV_POSITION", 0, 0, 4, 0}};
  const UINT stride = 7 * sizeof(float);
  D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
  pso.pRootSignature = root.Get();
  pso.VS = {shader.GetBufferPointer(), shader.GetBufferSize()};
  pso.StreamOutput = {outputs, 2, &stride, 1, D3D12_SO_NO_RASTERIZED_STREAM};
  pso.InputLayout = {inputs, UINT(std::size(inputs))};
  pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
  pso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
  pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
  pso.SampleMask = UINT_MAX;
  pso.SampleDesc.Count = 1;
  ComPtr<ID3D12PipelineState> pipeline;
  Check(device->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&pipeline)));

  // Match the production ExecuteIndirect ABI: CBV, VBV, IBV, indexed draw.
  struct Command
  {
    D3D12_GPU_VIRTUAL_ADDRESS cb;
    D3D12_VERTEX_BUFFER_VIEW vb;
    D3D12_INDEX_BUFFER_VIEW ib;
    D3D12_DRAW_INDEXED_ARGUMENTS draw;
    UINT padding = 0;
  };
  static_assert(sizeof(Command) == 64);
  std::array<Command, 3> commands{};
  for (UINT i = 0; i < commands.size(); ++i)
    commands[i] = {cbResource->GetGPUVirtualAddress() + i * ConstantStride,
      {vb->GetGPUVirtualAddress(), sizeof(vertex), sizeof(vertex)},
      {ib->GetGPUVirtualAddress(), sizeof(index), DXGI_FORMAT_R32_UINT}, {1, counts[i], 0, 0, 0}};
  auto args = Buffer(device, sizeof(commands), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, commands.data());
  D3D12_INDIRECT_ARGUMENT_DESC arguments[4]{};
  arguments[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW;
  arguments[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW;
  arguments[2].Type = D3D12_INDIRECT_ARGUMENT_TYPE_INDEX_BUFFER_VIEW;
  arguments[3].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
  D3D12_COMMAND_SIGNATURE_DESC signatureDesc{sizeof(Command), 4, arguments};
  ComPtr<ID3D12CommandSignature> signature;
  Check(device->CreateCommandSignature(&signatureDesc, root.Get(), IID_PPV_ARGS(&signature)));

  constexpr UINT DataBytes = 4 * 7 * sizeof(float), CounterOffset = 256, BufferBytes = 264;
  std::array<std::byte, BufferBytes> zero{};
  auto zeros = Buffer(device, BufferBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, zero.data());
  auto output = Buffer(device, BufferBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
  auto readback = Buffer(device, BufferBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
  ComPtr<ID3D12CommandQueue> queue;
  D3D12_COMMAND_QUEUE_DESC queueDesc{};
  Check(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)));
  ComPtr<ID3D12CommandAllocator> allocator;
  Check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
  ComPtr<ID3D12GraphicsCommandList> list;
  Check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), pipeline.Get(), IID_PPV_ARGS(&list)));
  auto transition = [&](D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
  {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {output.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    list->ResourceBarrier(1, &barrier);
  };
  list->CopyResource(output.Get(), zeros.Get());
  transition(D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_STREAM_OUT);
  list->SetGraphicsRootSignature(root.Get());
  list->SetGraphicsRootShaderResourceView(1, matrixResource->GetGPUVirtualAddress());
  list->SetGraphicsRootConstantBufferView(2, shadowResource->GetGPUVirtualAddress());
  list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
  D3D12_STREAM_OUTPUT_BUFFER_VIEW so{output->GetGPUVirtualAddress(), DataBytes, output->GetGPUVirtualAddress() + CounterOffset};
  list->SOSetTargets(0, 1, &so);
  list->ExecuteIndirect(signature.Get(), UINT(commands.size()), args.Get(), 0, nullptr, 0);
  transition(D3D12_RESOURCE_STATE_STREAM_OUT, D3D12_RESOURCE_STATE_COPY_SOURCE);
  list->CopyResource(readback.Get(), output.Get());
  Check(list->Close());
  ID3D12CommandList* lists[]{list.Get()};
  queue->ExecuteCommandLists(1, lists);
  ComPtr<ID3D12Fence> fence;
  Check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
  Check(queue->Signal(fence.Get(), 1));
  HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  if (!event) throw std::runtime_error("CreateEvent failed");
  const HRESULT result = fence->SetEventOnCompletion(1, event);
  const DWORD wait = SUCCEEDED(result) ? WaitForSingleObject(event, 30000) : WAIT_FAILED;
  CloseHandle(event);
  Check(result);
  if (wait != WAIT_OBJECT_0) throw std::runtime_error("GPU timeout");
  void* mapped;
  Check(readback->Map(0, nullptr, &mapped));
  std::array<std::byte, BufferBytes> actual{};
  std::memcpy(actual.data(), mapped, BufferBytes);
  readback->Unmap(0, nullptr);
  UINT64 written;
  std::memcpy(&written, actual.data() + CounterOffset, sizeof(written));
  if (written != DataBytes) throw std::runtime_error("Unexpected stream output size");
  std::array<float, DataBytes / sizeof(float)> positions;
  std::memcpy(positions.data(), actual.data(), DataBytes);
  UINT vertexIndex = 0;
  for (UINT draw = 0; draw < commands.size(); ++draw)
    for (UINT instance = 0; instance < counts[draw]; ++instance, ++vertexIndex)
    {
      const auto world = transforms[bases[draw] + instance] * (animated ? bone : glm::mat4(1)) * glm::vec4(vertex.position, 1);
      const auto clip = view * world;
      for (UINT component = 0; component < 7; ++component)
      {
        const float expected = component < 3 ? world[component] : clip[component - 3];
        const float value = positions[vertexIndex * 7 + component];
        if (!std::isfinite(value) || std::abs(value - expected) > 0.0001f)
          throw std::runtime_error(std::string(entry) + " draw " + std::to_string(draw) +
            " instance " + std::to_string(instance) + " component " + std::to_string(component) +
            ": expected " + std::to_string(expected) + ", got " + std::to_string(value));
      }
    }
  std::cout << "PASS " << entry << (animated ? " animated" : " static") << '\n';
}

static void CheckPointShadowOrientation()
{
  // Inverse of the TextureCube sampler's face-to-UV mapping, independently
  // specified here so a mirrored or rotated render matrix cannot pass.
  for (const auto light : {glm::vec3(0), glm::vec3(7, -3, 11)})
    for (const float u : {0.07f, 0.31f, 0.82f, 0.96f})
      for (const float v : {0.13f, 0.42f, 0.76f})
      {
        const float s = 2 * u - 1, t = 2 * v - 1;
        const std::array<glm::vec3, 6> rays{
          glm::vec3(1, -t, -s), glm::vec3(-1, -t, s),
          glm::vec3(s, 1, t), glm::vec3(s, -1, -t),
          glm::vec3(s, -t, 1), glm::vec3(-s, -t, -1)};
        for (uint32_t face = 0; face < rays.size(); ++face)
        {
          const auto matrix = DX12Shadow::PointViewProjection(light, face, 0.1f, 20.0f);
          const auto clip = matrix * glm::vec4(light + 5.0f * rays[face], 1);
          const glm::vec3 ndc = glm::vec3(clip) / clip.w;
          const glm::vec2 uv(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f);
          if (clip.w <= 0 || ndc.z <= 0 || ndc.z >= 1 ||
              glm::length(uv - glm::vec2(u, v)) > 0.00001f)
            throw std::runtime_error("Point shadow face " + std::to_string(face) +
              ": expected UV " + std::to_string(u) + "," + std::to_string(v) +
              ", got " + std::to_string(uv.x) + "," + std::to_string(uv.y));
        }
      }
  std::cout << "PASS point shadow orientation (six faces, translated light)\n";
}

int main()
{
  try
  {
    CheckPointShadowOrientation();
    ComPtr<IDXGIFactory4> factory;
    Check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
    ComPtr<IDXGIAdapter> warp;
    Check(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp)));
    ComPtr<ID3D12Device> device;
    Check(D3D12CreateDevice(warp.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));
    for (const auto* entry : {"VSMain", "ShadowVS"})
      for (bool animated : {false, true}) Run(device.Get(), entry, animated);
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
