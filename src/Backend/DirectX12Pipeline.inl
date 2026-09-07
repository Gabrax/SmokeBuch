// Internal DX12 pipeline stages. Included inside the renderer's anonymous
// namespace so device ownership and resource transitions stay in one backend.

struct PointShadowCasterState
{
  uint64_t Hash = 1469598103934665603ull;
  bool Animated = false;
};

PointShadowCasterState GetPointShadowCasterState(const glm::vec3& lightPosition)
{
  PointShadowCasterState state;
  auto hash = [&](const void* data, size_t size)
  {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) state.Hash = (state.Hash ^ bytes[i]) * 1099511628211ull;
  };
  for (const auto& name : ModelManager::GetModelNames())
  {
    const auto model = ModelManager::GetModel(name);
    if (!model || !model->m_IsRendered || model->GetPhysXMeshType() == MeshType::CONVEXMESH) continue;
    bool added = false;
    for (const auto& transform : model->m_InstanceTransforms)
    {
      const auto sphere = CalculateWorldBoundingSphere(*model, transform);
      if (glm::distance(sphere.center, lightPosition) > PointShadowRadius + sphere.radius) continue;
      if (!added)
      {
        hash(name.data(), name.size());
        const float radius = model->GetBoundsRadius();
        hash(&radius, sizeof(radius));
        added = true;
      }
      hash(&transform, sizeof(transform));
      state.Animated |= model->IsAnimated();
    }
  }
  return state;
}

ComPtr<ID3D12Resource> CreateGPUBuffer(uint64_t bytes)
{
  auto desc = BufferDescription(std::max(bytes, 256ull));
  desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  const auto heap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
  ComPtr<ID3D12Resource> resource;
  CheckHRESULT(s_Data.Device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
    &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&resource)), "Create DX12 GPU buffer");
  return resource;
}

D3D12_ROOT_PARAMETER RootBuffer(D3D12_ROOT_PARAMETER_TYPE type, uint32_t slot)
{
  D3D12_ROOT_PARAMETER parameter{};
  parameter.ParameterType = type;
  parameter.Descriptor.ShaderRegister = slot;
  return parameter;
}

D3D12_ROOT_PARAMETER RootTable(D3D12_DESCRIPTOR_RANGE& range,
                               D3D12_DESCRIPTOR_RANGE_TYPE type, uint32_t slot)
{
  range = {};
  range.RangeType = type;
  range.NumDescriptors = 1;
  range.BaseShaderRegister = slot;
  D3D12_ROOT_PARAMETER parameter{};
  parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  parameter.DescriptorTable = {1, &range};
  return parameter;
}

void CreateComputePipeline(const char* shader, const D3D12_ROOT_PARAMETER* parameters,
                           uint32_t count, ComPtr<ID3D12RootSignature>& root,
                           ComPtr<ID3D12PipelineState>& pipeline)
{
  D3D12_ROOT_SIGNATURE_DESC desc{};
  desc.NumParameters = count;
  desc.pParameters = parameters;
  root = CreateRootSignature(desc);
  const auto bytecode = CompileSlang(shader, "CSMain", "cs_5_1");
  D3D12_COMPUTE_PIPELINE_STATE_DESC state{};
  state.pRootSignature = root.Get();
  state.CS = {bytecode.GetBufferPointer(), bytecode.GetBufferSize()};
  CheckHRESULT(s_Data.Device->CreateComputePipelineState(&state, IID_PPV_ARGS(&pipeline)),
    "Create DX12 compute pipeline");
}

void CreateComputePipelines()
{
  D3D12_DESCRIPTOR_RANGE depthRange{}, sourceRange{}, outputRange{}, positionRange{};
  const std::array cullParameters{
    RootBuffer(D3D12_ROOT_PARAMETER_TYPE_CBV, 0),
    RootBuffer(D3D12_ROOT_PARAMETER_TYPE_SRV, 0),
    RootBuffer(D3D12_ROOT_PARAMETER_TYPE_SRV, 1),
    RootBuffer(D3D12_ROOT_PARAMETER_TYPE_SRV, 2),
    RootTable(depthRange, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3),
    RootBuffer(D3D12_ROOT_PARAMETER_TYPE_UAV, 0),
    RootBuffer(D3D12_ROOT_PARAMETER_TYPE_UAV, 1),
    RootBuffer(D3D12_ROOT_PARAMETER_TYPE_UAV, 2)
  };
  CreateComputePipeline("../res/shaders/gpu_cull.slang", cullParameters.data(),
    uint32_t(cullParameters.size()), s_Data.CullRootSignature, s_Data.CullPipeline);
  auto sizes = RootBuffer(D3D12_ROOT_PARAMETER_TYPE_CBV, 0);
  sizes.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  sizes.Constants = {0, 0, 4};
  const std::array hizParameters{sizes,
    RootTable(sourceRange, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0),
    RootTable(outputRange, D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0)};
  CreateComputePipeline("../res/shaders/hiz_build.slang", hizParameters.data(),
    uint32_t(hizParameters.size()), s_Data.HiZRootSignature, s_Data.HiZPipeline);
  const std::array tileParameters{
    RootBuffer(D3D12_ROOT_PARAMETER_TYPE_CBV, 0),
    RootTable(positionRange, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0),
    RootBuffer(D3D12_ROOT_PARAMETER_TYPE_UAV, 0),
    RootBuffer(D3D12_ROOT_PARAMETER_TYPE_UAV, 1)};
  CreateComputePipeline("../res/shaders/tiled_light_cull.slang", tileParameters.data(),
    uint32_t(tileParameters.size()), s_Data.TileRootSignature, s_Data.TilePipeline);
}

void CreateGPUResources(uint32_t width, uint32_t height)
{
  if (!s_Data.DepthDescriptor)
  {
    s_Data.DepthDescriptor = AllocateDescriptor();
    s_Data.HiZ.DescriptorIndex = AllocateDescriptor();
    s_Data.HiZSRVBase = AllocateDescriptor();
    for (uint32_t i = 1; i < MaxHiZMips; ++i) AllocateDescriptor();
    s_Data.HiZUAVBase = AllocateDescriptor();
    for (uint32_t i = 1; i < MaxHiZMips; ++i) AllocateDescriptor();
  }
  D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
  srv.Format = DXGI_FORMAT_R32_FLOAT;
  srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  srv.Texture2D.MipLevels = 1;
  s_Data.Device->CreateShaderResourceView(s_Data.DepthBuffer.Get(), &srv,
    GetSRVCPUHandle(s_Data.DepthDescriptor));
  s_Data.HiZMipCount = std::min(MaxHiZMips, 1u + uint32_t(std::floor(std::log2(std::max(width, height)))));
  D3D12_RESOURCE_DESC desc{};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  desc.Width = width;
  desc.Height = height;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = static_cast<UINT16>(s_Data.HiZMipCount);
  desc.Format = DXGI_FORMAT_R32_FLOAT;
  desc.SampleDesc.Count = 1;
  desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  const auto heap = HeapProperties(D3D12_HEAP_TYPE_DEFAULT);
  s_Data.HiZ.Resource.Reset();
  CheckHRESULT(s_Data.Device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
    &desc, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr,
    IID_PPV_ARGS(&s_Data.HiZ.Resource)), "Create DX12 Hi-Z pyramid");
  srv.Texture2D.MipLevels = s_Data.HiZMipCount;
  s_Data.Device->CreateShaderResourceView(s_Data.HiZ.Resource.Get(), &srv,
    GetSRVCPUHandle(s_Data.HiZ.DescriptorIndex));
  for (uint32_t mip = 0; mip < s_Data.HiZMipCount; ++mip)
  {
    srv.Texture2D.MostDetailedMip = mip;
    srv.Texture2D.MipLevels = 1;
    s_Data.Device->CreateShaderResourceView(s_Data.HiZ.Resource.Get(), &srv,
      GetSRVCPUHandle(s_Data.HiZSRVBase + mip));
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R32_FLOAT;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    uav.Texture2D.MipSlice = mip;
    s_Data.Device->CreateUnorderedAccessView(s_Data.HiZ.Resource.Get(), nullptr,
      &uav, GetSRVCPUHandle(s_Data.HiZUAVBase + mip));
  }
  s_Data.TileCountX = (width + 15) / 16;
  s_Data.TileCountY = (height + 15) / 16;
  const uint64_t tiles = uint64_t(s_Data.TileCountX) * s_Data.TileCountY;
  s_Data.TileGrid = CreateGPUBuffer(tiles * sizeof(glm::uvec2));
  s_Data.TileIndices = CreateGPUBuffer(tiles * MaxSceneLights * sizeof(uint32_t));
}

D3D12_GPU_VIRTUAL_ADDRESS UploadArray(const void* data, uint64_t bytes)
{
  auto& frame = s_Data.FrameUploads[s_Data.FrameIndex];
  const uint64_t offset = AlignUp(frame.ConstantsOffset, 256);
  if (offset + bytes > ConstantBufferBytes)
    throw std::runtime_error("DX12 scene upload buffer exhausted");
  if (bytes) std::memcpy(frame.ConstantsMapped + offset, data, bytes);
  frame.ConstantsOffset = offset + bytes;
  return frame.Constants->GetGPUVirtualAddress() + offset;
}

void PrepareGPUScene(const RenderEffectSettings& effects)
{
  auto& frame = s_Data.GPUFrames[s_Data.FrameIndex];
  if (frame.StatisticsReady)
  {
    const D3D12_RANGE readRange{0, sizeof(uint32_t)};
    void* mapped = nullptr;
    CheckHRESULT(frame.StatisticsReadback->Map(0, &readRange, &mapped), "Read DX12 culling statistics");
    RenderBackend::SetStatistics({*static_cast<const uint32_t*>(mapped), frame.RenderableInstances});
    const D3D12_RANGE noWrite{0, 0};
    frame.StatisticsReadback->Unmap(0, &noWrite);
  }
  frame.Culled = false;
  std::vector<IndirectDraw> commands;
  std::vector<DrawCullData> cullData;
  std::vector<glm::mat4> transforms;
  uint32_t outputCount = 0;
  RenderStatistics statistics;
  SceneConstants constants = BuildSceneConstants(effects);
  auto addModel = [&](const std::shared_ptr<Model>& model,
                      const std::vector<glm::mat4>& instances, bool preview, float brightness)
  {
    if (instances.empty()) return;
    const uint32_t sourceStart = uint32_t(transforms.size());
    transforms.insert(transforms.end(), instances.begin(), instances.end());
    const bool animated = model->IsAnimated() && !preview;
    constants.CameraPosition.w = animated ? 1.0f : 0.0f;
    constants.Bones.fill(glm::mat4(1.0f));
    if (animated)
    {
      const auto& bones = model->GetFinalBoneMatrices();
      std::copy_n(bones.begin(), std::min(bones.size(), constants.Bones.size()), constants.Bones.begin());
    }
    if (!preview)
    {
      statistics.RenderableInstances += uint32_t(instances.size());
    }
    bool firstMesh = !preview;
    for (const auto& mesh : model->GetMeshes())
    {
      const auto it = s_Data.Meshes.find(&mesh);
      if (it == s_Data.Meshes.end()) continue;
      const auto& gpu = it->second;
      constants.MaterialFlags = glm::vec4(gpu.HasNormalMap, gpu.HasSpecularMap, brightness, 0);
      // SV_InstanceID starts at zero for each draw. Pass the compacted buffer's
      // base explicitly; StartInstanceLocation only offsets IA instance data.
      constants.MaterialIndices = glm::uvec4(gpu.DiffuseDescriptorIndex, gpu.NormalDescriptorIndex, gpu.SpecularDescriptorIndex, outputCount);
      IndirectDraw command{};
      command.Constants = UploadSceneConstants(constants);
      command.Vertex = gpu.VertexView;
      command.Index = gpu.IndexView;
      command.Draw = {gpu.IndexCount, uint32_t(instances.size()), 0, 0, 0};
      commands.push_back(command);
      cullData.push_back({glm::vec4(model->GetBoundsCenter(), std::max(model->GetBoundsRadius(), 0.001f)),
        glm::uvec4(sourceStart, uint32_t(instances.size()), outputCount,
          ((!animated && !preview && model->GetPhysXMeshType() != MeshType::CONTROLLER) ? 1u : 0u) |
          (firstMesh ? 2u : 0u) | (preview ? 4u : 0u))});
      firstMesh = false;
      outputCount += uint32_t(instances.size());
    }
  };
  for (const auto& name : ModelManager::GetModelNames())
  {
    const auto model = ModelManager::GetModel(name);
    if (model && model->m_IsRendered && model->GetPhysXMeshType() != MeshType::CONVEXMESH)
      addModel(model, model->m_InstanceTransforms, false, 1.0f);
  }
  for (const auto& preview : s_Data.ModelPreviews)
  {
    const auto model = ModelManager::GetModel(preview.ModelName);
    if (model && model->GetPhysXMeshType() != MeshType::CONVEXMESH)
      addModel(model, {preview.Transform}, true, std::max(preview.Brightness, 1.0f));
  }
  frame.RenderableInstances = statistics.RenderableInstances;
  frame.DrawCount = uint32_t(commands.size());
  frame.SourceCommands = UploadArray(commands.data(), commands.size() * sizeof(IndirectDraw));
  frame.SourceTransforms = UploadArray(transforms.data(), transforms.size() * sizeof(glm::mat4));
  frame.CullData = UploadArray(cullData.data(), cullData.size() * sizeof(DrawCullData));
  const uint64_t commandBytes = std::max(uint64_t(commands.size() * sizeof(IndirectDraw)), uint64_t(256));
  const uint64_t transformBytes = std::max(uint64_t(outputCount) * sizeof(glm::mat4), uint64_t(256));
  // Only the current frame's resources can grow; BeginFrame has waited for its fence.
  if (commandBytes > frame.CommandBytes)
  {
    frame.CommandBytes = commandBytes * 2;
    frame.Commands = CreateGPUBuffer(frame.CommandBytes);
  }
  if (transformBytes > frame.TransformBytes)
  {
    frame.TransformBytes = transformBytes * 2;
    frame.Transforms = CreateGPUBuffer(frame.TransformBytes);
  }
  if (!frame.Count) frame.Count = CreateGPUBuffer(256);
  if (!frame.StatisticsReadback)
  {
    const auto heap = HeapProperties(D3D12_HEAP_TYPE_READBACK);
    const auto desc = BufferDescription(256);
    CheckHRESULT(s_Data.Device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
      &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&frame.StatisticsReadback)),
      "Create DX12 culling readback");
  }
}

void DispatchGPUCull(bool useHiZ, const glm::mat4& viewProjection,
                     const RenderEffectSettings& effects = {}, bool shadowPass = false)
{
  auto& frame = s_Data.GPUFrames[s_Data.FrameIndex];
  TransitionResource(frame.Count.Get(), frame.Culled ? D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT : D3D12_RESOURCE_STATE_COMMON,
    D3D12_RESOURCE_STATE_COPY_DEST);
  const glm::uvec2 zero(0);
  auto& upload = s_Data.FrameUploads[s_Data.FrameIndex];
  const auto zeroAddress = UploadArray(&zero, sizeof(zero));
  s_Data.CommandList->CopyBufferRegion(frame.Count.Get(), 0, upload.Constants.Get(),
    zeroAddress - upload.Constants->GetGPUVirtualAddress(), sizeof(zero));
  TransitionResource(frame.Count.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  TransitionResource(frame.Commands.Get(), frame.Culled ? D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT : D3D12_RESOURCE_STATE_COMMON,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  TransitionResource(frame.Transforms.Get(), frame.Culled ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE : D3D12_RESOURCE_STATE_COMMON,
    D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  struct CullConstants { glm::mat4 ViewProjection; glm::uvec4 Parameters; glm::vec4 SnapMargin; };
  const float virtualHeight = std::max(effects.PS1VirtualHeight, 1.0f);
  const float virtualWidth = std::max(std::round(virtualHeight * s_Data.Width / s_Data.Height), 1.0f);
  const CullConstants constants{viewProjection, glm::uvec4(frame.DrawCount, shadowPass ? 2u : uint32_t(useHiZ), s_Data.Width, s_Data.Height),
    effects.PS1Enabled ? glm::vec4(1.0f / virtualWidth, 1.0f / virtualHeight, 0, 0) : glm::vec4(0)};
  ID3D12DescriptorHeap* heaps[] = {s_Data.SRVHeap.Get()};
  s_Data.CommandList->SetDescriptorHeaps(1, heaps);
  s_Data.CommandList->SetComputeRootSignature(s_Data.CullRootSignature.Get());
  s_Data.CommandList->SetPipelineState(s_Data.CullPipeline.Get());
  s_Data.CommandList->SetComputeRootConstantBufferView(0, UploadConstants(constants));
  s_Data.CommandList->SetComputeRootShaderResourceView(1, frame.SourceCommands);
  s_Data.CommandList->SetComputeRootShaderResourceView(2, frame.SourceTransforms);
  s_Data.CommandList->SetComputeRootShaderResourceView(3, frame.CullData);
  s_Data.CommandList->SetComputeRootDescriptorTable(4, GetSRVGPUHandle(s_Data.HiZ.DescriptorIndex));
  s_Data.CommandList->SetComputeRootUnorderedAccessView(5, frame.Commands->GetGPUVirtualAddress());
  s_Data.CommandList->SetComputeRootUnorderedAccessView(6, frame.Transforms->GetGPUVirtualAddress());
  s_Data.CommandList->SetComputeRootUnorderedAccessView(7, frame.Count->GetGPUVirtualAddress());
  if (frame.DrawCount) s_Data.CommandList->Dispatch((frame.DrawCount + 63) / 64, 1, 1);
  TransitionResource(frame.Count.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
  TransitionResource(frame.Commands.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
  TransitionResource(frame.Transforms.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  frame.Culled = true;
  if (useHiZ)
  {
    TransitionResource(frame.Count.Get(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    s_Data.CommandList->CopyBufferRegion(frame.StatisticsReadback.Get(), 0, frame.Count.Get(), 4, 4);
    TransitionResource(frame.Count.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    frame.StatisticsReady = true;
  }
}

void DrawGPUScene(ID3D12PipelineState* pipeline, D3D12_GPU_VIRTUAL_ADDRESS shadowConstants = 0)
{
  const auto& frame = s_Data.GPUFrames[s_Data.FrameIndex];
  if (!frame.DrawCount) return;
  s_Data.CommandList->SetGraphicsRootSignature(s_Data.SceneRootSignature.Get());
  s_Data.CommandList->SetPipelineState(pipeline);
  s_Data.CommandList->SetGraphicsRootShaderResourceView(6, frame.Transforms->GetGPUVirtualAddress());
  s_Data.CommandList->SetGraphicsRootDescriptorTable(7, GetSRVGPUHandle(0));
  if (shadowConstants) s_Data.CommandList->SetGraphicsRootConstantBufferView(10, shadowConstants);
  s_Data.CommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  s_Data.CommandList->ExecuteIndirect(s_Data.DrawSignature.Get(), frame.DrawCount,
    frame.Commands.Get(), 0, frame.Count.Get(), 0);
}

void DrawDepthPrepass()
{
  const auto dsv = GetDSVHandle(0);
  s_Data.CommandList->OMSetRenderTargets(0, nullptr, FALSE, &dsv);
  s_Data.CommandList->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
  s_Data.CommandList->RSSetViewports(1, &s_Data.Viewport);
  s_Data.CommandList->RSSetScissorRects(1, &s_Data.ScissorRect);
  DrawGPUScene(s_Data.DepthPrepassPipeline.Get());
}

void BuildHiZ()
{
  TransitionResource(s_Data.DepthBuffer.Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  s_Data.CommandList->SetComputeRootSignature(s_Data.HiZRootSignature.Get());
  s_Data.CommandList->SetPipelineState(s_Data.HiZPipeline.Get());
  for (uint32_t mip = 0; mip < s_Data.HiZMipCount; ++mip)
  {
    const uint32_t width = std::max(1u, s_Data.Width >> mip);
    const uint32_t height = std::max(1u, s_Data.Height >> mip);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {s_Data.HiZ.Resource.Get(), mip,
      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
    s_Data.CommandList->ResourceBarrier(1, &barrier);
    const glm::uvec4 parameters(width, height, mip == 0, 0);
    s_Data.CommandList->SetComputeRoot32BitConstants(0, 4, &parameters, 0);
    s_Data.CommandList->SetComputeRootDescriptorTable(1, GetSRVGPUHandle(
      mip == 0 ? s_Data.DepthDescriptor : s_Data.HiZSRVBase + mip - 1));
    s_Data.CommandList->SetComputeRootDescriptorTable(2, GetSRVGPUHandle(s_Data.HiZUAVBase + mip));
    s_Data.CommandList->Dispatch((width + 7) / 8, (height + 7) / 8, 1);
    std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
    s_Data.CommandList->ResourceBarrier(1, &barrier);
  }
  TransitionResource(s_Data.DepthBuffer.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_DEPTH_WRITE);
}

void DispatchTileLights(const RenderEffectSettings& effects)
{
  TransitionResource(s_Data.GPosition.Resource.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  // Buffer states decay to COMMON after the preceding frame's command list.
  TransitionResource(s_Data.TileGrid.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  TransitionResource(s_Data.TileIndices.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  s_Data.CommandList->SetComputeRootSignature(s_Data.TileRootSignature.Get());
  s_Data.CommandList->SetPipelineState(s_Data.TilePipeline.Get());
  s_Data.CommandList->SetComputeRootConstantBufferView(0, UploadSceneConstants(BuildSceneConstants(effects)));
  s_Data.CommandList->SetComputeRootDescriptorTable(1, GetSRVGPUHandle(s_Data.GPosition.DescriptorIndex));
  s_Data.CommandList->SetComputeRootUnorderedAccessView(2, s_Data.TileGrid->GetGPUVirtualAddress());
  s_Data.CommandList->SetComputeRootUnorderedAccessView(3, s_Data.TileIndices->GetGPUVirtualAddress());
  s_Data.CommandList->Dispatch(s_Data.TileCountX, s_Data.TileCountY, 1);
  TransitionResource(s_Data.GPosition.Resource.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
  TransitionResource(s_Data.TileGrid.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
  TransitionResource(s_Data.TileIndices.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
}
