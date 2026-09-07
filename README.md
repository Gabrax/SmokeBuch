> [!IMPORTANT]
> All rights to the assets belong to their respective authors.

<div align="center">
  
## Game preview

</div>

<div align="center">

https://github.com/user-attachments/assets/dfeda195-f8d6-4073-aff7-d7bfe55fce39

</div>

<div align="center">
  
## Dependencies

</div>

<div align="center">

<p>
<a href="https://github.com/glfw/glfw">glfw</a> •
<a href="https://github.com/Dav1dde/glad">glad</a> •
<a href="https://github.com/g-truc/glm">glm</a> •
<a href="https://github.com/ocornut/imgui">imgui</a> •
<a href="https://github.com/CedricGuillemet/ImGuizmo">ImGuizmo</a> •
<a href="https://github.com/libsndfile/libsndfile">libsndfile</a> •
<a href="https://github.com/kcat/openal-soft">OpenAL-Soft</a> •
<a href="https://github.com/nothings/stb/blob/master/stb_image.h">stb_image</a> •
<a href="https://github.com/freetype/freetype">freetype</a> •
<a href="https://github.com/assimp/assimp">assimp</a> •
<a href="https://github.com/zeux/meshoptimizer">meshoptimizer</a> •
<a href="https://github.com/shader-slang/slang">Slang</a> •
<a href="https://github.com/nlohmann/json">nlohmann_json</a> •
<a href="https://github.com/syoyo/tinyexr">tinyEXR</a> •
<a href="https://github.com/NVIDIA-Omniverse/PhysX">PhysX</a>
• <a href="https://github.com/Gabrax/GABDEBUG">GABDEBUG</a>
</p>

</div>

On Windows, the optional DirectX 12
path can be enabled at configure time and selected without changing the saved
configuration:

```powershell
cmake -S . -B build -DGABGL_ENABLE_DX12=ON
cmake --build build --config Release
build\gl_engine.exe --renderer=dx12
```

Alternatively, set `graphics.api` in `gab.ini` to `"dx12"`. Use
`--renderer=opengl` to override that setting for a single run.

Both renderers are selected through the same backend contract. Models, particles,
screen UI, ImGui, debug layers, culling statistics and visual-effect settings are
submitted without backend-specific branches in scene code, leaving future APIs a
single interface to implement.

An opt-in desktop integration test exercises the real scene, DX12 effect quality
changes, PS1 on/off, particles, editor render targets and odd-size resizing. It also
supports an OpenGL regression run. Build with `BUILD_TESTING=ON` and
`GABGL_ENABLE_DX12=ON`, then run from the repository root:

```powershell
cmake --build cmake-build-release --target renderer_smoke
./tests/RunRendererSmoke.ps1
```

Pass `-BuildDirectory` for another build directory. The script checks process exit
codes and renderer error logs. `GABGL_DX12_VALIDATION=1` enables D3D12 validation in
Release builds when the Windows debug layer is installed. These integration tests require a
desktop and scene assets and are intentionally excluded from automatic CTest runs.

The `dx12_instances` CTest regression runs headlessly on D3D12 WARP. It reads back
positions from the actual scene and shadow vertex shaders through `ExecuteIndirect`,
covering nonzero instance offsets, translation, rotation, nonuniform scale and skinning.
DX12 transform buffers store explicit GLM columns; each draw passes its compacted
instance base in the constant buffer because `SV_InstanceID` starts at zero.
The same test checks all six point-shadow face projections against cubemap UV
coordinates. DX12 uses explicit left-handed views and a [0,1] depth projection
for those faces so rendering and `TextureCube` sampling have the same orientation.
