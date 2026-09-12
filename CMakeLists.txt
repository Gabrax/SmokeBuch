cmake_minimum_required(VERSION 3.16)
set(CMAKE_POLICY_VERSION_MINIMUM 3.16)

project(gabgl LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Do not update already populated dependencies during configure" FORCE)

# Set this to false if you don’t use clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    # PhysX binaries in vendor/ are built against the dynamic MSVC runtime.
    # Keep every source dependency (including GABDEBUG's C target) ABI-compatible.
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

    if(RELEASE)
        add_compile_options(/MP /O2)
    endif()

    if(DEBUG)
        add_compile_options(/MP /Zi /Od) 
        add_link_options(/DEBUG /INCREMENTAL)
    endif()

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wno-unused-function
        -Wno-unused-parameter
        -ferror-limit=0
    )
endif()

option(GABGL_ENABLE_DX12 "Build the optional DirectX 12 backend" ${WIN32})
find_package(Vulkan QUIET)
option(GABGL_ENABLE_VULKAN "Build the Vulkan backend" ${Vulkan_FOUND})

if(GABGL_ENABLE_VULKAN AND NOT Vulkan_FOUND)
  find_package(Vulkan REQUIRED)
endif()

# Slang is the single source-language compiler for the graphics backends.
# The Vulkan SDK ships a complete Slang SDK; standalone Slang installations are
# also supported through CMAKE_PREFIX_PATH or SLANG_ROOT.
find_package(slang CONFIG QUIET)
if(TARGET slang::slang)
  set(GABGL_SLANG_TARGET slang::slang)
else()
  find_path(GABGL_SLANG_INCLUDE_DIR
    NAMES slang.h
    HINTS "$ENV{VULKAN_SDK}/Include/slang" "$ENV{SLANG_ROOT}/include"
    PATH_SUFFIXES slang include/slang include
  )
  find_library(GABGL_SLANG_LIBRARY
    NAMES slang-compiler slang
    HINTS "$ENV{VULKAN_SDK}/Lib" "$ENV{SLANG_ROOT}/lib"
  )
  if(NOT GABGL_SLANG_INCLUDE_DIR OR NOT GABGL_SLANG_LIBRARY)
    message(FATAL_ERROR "Slang SDK not found. Install Slang or a Vulkan SDK containing Slang, then set VULKAN_SDK or SLANG_ROOT.")
  endif()
  add_library(gabgl_slang UNKNOWN IMPORTED)
  set_target_properties(gabgl_slang PROPERTIES
    IMPORTED_LOCATION "${GABGL_SLANG_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GABGL_SLANG_INCLUDE_DIR}"
  )
  set(GABGL_SLANG_TARGET gabgl_slang)
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries only" FORCE)
# Assimp 
FetchContent_Declare(
    assimp
    GIT_REPOSITORY https://github.com/assimp/assimp.git
    GIT_TAG master
)
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "Disable Assimp tests" FORCE)
set(ASSIMP_BUILD_DOCS OFF CACHE BOOL "Disable Assimp documentation" FORCE)

# Meshoptimizer
FetchContent_Declare(
    meshoptimizer
    GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
    GIT_TAG v0.18
)

# ImGuizmo
FetchContent_Declare(
    imguizmo
    GIT_REPOSITORY https://github.com/CedricGuillemet/ImGuizmo
    GIT_TAG master
)

# libsndfile
FetchContent_Declare(
    libsndfile
    GIT_REPOSITORY https://github.com/libsndfile/libsndfile
    GIT_TAG 1.2.2
)

# OpenAL Soft
FetchContent_Declare(
    openal-soft
    GIT_REPOSITORY https://github.com/kcat/openal-soft.git
    GIT_TAG 1.23.1 
)

# GABDEBUG supplies the engine logger and the CPU/OpenGL GPU profiler. Pin the
# exact revision so diagnostics do not change underneath reproducible builds.
FetchContent_Declare(
    gabdebug
    GIT_REPOSITORY https://github.com/Gabrax/GABDEBUG.git
    GIT_TAG 7be0eeb3b945423d872dc944e4f0644fd4a688e8
)
set(GABDEBUG_BUILD_TESTS OFF CACHE BOOL "Disable dependency tests" FORCE)
set(GABDEBUG_GPU_API OPENGL CACHE STRING "Use GABDEBUG's OpenGL profiler adapter" FORCE)

set(LIBTYPE STATIC CACHE STRING "Build static OpenAL Soft" FORCE)
# Freetype
FetchContent_Declare(
    freetype
    GIT_REPOSITORY https://gitlab.freedesktop.org/freetype/freetype.git
    GIT_TAG VER-2-13-2 
)

FetchContent_MakeAvailable(assimp meshoptimizer imguizmo libsndfile openal-soft freetype gabdebug)

# Keep GABDEBUG's short LOG_*/PROFILER_* aliases out of engine headers.
target_compile_definitions(gabdebug PUBLIC
  GABLOG_UNSTRIP_PREFIX=1
  GABPROFILER_UNSTRIP_PREFIX=1
)

add_subdirectory(vendor/glfw-3.3.2)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(vendor/glad)
add_subdirectory(vendor/stb_image)
add_subdirectory(vendor/glm)
add_subdirectory(vendor/imgui-docking)
add_subdirectory(vendor/nlohmann-json)
add_subdirectory(vendor/physx)

# The engine currently does not link the upstream ImGuizmo target (its single
# source integration below is optional). Exclude that target because current
# ImGuizmo master requires a newer ImGui API than the vendored docking branch.
if(TARGET imguizmo)
  set_target_properties(imguizmo PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

file(GLOB IMGUIMO_SRC "${imguizmo_SOURCE_DIR}/ImGuizmo.cpp")
file(GLOB IMGUIMO_HEADERS "${imguizmo_SOURCE_DIR}/ImGuizmo.h")

file(GLOB_RECURSE GABGL_ENGINE_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/Backend/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/Input/*.cpp"
)

# The engine target contains platform, input, rendering, audio, physics and
# generic scene infrastructure. It has no dependency on src/Game.
add_library(gabgl_engine STATIC ${GABGL_ENGINE_SOURCES} ${IMGUIMO_SRC})
set_property(TARGET gabgl_engine PROPERTY CXX_STANDARD 20)
target_include_directories(gabgl_engine
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Backend"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Input"
  PRIVATE
    ${imguizmo_SOURCE_DIR}
)
target_compile_definitions(gabgl_engine PUBLIC GLFW_INCLUDE_NONE=1)
if(MSVC)
  target_compile_definitions(gabgl_engine PUBLIC _CRT_SECURE_NO_WARNINGS)
endif()

# Gameplay state is independently testable and deliberately does not link the
# renderer or platform layer.
add_library(gabgl_gameplay STATIC
  src/Game/DialogueManager.cpp
  src/Game/InventoryManager.cpp
)
target_include_directories(gabgl_gameplay PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/Game")

# The executable is the composition root: it registers game scenes and
# interaction callbacks with the reusable engine library.
add_executable(gabgl_game
  src/Game/main.cpp
  src/Game/GameBootstrap.cpp
)
set_target_properties(gabgl_game PROPERTIES OUTPUT_NAME gl_engine)
target_include_directories(gabgl_game PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/Game")
target_link_libraries(gabgl_game PRIVATE gabgl_engine gabgl_gameplay)

# if(RELEASE)
#   if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
#     set_target_properties("${CMAKE_PROJECT_NAME}" PROPERTIES LINK_FLAGS "-Wl,-subsystem:windows -Wl,-entry:mainCRTStartup")
#   else()
#     set_target_properties("${CMAKE_PROJECT_NAME}" PROPERTIES LINK_FLAGS "/SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
#   endif()
# else()
#     add_compile_definitions(DEBUG)
# endif()

set(IMGUI_INI_FILE "${CMAKE_BINARY_DIR}/imgui.ini")
set(RES_FOLDER "${CMAKE_BINARY_DIR}/../res")
file(MAKE_DIRECTORY ${RES_FOLDER})

target_link_libraries(gabgl_engine PUBLIC
  glm glfw glad stb_image imgui assimp meshoptimizer JSONparser PhysX
  SndFile::sndfile OpenAL::OpenAL freetype gabdebug::gabdebug ${GABGL_SLANG_TARGET}
)

if(GABGL_ENABLE_VULKAN)
  target_compile_definitions(gabgl_engine PRIVATE GABGL_ENABLE_VULKAN=1)
  target_link_libraries(gabgl_engine PRIVATE Vulkan::Vulkan)
endif()

if(WIN32 AND NOT TARGET slang::slang)
  find_file(GABGL_SLANG_DLL
    NAMES slang-compiler.dll slang.dll
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{SLANG_ROOT}/bin"
  )
  if(GABGL_SLANG_DLL)
    add_custom_command(TARGET gabgl_game POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${GABGL_SLANG_DLL}" "$<TARGET_FILE_DIR:gabgl_game>"
    )
  endif()
endif()

if(GABGL_ENABLE_DX12)
  if(NOT WIN32)
    message(FATAL_ERROR "GABGL_ENABLE_DX12 is only supported on Windows")
  endif()

  target_compile_definitions(gabgl_engine PRIVATE GABGL_ENABLE_DX12=1)
  target_link_libraries(gabgl_engine PRIVATE d3d12 dxgi dxguid)
endif()

include(CTest)
if(BUILD_TESTING)
  if(GABGL_ENABLE_DX12)
    add_executable(dx12_instance_tests tests/DX12InstanceTests.cpp)
    target_link_libraries(dx12_instance_tests PRIVATE gabgl_engine d3dcompiler)
    add_test(NAME dx12_instances COMMAND dx12_instance_tests)
    set_tests_properties(dx12_instances PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" TIMEOUT 60)
    if(GABGL_SLANG_DLL)
      foreach(renderer_test IN ITEMS dx12_instance_tests)
        add_custom_command(TARGET ${renderer_test} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GABGL_SLANG_DLL}" "$<TARGET_FILE_DIR:${renderer_test}>"
        )
      endforeach()
    endif()
  endif()
  if(GABGL_ENABLE_DX12 OR GABGL_ENABLE_VULKAN)
    # Explicitly built/run: desktop/GPU integration is not a headless CTest test.
    add_executable(renderer_smoke EXCLUDE_FROM_ALL tests/RendererSmoke.cpp)
    target_link_libraries(renderer_smoke PRIVATE gabgl_engine)
    if(GABGL_SLANG_DLL)
      add_custom_command(TARGET renderer_smoke POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${GABGL_SLANG_DLL}" "$<TARGET_FILE_DIR:renderer_smoke>"
      )
    endif()
  endif()
  if(GABGL_ENABLE_VULKAN)
    add_executable(vulkan_bootstrap_smoke EXCLUDE_FROM_ALL tests/VulkanBootstrapSmoke.cpp)
    target_link_libraries(vulkan_bootstrap_smoke PRIVATE gabgl_engine)
  endif()
  add_executable(inventory_manager_tests
    tests/InventoryManagerTests.cpp
  )
  target_link_libraries(inventory_manager_tests PRIVATE gabgl_gameplay)
  add_test(NAME inventory_manager COMMAND inventory_manager_tests)

  add_executable(dialogue_manager_tests
    tests/DialogueManagerTests.cpp
  )
  target_link_libraries(dialogue_manager_tests PRIVATE gabgl_gameplay)
  add_test(NAME dialogue_manager COMMAND dialogue_manager_tests)
endif()
