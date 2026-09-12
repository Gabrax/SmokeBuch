#include "VulkanRenderer.h"

#if defined(GABGL_ENABLE_VULKAN)

#include <vulkan/vulkan.h>

#include "AudioManager.h"
#include "Camera.h"
#include "DeltaTime.hpp"
#include "FontManager.h"
#include "ModelManager.h"
#include "ParticleRenderer.h"
#include "PhysX.h"
#include "RenderBackend.h"
#include "Shader.h"
#include "Window.h"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>

#include <gabdebug.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
  constexpr uint32_t FramesInFlight = 2;
  constexpr uint32_t MaxSceneLights = 32;
  constexpr uint32_t MaxPointShadowLights = 4;
  constexpr float PointShadowRadius = 20.0f;
  constexpr uint32_t MaxParticleVertices = 4096;
  constexpr uint32_t MaxDebugLineVertices = 131072;
  constexpr uint32_t MaxBloomMips = 6;
  constexpr uint32_t TileSize = 16;
  constexpr uint32_t MaxTileLights = 32;
  constexpr uint32_t MaxHiZMips = 16;
  constexpr VkFormat SceneColorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
  constexpr std::array<const char*, 1> DeviceExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  struct QueueFamilies
  {
    std::optional<uint32_t> Graphics;
    std::optional<uint32_t> Present;
    [[nodiscard]] bool Complete() const { return Graphics.has_value() && Present.has_value(); }
  };

  struct GPUBuffer
  {
    VkBuffer Buffer = VK_NULL_HANDLE;
    VkDeviceMemory Memory = VK_NULL_HANDLE;
  };

  struct GPUTexture
  {
    VkImage Image = VK_NULL_HANDLE;
    VkDeviceMemory Memory = VK_NULL_HANDLE;
    VkImageView View = VK_NULL_HANDLE;
  };

  struct GPUFontAtlas
  {
    GPUTexture Texture;
    VkDescriptorSet Descriptor = VK_NULL_HANDLE;
  };

  struct GPUMesh
  {
    GPUBuffer Vertices;
    GPUBuffer Indices;
    std::array<VkDescriptorSet, FramesInFlight> Materials{};
    uint32_t IndexCount = 0;
  };

  struct GPUModel
  {
    std::array<GPUBuffer, FramesInFlight> Bones{};
  };

  struct ScenePushConstants
  {
    glm::mat4 ModelViewProjection;
    glm::vec4 DrawParams{1.0f, 0.0f, 0.0f, 0.0f}; // brightness, transform offset
  };
  static_assert(sizeof(ScenePushConstants) <= 128);

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

  struct alignas(16) CullInput
  {
    glm::vec4 LocalSphere{0.0f};
    glm::uvec4 Metadata{0u}; // source offset/count, output offset, unused
  };

  struct CullPushConstants
  {
    glm::mat4 ViewProjection{1.0f};
    glm::uvec4 Params{0u};
  };

  struct GPUCullFrame
  {
    GPUBuffer SourceTransforms;
    GPUBuffer Inputs;
    GPUBuffer VisibleTransforms;
    GPUBuffer Commands;
  };

  struct CullDrawRecord
  {
    GPUMesh* Mesh = nullptr;
    VkDescriptorSet Descriptor = VK_NULL_HANDLE;
    uint32_t CommandIndex = 0;
    uint32_t TransformOffset = 0;
    float Brightness = 1.0f;
  };

  struct HiZPyramid
  {
    GPUTexture Texture;
    std::array<VkImageView, MaxHiZMips> MipViews{};
    uint32_t MipCount = 0;
    bool Valid = false;
  };

  struct DepthAttachment
  {
    VkImage Image = VK_NULL_HANDLE;
    VkDeviceMemory Memory = VK_NULL_HANDLE;
    VkImageView View = VK_NULL_HANDLE;
  };

  struct ShadowAttachment
  {
    GPUTexture Texture;
    VkFramebuffer Framebuffer = VK_NULL_HANDLE;
  };

  struct PointShadowCube
  {
    GPUTexture Texture;
    std::array<VkImageView, 6> FaceViews{};
    std::array<VkFramebuffer, 6> Framebuffers{};
  };

  struct PointShadowFrame
  {
    std::array<PointShadowCube, MaxPointShadowLights> Cubes{};
  };

  struct PointShadowCacheEntry
  {
    uint64_t CasterHash = 0;
    glm::vec3 LightPosition{0.0f};
    int32_t LightIndex = -1;
    bool Valid = false;
  };

  struct PointShadowCasterState
  {
    uint64_t Hash = 1469598103934665603ull;
    bool Animated = false;
  };

  struct alignas(16) SceneFrameData
  {
    glm::mat4 LightViewProjection{1.0f};
    glm::mat4 InverseViewProjection{1.0f};
    glm::vec4 CameraPosition{0.0f};
    std::array<glm::vec4, MaxSceneLights> LightPositions{};
    std::array<glm::vec4, MaxSceneLights> LightDirections{};
    std::array<glm::vec4, MaxSceneLights> LightColors{};
    glm::vec4 Params{0.0f};
    glm::vec4 Effects{1.0f};
    glm::vec4 ShadowInfo{0.0f};
    glm::vec4 PointShadowIndices{-1.0f};
    glm::vec4 TileInfo{0.0f};
    glm::vec4 ViewportInfo{1.0f};
  };

  struct PostPushConstants
  {
    glm::vec4 Params{0.0f};
    glm::vec4 Effects{0.0f};
    glm::vec4 Resolution{1.0f};
  };

  struct VulkanData
  {
    VkInstance Instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT DebugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR Surface = VK_NULL_HANDLE;
    VkPhysicalDevice PhysicalDevice = VK_NULL_HANDLE;
    VkDevice Device = VK_NULL_HANDLE;
    VkQueue GraphicsQueue = VK_NULL_HANDLE;
    VkQueue PresentQueue = VK_NULL_HANDLE;
    uint32_t GraphicsQueueFamily = 0;
    uint32_t PresentQueueFamily = 0;

    VkSwapchainKHR Swapchain = VK_NULL_HANDLE;
    VkFormat SwapchainFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D Extent{};
    std::vector<VkImage> Images;
    std::vector<VkImageView> ImageViews;
    std::vector<VkFramebuffer> Framebuffers;
    std::vector<VkFramebuffer> DepthPrepassFramebuffers;
    std::vector<VkFramebuffer> LightingFramebuffers;
    std::vector<VkFramebuffer> ParticleFramebuffers;
    std::vector<VkFramebuffer> PresentFramebuffers;
    std::vector<VkFramebuffer> EditorFramebuffers;
    std::vector<GPUTexture> SceneColors;
    std::vector<GPUTexture> GBufferAlbedo;
    std::vector<GPUTexture> GBufferNormal;
    std::vector<GPUTexture> GBufferPosition;
    std::vector<GPUTexture> BloomPyramids;
    std::vector<GPUTexture> EditorColors;
    std::vector<VkDescriptorSet> EditorDescriptors;
    uint32_t BloomMipCount = 1;
    std::vector<GPUBuffer> TileGrids;
    std::vector<GPUBuffer> TileIndices;
    std::vector<DepthAttachment> DepthAttachments;
    std::vector<VkFence> ImagesInFlight;
    std::vector<VkSemaphore> RenderComplete;
    uint32_t MinImageCount = 2;

    VkRenderPass RenderPass = VK_NULL_HANDLE;
    VkRenderPass DepthPrepassRenderPass = VK_NULL_HANDLE;
    VkRenderPass LightingRenderPass = VK_NULL_HANDLE;
    VkRenderPass ParticleRenderPass = VK_NULL_HANDLE;
    VkRenderPass PresentRenderPass = VK_NULL_HANDLE;
    VkRenderPass EditorRenderPass = VK_NULL_HANDLE;
    VkRenderPass ShadowRenderPass = VK_NULL_HANDLE;
    VkDescriptorSetLayout SceneDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool SceneDescriptorPool = VK_NULL_HANDLE;
    VkSampler MaterialSampler = VK_NULL_HANDLE;
    VkSampler ShadowSampler = VK_NULL_HANDLE;
    GPUTexture WhiteTexture;
    GPUTexture NeutralNormalTexture;
    GPUTexture BlackTexture;
    VkPipelineLayout ScenePipelineLayout = VK_NULL_HANDLE;
    VkPipeline ScenePipeline = VK_NULL_HANDLE;
    VkPipelineLayout DepthPrepassPipelineLayout = VK_NULL_HANDLE;
    VkPipeline DepthPrepassPipeline = VK_NULL_HANDLE;
    VkPipelineLayout ShadowPipelineLayout = VK_NULL_HANDLE;
    VkPipeline ShadowPipeline = VK_NULL_HANDLE;
    VkRenderPass PointShadowRenderPass = VK_NULL_HANDLE;
    VkPipelineLayout PointShadowPipelineLayout = VK_NULL_HANDLE;
    VkPipeline PointShadowPipeline = VK_NULL_HANDLE;
    std::array<PointShadowFrame, FramesInFlight> PointShadows{};
    std::array<std::array<PointShadowCacheEntry, MaxPointShadowLights>,
               FramesInFlight> PointShadowCache{};
    uint32_t PointShadowSize = 0;
    std::array<int32_t, MaxPointShadowLights> ActivePointShadowLights{{-1, -1, -1, -1}};
    VkPipelineLayout ParticlePipelineLayout = VK_NULL_HANDLE;
    VkPipeline ParticlePipeline = VK_NULL_HANDLE;
    std::array<GPUBuffer, FramesInFlight> ParticleBuffers{};
    VkPipelineLayout DebugLinePipelineLayout = VK_NULL_HANDLE;
    VkPipeline DebugLinePipeline = VK_NULL_HANDLE;
    std::array<GPUBuffer, FramesInFlight> DebugLineBuffers{};
    std::vector<DebugLineVertex> PendingDebugLines;
    VkDescriptorSetLayout CullDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool CullDescriptorPool = VK_NULL_HANDLE;
    VkPipelineLayout CullPipelineLayout = VK_NULL_HANDLE;
    VkPipeline CullPipeline = VK_NULL_HANDLE;
    std::array<GPUCullFrame, FramesInFlight> CullFrames{};
    std::array<VkDescriptorSet, FramesInFlight> CullDescriptors{};
    size_t CullTransformCapacity = 0;
    size_t CullDrawCapacity = 0;
    std::vector<CullDrawRecord> CullDrawRecords;
    std::array<HiZPyramid, FramesInFlight> HiZ{};
    VkDescriptorSetLayout HiZDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool HiZDescriptorPool = VK_NULL_HANDLE;
    VkPipelineLayout HiZPipelineLayout = VK_NULL_HANDLE;
    VkPipeline HiZPipeline = VK_NULL_HANDLE;
    std::array<std::array<VkDescriptorSet, MaxHiZMips>, FramesInFlight> HiZDescriptors{};
    VkDescriptorSetLayout PostDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool PostDescriptorPool = VK_NULL_HANDLE;
    VkSampler PostSampler = VK_NULL_HANDLE;
    VkPipelineLayout PostPipelineLayout = VK_NULL_HANDLE;
    VkPipeline PostPipeline = VK_NULL_HANDLE;
    GPUBuffer PostVertices;
    std::vector<VkDescriptorSet> PostDescriptors;
    VkDescriptorSetLayout LightingDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool LightingDescriptorPool = VK_NULL_HANDLE;
    VkPipelineLayout LightingPipelineLayout = VK_NULL_HANDLE;
    VkPipeline LightingPipeline = VK_NULL_HANDLE;
    std::vector<std::array<VkDescriptorSet, FramesInFlight>> LightingDescriptors;
    VkDescriptorSetLayout TileDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool TileDescriptorPool = VK_NULL_HANDLE;
    VkPipelineLayout TilePipelineLayout = VK_NULL_HANDLE;
    VkPipeline TilePipeline = VK_NULL_HANDLE;
    std::vector<std::array<VkDescriptorSet, FramesInFlight>> TileDescriptors;
    GPUTexture SkyboxTexture;
    bool HasSkybox = false;
    RenderEffectSettings CurrentEffects{};
    glm::vec4 ClearColor{0.008f, 0.012f, 0.025f, 1.0f};
    std::array<ShadowAttachment, FramesInFlight> Shadows{};
    std::array<GPUBuffer, FramesInFlight> SceneFrames{};
    uint32_t ShadowSize = 0;
    VkFormat DepthFormat = VK_FORMAT_UNDEFINED;
    VkCommandPool CommandPool = VK_NULL_HANDLE;
    std::array<VkCommandBuffer, FramesInFlight> CommandBuffers{};
    std::array<VkSemaphore, FramesInFlight> ImageAvailable{};
    std::array<VkFence, FramesInFlight> FrameFences{};
    VkDescriptorPool ImGuiDescriptorPool = VK_NULL_HANDLE;
    std::unordered_map<const Mesh*, GPUMesh> Meshes;
    std::unordered_map<const Model*, GPUModel> Models;
    std::unordered_map<const Texture*, GPUTexture> Textures;
    std::unordered_map<uint64_t, GPUFontAtlas> FontAtlases;
    uint64_t NextFontAtlasHandle = 1;
    std::vector<RenderModelPreview> ModelPreviews;

    uint32_t Frame = 0;
    uint32_t ImageIndex = 0;
    uint32_t RequestedWidth = 0;
    uint32_t RequestedHeight = 0;
    bool VSync = false;
    bool SwapchainDirty = false;
    bool FrameStarted = false;
    bool RenderPassActive = false;
    bool DepthPrepared = false;
    bool ParticlePassActive = false;
    bool PresentPassActive = false;
    bool RenderForEditor = false;
    bool SceneInitialized = false;
    bool ImGuiInitialized = false;
    bool ImGuiFrameActive = false;
    bool ValidationEnabled = false;
    bool ValidationError = false;
    bool Initialized = false;
  } s_Data;

  bool PrepareGPUCull(bool useHiZ, const glm::mat4& viewProjection,
                      bool includePreviews);

  void Check(const VkResult result, const char* operation)
  {
    if (result != VK_SUCCESS)
      throw std::runtime_error(std::string(operation) + " failed (VkResult " +
                               std::to_string(static_cast<int>(result)) + ')');
  }

  VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
    void*)
  {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    {
      s_Data.ValidationError = true;
      gablog_log(LOG_ERROR, __FILE__, __LINE__, "Vulkan validation: %s", callbackData->pMessage);
    }
    else if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
      gablog_log(LOG_WARN, __FILE__, __LINE__, "Vulkan validation: %s", callbackData->pMessage);
    return VK_FALSE;
  }

  VkDebugUtilsMessengerCreateInfoEXT DebugMessengerInfo()
  {
    VkDebugUtilsMessengerCreateInfoEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = DebugCallback;
    return info;
  }

  bool HasInstanceLayer(const char* name)
  {
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    return std::ranges::any_of(layers, [name](const VkLayerProperties& layer)
    {
      return std::strcmp(layer.layerName, name) == 0;
    });
  }

  QueueFamilies FindQueueFamilies(const VkPhysicalDevice device)
  {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> properties(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, properties.data());

    QueueFamilies result;
    for (uint32_t index = 0; index < count; ++index)
    {
      if (properties[index].queueCount > 0 &&
          (properties[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0)
        result.Graphics = index;
      VkBool32 present = VK_FALSE;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, index, s_Data.Surface, &present);
      if (properties[index].queueCount > 0 && present) result.Present = index;
      if (result.Complete()) break;
    }
    return result;
  }

  bool SupportsDeviceExtensions(const VkPhysicalDevice device)
  {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> properties(count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, properties.data());
    return std::ranges::all_of(DeviceExtensions, [&](const char* required)
    {
      return std::ranges::any_of(properties, [required](const VkExtensionProperties& property)
      {
        return std::strcmp(property.extensionName, required) == 0;
      });
    });
  }

  bool HasSwapchainSupport(const VkPhysicalDevice device)
  {
    uint32_t formats = 0;
    uint32_t modes = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, s_Data.Surface, &formats, nullptr);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, s_Data.Surface, &modes, nullptr);
    return formats > 0 && modes > 0;
  }

  void CreateInstance()
  {
    if (glfwVulkanSupported() != GLFW_TRUE)
      throw std::runtime_error("GLFW could not find a Vulkan loader and physical device");

    uint32_t extensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&extensionCount);
    if (!glfwExtensions || extensionCount == 0)
      throw std::runtime_error("GLFW did not provide Vulkan surface extensions");
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + extensionCount);

    constexpr const char* ValidationLayer = "VK_LAYER_KHRONOS_validation";
    const char* validation = std::getenv("GABGL_VULKAN_VALIDATION");
    s_Data.ValidationEnabled = validation && std::strcmp(validation, "0") != 0 &&
                               HasInstanceLayer(ValidationLayer);
    if (s_Data.ValidationEnabled)
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "GABGL";
    application.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    application.pEngineName = "GABGL";
    application.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    application.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    createInfo.pApplicationInfo = &application;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    VkDebugUtilsMessengerCreateInfoEXT debugInfo{};
    if (s_Data.ValidationEnabled)
    {
      createInfo.enabledLayerCount = 1;
      createInfo.ppEnabledLayerNames = &ValidationLayer;
      debugInfo = DebugMessengerInfo();
      createInfo.pNext = &debugInfo;
    }
    Check(vkCreateInstance(&createInfo, nullptr, &s_Data.Instance), "vkCreateInstance");

    if (s_Data.ValidationEnabled)
    {
      const auto createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(s_Data.Instance, "vkCreateDebugUtilsMessengerEXT"));
      if (createMessenger)
        Check(createMessenger(s_Data.Instance, &debugInfo, nullptr, &s_Data.DebugMessenger),
              "vkCreateDebugUtilsMessengerEXT");
    }
  }

  void PickPhysicalDevice()
  {
    uint32_t count = 0;
    Check(vkEnumeratePhysicalDevices(s_Data.Instance, &count, nullptr),
          "vkEnumeratePhysicalDevices");
    if (count == 0) throw std::runtime_error("No Vulkan physical device is available");
    std::vector<VkPhysicalDevice> devices(count);
    Check(vkEnumeratePhysicalDevices(s_Data.Instance, &count, devices.data()),
          "vkEnumeratePhysicalDevices");

    int bestScore = -1;
    for (const VkPhysicalDevice device : devices)
    {
      const QueueFamilies families = FindQueueFamilies(device);
      if (!families.Complete() || !SupportsDeviceExtensions(device) || !HasSwapchainSupport(device))
        continue;
      VkPhysicalDeviceProperties properties{};
      vkGetPhysicalDeviceProperties(device, &properties);
      int score = static_cast<int>(properties.limits.maxImageDimension2D);
      if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 100000;
      if (score <= bestScore) continue;
      bestScore = score;
      s_Data.PhysicalDevice = device;
      s_Data.GraphicsQueueFamily = *families.Graphics;
      s_Data.PresentQueueFamily = *families.Present;
    }
    if (s_Data.PhysicalDevice == VK_NULL_HANDLE)
      throw std::runtime_error("No Vulkan device supports graphics and presentation");

    VkPhysicalDeviceProperties selected{};
    vkGetPhysicalDeviceProperties(s_Data.PhysicalDevice, &selected);
    gablog_log(LOG_INFO, __FILE__, __LINE__, "Vulkan device: %s (API %u.%u.%u)",
      selected.deviceName, VK_API_VERSION_MAJOR(selected.apiVersion),
      VK_API_VERSION_MINOR(selected.apiVersion), VK_API_VERSION_PATCH(selected.apiVersion));
  }

  void CreateDevice()
  {
    constexpr float priority = 1.0f;
    std::vector<uint32_t> uniqueFamilies{s_Data.GraphicsQueueFamily};
    if (s_Data.PresentQueueFamily != s_Data.GraphicsQueueFamily)
      uniqueFamilies.push_back(s_Data.PresentQueueFamily);
    std::vector<VkDeviceQueueCreateInfo> queues;
    for (const uint32_t family : uniqueFamilies)
    {
      VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
      queue.queueFamilyIndex = family;
      queue.queueCount = 1;
      queue.pQueuePriorities = &priority;
      queues.push_back(queue);
    }

    VkPhysicalDeviceVulkan11Features supported11{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceFeatures2 supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    supported.pNext = &supported11;
    vkGetPhysicalDeviceFeatures2(s_Data.PhysicalDevice, &supported);
    if (!supported11.shaderDrawParameters)
      throw std::runtime_error("Vulkan device does not support shader draw parameters");
    if (!supported.features.shaderStorageImageReadWithoutFormat ||
        !supported.features.shaderStorageImageWriteWithoutFormat)
      throw std::runtime_error("Vulkan device does not support unformatted storage images");
    VkPhysicalDeviceFeatures features{};
    features.shaderStorageImageReadWithoutFormat = VK_TRUE;
    features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    VkPhysicalDeviceVulkan11Features enabled11{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    enabled11.shaderDrawParameters = VK_TRUE;
    VkDeviceCreateInfo createInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    createInfo.pNext = &enabled11;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queues.size());
    createInfo.pQueueCreateInfos = queues.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(DeviceExtensions.size());
    createInfo.ppEnabledExtensionNames = DeviceExtensions.data();
    createInfo.pEnabledFeatures = &features;
    Check(vkCreateDevice(s_Data.PhysicalDevice, &createInfo, nullptr, &s_Data.Device),
          "vkCreateDevice");
    vkGetDeviceQueue(s_Data.Device, s_Data.GraphicsQueueFamily, 0, &s_Data.GraphicsQueue);
    vkGetDeviceQueue(s_Data.Device, s_Data.PresentQueueFamily, 0, &s_Data.PresentQueue);
  }

  VkSurfaceFormatKHR ChooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats)
  {
    const auto preferred = std::ranges::find_if(formats, [](const VkSurfaceFormatKHR& format)
    {
      return format.format == VK_FORMAT_B8G8R8A8_UNORM &&
             format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    });
    return preferred != formats.end() ? *preferred : formats.front();
  }

  VkPresentModeKHR ChoosePresentMode(const std::vector<VkPresentModeKHR>& modes)
  {
    if (s_Data.VSync) return VK_PRESENT_MODE_FIFO_KHR;
    if (std::ranges::find(modes, VK_PRESENT_MODE_MAILBOX_KHR) != modes.end())
      return VK_PRESENT_MODE_MAILBOX_KHR;
    if (std::ranges::find(modes, VK_PRESENT_MODE_IMMEDIATE_KHR) != modes.end())
      return VK_PRESENT_MODE_IMMEDIATE_KHR;
    return VK_PRESENT_MODE_FIFO_KHR;
  }

  uint32_t FindMemoryType(const uint32_t typeBits, const VkMemoryPropertyFlags properties)
  {
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(s_Data.PhysicalDevice, &memory);
    for (uint32_t index = 0; index < memory.memoryTypeCount; ++index)
      if ((typeBits & (1u << index)) != 0 &&
          (memory.memoryTypes[index].propertyFlags & properties) == properties)
        return index;
    throw std::runtime_error("No compatible Vulkan memory type is available");
  }

  void DestroyBuffer(GPUBuffer& buffer)
  {
    if (buffer.Buffer) vkDestroyBuffer(s_Data.Device, buffer.Buffer, nullptr);
    if (buffer.Memory) vkFreeMemory(s_Data.Device, buffer.Memory, nullptr);
    buffer = {};
  }

  void DestroyTexture(GPUTexture& texture)
  {
    if (texture.View) vkDestroyImageView(s_Data.Device, texture.View, nullptr);
    if (texture.Image) vkDestroyImage(s_Data.Device, texture.Image, nullptr);
    if (texture.Memory) vkFreeMemory(s_Data.Device, texture.Memory, nullptr);
    texture = {};
  }

  void DestroyDepthAttachment(DepthAttachment& depth)
  {
    if (depth.View) vkDestroyImageView(s_Data.Device, depth.View, nullptr);
    if (depth.Image) vkDestroyImage(s_Data.Device, depth.Image, nullptr);
    if (depth.Memory) vkFreeMemory(s_Data.Device, depth.Memory, nullptr);
    depth = {};
  }

  GPUBuffer CreateUploadBuffer(const void* data, const VkDeviceSize bytes,
                               const VkBufferUsageFlags usage)
  {
    GPUBuffer result;
    VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer.size = std::max<VkDeviceSize>(bytes, 4);
    buffer.usage = usage;
    buffer.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    Check(vkCreateBuffer(s_Data.Device, &buffer, nullptr, &result.Buffer), "vkCreateBuffer");
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(s_Data.Device, result.Buffer, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = FindMemoryType(requirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    try
    {
      Check(vkAllocateMemory(s_Data.Device, &allocation, nullptr, &result.Memory),
            "vkAllocateMemory (buffer)");
      Check(vkBindBufferMemory(s_Data.Device, result.Buffer, result.Memory, 0),
            "vkBindBufferMemory");
      void* mapped = nullptr;
      Check(vkMapMemory(s_Data.Device, result.Memory, 0, bytes, 0, &mapped), "vkMapMemory");
      std::memcpy(mapped, data, static_cast<size_t>(bytes));
      vkUnmapMemory(s_Data.Device, result.Memory);
      return result;
    }
    catch (...)
    {
      DestroyBuffer(result);
      throw;
    }
  }

  VkCommandBuffer BeginImmediateCommands()
  {
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = s_Data.CommandPool;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = 1;
    Check(vkAllocateCommandBuffers(s_Data.Device, &allocation, &command),
          "vkAllocateCommandBuffers (upload)");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    try
    {
      Check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer (upload)");
      return command;
    }
    catch (...)
    {
      vkFreeCommandBuffers(s_Data.Device, s_Data.CommandPool, 1, &command);
      throw;
    }
  }

  void EndImmediateCommands(const VkCommandBuffer command)
  {
    try
    {
      Check(vkEndCommandBuffer(command), "vkEndCommandBuffer (upload)");
      VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      submit.commandBufferCount = 1;
      submit.pCommandBuffers = &command;
      Check(vkQueueSubmit(s_Data.GraphicsQueue, 1, &submit, VK_NULL_HANDLE),
            "vkQueueSubmit (upload)");
      Check(vkQueueWaitIdle(s_Data.GraphicsQueue), "vkQueueWaitIdle (upload)");
    }
    catch (...)
    {
      vkFreeCommandBuffers(s_Data.Device, s_Data.CommandPool, 1, &command);
      throw;
    }
    vkFreeCommandBuffers(s_Data.Device, s_Data.CommandPool, 1, &command);
  }

  GPUTexture CreateTextureRGBA(const uint8_t* pixels, const uint32_t width,
                               const uint32_t height)
  {
    if (!pixels || width == 0 || height == 0)
      throw std::runtime_error("Cannot upload an empty Vulkan texture");
    const VkDeviceSize byteCount = static_cast<VkDeviceSize>(width) * height * 4;
    GPUBuffer staging = CreateUploadBuffer(pixels, byteCount, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    GPUTexture result;
    try
    {
      VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      image.imageType = VK_IMAGE_TYPE_2D;
      image.extent = {width, height, 1};
      image.mipLevels = 1;
      image.arrayLayers = 1;
      image.format = VK_FORMAT_R8G8B8A8_UNORM;
      image.tiling = VK_IMAGE_TILING_OPTIMAL;
      image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
      image.samples = VK_SAMPLE_COUNT_1_BIT;
      image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      Check(vkCreateImage(s_Data.Device, &image, nullptr, &result.Image),
            "vkCreateImage (texture)");
      VkMemoryRequirements requirements{};
      vkGetImageMemoryRequirements(s_Data.Device, result.Image, &requirements);
      VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      allocation.allocationSize = requirements.size;
      allocation.memoryTypeIndex = FindMemoryType(
        requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      Check(vkAllocateMemory(s_Data.Device, &allocation, nullptr, &result.Memory),
            "vkAllocateMemory (texture)");
      Check(vkBindImageMemory(s_Data.Device, result.Image, result.Memory, 0),
            "vkBindImageMemory (texture)");

      const VkCommandBuffer command = BeginImmediateCommands();
      VkImageMemoryBarrier toTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      toTransfer.image = result.Image;
      toTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      toTransfer.subresourceRange.levelCount = 1;
      toTransfer.subresourceRange.layerCount = 1;
      toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                           1, &toTransfer);
      VkBufferImageCopy copy{};
      copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      copy.imageSubresource.layerCount = 1;
      copy.imageExtent = {width, height, 1};
      vkCmdCopyBufferToImage(command, staging.Buffer, result.Image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
      VkImageMemoryBarrier toShader{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      toShader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      toShader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      toShader.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      toShader.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      toShader.image = result.Image;
      toShader.subresourceRange = toTransfer.subresourceRange;
      toShader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      toShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                           1, &toShader);
      EndImmediateCommands(command);

      VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      view.image = result.Image;
      view.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view.format = VK_FORMAT_R8G8B8A8_UNORM;
      view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      view.subresourceRange.levelCount = 1;
      view.subresourceRange.layerCount = 1;
      Check(vkCreateImageView(s_Data.Device, &view, nullptr, &result.View),
            "vkCreateImageView (texture)");
      DestroyBuffer(staging);
      return result;
    }
    catch (...)
    {
      DestroyBuffer(staging);
      DestroyTexture(result);
      throw;
    }
  }

  GPUTexture CreateCubeTextureRGBA(const uint8_t* pixels, const uint32_t width,
                                   const uint32_t height)
  {
    if (!pixels || width == 0 || height == 0)
      throw std::runtime_error("Cannot upload an empty Vulkan cubemap");
    const VkDeviceSize faceBytes = static_cast<VkDeviceSize>(width) * height * 4;
    GPUBuffer staging = CreateUploadBuffer(
      pixels, faceBytes * 6, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    GPUTexture result;
    try
    {
      VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      image.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
      image.imageType = VK_IMAGE_TYPE_2D;
      image.extent = {width, height, 1};
      image.mipLevels = 1;
      image.arrayLayers = 6;
      image.format = VK_FORMAT_R8G8B8A8_UNORM;
      image.tiling = VK_IMAGE_TILING_OPTIMAL;
      image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
      image.samples = VK_SAMPLE_COUNT_1_BIT;
      image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      Check(vkCreateImage(s_Data.Device, &image, nullptr, &result.Image),
            "vkCreateImage (cubemap)");
      VkMemoryRequirements requirements{};
      vkGetImageMemoryRequirements(s_Data.Device, result.Image, &requirements);
      VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      allocation.allocationSize = requirements.size;
      allocation.memoryTypeIndex = FindMemoryType(
        requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      Check(vkAllocateMemory(s_Data.Device, &allocation, nullptr, &result.Memory),
            "vkAllocateMemory (cubemap)");
      Check(vkBindImageMemory(s_Data.Device, result.Image, result.Memory, 0),
            "vkBindImageMemory (cubemap)");

      const VkCommandBuffer command = BeginImmediateCommands();
      VkImageMemoryBarrier toTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      toTransfer.image = result.Image;
      toTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      toTransfer.subresourceRange.levelCount = 1;
      toTransfer.subresourceRange.layerCount = 6;
      toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                           1, &toTransfer);
      std::array<VkBufferImageCopy, 6> copies{};
      for (uint32_t face = 0; face < copies.size(); ++face)
      {
        copies[face].bufferOffset = faceBytes * face;
        copies[face].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copies[face].imageSubresource.baseArrayLayer = face;
        copies[face].imageSubresource.layerCount = 1;
        copies[face].imageExtent = {width, height, 1};
      }
      vkCmdCopyBufferToImage(command, staging.Buffer, result.Image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             static_cast<uint32_t>(copies.size()), copies.data());
      VkImageMemoryBarrier toShader{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      toShader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      toShader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      toShader.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      toShader.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      toShader.image = result.Image;
      toShader.subresourceRange = toTransfer.subresourceRange;
      toShader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      toShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                           0, nullptr, 0, nullptr, 1, &toShader);
      EndImmediateCommands(command);

      VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      view.image = result.Image;
      view.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
      view.format = VK_FORMAT_R8G8B8A8_UNORM;
      view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      view.subresourceRange.levelCount = 1;
      view.subresourceRange.layerCount = 6;
      Check(vkCreateImageView(s_Data.Device, &view, nullptr, &result.View),
            "vkCreateImageView (cubemap)");
      DestroyBuffer(staging);
      return result;
    }
    catch (...)
    {
      DestroyBuffer(staging);
      DestroyTexture(result);
      throw;
    }
  }

  GPUTexture CreateCubeTexture(Texture& source)
  {
    const uint32_t width = source.GetWidth();
    const uint32_t height = source.GetHeight();
    const int channels = std::clamp(source.GetChannels(), 1, 4);
    auto& faces = source.GetPixels();
    for (const uint8_t* face : faces)
      if (!face) throw std::runtime_error("Cubemap has a missing face");
    std::vector<uint8_t> rgba(static_cast<size_t>(width) * height * 4 * 6);
    const size_t pixelCount = static_cast<size_t>(width) * height;
    for (size_t faceIndex = 0; faceIndex < faces.size(); ++faceIndex)
      for (size_t pixel = 0; pixel < pixelCount; ++pixel)
      {
        const uint8_t* input = faces[faceIndex] + pixel * channels;
        uint8_t* output = rgba.data() + (faceIndex * pixelCount + pixel) * 4;
        output[0] = input[0];
        output[1] = channels > 1 ? input[1] : input[0];
        output[2] = channels > 2 ? input[2] : input[0];
        output[3] = channels > 3 ? input[3] : 255;
      }
    return CreateCubeTextureRGBA(rgba.data(), width, height);
  }

  GPUTexture CreateTexture(const Texture& source)
  {
    const uint8_t* raw = source.GetRawData();
    const uint32_t width = source.GetWidth();
    const uint32_t height = source.GetHeight();
    const int channels = std::clamp(source.GetChannels(), 1, 4);
    if (!raw || width == 0 || height == 0)
      throw std::runtime_error("Texture has no CPU pixels for Vulkan upload: " + source.GetPath());
    std::vector<uint8_t> rgba(static_cast<size_t>(width) * height * 4);
    for (size_t pixel = 0; pixel < static_cast<size_t>(width) * height; ++pixel)
    {
      const uint8_t* input = raw + pixel * channels;
      uint8_t* output = rgba.data() + pixel * 4;
      if (channels == 1)
      {
        output[0] = output[1] = output[2] = input[0];
        output[3] = 255;
      }
      else if (channels == 2)
      {
        output[0] = output[1] = output[2] = input[0];
        output[3] = input[1];
      }
      else
      {
        output[0] = input[0];
        output[1] = input[1];
        output[2] = input[2];
        output[3] = channels == 4 ? input[3] : 255;
      }
    }
    return CreateTextureRGBA(rgba.data(), width, height);
  }

  const GPUTexture& ResolveTexture(const std::shared_ptr<Texture>& source,
                                   const GPUTexture& fallback)
  {
    if (!source) return fallback;
    const Texture* key = source.get();
    auto uploaded = s_Data.Textures.find(key);
    if (uploaded == s_Data.Textures.end() && key->GetRawData() &&
        key->GetWidth() > 0 && key->GetHeight() > 0)
    {
      GPUTexture texture = CreateTexture(*key);
      uploaded = s_Data.Textures.emplace(key, std::move(texture)).first;
    }
    return uploaded != s_Data.Textures.end() ? uploaded->second : fallback;
  }

  VkFormat ChooseDepthFormat()
  {
    constexpr std::array candidates{
      VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT};
    for (const VkFormat format : candidates)
    {
      VkFormatProperties properties{};
      vkGetPhysicalDeviceFormatProperties(s_Data.PhysicalDevice, format, &properties);
      if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0)
        return format;
    }
    throw std::runtime_error("No Vulkan depth attachment format is available");
  }

  DepthAttachment CreateDepthAttachment(uint32_t width = 0, uint32_t height = 0)
  {
    DepthAttachment result;
    if (width == 0) width = s_Data.Extent.width;
    if (height == 0) height = s_Data.Extent.height;
    VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image.imageType = VK_IMAGE_TYPE_2D;
    image.extent = {width, height, 1};
    image.mipLevels = 1;
    image.arrayLayers = 1;
    image.format = s_Data.DepthFormat;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                  VK_IMAGE_USAGE_SAMPLED_BIT;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    Check(vkCreateImage(s_Data.Device, &image, nullptr, &result.Image), "vkCreateImage (depth)");
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(s_Data.Device, result.Image, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = FindMemoryType(
      requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Check(vkAllocateMemory(s_Data.Device, &allocation, nullptr, &result.Memory),
          "vkAllocateMemory (depth)");
    Check(vkBindImageMemory(s_Data.Device, result.Image, result.Memory, 0),
          "vkBindImageMemory (depth)");
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = result.Image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = s_Data.DepthFormat;
    view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    view.subresourceRange.levelCount = 1;
    view.subresourceRange.layerCount = 1;
    Check(vkCreateImageView(s_Data.Device, &view, nullptr, &result.View),
          "vkCreateImageView (depth)");
    return result;
  }

  GPUTexture CreateColorAttachment(const VkFormat format)
  {
    GPUTexture result;
    VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image.imageType = VK_IMAGE_TYPE_2D;
    image.extent = {s_Data.Extent.width, s_Data.Extent.height, 1};
    image.mipLevels = 1;
    image.arrayLayers = 1;
    image.format = format;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    Check(vkCreateImage(s_Data.Device, &image, nullptr, &result.Image),
          "vkCreateImage (scene color)");
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(s_Data.Device, result.Image, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = FindMemoryType(
      requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Check(vkAllocateMemory(s_Data.Device, &allocation, nullptr, &result.Memory),
          "vkAllocateMemory (scene color)");
    Check(vkBindImageMemory(s_Data.Device, result.Image, result.Memory, 0),
          "vkBindImageMemory (scene color)");
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = result.Image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = format;
    view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view.subresourceRange.levelCount = 1;
    view.subresourceRange.layerCount = 1;
    Check(vkCreateImageView(s_Data.Device, &view, nullptr, &result.View),
          "vkCreateImageView (scene color)");
    return result;
  }

  GPUTexture CreateBloomPyramid()
  {
    GPUTexture result;
    VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image.imageType = VK_IMAGE_TYPE_2D;
    image.extent = {s_Data.Extent.width, s_Data.Extent.height, 1};
    image.mipLevels = s_Data.BloomMipCount;
    image.arrayLayers = 1;
    image.format = SceneColorFormat;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                  VK_IMAGE_USAGE_SAMPLED_BIT;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    Check(vkCreateImage(s_Data.Device, &image, nullptr, &result.Image),
          "vkCreateImage (bloom pyramid)");
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(s_Data.Device, result.Image, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = FindMemoryType(
      requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Check(vkAllocateMemory(s_Data.Device, &allocation, nullptr, &result.Memory),
          "vkAllocateMemory (bloom pyramid)");
    Check(vkBindImageMemory(s_Data.Device, result.Image, result.Memory, 0),
          "vkBindImageMemory (bloom pyramid)");
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = result.Image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = SceneColorFormat;
    view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view.subresourceRange.levelCount = s_Data.BloomMipCount;
    view.subresourceRange.layerCount = 1;
    Check(vkCreateImageView(s_Data.Device, &view, nullptr, &result.View),
          "vkCreateImageView (bloom pyramid)");
    return result;
  }

  void DestroyHiZPyramids()
  {
    for (HiZPyramid& pyramid : s_Data.HiZ)
    {
      for (VkImageView& view : pyramid.MipViews)
      {
        if (view) vkDestroyImageView(s_Data.Device, view, nullptr);
        view = VK_NULL_HANDLE;
      }
      DestroyTexture(pyramid.Texture);
      pyramid = {};
    }
  }

  void CreateHiZPyramids()
  {
    DestroyHiZPyramids();
    const uint32_t mipCount = std::min(MaxHiZMips, 1u + static_cast<uint32_t>(
      std::floor(std::log2(static_cast<float>(std::max(
        s_Data.Extent.width, s_Data.Extent.height))))));
    for (HiZPyramid& pyramid : s_Data.HiZ)
    {
      pyramid.MipCount = mipCount;
      VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      image.imageType = VK_IMAGE_TYPE_2D;
      image.extent = {s_Data.Extent.width, s_Data.Extent.height, 1};
      image.mipLevels = mipCount;
      image.arrayLayers = 1;
      image.format = VK_FORMAT_R32_SFLOAT;
      image.tiling = VK_IMAGE_TILING_OPTIMAL;
      image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      image.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
      image.samples = VK_SAMPLE_COUNT_1_BIT;
      image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      Check(vkCreateImage(s_Data.Device, &image, nullptr, &pyramid.Texture.Image),
            "vkCreateImage (Hi-Z)");
      VkMemoryRequirements requirements{};
      vkGetImageMemoryRequirements(s_Data.Device, pyramid.Texture.Image, &requirements);
      VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      allocation.allocationSize = requirements.size;
      allocation.memoryTypeIndex = FindMemoryType(
        requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      Check(vkAllocateMemory(s_Data.Device, &allocation, nullptr, &pyramid.Texture.Memory),
            "vkAllocateMemory (Hi-Z)");
      Check(vkBindImageMemory(s_Data.Device, pyramid.Texture.Image,
                              pyramid.Texture.Memory, 0),
            "vkBindImageMemory (Hi-Z)");
      VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      view.image = pyramid.Texture.Image;
      view.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view.format = VK_FORMAT_R32_SFLOAT;
      view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      view.subresourceRange.levelCount = mipCount;
      view.subresourceRange.layerCount = 1;
      Check(vkCreateImageView(s_Data.Device, &view, nullptr, &pyramid.Texture.View),
            "vkCreateImageView (Hi-Z)");
      for (uint32_t mip = 0; mip < mipCount; ++mip)
      {
        VkImageViewCreateInfo mipView = view;
        mipView.subresourceRange.baseMipLevel = mip;
        mipView.subresourceRange.levelCount = 1;
        Check(vkCreateImageView(s_Data.Device, &mipView, nullptr,
                                &pyramid.MipViews[mip]),
              "vkCreateImageView (Hi-Z mip)");
      }
    }
    const VkCommandBuffer command = BeginImmediateCommands();
    std::array<VkImageMemoryBarrier, FramesInFlight> barriers{};
    for (uint32_t frame = 0; frame < FramesInFlight; ++frame)
    {
      barriers[frame].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barriers[frame].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      barriers[frame].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      barriers[frame].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[frame].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[frame].image = s_Data.HiZ[frame].Texture.Image;
      barriers[frame].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      barriers[frame].subresourceRange.levelCount = mipCount;
      barriers[frame].subresourceRange.layerCount = 1;
      barriers[frame].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    }
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         0, nullptr, 0, nullptr,
                         static_cast<uint32_t>(barriers.size()), barriers.data());
    EndImmediateCommands(command);
  }

  VkShaderModule CreateShaderModule(const Shader::Bytecode& bytecode)
  {
    if (bytecode.Bytes.empty() || bytecode.Bytes.size() % sizeof(uint32_t) != 0)
      throw std::runtime_error("Slang produced invalid SPIR-V bytecode");
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = bytecode.Bytes.size();
    info.pCode = reinterpret_cast<const uint32_t*>(bytecode.Bytes.data());
    VkShaderModule module = VK_NULL_HANDLE;
    Check(vkCreateShaderModule(s_Data.Device, &info, nullptr, &module),
          "vkCreateShaderModule");
    return module;
  }

  void CreateSceneDescriptors()
  {
    const std::array<VkDescriptorSetLayoutBinding, 9> bindings{{
      {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
      {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {7, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr}}};
    VkDescriptorSetLayoutCreateInfo layout{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout.bindingCount = static_cast<uint32_t>(bindings.size());
    layout.pBindings = bindings.data();
    Check(vkCreateDescriptorSetLayout(s_Data.Device, &layout, nullptr,
                                      &s_Data.SceneDescriptorSetLayout),
          "vkCreateDescriptorSetLayout (scene)");

    constexpr uint32_t MaxMaterials = 4096;
    const std::array<VkDescriptorPoolSize, 4> sizes{{
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, MaxMaterials * 4},
      {VK_DESCRIPTOR_TYPE_SAMPLER, MaxMaterials * 2},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MaxMaterials * 2},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MaxMaterials}}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets = MaxMaterials;
    pool.poolSizeCount = static_cast<uint32_t>(sizes.size());
    pool.pPoolSizes = sizes.data();
    Check(vkCreateDescriptorPool(s_Data.Device, &pool, nullptr, &s_Data.SceneDescriptorPool),
          "vkCreateDescriptorPool (scene)");

    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler.maxLod = 0.0f;
    Check(vkCreateSampler(s_Data.Device, &sampler, nullptr, &s_Data.MaterialSampler),
          "vkCreateSampler (scene)");
    VkSamplerCreateInfo shadowSampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    shadowSampler.magFilter = VK_FILTER_NEAREST;
    shadowSampler.minFilter = VK_FILTER_NEAREST;
    shadowSampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    shadowSampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    shadowSampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    shadowSampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    shadowSampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    shadowSampler.maxLod = 0.0f;
    Check(vkCreateSampler(s_Data.Device, &shadowSampler, nullptr, &s_Data.ShadowSampler),
          "vkCreateSampler (shadow)");

    constexpr std::array<uint8_t, 4> white{255, 255, 255, 255};
    constexpr std::array<uint8_t, 4> neutralNormal{128, 128, 255, 255};
    constexpr std::array<uint8_t, 4> black{0, 0, 0, 255};
    s_Data.WhiteTexture = CreateTextureRGBA(white.data(), 1, 1);
    s_Data.NeutralNormalTexture = CreateTextureRGBA(neutralNormal.data(), 1, 1);
    s_Data.BlackTexture = CreateTextureRGBA(black.data(), 1, 1);
  }

  VkDescriptorSet CreateMaterialDescriptor(const GPUTexture& diffuse,
                                           const GPUTexture& normal,
                                           const GPUTexture& specular,
                                           const GPUBuffer& bones,
                                           const GPUBuffer& sceneFrame,
                                           const GPUTexture& shadowTexture,
                                           const GPUBuffer& visibleTransforms)
  {
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = s_Data.SceneDescriptorPool;
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &s_Data.SceneDescriptorSetLayout;
    VkDescriptorSet result = VK_NULL_HANDLE;
    Check(vkAllocateDescriptorSets(s_Data.Device, &allocation, &result),
          "vkAllocateDescriptorSets (material)");
    VkDescriptorImageInfo diffuseImage{};
    diffuseImage.imageView = diffuse.View;
    diffuseImage.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo sampler{};
    sampler.sampler = s_Data.MaterialSampler;
    VkDescriptorImageInfo normalImage{};
    normalImage.imageView = normal.View;
    normalImage.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo specularImage{};
    specularImage.imageView = specular.View;
    specularImage.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorBufferInfo boneBuffer{};
    boneBuffer.buffer = bones.Buffer;
    boneBuffer.range = MAX_BONES * sizeof(glm::mat4);
    VkDescriptorBufferInfo frameBuffer{};
    frameBuffer.buffer = sceneFrame.Buffer;
    frameBuffer.range = sizeof(SceneFrameData);
    VkDescriptorImageInfo shadowImage{};
    shadowImage.imageView = shadowTexture.View;
    shadowImage.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo shadowSampler{};
    shadowSampler.sampler = s_Data.ShadowSampler;
    VkDescriptorBufferInfo visibleBuffer{};
    visibleBuffer.buffer = visibleTransforms.Buffer;
    visibleBuffer.range = VK_WHOLE_SIZE;
    const std::array<VkWriteDescriptorSet, 9> writes{{
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, result, 0, 0, 1,
       VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &diffuseImage, nullptr, nullptr},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, result, 1, 0, 1,
       VK_DESCRIPTOR_TYPE_SAMPLER, &sampler, nullptr, nullptr},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, result, 2, 0, 1,
       VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &normalImage, nullptr, nullptr},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, result, 3, 0, 1,
       VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &specularImage, nullptr, nullptr},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, result, 4, 0, 1,
       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &boneBuffer, nullptr},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, result, 5, 0, 1,
       VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &frameBuffer, nullptr},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, result, 6, 0, 1,
       VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &shadowImage, nullptr, nullptr},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, result, 7, 0, 1,
       VK_DESCRIPTOR_TYPE_SAMPLER, &shadowSampler, nullptr, nullptr},
      {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, result, 8, 0, 1,
       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &visibleBuffer, nullptr}}};
    vkUpdateDescriptorSets(s_Data.Device, static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
    return result;
  }

  void CreateScenePipeline()
  {
    std::filesystem::path shaderPath = "../res/shaders/vulkan_basic.slang";
    if (!std::filesystem::exists(shaderPath)) shaderPath = "res/shaders/vulkan_basic.slang";
    const auto vertexBytecode = Shader::CompileSlangSPIRV(
      shaderPath, "VSMain", "vertex");
    const auto fragmentBytecode = Shader::CompileSlangSPIRV(
      shaderPath, "PSMain", "fragment");
    const VkShaderModule vertex = CreateShaderModule(vertexBytecode);
    const VkShaderModule fragment = CreateShaderModule(fragmentBytecode);
    try
    {
      const VkPipelineShaderStageCreateInfo stages[]{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_VERTEX_BIT, vertex, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_FRAGMENT_BIT, fragment, "main", nullptr}};
      const VkVertexInputBindingDescription binding{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
      const std::array<VkVertexInputAttributeDescription, 7> attributes{{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, Position))},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, Normal))},
        {2, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, TexCoords))},
        {3, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, Tangent))},
        {4, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, Bitangent))},
        {5, 0, VK_FORMAT_R32G32B32A32_SINT, static_cast<uint32_t>(offsetof(Vertex, m_BoneIDs))},
        {6, 0, VK_FORMAT_R32G32B32A32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, m_Weights))}}};
      VkPipelineVertexInputStateCreateInfo vertexInput{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
      vertexInput.vertexBindingDescriptionCount = 1;
      vertexInput.pVertexBindingDescriptions = &binding;
      vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
      vertexInput.pVertexAttributeDescriptions = attributes.data();
      VkPipelineInputAssemblyStateCreateInfo assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
      assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      VkPipelineViewportStateCreateInfo viewport{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
      viewport.viewportCount = 1;
      viewport.scissorCount = 1;
      VkPipelineRasterizationStateCreateInfo raster{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
      raster.polygonMode = VK_POLYGON_MODE_FILL;
      raster.cullMode = VK_CULL_MODE_NONE;
      raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
      raster.lineWidth = 1.0f;
      VkPipelineMultisampleStateCreateInfo multisample{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
      multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
      VkPipelineDepthStencilStateCreateInfo depth{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
      depth.depthTestEnable = VK_TRUE;
      depth.depthWriteEnable = VK_FALSE;
      depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
      VkPipelineColorBlendAttachmentState blend{};
      blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      VkPipelineColorBlendStateCreateInfo blending{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
      const std::array sceneBlends{blend, blend, blend};
      blending.attachmentCount = static_cast<uint32_t>(sceneBlends.size());
      blending.pAttachments = sceneBlends.data();
      constexpr std::array dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
      VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
      dynamic.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
      dynamic.pDynamicStates = dynamicStates.data();
      VkPushConstantRange push{};
      push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
      push.size = sizeof(ScenePushConstants);
      VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      layout.setLayoutCount = 1;
      layout.pSetLayouts = &s_Data.SceneDescriptorSetLayout;
      layout.pushConstantRangeCount = 1;
      layout.pPushConstantRanges = &push;
      Check(vkCreatePipelineLayout(s_Data.Device, &layout, nullptr,
                                   &s_Data.ScenePipelineLayout), "vkCreatePipelineLayout (scene)");
      VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
      pipeline.stageCount = 2;
      pipeline.pStages = stages;
      pipeline.pVertexInputState = &vertexInput;
      pipeline.pInputAssemblyState = &assembly;
      pipeline.pViewportState = &viewport;
      pipeline.pRasterizationState = &raster;
      pipeline.pMultisampleState = &multisample;
      pipeline.pDepthStencilState = &depth;
      pipeline.pColorBlendState = &blending;
      pipeline.pDynamicState = &dynamic;
      pipeline.layout = s_Data.ScenePipelineLayout;
      pipeline.renderPass = s_Data.RenderPass;
      Check(vkCreateGraphicsPipelines(s_Data.Device, VK_NULL_HANDLE, 1, &pipeline,
                                      nullptr, &s_Data.ScenePipeline),
            "vkCreateGraphicsPipelines (scene)");
    }
    catch (...)
    {
      vkDestroyShaderModule(s_Data.Device, fragment, nullptr);
      vkDestroyShaderModule(s_Data.Device, vertex, nullptr);
      throw;
    }
    vkDestroyShaderModule(s_Data.Device, fragment, nullptr);
    vkDestroyShaderModule(s_Data.Device, vertex, nullptr);
  }

  void CreateDepthPrepassPipeline()
  {
    std::filesystem::path shaderPath = "../res/shaders/vulkan_depth_prepass.slang";
    if (!std::filesystem::exists(shaderPath)) shaderPath = "res/shaders/vulkan_depth_prepass.slang";
    const VkShaderModule vertex = CreateShaderModule(
      Shader::CompileSlangSPIRV(shaderPath, "VSMain", "vertex"));
    try
    {
      VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
      stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
      stage.module = vertex;
      stage.pName = "main";
      const VkVertexInputBindingDescription binding{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
      const std::array<VkVertexInputAttributeDescription, 3> attributes{{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, Position))},
        {1, 0, VK_FORMAT_R32G32B32A32_SINT, static_cast<uint32_t>(offsetof(Vertex, m_BoneIDs))},
        {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, m_Weights))}}};
      VkPipelineVertexInputStateCreateInfo vertexInput{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
      vertexInput.vertexBindingDescriptionCount = 1;
      vertexInput.pVertexBindingDescriptions = &binding;
      vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
      vertexInput.pVertexAttributeDescriptions = attributes.data();
      VkPipelineInputAssemblyStateCreateInfo assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
      assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
      viewport.viewportCount = 1;
      viewport.scissorCount = 1;
      VkPipelineRasterizationStateCreateInfo raster{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
      raster.polygonMode = VK_POLYGON_MODE_FILL;
      raster.cullMode = VK_CULL_MODE_NONE;
      raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
      raster.depthBiasEnable = VK_TRUE;
      raster.lineWidth = 1.0f;
      VkPipelineMultisampleStateCreateInfo multisample{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
      multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
      VkPipelineDepthStencilStateCreateInfo depth{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
      depth.depthTestEnable = VK_TRUE;
      depth.depthWriteEnable = VK_TRUE;
      depth.depthCompareOp = VK_COMPARE_OP_LESS;
      constexpr std::array dynamicStates{
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS};
      VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
      dynamic.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
      dynamic.pDynamicStates = dynamicStates.data();
      VkPushConstantRange push{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ScenePushConstants)};
      VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      layout.setLayoutCount = 1;
      layout.pSetLayouts = &s_Data.SceneDescriptorSetLayout;
      layout.pushConstantRangeCount = 1;
      layout.pPushConstantRanges = &push;
      Check(vkCreatePipelineLayout(s_Data.Device, &layout, nullptr,
                                   &s_Data.DepthPrepassPipelineLayout),
            "vkCreatePipelineLayout (depth prepass)");
      VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
      pipeline.stageCount = 1;
      pipeline.pStages = &stage;
      pipeline.pVertexInputState = &vertexInput;
      pipeline.pInputAssemblyState = &assembly;
      pipeline.pViewportState = &viewport;
      pipeline.pRasterizationState = &raster;
      pipeline.pMultisampleState = &multisample;
      pipeline.pDepthStencilState = &depth;
      pipeline.pDynamicState = &dynamic;
      pipeline.layout = s_Data.DepthPrepassPipelineLayout;
      pipeline.renderPass = s_Data.DepthPrepassRenderPass;
      Check(vkCreateGraphicsPipelines(s_Data.Device, VK_NULL_HANDLE, 1, &pipeline,
                                      nullptr, &s_Data.DepthPrepassPipeline),
            "vkCreateGraphicsPipelines (depth prepass)");
    }
    catch (...)
    {
      vkDestroyShaderModule(s_Data.Device, vertex, nullptr);
      throw;
    }
    vkDestroyShaderModule(s_Data.Device, vertex, nullptr);
  }

  void CreateShadowRenderPass()
  {
    VkAttachmentDescription depth{};
    depth.format = s_Data.DepthFormat;
    depth.samples = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    VkAttachmentReference depthReference{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.pDepthStencilAttachment = &depthReference;
    const std::array<VkSubpassDependency, 2> dependencies{{
      {VK_SUBPASS_EXTERNAL, 0, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_SHADER_READ_BIT,
       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_DEPENDENCY_BY_REGION_BIT},
      {0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
       VK_ACCESS_SHADER_READ_BIT, VK_DEPENDENCY_BY_REGION_BIT}}};
    VkRenderPassCreateInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    info.attachmentCount = 1;
    info.pAttachments = &depth;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    info.dependencyCount = static_cast<uint32_t>(dependencies.size());
    info.pDependencies = dependencies.data();
    Check(vkCreateRenderPass(s_Data.Device, &info, nullptr, &s_Data.ShadowRenderPass),
          "vkCreateRenderPass (shadow)");
  }

  void DestroyShadowAttachments()
  {
    for (ShadowAttachment& shadow : s_Data.Shadows)
    {
      if (shadow.Framebuffer)
        vkDestroyFramebuffer(s_Data.Device, shadow.Framebuffer, nullptr);
      DestroyTexture(shadow.Texture);
      shadow = {};
    }
    s_Data.ShadowSize = 0;
  }

  void CreateShadowAttachments(const uint32_t size)
  {
    DestroyShadowAttachments();
    for (ShadowAttachment& shadow : s_Data.Shadows)
    {
      VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
      image.imageType = VK_IMAGE_TYPE_2D;
      image.extent = {size, size, 1};
      image.mipLevels = 1;
      image.arrayLayers = 1;
      image.format = s_Data.DepthFormat;
      image.tiling = VK_IMAGE_TILING_OPTIMAL;
      image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
      image.samples = VK_SAMPLE_COUNT_1_BIT;
      image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      Check(vkCreateImage(s_Data.Device, &image, nullptr, &shadow.Texture.Image),
            "vkCreateImage (shadow)");
      VkMemoryRequirements requirements{};
      vkGetImageMemoryRequirements(s_Data.Device, shadow.Texture.Image, &requirements);
      VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      allocation.allocationSize = requirements.size;
      allocation.memoryTypeIndex = FindMemoryType(
        requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      Check(vkAllocateMemory(s_Data.Device, &allocation, nullptr, &shadow.Texture.Memory),
            "vkAllocateMemory (shadow)");
      Check(vkBindImageMemory(s_Data.Device, shadow.Texture.Image, shadow.Texture.Memory, 0),
            "vkBindImageMemory (shadow)");
      VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      view.image = shadow.Texture.Image;
      view.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view.format = s_Data.DepthFormat;
      view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      view.subresourceRange.levelCount = 1;
      view.subresourceRange.layerCount = 1;
      Check(vkCreateImageView(s_Data.Device, &view, nullptr, &shadow.Texture.View),
            "vkCreateImageView (shadow)");
      VkFramebufferCreateInfo framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
      framebuffer.renderPass = s_Data.ShadowRenderPass;
      framebuffer.attachmentCount = 1;
      framebuffer.pAttachments = &shadow.Texture.View;
      framebuffer.width = size;
      framebuffer.height = size;
      framebuffer.layers = 1;
      Check(vkCreateFramebuffer(s_Data.Device, &framebuffer, nullptr, &shadow.Framebuffer),
            "vkCreateFramebuffer (shadow)");
    }
    const VkCommandBuffer command = BeginImmediateCommands();
    std::array<VkImageMemoryBarrier, FramesInFlight> barriers{};
    for (uint32_t frame = 0; frame < FramesInFlight; ++frame)
    {
      barriers[frame].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barriers[frame].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      barriers[frame].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
      barriers[frame].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[frame].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[frame].image = s_Data.Shadows[frame].Texture.Image;
      barriers[frame].subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      barriers[frame].subresourceRange.levelCount = 1;
      barriers[frame].subresourceRange.layerCount = 1;
      barriers[frame].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    }
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                         0, nullptr, 0, nullptr,
                         static_cast<uint32_t>(barriers.size()), barriers.data());
    EndImmediateCommands(command);
    s_Data.ShadowSize = size;
  }

  void CreateShadowPipeline()
  {
    std::filesystem::path shaderPath = "../res/shaders/vulkan_shadow.slang";
    if (!std::filesystem::exists(shaderPath)) shaderPath = "res/shaders/vulkan_shadow.slang";
    const VkShaderModule vertex = CreateShaderModule(
      Shader::CompileSlangSPIRV(shaderPath, "VSMain", "vertex"));
    try
    {
      VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
      stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
      stage.module = vertex;
      stage.pName = "main";
      const VkVertexInputBindingDescription binding{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
      const std::array<VkVertexInputAttributeDescription, 3> attributes{{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, Position))},
        {1, 0, VK_FORMAT_R32G32B32A32_SINT, static_cast<uint32_t>(offsetof(Vertex, m_BoneIDs))},
        {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, m_Weights))}}};
      VkPipelineVertexInputStateCreateInfo vertexInput{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
      vertexInput.vertexBindingDescriptionCount = 1;
      vertexInput.pVertexBindingDescriptions = &binding;
      vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
      vertexInput.pVertexAttributeDescriptions = attributes.data();
      VkPipelineInputAssemblyStateCreateInfo assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
      assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
      viewport.viewportCount = 1;
      viewport.scissorCount = 1;
      VkPipelineRasterizationStateCreateInfo raster{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
      raster.polygonMode = VK_POLYGON_MODE_FILL;
      raster.cullMode = VK_CULL_MODE_NONE;
      raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
      raster.depthBiasEnable = VK_TRUE;
      raster.lineWidth = 1.0f;
      VkPipelineMultisampleStateCreateInfo multisample{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
      multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
      VkPipelineDepthStencilStateCreateInfo depth{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
      depth.depthTestEnable = VK_TRUE;
      depth.depthWriteEnable = VK_TRUE;
      depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
      constexpr std::array dynamicStates{
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS};
      VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
      dynamic.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
      dynamic.pDynamicStates = dynamicStates.data();
      VkPushConstantRange push{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ScenePushConstants)};
      VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      layout.setLayoutCount = 1;
      layout.pSetLayouts = &s_Data.SceneDescriptorSetLayout;
      layout.pushConstantRangeCount = 1;
      layout.pPushConstantRanges = &push;
      Check(vkCreatePipelineLayout(s_Data.Device, &layout, nullptr,
                                   &s_Data.ShadowPipelineLayout),
            "vkCreatePipelineLayout (shadow)");
      VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
      pipeline.stageCount = 1;
      pipeline.pStages = &stage;
      pipeline.pVertexInputState = &vertexInput;
      pipeline.pInputAssemblyState = &assembly;
      pipeline.pViewportState = &viewport;
      pipeline.pRasterizationState = &raster;
      pipeline.pMultisampleState = &multisample;
      pipeline.pDepthStencilState = &depth;
      pipeline.pDynamicState = &dynamic;
      pipeline.layout = s_Data.ShadowPipelineLayout;
      pipeline.renderPass = s_Data.ShadowRenderPass;
      Check(vkCreateGraphicsPipelines(s_Data.Device, VK_NULL_HANDLE, 1, &pipeline,
                                      nullptr, &s_Data.ShadowPipeline),
            "vkCreateGraphicsPipelines (shadow)");
    }
    catch (...)
    {
      vkDestroyShaderModule(s_Data.Device, vertex, nullptr);
      throw;
    }
    vkDestroyShaderModule(s_Data.Device, vertex, nullptr);
  }

  void CreatePointShadowRenderPass()
  {
    VkAttachmentDescription depth{};
    depth.format = s_Data.DepthFormat;
    depth.samples = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    VkAttachmentReference depthReference{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.pDepthStencilAttachment = &depthReference;
    const std::array<VkSubpassDependency, 2> dependencies{{
      {VK_SUBPASS_EXTERNAL, 0, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_SHADER_READ_BIT,
       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_DEPENDENCY_BY_REGION_BIT},
      {0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
       VK_ACCESS_SHADER_READ_BIT, VK_DEPENDENCY_BY_REGION_BIT}}};
    VkRenderPassCreateInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    info.attachmentCount = 1;
    info.pAttachments = &depth;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    info.dependencyCount = static_cast<uint32_t>(dependencies.size());
    info.pDependencies = dependencies.data();
    Check(vkCreateRenderPass(s_Data.Device, &info, nullptr,
                             &s_Data.PointShadowRenderPass),
          "vkCreateRenderPass (point shadow)");
  }

  void DestroyPointShadowAttachments()
  {
    for (PointShadowFrame& frame : s_Data.PointShadows)
      for (PointShadowCube& cube : frame.Cubes)
      {
        for (VkFramebuffer& framebuffer : cube.Framebuffers)
        {
          if (framebuffer) vkDestroyFramebuffer(s_Data.Device, framebuffer, nullptr);
          framebuffer = VK_NULL_HANDLE;
        }
        for (VkImageView& view : cube.FaceViews)
        {
          if (view) vkDestroyImageView(s_Data.Device, view, nullptr);
          view = VK_NULL_HANDLE;
        }
        DestroyTexture(cube.Texture);
      }
    for (auto& frameCache : s_Data.PointShadowCache)
      for (PointShadowCacheEntry& cache : frameCache) cache = {};
    s_Data.PointShadowSize = 0;
  }

  void CreatePointShadowAttachments(const uint32_t size)
  {
    DestroyPointShadowAttachments();
    std::vector<VkImageMemoryBarrier> barriers;
    barriers.reserve(FramesInFlight * MaxPointShadowLights);
    for (PointShadowFrame& frame : s_Data.PointShadows)
      for (PointShadowCube& cube : frame.Cubes)
      {
        VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        image.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        image.imageType = VK_IMAGE_TYPE_2D;
        image.extent = {size, size, 1};
        image.mipLevels = 1;
        image.arrayLayers = 6;
        image.format = s_Data.DepthFormat;
        image.tiling = VK_IMAGE_TILING_OPTIMAL;
        image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        image.samples = VK_SAMPLE_COUNT_1_BIT;
        image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        Check(vkCreateImage(s_Data.Device, &image, nullptr, &cube.Texture.Image),
              "vkCreateImage (point shadow cube)");
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(s_Data.Device, cube.Texture.Image, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = FindMemoryType(
          requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Check(vkAllocateMemory(s_Data.Device, &allocation, nullptr, &cube.Texture.Memory),
              "vkAllocateMemory (point shadow cube)");
        Check(vkBindImageMemory(s_Data.Device, cube.Texture.Image, cube.Texture.Memory, 0),
              "vkBindImageMemory (point shadow cube)");
        VkImageViewCreateInfo cubeView{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        cubeView.image = cube.Texture.Image;
        cubeView.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        cubeView.format = s_Data.DepthFormat;
        cubeView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        cubeView.subresourceRange.levelCount = 1;
        cubeView.subresourceRange.layerCount = 6;
        Check(vkCreateImageView(s_Data.Device, &cubeView, nullptr, &cube.Texture.View),
              "vkCreateImageView (point shadow cube)");
        for (uint32_t face = 0; face < 6; ++face)
        {
          VkImageViewCreateInfo faceView = cubeView;
          faceView.viewType = VK_IMAGE_VIEW_TYPE_2D;
          faceView.subresourceRange.baseArrayLayer = face;
          faceView.subresourceRange.layerCount = 1;
          Check(vkCreateImageView(s_Data.Device, &faceView, nullptr,
                                  &cube.FaceViews[face]),
                "vkCreateImageView (point shadow face)");
          VkFramebufferCreateInfo framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
          framebuffer.renderPass = s_Data.PointShadowRenderPass;
          framebuffer.attachmentCount = 1;
          framebuffer.pAttachments = &cube.FaceViews[face];
          framebuffer.width = size;
          framebuffer.height = size;
          framebuffer.layers = 1;
          Check(vkCreateFramebuffer(s_Data.Device, &framebuffer, nullptr,
                                    &cube.Framebuffers[face]),
                "vkCreateFramebuffer (point shadow face)");
        }
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = cube.Texture.Image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 6;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barriers.push_back(barrier);
      }
    const VkCommandBuffer command = BeginImmediateCommands();
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                         static_cast<uint32_t>(barriers.size()), barriers.data());
    EndImmediateCommands(command);
    s_Data.PointShadowSize = size;
  }

  void CreatePointShadowPipeline()
  {
    std::filesystem::path shaderPath = "../res/shaders/vulkan_shadow.slang";
    if (!std::filesystem::exists(shaderPath)) shaderPath = "res/shaders/vulkan_shadow.slang";
    const VkShaderModule vertex = CreateShaderModule(
      Shader::CompileSlangSPIRV(shaderPath, "VSMain", "vertex"));
    try
    {
      VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
      stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
      stage.module = vertex;
      stage.pName = "main";
      const VkVertexInputBindingDescription binding{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
      const std::array<VkVertexInputAttributeDescription, 3> attributes{{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, Position))},
        {1, 0, VK_FORMAT_R32G32B32A32_SINT, static_cast<uint32_t>(offsetof(Vertex, m_BoneIDs))},
        {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, m_Weights))}}};
      VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
      vertexInput.vertexBindingDescriptionCount = 1;
      vertexInput.pVertexBindingDescriptions = &binding;
      vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
      vertexInput.pVertexAttributeDescriptions = attributes.data();
      VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
      assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
      viewport.viewportCount = 1;
      viewport.scissorCount = 1;
      VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
      raster.polygonMode = VK_POLYGON_MODE_FILL;
      raster.cullMode = VK_CULL_MODE_NONE;
      raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
      raster.depthBiasEnable = VK_TRUE;
      raster.lineWidth = 1.0f;
      VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
      multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
      VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
      depth.depthTestEnable = VK_TRUE;
      depth.depthWriteEnable = VK_TRUE;
      depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
      constexpr std::array dynamicStates{
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS};
      VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
      dynamic.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
      dynamic.pDynamicStates = dynamicStates.data();
      VkPushConstantRange push{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ScenePushConstants)};
      VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      layout.setLayoutCount = 1;
      layout.pSetLayouts = &s_Data.SceneDescriptorSetLayout;
      layout.pushConstantRangeCount = 1;
      layout.pPushConstantRanges = &push;
      Check(vkCreatePipelineLayout(s_Data.Device, &layout, nullptr,
                                   &s_Data.PointShadowPipelineLayout),
            "vkCreatePipelineLayout (point shadow)");
      VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
      pipeline.stageCount = 1;
      pipeline.pStages = &stage;
      pipeline.pVertexInputState = &vertexInput;
      pipeline.pInputAssemblyState = &assembly;
      pipeline.pViewportState = &viewport;
      pipeline.pRasterizationState = &raster;
      pipeline.pMultisampleState = &multisample;
      pipeline.pDepthStencilState = &depth;
      pipeline.pDynamicState = &dynamic;
      pipeline.layout = s_Data.PointShadowPipelineLayout;
      pipeline.renderPass = s_Data.PointShadowRenderPass;
      Check(vkCreateGraphicsPipelines(s_Data.Device, VK_NULL_HANDLE, 1, &pipeline,
                                      nullptr, &s_Data.PointShadowPipeline),
            "vkCreateGraphicsPipelines (point shadow)");
    }
    catch (...)
    {
      vkDestroyShaderModule(s_Data.Device, vertex, nullptr);
      throw;
    }
    vkDestroyShaderModule(s_Data.Device, vertex, nullptr);
  }

  void CreateParticlePipeline()
  {
    std::filesystem::path shaderPath = "../res/shaders/vulkan_particle.slang";
    if (!std::filesystem::exists(shaderPath)) shaderPath = "res/shaders/vulkan_particle.slang";
    const VkShaderModule vertex = CreateShaderModule(
      Shader::CompileSlangSPIRV(shaderPath, "VSMain", "vertex"));
    const VkShaderModule fragment = CreateShaderModule(
      Shader::CompileSlangSPIRV(shaderPath, "PSMain", "fragment"));
    try
    {
      const VkPipelineShaderStageCreateInfo stages[]{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_VERTEX_BIT, vertex, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_FRAGMENT_BIT, fragment, "main", nullptr}};
      const VkVertexInputBindingDescription binding{
        0, sizeof(ParticleVertex), VK_VERTEX_INPUT_RATE_VERTEX};
      const std::array<VkVertexInputAttributeDescription, 4> attributes{{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT,
         static_cast<uint32_t>(offsetof(ParticleVertex, Position))},
        {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT,
         static_cast<uint32_t>(offsetof(ParticleVertex, Color))},
        {2, 0, VK_FORMAT_R32G32_SFLOAT,
         static_cast<uint32_t>(offsetof(ParticleVertex, LocalPosition))},
        {3, 0, VK_FORMAT_R32_SFLOAT,
         static_cast<uint32_t>(offsetof(ParticleVertex, IsSquare))}}};
      VkPipelineVertexInputStateCreateInfo vertexInput{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
      vertexInput.vertexBindingDescriptionCount = 1;
      vertexInput.pVertexBindingDescriptions = &binding;
      vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
      vertexInput.pVertexAttributeDescriptions = attributes.data();
      VkPipelineInputAssemblyStateCreateInfo assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
      assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
      viewport.viewportCount = 1;
      viewport.scissorCount = 1;
      VkPipelineRasterizationStateCreateInfo raster{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
      raster.polygonMode = VK_POLYGON_MODE_FILL;
      raster.cullMode = VK_CULL_MODE_NONE;
      raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
      raster.lineWidth = 1.0f;
      VkPipelineMultisampleStateCreateInfo multisample{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
      multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
      VkPipelineDepthStencilStateCreateInfo depth{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
      depth.depthTestEnable = VK_TRUE;
      depth.depthWriteEnable = VK_FALSE;
      depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
      VkPipelineColorBlendAttachmentState blend{};
      blend.blendEnable = VK_TRUE;
      blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      blend.colorBlendOp = VK_BLEND_OP_ADD;
      blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      blend.alphaBlendOp = VK_BLEND_OP_ADD;
      blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      VkPipelineColorBlendStateCreateInfo blending{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
      blending.attachmentCount = 1;
      blending.pAttachments = &blend;
      constexpr std::array dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
      VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
      dynamic.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
      dynamic.pDynamicStates = dynamicStates.data();
      VkPushConstantRange push{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4)};
      VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      layout.pushConstantRangeCount = 1;
      layout.pPushConstantRanges = &push;
      Check(vkCreatePipelineLayout(s_Data.Device, &layout, nullptr,
                                   &s_Data.ParticlePipelineLayout),
            "vkCreatePipelineLayout (particles)");
      VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
      pipeline.stageCount = 2;
      pipeline.pStages = stages;
      pipeline.pVertexInputState = &vertexInput;
      pipeline.pInputAssemblyState = &assembly;
      pipeline.pViewportState = &viewport;
      pipeline.pRasterizationState = &raster;
      pipeline.pMultisampleState = &multisample;
      pipeline.pDepthStencilState = &depth;
      pipeline.pColorBlendState = &blending;
      pipeline.pDynamicState = &dynamic;
      pipeline.layout = s_Data.ParticlePipelineLayout;
      pipeline.renderPass = s_Data.ParticleRenderPass;
      Check(vkCreateGraphicsPipelines(s_Data.Device, VK_NULL_HANDLE, 1, &pipeline,
                                      nullptr, &s_Data.ParticlePipeline),
            "vkCreateGraphicsPipelines (particles)");
    }
    catch (...)
    {
      vkDestroyShaderModule(s_Data.Device, fragment, nullptr);
      vkDestroyShaderModule(s_Data.Device, vertex, nullptr);
      throw;
    }
    vkDestroyShaderModule(s_Data.Device, fragment, nullptr);
    vkDestroyShaderModule(s_Data.Device, vertex, nullptr);
  }

  void CreateDebugLinePipeline()
  {
    std::filesystem::path shaderPath = "../res/shaders/vulkan_debug_line.slang";
    if (!std::filesystem::exists(shaderPath)) shaderPath = "res/shaders/vulkan_debug_line.slang";
    const VkShaderModule vertex = CreateShaderModule(
      Shader::CompileSlangSPIRV(shaderPath, "VSMain", "vertex"));
    const VkShaderModule fragment = CreateShaderModule(
      Shader::CompileSlangSPIRV(shaderPath, "PSMain", "fragment"));
    try
    {
      const VkPipelineShaderStageCreateInfo stages[]{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_VERTEX_BIT, vertex, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_FRAGMENT_BIT, fragment, "main", nullptr}};
      const VkVertexInputBindingDescription binding{
        0, sizeof(DebugLineVertex), VK_VERTEX_INPUT_RATE_VERTEX};
      const std::array<VkVertexInputAttributeDescription, 2> attributes{{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT,
         static_cast<uint32_t>(offsetof(DebugLineVertex, Position))},
        {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT,
         static_cast<uint32_t>(offsetof(DebugLineVertex, Color))}}};
      VkPipelineVertexInputStateCreateInfo vertexInput{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
      vertexInput.vertexBindingDescriptionCount = 1;
      vertexInput.pVertexBindingDescriptions = &binding;
      vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
      vertexInput.pVertexAttributeDescriptions = attributes.data();
      VkPipelineInputAssemblyStateCreateInfo assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
      assembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
      VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
      viewport.viewportCount = 1;
      viewport.scissorCount = 1;
      VkPipelineRasterizationStateCreateInfo raster{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
      raster.polygonMode = VK_POLYGON_MODE_FILL;
      raster.cullMode = VK_CULL_MODE_NONE;
      raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
      raster.lineWidth = 1.0f;
      VkPipelineMultisampleStateCreateInfo multisample{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
      multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
      VkPipelineDepthStencilStateCreateInfo depth{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
      depth.depthTestEnable = VK_TRUE;
      depth.depthWriteEnable = VK_FALSE;
      depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
      VkPipelineColorBlendAttachmentState blend{};
      blend.blendEnable = VK_TRUE;
      blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      blend.colorBlendOp = VK_BLEND_OP_ADD;
      blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      blend.alphaBlendOp = VK_BLEND_OP_ADD;
      blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      VkPipelineColorBlendStateCreateInfo blending{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
      blending.attachmentCount = 1;
      blending.pAttachments = &blend;
      constexpr std::array dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
      VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
      dynamic.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
      dynamic.pDynamicStates = dynamicStates.data();
      VkPushConstantRange push{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4)};
      VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      layout.pushConstantRangeCount = 1;
      layout.pPushConstantRanges = &push;
      Check(vkCreatePipelineLayout(s_Data.Device, &layout, nullptr,
                                   &s_Data.DebugLinePipelineLayout),
            "vkCreatePipelineLayout (debug lines)");
      VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
      pipeline.stageCount = 2;
      pipeline.pStages = stages;
      pipeline.pVertexInputState = &vertexInput;
      pipeline.pInputAssemblyState = &assembly;
      pipeline.pViewportState = &viewport;
      pipeline.pRasterizationState = &raster;
      pipeline.pMultisampleState = &multisample;
      pipeline.pDepthStencilState = &depth;
      pipeline.pColorBlendState = &blending;
      pipeline.pDynamicState = &dynamic;
      pipeline.layout = s_Data.DebugLinePipelineLayout;
      pipeline.renderPass = s_Data.ParticleRenderPass;
      Check(vkCreateGraphicsPipelines(s_Data.Device, VK_NULL_HANDLE, 1, &pipeline,
                                      nullptr, &s_Data.DebugLinePipeline),
            "vkCreateGraphicsPipelines (debug lines)");
    }
    catch (...)
    {
      vkDestroyShaderModule(s_Data.Device, fragment, nullptr);
      vkDestroyShaderModule(s_Data.Device, vertex, nullptr);
      throw;
    }
    vkDestroyShaderModule(s_Data.Device, fragment, nullptr);
    vkDestroyShaderModule(s_Data.Device, vertex, nullptr);
  }

  void AllocatePostDescriptors()
  {
    if (!s_Data.PostDescriptorPool) return;
    s_Data.PostDescriptors.assign(s_Data.SceneColors.size(), VK_NULL_HANDLE);
    std::vector<VkDescriptorSetLayout> layouts(
      s_Data.PostDescriptors.size(), s_Data.PostDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = s_Data.PostDescriptorPool;
    allocation.descriptorSetCount = static_cast<uint32_t>(layouts.size());
    allocation.pSetLayouts = layouts.data();
    Check(vkAllocateDescriptorSets(s_Data.Device, &allocation,
                                   s_Data.PostDescriptors.data()),
          "vkAllocateDescriptorSets (post)");
    for (size_t index = 0; index < s_Data.PostDescriptors.size(); ++index)
    {
      VkDescriptorImageInfo image{};
      image.imageView = s_Data.SceneColors[index].View;
      image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      VkDescriptorImageInfo sampler{};
      sampler.sampler = s_Data.PostSampler;
      VkDescriptorImageInfo bloom{};
      bloom.imageView = s_Data.BloomPyramids[index].View;
      bloom.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      const std::array<VkWriteDescriptorSet, 3> writes{{
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, s_Data.PostDescriptors[index],
         0, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &image, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, s_Data.PostDescriptors[index],
         1, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLER, &sampler, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, s_Data.PostDescriptors[index],
         2, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &bloom, nullptr, nullptr}}};
      vkUpdateDescriptorSets(s_Data.Device, static_cast<uint32_t>(writes.size()),
                             writes.data(), 0, nullptr);
    }
  }

  void CreatePostPipeline()
  {
    const std::array<VkDescriptorSetLayoutBinding, 3> bindings{{
      {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}}};
    VkDescriptorSetLayoutCreateInfo setLayout{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    setLayout.bindingCount = static_cast<uint32_t>(bindings.size());
    setLayout.pBindings = bindings.data();
    Check(vkCreateDescriptorSetLayout(s_Data.Device, &setLayout, nullptr,
                                      &s_Data.PostDescriptorSetLayout),
          "vkCreateDescriptorSetLayout (post)");
    const uint32_t imageCount = static_cast<uint32_t>(s_Data.Images.size());
    const std::array<VkDescriptorPoolSize, 2> sizes{{
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, std::max(imageCount * 2, 16u)},
      {VK_DESCRIPTOR_TYPE_SAMPLER, std::max(imageCount, 8u)}}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets = std::max(imageCount, 8u);
    pool.poolSizeCount = static_cast<uint32_t>(sizes.size());
    pool.pPoolSizes = sizes.data();
    Check(vkCreateDescriptorPool(s_Data.Device, &pool, nullptr, &s_Data.PostDescriptorPool),
          "vkCreateDescriptorPool (post)");
    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.magFilter = VK_FILTER_LINEAR;
    sampler.minFilter = VK_FILTER_LINEAR;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.maxLod = static_cast<float>(MaxHiZMips - 1);
    Check(vkCreateSampler(s_Data.Device, &sampler, nullptr, &s_Data.PostSampler),
          "vkCreateSampler (post)");
    AllocatePostDescriptors();

    std::filesystem::path shaderPath = "../res/shaders/vulkan_post.slang";
    if (!std::filesystem::exists(shaderPath)) shaderPath = "res/shaders/vulkan_post.slang";
    const VkShaderModule vertex = CreateShaderModule(
      Shader::CompileSlangSPIRV(shaderPath, "VSMain", "vertex"));
    const VkShaderModule fragment = CreateShaderModule(
      Shader::CompileSlangSPIRV(shaderPath, "PSMain", "fragment"));
    try
    {
      const VkPipelineShaderStageCreateInfo stages[]{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_VERTEX_BIT, vertex, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_FRAGMENT_BIT, fragment, "main", nullptr}};
      VkPipelineVertexInputStateCreateInfo vertexInput{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
      const VkVertexInputBindingDescription binding{
        0, sizeof(glm::vec2), VK_VERTEX_INPUT_RATE_VERTEX};
      const VkVertexInputAttributeDescription attribute{
        0, 0, VK_FORMAT_R32G32_SFLOAT, 0};
      vertexInput.vertexBindingDescriptionCount = 1;
      vertexInput.pVertexBindingDescriptions = &binding;
      vertexInput.vertexAttributeDescriptionCount = 1;
      vertexInput.pVertexAttributeDescriptions = &attribute;
      VkPipelineInputAssemblyStateCreateInfo assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
      assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
      viewport.viewportCount = 1;
      viewport.scissorCount = 1;
      VkPipelineRasterizationStateCreateInfo raster{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
      raster.polygonMode = VK_POLYGON_MODE_FILL;
      raster.cullMode = VK_CULL_MODE_NONE;
      raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
      raster.lineWidth = 1.0f;
      VkPipelineMultisampleStateCreateInfo multisample{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
      multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
      VkPipelineColorBlendAttachmentState blend{};
      blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      VkPipelineColorBlendStateCreateInfo blending{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
      blending.attachmentCount = 1;
      blending.pAttachments = &blend;
      constexpr std::array dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
      VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
      dynamic.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
      dynamic.pDynamicStates = dynamicStates.data();
      VkPushConstantRange push{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PostPushConstants)};
      VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      layout.setLayoutCount = 1;
      layout.pSetLayouts = &s_Data.PostDescriptorSetLayout;
      layout.pushConstantRangeCount = 1;
      layout.pPushConstantRanges = &push;
      Check(vkCreatePipelineLayout(s_Data.Device, &layout, nullptr,
                                   &s_Data.PostPipelineLayout),
            "vkCreatePipelineLayout (post)");
      VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
      pipeline.stageCount = 2;
      pipeline.pStages = stages;
      pipeline.pVertexInputState = &vertexInput;
      pipeline.pInputAssemblyState = &assembly;
      pipeline.pViewportState = &viewport;
      pipeline.pRasterizationState = &raster;
      pipeline.pMultisampleState = &multisample;
      pipeline.pColorBlendState = &blending;
      pipeline.pDynamicState = &dynamic;
      pipeline.layout = s_Data.PostPipelineLayout;
      pipeline.renderPass = s_Data.PresentRenderPass;
      Check(vkCreateGraphicsPipelines(s_Data.Device, VK_NULL_HANDLE, 1, &pipeline,
                                      nullptr, &s_Data.PostPipeline),
            "vkCreateGraphicsPipelines (post)");
    }
    catch (...)
    {
      vkDestroyShaderModule(s_Data.Device, fragment, nullptr);
      vkDestroyShaderModule(s_Data.Device, vertex, nullptr);
      throw;
    }
    vkDestroyShaderModule(s_Data.Device, fragment, nullptr);
    vkDestroyShaderModule(s_Data.Device, vertex, nullptr);
  }

  void AllocateLightingDescriptors()
  {
    if (!s_Data.LightingDescriptorPool) return;
    s_Data.LightingDescriptors.resize(s_Data.GBufferAlbedo.size());
    for (size_t imageIndex = 0; imageIndex < s_Data.LightingDescriptors.size(); ++imageIndex)
      for (uint32_t frame = 0; frame < FramesInFlight; ++frame)
      {
        VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocation.descriptorPool = s_Data.LightingDescriptorPool;
        allocation.descriptorSetCount = 1;
        allocation.pSetLayouts = &s_Data.LightingDescriptorSetLayout;
        VkDescriptorSet& descriptor = s_Data.LightingDescriptors[imageIndex][frame];
        Check(vkAllocateDescriptorSets(s_Data.Device, &allocation, &descriptor),
              "vkAllocateDescriptorSets (lighting)");
        const std::array<VkDescriptorImageInfo, 3> gbuffer{{
          {VK_NULL_HANDLE, s_Data.GBufferAlbedo[imageIndex].View,
           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
          {VK_NULL_HANDLE, s_Data.GBufferNormal[imageIndex].View,
           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
          {VK_NULL_HANDLE, s_Data.GBufferPosition[imageIndex].View,
           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}}};
        VkDescriptorImageInfo materialSampler{};
        materialSampler.sampler = s_Data.PostSampler;
        VkDescriptorBufferInfo sceneFrame{};
        sceneFrame.buffer = s_Data.SceneFrames[frame].Buffer;
        sceneFrame.range = sizeof(SceneFrameData);
        VkDescriptorImageInfo shadowImage{};
        shadowImage.imageView = s_Data.Shadows[frame].Texture.View;
        shadowImage.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo shadowSampler{};
        shadowSampler.sampler = s_Data.ShadowSampler;
        VkDescriptorImageInfo skyboxImage{};
        skyboxImage.imageView = s_Data.SkyboxTexture.View;
        skyboxImage.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        std::array<VkDescriptorImageInfo, MaxPointShadowLights> pointShadowImages{};
        for (uint32_t shadow = 0; shadow < MaxPointShadowLights; ++shadow)
        {
          pointShadowImages[shadow].imageView =
            s_Data.PointShadows[frame].Cubes[shadow].Texture.View;
          pointShadowImages[shadow].imageLayout =
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        }
        VkDescriptorImageInfo pointShadowSampler{};
        pointShadowSampler.sampler = s_Data.ShadowSampler;
        VkDescriptorBufferInfo tileGrid{};
        tileGrid.buffer = s_Data.TileGrids[imageIndex].Buffer;
        tileGrid.range = VK_WHOLE_SIZE;
        VkDescriptorBufferInfo tileIndices{};
        tileIndices.buffer = s_Data.TileIndices[imageIndex].Buffer;
        tileIndices.range = VK_WHOLE_SIZE;
        std::array<VkWriteDescriptorSet, 15> writes{};
        for (uint32_t binding = 0; binding < 3; ++binding)
        {
          writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
          writes[binding].dstSet = descriptor;
          writes[binding].dstBinding = binding;
          writes[binding].descriptorCount = 1;
          writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
          writes[binding].pImageInfo = &gbuffer[binding];
        }
        writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor,
          3, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLER, &materialSampler, nullptr, nullptr};
        writes[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor,
          4, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &sceneFrame, nullptr};
        writes[5] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor,
          5, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &shadowImage, nullptr, nullptr};
        writes[6] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor,
          6, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLER, &shadowSampler, nullptr, nullptr};
        writes[7] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor,
          7, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &skyboxImage, nullptr, nullptr};
        for (uint32_t shadow = 0; shadow < MaxPointShadowLights; ++shadow)
          writes[8 + shadow] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor,
            8 + shadow, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            &pointShadowImages[shadow], nullptr, nullptr};
        writes[12] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor,
          12, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLER, &pointShadowSampler, nullptr, nullptr};
        writes[13] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor,
          13, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &tileGrid, nullptr};
        writes[14] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor,
          14, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &tileIndices, nullptr};
        vkUpdateDescriptorSets(s_Data.Device, static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
      }
  }

  void CreateLightingPipeline()
  {
    const std::array<VkDescriptorSetLayoutBinding, 15> bindings{{
      {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {3, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {6, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {7, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {8, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {9, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {10, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {11, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {12, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
      {14, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}}};
    VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout.bindingCount = static_cast<uint32_t>(bindings.size());
    layout.pBindings = bindings.data();
    Check(vkCreateDescriptorSetLayout(s_Data.Device, &layout, nullptr,
                                      &s_Data.LightingDescriptorSetLayout),
          "vkCreateDescriptorSetLayout (lighting)");
    constexpr uint32_t MaxLightingSets = 16;
    const std::array<VkDescriptorPoolSize, 4> sizes{{
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, MaxLightingSets * 9},
      {VK_DESCRIPTOR_TYPE_SAMPLER, MaxLightingSets * 3},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MaxLightingSets},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MaxLightingSets * 2}}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets = MaxLightingSets;
    pool.poolSizeCount = static_cast<uint32_t>(sizes.size());
    pool.pPoolSizes = sizes.data();
    Check(vkCreateDescriptorPool(s_Data.Device, &pool, nullptr,
                                 &s_Data.LightingDescriptorPool),
          "vkCreateDescriptorPool (lighting)");
    AllocateLightingDescriptors();

    std::filesystem::path shaderPath = "../res/shaders/vulkan_lighting.slang";
    if (!std::filesystem::exists(shaderPath)) shaderPath = "res/shaders/vulkan_lighting.slang";
    const VkShaderModule vertex = CreateShaderModule(
      Shader::CompileSlangSPIRV(shaderPath, "VSMain", "vertex"));
    const VkShaderModule fragment = CreateShaderModule(
      Shader::CompileSlangSPIRV(shaderPath, "PSMain", "fragment"));
    try
    {
      const VkPipelineShaderStageCreateInfo stages[]{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_VERTEX_BIT, vertex, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
         VK_SHADER_STAGE_FRAGMENT_BIT, fragment, "main", nullptr}};
      const VkVertexInputBindingDescription binding{
        0, sizeof(glm::vec2), VK_VERTEX_INPUT_RATE_VERTEX};
      const VkVertexInputAttributeDescription attribute{0, 0, VK_FORMAT_R32G32_SFLOAT, 0};
      VkPipelineVertexInputStateCreateInfo vertexInput{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
      vertexInput.vertexBindingDescriptionCount = 1;
      vertexInput.pVertexBindingDescriptions = &binding;
      vertexInput.vertexAttributeDescriptionCount = 1;
      vertexInput.pVertexAttributeDescriptions = &attribute;
      VkPipelineInputAssemblyStateCreateInfo assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
      assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
      VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
      viewport.viewportCount = 1;
      viewport.scissorCount = 1;
      VkPipelineRasterizationStateCreateInfo raster{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
      raster.polygonMode = VK_POLYGON_MODE_FILL;
      raster.cullMode = VK_CULL_MODE_NONE;
      raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
      raster.lineWidth = 1.0f;
      VkPipelineMultisampleStateCreateInfo multisample{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
      multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
      VkPipelineColorBlendAttachmentState blend{};
      blend.blendEnable = VK_TRUE;
      blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      blend.colorBlendOp = VK_BLEND_OP_ADD;
      blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      blend.alphaBlendOp = VK_BLEND_OP_ADD;
      blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      VkPipelineColorBlendStateCreateInfo blending{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
      blending.attachmentCount = 1;
      blending.pAttachments = &blend;
      constexpr std::array dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
      VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
      dynamic.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
      dynamic.pDynamicStates = dynamicStates.data();
      VkPipelineLayoutCreateInfo pipelineLayout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      pipelineLayout.setLayoutCount = 1;
      pipelineLayout.pSetLayouts = &s_Data.LightingDescriptorSetLayout;
      Check(vkCreatePipelineLayout(s_Data.Device, &pipelineLayout, nullptr,
                                   &s_Data.LightingPipelineLayout),
            "vkCreatePipelineLayout (lighting)");
      VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
      pipeline.stageCount = 2;
      pipeline.pStages = stages;
      pipeline.pVertexInputState = &vertexInput;
      pipeline.pInputAssemblyState = &assembly;
      pipeline.pViewportState = &viewport;
      pipeline.pRasterizationState = &raster;
      pipeline.pMultisampleState = &multisample;
      pipeline.pColorBlendState = &blending;
      pipeline.pDynamicState = &dynamic;
      pipeline.layout = s_Data.LightingPipelineLayout;
      pipeline.renderPass = s_Data.LightingRenderPass;
      Check(vkCreateGraphicsPipelines(s_Data.Device, VK_NULL_HANDLE, 1, &pipeline,
                                      nullptr, &s_Data.LightingPipeline),
            "vkCreateGraphicsPipelines (lighting)");
    }
    catch (...)
    {
      vkDestroyShaderModule(s_Data.Device, fragment, nullptr);
      vkDestroyShaderModule(s_Data.Device, vertex, nullptr);
      throw;
    }
    vkDestroyShaderModule(s_Data.Device, fragment, nullptr);
    vkDestroyShaderModule(s_Data.Device, vertex, nullptr);
  }

  void AllocateTileDescriptors()
  {
    if (!s_Data.TileDescriptorPool) return;
    s_Data.TileDescriptors.resize(s_Data.GBufferPosition.size());
    for (size_t imageIndex = 0; imageIndex < s_Data.TileDescriptors.size(); ++imageIndex)
      for (uint32_t frame = 0; frame < FramesInFlight; ++frame)
      {
        VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocation.descriptorPool = s_Data.TileDescriptorPool;
        allocation.descriptorSetCount = 1;
        allocation.pSetLayouts = &s_Data.TileDescriptorSetLayout;
        VkDescriptorSet& descriptor = s_Data.TileDescriptors[imageIndex][frame];
        Check(vkAllocateDescriptorSets(s_Data.Device, &allocation, &descriptor),
              "vkAllocateDescriptorSets (tile lights)");
        VkDescriptorImageInfo position{};
        position.imageView = s_Data.GBufferPosition[imageIndex].View;
        position.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorImageInfo sampler{};
        sampler.sampler = s_Data.PostSampler;
        VkDescriptorBufferInfo sceneFrame{};
        sceneFrame.buffer = s_Data.SceneFrames[frame].Buffer;
        sceneFrame.range = sizeof(SceneFrameData);
        VkDescriptorBufferInfo grid{};
        grid.buffer = s_Data.TileGrids[imageIndex].Buffer;
        grid.range = VK_WHOLE_SIZE;
        VkDescriptorBufferInfo indices{};
        indices.buffer = s_Data.TileIndices[imageIndex].Buffer;
        indices.range = VK_WHOLE_SIZE;
        const std::array<VkWriteDescriptorSet, 5> writes{{
          {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor, 0, 0, 1,
           VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &position, nullptr, nullptr},
          {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor, 1, 0, 1,
           VK_DESCRIPTOR_TYPE_SAMPLER, &sampler, nullptr, nullptr},
          {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor, 2, 0, 1,
           VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &sceneFrame, nullptr},
          {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor, 3, 0, 1,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &grid, nullptr},
          {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor, 4, 0, 1,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &indices, nullptr}}};
        vkUpdateDescriptorSets(s_Data.Device, static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
      }
  }

  void CreateTilePipeline()
  {
    const std::array<VkDescriptorSetLayoutBinding, 5> bindings{{
      {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}}};
    VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout.bindingCount = static_cast<uint32_t>(bindings.size());
    layout.pBindings = bindings.data();
    Check(vkCreateDescriptorSetLayout(s_Data.Device, &layout, nullptr,
                                      &s_Data.TileDescriptorSetLayout),
          "vkCreateDescriptorSetLayout (tile lights)");
    constexpr uint32_t MaxTileSets = 16;
    const std::array<VkDescriptorPoolSize, 4> sizes{{
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, MaxTileSets},
      {VK_DESCRIPTOR_TYPE_SAMPLER, MaxTileSets},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MaxTileSets},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MaxTileSets * 2}}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets = MaxTileSets;
    pool.poolSizeCount = static_cast<uint32_t>(sizes.size());
    pool.pPoolSizes = sizes.data();
    Check(vkCreateDescriptorPool(s_Data.Device, &pool, nullptr,
                                 &s_Data.TileDescriptorPool),
          "vkCreateDescriptorPool (tile lights)");
    AllocateTileDescriptors();
    std::filesystem::path shaderPath = "../res/shaders/vulkan_tiled_light.slang";
    if (!std::filesystem::exists(shaderPath)) shaderPath = "res/shaders/vulkan_tiled_light.slang";
    const VkShaderModule compute = CreateShaderModule(
      Shader::CompileSlangSPIRV(shaderPath, "CSMain", "compute"));
    try
    {
      VkPipelineLayoutCreateInfo pipelineLayout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      pipelineLayout.setLayoutCount = 1;
      pipelineLayout.pSetLayouts = &s_Data.TileDescriptorSetLayout;
      Check(vkCreatePipelineLayout(s_Data.Device, &pipelineLayout, nullptr,
                                   &s_Data.TilePipelineLayout),
            "vkCreatePipelineLayout (tile lights)");
      VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
      pipeline.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      pipeline.stage.module = compute;
      pipeline.stage.pName = "main";
      pipeline.layout = s_Data.TilePipelineLayout;
      Check(vkCreateComputePipelines(s_Data.Device, VK_NULL_HANDLE, 1, &pipeline,
                                     nullptr, &s_Data.TilePipeline),
            "vkCreateComputePipelines (tile lights)");
    }
    catch (...)
    {
      vkDestroyShaderModule(s_Data.Device, compute, nullptr);
      throw;
    }
    vkDestroyShaderModule(s_Data.Device, compute, nullptr);
  }

  void UpdateVisibleTransformDescriptors()
  {
    for (auto& [mesh, gpu] : s_Data.Meshes)
      for (uint32_t frame = 0; frame < FramesInFlight; ++frame)
      {
        VkDescriptorBufferInfo visible{};
        visible.buffer = s_Data.CullFrames[frame].VisibleTransforms.Buffer;
        visible.range = VK_WHOLE_SIZE;
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = gpu.Materials[frame];
        write.dstBinding = 8;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &visible;
        vkUpdateDescriptorSets(s_Data.Device, 1, &write, 0, nullptr);
      }
  }

  void AllocateCullDescriptors()
  {
    std::array<VkDescriptorSetLayout, FramesInFlight> layouts{};
    layouts.fill(s_Data.CullDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = s_Data.CullDescriptorPool;
    allocation.descriptorSetCount = FramesInFlight;
    allocation.pSetLayouts = layouts.data();
    Check(vkAllocateDescriptorSets(s_Data.Device, &allocation,
                                   s_Data.CullDescriptors.data()),
          "vkAllocateDescriptorSets (GPU cull)");
    for (uint32_t frame = 0; frame < FramesInFlight; ++frame)
    {
      const std::array<VkDescriptorBufferInfo, 4> buffers{{
        {s_Data.CullFrames[frame].SourceTransforms.Buffer, 0, VK_WHOLE_SIZE},
        {s_Data.CullFrames[frame].Inputs.Buffer, 0, VK_WHOLE_SIZE},
        {s_Data.CullFrames[frame].VisibleTransforms.Buffer, 0, VK_WHOLE_SIZE},
        {s_Data.CullFrames[frame].Commands.Buffer, 0, VK_WHOLE_SIZE}}};
      std::array<VkWriteDescriptorSet, 6> writes{};
      for (uint32_t binding = 0; binding < buffers.size(); ++binding)
      {
        writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[binding].dstSet = s_Data.CullDescriptors[frame];
        writes[binding].dstBinding = binding;
        writes[binding].descriptorCount = 1;
        writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[binding].pBufferInfo = &buffers[binding];
      }
      VkDescriptorImageInfo hizImage{};
      hizImage.imageView = s_Data.HiZ[frame].Texture.View;
      hizImage.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      writes[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
        s_Data.CullDescriptors[frame], 4, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        &hizImage, nullptr, nullptr};
      VkDescriptorImageInfo hizSampler{};
      hizSampler.sampler = s_Data.PostSampler;
      writes[5] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
        s_Data.CullDescriptors[frame], 5, 0, 1, VK_DESCRIPTOR_TYPE_SAMPLER,
        &hizSampler, nullptr, nullptr};
      vkUpdateDescriptorSets(s_Data.Device, static_cast<uint32_t>(writes.size()),
                             writes.data(), 0, nullptr);
    }
  }

  void ResizeCullBuffers(const size_t transformCapacity, const size_t drawCapacity)
  {
    Check(vkDeviceWaitIdle(s_Data.Device), "vkDeviceWaitIdle (GPU cull resize)");
    for (GPUCullFrame& frame : s_Data.CullFrames)
    {
      DestroyBuffer(frame.SourceTransforms);
      DestroyBuffer(frame.Inputs);
      DestroyBuffer(frame.VisibleTransforms);
      DestroyBuffer(frame.Commands);
    }
    s_Data.CullTransformCapacity = std::max<size_t>(transformCapacity, 1);
    s_Data.CullDrawCapacity = std::max<size_t>(drawCapacity, 1);
    const std::vector<glm::mat4> transforms(s_Data.CullTransformCapacity, glm::mat4(1.0f));
    const std::vector<CullInput> inputs(s_Data.CullDrawCapacity);
    const std::vector<VkDrawIndexedIndirectCommand> commands(s_Data.CullDrawCapacity);
    for (GPUCullFrame& frame : s_Data.CullFrames)
    {
      frame.SourceTransforms = CreateUploadBuffer(
        transforms.data(), transforms.size() * sizeof(glm::mat4),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      frame.Inputs = CreateUploadBuffer(
        inputs.data(), inputs.size() * sizeof(CullInput),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      frame.VisibleTransforms = CreateUploadBuffer(
        transforms.data(), transforms.size() * sizeof(glm::mat4),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      frame.Commands = CreateUploadBuffer(
        commands.data(), commands.size() * sizeof(VkDrawIndexedIndirectCommand),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    }
    Check(vkResetDescriptorPool(s_Data.Device, s_Data.CullDescriptorPool, 0),
          "vkResetDescriptorPool (GPU cull)");
    AllocateCullDescriptors();
    UpdateVisibleTransformDescriptors();
  }

  void CreateCullPipeline()
  {
    std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
    for (uint32_t binding = 0; binding < 4; ++binding)
    {
      bindings[binding].binding = binding;
      bindings[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[binding].descriptorCount = 1;
      bindings[binding].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    bindings[4] = {4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                   VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[5] = {5, VK_DESCRIPTOR_TYPE_SAMPLER, 1,
                   VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout.bindingCount = static_cast<uint32_t>(bindings.size());
    layout.pBindings = bindings.data();
    Check(vkCreateDescriptorSetLayout(s_Data.Device, &layout, nullptr,
                                      &s_Data.CullDescriptorSetLayout),
          "vkCreateDescriptorSetLayout (GPU cull)");
    const std::array<VkDescriptorPoolSize, 3> sizes{{
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, FramesInFlight * 4},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, FramesInFlight},
      {VK_DESCRIPTOR_TYPE_SAMPLER, FramesInFlight}}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets = FramesInFlight;
    pool.poolSizeCount = static_cast<uint32_t>(sizes.size());
    pool.pPoolSizes = sizes.data();
    Check(vkCreateDescriptorPool(s_Data.Device, &pool, nullptr,
                                 &s_Data.CullDescriptorPool),
          "vkCreateDescriptorPool (GPU cull)");
    const std::vector<glm::mat4> transforms(1, glm::mat4(1.0f));
    const std::vector<CullInput> inputs(1);
    const std::vector<VkDrawIndexedIndirectCommand> commands(1);
    for (GPUCullFrame& frame : s_Data.CullFrames)
    {
      frame.SourceTransforms = CreateUploadBuffer(transforms.data(), sizeof(glm::mat4),
                                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      frame.Inputs = CreateUploadBuffer(inputs.data(), sizeof(CullInput),
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      frame.VisibleTransforms = CreateUploadBuffer(transforms.data(), sizeof(glm::mat4),
                                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      frame.Commands = CreateUploadBuffer(commands.data(), sizeof(VkDrawIndexedIndirectCommand),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    }
    s_Data.CullTransformCapacity = 1;
    s_Data.CullDrawCapacity = 1;
    AllocateCullDescriptors();
    std::filesystem::path shaderPath = "../res/shaders/vulkan_gpu_cull.slang";
    if (!std::filesystem::exists(shaderPath)) shaderPath = "res/shaders/vulkan_gpu_cull.slang";
    const VkShaderModule compute = CreateShaderModule(
      Shader::CompileSlangSPIRV(shaderPath, "CSMain", "compute"));
    try
    {
      VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CullPushConstants)};
      VkPipelineLayoutCreateInfo pipelineLayout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      pipelineLayout.setLayoutCount = 1;
      pipelineLayout.pSetLayouts = &s_Data.CullDescriptorSetLayout;
      pipelineLayout.pushConstantRangeCount = 1;
      pipelineLayout.pPushConstantRanges = &push;
      Check(vkCreatePipelineLayout(s_Data.Device, &pipelineLayout, nullptr,
                                   &s_Data.CullPipelineLayout),
            "vkCreatePipelineLayout (GPU cull)");
      VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
      pipeline.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      pipeline.stage.module = compute;
      pipeline.stage.pName = "main";
      pipeline.layout = s_Data.CullPipelineLayout;
      Check(vkCreateComputePipelines(s_Data.Device, VK_NULL_HANDLE, 1, &pipeline,
                                     nullptr, &s_Data.CullPipeline),
            "vkCreateComputePipelines (GPU cull)");
    }
    catch (...)
    {
      vkDestroyShaderModule(s_Data.Device, compute, nullptr);
      throw;
    }
    vkDestroyShaderModule(s_Data.Device, compute, nullptr);
  }

  void AllocateHiZDescriptors()
  {
    const uint32_t mipCount = s_Data.HiZ[0].MipCount;
    for (uint32_t frame = 0; frame < FramesInFlight; ++frame)
      for (uint32_t mip = 0; mip < mipCount; ++mip)
      {
        VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocation.descriptorPool = s_Data.HiZDescriptorPool;
        allocation.descriptorSetCount = 1;
        allocation.pSetLayouts = &s_Data.HiZDescriptorSetLayout;
        Check(vkAllocateDescriptorSets(s_Data.Device, &allocation,
                                       &s_Data.HiZDescriptors[frame][mip]),
              "vkAllocateDescriptorSets (Hi-Z)");
        VkDescriptorImageInfo sampler{};
        sampler.sampler = s_Data.PostSampler;
        VkDescriptorImageInfo destination{};
        destination.imageView = s_Data.HiZ[frame].MipViews[mip];
        destination.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        const std::array<VkWriteDescriptorSet, 2> writes{{
          {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
           s_Data.HiZDescriptors[frame][mip], 1, 0, 1,
           VK_DESCRIPTOR_TYPE_SAMPLER, &sampler, nullptr, nullptr},
          {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
           s_Data.HiZDescriptors[frame][mip], 2, 0, 1,
           VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &destination, nullptr, nullptr}}};
        vkUpdateDescriptorSets(s_Data.Device, static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
      }
  }

  void CreateHiZPipeline()
  {
    const std::array<VkDescriptorSetLayoutBinding, 3> bindings{{
      {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}}};
    VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout.bindingCount = static_cast<uint32_t>(bindings.size());
    layout.pBindings = bindings.data();
    Check(vkCreateDescriptorSetLayout(s_Data.Device, &layout, nullptr,
                                      &s_Data.HiZDescriptorSetLayout),
          "vkCreateDescriptorSetLayout (Hi-Z)");
    const uint32_t setCount = FramesInFlight * MaxHiZMips;
    const std::array<VkDescriptorPoolSize, 3> sizes{{
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, setCount},
      {VK_DESCRIPTOR_TYPE_SAMPLER, setCount},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, setCount}}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets = setCount;
    pool.poolSizeCount = static_cast<uint32_t>(sizes.size());
    pool.pPoolSizes = sizes.data();
    Check(vkCreateDescriptorPool(s_Data.Device, &pool, nullptr,
                                 &s_Data.HiZDescriptorPool),
          "vkCreateDescriptorPool (Hi-Z)");
    AllocateHiZDescriptors();
    std::filesystem::path shaderPath = "../res/shaders/vulkan_hiz.slang";
    if (!std::filesystem::exists(shaderPath)) shaderPath = "res/shaders/vulkan_hiz.slang";
    const VkShaderModule compute = CreateShaderModule(
      Shader::CompileSlangSPIRV(shaderPath, "CSMain", "compute"));
    try
    {
      VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(glm::uvec4)};
      VkPipelineLayoutCreateInfo pipelineLayout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      pipelineLayout.setLayoutCount = 1;
      pipelineLayout.pSetLayouts = &s_Data.HiZDescriptorSetLayout;
      pipelineLayout.pushConstantRangeCount = 1;
      pipelineLayout.pPushConstantRanges = &push;
      Check(vkCreatePipelineLayout(s_Data.Device, &pipelineLayout, nullptr,
                                   &s_Data.HiZPipelineLayout),
            "vkCreatePipelineLayout (Hi-Z)");
      VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
      pipeline.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      pipeline.stage.module = compute;
      pipeline.stage.pName = "main";
      pipeline.layout = s_Data.HiZPipelineLayout;
      Check(vkCreateComputePipelines(s_Data.Device, VK_NULL_HANDLE, 1, &pipeline,
                                     nullptr, &s_Data.HiZPipeline),
            "vkCreateComputePipelines (Hi-Z)");
    }
    catch (...)
    {
      vkDestroyShaderModule(s_Data.Device, compute, nullptr);
      throw;
    }
    vkDestroyShaderModule(s_Data.Device, compute, nullptr);
  }

  void DestroySwapchain()
  {
    if (s_Data.ImGuiInitialized)
      for (const VkDescriptorSet descriptor : s_Data.EditorDescriptors)
        if (descriptor) ImGui_ImplVulkan_RemoveTexture(descriptor);
    s_Data.EditorDescriptors.clear();
    for (const VkSemaphore semaphore : s_Data.RenderComplete)
      vkDestroySemaphore(s_Data.Device, semaphore, nullptr);
    for (const VkFramebuffer framebuffer : s_Data.Framebuffers)
      vkDestroyFramebuffer(s_Data.Device, framebuffer, nullptr);
    for (const VkFramebuffer framebuffer : s_Data.DepthPrepassFramebuffers)
      vkDestroyFramebuffer(s_Data.Device, framebuffer, nullptr);
    for (const VkFramebuffer framebuffer : s_Data.LightingFramebuffers)
      vkDestroyFramebuffer(s_Data.Device, framebuffer, nullptr);
    for (const VkFramebuffer framebuffer : s_Data.ParticleFramebuffers)
      vkDestroyFramebuffer(s_Data.Device, framebuffer, nullptr);
    for (const VkFramebuffer framebuffer : s_Data.PresentFramebuffers)
      vkDestroyFramebuffer(s_Data.Device, framebuffer, nullptr);
    for (const VkFramebuffer framebuffer : s_Data.EditorFramebuffers)
      vkDestroyFramebuffer(s_Data.Device, framebuffer, nullptr);
    for (GPUTexture& color : s_Data.SceneColors) DestroyTexture(color);
    for (GPUTexture& color : s_Data.GBufferAlbedo) DestroyTexture(color);
    for (GPUTexture& color : s_Data.GBufferNormal) DestroyTexture(color);
    for (GPUTexture& color : s_Data.GBufferPosition) DestroyTexture(color);
    for (GPUTexture& bloom : s_Data.BloomPyramids) DestroyTexture(bloom);
    for (GPUTexture& color : s_Data.EditorColors) DestroyTexture(color);
    for (GPUBuffer& grid : s_Data.TileGrids) DestroyBuffer(grid);
    for (GPUBuffer& indices : s_Data.TileIndices) DestroyBuffer(indices);
    DestroyHiZPyramids();
    for (const DepthAttachment& depth : s_Data.DepthAttachments)
    {
      if (depth.View) vkDestroyImageView(s_Data.Device, depth.View, nullptr);
      if (depth.Image) vkDestroyImage(s_Data.Device, depth.Image, nullptr);
      if (depth.Memory) vkFreeMemory(s_Data.Device, depth.Memory, nullptr);
    }
    for (const VkImageView view : s_Data.ImageViews)
      vkDestroyImageView(s_Data.Device, view, nullptr);
    if (s_Data.Swapchain) vkDestroySwapchainKHR(s_Data.Device, s_Data.Swapchain, nullptr);
    s_Data.Framebuffers.clear();
    s_Data.DepthPrepassFramebuffers.clear();
    s_Data.LightingFramebuffers.clear();
    s_Data.ParticleFramebuffers.clear();
    s_Data.PresentFramebuffers.clear();
    s_Data.EditorFramebuffers.clear();
    s_Data.SceneColors.clear();
    s_Data.GBufferAlbedo.clear();
    s_Data.GBufferNormal.clear();
    s_Data.GBufferPosition.clear();
    s_Data.BloomPyramids.clear();
    s_Data.EditorColors.clear();
    s_Data.TileGrids.clear();
    s_Data.TileIndices.clear();
    s_Data.TileDescriptors.clear();
    s_Data.DepthAttachments.clear();
    s_Data.ImageViews.clear();
    s_Data.Images.clear();
    s_Data.ImagesInFlight.clear();
    s_Data.RenderComplete.clear();
    s_Data.PostDescriptors.clear();
    s_Data.LightingDescriptors.clear();
    s_Data.Swapchain = VK_NULL_HANDLE;
  }

  void CreateRenderPass()
  {
    VkAttachmentDescription prepassDepth{};
    prepassDepth.format = s_Data.DepthFormat;
    prepassDepth.samples = VK_SAMPLE_COUNT_1_BIT;
    prepassDepth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    prepassDepth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    prepassDepth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    prepassDepth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    prepassDepth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    prepassDepth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference prepassDepthReference{
      0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription prepassSubpass{};
    prepassSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    prepassSubpass.pDepthStencilAttachment = &prepassDepthReference;
    const std::array<VkSubpassDependency, 2> prepassDependencies{{
      {VK_SUBPASS_EXTERNAL, 0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0,
       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_DEPENDENCY_BY_REGION_BIT},
      {0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
       VK_DEPENDENCY_BY_REGION_BIT}}};
    VkRenderPassCreateInfo prepassInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    prepassInfo.attachmentCount = 1;
    prepassInfo.pAttachments = &prepassDepth;
    prepassInfo.subpassCount = 1;
    prepassInfo.pSubpasses = &prepassSubpass;
    prepassInfo.dependencyCount = static_cast<uint32_t>(prepassDependencies.size());
    prepassInfo.pDependencies = prepassDependencies.data();
    Check(vkCreateRenderPass(s_Data.Device, &prepassInfo, nullptr,
                             &s_Data.DepthPrepassRenderPass),
          "vkCreateRenderPass (depth prepass)");

    VkAttachmentDescription albedo{};
    albedo.format = VK_FORMAT_R8G8B8A8_UNORM;
    albedo.samples = VK_SAMPLE_COUNT_1_BIT;
    albedo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    albedo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    albedo.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    albedo.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    albedo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    albedo.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkAttachmentDescription normal = albedo;
    normal.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkAttachmentDescription position = normal;
    VkAttachmentDescription depth{};
    depth.format = s_Data.DepthFormat;
    depth.samples = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    const std::array<VkAttachmentReference, 3> references{{
      {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
      {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
      {2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}}};
    VkAttachmentReference depthReference{3, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = static_cast<uint32_t>(references.size());
    subpass.pColorAttachments = references.data();
    subpass.pDepthStencilAttachment = &depthReference;
    const std::array<VkSubpassDependency, 2> dependencies{{
      {VK_SUBPASS_EXTERNAL, 0,
       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
       VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
       VK_DEPENDENCY_BY_REGION_BIT},
      {0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
       VK_ACCESS_SHADER_READ_BIT, VK_DEPENDENCY_BY_REGION_BIT}}};
    VkRenderPassCreateInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    const std::array attachments{albedo, normal, position, depth};
    info.attachmentCount = static_cast<uint32_t>(attachments.size());
    info.pAttachments = attachments.data();
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    info.dependencyCount = static_cast<uint32_t>(dependencies.size());
    info.pDependencies = dependencies.data();
    Check(vkCreateRenderPass(s_Data.Device, &info, nullptr, &s_Data.RenderPass),
          "vkCreateRenderPass (geometry)");

    VkAttachmentDescription litColor{};
    litColor.format = SceneColorFormat;
    litColor.samples = VK_SAMPLE_COUNT_1_BIT;
    litColor.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    litColor.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    litColor.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    litColor.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    litColor.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    litColor.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentReference litReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription litSubpass{};
    litSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    litSubpass.colorAttachmentCount = 1;
    litSubpass.pColorAttachments = &litReference;
    const std::array<VkSubpassDependency, 2> litDependencies{{
      {VK_SUBPASS_EXTERNAL, 0, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
       VK_DEPENDENCY_BY_REGION_BIT},
      {0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
       VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
       VK_DEPENDENCY_BY_REGION_BIT}}};
    VkRenderPassCreateInfo litInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    litInfo.attachmentCount = 1;
    litInfo.pAttachments = &litColor;
    litInfo.subpassCount = 1;
    litInfo.pSubpasses = &litSubpass;
    litInfo.dependencyCount = static_cast<uint32_t>(litDependencies.size());
    litInfo.pDependencies = litDependencies.data();
    Check(vkCreateRenderPass(s_Data.Device, &litInfo, nullptr,
                             &s_Data.LightingRenderPass),
          "vkCreateRenderPass (lighting)");

    VkAttachmentDescription particleColor = litColor;
    particleColor.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    particleColor.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    particleColor.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkAttachmentDescription particleDepth = depth;
    particleDepth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    particleDepth.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference particleColorReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference particleDepthReference{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription particleSubpass{};
    particleSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    particleSubpass.colorAttachmentCount = 1;
    particleSubpass.pColorAttachments = &particleColorReference;
    particleSubpass.pDepthStencilAttachment = &particleDepthReference;
    const std::array<VkSubpassDependency, 2> particleDependencies{{
      {VK_SUBPASS_EXTERNAL, 0,
       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
       VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT, VK_DEPENDENCY_BY_REGION_BIT},
      {0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
       VK_ACCESS_SHADER_READ_BIT, VK_DEPENDENCY_BY_REGION_BIT}}};
    const std::array particleAttachments{particleColor, particleDepth};
    VkRenderPassCreateInfo particleInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    particleInfo.attachmentCount = static_cast<uint32_t>(particleAttachments.size());
    particleInfo.pAttachments = particleAttachments.data();
    particleInfo.subpassCount = 1;
    particleInfo.pSubpasses = &particleSubpass;
    particleInfo.dependencyCount = static_cast<uint32_t>(particleDependencies.size());
    particleInfo.pDependencies = particleDependencies.data();
    Check(vkCreateRenderPass(s_Data.Device, &particleInfo, nullptr,
                             &s_Data.ParticleRenderPass),
          "vkCreateRenderPass (particles)");

    VkAttachmentDescription present{};
    present.format = s_Data.SwapchainFormat;
    present.samples = VK_SAMPLE_COUNT_1_BIT;
    present.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    present.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    present.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    present.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    present.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    present.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference presentReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription presentSubpass{};
    presentSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    presentSubpass.colorAttachmentCount = 1;
    presentSubpass.pColorAttachments = &presentReference;
    VkSubpassDependency presentDependency{};
    presentDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    presentDependency.dstSubpass = 0;
    presentDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    presentDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    presentDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    VkRenderPassCreateInfo presentInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    presentInfo.attachmentCount = 1;
    presentInfo.pAttachments = &present;
    presentInfo.subpassCount = 1;
    presentInfo.pSubpasses = &presentSubpass;
    presentInfo.dependencyCount = 1;
    presentInfo.pDependencies = &presentDependency;
    Check(vkCreateRenderPass(s_Data.Device, &presentInfo, nullptr,
                             &s_Data.PresentRenderPass),
          "vkCreateRenderPass (present)");

    VkAttachmentDescription editor = present;
    editor.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkRenderPassCreateInfo editorInfo = presentInfo;
    editorInfo.pAttachments = &editor;
    Check(vkCreateRenderPass(s_Data.Device, &editorInfo, nullptr,
                             &s_Data.EditorRenderPass),
          "vkCreateRenderPass (editor output)");
  }

  void CreateSwapchain()
  {
    VkSurfaceCapabilitiesKHR capabilities{};
    Check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
      s_Data.PhysicalDevice, s_Data.Surface, &capabilities),
      "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    uint32_t formatCount = 0;
    uint32_t modeCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(s_Data.PhysicalDevice, s_Data.Surface,
                                         &formatCount, nullptr);
    vkGetPhysicalDeviceSurfacePresentModesKHR(s_Data.PhysicalDevice, s_Data.Surface,
                                              &modeCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    std::vector<VkPresentModeKHR> modes(modeCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(s_Data.PhysicalDevice, s_Data.Surface,
                                         &formatCount, formats.data());
    vkGetPhysicalDeviceSurfacePresentModesKHR(s_Data.PhysicalDevice, s_Data.Surface,
                                              &modeCount, modes.data());
    if (formats.empty() || modes.empty()) throw std::runtime_error("Vulkan surface is not presentable");

    const VkSurfaceFormatKHR format = ChooseSurfaceFormat(formats);
    if (s_Data.RenderPass && format.format != s_Data.SwapchainFormat)
      throw std::runtime_error("Vulkan swapchain format changed during resize");
    s_Data.SwapchainFormat = format.format;
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
      s_Data.Extent = capabilities.currentExtent;
    else
    {
      s_Data.Extent.width = std::clamp(s_Data.RequestedWidth,
        capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
      s_Data.Extent.height = std::clamp(s_Data.RequestedHeight,
        capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    }

    uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0) imageCount = std::min(imageCount, capabilities.maxImageCount);
    s_Data.MinImageCount = std::max(2u, capabilities.minImageCount);
    VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    info.surface = s_Data.Surface;
    info.minImageCount = imageCount;
    info.imageFormat = format.format;
    info.imageColorSpace = format.colorSpace;
    info.imageExtent = s_Data.Extent;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    const uint32_t families[]{s_Data.GraphicsQueueFamily, s_Data.PresentQueueFamily};
    if (families[0] != families[1])
    {
      info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      info.queueFamilyIndexCount = 2;
      info.pQueueFamilyIndices = families;
    }
    else info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.preTransform = capabilities.currentTransform;
    info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    info.presentMode = ChoosePresentMode(modes);
    info.clipped = VK_TRUE;
    Check(vkCreateSwapchainKHR(s_Data.Device, &info, nullptr, &s_Data.Swapchain),
          "vkCreateSwapchainKHR");

    vkGetSwapchainImagesKHR(s_Data.Device, s_Data.Swapchain, &imageCount, nullptr);
    s_Data.Images.resize(imageCount);
    vkGetSwapchainImagesKHR(s_Data.Device, s_Data.Swapchain, &imageCount, s_Data.Images.data());
    s_Data.ImageViews.resize(imageCount);
    for (uint32_t index = 0; index < imageCount; ++index)
    {
      VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      view.image = s_Data.Images[index];
      view.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view.format = s_Data.SwapchainFormat;
      view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      view.subresourceRange.levelCount = 1;
      view.subresourceRange.layerCount = 1;
      Check(vkCreateImageView(s_Data.Device, &view, nullptr, &s_Data.ImageViews[index]),
            "vkCreateImageView");
    }

    if (!s_Data.RenderPass)
    {
      s_Data.DepthFormat = ChooseDepthFormat();
      CreateRenderPass();
    }
    s_Data.DepthAttachments.reserve(imageCount);
    s_Data.SceneColors.reserve(imageCount);
    s_Data.GBufferAlbedo.reserve(imageCount);
    s_Data.GBufferNormal.reserve(imageCount);
    s_Data.GBufferPosition.reserve(imageCount);
    s_Data.BloomPyramids.reserve(imageCount);
    s_Data.EditorColors.reserve(imageCount);
    s_Data.TileGrids.reserve(imageCount);
    s_Data.TileIndices.reserve(imageCount);
    s_Data.BloomMipCount = std::min(MaxBloomMips, 1u + static_cast<uint32_t>(
      std::floor(std::log2(static_cast<float>(std::max(
        s_Data.Extent.width, s_Data.Extent.height))))));
    for (uint32_t index = 0; index < imageCount; ++index)
    {
      s_Data.DepthAttachments.push_back(CreateDepthAttachment());
      s_Data.SceneColors.push_back(CreateColorAttachment(SceneColorFormat));
      s_Data.GBufferAlbedo.push_back(CreateColorAttachment(VK_FORMAT_R8G8B8A8_UNORM));
      s_Data.GBufferNormal.push_back(CreateColorAttachment(VK_FORMAT_R16G16B16A16_SFLOAT));
      s_Data.GBufferPosition.push_back(CreateColorAttachment(VK_FORMAT_R16G16B16A16_SFLOAT));
      s_Data.BloomPyramids.push_back(CreateBloomPyramid());
      s_Data.EditorColors.push_back(CreateColorAttachment(s_Data.SwapchainFormat));
      const uint32_t tileCount = ((s_Data.Extent.width + TileSize - 1) / TileSize) *
                                 ((s_Data.Extent.height + TileSize - 1) / TileSize);
      const std::vector<uint32_t> emptyGrid(static_cast<size_t>(tileCount) * 2);
      const std::vector<uint32_t> emptyIndices(static_cast<size_t>(tileCount) * MaxTileLights);
      s_Data.TileGrids.push_back(CreateUploadBuffer(
        emptyGrid.data(), emptyGrid.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
      s_Data.TileIndices.push_back(CreateUploadBuffer(
        emptyIndices.data(), emptyIndices.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    }
    CreateHiZPyramids();
    {
      const VkCommandBuffer command = BeginImmediateCommands();
      std::vector<VkImageMemoryBarrier> barriers(imageCount);
      for (uint32_t index = 0; index < imageCount; ++index)
      {
        barriers[index].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[index].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[index].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[index].image = s_Data.BloomPyramids[index].Image;
        barriers[index].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[index].subresourceRange.levelCount = s_Data.BloomMipCount;
        barriers[index].subresourceRange.layerCount = 1;
        barriers[index].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      }
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                           0, nullptr, 0, nullptr,
                           static_cast<uint32_t>(barriers.size()), barriers.data());
      EndImmediateCommands(command);
    }
    s_Data.Framebuffers.resize(imageCount);
    s_Data.DepthPrepassFramebuffers.resize(imageCount);
    s_Data.LightingFramebuffers.resize(imageCount);
    s_Data.ParticleFramebuffers.resize(imageCount);
    s_Data.PresentFramebuffers.resize(imageCount);
    s_Data.EditorFramebuffers.resize(imageCount);
    for (uint32_t index = 0; index < imageCount; ++index)
    {
      VkFramebufferCreateInfo depthFramebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
      depthFramebuffer.renderPass = s_Data.DepthPrepassRenderPass;
      depthFramebuffer.attachmentCount = 1;
      depthFramebuffer.pAttachments = &s_Data.DepthAttachments[index].View;
      depthFramebuffer.width = s_Data.Extent.width;
      depthFramebuffer.height = s_Data.Extent.height;
      depthFramebuffer.layers = 1;
      Check(vkCreateFramebuffer(s_Data.Device, &depthFramebuffer, nullptr,
                                 &s_Data.DepthPrepassFramebuffers[index]),
            "vkCreateFramebuffer (depth prepass)");
      VkFramebufferCreateInfo framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
      framebuffer.renderPass = s_Data.RenderPass;
      const std::array attachments{
        s_Data.GBufferAlbedo[index].View, s_Data.GBufferNormal[index].View,
        s_Data.GBufferPosition[index].View, s_Data.DepthAttachments[index].View};
      framebuffer.attachmentCount = static_cast<uint32_t>(attachments.size());
      framebuffer.pAttachments = attachments.data();
      framebuffer.width = s_Data.Extent.width;
      framebuffer.height = s_Data.Extent.height;
      framebuffer.layers = 1;
      Check(vkCreateFramebuffer(s_Data.Device, &framebuffer, nullptr,
                                &s_Data.Framebuffers[index]), "vkCreateFramebuffer");
      VkFramebufferCreateInfo lightingFramebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
      lightingFramebuffer.renderPass = s_Data.LightingRenderPass;
      lightingFramebuffer.attachmentCount = 1;
      lightingFramebuffer.pAttachments = &s_Data.SceneColors[index].View;
      lightingFramebuffer.width = s_Data.Extent.width;
      lightingFramebuffer.height = s_Data.Extent.height;
      lightingFramebuffer.layers = 1;
      Check(vkCreateFramebuffer(s_Data.Device, &lightingFramebuffer, nullptr,
                                &s_Data.LightingFramebuffers[index]),
            "vkCreateFramebuffer (lighting)");
      VkFramebufferCreateInfo particleFramebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
      particleFramebuffer.renderPass = s_Data.ParticleRenderPass;
      const std::array particleAttachments{
        s_Data.SceneColors[index].View, s_Data.DepthAttachments[index].View};
      particleFramebuffer.attachmentCount = static_cast<uint32_t>(particleAttachments.size());
      particleFramebuffer.pAttachments = particleAttachments.data();
      particleFramebuffer.width = s_Data.Extent.width;
      particleFramebuffer.height = s_Data.Extent.height;
      particleFramebuffer.layers = 1;
      Check(vkCreateFramebuffer(s_Data.Device, &particleFramebuffer, nullptr,
                                &s_Data.ParticleFramebuffers[index]),
            "vkCreateFramebuffer (particles)");
      VkFramebufferCreateInfo presentFramebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
      presentFramebuffer.renderPass = s_Data.PresentRenderPass;
      presentFramebuffer.attachmentCount = 1;
      presentFramebuffer.pAttachments = &s_Data.ImageViews[index];
      presentFramebuffer.width = s_Data.Extent.width;
      presentFramebuffer.height = s_Data.Extent.height;
      presentFramebuffer.layers = 1;
      Check(vkCreateFramebuffer(s_Data.Device, &presentFramebuffer, nullptr,
                                 &s_Data.PresentFramebuffers[index]),
            "vkCreateFramebuffer (present)");
      VkFramebufferCreateInfo editorFramebuffer = presentFramebuffer;
      editorFramebuffer.renderPass = s_Data.EditorRenderPass;
      editorFramebuffer.pAttachments = &s_Data.EditorColors[index].View;
      Check(vkCreateFramebuffer(s_Data.Device, &editorFramebuffer, nullptr,
                                 &s_Data.EditorFramebuffers[index]),
            "vkCreateFramebuffer (editor output)");
    }
    s_Data.ImagesInFlight.assign(imageCount, VK_NULL_HANDLE);
    s_Data.RenderComplete.resize(imageCount);
    VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (VkSemaphore& renderComplete : s_Data.RenderComplete)
      Check(vkCreateSemaphore(s_Data.Device, &semaphore, nullptr, &renderComplete),
            "vkCreateSemaphore (present)");
    s_Data.SwapchainDirty = false;
  }

  void RegisterEditorTextures()
  {
    s_Data.EditorDescriptors.clear();
    if (!s_Data.ImGuiInitialized || !s_Data.PostSampler) return;
    s_Data.EditorDescriptors.reserve(s_Data.EditorColors.size());
    for (const GPUTexture& color : s_Data.EditorColors)
      s_Data.EditorDescriptors.push_back(ImGui_ImplVulkan_AddTexture(
        s_Data.PostSampler, color.View, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
  }

  void RecreateSwapchain()
  {
    Check(vkDeviceWaitIdle(s_Data.Device), "vkDeviceWaitIdle");
    if (s_Data.PostDescriptorPool)
      Check(vkResetDescriptorPool(s_Data.Device, s_Data.PostDescriptorPool, 0),
            "vkResetDescriptorPool (post)");
    if (s_Data.LightingDescriptorPool)
      Check(vkResetDescriptorPool(s_Data.Device, s_Data.LightingDescriptorPool, 0),
            "vkResetDescriptorPool (lighting)");
    if (s_Data.TileDescriptorPool)
      Check(vkResetDescriptorPool(s_Data.Device, s_Data.TileDescriptorPool, 0),
            "vkResetDescriptorPool (tile lights)");
    if (s_Data.CullDescriptorPool)
      Check(vkResetDescriptorPool(s_Data.Device, s_Data.CullDescriptorPool, 0),
            "vkResetDescriptorPool (GPU cull resize)");
    if (s_Data.HiZDescriptorPool)
      Check(vkResetDescriptorPool(s_Data.Device, s_Data.HiZDescriptorPool, 0),
            "vkResetDescriptorPool (Hi-Z resize)");
    DestroySwapchain();
    CreateSwapchain();
    AllocatePostDescriptors();
    AllocateLightingDescriptors();
    AllocateTileDescriptors();
    if (s_Data.CullDescriptorPool) AllocateCullDescriptors();
    if (s_Data.HiZDescriptorPool) AllocateHiZDescriptors();
    if (s_Data.ImGuiInitialized)
    {
      ImGui_ImplVulkan_SetMinImageCount(s_Data.MinImageCount);
      RegisterEditorTextures();
    }
  }

  void CreateCommandsAndSync()
  {
    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool.queueFamilyIndex = s_Data.GraphicsQueueFamily;
    Check(vkCreateCommandPool(s_Data.Device, &pool, nullptr, &s_Data.CommandPool),
          "vkCreateCommandPool");
    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = s_Data.CommandPool;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = FramesInFlight;
    Check(vkAllocateCommandBuffers(s_Data.Device, &allocation, s_Data.CommandBuffers.data()),
          "vkAllocateCommandBuffers");

    VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (uint32_t frame = 0; frame < FramesInFlight; ++frame)
    {
      Check(vkCreateSemaphore(s_Data.Device, &semaphore, nullptr,
                              &s_Data.ImageAvailable[frame]), "vkCreateSemaphore");
      Check(vkCreateFence(s_Data.Device, &fence, nullptr, &s_Data.FrameFences[frame]),
            "vkCreateFence");
    }
  }

  void ImGuiCheckResult(const VkResult result)
  {
    if (result < 0)
      gablog_log(LOG_ERROR, __FILE__, __LINE__, "Dear ImGui Vulkan error: %d", result);
  }

  void UploadImGuiFonts()
  {
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = s_Data.CommandPool;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = 1;
    Check(vkAllocateCommandBuffers(s_Data.Device, &allocation, &command),
          "vkAllocateCommandBuffers (ImGui fonts)");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    Check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer (ImGui fonts)");
    if (!ImGui_ImplVulkan_CreateFontsTexture(command))
      throw std::runtime_error("ImGui_ImplVulkan_CreateFontsTexture failed");
    Check(vkEndCommandBuffer(command), "vkEndCommandBuffer (ImGui fonts)");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    Check(vkQueueSubmit(s_Data.GraphicsQueue, 1, &submit, VK_NULL_HANDLE),
          "vkQueueSubmit (ImGui fonts)");
    Check(vkQueueWaitIdle(s_Data.GraphicsQueue), "vkQueueWaitIdle (ImGui fonts)");
    ImGui_ImplVulkan_DestroyFontUploadObjects();
    vkFreeCommandBuffers(s_Data.Device, s_Data.CommandPool, 1, &command);
  }

  glm::mat4 CalculateLightViewProjection()
  {
    glm::vec3 direction(-1.0f, -2.0f, -1.0f);
    for (const RenderLight& light : RenderBackend::Lights())
      if (light.Type == LightType::DIRECT) { direction = light.Direction; break; }
    if (glm::dot(direction, direction) < 0.0001f)
      direction = glm::vec3(-1.0f, -2.0f, -1.0f);
    direction = glm::normalize(direction);
    constexpr float orthoSize = 90.0f;
    glm::vec3 focus = Camera::GetPosition() + Camera::GetForwardDirection() * 45.0f;
    const glm::vec3 fallbackUp =
      std::abs(glm::dot(direction, glm::vec3(0.0f, 1.0f, 0.0f))) > 0.98f
        ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
    const glm::vec3 right = glm::normalize(glm::cross(direction, fallbackUp));
    const glm::vec3 lightUp = glm::normalize(glm::cross(right, direction));
    const float unitsPerTexel = (2.0f * orthoSize) /
      static_cast<float>(std::max(s_Data.ShadowSize, 1u));
    focus += right * (std::round(glm::dot(focus, right) / unitsPerTexel) * unitsPerTexel -
                      glm::dot(focus, right));
    focus += lightUp * (std::round(glm::dot(focus, lightUp) / unitsPerTexel) * unitsPerTexel -
                        glm::dot(focus, lightUp));
    const glm::mat4 view = glm::lookAt(focus - direction * 180.0f, focus, lightUp);
    const glm::mat4 projection = glm::ortho(
      -orthoSize, orthoSize, -orthoSize, orthoSize, 0.1f, 400.0f);
    glm::mat4 correction(1.0f);
    correction[1][1] = -1.0f;
    correction[2][2] = 0.5f;
    correction[3][2] = 0.5f;
    return correction * projection * view;
  }

  void UpdateShadowDescriptorImages()
  {
    for (auto& [mesh, gpu] : s_Data.Meshes)
      for (uint32_t frame = 0; frame < FramesInFlight; ++frame)
      {
        VkDescriptorImageInfo image{};
        image.imageView = s_Data.Shadows[frame].Texture.View;
        image.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = gpu.Materials[frame];
        write.dstBinding = 6;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        write.pImageInfo = &image;
        vkUpdateDescriptorSets(s_Data.Device, 1, &write, 0, nullptr);
      }
    for (auto& descriptors : s_Data.LightingDescriptors)
      for (uint32_t frame = 0; frame < FramesInFlight; ++frame)
      {
        VkDescriptorImageInfo image{};
        image.imageView = s_Data.Shadows[frame].Texture.View;
        image.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = descriptors[frame];
        write.dstBinding = 5;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        write.pImageInfo = &image;
        vkUpdateDescriptorSets(s_Data.Device, 1, &write, 0, nullptr);
      }
  }

  void UpdateSkyboxDescriptorImages()
  {
    for (auto& descriptors : s_Data.LightingDescriptors)
      for (VkDescriptorSet descriptor : descriptors)
      {
        VkDescriptorImageInfo image{};
        image.imageView = s_Data.SkyboxTexture.View;
        image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = descriptor;
        write.dstBinding = 7;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        write.pImageInfo = &image;
        vkUpdateDescriptorSets(s_Data.Device, 1, &write, 0, nullptr);
      }
  }

  void UpdatePointShadowDescriptorImages()
  {
    for (auto& descriptors : s_Data.LightingDescriptors)
      for (uint32_t frame = 0; frame < FramesInFlight; ++frame)
      {
        std::array<VkDescriptorImageInfo, MaxPointShadowLights> images{};
        std::array<VkWriteDescriptorSet, MaxPointShadowLights> writes{};
        for (uint32_t shadow = 0; shadow < MaxPointShadowLights; ++shadow)
        {
          images[shadow].imageView = s_Data.PointShadows[frame].Cubes[shadow].Texture.View;
          images[shadow].imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
          writes[shadow].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
          writes[shadow].dstSet = descriptors[frame];
          writes[shadow].dstBinding = 8 + shadow;
          writes[shadow].descriptorCount = 1;
          writes[shadow].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
          writes[shadow].pImageInfo = &images[shadow];
        }
        vkUpdateDescriptorSets(s_Data.Device, static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
      }
  }

  void EnsureShadowSize(const uint32_t size)
  {
    if (size == s_Data.ShadowSize) return;
    Check(vkDeviceWaitIdle(s_Data.Device), "vkDeviceWaitIdle (shadow resize)");
    CreateShadowAttachments(size);
    UpdateShadowDescriptorImages();
  }

  void EnsurePointShadowSize(const uint32_t size)
  {
    if (size == s_Data.PointShadowSize) return;
    Check(vkDeviceWaitIdle(s_Data.Device), "vkDeviceWaitIdle (point shadow resize)");
    CreatePointShadowAttachments(size);
    UpdatePointShadowDescriptorImages();
  }

  void UpdateBoneBuffers()
  {
    for (const std::string& name : ModelManager::GetModelNames())
    {
      const auto model = ModelManager::GetModel(name);
      if (!model || !model->IsAnimated()) continue;
      const auto gpu = s_Data.Models.find(model.get());
      if (gpu == s_Data.Models.end()) continue;
      const auto& matrices = model->GetFinalBoneMatrices();
      const size_t count = std::min(matrices.size(), static_cast<size_t>(MAX_BONES));
      if (count == 0) continue;
      void* mapped = nullptr;
      Check(vkMapMemory(s_Data.Device, gpu->second.Bones[s_Data.Frame].Memory,
                        0, count * sizeof(glm::mat4), 0, &mapped),
            "vkMapMemory (bones)");
      std::memcpy(mapped, matrices.data(), count * sizeof(glm::mat4));
      vkUnmapMemory(s_Data.Device, gpu->second.Bones[s_Data.Frame].Memory);
    }
  }

  void SelectPointShadowLights(const RenderEffectSettings& effects)
  {
    s_Data.ActivePointShadowLights.fill(-1);
    if (effects.ShadowQuality == GraphicsQuality::Off) return;
    struct Candidate { float DistanceSquared; int32_t Index; };
    std::vector<Candidate> candidates;
    const glm::vec3 camera = Camera::GetPosition();
    const RenderFrustum frustum(Camera::GetViewProjection());
    const auto& lights = RenderBackend::Lights();
    for (uint32_t index = 0; index < lights.size() && index < MaxSceneLights; ++index)
    {
      if (lights[index].Type != LightType::POINT ||
          !frustum.IntersectsSphere(lights[index].Position, PointShadowRadius)) continue;
      const glm::vec3 delta = lights[index].Position - camera;
      candidates.push_back({glm::dot(delta, delta), static_cast<int32_t>(index)});
    }
    std::ranges::sort(candidates, {}, &Candidate::DistanceSquared);
    for (size_t slot = 0; slot < std::min(candidates.size(),
                                          static_cast<size_t>(MaxPointShadowLights)); ++slot)
      s_Data.ActivePointShadowLights[slot] = candidates[slot].Index;
  }

  glm::mat4 PointShadowViewProjection(const glm::vec3& position, const uint32_t face)
  {
    static constexpr std::array<glm::vec3, 6> directions{{
      {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}};
    static constexpr std::array<glm::vec3, 6> up{{
      {0, -1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}, {0, -1, 0}, {0, -1, 0}}};
    const glm::mat4 projection = glm::perspective(
      glm::half_pi<float>(), 1.0f, 0.1f, PointShadowRadius);
    glm::mat4 correction(1.0f);
    correction[1][1] = -1.0f;
    correction[2][2] = 0.5f;
    correction[3][2] = 0.5f;
    return correction * projection * glm::lookAt(
      position, position + directions[face], up[face]);
  }

  PointShadowCasterState GetPointShadowCasterState(const glm::vec3& lightPosition)
  {
    PointShadowCasterState state;
    auto hashBytes = [&](const void* data, const size_t size)
    {
      const auto* bytes = static_cast<const uint8_t*>(data);
      for (size_t index = 0; index < size; ++index)
        state.Hash = (state.Hash ^ bytes[index]) * 1099511628211ull;
    };
    for (const std::string& name : ModelManager::GetModelNames())
    {
      const auto model = ModelManager::GetModel(name);
      if (!model || !model->m_IsRendered ||
          model->GetPhysXMeshType() == MeshType::CONVEXMESH) continue;
      bool modelIncluded = false;
      for (const glm::mat4& transform : model->m_InstanceTransforms)
      {
        const WorldBoundingSphere sphere = CalculateWorldBoundingSphere(*model, transform);
        const glm::vec3 delta = sphere.center - lightPosition;
        const float maximumDistance = PointShadowRadius + sphere.radius;
        if (glm::dot(delta, delta) > maximumDistance * maximumDistance) continue;
        if (!modelIncluded)
        {
          hashBytes(name.data(), name.size());
          const float radius = model->GetBoundsRadius();
          hashBytes(&radius, sizeof(radius));
          modelIncluded = true;
        }
        hashBytes(&transform, sizeof(transform));
        state.Animated |= model->IsAnimated();
      }
    }
    return state;
  }

  void DrawPointShadowPass(const RenderEffectSettings& effects)
  {
    if (!s_Data.PointShadowPipeline || effects.ShadowQuality == GraphicsQuality::Off) return;
    const auto& lights = RenderBackend::Lights();
    VkCommandBuffer command = s_Data.CommandBuffers[s_Data.Frame];
    const VkViewport viewport{0.0f, 0.0f, static_cast<float>(s_Data.PointShadowSize),
                              static_cast<float>(s_Data.PointShadowSize), 0.0f, 1.0f};
    const VkRect2D scissor{{0, 0}, {s_Data.PointShadowSize, s_Data.PointShadowSize}};
    for (uint32_t slot = 0; slot < MaxPointShadowLights; ++slot)
    {
      const int32_t lightIndex = s_Data.ActivePointShadowLights[slot];
      PointShadowCacheEntry& cache = s_Data.PointShadowCache[s_Data.Frame][slot];
      if (lightIndex < 0 || static_cast<size_t>(lightIndex) >= lights.size())
      {
        cache.Valid = false;
        continue;
      }
      const PointShadowCasterState casterState = GetPointShadowCasterState(
        lights[lightIndex].Position);
      const glm::vec3 lightDelta = cache.LightPosition - lights[lightIndex].Position;
      const bool lightMoved = glm::dot(lightDelta, lightDelta) > 0.000001f;
      if (cache.Valid && cache.LightIndex == lightIndex && !lightMoved &&
          cache.CasterHash == casterState.Hash && !casterState.Animated)
        continue;
      for (uint32_t face = 0; face < 6; ++face)
      {
        const glm::mat4 lightViewProjection = PointShadowViewProjection(
          lights[lightIndex].Position, face);
        const bool gpuCulled = PrepareGPUCull(false, lightViewProjection, false);
        VkClearValue clear{};
        clear.depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        begin.renderPass = s_Data.PointShadowRenderPass;
        begin.framebuffer = s_Data.PointShadows[s_Data.Frame].Cubes[slot].Framebuffers[face];
        begin.renderArea.extent = {s_Data.PointShadowSize, s_Data.PointShadowSize};
        begin.clearValueCount = 1;
        begin.pClearValues = &clear;
        vkCmdBeginRenderPass(command, &begin, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(command, 0, 1, &viewport);
        vkCmdSetScissor(command, 0, 1, &scissor);
        vkCmdSetDepthBias(command, 1.25f, 0.0f, 1.75f);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          s_Data.PointShadowPipeline);
        if (gpuCulled)
        {
          for (const CullDrawRecord& record : s_Data.CullDrawRecords)
          {
            const ScenePushConstants constants{
              lightViewProjection,
              glm::vec4(1.0f, static_cast<float>(record.TransformOffset), 0.0f, 0.0f)};
            vkCmdPushConstants(command, s_Data.PointShadowPipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(constants), &constants);
            const VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(command, 0, 1, &record.Mesh->Vertices.Buffer, &offset);
            vkCmdBindIndexBuffer(command, record.Mesh->Indices.Buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    s_Data.PointShadowPipelineLayout, 0, 1,
                                    &record.Descriptor, 0, nullptr);
            vkCmdDrawIndexedIndirect(command, s_Data.CullFrames[s_Data.Frame].Commands.Buffer,
              static_cast<VkDeviceSize>(record.CommandIndex) *
                sizeof(VkDrawIndexedIndirectCommand),
              1, sizeof(VkDrawIndexedIndirectCommand));
          }
        }
        vkCmdEndRenderPass(command);
      }
      cache.CasterHash = casterState.Hash;
      cache.LightPosition = lights[lightIndex].Position;
      cache.LightIndex = lightIndex;
      cache.Valid = true;
    }
  }

  void UpdateSceneFrame(const RenderEffectSettings& effects,
                        const glm::mat4& lightViewProjection)
  {
    SceneFrameData data;
    data.LightViewProjection = lightViewProjection;
    glm::mat4 clipCorrection(1.0f);
    clipCorrection[1][1] = -1.0f;
    clipCorrection[2][2] = 0.5f;
    clipCorrection[3][2] = 0.5f;
    data.InverseViewProjection = glm::inverse(
      clipCorrection * Camera::GetViewProjection());
    data.CameraPosition = glm::vec4(Camera::GetPosition(), 1.0f);
    const auto& lights = RenderBackend::Lights();
    const uint32_t count = static_cast<uint32_t>(
      std::min(lights.size(), static_cast<size_t>(MaxSceneLights)));
    bool hasDirectional = false;
    for (uint32_t index = 0; index < count; ++index)
    {
      data.LightPositions[index] = glm::vec4(
        lights[index].Position, static_cast<float>(lights[index].Type));
      data.LightDirections[index] = glm::vec4(lights[index].Direction, 0.0f);
      data.LightColors[index] = glm::vec4(lights[index].Color, 1.0f);
      hasDirectional |= lights[index].Type == LightType::DIRECT;
    }
    data.Params = glm::vec4(
      static_cast<float>(count),
      effects.ShadowQuality != GraphicsQuality::Off && hasDirectional ? 1.0f : 0.0f,
      effects.PS1Enabled ? 1.0f : 0.0f,
      effects.PS1ColorLevels);
    data.Effects = glm::vec4(
      effects.Gamma, effects.BloomQuality == GraphicsQuality::Off ? 0.0f : effects.BloomStrength,
      effects.BloomThreshold, effects.BloomExposure);
    data.ShadowInfo.x = 1.0f / static_cast<float>(std::max(s_Data.ShadowSize, 1u));
    data.ShadowInfo.y = s_Data.HasSkybox ? 1.0f : 0.0f;
    data.ShadowInfo.z = effects.ShadowQuality == GraphicsQuality::Off ? 0.0f : 1.0f;
    data.ShadowInfo.w = PointShadowRadius;
    for (uint32_t slot = 0; slot < MaxPointShadowLights; ++slot)
    data.PointShadowIndices[slot] = static_cast<float>(s_Data.ActivePointShadowLights[slot]);
    data.TileInfo = glm::vec4(
      static_cast<float>((s_Data.Extent.width + TileSize - 1) / TileSize),
      s_Data.TilePipeline ? 1.0f : 0.0f,
      static_cast<float>(RenderBackend::DebugSettings().TiledLightingMode), 0.0f);
    data.ViewportInfo = glm::vec4(
      static_cast<float>(s_Data.Extent.width), static_cast<float>(s_Data.Extent.height),
      1.0f / static_cast<float>(std::max(s_Data.Extent.width, 1u)),
      effects.PS1VirtualHeight);
    void* mapped = nullptr;
    Check(vkMapMemory(s_Data.Device, s_Data.SceneFrames[s_Data.Frame].Memory,
                      0, sizeof(data), 0, &mapped), "vkMapMemory (scene frame)");
    std::memcpy(mapped, &data, sizeof(data));
    vkUnmapMemory(s_Data.Device, s_Data.SceneFrames[s_Data.Frame].Memory);
  }

  void DrawShadowPass(const RenderEffectSettings& effects,
                      const glm::mat4& lightViewProjection, const bool gpuCulled)
  {
    if (!s_Data.ShadowPipeline) return;
    const bool hasDirectional = std::ranges::any_of(
      RenderBackend::Lights(), [](const RenderLight& light)
      { return light.Type == LightType::DIRECT; });
    const bool renderCasters = effects.ShadowQuality != GraphicsQuality::Off && hasDirectional;
    VkCommandBuffer command = s_Data.CommandBuffers[s_Data.Frame];
    VkClearValue clear{};
    clear.depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass = s_Data.ShadowRenderPass;
    begin.framebuffer = s_Data.Shadows[s_Data.Frame].Framebuffer;
    begin.renderArea.extent = {s_Data.ShadowSize, s_Data.ShadowSize};
    begin.clearValueCount = 1;
    begin.pClearValues = &clear;
    vkCmdBeginRenderPass(command, &begin, VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport{0.0f, 0.0f, static_cast<float>(s_Data.ShadowSize),
                              static_cast<float>(s_Data.ShadowSize), 0.0f, 1.0f};
    const VkRect2D scissor{{0, 0}, {s_Data.ShadowSize, s_Data.ShadowSize}};
    vkCmdSetViewport(command, 0, 1, &viewport);
    vkCmdSetScissor(command, 0, 1, &scissor);
    vkCmdSetDepthBias(command, 1.25f, 0.0f, 1.75f);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, s_Data.ShadowPipeline);
    if (renderCasters && gpuCulled)
    {
      for (const CullDrawRecord& record : s_Data.CullDrawRecords)
      {
        const ScenePushConstants constants{
          lightViewProjection,
          glm::vec4(1.0f, static_cast<float>(record.TransformOffset), 0.0f, 0.0f)};
        vkCmdPushConstants(command, s_Data.ShadowPipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(constants), &constants);
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(command, 0, 1, &record.Mesh->Vertices.Buffer, &offset);
        vkCmdBindIndexBuffer(command, record.Mesh->Indices.Buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                s_Data.ShadowPipelineLayout, 0, 1,
                                &record.Descriptor, 0, nullptr);
        vkCmdDrawIndexedIndirect(command, s_Data.CullFrames[s_Data.Frame].Commands.Buffer,
          static_cast<VkDeviceSize>(record.CommandIndex) * sizeof(VkDrawIndexedIndirectCommand),
          1, sizeof(VkDrawIndexedIndirectCommand));
      }
    }
    vkCmdEndRenderPass(command);
  }

  bool PrepareGPUCull(const bool useHiZ, const glm::mat4& viewProjection,
                      const bool includePreviews)
  {
    if (!s_Data.CullPipeline) return false;
    std::vector<glm::mat4> sourceTransforms;
    std::vector<CullInput> inputs;
    s_Data.CullDrawRecords.clear();
    size_t outputTransformCount = 0;
    RenderStatistics statistics;
    const RenderFrustum cameraFrustum(Camera::GetViewProjection());
    for (const std::string& name : ModelManager::GetModelNames())
    {
      const auto model = ModelManager::GetModel(name);
      if (!model || !model->m_IsRendered ||
          model->GetPhysXMeshType() == MeshType::CONVEXMESH) continue;
      const uint32_t sourceOffset = static_cast<uint32_t>(sourceTransforms.size());
      sourceTransforms.insert(sourceTransforms.end(), model->m_InstanceTransforms.begin(),
                              model->m_InstanceTransforms.end());
      statistics.RenderableInstances += static_cast<uint32_t>(model->m_InstanceTransforms.size());
      for (const glm::mat4& transform : model->m_InstanceTransforms)
      {
        const WorldBoundingSphere sphere = CalculateWorldBoundingSphere(*model, transform);
        if (cameraFrustum.IntersectsSphere(sphere.center, sphere.radius))
          ++statistics.VisibleInstances;
      }
      for (Mesh& mesh : model->GetMeshes())
      {
        const auto gpu = s_Data.Meshes.find(&mesh);
        if (gpu == s_Data.Meshes.end()) continue;
        CullInput input;
        input.LocalSphere = glm::vec4(model->GetBoundsCenter(),
                                      std::max(model->GetBoundsRadius(), 0.001f));
        input.Metadata = glm::uvec4(
          sourceOffset, static_cast<uint32_t>(model->m_InstanceTransforms.size()),
          static_cast<uint32_t>(outputTransformCount), gpu->second.IndexCount);
        inputs.push_back(input);
        s_Data.CullDrawRecords.push_back(
          {&gpu->second, gpu->second.Materials[s_Data.Frame],
           static_cast<uint32_t>(inputs.size() - 1),
           static_cast<uint32_t>(outputTransformCount), 1.0f});
        outputTransformCount += model->m_InstanceTransforms.size();
      }
    }
    if (includePreviews)
    for (const RenderModelPreview& preview : s_Data.ModelPreviews)
    {
      const auto model = ModelManager::GetModel(preview.ModelName);
      if (!model || model->GetPhysXMeshType() == MeshType::CONVEXMESH) continue;
      const uint32_t sourceOffset = static_cast<uint32_t>(sourceTransforms.size());
      sourceTransforms.push_back(preview.Transform);
      for (Mesh& mesh : model->GetMeshes())
      {
        const auto gpu = s_Data.Meshes.find(&mesh);
        if (gpu == s_Data.Meshes.end()) continue;
        CullInput input;
        input.LocalSphere = glm::vec4(model->GetBoundsCenter(),
                                      std::max(model->GetBoundsRadius(), 0.001f));
        input.Metadata = glm::uvec4(sourceOffset, 1u,
          static_cast<uint32_t>(outputTransformCount), gpu->second.IndexCount | 0x80000000u);
        inputs.push_back(input);
        s_Data.CullDrawRecords.push_back(
          {&gpu->second, gpu->second.Materials[s_Data.Frame],
           static_cast<uint32_t>(inputs.size() - 1),
           static_cast<uint32_t>(outputTransformCount),
           std::max(preview.Brightness, 0.0f)});
        ++outputTransformCount;
      }
    }
    RenderBackend::SetStatistics(statistics);
    if (inputs.empty()) return false;
    const size_t requiredTransforms = std::max(sourceTransforms.size(), outputTransformCount);
    if (requiredTransforms > s_Data.CullTransformCapacity ||
        inputs.size() > s_Data.CullDrawCapacity)
      ResizeCullBuffers(std::max(requiredTransforms, s_Data.CullTransformCapacity * 2),
                        std::max(inputs.size(), s_Data.CullDrawCapacity * 2));
    GPUCullFrame& frame = s_Data.CullFrames[s_Data.Frame];
    auto upload = [&](GPUBuffer& buffer, const void* data, const VkDeviceSize bytes,
                      const char* operation)
    {
      if (bytes == 0) return;
      void* mapped = nullptr;
      Check(vkMapMemory(s_Data.Device, buffer.Memory, 0, bytes, 0, &mapped), operation);
      std::memcpy(mapped, data, static_cast<size_t>(bytes));
      vkUnmapMemory(s_Data.Device, buffer.Memory);
    };
    upload(frame.SourceTransforms, sourceTransforms.data(),
           sourceTransforms.size() * sizeof(glm::mat4), "vkMapMemory (cull transforms)");
    upload(frame.Inputs, inputs.data(), inputs.size() * sizeof(CullInput),
           "vkMapMemory (cull inputs)");
    VkCommandBuffer command = s_Data.CommandBuffers[s_Data.Frame];
    std::array<VkBufferMemoryBarrier, 2> reuseBarriers{};
    reuseBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    reuseBarriers[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    reuseBarriers[0].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    reuseBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    reuseBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    reuseBarriers[0].buffer = frame.VisibleTransforms.Buffer;
    reuseBarriers[0].size = VK_WHOLE_SIZE;
    reuseBarriers[1] = reuseBarriers[0];
    reuseBarriers[1].srcAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    reuseBarriers[1].buffer = frame.Commands.Buffer;
    vkCmdPipelineBarrier(command,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                         VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         0, nullptr, static_cast<uint32_t>(reuseBarriers.size()),
                         reuseBarriers.data(), 0, nullptr);
    std::array<VkBufferMemoryBarrier, 2> hostBarriers{};
    const std::array<VkBuffer, 2> hostBuffers{frame.SourceTransforms.Buffer, frame.Inputs.Buffer};
    for (size_t index = 0; index < hostBarriers.size(); ++index)
    {
      hostBarriers[index].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      hostBarriers[index].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
      hostBarriers[index].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      hostBarriers[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      hostBarriers[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      hostBarriers[index].buffer = hostBuffers[index];
      hostBarriers[index].size = VK_WHOLE_SIZE;
    }
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         0, nullptr, static_cast<uint32_t>(hostBarriers.size()),
                         hostBarriers.data(), 0, nullptr);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, s_Data.CullPipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            s_Data.CullPipelineLayout, 0, 1,
                            &s_Data.CullDescriptors[s_Data.Frame], 0, nullptr);
    CullPushConstants constants;
    constants.ViewProjection = viewProjection;
    constants.Params.x = static_cast<uint32_t>(inputs.size());
    constants.Params.y = useHiZ && s_Data.HiZ[s_Data.Frame].Valid ? 1u : 0u;
    constants.Params.z = s_Data.Extent.width;
    constants.Params.w = s_Data.Extent.height;
    vkCmdPushConstants(command, s_Data.CullPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(constants), &constants);
    vkCmdDispatch(command, (constants.Params.x + 63) / 64, 1, 1);
    std::array<VkBufferMemoryBarrier, 2> outputBarriers{};
    outputBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    outputBarriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    outputBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    outputBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputBarriers[0].buffer = frame.VisibleTransforms.Buffer;
    outputBarriers[0].size = VK_WHOLE_SIZE;
    outputBarriers[1] = outputBarriers[0];
    outputBarriers[1].dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    outputBarriers[1].buffer = frame.Commands.Buffer;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                         VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0,
                         0, nullptr, static_cast<uint32_t>(outputBarriers.size()),
                         outputBarriers.data(), 0, nullptr);
    return true;
  }

  void DrawDepthPrepass(const bool drawScene)
  {
    if (!s_Data.FrameStarted || s_Data.DepthPrepared ||
        s_Data.ImageIndex >= s_Data.DepthPrepassFramebuffers.size()) return;
    VkCommandBuffer command = s_Data.CommandBuffers[s_Data.Frame];
    VkClearValue clear{};
    clear.depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass = s_Data.DepthPrepassRenderPass;
    begin.framebuffer = s_Data.DepthPrepassFramebuffers[s_Data.ImageIndex];
    begin.renderArea.extent = s_Data.Extent;
    begin.clearValueCount = 1;
    begin.pClearValues = &clear;
    vkCmdBeginRenderPass(command, &begin, VK_SUBPASS_CONTENTS_INLINE);
    if (drawScene && s_Data.DepthPrepassPipeline)
    {
      const VkViewport viewport{0.0f, 0.0f, static_cast<float>(s_Data.Extent.width),
                                static_cast<float>(s_Data.Extent.height), 0.0f, 1.0f};
      const VkRect2D scissor{{0, 0}, s_Data.Extent};
      vkCmdSetViewport(command, 0, 1, &viewport);
      vkCmdSetScissor(command, 0, 1, &scissor);
      vkCmdSetDepthBias(command, 1.0f, 0.0f, 0.0f);
      vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        s_Data.DepthPrepassPipeline);
      glm::mat4 correction(1.0f);
      correction[1][1] = -1.0f;
      correction[2][2] = 0.5f;
      correction[3][2] = 0.5f;
      const ScenePushConstants base{
        correction * Camera::GetViewProjection(), glm::vec4(1.0f, 0.0f, 0.0f, 0.0f)};
      for (const CullDrawRecord& record : s_Data.CullDrawRecords)
      {
        ScenePushConstants constants = base;
        constants.DrawParams.x = record.Brightness;
        constants.DrawParams.y = static_cast<float>(record.TransformOffset);
        vkCmdPushConstants(command, s_Data.DepthPrepassPipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(constants), &constants);
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(command, 0, 1, &record.Mesh->Vertices.Buffer, &offset);
        vkCmdBindIndexBuffer(command, record.Mesh->Indices.Buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                s_Data.DepthPrepassPipelineLayout, 0, 1,
                                &record.Descriptor, 0, nullptr);
        vkCmdDrawIndexedIndirect(command, s_Data.CullFrames[s_Data.Frame].Commands.Buffer,
          static_cast<VkDeviceSize>(record.CommandIndex) * sizeof(VkDrawIndexedIndirectCommand),
          1, sizeof(VkDrawIndexedIndirectCommand));
      }
    }
    vkCmdEndRenderPass(command);
    s_Data.DepthPrepared = true;
  }

  void BeginMainRenderPass()
  {
    if (!s_Data.FrameStarted || s_Data.RenderPassActive ||
        s_Data.ParticlePassActive || s_Data.PresentPassActive) return;
    DrawDepthPrepass(false);
    std::array<VkClearValue, 4> clear{};
    clear[3].depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo renderPass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    renderPass.renderPass = s_Data.RenderPass;
    renderPass.framebuffer = s_Data.Framebuffers[s_Data.ImageIndex];
    renderPass.renderArea.extent = s_Data.Extent;
    renderPass.clearValueCount = static_cast<uint32_t>(clear.size());
    renderPass.pClearValues = clear.data();
    vkCmdBeginRenderPass(s_Data.CommandBuffers[s_Data.Frame], &renderPass,
                         VK_SUBPASS_CONTENTS_INLINE);
    s_Data.RenderPassActive = true;
  }

  void BuildHiZ()
  {
    if (!s_Data.HiZPipeline || s_Data.ImageIndex >= s_Data.DepthAttachments.size()) return;
    VkCommandBuffer command = s_Data.CommandBuffers[s_Data.Frame];
    HiZPyramid& pyramid = s_Data.HiZ[s_Data.Frame];
    VkImageMemoryBarrier depthToRead{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    depthToRead.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthToRead.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    depthToRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depthToRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depthToRead.image = s_Data.DepthAttachments[s_Data.ImageIndex].Image;
    depthToRead.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthToRead.subresourceRange.levelCount = 1;
    depthToRead.subresourceRange.layerCount = 1;
    depthToRead.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    depthToRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &depthToRead);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, s_Data.HiZPipeline);
    for (uint32_t mip = 0; mip < pyramid.MipCount; ++mip)
    {
      VkImageMemoryBarrier toGeneral{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      toGeneral.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
      toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      toGeneral.image = pyramid.Texture.Image;
      toGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      toGeneral.subresourceRange.baseMipLevel = mip;
      toGeneral.subresourceRange.levelCount = 1;
      toGeneral.subresourceRange.layerCount = 1;
      toGeneral.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
      toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                           0, nullptr, 0, nullptr, 1, &toGeneral);
      VkDescriptorImageInfo source{};
      source.imageView = mip == 0 ? s_Data.DepthAttachments[s_Data.ImageIndex].View
                                  : pyramid.MipViews[mip - 1];
      source.imageLayout = mip == 0 ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                    : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      write.dstSet = s_Data.HiZDescriptors[s_Data.Frame][mip];
      write.dstBinding = 0;
      write.descriptorCount = 1;
      write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      write.pImageInfo = &source;
      vkUpdateDescriptorSets(s_Data.Device, 1, &write, 0, nullptr);
      vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                              s_Data.HiZPipelineLayout, 0, 1,
                              &s_Data.HiZDescriptors[s_Data.Frame][mip], 0, nullptr);
      const uint32_t width = std::max(1u, s_Data.Extent.width >> mip);
      const uint32_t height = std::max(1u, s_Data.Extent.height >> mip);
      const glm::uvec4 constants(width, height, 0u,
                                 mip == 0 ? 1u : 0u);
      vkCmdPushConstants(command, s_Data.HiZPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                         0, sizeof(constants), &constants);
      vkCmdDispatch(command, (width + 7) / 8, (height + 7) / 8, 1);
      VkImageMemoryBarrier toRead = toGeneral;
      toRead.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
      toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      toRead.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                           0, nullptr, 0, nullptr, 1, &toRead);
    }
    VkImageMemoryBarrier depthToAttachment = depthToRead;
    depthToAttachment.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    depthToAttachment.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthToAttachment.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    depthToAttachment.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &depthToAttachment);
    pyramid.Valid = true;
  }

  void DispatchTileLights()
  {
    if (!s_Data.TilePipeline || s_Data.ImageIndex >= s_Data.TileDescriptors.size()) return;
    VkCommandBuffer command = s_Data.CommandBuffers[s_Data.Frame];
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, s_Data.TilePipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            s_Data.TilePipelineLayout, 0, 1,
                            &s_Data.TileDescriptors[s_Data.ImageIndex][s_Data.Frame],
                            0, nullptr);
    const uint32_t tileCountX = (s_Data.Extent.width + TileSize - 1) / TileSize;
    const uint32_t tileCountY = (s_Data.Extent.height + TileSize - 1) / TileSize;
    vkCmdDispatch(command, tileCountX, tileCountY, 1);
    std::array<VkBufferMemoryBarrier, 2> barriers{};
    const std::array<VkBuffer, 2> buffers{
      s_Data.TileGrids[s_Data.ImageIndex].Buffer,
      s_Data.TileIndices[s_Data.ImageIndex].Buffer};
    for (size_t index = 0; index < barriers.size(); ++index)
    {
      barriers[index].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      barriers[index].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barriers[index].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      barriers[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[index].buffer = buffers[index];
      barriers[index].size = VK_WHOLE_SIZE;
    }
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                         0, nullptr, static_cast<uint32_t>(barriers.size()),
                         barriers.data(), 0, nullptr);
  }

  void BeginParticleRenderPass()
  {
    if (!s_Data.FrameStarted || s_Data.ParticlePassActive || s_Data.PresentPassActive) return;
    BeginMainRenderPass();
    VkCommandBuffer command = s_Data.CommandBuffers[s_Data.Frame];
    if (s_Data.RenderPassActive)
    {
      vkCmdEndRenderPass(command);
      s_Data.RenderPassActive = false;
    }
    DispatchTileLights();

    VkClearValue clear{};
    clear.color = {{s_Data.ClearColor.r, s_Data.ClearColor.g,
                    s_Data.ClearColor.b, s_Data.ClearColor.a}};
    VkRenderPassBeginInfo lighting{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    lighting.renderPass = s_Data.LightingRenderPass;
    lighting.framebuffer = s_Data.LightingFramebuffers[s_Data.ImageIndex];
    lighting.renderArea.extent = s_Data.Extent;
    lighting.clearValueCount = 1;
    lighting.pClearValues = &clear;
    vkCmdBeginRenderPass(command, &lighting, VK_SUBPASS_CONTENTS_INLINE);
    if (s_Data.LightingPipeline &&
        s_Data.ImageIndex < s_Data.LightingDescriptors.size())
    {
      const VkViewport viewport{0.0f, 0.0f, static_cast<float>(s_Data.Extent.width),
                                static_cast<float>(s_Data.Extent.height), 0.0f, 1.0f};
      const VkRect2D scissor{{0, 0}, s_Data.Extent};
      vkCmdSetViewport(command, 0, 1, &viewport);
      vkCmdSetScissor(command, 0, 1, &scissor);
      vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, s_Data.LightingPipeline);
      vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              s_Data.LightingPipelineLayout, 0, 1,
                              &s_Data.LightingDescriptors[s_Data.ImageIndex][s_Data.Frame],
                              0, nullptr);
      const VkDeviceSize offset = 0;
      vkCmdBindVertexBuffers(command, 0, 1, &s_Data.PostVertices.Buffer, &offset);
      vkCmdDraw(command, 3, 1, 0, 0);
    }
    vkCmdEndRenderPass(command);

    VkRenderPassBeginInfo particles{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    particles.renderPass = s_Data.ParticleRenderPass;
    particles.framebuffer = s_Data.ParticleFramebuffers[s_Data.ImageIndex];
    particles.renderArea.extent = s_Data.Extent;
    vkCmdBeginRenderPass(command, &particles, VK_SUBPASS_CONTENTS_INLINE);
    s_Data.ParticlePassActive = true;
  }

  void BuildBloomPyramid()
  {
    if (s_Data.CurrentEffects.BloomQuality == GraphicsQuality::Off ||
        s_Data.ImageIndex >= s_Data.BloomPyramids.size()) return;
    VkCommandBuffer command = s_Data.CommandBuffers[s_Data.Frame];
    const uint32_t mipCount = std::clamp(
      s_Data.CurrentEffects.BloomPassCount(), 1u, s_Data.BloomMipCount);
    VkImageMemoryBarrier sceneToTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    sceneToTransfer.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    sceneToTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    sceneToTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    sceneToTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    sceneToTransfer.image = s_Data.SceneColors[s_Data.ImageIndex].Image;
    sceneToTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    sceneToTransfer.subresourceRange.levelCount = 1;
    sceneToTransfer.subresourceRange.layerCount = 1;
    sceneToTransfer.srcAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    sceneToTransfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    VkImageMemoryBarrier bloomToTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    bloomToTransfer.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bloomToTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bloomToTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bloomToTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bloomToTransfer.image = s_Data.BloomPyramids[s_Data.ImageIndex].Image;
    bloomToTransfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bloomToTransfer.subresourceRange.levelCount = 1;
    bloomToTransfer.subresourceRange.layerCount = 1;
    bloomToTransfer.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bloomToTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    const std::array initialBarriers{sceneToTransfer, bloomToTransfer};
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                         static_cast<uint32_t>(initialBarriers.size()),
                         initialBarriers.data());
    VkImageCopy copy{};
    copy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.srcSubresource.layerCount = 1;
    copy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.dstSubresource.layerCount = 1;
    copy.extent = {s_Data.Extent.width, s_Data.Extent.height, 1};
    vkCmdCopyImage(command, s_Data.SceneColors[s_Data.ImageIndex].Image,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   s_Data.BloomPyramids[s_Data.ImageIndex].Image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    for (uint32_t mip = 1; mip < mipCount; ++mip)
    {
      VkImageMemoryBarrier barriers[2]{};
      barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[0].image = s_Data.BloomPyramids[s_Data.ImageIndex].Image;
      barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      barriers[0].subresourceRange.baseMipLevel = mip - 1;
      barriers[0].subresourceRange.levelCount = 1;
      barriers[0].subresourceRange.layerCount = 1;
      barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      barriers[1] = bloomToTransfer;
      barriers[1].subresourceRange.baseMipLevel = mip;
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT |
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                           2, barriers);
      const int32_t sourceWidth = static_cast<int32_t>(std::max(1u, s_Data.Extent.width >> (mip - 1)));
      const int32_t sourceHeight = static_cast<int32_t>(std::max(1u, s_Data.Extent.height >> (mip - 1)));
      const int32_t targetWidth = static_cast<int32_t>(std::max(1u, s_Data.Extent.width >> mip));
      const int32_t targetHeight = static_cast<int32_t>(std::max(1u, s_Data.Extent.height >> mip));
      VkImageBlit blit{};
      blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      blit.srcSubresource.mipLevel = mip - 1;
      blit.srcSubresource.layerCount = 1;
      blit.srcOffsets[1] = {sourceWidth, sourceHeight, 1};
      blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      blit.dstSubresource.mipLevel = mip;
      blit.dstSubresource.layerCount = 1;
      blit.dstOffsets[1] = {targetWidth, targetHeight, 1};
      vkCmdBlitImage(command, s_Data.BloomPyramids[s_Data.ImageIndex].Image,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     s_Data.BloomPyramids[s_Data.ImageIndex].Image,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
    }

    std::vector<VkImageMemoryBarrier> finalBarriers;
    finalBarriers.reserve(mipCount + 1);
    for (uint32_t mip = 0; mip < mipCount; ++mip)
    {
      VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      barrier.oldLayout = mip + 1 < mipCount
        ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image = s_Data.BloomPyramids[s_Data.ImageIndex].Image;
      barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      barrier.subresourceRange.baseMipLevel = mip;
      barrier.subresourceRange.levelCount = 1;
      barrier.subresourceRange.layerCount = 1;
      barrier.srcAccessMask = mip + 1 < mipCount
        ? VK_ACCESS_TRANSFER_READ_BIT : VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      finalBarriers.push_back(barrier);
    }
    VkImageMemoryBarrier sceneToShader = sceneToTransfer;
    sceneToShader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    sceneToShader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    sceneToShader.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    sceneToShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    finalBarriers.push_back(sceneToShader);
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                         0, nullptr, 0, nullptr,
                         static_cast<uint32_t>(finalBarriers.size()),
                         finalBarriers.data());
  }

  void DrawPostProcess()
  {
    if (!s_Data.PostPipeline || s_Data.ImageIndex >= s_Data.PostDescriptors.size()) return;
    VkCommandBuffer command = s_Data.CommandBuffers[s_Data.Frame];
    const VkViewport viewport{0.0f, 0.0f, static_cast<float>(s_Data.Extent.width),
                              static_cast<float>(s_Data.Extent.height), 0.0f, 1.0f};
    const VkRect2D scissor{{0, 0}, s_Data.Extent};
    vkCmdSetViewport(command, 0, 1, &viewport);
    vkCmdSetScissor(command, 0, 1, &scissor);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, s_Data.PostPipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            s_Data.PostPipelineLayout, 0, 1,
                            &s_Data.PostDescriptors[s_Data.ImageIndex], 0, nullptr);
    const VkDeviceSize vertexOffset = 0;
    vkCmdBindVertexBuffers(command, 0, 1, &s_Data.PostVertices.Buffer, &vertexOffset);
    const RenderEffectSettings& effects = s_Data.CurrentEffects;
    const float bloomPassCount = static_cast<float>(std::min(
      effects.BloomPassCount(), s_Data.BloomMipCount));
    PostPushConstants constants;
    constants.Params = glm::vec4(
      effects.BloomQuality == GraphicsQuality::Off ? 0.0f : effects.BloomStrength,
      effects.BloomThreshold, effects.BloomExposure, effects.Gamma);
    constants.Effects = glm::vec4(
      effects.PS1Enabled ? 1.0f : 0.0f, effects.PS1ColorLevels,
      effects.PS1VirtualHeight, bloomPassCount);
    constants.Resolution = glm::vec4(
      static_cast<float>(s_Data.Extent.width), static_cast<float>(s_Data.Extent.height),
      1.0f / static_cast<float>(s_Data.Extent.width),
      1.0f / static_cast<float>(s_Data.Extent.height));
    vkCmdPushConstants(command, s_Data.PostPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(constants), &constants);
    vkCmdDraw(command, 3, 1, 0, 0);
  }

  void BeginPresentRenderPass()
  {
    if (!s_Data.FrameStarted || s_Data.PresentPassActive) return;
    BeginParticleRenderPass();
    VkCommandBuffer command = s_Data.CommandBuffers[s_Data.Frame];
    if (s_Data.ParticlePassActive)
    {
      vkCmdEndRenderPass(command);
      s_Data.ParticlePassActive = false;
    }
    BuildBloomPyramid();
    VkClearValue clear{};
    clear.color = {{0.008f, 0.012f, 0.025f, 1.0f}};
    if (s_Data.RenderForEditor &&
        s_Data.ImageIndex < s_Data.EditorFramebuffers.size())
    {
      VkRenderPassBeginInfo editorBegin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      editorBegin.renderPass = s_Data.EditorRenderPass;
      editorBegin.framebuffer = s_Data.EditorFramebuffers[s_Data.ImageIndex];
      editorBegin.renderArea.extent = s_Data.Extent;
      editorBegin.clearValueCount = 1;
      editorBegin.pClearValues = &clear;
      vkCmdBeginRenderPass(command, &editorBegin, VK_SUBPASS_CONTENTS_INLINE);
      DrawPostProcess();
      vkCmdEndRenderPass(command);
      VkImageMemoryBarrier editorToImGui{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      editorToImGui.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      editorToImGui.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      editorToImGui.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      editorToImGui.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      editorToImGui.image = s_Data.EditorColors[s_Data.ImageIndex].Image;
      editorToImGui.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      editorToImGui.subresourceRange.levelCount = 1;
      editorToImGui.subresourceRange.layerCount = 1;
      editorToImGui.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      editorToImGui.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                           0, nullptr, 0, nullptr, 1, &editorToImGui);
    }
    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass = s_Data.PresentRenderPass;
    begin.framebuffer = s_Data.PresentFramebuffers[s_Data.ImageIndex];
    begin.renderArea.extent = s_Data.Extent;
    begin.clearValueCount = 1;
    begin.pClearValues = &clear;
    vkCmdBeginRenderPass(command, &begin, VK_SUBPASS_CONTENTS_INLINE);
    s_Data.PresentPassActive = true;
    if (!s_Data.RenderForEditor) DrawPostProcess();
  }

  void QueueWireSphere(const glm::vec3& center, const float radius,
                       const glm::vec4& color, const uint32_t segments = 24)
  {
    for (uint32_t segment = 0; segment < segments; ++segment)
    {
      const float a0 = glm::two_pi<float>() * static_cast<float>(segment) /
                       static_cast<float>(segments);
      const float a1 = glm::two_pi<float>() * static_cast<float>(segment + 1) /
                       static_cast<float>(segments);
      const float c0 = std::cos(a0) * radius;
      const float s0 = std::sin(a0) * radius;
      const float c1 = std::cos(a1) * radius;
      const float s1 = std::sin(a1) * radius;
      VulkanRenderer::DrawDebugLine(center + glm::vec3(c0, s0, 0.0f),
                                    center + glm::vec3(c1, s1, 0.0f), color);
      VulkanRenderer::DrawDebugLine(center + glm::vec3(c0, 0.0f, s0),
                                    center + glm::vec3(c1, 0.0f, s1), color);
      VulkanRenderer::DrawDebugLine(center + glm::vec3(0.0f, c0, s0),
                                    center + glm::vec3(0.0f, c1, s1), color);
    }
  }

  void DrawNativeDebugVisualizations()
  {
    const RenderDebugSettings& debug = RenderBackend::DebugSettings();
    if (!debug.Physics && !debug.Lights && !debug.CullingBounds) return;
    VulkanRenderer::BeginDebugLines();
    if (debug.Physics) VulkanRenderer::DrawPhysicsDebug();
    if (debug.CullingBounds)
    {
      const RenderFrustum frustum(Camera::GetViewProjection());
      for (const std::string& name : ModelManager::GetModelNames())
      {
        const auto model = ModelManager::GetModel(name);
        if (!model || !model->m_IsRendered) continue;
        for (const glm::mat4& transform : model->m_InstanceTransforms)
        {
          const WorldBoundingSphere sphere = CalculateWorldBoundingSphere(*model, transform);
          const bool visible = frustum.IntersectsSphere(sphere.center, sphere.radius);
          QueueWireSphere(sphere.center, sphere.radius,
            visible ? glm::vec4(0.15f, 1.0f, 0.3f, 0.8f)
                    : glm::vec4(1.0f, 0.2f, 0.15f, 0.8f));
        }
      }
    }
    if (debug.Lights)
      for (const RenderLight& light : RenderBackend::Lights())
      {
        const glm::vec4 color(light.Color, 0.9f);
        if (light.Type == LightType::DIRECT)
        {
          const glm::vec3 origin = Camera::GetPosition() +
                                   Camera::GetForwardDirection() * 8.0f;
          const glm::vec3 direction = glm::length(light.Direction) > 0.0001f
            ? glm::normalize(light.Direction) : glm::vec3(0.0f, -1.0f, 0.0f);
          QueueWireSphere(origin, 0.5f, color);
          VulkanRenderer::DrawDebugLine(origin, origin + direction * 12.0f, color);
        }
        else
        {
          QueueWireSphere(light.Position, 0.5f, color);
          VulkanRenderer::DrawDebugLine(light.Position - glm::vec3(1, 0, 0),
                                        light.Position + glm::vec3(1, 0, 0), color);
          VulkanRenderer::DrawDebugLine(light.Position - glm::vec3(0, 1, 0),
                                        light.Position + glm::vec3(0, 1, 0), color);
          VulkanRenderer::DrawDebugLine(light.Position - glm::vec3(0, 0, 1),
                                        light.Position + glm::vec3(0, 0, 1), color);
          if (light.Type == LightType::POINT)
            QueueWireSphere(light.Position, 20.0f, glm::vec4(light.Color, 0.2f), 32);
        }
      }
    VulkanRenderer::EndDebugLines();
  }
}

bool VulkanRenderer::Init(const uint32_t width, const uint32_t height)
{
  if (s_Data.Initialized) return true;
  s_Data.RequestedWidth = width;
  s_Data.RequestedHeight = height;
  try
  {
    CreateInstance();
    Check(glfwCreateWindowSurface(s_Data.Instance, Window::GetWindowPtr(), nullptr,
                                  &s_Data.Surface), "glfwCreateWindowSurface");
    PickPhysicalDevice();
    CreateDevice();
    CreateCommandsAndSync();
    CreateSwapchain();
    s_Data.Initialized = true;
    return true;
  }
  catch (const std::exception& error)
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "Vulkan initialization failed: %s", error.what());
    Shutdown();
    return false;
  }
}

void VulkanRenderer::Shutdown()
{
  if (s_Data.Device) vkDeviceWaitIdle(s_Data.Device);
  ShutdownImGui();
  ShutdownSceneRenderer();
  DestroySwapchain();
  if (s_Data.RenderPass) vkDestroyRenderPass(s_Data.Device, s_Data.RenderPass, nullptr);
  if (s_Data.DepthPrepassRenderPass)
    vkDestroyRenderPass(s_Data.Device, s_Data.DepthPrepassRenderPass, nullptr);
  if (s_Data.LightingRenderPass)
    vkDestroyRenderPass(s_Data.Device, s_Data.LightingRenderPass, nullptr);
  if (s_Data.ParticleRenderPass)
    vkDestroyRenderPass(s_Data.Device, s_Data.ParticleRenderPass, nullptr);
  if (s_Data.PresentRenderPass)
    vkDestroyRenderPass(s_Data.Device, s_Data.PresentRenderPass, nullptr);
  if (s_Data.EditorRenderPass)
    vkDestroyRenderPass(s_Data.Device, s_Data.EditorRenderPass, nullptr);
  for (uint32_t frame = 0; frame < FramesInFlight; ++frame)
  {
    if (s_Data.ImageAvailable[frame]) vkDestroySemaphore(s_Data.Device, s_Data.ImageAvailable[frame], nullptr);
    if (s_Data.FrameFences[frame]) vkDestroyFence(s_Data.Device, s_Data.FrameFences[frame], nullptr);
  }
  if (s_Data.CommandPool) vkDestroyCommandPool(s_Data.Device, s_Data.CommandPool, nullptr);
  if (s_Data.Device) vkDestroyDevice(s_Data.Device, nullptr);
  if (s_Data.Surface) vkDestroySurfaceKHR(s_Data.Instance, s_Data.Surface, nullptr);
  if (s_Data.DebugMessenger)
  {
    const auto destroyMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
      vkGetInstanceProcAddr(s_Data.Instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (destroyMessenger) destroyMessenger(s_Data.Instance, s_Data.DebugMessenger, nullptr);
  }
  if (s_Data.Instance) vkDestroyInstance(s_Data.Instance, nullptr);
  s_Data = {};
}

bool VulkanRenderer::Resize(const uint32_t width, const uint32_t height)
{
  if (!s_Data.Initialized || width == 0 || height == 0 || s_Data.FrameStarted) return false;
  if (width != s_Data.RequestedWidth || height != s_Data.RequestedHeight)
  {
    s_Data.RequestedWidth = width;
    s_Data.RequestedHeight = height;
    s_Data.SwapchainDirty = true;
  }
  if (!s_Data.SwapchainDirty) return true;
  try { RecreateSwapchain(); return true; }
  catch (const std::exception& error)
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "Vulkan resize failed: %s", error.what());
    return false;
  }
}

bool VulkanRenderer::BeginFrame()
{
  if (!s_Data.Initialized || s_Data.FrameStarted) return false;
  try
  {
    const VkFence frameFence = s_Data.FrameFences[s_Data.Frame];
    Check(vkWaitForFences(s_Data.Device, 1, &frameFence, VK_TRUE, UINT64_MAX),
          "vkWaitForFences");
    VkResult acquired = vkAcquireNextImageKHR(s_Data.Device, s_Data.Swapchain, UINT64_MAX,
      s_Data.ImageAvailable[s_Data.Frame], VK_NULL_HANDLE, &s_Data.ImageIndex);
    if (acquired == VK_ERROR_OUT_OF_DATE_KHR)
    {
      RecreateSwapchain();
      acquired = vkAcquireNextImageKHR(s_Data.Device, s_Data.Swapchain, UINT64_MAX,
        s_Data.ImageAvailable[s_Data.Frame], VK_NULL_HANDLE, &s_Data.ImageIndex);
    }
    if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR)
      Check(acquired, "vkAcquireNextImageKHR");
    if (acquired == VK_SUBOPTIMAL_KHR) s_Data.SwapchainDirty = true;
    if (s_Data.ImagesInFlight[s_Data.ImageIndex] != VK_NULL_HANDLE)
      Check(vkWaitForFences(s_Data.Device, 1, &s_Data.ImagesInFlight[s_Data.ImageIndex],
                            VK_TRUE, UINT64_MAX), "vkWaitForFences (swapchain image)");
    s_Data.ImagesInFlight[s_Data.ImageIndex] = frameFence;
    Check(vkResetFences(s_Data.Device, 1, &frameFence), "vkResetFences");

    VkCommandBuffer command = s_Data.CommandBuffers[s_Data.Frame];
    Check(vkResetCommandBuffer(command, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    Check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");
    s_Data.FrameStarted = true;
    s_Data.RenderPassActive = false;
    s_Data.DepthPrepared = false;
    s_Data.ParticlePassActive = false;
    s_Data.PresentPassActive = false;
    s_Data.RenderForEditor = false;
    s_Data.ClearColor = {0.008f, 0.012f, 0.025f, 1.0f};
    s_Data.CurrentEffects = RenderBackend::GetEffectSettings();
    return true;
  }
  catch (const std::exception& error)
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "Vulkan frame start failed: %s", error.what());
    return false;
  }
}

bool VulkanRenderer::EndFrame(const bool vSync)
{
  if (!s_Data.Initialized || !s_Data.FrameStarted) return false;
  try
  {
    BeginPresentRenderPass();
    if (s_Data.ImGuiFrameActive)
    {
      ImGui::Render();
      RenderImGuiDrawData();
    }
    VkCommandBuffer command = s_Data.CommandBuffers[s_Data.Frame];
    if (s_Data.PresentPassActive) vkCmdEndRenderPass(command);
    Check(vkEndCommandBuffer(command), "vkEndCommandBuffer");
    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &s_Data.ImageAvailable[s_Data.Frame];
    submit.pWaitDstStageMask = &waitStage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &s_Data.RenderComplete[s_Data.ImageIndex];
    Check(vkQueueSubmit(s_Data.GraphicsQueue, 1, &submit, s_Data.FrameFences[s_Data.Frame]),
          "vkQueueSubmit");
    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &s_Data.RenderComplete[s_Data.ImageIndex];
    present.swapchainCount = 1;
    present.pSwapchains = &s_Data.Swapchain;
    present.pImageIndices = &s_Data.ImageIndex;
    const VkResult result = vkQueuePresentKHR(s_Data.PresentQueue, &present);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
      s_Data.SwapchainDirty = true;
    else Check(result, "vkQueuePresentKHR");
    if (vSync != s_Data.VSync)
    {
      s_Data.VSync = vSync;
      s_Data.SwapchainDirty = true;
    }
    s_Data.FrameStarted = false;
    s_Data.RenderPassActive = false;
    s_Data.ParticlePassActive = false;
    s_Data.PresentPassActive = false;
    s_Data.Frame = (s_Data.Frame + 1) % FramesInFlight;
    return !s_Data.ValidationError;
  }
  catch (const std::exception& error)
  {
    s_Data.FrameStarted = false;
    s_Data.RenderPassActive = false;
    s_Data.ParticlePassActive = false;
    s_Data.PresentPassActive = false;
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "Vulkan presentation failed: %s", error.what());
    return false;
  }
}

bool VulkanRenderer::InitSceneRenderer()
{
  if (!s_Data.Initialized) return false;
  if (s_Data.SceneInitialized) return true;
  try
  {
    CreateSceneDescriptors();
    CreateShadowRenderPass();
    CreateShadowAttachments(RenderEffectSettings{}.DirectionalShadowResolution());
    CreatePointShadowRenderPass();
    CreatePointShadowAttachments(RenderEffectSettings{}.PointShadowResolution());
    const SceneFrameData emptyFrame{};
    for (GPUBuffer& frame : s_Data.SceneFrames)
      frame = CreateUploadBuffer(&emptyFrame, sizeof(emptyFrame),
                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    const std::vector<ParticleVertex> emptyParticles(MaxParticleVertices);
    for (GPUBuffer& buffer : s_Data.ParticleBuffers)
      buffer = CreateUploadBuffer(emptyParticles.data(),
        emptyParticles.size() * sizeof(ParticleVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    const std::vector<DebugLineVertex> emptyDebugLines(MaxDebugLineVertices);
    for (GPUBuffer& buffer : s_Data.DebugLineBuffers)
      buffer = CreateUploadBuffer(emptyDebugLines.data(),
        emptyDebugLines.size() * sizeof(DebugLineVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    constexpr std::array<glm::vec2, 3> postVertices{
      glm::vec2(-1.0f, -1.0f), glm::vec2(3.0f, -1.0f), glm::vec2(-1.0f, 3.0f)};
    s_Data.PostVertices = CreateUploadBuffer(
      postVertices.data(), sizeof(postVertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    constexpr std::array<uint8_t, 24> blackCube{};
    s_Data.SkyboxTexture = CreateCubeTextureRGBA(blackCube.data(), 1, 1);
    CreateScenePipeline();
    CreateDepthPrepassPipeline();
    CreateShadowPipeline();
    CreatePointShadowPipeline();
    CreateParticlePipeline();
    CreateDebugLinePipeline();
    CreatePostPipeline();
    CreateLightingPipeline();
    CreateTilePipeline();
    CreateCullPipeline();
    CreateHiZPipeline();
    s_Data.SceneInitialized = true;
    return true;
  }
  catch (const std::exception& error)
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "Vulkan scene pipeline initialization failed: %s", error.what());
    ShutdownSceneRenderer();
    return false;
  }
}

void VulkanRenderer::ResetSceneResources()
{
  if (!s_Data.Device) return;
  vkDeviceWaitIdle(s_Data.Device);
  for (auto& mesh : s_Data.Meshes)
  {
    DestroyBuffer(mesh.second.Vertices);
    DestroyBuffer(mesh.second.Indices);
  }
  s_Data.Meshes.clear();
  for (auto& model : s_Data.Models)
    for (GPUBuffer& bones : model.second.Bones)
      DestroyBuffer(bones);
  s_Data.Models.clear();
  for (auto& texture : s_Data.Textures)
    DestroyTexture(texture.second);
  s_Data.Textures.clear();
  s_Data.HasSkybox = false;
  if (s_Data.SceneDescriptorPool)
    Check(vkResetDescriptorPool(s_Data.Device, s_Data.SceneDescriptorPool, 0),
          "vkResetDescriptorPool (scene)");
}

void VulkanRenderer::ShutdownSceneRenderer()
{
  if (!s_Data.Device) return;
  vkDeviceWaitIdle(s_Data.Device);
  ResetSceneResources();
  if (s_Data.ScenePipeline)
    vkDestroyPipeline(s_Data.Device, s_Data.ScenePipeline, nullptr);
  if (s_Data.DepthPrepassPipeline)
    vkDestroyPipeline(s_Data.Device, s_Data.DepthPrepassPipeline, nullptr);
  if (s_Data.ShadowPipeline)
    vkDestroyPipeline(s_Data.Device, s_Data.ShadowPipeline, nullptr);
  if (s_Data.PointShadowPipeline)
    vkDestroyPipeline(s_Data.Device, s_Data.PointShadowPipeline, nullptr);
  if (s_Data.ParticlePipeline)
    vkDestroyPipeline(s_Data.Device, s_Data.ParticlePipeline, nullptr);
  if (s_Data.DebugLinePipeline)
    vkDestroyPipeline(s_Data.Device, s_Data.DebugLinePipeline, nullptr);
  if (s_Data.PostPipeline)
    vkDestroyPipeline(s_Data.Device, s_Data.PostPipeline, nullptr);
  if (s_Data.LightingPipeline)
    vkDestroyPipeline(s_Data.Device, s_Data.LightingPipeline, nullptr);
  if (s_Data.TilePipeline)
    vkDestroyPipeline(s_Data.Device, s_Data.TilePipeline, nullptr);
  if (s_Data.CullPipeline)
    vkDestroyPipeline(s_Data.Device, s_Data.CullPipeline, nullptr);
  if (s_Data.HiZPipeline)
    vkDestroyPipeline(s_Data.Device, s_Data.HiZPipeline, nullptr);
  if (s_Data.ScenePipelineLayout)
    vkDestroyPipelineLayout(s_Data.Device, s_Data.ScenePipelineLayout, nullptr);
  if (s_Data.DepthPrepassPipelineLayout)
    vkDestroyPipelineLayout(s_Data.Device, s_Data.DepthPrepassPipelineLayout, nullptr);
  if (s_Data.ShadowPipelineLayout)
    vkDestroyPipelineLayout(s_Data.Device, s_Data.ShadowPipelineLayout, nullptr);
  if (s_Data.PointShadowPipelineLayout)
    vkDestroyPipelineLayout(s_Data.Device, s_Data.PointShadowPipelineLayout, nullptr);
  if (s_Data.ParticlePipelineLayout)
    vkDestroyPipelineLayout(s_Data.Device, s_Data.ParticlePipelineLayout, nullptr);
  if (s_Data.DebugLinePipelineLayout)
    vkDestroyPipelineLayout(s_Data.Device, s_Data.DebugLinePipelineLayout, nullptr);
  if (s_Data.PostPipelineLayout)
    vkDestroyPipelineLayout(s_Data.Device, s_Data.PostPipelineLayout, nullptr);
  if (s_Data.LightingPipelineLayout)
    vkDestroyPipelineLayout(s_Data.Device, s_Data.LightingPipelineLayout, nullptr);
  if (s_Data.TilePipelineLayout)
    vkDestroyPipelineLayout(s_Data.Device, s_Data.TilePipelineLayout, nullptr);
  if (s_Data.CullPipelineLayout)
    vkDestroyPipelineLayout(s_Data.Device, s_Data.CullPipelineLayout, nullptr);
  if (s_Data.HiZPipelineLayout)
    vkDestroyPipelineLayout(s_Data.Device, s_Data.HiZPipelineLayout, nullptr);
  for (GPUBuffer& frame : s_Data.SceneFrames) DestroyBuffer(frame);
  for (GPUBuffer& buffer : s_Data.ParticleBuffers) DestroyBuffer(buffer);
  for (GPUBuffer& buffer : s_Data.DebugLineBuffers) DestroyBuffer(buffer);
  for (GPUCullFrame& frame : s_Data.CullFrames)
  {
    DestroyBuffer(frame.SourceTransforms);
    DestroyBuffer(frame.Inputs);
    DestroyBuffer(frame.VisibleTransforms);
    DestroyBuffer(frame.Commands);
  }
  DestroyBuffer(s_Data.PostVertices);
  DestroyShadowAttachments();
  DestroyPointShadowAttachments();
  if (s_Data.ShadowRenderPass)
    vkDestroyRenderPass(s_Data.Device, s_Data.ShadowRenderPass, nullptr);
  if (s_Data.PointShadowRenderPass)
    vkDestroyRenderPass(s_Data.Device, s_Data.PointShadowRenderPass, nullptr);
  DestroyTexture(s_Data.WhiteTexture);
  DestroyTexture(s_Data.NeutralNormalTexture);
  DestroyTexture(s_Data.BlackTexture);
  DestroyTexture(s_Data.SkyboxTexture);
  if (s_Data.MaterialSampler)
    vkDestroySampler(s_Data.Device, s_Data.MaterialSampler, nullptr);
  if (s_Data.ShadowSampler)
    vkDestroySampler(s_Data.Device, s_Data.ShadowSampler, nullptr);
  if (s_Data.SceneDescriptorPool)
    vkDestroyDescriptorPool(s_Data.Device, s_Data.SceneDescriptorPool, nullptr);
  if (s_Data.SceneDescriptorSetLayout)
    vkDestroyDescriptorSetLayout(s_Data.Device, s_Data.SceneDescriptorSetLayout, nullptr);
  if (s_Data.PostSampler)
    vkDestroySampler(s_Data.Device, s_Data.PostSampler, nullptr);
  if (s_Data.PostDescriptorPool)
    vkDestroyDescriptorPool(s_Data.Device, s_Data.PostDescriptorPool, nullptr);
  if (s_Data.PostDescriptorSetLayout)
    vkDestroyDescriptorSetLayout(s_Data.Device, s_Data.PostDescriptorSetLayout, nullptr);
  if (s_Data.LightingDescriptorPool)
    vkDestroyDescriptorPool(s_Data.Device, s_Data.LightingDescriptorPool, nullptr);
  if (s_Data.LightingDescriptorSetLayout)
    vkDestroyDescriptorSetLayout(s_Data.Device, s_Data.LightingDescriptorSetLayout, nullptr);
  if (s_Data.TileDescriptorPool)
    vkDestroyDescriptorPool(s_Data.Device, s_Data.TileDescriptorPool, nullptr);
  if (s_Data.TileDescriptorSetLayout)
    vkDestroyDescriptorSetLayout(s_Data.Device, s_Data.TileDescriptorSetLayout, nullptr);
  if (s_Data.CullDescriptorPool)
    vkDestroyDescriptorPool(s_Data.Device, s_Data.CullDescriptorPool, nullptr);
  if (s_Data.CullDescriptorSetLayout)
    vkDestroyDescriptorSetLayout(s_Data.Device, s_Data.CullDescriptorSetLayout, nullptr);
  if (s_Data.HiZDescriptorPool)
    vkDestroyDescriptorPool(s_Data.Device, s_Data.HiZDescriptorPool, nullptr);
  if (s_Data.HiZDescriptorSetLayout)
    vkDestroyDescriptorSetLayout(s_Data.Device, s_Data.HiZDescriptorSetLayout, nullptr);
  s_Data.ScenePipeline = VK_NULL_HANDLE;
  s_Data.DepthPrepassPipeline = VK_NULL_HANDLE;
  s_Data.ShadowPipeline = VK_NULL_HANDLE;
  s_Data.PointShadowPipeline = VK_NULL_HANDLE;
  s_Data.ParticlePipeline = VK_NULL_HANDLE;
  s_Data.DebugLinePipeline = VK_NULL_HANDLE;
  s_Data.PostPipeline = VK_NULL_HANDLE;
  s_Data.LightingPipeline = VK_NULL_HANDLE;
  s_Data.TilePipeline = VK_NULL_HANDLE;
  s_Data.CullPipeline = VK_NULL_HANDLE;
  s_Data.HiZPipeline = VK_NULL_HANDLE;
  s_Data.ScenePipelineLayout = VK_NULL_HANDLE;
  s_Data.DepthPrepassPipelineLayout = VK_NULL_HANDLE;
  s_Data.ShadowPipelineLayout = VK_NULL_HANDLE;
  s_Data.PointShadowPipelineLayout = VK_NULL_HANDLE;
  s_Data.ParticlePipelineLayout = VK_NULL_HANDLE;
  s_Data.DebugLinePipelineLayout = VK_NULL_HANDLE;
  s_Data.PostPipelineLayout = VK_NULL_HANDLE;
  s_Data.LightingPipelineLayout = VK_NULL_HANDLE;
  s_Data.TilePipelineLayout = VK_NULL_HANDLE;
  s_Data.CullPipelineLayout = VK_NULL_HANDLE;
  s_Data.HiZPipelineLayout = VK_NULL_HANDLE;
  s_Data.ShadowRenderPass = VK_NULL_HANDLE;
  s_Data.PointShadowRenderPass = VK_NULL_HANDLE;
  s_Data.MaterialSampler = VK_NULL_HANDLE;
  s_Data.ShadowSampler = VK_NULL_HANDLE;
  s_Data.SceneDescriptorPool = VK_NULL_HANDLE;
  s_Data.SceneDescriptorSetLayout = VK_NULL_HANDLE;
  s_Data.PostSampler = VK_NULL_HANDLE;
  s_Data.PostDescriptorPool = VK_NULL_HANDLE;
  s_Data.PostDescriptorSetLayout = VK_NULL_HANDLE;
  s_Data.LightingDescriptorPool = VK_NULL_HANDLE;
  s_Data.LightingDescriptorSetLayout = VK_NULL_HANDLE;
  s_Data.TileDescriptorPool = VK_NULL_HANDLE;
  s_Data.TileDescriptorSetLayout = VK_NULL_HANDLE;
  s_Data.CullDescriptorPool = VK_NULL_HANDLE;
  s_Data.CullDescriptorSetLayout = VK_NULL_HANDLE;
  s_Data.HiZDescriptorPool = VK_NULL_HANDLE;
  s_Data.HiZDescriptorSetLayout = VK_NULL_HANDLE;
  s_Data.HasSkybox = false;
  s_Data.ModelPreviews.clear();
  s_Data.SceneInitialized = false;
}

bool VulkanRenderer::UploadModel(const std::shared_ptr<Model>& model)
{
  if (!s_Data.SceneInitialized || !model) return false;
  try
  {
    auto gpuModel = s_Data.Models.find(model.get());
    if (gpuModel == s_Data.Models.end())
    {
      std::array<glm::mat4, MAX_BONES> identityBones;
      identityBones.fill(glm::mat4(1.0f));
      GPUModel resources;
      try
      {
        for (GPUBuffer& bones : resources.Bones)
          bones = CreateUploadBuffer(identityBones.data(), sizeof(identityBones),
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      }
      catch (...)
      {
        for (GPUBuffer& bones : resources.Bones) DestroyBuffer(bones);
        throw;
      }
      gpuModel = s_Data.Models.emplace(model.get(), std::move(resources)).first;
    }
    for (Mesh& mesh : model->GetMeshes())
    {
      if (mesh.m_Vertices.empty() || mesh.m_Indices.empty()) continue;
      if (s_Data.Meshes.contains(&mesh)) continue;
      GPUMesh gpu;
      try
      {
        gpu.Vertices = CreateUploadBuffer(mesh.m_Vertices.data(),
          mesh.m_Vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        gpu.Indices = CreateUploadBuffer(mesh.m_Indices.data(),
          mesh.m_Indices.size() * sizeof(GLuint), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        gpu.IndexCount = static_cast<uint32_t>(mesh.m_Indices.size());
        const GPUTexture& diffuse = ResolveTexture(mesh.m_DiffuseTexture, s_Data.WhiteTexture);
        const GPUTexture& normal = ResolveTexture(mesh.m_NormalTexture,
                                                  s_Data.NeutralNormalTexture);
        const GPUTexture& specular = ResolveTexture(mesh.m_SpecularTexture,
                                                    s_Data.BlackTexture);
        for (uint32_t frame = 0; frame < FramesInFlight; ++frame)
          gpu.Materials[frame] = CreateMaterialDescriptor(
            diffuse, normal, specular, gpuModel->second.Bones[frame],
            s_Data.SceneFrames[frame], s_Data.Shadows[frame].Texture,
            s_Data.CullFrames[frame].VisibleTransforms);
        s_Data.Meshes.emplace(&mesh, std::move(gpu));
      }
      catch (...)
      {
        DestroyBuffer(gpu.Vertices);
        DestroyBuffer(gpu.Indices);
        throw;
      }
    }
    return true;
  }
  catch (const std::exception& error)
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "Vulkan model upload failed: %s", error.what());
    return false;
  }
}

bool VulkanRenderer::UploadSkybox(const std::shared_ptr<Texture>& cubemap)
{
  if (!s_Data.SceneInitialized || !cubemap) return false;
  try
  {
    GPUTexture uploaded = CreateCubeTexture(*cubemap);
    Check(vkDeviceWaitIdle(s_Data.Device), "vkDeviceWaitIdle (skybox upload)");
    DestroyTexture(s_Data.SkyboxTexture);
    s_Data.SkyboxTexture = uploaded;
    s_Data.HasSkybox = true;
    UpdateSkyboxDescriptorImages();
    return true;
  }
  catch (const std::exception& error)
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "Vulkan skybox upload failed: %s", error.what());
    return false;
  }
}

void VulkanRenderer::SetModelPreviews(const std::vector<RenderModelPreview>& previews)
{
  s_Data.ModelPreviews = previews;
}

void VulkanRenderer::DrawScene(DeltaTime& dt, const std::function<void()>& sceneLogic,
                               const bool advanceSimulation, const bool renderForEditor,
                               const RenderEffectSettings& effects)
{
  if (!s_Data.FrameStarted || !s_Data.SceneInitialized) return;
  s_Data.CurrentEffects = effects;
  s_Data.RenderForEditor = renderForEditor;
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

  EnsureShadowSize(effects.DirectionalShadowResolution());
  EnsurePointShadowSize(effects.PointShadowResolution());
  UpdateBoneBuffers();
  const glm::mat4 lightViewProjection = CalculateLightViewProjection();
  SelectPointShadowLights(effects);
  UpdateSceneFrame(effects, lightViewProjection);
  const bool directionalCulled = effects.ShadowQuality != GraphicsQuality::Off &&
    PrepareGPUCull(false, lightViewProjection, false);
  DrawShadowPass(effects, lightViewProjection, directionalCulled);
  DrawPointShadowPass(effects);
  glm::mat4 cameraViewProjectionCorrection(1.0f);
  cameraViewProjectionCorrection[1][1] = -1.0f;
  cameraViewProjectionCorrection[2][2] = 0.5f;
  cameraViewProjectionCorrection[3][2] = 0.5f;
  const glm::mat4 cameraViewProjection =
    cameraViewProjectionCorrection * Camera::GetViewProjection();
  const bool prepassCulled = PrepareGPUCull(false, cameraViewProjection, true);
  DrawDepthPrepass(prepassCulled);
  BuildHiZ();
  const bool gpuCulled = PrepareGPUCull(true, cameraViewProjection, true);
  BeginMainRenderPass();

  if (s_Data.ScenePipeline && gpuCulled)
  {
    VkCommandBuffer command = s_Data.CommandBuffers[s_Data.Frame];
    const VkViewport viewport{
      0.0f, 0.0f, static_cast<float>(s_Data.Extent.width),
      static_cast<float>(s_Data.Extent.height), 0.0f, 1.0f};
    const VkRect2D scissor{{0, 0}, s_Data.Extent};
    vkCmdSetViewport(command, 0, 1, &viewport);
    vkCmdSetScissor(command, 0, 1, &scissor);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, s_Data.ScenePipeline);

    glm::mat4 clipCorrection(1.0f);
    clipCorrection[1][1] = -1.0f;
    clipCorrection[2][2] = 0.5f;
    clipCorrection[3][2] = 0.5f;
    const ScenePushConstants constants{
      clipCorrection * Camera::GetViewProjection(), glm::vec4(1.0f, 0.0f, 0.0f, 0.0f)};
    for (const CullDrawRecord& record : s_Data.CullDrawRecords)
    {
      ScenePushConstants drawConstants = constants;
      drawConstants.DrawParams.x = record.Brightness;
      drawConstants.DrawParams.y = static_cast<float>(record.TransformOffset);
      vkCmdPushConstants(command, s_Data.ScenePipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                         0, sizeof(drawConstants), &drawConstants);
      const VkDeviceSize offset = 0;
      vkCmdBindVertexBuffers(command, 0, 1, &record.Mesh->Vertices.Buffer, &offset);
      vkCmdBindIndexBuffer(command, record.Mesh->Indices.Buffer, 0, VK_INDEX_TYPE_UINT32);
      vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              s_Data.ScenePipelineLayout, 0, 1,
                              &record.Descriptor, 0, nullptr);
      vkCmdDrawIndexedIndirect(command, s_Data.CullFrames[s_Data.Frame].Commands.Buffer,
        static_cast<VkDeviceSize>(record.CommandIndex) * sizeof(VkDrawIndexedIndirectCommand),
        1, sizeof(VkDrawIndexedIndirectCommand));
    }
  }
  DrawNativeDebugVisualizations();
  ParticleRenderer::UpdateAndRender(dt);
  BeginPresentRenderPass();
}

bool VulkanRenderer::DrawParticles(const std::vector<ParticleRenderInstance>& instances)
{
  if (!s_Data.FrameStarted || !s_Data.ParticlePipeline || instances.empty()) return false;
  const size_t instanceCount = std::min(
    instances.size(), static_cast<size_t>(MaxParticleVertices / 6));
  std::vector<ParticleVertex> vertices;
  vertices.reserve(instanceCount * 6);
  static constexpr std::array<glm::vec2, 6> corners{
    glm::vec2(-0.5f, -0.5f), glm::vec2(0.5f, -0.5f), glm::vec2(0.5f, 0.5f),
    glm::vec2(0.5f, 0.5f), glm::vec2(-0.5f, 0.5f), glm::vec2(-0.5f, -0.5f)};
  for (size_t index = 0; index < instanceCount; ++index)
  {
    const ParticleRenderInstance& instance = instances[index];
    const float cosine = std::cos(instance.Rotation);
    const float sine = std::sin(instance.Rotation);
    for (const glm::vec2 corner : corners)
    {
      const glm::vec2 rotated(cosine * corner.x - sine * corner.y,
                              sine * corner.x + cosine * corner.y);
      const glm::vec3 position = glm::vec3(instance.PositionAndSize) +
        glm::vec3(instance.RightAndStyle) * rotated.x * instance.PositionAndSize.w +
        glm::vec3(instance.Up) * rotated.y * instance.PositionAndSize.w;
      vertices.push_back({position, instance.Color, corner * 2.0f, instance.RightAndStyle.w});
    }
  }
  void* mapped = nullptr;
  const VkDeviceSize bytes = vertices.size() * sizeof(ParticleVertex);
  Check(vkMapMemory(s_Data.Device, s_Data.ParticleBuffers[s_Data.Frame].Memory,
                    0, bytes, 0, &mapped), "vkMapMemory (particles)");
  std::memcpy(mapped, vertices.data(), static_cast<size_t>(bytes));
  vkUnmapMemory(s_Data.Device, s_Data.ParticleBuffers[s_Data.Frame].Memory);

  BeginParticleRenderPass();
  VkCommandBuffer command = s_Data.CommandBuffers[s_Data.Frame];
  const VkViewport viewport{0.0f, 0.0f, static_cast<float>(s_Data.Extent.width),
                            static_cast<float>(s_Data.Extent.height), 0.0f, 1.0f};
  const VkRect2D scissor{{0, 0}, s_Data.Extent};
  vkCmdSetViewport(command, 0, 1, &viewport);
  vkCmdSetScissor(command, 0, 1, &scissor);
  vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, s_Data.ParticlePipeline);
  const VkDeviceSize offset = 0;
  vkCmdBindVertexBuffers(command, 0, 1,
                         &s_Data.ParticleBuffers[s_Data.Frame].Buffer, &offset);
  glm::mat4 correction(1.0f);
  correction[1][1] = -1.0f;
  correction[2][2] = 0.5f;
  correction[3][2] = 0.5f;
  const glm::mat4 viewProjection = correction * Camera::GetViewProjection();
  vkCmdPushConstants(command, s_Data.ParticlePipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                     0, sizeof(viewProjection), &viewProjection);
  vkCmdDraw(command, static_cast<uint32_t>(vertices.size()), 1, 0, 0);
  return true;
}

bool VulkanRenderer::BeginDebugLines()
{
  if (!s_Data.FrameStarted || !s_Data.DebugLinePipeline) return false;
  s_Data.PendingDebugLines.clear();
  return true;
}

bool VulkanRenderer::DrawDebugLine(const glm::vec3& start, const glm::vec3& end,
                                   const glm::vec4& color)
{
  if (!s_Data.FrameStarted ||
      s_Data.PendingDebugLines.size() + 2 > MaxDebugLineVertices) return false;
  s_Data.PendingDebugLines.push_back({start, color});
  s_Data.PendingDebugLines.push_back({end, color});
  return true;
}

bool VulkanRenderer::EndDebugLines()
{
  if (!s_Data.FrameStarted || !s_Data.DebugLinePipeline) return false;
  if (s_Data.PendingDebugLines.empty()) return true;
  const VkDeviceSize bytes = s_Data.PendingDebugLines.size() * sizeof(DebugLineVertex);
  void* mapped = nullptr;
  Check(vkMapMemory(s_Data.Device, s_Data.DebugLineBuffers[s_Data.Frame].Memory,
                    0, bytes, 0, &mapped), "vkMapMemory (debug lines)");
  std::memcpy(mapped, s_Data.PendingDebugLines.data(), static_cast<size_t>(bytes));
  vkUnmapMemory(s_Data.Device, s_Data.DebugLineBuffers[s_Data.Frame].Memory);
  BeginParticleRenderPass();
  VkCommandBuffer command = s_Data.CommandBuffers[s_Data.Frame];
  const VkViewport viewport{0.0f, 0.0f, static_cast<float>(s_Data.Extent.width),
                            static_cast<float>(s_Data.Extent.height), 0.0f, 1.0f};
  const VkRect2D scissor{{0, 0}, s_Data.Extent};
  vkCmdSetViewport(command, 0, 1, &viewport);
  vkCmdSetScissor(command, 0, 1, &scissor);
  vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, s_Data.DebugLinePipeline);
  const VkDeviceSize offset = 0;
  vkCmdBindVertexBuffers(command, 0, 1,
                         &s_Data.DebugLineBuffers[s_Data.Frame].Buffer, &offset);
  glm::mat4 correction(1.0f);
  correction[1][1] = -1.0f;
  correction[2][2] = 0.5f;
  correction[3][2] = 0.5f;
  const glm::mat4 viewProjection = correction * Camera::GetViewProjection();
  vkCmdPushConstants(command, s_Data.DebugLinePipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                     0, sizeof(viewProjection), &viewProjection);
  vkCmdDraw(command, static_cast<uint32_t>(s_Data.PendingDebugLines.size()), 1, 0, 0);
  s_Data.PendingDebugLines.clear();
  return true;
}

bool VulkanRenderer::DrawPhysicsDebug()
{
  if (!s_Data.FrameStarted) return false;
  constexpr glm::vec4 color(0.15f, 1.0f, 0.35f, 0.85f);
  for (const std::string& name : ModelManager::GetModelNames())
  {
    const auto model = ModelManager::GetModel(name);
    if (!model || model->GetPhysXMeshType() != MeshType::CONVEXMESH) continue;
    for (const glm::mat4& transform : model->m_InstanceTransforms)
      for (const Mesh& mesh : model->GetMeshes())
        for (size_t index = 0; index + 2 < mesh.m_Indices.size(); index += 3)
        {
          const glm::vec3 a = glm::vec3(transform *
            glm::vec4(mesh.m_Vertices[mesh.m_Indices[index]].Position, 1.0f));
          const glm::vec3 b = glm::vec3(transform *
            glm::vec4(mesh.m_Vertices[mesh.m_Indices[index + 1]].Position, 1.0f));
          const glm::vec3 c = glm::vec3(transform *
            glm::vec4(mesh.m_Vertices[mesh.m_Indices[index + 2]].Position, 1.0f));
          if (!DrawDebugLine(a, b, color) || !DrawDebugLine(b, c, color) ||
              !DrawDebugLine(c, a, color)) return true;
        }
  }
  return true;
}

uint64_t VulkanRenderer::CreateFontAtlas(const uint8_t* pixels,
                                         const uint32_t width,
                                         const uint32_t height)
{
  if (!pixels || width == 0 || height == 0 || !s_Data.ImGuiInitialized ||
      !s_Data.PostSampler) return 0;
  try
  {
    std::vector<uint8_t> rgba(static_cast<size_t>(width) * height * 4, 255);
    for (size_t index = 0; index < static_cast<size_t>(width) * height; ++index)
      rgba[index * 4 + 3] = pixels[index];
    GPUFontAtlas atlas;
    atlas.Texture = CreateTextureRGBA(rgba.data(), width, height);
    atlas.Descriptor = ImGui_ImplVulkan_AddTexture(
      s_Data.PostSampler, atlas.Texture.View, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (!atlas.Descriptor)
    {
      DestroyTexture(atlas.Texture);
      return 0;
    }
    const uint64_t handle = s_Data.NextFontAtlasHandle++;
    s_Data.FontAtlases.emplace(handle, atlas);
    return handle;
  }
  catch (const std::exception& error)
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__,
               "Vulkan font atlas upload failed: %s", error.what());
    return 0;
  }
}

void VulkanRenderer::DestroyFontAtlas(const uint64_t handle)
{
  const auto atlas = s_Data.FontAtlases.find(handle);
  if (atlas == s_Data.FontAtlases.end()) return;
  if (s_Data.Device) vkDeviceWaitIdle(s_Data.Device);
  if (s_Data.ImGuiInitialized && atlas->second.Descriptor)
    ImGui_ImplVulkan_RemoveTexture(atlas->second.Descriptor);
  DestroyTexture(atlas->second.Texture);
  s_Data.FontAtlases.erase(atlas);
}

bool VulkanRenderer::InitImGui()
{
  if (!s_Data.SceneInitialized || !s_Data.Device || !s_Data.RenderPass ||
      ImGui::GetCurrentContext() == nullptr)
    return false;
  if (s_Data.ImGuiInitialized) return true;
  try
  {
    constexpr std::array<VkDescriptorPoolSize, 11> sizes{{
      {VK_DESCRIPTOR_TYPE_SAMPLER, 256},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
      {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 256},
      {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 256},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 256},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 256},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 256},
      {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 256}
    }};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool.maxSets = 256u * static_cast<uint32_t>(sizes.size());
    pool.poolSizeCount = static_cast<uint32_t>(sizes.size());
    pool.pPoolSizes = sizes.data();
    Check(vkCreateDescriptorPool(s_Data.Device, &pool, nullptr,
                                 &s_Data.ImGuiDescriptorPool), "vkCreateDescriptorPool (ImGui)");
    ImGui_ImplVulkan_InitInfo info{};
    info.Instance = s_Data.Instance;
    info.PhysicalDevice = s_Data.PhysicalDevice;
    info.Device = s_Data.Device;
    info.QueueFamily = s_Data.GraphicsQueueFamily;
    info.Queue = s_Data.GraphicsQueue;
    info.DescriptorPool = s_Data.ImGuiDescriptorPool;
    info.MinImageCount = s_Data.MinImageCount;
    info.ImageCount = static_cast<uint32_t>(s_Data.Images.size());
    info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    info.CheckVkResultFn = ImGuiCheckResult;
    if (!ImGui_ImplVulkan_Init(&info, s_Data.PresentRenderPass))
      throw std::runtime_error("ImGui_ImplVulkan_Init failed");
    s_Data.ImGuiInitialized = true;
    UploadImGuiFonts();
    RegisterEditorTextures();
    return true;
  }
  catch (const std::exception& error)
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "Dear ImGui Vulkan initialization failed: %s", error.what());
    ShutdownImGui();
    return false;
  }
}

void VulkanRenderer::ShutdownImGui()
{
  if (s_Data.ImGuiInitialized)
  {
    if (s_Data.Device) vkDeviceWaitIdle(s_Data.Device);
    for (auto& [handle, atlas] : s_Data.FontAtlases)
    {
      if (atlas.Descriptor) ImGui_ImplVulkan_RemoveTexture(atlas.Descriptor);
      DestroyTexture(atlas.Texture);
    }
    s_Data.FontAtlases.clear();
    for (const VkDescriptorSet descriptor : s_Data.EditorDescriptors)
      if (descriptor) ImGui_ImplVulkan_RemoveTexture(descriptor);
    s_Data.EditorDescriptors.clear();
    ImGui_ImplVulkan_Shutdown();
    s_Data.ImGuiInitialized = false;
  }
  if (s_Data.ImGuiDescriptorPool)
  {
    vkDestroyDescriptorPool(s_Data.Device, s_Data.ImGuiDescriptorPool, nullptr);
    s_Data.ImGuiDescriptorPool = VK_NULL_HANDLE;
  }
}

void VulkanRenderer::BeginImGuiFrame()
{
  if (s_Data.ImGuiInitialized && !s_Data.ImGuiFrameActive)
  {
    ImGui_ImplVulkan_NewFrame();
    s_Data.ImGuiFrameActive = true;
  }
}

void VulkanRenderer::RenderImGuiDrawData()
{
  if (s_Data.ImGuiInitialized && s_Data.FrameStarted && ImGui::GetDrawData())
  {
    BeginPresentRenderPass();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), s_Data.CommandBuffers[s_Data.Frame]);
  }
  s_Data.ImGuiFrameActive = false;
}

void VulkanRenderer::RenderImGuiPlatformWindows()
{
  if (!s_Data.ImGuiInitialized) return;
  ImGui::UpdatePlatformWindows();
  ImGui::RenderPlatformWindowsDefault();
}

uint64_t VulkanRenderer::GetEditorTextureID()
{
  if (!s_Data.ImGuiInitialized ||
      s_Data.ImageIndex >= s_Data.EditorDescriptors.size()) return 0;
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
    s_Data.EditorDescriptors[s_Data.ImageIndex]));
}

bool VulkanRenderer::BeginScreenUI()
{
  if (!s_Data.ImGuiInitialized || !s_Data.FrameStarted) return false;
  BeginPresentRenderPass();
  if (!s_Data.ImGuiFrameActive)
  {
    BeginImGuiFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
  }
  return true;
}

bool VulkanRenderer::DrawScreenQuad(const glm::mat4& transform, const glm::vec4& color)
{
  if (!BeginScreenUI()) return false;
  static constexpr std::array<glm::vec4, 4> corners{
    glm::vec4(-0.5f, -0.5f, 0.0f, 1.0f), glm::vec4(0.5f, -0.5f, 0.0f, 1.0f),
    glm::vec4(0.5f, 0.5f, 0.0f, 1.0f), glm::vec4(-0.5f, 0.5f, 0.0f, 1.0f)};
  std::array<ImVec2, 4> positions{};
  for (size_t index = 0; index < corners.size(); ++index)
  {
    const glm::vec4 point = transform * corners[index];
    positions[index] = {point.x, static_cast<float>(s_Data.Extent.height) - point.y};
  }
  ImGui::GetBackgroundDrawList()->AddQuadFilled(
    positions[0], positions[1], positions[2], positions[3],
    ImGui::ColorConvertFloat4ToU32(ImVec4(color.r, color.g, color.b, color.a)));
  return true;
}

bool VulkanRenderer::DrawScreenText(const Font* font, const std::string& text,
                                    const glm::vec2& position, const float size,
                                    const glm::vec4& color)
{
  if (text.empty() || !BeginScreenUI()) return false;
  if (font && font->m_AtlasHandle != 0 && !font->m_Characters.empty())
  {
    const auto atlas = s_Data.FontAtlases.find(font->m_AtlasHandle);
    if (atlas != s_Data.FontAtlases.end())
    {
      float textWidth = 0.0f;
      for (const unsigned char character : text)
        if (const auto glyph = font->m_Characters.find(static_cast<char>(character));
            glyph != font->m_Characters.end())
          textWidth += static_cast<float>(glyph->second.Advance >> 6) * size;
      const float baselineY = (font->m_Descender - font->m_Ascender) * size * 0.5f;
      float cursorX = -textWidth * 0.5f;
      const ImU32 packedColor = ImGui::ColorConvertFloat4ToU32(
        ImVec4(color.r, color.g, color.b, color.a));
      const ImTextureID texture = reinterpret_cast<ImTextureID>(
        static_cast<uintptr_t>(reinterpret_cast<uintptr_t>(atlas->second.Descriptor)));
      ImDrawList* drawList = ImGui::GetBackgroundDrawList();
      for (const unsigned char character : text)
      {
        const auto glyphIt = font->m_Characters.find(static_cast<char>(character));
        if (glyphIt == font->m_Characters.end()) continue;
        const Character& glyph = glyphIt->second;
        const float x = position.x + cursorX + glyph.Bearing.x * size;
        const float y = position.y + baselineY + (glyph.Bearing.y - glyph.Size.y) * size;
        const float glyphWidth = glyph.Size.x * size;
        const float glyphHeight = glyph.Size.y * size;
        if (glyphWidth > 0.0f && glyphHeight > 0.0f)
        {
          const float top = static_cast<float>(s_Data.Extent.height) - (y + glyphHeight);
          const float bottom = static_cast<float>(s_Data.Extent.height) - y;
          const ImVec2 uvTopLeft(glyph.UVTopLeft.x, glyph.UVTopLeft.y);
          const ImVec2 uvBottomRight(glyph.UVBottomRight.x, glyph.UVBottomRight.y);
          drawList->AddImageQuad(texture,
            ImVec2(x, top), ImVec2(x + glyphWidth, top),
            ImVec2(x + glyphWidth, bottom), ImVec2(x, bottom),
            uvTopLeft, ImVec2(uvBottomRight.x, uvTopLeft.y),
            uvBottomRight, ImVec2(uvTopLeft.x, uvBottomRight.y), packedColor);
        }
        cursorX += static_cast<float>(glyph.Advance >> 6) * size;
      }
      return true;
    }
  }
  ImFont* imguiFont = ImGui::GetFont();
  const float fontSize = std::max(1.0f, 48.0f * size);
  const ImVec2 bounds = imguiFont->CalcTextSizeA(
    fontSize, std::numeric_limits<float>::max(), 0.0f, text.c_str());
  const ImVec2 topLeft{
    position.x - bounds.x * 0.5f,
    static_cast<float>(s_Data.Extent.height) - position.y - bounds.y * 0.5f};
  ImGui::GetBackgroundDrawList()->AddText(
    imguiFont, fontSize, topLeft,
    ImGui::ColorConvertFloat4ToU32(ImVec4(color.r, color.g, color.b, color.a)),
    text.c_str());
  return true;
}

void VulkanRenderer::Clear(const glm::vec4& color)
{
  if (!s_Data.FrameStarted) return;
  s_Data.ClearColor = color;
  BeginMainRenderPass();
}

#else

bool VulkanRenderer::Init(uint32_t, uint32_t) { return false; }
void VulkanRenderer::Shutdown() {}
bool VulkanRenderer::Resize(uint32_t, uint32_t) { return false; }
bool VulkanRenderer::BeginFrame() { return false; }
bool VulkanRenderer::EndFrame(bool) { return false; }
bool VulkanRenderer::InitSceneRenderer() { return false; }
void VulkanRenderer::ShutdownSceneRenderer() {}
bool VulkanRenderer::UploadModel(const std::shared_ptr<Model>&) { return false; }
bool VulkanRenderer::UploadSkybox(const std::shared_ptr<Texture>&) { return false; }
void VulkanRenderer::ResetSceneResources() {}
void VulkanRenderer::SetModelPreviews(const std::vector<RenderModelPreview>&) {}
void VulkanRenderer::DrawScene(DeltaTime&, const std::function<void()>&, bool, bool,
                               const RenderEffectSettings&) {}
bool VulkanRenderer::DrawParticles(const std::vector<ParticleRenderInstance>&) { return false; }
bool VulkanRenderer::BeginDebugLines() { return false; }
bool VulkanRenderer::DrawDebugLine(const glm::vec3&, const glm::vec3&, const glm::vec4&) { return false; }
bool VulkanRenderer::EndDebugLines() { return false; }
bool VulkanRenderer::DrawPhysicsDebug() { return false; }
uint64_t VulkanRenderer::CreateFontAtlas(const uint8_t*, uint32_t, uint32_t) { return 0; }
void VulkanRenderer::DestroyFontAtlas(uint64_t) {}
bool VulkanRenderer::InitImGui() { return false; }
void VulkanRenderer::ShutdownImGui() {}
void VulkanRenderer::BeginImGuiFrame() {}
void VulkanRenderer::RenderImGuiDrawData() {}
void VulkanRenderer::RenderImGuiPlatformWindows() {}
uint64_t VulkanRenderer::GetEditorTextureID() { return 0; }
bool VulkanRenderer::BeginScreenUI() { return false; }
bool VulkanRenderer::DrawScreenQuad(const glm::mat4&, const glm::vec4&) { return false; }
bool VulkanRenderer::DrawScreenText(const Font*, const std::string&, const glm::vec2&, float,
                                    const glm::vec4&) { return false; }
void VulkanRenderer::Clear(const glm::vec4&) {}

#endif
