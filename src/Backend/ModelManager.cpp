#include "ModelManager.h"
#include "Buffer.h"
#include "Camera.h"
#include <gabdebug.h>
#include "glad/glad.h"
#include "meshoptimizer.h"
#include <filesystem>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "RenderBackend.h"
#include "RenderSystem.h"
#include "Timer.hpp"
#include <array>
#include <cstring>
#include <cmath>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <unordered_set>

static glm::mat4 AssimpMatToGLMMat(const aiMatrix4x4& from)
{
  glm::mat4 to;
  //the a,b,c,d in assimp is the row ; the 1,2,3,4 is the column
  to[0][0] = from.a1; to[1][0] = from.a2; to[2][0] = from.a3; to[3][0] = from.a4;
  to[0][1] = from.b1; to[1][1] = from.b2; to[2][1] = from.b3; to[3][1] = from.b4;
  to[0][2] = from.c1; to[1][2] = from.c2; to[2][2] = from.c3; to[3][2] = from.c4;
  to[0][3] = from.d1; to[1][3] = from.d2; to[2][3] = from.d3; to[3][3] = from.d4;
  return to;
}

static glm::vec3 AssimpVecToGLMVec(const aiVector3D& vec) 
{ 
  return {vec.x, vec.y, vec.z};
}

static glm::quat AssimpQuatToGLMQuat(const aiQuaternion& pOrientation)
{
  return {pOrientation.w, pOrientation.x, pOrientation.y, pOrientation.z};
}

static float WrapAnimationTime(float time, float duration)
{
  if (!std::isfinite(time) || !std::isfinite(duration) || duration <= std::numeric_limits<float>::epsilon())
    return 0.0f;

  const float wrapped = std::fmod(time, duration);
  return wrapped < 0.0f ? wrapped + duration : wrapped;
}

static glm::mat4 BlendLocalTransforms(const glm::mat4& from, const glm::mat4& to, float factor)
{
  glm::vec3 fromScale(1.0f);
  glm::quat fromRotation(1.0f, 0.0f, 0.0f, 0.0f);
  glm::vec3 fromTranslation(0.0f);
  glm::vec3 fromSkew(0.0f);
  glm::vec4 fromPerspective(0.0f);

  glm::vec3 toScale(1.0f);
  glm::quat toRotation(1.0f, 0.0f, 0.0f, 0.0f);
  glm::vec3 toTranslation(0.0f);
  glm::vec3 toSkew(0.0f);

  if (glm::vec4 toPerspective(0.0f); !glm::decompose(from, fromScale, fromRotation, fromTranslation, fromSkew, fromPerspective) ||
                                     !glm::decompose(to, toScale, toRotation, toTranslation, toSkew, toPerspective))
  {
    const float t = glm::clamp(factor, 0.0f, 1.0f);
    return from * (1.0f - t) + to * t;
  }

  fromRotation = glm::normalize(fromRotation);
  toRotation = glm::normalize(toRotation);
  if (glm::dot(fromRotation, toRotation) < 0.0f)
    toRotation = -toRotation;

  const float t = glm::clamp(factor, 0.0f, 1.0f);
  const glm::vec3 scale = glm::mix(fromScale, toScale, t);
  const glm::quat rotation = glm::normalize(glm::slerp(fromRotation, toRotation, t));
  const glm::vec3 translation = glm::mix(fromTranslation, toTranslation, t);

  return glm::translate(glm::mat4(1.0f), translation) *
         glm::toMat4(rotation) *
         glm::scale(glm::mat4(1.0f), scale);
}

struct MeshTextureRange
{
  uint32_t StartIndex; // Offset in the textureHandles array
  uint32_t Count;      // How many textures this mesh has
};

struct ModelsData
{
  std::unordered_map<std::string, std::shared_ptr<Model>> m_Models;
  std::vector<std::string> m_ModelsNames;

  std::shared_ptr<StorageBuffer> m_ModelsTransforms;
  std::shared_ptr<StorageBuffer> m_MeshToTransformSSBO;
  std::shared_ptr<StorageBuffer> m_BindlessTextureSSBO;
  std::shared_ptr<StorageBuffer> m_NormalMapFlagsSSBO;
  std::shared_ptr<StorageBuffer> m_SpecularMapFlagsSSBO;
  std::shared_ptr<StorageBuffer> m_MeshToTextureRangeSSBO;
  std::shared_ptr<StorageBuffer> m_FinalBoneMatricesSSBO;
  std::shared_ptr<StorageBuffer> m_ModelIsAnimatedSSBO; 
  std::shared_ptr<StorageBuffer> m_InstanceTransformsSSBO;
  std::shared_ptr<StorageBuffer> m_VisibleInstanceTransformsSSBO;

  GLuint m_BoneMatrixRingBuffer = 0;
  glm::mat4* m_BoneMatrixMapped = nullptr;
  std::vector<glm::mat4> m_BoneMatricesCPU;
  size_t m_BoneMatrixSegmentStride = 0;
  uint32_t m_BoneMatrixRingIndex = 0;
  std::array<GLsync, 3> m_BoneMatrixFences{};

  GLuint sharedVBO, sharedEBO, sharedVAO;
  GLuint fallbackTexture = 0;
  GLuint64 fallbackTextureHandle = 0;

  std::vector<Vertex> allVertices;
  std::vector<uint32_t> allIndices;

} s_Data; 

static size_t AlignBoneBufferOffset(size_t value, size_t alignment)
{
  return (value + alignment - 1) / alignment * alignment;
}

static void WaitForBoneFence(GLsync& fence)
{
  if (!fence) return;
  GLenum result = glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 0);
  while (result == GL_TIMEOUT_EXPIRED)
    result = glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 1'000'000);
  glDeleteSync(fence);
  fence = nullptr;
}

static void DestroyBoneMatrixRing()
{
  for (GLsync& fence : s_Data.m_BoneMatrixFences)
    WaitForBoneFence(fence);
  if (s_Data.m_BoneMatrixMapped && s_Data.m_BoneMatrixRingBuffer)
    glUnmapNamedBuffer(s_Data.m_BoneMatrixRingBuffer);
  if (s_Data.m_BoneMatrixRingBuffer)
    glDeleteBuffers(1, &s_Data.m_BoneMatrixRingBuffer);
  s_Data.m_BoneMatrixRingBuffer = 0;
  s_Data.m_BoneMatrixMapped = nullptr;
  s_Data.m_BoneMatricesCPU.clear();
  s_Data.m_BoneMatrixSegmentStride = 0;
  s_Data.m_BoneMatrixRingIndex = 0;
}

static void CreateBoneMatrixRing(std::vector<glm::mat4> matrices)
{
  DestroyBoneMatrixRing();
  if (matrices.empty()) matrices.emplace_back(1.0f);
  s_Data.m_BoneMatricesCPU = std::move(matrices);
  GLint alignment = 256;
  glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &alignment);
  const size_t dataBytes = s_Data.m_BoneMatricesCPU.size() * sizeof(glm::mat4);
  s_Data.m_BoneMatrixSegmentStride = AlignBoneBufferOffset(
    dataBytes, static_cast<size_t>(std::max(alignment, 1)));
  const size_t totalBytes = s_Data.m_BoneMatrixSegmentStride *
    s_Data.m_BoneMatrixFences.size();
  constexpr GLbitfield flags =
    GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
  glCreateBuffers(1, &s_Data.m_BoneMatrixRingBuffer);
  glNamedBufferStorage(s_Data.m_BoneMatrixRingBuffer,
    static_cast<GLsizeiptr>(totalBytes), nullptr, flags);
  s_Data.m_BoneMatrixMapped = static_cast<glm::mat4*>(glMapNamedBufferRange(
    s_Data.m_BoneMatrixRingBuffer, 0, static_cast<GLsizeiptr>(totalBytes), flags));
  if (!s_Data.m_BoneMatrixMapped)
    throw std::runtime_error("Failed to persistently map bone-matrix ring");
  for (size_t frame = 0; frame < s_Data.m_BoneMatrixFences.size(); ++frame)
    std::memcpy(reinterpret_cast<uint8_t*>(s_Data.m_BoneMatrixMapped) +
      frame * s_Data.m_BoneMatrixSegmentStride,
      s_Data.m_BoneMatricesCPU.data(), dataBytes);
  glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
  glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 9, s_Data.m_BoneMatrixRingBuffer,
    0, static_cast<GLsizeiptr>(dataBytes));
}

static void UploadBoneMatrixFrame()
{
  if (!s_Data.m_BoneMatrixRingBuffer || !s_Data.m_BoneMatrixMapped ||
      s_Data.m_BoneMatricesCPU.empty())
    return;
  s_Data.m_BoneMatrixRingIndex = (s_Data.m_BoneMatrixRingIndex + 1u) %
    static_cast<uint32_t>(s_Data.m_BoneMatrixFences.size());
  WaitForBoneFence(s_Data.m_BoneMatrixFences[s_Data.m_BoneMatrixRingIndex]);
  const size_t dataBytes = s_Data.m_BoneMatricesCPU.size() * sizeof(glm::mat4);
  const size_t offset = s_Data.m_BoneMatrixRingIndex * s_Data.m_BoneMatrixSegmentStride;
  std::memcpy(reinterpret_cast<uint8_t*>(s_Data.m_BoneMatrixMapped) + offset,
    s_Data.m_BoneMatricesCPU.data(), dataBytes);
  glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
  glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 9, s_Data.m_BoneMatrixRingBuffer,
    static_cast<GLintptr>(offset), static_cast<GLsizeiptr>(dataBytes));
}

static void RefreshInstanceTransforms()
{
  std::vector<glm::mat4> transforms;

  for (const auto& modelName : s_Data.m_ModelsNames)
  {
    const auto& model = s_Data.m_Models.at(modelName);
    model->m_InstanceBase = static_cast<uint32_t>(transforms.size());
    transforms.insert(transforms.end(), model->m_InstanceTransforms.begin(), model->m_InstanceTransforms.end());
    RenderSystem::UpdateModelInstances(model);
  }

  if (s_Data.m_InstanceTransformsSSBO)
    s_Data.m_InstanceTransformsSSBO->SetData(transforms.size() * sizeof(glm::mat4), transforms.data());
}

void ModelManager::Init()
{
  if (RenderBackend::Capabilities().NativeModelResources) return;

  if (s_Data.fallbackTexture == 0)
  {
    constexpr std::array<uint8_t, 4> whitePixel = {255, 255, 255, 255};
    glCreateTextures(GL_TEXTURE_2D, 1, &s_Data.fallbackTexture);
    glTextureStorage2D(s_Data.fallbackTexture, 1, GL_RGBA8, 1, 1);
    glTextureSubImage2D(s_Data.fallbackTexture, 0, 0, 0, 1, 1,
                        GL_RGBA, GL_UNSIGNED_BYTE, whitePixel.data());
    glTextureParameteri(s_Data.fallbackTexture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(s_Data.fallbackTexture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(s_Data.fallbackTexture, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTextureParameteri(s_Data.fallbackTexture, GL_TEXTURE_WRAP_T, GL_REPEAT);
    s_Data.fallbackTextureHandle = glGetTextureHandleARB(s_Data.fallbackTexture);
    glMakeTextureHandleResidentARB(s_Data.fallbackTextureHandle);
  }

  if (s_Data.sharedVBO == 0)
    glCreateBuffers(1, &s_Data.sharedVBO);
  if (s_Data.sharedEBO == 0)
    glCreateBuffers(1, &s_Data.sharedEBO);
  if (s_Data.sharedVAO == 0)
    glCreateVertexArrays(1, &s_Data.sharedVAO);

  glVertexArrayVertexBuffer(s_Data.sharedVAO, 0, s_Data.sharedVBO, 0, sizeof(Vertex));
  glVertexArrayElementBuffer(s_Data.sharedVAO, s_Data.sharedEBO);

  struct Attribute
  {
    GLint size;
    GLenum type;
    GLboolean normalized;
    size_t offset;
  };

  std::array<Attribute, 7> attributes =
  {{
    {3, GL_FLOAT, GL_FALSE, offsetof(Vertex, Position)},
    {3, GL_FLOAT, GL_FALSE, offsetof(Vertex, Normal)},
    {2, GL_FLOAT, GL_FALSE, offsetof(Vertex, TexCoords)},
    {3, GL_FLOAT, GL_FALSE, offsetof(Vertex, Tangent)},
    {3, GL_FLOAT, GL_FALSE, offsetof(Vertex, Bitangent)},
    {4, GL_INT,   GL_FALSE, offsetof(Vertex, m_BoneIDs)},
    {4, GL_FLOAT, GL_FALSE, offsetof(Vertex, m_Weights)}
  }};

  for (GLuint i = 0; i < attributes.size(); ++i)
  {
    glEnableVertexArrayAttrib(s_Data.sharedVAO, i);
    if (attributes[i].type == GL_INT)
      glVertexArrayAttribIFormat(s_Data.sharedVAO, i, attributes[i].size, attributes[i].type, attributes[i].offset);
    else
      glVertexArrayAttribFormat(s_Data.sharedVAO, i, attributes[i].size, attributes[i].type, attributes[i].normalized, attributes[i].offset);
    glVertexArrayAttribBinding(s_Data.sharedVAO, i, 0);
  }
}

void ModelManager::BakeModel(const std::string& path, const std::shared_ptr<Model>& model)
{
  Timer timer;

  if (RenderBackend::Capabilities().NativeModelResources)
  {
    const std::string name = std::filesystem::path(path).stem().string();
    model->m_Name = name;
    model->m_IsRendered = model->GetPhysXMeshType() != MeshType::CONVEXMESH;

    for (auto& mesh : model->GetMeshes())
    {
      if (model->GetPhysXMeshType() == MeshType::TRIANGLEMESH)
        model->CreatePhysXStaticMesh(mesh.m_Vertices, mesh.m_Indices);
      else if (model->GetPhysXMeshType() == MeshType::CONVEXMESH)
        model->CreatePhysXDynamicMesh(mesh.m_Vertices);
    }

    s_Data.m_Models[name] = model;
    s_Data.m_ModelsNames.emplace_back(name);
    RenderBackend::Get().UploadModel(model);

    for (auto& mesh : model->GetMeshes())
      for (auto& texture : mesh.m_Textures)
        if (texture) texture->ClearRawData();

    gablog_log(LOG_WARN, __FILE__, __LINE__, "Model: %s %s upload took %.3f ms",
      name.c_str(), RenderBackend::Get().GetName(), timer.ElapsedMillis());
    return;
  }

  constexpr int NUM_BUFFERS = 2;
  std::array<std::unique_ptr<PixelBuffer>, NUM_BUFFERS> pboBuffers;
  int currentPBO = 0;

  // Check if model has exactly one texture total shared by all meshes
  bool singleTextureModel = false;

  if (!model->GetMeshes().empty())
  {
    if (auto& firstMesh = model->GetMeshes()[0]; firstMesh.m_Textures.size() == 1)
    {
      auto sharedTexture = firstMesh.m_Textures[0];
      singleTextureModel = true;

      for (size_t i = 1; i < model->GetMeshes().size(); ++i)
      {
        auto& mesh = model->GetMeshes()[i];
        if (mesh.m_Textures.size() != 1 || mesh.m_Textures[0] != sharedTexture)
        {
          singleTextureModel = false;
          break;
        }
      }
    }
  }

  if (singleTextureModel)
  {
    GLuint64 sharedTextureHandle = 0;
    // Bake the texture only for the first mesh
    auto& firstMesh = model->GetMeshes()[0];
    firstMesh.m_TexturesBindlessHandles.clear();

    if (auto& texture = firstMesh.m_Textures[0])
    {
      int width = 0, height = 0;
      GLenum format;
      const void* srcData;
      GLsizei dataSize = 0;

      width = texture->GetWidth();
      height = texture->GetHeight();
      format = GL_RGBA;
      dataSize = width * height * 4;
      srcData = texture->GetRawData();

      if (srcData && width > 0 && height > 0)
      {
        if (!pboBuffers[currentPBO] || pboBuffers[currentPBO]->GetSize() != dataSize)
          pboBuffers[currentPBO] = std::make_unique<PixelBuffer>(dataSize);

        auto& pbo = pboBuffers[currentPBO];
        pbo->WaitForCompletion();

        if (void* ptr = pbo->Map())
        {
          memcpy(ptr, srcData, dataSize);
          pbo->Unmap(); // Also inserts a sync
        }
        else
        {
          gablog_log(LOG_ERROR, __FILE__, __LINE__, "Failed to map PixelBuffer for texture upload.");
          // fallback - don't assign texture
        }

        GLuint id;
        glCreateTextures(GL_TEXTURE_2D, 1, &id);
        texture->SetRendererID(id);

        const GLsizei mipLevels = 1 + static_cast<GLsizei>(
          std::floor(std::log2(static_cast<double>(std::max(width, height)))));
        glTextureStorage2D(id, mipLevels, GL_RGBA8, width, height);
        pbo->Bind();
        glTextureSubImage2D(id, 0, 0, 0, width, height, format, GL_UNSIGNED_BYTE, nullptr);
        pbo->Unbind();

        glGenerateTextureMipmap(id);
        glTextureParameteri(id, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTextureParameteri(id, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTextureParameteri(id, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTextureParameteri(id, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        sharedTextureHandle = glGetTextureHandleARB(id);
        glMakeTextureHandleResidentARB(sharedTextureHandle);

        // The shader material layout is always diffuse/normal/specular.
        // Missing channels still need valid handles even when their flags are off.
        firstMesh.m_TexturesBindlessHandles.assign(3, sharedTextureHandle);

        currentPBO = (currentPBO + 1) % NUM_BUFFERS;
      }
    }

    // Now assign the same handle to all other meshes
    for (size_t i = 1; i < model->GetMeshes().size(); ++i)
    {
      auto& mesh = model->GetMeshes()[i];
      mesh.m_TexturesBindlessHandles.clear();
      if (sharedTextureHandle != 0)
        mesh.m_TexturesBindlessHandles.assign(3, sharedTextureHandle);
    }
  }
  else
  {
    // Normal per-mesh texture baking
    for (auto& mesh : model->GetMeshes())
    {
      mesh.m_TexturesBindlessHandles.clear();
      std::unordered_map<const Texture*, GLuint64> uploadedHandles;

      for (auto& texture : mesh.m_Textures)
      {
        if (!texture)
            continue;

        int width, height;
        GLenum format;
        const void* srcData;
        GLsizei dataSize;

        width = texture->GetWidth();
        height = texture->GetHeight();
        format = GL_RGBA;
        dataSize = width * height * 4;
        srcData = texture->GetRawData();

        if (!srcData || width <= 0 || height <= 0)
            continue;

        if (!pboBuffers[currentPBO] || pboBuffers[currentPBO]->GetSize() != dataSize)
            pboBuffers[currentPBO] = std::make_unique<PixelBuffer>(dataSize);

        auto& pbo = pboBuffers[currentPBO];

        pbo->WaitForCompletion();

        if (void* ptr = pbo->Map())
        {
            memcpy(ptr, srcData, dataSize);
            pbo->Unmap(); // Also inserts a sync
        }
        else
        {
            gablog_log(LOG_ERROR, __FILE__, __LINE__, "Failed to map PixelBuffer for texture upload.");
            continue;
        }

        GLuint id;
        glCreateTextures(GL_TEXTURE_2D, 1, &id);
        texture->SetRendererID(id);

        const GLsizei mipLevels = 1 + static_cast<GLsizei>(
          std::floor(std::log2(static_cast<double>(std::max(width, height)))));
        glTextureStorage2D(id, mipLevels, GL_RGBA8, width, height);
        pbo->Bind();
        glTextureSubImage2D(id, 0, 0, 0, width, height, format, GL_UNSIGNED_BYTE, nullptr);
        pbo->Unbind();

        glGenerateTextureMipmap(id);
        glTextureParameteri(id, GL_TEXTURE_WRAP_S, GL_REPEAT );
        glTextureParameteri(id, GL_TEXTURE_WRAP_T, GL_REPEAT );
        glTextureParameteri(id, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTextureParameteri(id, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        GLuint64 handle = glGetTextureHandleARB(id);
        glMakeTextureHandleResidentARB(handle);

        uploadedHandles[texture.get()] = handle;

        currentPBO = (currentPBO + 1) % NUM_BUFFERS;
      }

      if (!uploadedHandles.empty())
      {
        const GLuint64 fallback = s_Data.fallbackTextureHandle;
        const auto materialHandle = [&](const std::shared_ptr<Texture>& texture)
        {
          if (texture)
            if (const auto found = uploadedHandles.find(texture.get()); found != uploadedHandles.end())
              return found->second;
          return fallback;
        };
        mesh.m_TexturesBindlessHandles = {
          materialHandle(mesh.m_DiffuseTexture),
          materialHandle(mesh.m_NormalTexture),
          materialHandle(mesh.m_SpecularTexture)
        };
      }
    }
  }

  // Every shader-visible material occupies exactly three consecutive entries:
  // diffuse, normal, specular. Never let a missing map consume zero entries or
  // borrow a handle from the next mesh.
  for (auto& mesh : model->GetMeshes())
  {
    if (mesh.m_TexturesBindlessHandles.size() != 3)
      mesh.m_TexturesBindlessHandles.assign(3, s_Data.fallbackTextureHandle);
  }

  std::string name = std::filesystem::path(path).stem().string();
  model->m_IsRendered = model->GetPhysXMeshType() != MeshType::CONVEXMESH;

  for (auto& mesh : model->GetMeshes())
  {
    s_Data.allVertices.insert(s_Data.allVertices.end(), mesh.m_Vertices.begin(), mesh.m_Vertices.end());
    s_Data.allIndices.insert(s_Data.allIndices.end(), mesh.m_Indices.begin(), mesh.m_Indices.end());

    RenderSystem::RegisterModelDrawCommand(name, static_cast<uint32_t>(mesh.m_Vertices.size()),
      static_cast<uint32_t>(mesh.m_Indices.size()));

    for(auto& tex : mesh.m_Textures) tex->ClearRawData();

    if(model->GetPhysXMeshType() == MeshType::TRIANGLEMESH) model->CreatePhysXStaticMesh(mesh.m_Vertices, mesh.m_Indices);
    else if(model->GetPhysXMeshType() == MeshType::CONVEXMESH) model->CreatePhysXDynamicMesh(mesh.m_Vertices);
  }

  model->m_Name = name;
  s_Data.m_Models[name] = model;
  s_Data.m_ModelsNames.emplace_back(name);

  gablog_log(LOG_WARN, __FILE__, __LINE__, "Model: %s baking took %.3f ms", name.c_str(), timer.ElapsedMillis());
}

void ModelManager::SetInitialControllerTransform(const std::string& name, const Transform& transform, float radius, float height, bool slopeLimit)
{
  auto it = s_Data.m_Models.find(name);
  if (it == s_Data.m_Models.end())
  {
      gablog_log(LOG_WARN, __FILE__, __LINE__, "Model '%s' not found in ModelManager!", name.c_str());
      return;
  }

  std::shared_ptr<Model> model = it->second;

  if(model->GetPhysXMeshType() != MeshType::CONTROLLER)
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "This function is for Controller model!");
    return;
  }

  model->m_ControllerTransform = transform;
  model->m_ControllerRadius = radius;
  model->m_ControllerHeight = height;
  model->m_ControllerSlopeLimit = slopeLimit;

  model->CreateCharacterController(PhysX::GlmVec3ToPxVec3(model->m_ControllerTransform.GetPosition()), radius, height, slopeLimit);

  // Find index in the names vector
  auto vecIt = std::ranges::find(s_Data.m_ModelsNames, name);
  if (vecIt == s_Data.m_ModelsNames.end())
  {
      gablog_log(LOG_WARN, __FILE__, __LINE__, "Model '%s' not found in name list for SSBO!", name.c_str());
      return;
  }

  int ssboIndex = static_cast<int>(std::distance(s_Data.m_ModelsNames.begin(), vecIt));
  const glm::mat4 controllerTransform = model->m_ControllerTransform.GetTransform();
  if (s_Data.m_ModelsTransforms)
    s_Data.m_ModelsTransforms->SetSubData(ssboIndex * sizeof(glm::mat4), sizeof(glm::mat4), glm::value_ptr(controllerTransform));
  SetModelInstanceTransform(name, 0, controllerTransform);
}

void ModelManager::SetControllerTransform(const std::string& name, const Transform& transform)
{
  auto it = s_Data.m_Models.find(name);
  if (it == s_Data.m_Models.end() || it->second->GetPhysXMeshType() != MeshType::CONTROLLER)
  {
    gablog_log(LOG_WARN, __FILE__, __LINE__, "Controller model '%s' not found in ModelManager!", name.c_str());
    return;
  }

  const auto& model = it->second;
  model->m_ControllerTransform = transform;
  if (model->m_ActorController)
  {
    const glm::vec3 position = transform.GetPosition();
    model->m_ActorController->setFootPosition(PxExtendedVec3(position.x, position.y, position.z));
  }

  const glm::mat4 matrix = transform.GetTransform();
  if (const auto nameIt = std::ranges::find(s_Data.m_ModelsNames, name); s_Data.m_ModelsTransforms && nameIt != s_Data.m_ModelsNames.end())
  {
    const auto modelIndex = static_cast<size_t>(std::distance(s_Data.m_ModelsNames.begin(), nameIt));
    s_Data.m_ModelsTransforms->SetSubData(modelIndex * sizeof(glm::mat4), sizeof(glm::mat4), glm::value_ptr(matrix));
  }
  SetModelInstanceTransform(name, 0, matrix);
}

bool ModelManager::SetControllerSize(const std::string& name, float radius, float height)
{
  if (!std::isfinite(radius) || !std::isfinite(height) || radius <= 0.0f || height <= 0.0f)
    return false;

  const auto it = s_Data.m_Models.find(name);
  if (it == s_Data.m_Models.end() || it->second->GetPhysXMeshType() != MeshType::CONTROLLER)
  {
    gablog_log(LOG_WARN, __FILE__, __LINE__, "Controller model '%s' not found in ModelManager!", name.c_str());
    return false;
  }

  const auto& model = it->second;
  PxController* controller = model->m_ActorController;
  if (!controller || controller->getType() != PxControllerShapeType::eCAPSULE)
    return false;

  auto* capsule = static_cast<PxCapsuleController*>(controller);
  const PxExtendedVec3 footPosition = controller->getFootPosition();
  if (!capsule->setRadius(radius) || !capsule->setHeight(height))
    return false;

  controller->setStepOffset(std::min(0.3f, height + radius * 2.0f));
  controller->setFootPosition(footPosition);
  model->m_ControllerRadius = radius;
  model->m_ControllerHeight = height;
  return true;
}

void ModelManager::SetInitialModelTransform(const std::string& name, const glm::mat4& transform)
{
  auto it = s_Data.m_Models.find(name);
  if (it == s_Data.m_Models.end())
  {
      gablog_log(LOG_WARN, __FILE__, __LINE__, "Model '%s' not found in ModelManager!", name.c_str());
      return;
  }

  const auto& model = it->second;

  if(model->GetPhysXMeshType() == MeshType::CONTROLLER)
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "This function is for non Controller model!");
    return;
  }

  glm::mat4 resolvedTransform = transform;
  std::string convexName = name + "_convex";

  if (auto convexIt = s_Data.m_Models.find(convexName); convexIt != s_Data.m_Models.end())
  {
    gablog_log(LOG_WARN, __FILE__, __LINE__, "Convex version '%s' found for model '%s'. Applying same transform.",
      convexName.c_str(), name.c_str());

    const auto& convex = convexIt->second;

    auto pxTransform = PxTransform(PhysX::GlmMat4ToPxTransform(transform));

    if(convex->GetPhysXMeshType() == MeshType::TRIANGLEMESH) convex->m_StaticMeshActor->setGlobalPose(pxTransform);
    else if(convex->GetPhysXMeshType() == MeshType::CONVEXMESH)
    {
      if (convex->m_isKinematic) convex->m_DynamicMeshActor->setKinematicTarget(pxTransform);
      else convex->m_DynamicMeshActor->setGlobalPose(pxTransform);
    }

    glm::mat4 convexTransform = PhysX::PxMat44ToGlmMat4(convexIt->second->GetDynamicActor()->getGlobalPose());
    resolvedTransform = convexTransform;
    SetModelInstanceTransform(convexName, 0, convexTransform);

    pxTransform = PxTransform(PhysX::GlmMat4ToPxTransform(convexTransform));

    if(model->GetPhysXMeshType() == MeshType::TRIANGLEMESH) model->m_StaticMeshActor->setGlobalPose(pxTransform);
    else if(model->GetPhysXMeshType() == MeshType::CONVEXMESH)
    {
      if (model->m_isKinematic) model->m_DynamicMeshActor->setKinematicTarget(pxTransform);
      else model->m_DynamicMeshActor->setGlobalPose(pxTransform);
    }

    if (auto vec2It = std::ranges::find(s_Data.m_ModelsNames, name); vec2It != s_Data.m_ModelsNames.end())
    {
      int ssboIndex = static_cast<int>(std::distance(s_Data.m_ModelsNames.begin(), vec2It));
      if (s_Data.m_ModelsTransforms)
        s_Data.m_ModelsTransforms->SetSubData(ssboIndex * sizeof(glm::mat4), sizeof(glm::mat4), glm::value_ptr(convexTransform));
    }
    else
    {
      gablog_log(LOG_WARN, __FILE__, __LINE__, "Main model '%s' not found in name list for SSBO!", name.c_str());
    }
  }
  else
  {
    // No convex model, fallback to normal update
    auto vecIt = std::ranges::find(s_Data.m_ModelsNames, name);
    if (vecIt == s_Data.m_ModelsNames.end())
    {
      gablog_log(LOG_WARN, __FILE__, __LINE__, "Model '%s' not found in name list for SSBO!", name.c_str());
      return;
    }

    int ssboIndex = static_cast<int>(std::distance(s_Data.m_ModelsNames.begin(), vecIt));
    if (s_Data.m_ModelsTransforms)
      s_Data.m_ModelsTransforms->SetSubData(ssboIndex * sizeof(glm::mat4), sizeof(glm::mat4), glm::value_ptr(transform));
  }

  SetModelInstanceTransform(name, 0, resolvedTransform);
}

uint32_t ModelManager::AddModelInstance(const std::string& name, const glm::mat4& transform)
{
  auto it = s_Data.m_Models.find(name);
  if (it == s_Data.m_Models.end())
  {
    gablog_log(LOG_WARN, __FILE__, __LINE__, "Model '%s' not found in ModelManager!", name.c_str());
    return std::numeric_limits<uint32_t>::max();
  }

  auto& instances = it->second->m_InstanceTransforms;
  const auto instanceIndex = static_cast<uint32_t>(instances.size());
  instances.push_back(transform);
  RefreshInstanceTransforms();
  return instanceIndex;
}

bool ModelManager::RemoveModelInstance(const std::string& name, uint32_t instanceIndex)
{
  auto it = s_Data.m_Models.find(name);
  if (it == s_Data.m_Models.end() || instanceIndex >= it->second->m_InstanceTransforms.size())
    return false;

  auto& model = it->second;
  model->m_InstanceTransforms.erase(model->m_InstanceTransforms.begin() + instanceIndex);
  if (model->m_InstanceTransforms.empty())
  {
    model->m_IsRendered = false;

    if (model->m_ActorController)
    {
      model->m_ActorController->release();
      model->m_ActorController = nullptr;
    }

    auto removePhysicsActors = [](const std::shared_ptr<Model>& physicsModel)
    {
      if (physicsModel->m_StaticMeshActor)
      {
        physicsModel->m_StaticMeshActor->release();
        physicsModel->m_StaticMeshActor = nullptr;
      }
      if (physicsModel->m_DynamicMeshActor)
      {
        physicsModel->m_DynamicMeshActor->release();
        physicsModel->m_DynamicMeshActor = nullptr;
      }
    };

    removePhysicsActors(model);
    if (const auto convex = s_Data.m_Models.find(name + "_convex"); convex != s_Data.m_Models.end())
    {
      removePhysicsActors(convex->second);
      convex->second->m_InstanceTransforms.clear();
    }
  }

  RefreshInstanceTransforms();
  return true;
}

void ModelManager::SetModelInstances(const std::string& name, const std::vector<Transform>& instances)
{
  auto it = s_Data.m_Models.find(name);
  if (it == s_Data.m_Models.end())
  {
    gablog_log(LOG_WARN, __FILE__, __LINE__, "Model '%s' not found in ModelManager!", name.c_str());
    return;
  }

  auto& modelInstances = it->second->m_InstanceTransforms;
  modelInstances.clear();
  modelInstances.reserve(instances.size());
  for (const auto& instance : instances)
    modelInstances.push_back(instance.GetTransform());

  RefreshInstanceTransforms();
}

void ModelManager::SetModelInstanceTransform(const std::string& name, uint32_t instanceIndex, const glm::mat4& transform)
{
  const auto it = s_Data.m_Models.find(name);
  if (it == s_Data.m_Models.end())
  {
    gablog_log(LOG_WARN, __FILE__, __LINE__, "Model '%s' not found in ModelManager!", name.c_str());
    return;
  }

  auto& model = it->second;
  auto& instances = model->m_InstanceTransforms;
  const bool countChanged = instanceIndex >= instances.size();
  if (countChanged)
    instances.resize(static_cast<size_t>(instanceIndex) + 1, glm::mat4(1.0f));
  instances[instanceIndex] = transform;

  if (countChanged)
  {
    RefreshInstanceTransforms();
  }
  else if (s_Data.m_InstanceTransformsSSBO)
  {
    const size_t offset = (static_cast<size_t>(model->m_InstanceBase) + instanceIndex) * sizeof(glm::mat4);
    s_Data.m_InstanceTransformsSSBO->SetSubData(offset, sizeof(glm::mat4), glm::value_ptr(transform));
  }
}

static void ReleaseModelResources()
{
  std::unordered_set<GLuint64> residentHandles;

  if (s_Data.fallbackTextureHandle != 0)
    residentHandles.insert(s_Data.fallbackTextureHandle);

  for (const auto &model: s_Data.m_Models | std::views::values)
  {
    if (model->m_ActorController)
    {
      model->m_ActorController->release();
      model->m_ActorController = nullptr;
    }
    if (model->m_StaticMeshActor)
    {
      model->m_StaticMeshActor->release();
      model->m_StaticMeshActor = nullptr;
    }
    if (model->m_DynamicMeshActor)
    {
      model->m_DynamicMeshActor->release();
      model->m_DynamicMeshActor = nullptr;
    }

    for (const auto& mesh : model->m_Meshes)
      residentHandles.insert(mesh.m_TexturesBindlessHandles.begin(), mesh.m_TexturesBindlessHandles.end());
  }

  for (const GLuint64 handle : residentHandles)
  {
    if (RenderBackend::Capabilities().OpenGLContext && handle != 0)
      glMakeTextureHandleNonResidentARB(handle);
  }

  if (RenderBackend::Capabilities().NativeModelResources)
    RenderBackend::Get().ResetSceneResources();

  s_Data.m_Models.clear();
  s_Data.m_ModelsNames.clear();
  s_Data.allVertices.clear();
  s_Data.allIndices.clear();

  s_Data.m_ModelsTransforms.reset();
  s_Data.m_MeshToTransformSSBO.reset();
  s_Data.m_BindlessTextureSSBO.reset();
  s_Data.m_NormalMapFlagsSSBO.reset();
  s_Data.m_SpecularMapFlagsSSBO.reset();
  s_Data.m_MeshToTextureRangeSSBO.reset();
  s_Data.m_FinalBoneMatricesSSBO.reset();
  s_Data.m_ModelIsAnimatedSSBO.reset();
  s_Data.m_InstanceTransformsSSBO.reset();
  s_Data.m_VisibleInstanceTransformsSSBO.reset();

  if (!RenderBackend::Capabilities().NativeModelResources)
  {
    DestroyBoneMatrixRing();
    if (s_Data.sharedVBO) glDeleteBuffers(1, &s_Data.sharedVBO);
    if (s_Data.sharedEBO) glDeleteBuffers(1, &s_Data.sharedEBO);
    if (s_Data.sharedVAO) glDeleteVertexArrays(1, &s_Data.sharedVAO);
    if (s_Data.fallbackTexture) glDeleteTextures(1, &s_Data.fallbackTexture);
  }
  s_Data.sharedVBO = 0;
  s_Data.sharedEBO = 0;
  s_Data.sharedVAO = 0;
  s_Data.fallbackTexture = 0;
  s_Data.fallbackTextureHandle = 0;
}

void ModelManager::Reset()
{
  ReleaseModelResources();
  Init();
}

void ModelManager::Shutdown()
{
  ReleaseModelResources();
}

void ModelManager::UpdateTransforms(const DeltaTime& dt)
{
  for (const auto& [key, model] : s_Data.m_Models)
  {
    if(model->IsAnimated() && model->m_IsRendered)
    { 
      model->UpdateAnimation(dt);
      auto& transforms = model->GetFinalBoneMatrices();

      auto nameIt = std::ranges::find(s_Data.m_ModelsNames, key);
      if (nameIt != s_Data.m_ModelsNames.end())
      {
        int ssboIndex = static_cast<int>(std::distance(s_Data.m_ModelsNames.begin(), nameIt));
        const size_t offset = static_cast<size_t>(ssboIndex) * MAX_BONES * sizeof(glm::mat4);
        const size_t matrixCount = std::min(transforms.size(), static_cast<size_t>(MAX_BONES));
        const size_t size = matrixCount * sizeof(glm::mat4);

        if (size > 0 && offset + size <=
            s_Data.m_BoneMatricesCPU.size() * sizeof(glm::mat4))
          std::memcpy(reinterpret_cast<uint8_t*>(s_Data.m_BoneMatricesCPU.data()) + offset,
            transforms.data(), size);
      }
      else
      {
        gablog_log(LOG_WARN, __FILE__, __LINE__, "Animated model '%s' not found in name list for bone SSBO!", key.c_str());
      }
    }

    const std::string& convexName = key;
    constexpr const char* suffix = "_convex";
    if (convexName.size() < 7 || convexName.compare(convexName.size() - 7, 7, suffix) != 0) continue;

    std::string baseName = convexName.substr(0, convexName.size() - 7);

    auto baseModelIt = s_Data.m_Models.find(baseName);
    if (baseModelIt == s_Data.m_Models.end())
    {
      gablog_log(LOG_WARN, __FILE__, __LINE__, "Base model '%s' not found for convex '%s'",
        baseName.c_str(), convexName.c_str());
      continue;
    }

    if(baseModelIt->second->m_IsRendered)
    {
      glm::mat4 convexTransform = PhysX::PxMat44ToGlmMat4(model->GetDynamicActor()->getGlobalPose());
      SetModelInstanceTransform(convexName, 0, convexTransform);

      if (auto nameIt = std::ranges::find(s_Data.m_ModelsNames, baseName); nameIt != s_Data.m_ModelsNames.end())
      {
        int ssboIndex = static_cast<int>(std::distance(s_Data.m_ModelsNames.begin(), nameIt));
        if (s_Data.m_ModelsTransforms)
          s_Data.m_ModelsTransforms->SetSubData(ssboIndex * sizeof(glm::mat4), sizeof(glm::mat4), glm::value_ptr(convexTransform));
        SetModelInstanceTransform(baseName, 0, convexTransform);
      }
      else
      {
        gablog_log(LOG_WARN, __FILE__, __LINE__, "Base model '%s' not found in name list for SSBO!", baseName.c_str());
      }
    }
  }
  UploadBoneMatrixFrame();
}

void ModelManager::EndGPUFrame()
{
  if (!s_Data.m_BoneMatrixRingBuffer) return;
  GLsync& fence = s_Data.m_BoneMatrixFences[s_Data.m_BoneMatrixRingIndex];
  if (fence) glDeleteSync(fence);
  fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}

void ModelManager::SetRender(const std::string& name, bool render)
{
  auto it = s_Data.m_Models.find(name);
  if (it == s_Data.m_Models.end())
  {
    gablog_log(LOG_WARN, __FILE__, __LINE__, "Model '%s' not found in ModelManager!", name.c_str());
    return;
  }

  it->second->m_IsRendered = render;

  RenderSystem::SetModelRendered(it->second, render);
}

GLsizei ModelManager::GetModelsQuantity()
{
  return s_Data.m_Models.size();
}

GLuint ModelManager::GetModelsVAO()
{
  return s_Data.sharedVAO;
}

GLuint ModelManager::GetAllInstanceTransformsBuffer()
{
  return s_Data.m_InstanceTransformsSSBO
    ? s_Data.m_InstanceTransformsSSBO->GetID() : 0;
}

void ModelManager::UploadVisibleInstanceTransforms(const std::vector<glm::mat4>& transforms)
{
  if (!s_Data.m_VisibleInstanceTransformsSSBO)
    return;

  if (!transforms.empty())
    s_Data.m_VisibleInstanceTransformsSSBO->SetData(transforms.size() * sizeof(glm::mat4),
      transforms.data());
  s_Data.m_VisibleInstanceTransformsSSBO->Bind();
}

void ModelManager::BindAllInstanceTransforms()
{
  if (s_Data.m_InstanceTransformsSSBO)
    s_Data.m_InstanceTransformsSSBO->Bind();
}

void ModelManager::BindVisibleInstanceTransforms()
{
  if (s_Data.m_VisibleInstanceTransformsSSBO)
    s_Data.m_VisibleInstanceTransformsSSBO->Bind();
}

void ModelManager::UploadToGPU()
{
  if (RenderBackend::Capabilities().NativeModelResources) return;

  glNamedBufferStorage(s_Data.sharedVBO, s_Data.allVertices.size() * sizeof(Vertex), s_Data.allVertices.data(), 0);
  glNamedBufferStorage(s_Data.sharedEBO, s_Data.allIndices.size() * sizeof(uint32_t), s_Data.allIndices.data(), 0);

  s_Data.m_ModelsTransforms = StorageBuffer::Create(sizeof(glm::mat4) * s_Data.m_Models.size(), 5);

  auto transform = GetTransforms();
  s_Data.m_ModelsTransforms->SetData(transform.size() * sizeof(glm::mat4), transform.data());

  s_Data.m_InstanceTransformsSSBO = StorageBuffer::Create(sizeof(glm::mat4), 13);
  s_Data.m_VisibleInstanceTransformsSSBO = StorageBuffer::Create(sizeof(glm::mat4), 13);
  RefreshInstanceTransforms();
  s_Data.m_InstanceTransformsSSBO->Bind();

  std::vector<int> meshToTransformIndex;

  int currentMeshIndex = 0;
  for (int modelIndex = 0; modelIndex < s_Data.m_ModelsNames.size(); ++modelIndex)
  {
      const std::string& modelName = s_Data.m_ModelsNames[modelIndex];
      std::shared_ptr<Model> model = s_Data.m_Models[modelName];

      int meshCount = model->GetMeshes().size(); 

      for (int i = 0; i < meshCount; ++i)
      {
          meshToTransformIndex.push_back(modelIndex); 
          currentMeshIndex++;
      }
  }

  s_Data.m_MeshToTransformSSBO = StorageBuffer::Create(meshToTransformIndex.size() * sizeof(int), 6);
  s_Data.m_MeshToTransformSSBO->SetData(meshToTransformIndex.size() * sizeof(int), meshToTransformIndex.data());

  std::vector<GLuint64> textureHandles;
  std::vector<MeshTextureRange> meshTextureRanges;
  std::vector<int32_t> normalMapFlags;
  std::vector<int32_t> specularMapFlags;

  for (const auto& modelName : s_Data.m_ModelsNames)
  {
    auto& model = s_Data.m_Models[modelName];

    for (const auto& meshes = model->GetMeshes(); const auto & mesh : meshes)
    {
      mesh.hasNormalMap ? normalMapFlags.push_back(1) : normalMapFlags.push_back(0);
      mesh.hasSpecularMap ? specularMapFlags.push_back(1) : specularMapFlags.push_back(0);  

      MeshTextureRange range{};
      range.StartIndex = static_cast<uint32_t>(textureHandles.size());
      range.Count = static_cast<uint32_t>(mesh.m_TexturesBindlessHandles.size());

      for (GLuint64 handle : mesh.m_TexturesBindlessHandles) 
      {
       textureHandles.push_back(handle);
      }

      meshTextureRanges.push_back(range);
    }
  }

  s_Data.m_BindlessTextureSSBO = StorageBuffer::Create(textureHandles.size() * sizeof(GLuint64), 7);
  s_Data.m_BindlessTextureSSBO->SetData(textureHandles.size() * sizeof(GLuint64), textureHandles.data());

  s_Data.m_MeshToTextureRangeSSBO = StorageBuffer::Create(meshTextureRanges.size() * sizeof(MeshTextureRange), 8);
  s_Data.m_MeshToTextureRangeSSBO->SetData(meshTextureRanges.size() * sizeof(MeshTextureRange), meshTextureRanges.data());
  
  std::vector identityBones(s_Data.m_Models.size() * MAX_BONES, glm::mat4(1.0f));

  CreateBoneMatrixRing(std::move(identityBones));

  std::vector<int> isAnimatedFlags;

  for (const auto& modelName : s_Data.m_ModelsNames)
  {
      const auto& model = s_Data.m_Models[modelName];
      isAnimatedFlags.push_back(model->IsAnimated() ? 1 : 0);
  }

  s_Data.m_ModelIsAnimatedSSBO = StorageBuffer::Create(isAnimatedFlags.size() * sizeof(int), 10); 
  s_Data.m_ModelIsAnimatedSSBO->SetData(isAnimatedFlags.size() * sizeof(int), isAnimatedFlags.data());

  s_Data.m_NormalMapFlagsSSBO = StorageBuffer::Create(normalMapFlags.size() * sizeof(int), 11); 
  s_Data.m_NormalMapFlagsSSBO->SetData(normalMapFlags.size() * sizeof(int), normalMapFlags.data());

  s_Data.m_SpecularMapFlagsSSBO = StorageBuffer::Create(specularMapFlags.size() * sizeof(int), 12); 
  s_Data.m_SpecularMapFlagsSSBO->SetData(specularMapFlags.size() * sizeof(int), specularMapFlags.data());

  for (const auto& modelName : s_Data.m_Models)
  {
    for(auto& bruh : modelName.second->m_Meshes)
    {
      for(auto& hehe : bruh.m_Textures)
      {
        gablog_log(LOG_WARN, __FILE__, __LINE__, "TYPE OF TEXTURE: %s", hehe->GetType().c_str());
      }
    }
  }

  s_Data.allVertices.clear();
  s_Data.allIndices.clear();
}

void ModelManager::MoveController(const std::string& name, const Movement& movement, float speed, const DeltaTime& dt)
{
  auto it = s_Data.m_Models.find(name);
  if (it == s_Data.m_Models.end())
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "Model '%s' doesnt exist!", name.c_str());
    return;
  }

  const auto& model = it->second;

  if (model->GetPhysXMeshType() != MeshType::CONTROLLER || !model->GetController())
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "Model '%s' is not a valid PhysX controller!", name.c_str());
    return;
  }

  glm::vec3 camForward = Camera::GetForwardDirection();
  camForward.y = 0.0f;
  if (glm::length2(camForward) > std::numeric_limits<float>::epsilon())
    camForward = glm::normalize(camForward);
  else
    camForward = glm::vec3(0.0f, 0.0f, -1.0f);

  glm::vec3 camRight = Camera::GetRightDirection();
  camRight.y = 0.0f;
  if (glm::length2(camRight) > std::numeric_limits<float>::epsilon())
    camRight = glm::normalize(camRight);
  else
    camRight = glm::vec3(1.0f, 0.0f, 0.0f);

  switch (movement)
  {
    case Movement::FORWARD:  model->m_ControllerMoveDirection += camForward; break;
    case Movement::BACKWARD: model->m_ControllerMoveDirection -= camForward; break;
    case Movement::LEFT:     model->m_ControllerMoveDirection -= camRight;   break;
    case Movement::RIGHT:    model->m_ControllerMoveDirection += camRight;   break;
    default: break;
  }

  model->m_ControllerMoveSpeed = std::max(model->m_ControllerMoveSpeed, speed);
  (void)dt;
}

void ModelManager::UpdateControllers(const DeltaTime& dt)
{
  constexpr float gravity = -9.81f;
  constexpr float terminalVelocity = -55.0f;
  constexpr float horizontalDamping = 10.0f;
  constexpr float rotationResponse = 10.0f;

  const float deltaTime = std::clamp(dt.GetSeconds(), 0.0f, 0.1f);
  if (deltaTime <= 0.0f)
    return;

  for (size_t modelIndex = 0; modelIndex < s_Data.m_ModelsNames.size(); ++modelIndex)
  {
    const std::string& name = s_Data.m_ModelsNames[modelIndex];
    auto modelIt = s_Data.m_Models.find(name);
    if (modelIt == s_Data.m_Models.end())
      continue;

    const auto& model = modelIt->second;
    PxController* controller = model->GetController();
    if (model->GetPhysXMeshType() != MeshType::CONTROLLER || !controller)
      continue;

    model->m_ControllerVelocity.y = std::max(
      model->m_ControllerVelocity.y + gravity * deltaTime,
      terminalVelocity);

    glm::vec3 moveDirection = model->m_ControllerMoveDirection;
    if (glm::length2(moveDirection) > std::numeric_limits<float>::epsilon())
    {
      moveDirection = glm::normalize(moveDirection);

      model->m_ControllerVelocity.x = moveDirection.x * model->m_ControllerMoveSpeed;
      model->m_ControllerVelocity.z = moveDirection.z * model->m_ControllerMoveSpeed;

      const float targetYaw = std::atan2(moveDirection.x, moveDirection.z);
      float yawDelta = targetYaw - model->m_ControllerCurrentYaw;
      yawDelta = std::atan2(std::sin(yawDelta), std::cos(yawDelta));

      model->m_ControllerCurrentYaw += yawDelta * glm::clamp(deltaTime * rotationResponse, 0.0f, 1.0f);
      model->m_ControllerTransform.SetRotation(
        { 0.0f, glm::degrees(model->m_ControllerCurrentYaw), 0.0f });
    }
    else
    {
      const float damping = std::exp(-horizontalDamping * deltaTime);
      model->m_ControllerVelocity.x *= damping;
      model->m_ControllerVelocity.z *= damping;
    }

    const PxVec3 displacement = model->m_ControllerVelocity * deltaTime;
    const PxControllerCollisionFlags flags = controller->move(
      displacement, 0.001f, deltaTime, PxControllerFilters());

    if (flags.isSet(PxControllerCollisionFlag::eCOLLISION_DOWN) && model->m_ControllerVelocity.y < 0.0f)
      model->m_ControllerVelocity.y = 0.0f;
    if (flags.isSet(PxControllerCollisionFlag::eCOLLISION_UP) && model->m_ControllerVelocity.y > 0.0f)
      model->m_ControllerVelocity.y = 0.0f;
    model->m_ControllerIsGrounded = flags.isSet(PxControllerCollisionFlag::eCOLLISION_DOWN);

    const PxExtendedVec3 footPosition = controller->getFootPosition();
    model->m_ControllerTransform.SetPosition(glm::vec3(
      static_cast<float>(footPosition.x),
      static_cast<float>(footPosition.y),
      static_cast<float>(footPosition.z)));

    const glm::mat4 controllerTransform = model->m_ControllerTransform.GetTransform();
    if (s_Data.m_ModelsTransforms)
      s_Data.m_ModelsTransforms->SetSubData(
        modelIndex * sizeof(glm::mat4), sizeof(glm::mat4), glm::value_ptr(controllerTransform));
    SetModelInstanceTransform(name, 0, controllerTransform);

    model->m_ControllerMoveDirection = glm::vec3(0.0f);
    model->m_ControllerMoveSpeed = 0.0f;
  }
}

std::shared_ptr<Model> ModelManager::GetModel(const std::string& name)
{
  auto it = s_Data.m_Models.find(name);
  if (it != s_Data.m_Models.end())
      return it->second;
  return nullptr;
}

const std::vector<std::string>& ModelManager::GetModelNames()
{
  return s_Data.m_ModelsNames;
}

std::vector<glm::mat4> ModelManager::GetTransforms()
{
  std::vector<glm::mat4> transforms;
  transforms.reserve(s_Data.m_Models.size());

  for (const auto& [key, model] : s_Data.m_Models)
  {
    if (model->GetPhysXMeshType() == MeshType::TRIANGLEMESH)
    {
        glm::mat4 mat = PhysX::PxMat44ToGlmMat4(model->GetStaticActor()->getGlobalPose());
        transforms.push_back(mat);
    }
    else if (model->GetPhysXMeshType() == MeshType::CONVEXMESH)
    {
        glm::mat4 mat = PhysX::PxMat44ToGlmMat4(model->GetDynamicActor()->getGlobalPose());
        transforms.push_back(mat);
    }
    else if (model->GetPhysXMeshType() == MeshType::CONTROLLER)
    {
        glm::mat4 mat = model->GetControllerTransform().GetTransform();
        transforms.push_back(mat);
    }
  }

  return transforms;
}

Model::Model(const char* path, float optimizerStrength, bool isAnimated, bool isKinematic, const MeshType& type) :  m_isKinematic(isKinematic), m_OptimizerStrength(optimizerStrength), m_isAnimated(isAnimated), m_meshType(type)
{
  Timer timer;

  Assimp::Importer importer;
  if(m_isAnimated) m_Scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices | aiProcess_CalcTangentSpace);
  else m_Scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices | aiProcess_CalcTangentSpace);

  if (!m_Scene || m_Scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !m_Scene->mRootNode)
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "[MODEL]: %s", importer.GetErrorString());
    return;
  }
  std::string dirStr = std::filesystem::path(path).parent_path().string();
  m_Directory = dirStr;

  const glm::mat4 rootTransform = AssimpMatToGLMMat(m_Scene->mRootNode->mTransformation);
  const float rootDeterminant = glm::determinant(rootTransform);
  if (std::abs(rootDeterminant) > std::numeric_limits<float>::epsilon())
    m_GlobalInverseTransform = glm::inverse(rootTransform);

  processNode(m_Scene->mRootNode, m_Scene);

  glm::vec3 boundsMin(std::numeric_limits<float>::max());
  glm::vec3 boundsMax(std::numeric_limits<float>::lowest());
  bool hasVertices = false;
  for (const auto& mesh : m_Meshes)
  {
    for (const auto& vertex : mesh.m_Vertices)
    {
      boundsMin = glm::min(boundsMin, vertex.Position);
      boundsMax = glm::max(boundsMax, vertex.Position);
      hasVertices = true;
    }
  }
  if (hasVertices)
  {
    m_BoundsCenter = (boundsMin + boundsMax) * 0.5f;
    for (const auto& mesh : m_Meshes)
      for (const auto& vertex : mesh.m_Vertices)
        m_BoundsRadius = std::max(m_BoundsRadius, glm::distance(m_BoundsCenter, vertex.Position));

    // Bind-pose vertices do not contain the full animation envelope. Keep the
    // sphere conservative so animated limbs are not clipped at a frustum edge.
    if (m_isAnimated)
      m_BoundsRadius *= 1.5f;
  }

  if(isAnimated)
  {
    for (unsigned int i = 0; i < m_Scene->mNumAnimations; ++i)
    {
      aiAnimation* animation = m_Scene->mAnimations[i];

      AnimationData animData;
      animData.name = animation->mName.C_Str();
      animData.duration = static_cast<float>(animation->mDuration);
      animData.ticksPerSecond = animation->mTicksPerSecond > 0.0
        ? static_cast<float>(animation->mTicksPerSecond)
        : 25.0f;

      if (!std::isfinite(animData.duration) ||
          animData.duration <= std::numeric_limits<float>::epsilon())
      {
        double lastKeyTime = 0.0;
        for (unsigned int channelIndex = 0; channelIndex < animation->mNumChannels; ++channelIndex)
        {
          const aiNodeAnim* channel = animation->mChannels[channelIndex];
          if (channel->mNumPositionKeys > 0)
            lastKeyTime = std::max(lastKeyTime, channel->mPositionKeys[channel->mNumPositionKeys - 1].mTime);
          if (channel->mNumRotationKeys > 0)
            lastKeyTime = std::max(lastKeyTime, channel->mRotationKeys[channel->mNumRotationKeys - 1].mTime);
          if (channel->mNumScalingKeys > 0)
            lastKeyTime = std::max(lastKeyTime, channel->mScalingKeys[channel->mNumScalingKeys - 1].mTime);
        }
        animData.duration = std::max(static_cast<float>(lastKeyTime), 1.0f);
      }

      ReadHierarchyData(animData.hierarchy, m_Scene->mRootNode);
      ReadMissingBones(animation);

      animData.bones = m_Bones;

      gablog_log(LOG_INFO, __FILE__, __LINE__, "Model: %s, Animation at index: %u, %s",
        std::filesystem::path(path).stem().string().c_str(), i, animData.name.c_str());

      m_ProcessedAnimations.emplace_back(animData);
    }

    if (m_ProcessedAnimations.empty())
    {
      gablog_log(LOG_ASSERT, __FILE__, __LINE__, "[MODEL]: Model doesnt contain animations");
      gabdebug_break();
    }

    SetAnimationbyIndex(0);

    ResizeFinalBoneMatrices();
  }

  m_TexturesLoaded.clear();

  gablog_log(LOG_WARN, __FILE__, __LINE__, "Model loading took %.3f ms", timer.ElapsedMillis());
}

void Model::processNode(aiNode* node, const aiScene* scene)
{
  for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
      aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
      m_Meshes.emplace_back(processMesh(mesh, scene));
  }
  for (unsigned int i = 0; i < node->mNumChildren; ++i) {
      processNode(node->mChildren[i], scene);
  }
}

Mesh Model::processMesh(aiMesh* mesh, const aiScene* scene)
{
  std::vector<Vertex> vertices;
  vertices.reserve(mesh->mNumVertices);

  for (unsigned int i = 0; i < mesh->mNumVertices; ++i)
  {
    Vertex vertex{};
    if(m_isAnimated) SetDefaultBoneData(vertex);
    vertex.Position = {mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z};
    vertex.Normal = mesh->HasNormals() ? glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z) : glm::vec3(0.0f);
    if (mesh->mTextureCoords[0]) {
        vertex.TexCoords = {mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y};
        vertex.Tangent = {mesh->mTangents[i].x, mesh->mTangents[i].y, mesh->mTangents[i].z};
        vertex.Bitangent = {mesh->mBitangents[i].x, mesh->mBitangents[i].y, mesh->mBitangents[i].z};
    } else {
        vertex.TexCoords = glm::vec2(0.0f);
    }
    vertices.emplace_back(vertex);
  }

  std::vector<GLuint> indices;
  for (unsigned int i = 0; i < mesh->mNumFaces; ++i)
  {
    const aiFace& face = mesh->mFaces[i];
    indices.insert(indices.end(), face.mIndices, face.mIndices + face.mNumIndices);
  }

  std::vector<std::shared_ptr<Texture>> textures;

  if (mesh->mMaterialIndex >= 0)
  {
    aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

    loadMaterialTextures(material, aiTextureType_DIFFUSE, "texture_diffuse", textures);
    loadMaterialTextures(material, aiTextureType_NORMALS, "texture_normal", textures);
    loadMaterialTextures(material, aiTextureType_HEIGHT, "texture_normal", textures); // OBJ fallback
    loadMaterialTextures(material, aiTextureType_SPECULAR, "texture_specular", textures);

    // loadMaterialTextures(material, aiTextureType_BASE_COLOR, "texture_albedo", textures);
  }

  if (m_isAnimated) ExtractBoneWeightForVertices(vertices, mesh);
  OptimizeMesh(vertices, indices);

  Mesh result(vertices, indices, textures);
  for (const auto& texture : textures)
  {
    if (!texture) continue;
    const std::string& textureType = texture->GetType();
    if (!result.m_NormalTexture && textureType.find("normal") != std::string::npos)
      result.m_NormalTexture = texture;
    else if (!result.m_SpecularTexture && textureType.find("specular") != std::string::npos)
      result.m_SpecularTexture = texture;
    else if (!result.m_DiffuseTexture &&
             (textureType.find("diffuse") != std::string::npos ||
              textureType.find("albedo") != std::string::npos ||
              textureType.find("base_color") != std::string::npos))
      result.m_DiffuseTexture = texture;
  }
  if (!result.m_DiffuseTexture && !textures.empty())
    result.m_DiffuseTexture = textures.front();
  result.hasNormalMap = result.m_NormalTexture && result.m_NormalTexture->IsLoaded();
  result.hasSpecularMap = result.m_SpecularTexture && result.m_SpecularTexture->IsLoaded();

  return result;
}

bool Model::loadMaterialTextures(aiMaterial* mat, aiTextureType type, const std::string& typeName, std::vector<std::shared_ptr<Texture>>& textures)
{
  bool loadedAny = false;

  for (unsigned int i = 0; i < mat->GetTextureCount(type); ++i) {
      aiString str;
      mat->GetTexture(type, i, &str);
      std::string texturePath = str.C_Str();
      // A single image can legitimately be referenced by multiple material
      // channels. Texture::m_Type describes the channel, so sharing one
      // Texture object between diffuse/normal/specular would make the last
      // channel win and bind the wrong DX12 descriptor.
      const std::string cacheKey = typeName + '\n' + texturePath;

      if (m_TexturesLoaded.contains(cacheKey)) {
          const auto& cachedTexture = m_TexturesLoaded[cacheKey];
          if (cachedTexture && cachedTexture->IsLoaded()) {
              textures.emplace_back(cachedTexture);
              loadedAny = true;
          }
          continue;
      }

      std::shared_ptr<Texture> texture;

      if (texturePath[0] == '*') {
        if (const aiTexture* aitexture = m_Scene->GetEmbeddedTexture(str.C_Str())) {
              texture = Texture::CreateEMBEDDED(aitexture, texturePath);
          }
      } else {
          texture = Texture::Create(texturePath, m_Directory);
      }

      if (texture && texture->IsLoaded()) {
          texture->SetType(typeName);
          textures.emplace_back(texture);
          m_TexturesLoaded[cacheKey] = texture;
          loadedAny = true;
      }
  }

  return loadedAny;
}

void Model::OptimizeMesh(std::vector<Vertex>& m_Vertices, std::vector<GLuint>& m_Indices)
{
  std::vector<GLuint> remap(m_Indices.size());

  size_t OptVertexCount = meshopt_generateVertexRemap(remap.data(),m_Indices.data(),m_Indices.size(),m_Vertices.data(),m_Vertices.size(),sizeof(Vertex));

  std::vector<Vertex> OptVertices;
  std::vector<GLuint> OptIndices;
  OptVertices.resize(OptVertexCount);
  OptIndices.resize(m_Indices.size());

  meshopt_remapIndexBuffer(OptIndices.data(),m_Indices.data(),m_Indices.size(),remap.data());
  meshopt_remapVertexBuffer(OptVertices.data(),m_Vertices.data(),m_Vertices.size(),sizeof(Vertex),remap.data());
  meshopt_optimizeVertexCache(OptIndices.data(), OptIndices.data(), m_Indices.size(), OptVertexCount);
  meshopt_optimizeOverdraw(OptIndices.data(), OptIndices.data(), m_Indices.size(), &OptVertices[0].Position.x, OptVertexCount, sizeof(Vertex), 1.05f);  // Overdraw threshold (1.0 = minimal overdraw)
  meshopt_optimizeVertexFetch(OptVertices.data(), OptIndices.data(), m_Indices.size(), OptVertices.data(), OptVertexCount, sizeof(Vertex));

  std::vector<GLuint> SimplifiedIndices(OptIndices.size());
  size_t OptIndexCount = meshopt_simplify(SimplifiedIndices.data(),OptIndices.data(),m_Indices.size(),&OptVertices[0].Position.x,OptVertexCount,sizeof(Vertex),(size_t)(m_Indices.size() * m_OptimizerStrength),0.2f);
  SimplifiedIndices.resize(OptIndexCount);

  m_Indices = std::move(SimplifiedIndices);
  m_Vertices = std::move(OptVertices);
}

void Model::CreatePhysXStaticMesh(std::vector<Vertex>& m_Vertices, std::vector<GLuint>& m_Indices)
{
  std::vector<PxVec3> physxVertices(m_Vertices.size());
  for (size_t i = 0; i < m_Vertices.size(); ++i) {
      physxVertices[i] = PxVec3(
          m_Vertices[i].Position.x,
          m_Vertices[i].Position.y,
          m_Vertices[i].Position.z
      );
  }

  PxTriangleMesh* physxMesh = PhysX::CreateTriangleMesh(
      static_cast<PxU32>(physxVertices.size()), physxVertices.data(),
      static_cast<PxU32>(m_Indices.size() / 3), m_Indices.data()
  );

  if (!physxMesh) {
      gablog_log(LOG_ERROR, __FILE__, __LINE__, "Failed to create PhysX triangle mesh");
      return;
  }

  PxPhysics* physics = PhysX::getPhysics();
  PxScene* scene = PhysX::getScene();
  PxMaterial* material = PhysX::getMaterial();

  PxTriangleMeshGeometry triGeom;
  triGeom.triangleMesh = physxMesh;

  const bool actorCreated = m_StaticMeshActor == nullptr;
  if (actorCreated)
  {
    const PxTransform pose(PxVec3(0));
    m_StaticMeshActor = physics->createRigidStatic(pose);
  }

  PxShape* meshShape = m_StaticMeshActor
    ? PxRigidActorExt::createExclusiveShape(*m_StaticMeshActor, triGeom, *material)
    : nullptr;
  physxMesh->release();

  if (!meshShape)
  {
    if (actorCreated && m_StaticMeshActor)
    {
      m_StaticMeshActor->release();
      m_StaticMeshActor = nullptr;
    }
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "Failed to create PhysX triangle mesh shape");
    return;
  }

  if (actorCreated)
    scene->addActor(*m_StaticMeshActor);
}

void Model::CreatePhysXDynamicMesh(std::vector<Vertex>& m_Vertices)
{
  std::vector<PxVec3> physxVertices(m_Vertices.size());
  for (size_t i = 0; i < m_Vertices.size(); ++i)
  {
      physxVertices[i] = PxVec3(m_Vertices[i].Position.x, m_Vertices[i].Position.y,m_Vertices[i].Position.z);
  }

  PxConvexMesh* convexMesh = PhysX::CreateConvexMesh(static_cast<PxU32>(physxVertices.size()), physxVertices.data());

  if (!convexMesh) {
      gablog_log(LOG_ERROR, __FILE__, __LINE__, "Failed to create PhysX convex mesh");
      return;
  }

  PxPhysics* physics = PhysX::getPhysics();
  PxScene* scene = PhysX::getScene();
  PxMaterial* material = PhysX::getMaterial();

  PxConvexMeshGeometry convexGeom;
  convexGeom.convexMesh = convexMesh;
  convexGeom.scale = PxMeshScale(PxVec3(1.0f)); // Optional scaling

  const bool actorCreated = m_DynamicMeshActor == nullptr;
  if (actorCreated)
  {
    const PxTransform pose(PxVec3(0));
    m_DynamicMeshActor = physics->createRigidDynamic(pose);
  }

  PxShape* shape = m_DynamicMeshActor
    ? PxRigidActorExt::createExclusiveShape(*m_DynamicMeshActor, convexGeom, *material)
    : nullptr;
  convexMesh->release();

  if (!shape)
  {
    if (actorCreated && m_DynamicMeshActor)
    {
      m_DynamicMeshActor->release();
      m_DynamicMeshActor = nullptr;
    }
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "Failed to create PhysX convex mesh shape");
    return;
  }

  material->setRestitution(0.0f);
  if (m_isKinematic)
    m_DynamicMeshActor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
  PxRigidBodyExt::setMassAndUpdateInertia(*m_DynamicMeshActor, 1.0f);

  if (actorCreated)
    scene->addActor(*m_DynamicMeshActor);
}

void Model::CreateCharacterController(const PxVec3& footPosition, float radius, float height, bool slopeLimit)
{
  m_ActorController = PhysX::CreateCharacterController(footPosition, radius, height, slopeLimit);
}

void Model::ExtractBoneWeightForVertices(std::vector<Vertex>& vertices, aiMesh* mesh)
{
  for (unsigned int i = 0; i < mesh->mNumBones; ++i) {
      std::string boneName = mesh->mBones[i]->mName.C_Str();
      int boneID = -1;

      if (m_BoneInfoMap.find(boneName) == m_BoneInfoMap.end())
      {
          if (m_BoneCounter >= MAX_BONES)
          {
              gablog_log(LOG_ERROR, __FILE__, __LINE__, "Model '%s' exceeds the supported limit of %d deforming bones; ignoring '%s'",
                m_Name.c_str(), MAX_BONES, boneName.c_str());
              continue;
          }

          boneID = m_BoneCounter++;
          m_BoneInfoMap[boneName] = { boneID, AssimpMatToGLMMat(mesh->mBones[i]->mOffsetMatrix) };
      }
      else
      {
          boneID = m_BoneInfoMap[boneName].id;
      }

      for (unsigned int j = 0; j < mesh->mBones[i]->mNumWeights; ++j) {
          int vertexID = mesh->mBones[i]->mWeights[j].mVertexId;
          float weight = mesh->mBones[i]->mWeights[j].mWeight;

          if (vertexID < vertices.size()) {
              SetBoneData(vertices[vertexID], boneID, weight);
          }
      }
  }

  for (Vertex& vertex : vertices)
    NormalizeBoneWeights(vertex);
}

// Set default bone data for a vertex
void Model::SetDefaultBoneData(Vertex& vertex)
{
  for (int i = 0; i < MAX_BONE_INFLUENCE; ++i) {
      vertex.m_BoneIDs[i] = -1;
      vertex.m_Weights[i] = 0.0f;
  }
}

// Set bone data for a vertex
void Model::SetBoneData(Vertex& vertex, int boneID, float weight)
{
  if (boneID < 0 || boneID >= MAX_BONES || weight <= 0.0f)
    return;

  for (int i = 0; i < MAX_BONE_INFLUENCE; ++i)
  {
      if (vertex.m_BoneIDs[i] < 0)
      {
          vertex.m_BoneIDs[i] = boneID;
          vertex.m_Weights[i] = weight;
          return;
      }
  }

  int weakestIndex = 0;
  for (int i = 1; i < MAX_BONE_INFLUENCE; ++i)
    if (vertex.m_Weights[i] < vertex.m_Weights[weakestIndex])
      weakestIndex = i;

  if (weight > vertex.m_Weights[weakestIndex])
  {
    vertex.m_BoneIDs[weakestIndex] = boneID;
    vertex.m_Weights[weakestIndex] = weight;
  }
}

void Model::NormalizeBoneWeights(Vertex& vertex)
{
  float totalWeight = 0.0f;
  for (int i = 0; i < MAX_BONE_INFLUENCE; ++i)
    if (vertex.m_BoneIDs[i] >= 0)
      totalWeight += vertex.m_Weights[i];

  if (totalWeight <= std::numeric_limits<float>::epsilon())
    return;

  for (int i = 0; i < MAX_BONE_INFLUENCE; ++i)
    if (vertex.m_BoneIDs[i] >= 0)
      vertex.m_Weights[i] /= totalWeight;
}

void Model::UpdateAnimation(const DeltaTime& dt)
{
  if (!m_isAnimated || m_ProcessedAnimations.empty() ||
      m_CurrentAnimationIndex < 0 ||
      m_CurrentAnimationIndex >= static_cast<int>(m_ProcessedAnimations.size()))
    return;

  const float rawDelta = static_cast<float>(dt);
  const float delta = std::isfinite(rawDelta) && rawDelta > 0.0f ? rawDelta : 0.0f;

  if (!m_IsBlending)
  {
      m_CurrentTime = WrapAnimationTime(m_CurrentTime + m_TicksPerSecond * delta, m_Duration);

      CalculateBoneTransform(&m_RootNode, glm::mat4(1.0f));
  }
  else
  {
    m_BlendTime += delta;
    const float linearBlendFactor = glm::clamp(m_BlendTime / m_BlendDuration, 0.0f, 1.0f);
    const float blendFactor = linearBlendFactor * linearBlendFactor * (3.0f - 2.0f * linearBlendFactor);

    if (!m_BlendFromSnapshot)
      m_CurrentTime = WrapAnimationTime(m_CurrentTime + m_TicksPerSecond * delta, m_Duration);
    m_NextTime = WrapAnimationTime(m_NextTime + m_TicksPerSecondNext * delta, m_DurationNext);

    CalculateBlendedBoneTransform(&m_RootNode, m_CurrentTime, m_NextTime, glm::mat4(1.0f), blendFactor);

    if (linearBlendFactor >= 1.0f)
    {
        const int completedAnimationIndex = m_NextAnimationIndex;
        const float completedAnimationTime = m_NextTime;
        SetAnimationbyIndex(completedAnimationIndex);
        m_CurrentTime = completedAnimationTime;
    }
  }
}

bool Model::IsInAnimation(int index) const
{
  if (index < 0 || index >= static_cast<int>(m_ProcessedAnimations.size()))
    return false;

  return (!m_IsBlending && m_CurrentAnimationIndex == index) ||
         (m_IsBlending && m_NextAnimationIndex == index);
}

void Model::StartBlendToAnimation(int32_t nextAnimationIndex, float blendDuration)
{
  if (nextAnimationIndex < 0 ||
      nextAnimationIndex >= static_cast<int32_t>(m_ProcessedAnimations.size()))
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "Invalid animation index %d for model '%s'",
      nextAnimationIndex, m_Name.c_str());
    return;
  }

  if (IsInAnimation(nextAnimationIndex))
    return;

  if (!std::isfinite(blendDuration) ||
      blendDuration <= std::numeric_limits<float>::epsilon())
  {
    SetAnimationbyIndex(nextAnimationIndex);
    return;
  }

  std::unordered_map<std::string, glm::mat4> interruptedPose;
  if (m_IsBlending)
  {
    const float linearFactor = glm::clamp(m_BlendTime / m_BlendDuration, 0.0f, 1.0f);
    const float smoothFactor = linearFactor * linearFactor * (3.0f - 2.0f * linearFactor);
    interruptedPose.reserve(m_BlendSourcePose.size() + m_BoneInfoMap.size());
    CaptureBlendedLocalPose(&m_RootNode, smoothFactor, interruptedPose);
  }

  m_BlendTime = 0.0f;
  m_BlendDuration = std::max(blendDuration, std::numeric_limits<float>::epsilon());
  m_IsBlending = true;
  m_BlendFromSnapshot = !interruptedPose.empty();
  m_BlendSourcePose = std::move(interruptedPose);
  m_NextAnimationIndex = nextAnimationIndex;
  m_NextTime = 0.0f;

  const AnimationData& animData = m_ProcessedAnimations[nextAnimationIndex];
  m_BonesNext           = animData.bones;
  m_TicksPerSecondNext  = animData.ticksPerSecond;
  m_DurationNext        = animData.duration;
}

void Model::SetAnimationbyIndex(int animationIndex)
{
  if (animationIndex < 0 || animationIndex >= static_cast<int>(m_ProcessedAnimations.size()))
  {
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "Invalid animation index %d for model '%s'",
      animationIndex, m_Name.c_str());
    return;
  }

  m_CurrentAnimationIndex = animationIndex;
  m_CurrentTime = 0.0f;
  m_IsBlending = false;
  m_BlendFromSnapshot = false;
  m_BlendSourcePose.clear();
  m_NextAnimationIndex = -1;
  m_NextTime = 0.0f;
  
  const AnimationData& animData = m_ProcessedAnimations[animationIndex];

  m_Duration = std::isfinite(animData.duration) &&
               animData.duration > std::numeric_limits<float>::epsilon()
    ? animData.duration
    : 1.0f;
  m_TicksPerSecond = std::isfinite(animData.ticksPerSecond) && animData.ticksPerSecond > 0.0f
    ? animData.ticksPerSecond
    : 25.0f;

  m_RootNode = animData.hierarchy;
  m_Bones = animData.bones;

  ResizeFinalBoneMatrices();
}

void Model::SetAnimationByName(const std::string& animationName)
{
  auto it = std::ranges::find_if(m_ProcessedAnimations,
                                 [&animationName](const AnimationData& animData) {
                                   return animData.name == animationName;
                                 });

  if (it != m_ProcessedAnimations.end()) {
      const int animationIndex = static_cast<int>(std::distance(m_ProcessedAnimations.begin(), it));
      SetAnimationbyIndex(animationIndex); 
  } else {
      gablog_log(LOG_ERROR, __FILE__, __LINE__, "Animation not found: %s", animationName.c_str());
  }
}

void Model::CalculateBoneTransform(const AssimpNodeData* node, const glm::mat4& parentTransform)
{
  const std::string& nodeName = node->name;
  const glm::mat4 nodeTransform = SampleLocalTransform(*node, m_Bones, m_CurrentTime);

  const glm::mat4 globalTransformation = parentTransform * nodeTransform;

  auto it = m_BoneInfoMap.find(nodeName);
  if (it != m_BoneInfoMap.end())
  {
      const int index = it->second.id;
      if (index >= 0 && index < static_cast<int>(m_FinalBoneMatrices.size()))
        m_FinalBoneMatrices[index] = m_GlobalInverseTransform * globalTransformation * it->second.offset;
  }

  for (const auto & i : node->children)
      CalculateBoneTransform(&i, globalTransformation);
}

glm::mat4 Model::SampleLocalTransform(const AssimpNodeData& node, const std::vector<Bone>& bones,
  float animationTime) const
{
  if (const Bone* bone = FindBoneInList(node.name, bones))
    return bone->GetInterpolatedTransform(animationTime, node.transformation);
  return node.transformation;
}

void Model::CaptureBlendedLocalPose(const AssimpNodeData* node, float blendFactor,
  std::unordered_map<std::string, glm::mat4>& outPose) const
{
  const std::string& nodeName = node->name;
  glm::mat4 transformCurrent;
  if (m_BlendFromSnapshot)
  {
    const auto snapshotIt = m_BlendSourcePose.find(nodeName);
    transformCurrent = snapshotIt != m_BlendSourcePose.end()
      ? snapshotIt->second
      : SampleLocalTransform(*node, m_Bones, m_CurrentTime);
  }
  else
  {
    transformCurrent = SampleLocalTransform(*node, m_Bones, m_CurrentTime);
  }

  const glm::mat4 transformNext = SampleLocalTransform(*node, m_BonesNext, m_NextTime);
  outPose[nodeName] = BlendLocalTransforms(transformCurrent, transformNext, blendFactor);

  for (const AssimpNodeData& child : node->children)
    CaptureBlendedLocalPose(&child, blendFactor, outPose);
}

void Model::CalculateBlendedBoneTransform(const AssimpNodeData* node, float timeCurrent,
  float timeNext, const glm::mat4& parentTransform, float blendFactor)
{
  const std::string& nodeName = node->name;
  glm::mat4 transformCurrent;
  if (m_BlendFromSnapshot)
  {
    const auto snapshotIt = m_BlendSourcePose.find(nodeName);
    transformCurrent = snapshotIt != m_BlendSourcePose.end()
      ? snapshotIt->second
      : SampleLocalTransform(*node, m_Bones, timeCurrent);
  }
  else
  {
    transformCurrent = SampleLocalTransform(*node, m_Bones, timeCurrent);
  }

  const glm::mat4 transformNext = SampleLocalTransform(*node, m_BonesNext, timeNext);
  const glm::mat4 blendedTransform = BlendLocalTransforms(transformCurrent, transformNext, blendFactor);
  const glm::mat4 globalTransform = parentTransform * blendedTransform;

  if (const auto it = m_BoneInfoMap.find(nodeName); it != m_BoneInfoMap.end())
  {
      const int index = it->second.id;
      if (index >= 0 && index < static_cast<int>(m_FinalBoneMatrices.size()))
        m_FinalBoneMatrices[index] = m_GlobalInverseTransform * globalTransform * it->second.offset;
  }

  for (const AssimpNodeData& child : node->children)
    CalculateBlendedBoneTransform(&child, timeCurrent, timeNext, globalTransform, blendFactor);
}

const Bone* Model::FindBoneInList(const std::string& name, const std::vector<Bone>& bones) const
{
  auto iter = std::ranges::find_if(bones,
                                   [&](const Bone& bone) {
                                     return bone.GetBoneName() == name;
                                   });
  return iter != bones.end() ? &(*iter) : nullptr;
}

void Model::ResizeFinalBoneMatrices()
{
  m_FinalBoneMatrices.resize(MAX_BONES, glm::mat4(1.0f));
}

void Model::ReadHierarchyData(AssimpNodeData& dest, const aiNode* src)
{
  assert(src);  

  dest.name = "";
  dest.transformation = glm::mat4(1.0f);  // Reset to identity matrix
  dest.children.clear();  // Clear previous children

  dest.name = src->mName.data;
  dest.transformation = AssimpMatToGLMMat(src->mTransformation);
  dest.childrenCount = static_cast<int>(src->mNumChildren);

  for (unsigned int i = 0; i < src->mNumChildren; i++)
  {
      if (src->mChildren[i] == nullptr) 
      {
          gablog_log(LOG_ERROR, __FILE__, __LINE__, "Null child node found at index %u", i);
          continue;  // Skip if child node is null
      }

      AssimpNodeData newData;
      ReadHierarchyData(newData, src->mChildren[i]);
      dest.children.emplace_back(newData);
  }
}

void Model::ReadMissingBones(const aiAnimation* animation)
{
  assert(animation);  

  m_Bones.clear();
  m_Bones.reserve(animation->mNumChannels);

  for (unsigned int i = 0; i < animation->mNumChannels; i++)
  {
      const aiNodeAnim* channel = animation->mChannels[i];
      const std::string boneName = channel->mNodeName.data;
      const auto boneInfoIt = m_BoneInfoMap.find(boneName);
      const int boneId = boneInfoIt != m_BoneInfoMap.end() ? boneInfoIt->second.id : -1;
      m_Bones.emplace_back(boneName, boneId, channel);
  }
}


std::shared_ptr<Model> Model::CreateSTATIC(const char* path, float optimizerStrength, bool isKinematic, MeshType type)
{
	return std::make_shared<Model>(path,optimizerStrength,false,isKinematic,type);
}

std::shared_ptr<Model> Model::CreateANIMATED(const char* path, float optimizerStrength, bool isKinematic, MeshType type)
{
	return std::make_shared<Model>(path,optimizerStrength,true,isKinematic,type);
}


Bone::Bone(const std::string& name, int ID, const aiNodeAnim* channel) : m_Name(name), m_ID(ID)
{
  // Extract position keyframes
  m_NumPositions = static_cast<int>(channel->mNumPositionKeys);
  m_Positions.reserve(channel->mNumPositionKeys);
  for (int i = 0; i < m_NumPositions; ++i) {
      aiVector3D aiPosition = channel->mPositionKeys[i].mValue;
      float timeStamp = static_cast<float>(channel->mPositionKeys[i].mTime);
      KeyPosition data = { AssimpVecToGLMVec(aiPosition), timeStamp };
      m_Positions.emplace_back(data);
  }

  // Extract rotation keyframes
  m_NumRotations = static_cast<int>(channel->mNumRotationKeys);
  m_Rotations.reserve(channel->mNumRotationKeys);
  for (int i = 0; i < m_NumRotations; ++i) {
      aiQuaternion aiOrientation = channel->mRotationKeys[i].mValue;
      float timeStamp = static_cast<float>(channel->mRotationKeys[i].mTime);
      KeyRotation data = { glm::normalize(AssimpQuatToGLMQuat(aiOrientation)), timeStamp };
      m_Rotations.emplace_back(data);
  }

  // Extract scaling keyframes
  m_NumScalings = static_cast<int>(channel->mNumScalingKeys);
  m_Scales.reserve(channel->mNumScalingKeys);
  for (int i = 0; i < m_NumScalings; ++i) {
      aiVector3D aiScale = channel->mScalingKeys[i].mValue;
      float timeStamp = static_cast<float>(channel->mScalingKeys[i].mTime);
      KeyScale data = { AssimpVecToGLMVec(aiScale), timeStamp };
      m_Scales.emplace_back(data);
  }
}

glm::mat4 Bone::GetInterpolatedTransform(float animationTime, const glm::mat4& fallbackTransform) const
{
  glm::vec3 fallbackScale(1.0f);
  glm::quat fallbackRotation(1.0f, 0.0f, 0.0f, 0.0f);
  glm::vec3 fallbackTranslation(0.0f);
  glm::vec3 skew(0.0f);
  glm::vec4 perspective(0.0f);
  if (!glm::decompose(fallbackTransform, fallbackScale, fallbackRotation,
      fallbackTranslation, skew, perspective))
  {
    fallbackScale = glm::vec3(1.0f);
    fallbackRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    fallbackTranslation = glm::vec3(0.0f);
  }

  const glm::mat4 translation = m_NumPositions > 0
    ? InterpolatePosition(animationTime)
    : glm::translate(glm::mat4(1.0f), fallbackTranslation);
  const glm::mat4 rotation = m_NumRotations > 0
    ? InterpolateRotation(animationTime)
    : glm::toMat4(glm::normalize(fallbackRotation));
  const glm::mat4 scale = m_NumScalings > 0
    ? InterpolateScaling(animationTime)
    : glm::scale(glm::mat4(1.0f), fallbackScale);

  return translation * rotation * scale;
}

// Interpolation helper functions
float Bone::GetScaleFactor(float lastTimeStamp, float nextTimeStamp, float animationTime) const
{
    const float timeRange = nextTimeStamp - lastTimeStamp;
    if (std::abs(timeRange) <= std::numeric_limits<float>::epsilon())
      return 0.0f;

    const float scaleFactor = (animationTime - lastTimeStamp) / timeRange;
    return glm::clamp(scaleFactor, 0.0f, 1.0f);
}

glm::mat4 Bone::InterpolatePosition(float animationTime) const
{
    if (m_NumPositions == 0)
        return glm::mat4(1.0f);

    if (m_NumPositions == 1) {
        return glm::translate(glm::mat4(1.0f), m_Positions[0].position);
    }

    int p0Index = GetPositionIndex(animationTime);
    int p1Index = p0Index + 1;

    float scaleFactor = GetScaleFactor(m_Positions[p0Index].timeStamp, m_Positions[p1Index].timeStamp, animationTime);
    glm::vec3 finalPosition = glm::mix(m_Positions[p0Index].position, m_Positions[p1Index].position, scaleFactor);

    return glm::translate(glm::mat4(1.0f), finalPosition);
}

glm::mat4 Bone::InterpolateRotation(float animationTime) const
{
    if (m_NumRotations == 0)
        return glm::mat4(1.0f);

    if (m_NumRotations == 1) {
        return glm::toMat4(glm::normalize(m_Rotations[0].orientation));
    }

    int p0Index = GetRotationIndex(animationTime);
    int p1Index = p0Index + 1;

    float scaleFactor = GetScaleFactor(m_Rotations[p0Index].timeStamp, m_Rotations[p1Index].timeStamp, animationTime);
    glm::quat finalRotation = glm::slerp(m_Rotations[p0Index].orientation, m_Rotations[p1Index].orientation, scaleFactor);

    return glm::toMat4(glm::normalize(finalRotation));
}

glm::mat4 Bone::InterpolateScaling(float animationTime) const
{
    if (m_NumScalings == 0)
        return glm::mat4(1.0f);

    if (m_NumScalings == 1) {
        return glm::scale(glm::mat4(1.0f), m_Scales[0].scale);
    }

    int p0Index = GetScaleIndex(animationTime);
    int p1Index = p0Index + 1;

    float scaleFactor = GetScaleFactor(m_Scales[p0Index].timeStamp, m_Scales[p1Index].timeStamp, animationTime);
    glm::vec3 finalScale = glm::mix(m_Scales[p0Index].scale, m_Scales[p1Index].scale, scaleFactor);

    return glm::scale(glm::mat4(1.0f), finalScale);
}

int Bone::GetPositionIndex(float animationTime) const
{
    if (m_NumPositions <= 1)
        return 0;

    for (int i = 0; i < m_NumPositions - 1; ++i) {
        if (animationTime < m_Positions[i + 1].timeStamp) {
            return i;
        }
    }
    return m_NumPositions - 2;
}

int Bone::GetRotationIndex(float animationTime) const
{
    if (m_NumRotations <= 1)
        return 0;

    for (int i = 0; i < m_NumRotations - 1; ++i) {
        if (animationTime < m_Rotations[i + 1].timeStamp) {
            return i;
        }
    }
    return m_NumRotations - 2;
}

int Bone::GetScaleIndex(float animationTime) const
{
    if (m_NumScalings <= 1)
        return 0;

    for (int i = 0; i < m_NumScalings - 1; ++i) {
        if (animationTime < m_Scales[i + 1].timeStamp) {
            return i;
        }
    }
    return m_NumScalings - 2;
}



