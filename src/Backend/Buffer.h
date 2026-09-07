#pragma once
#include <cstdint>
#include <gabdebug.h>
#include <string>
#include <vector>
#include <glad/glad.h>
#include "Shader.h"
#include <span>
#include <array>
#include <unordered_map>

enum class ShaderDataType
{
	None = 0, Float, Float2, Float3, Float4, Mat3, Mat4, Int, Int2, Int3, Int4, Bool
};

static uint32_t ShaderDataTypeSize(ShaderDataType type)
{
	switch (type)
	{
		case ShaderDataType::Float:    return 4;
		case ShaderDataType::Float2:   return 4 * 2;
		case ShaderDataType::Float3:   return 4 * 3;
		case ShaderDataType::Float4:   return 4 * 4;
		case ShaderDataType::Mat3:     return 4 * 3 * 3;
		case ShaderDataType::Mat4:     return 4 * 4 * 4;
		case ShaderDataType::Int:      return 4;
		case ShaderDataType::Int2:     return 4 * 2;
		case ShaderDataType::Int3:     return 4 * 3;
		case ShaderDataType::Int4:     return 4 * 4;
		case ShaderDataType::Bool:     return 1;
	}

	gablog_log(LOG_ASSERT, __FILE__, __LINE__, "Unknown ShaderDataType!");
	gabdebug_break();
	return 0;
}

struct BufferElement
{
	std::string Name;
	ShaderDataType Type;
	uint32_t Size;
	size_t Offset;
	bool Normalized;

	BufferElement() = default;

	BufferElement(ShaderDataType type, const std::string& name, bool normalized = false)
		: Name(name), Type(type), Size(ShaderDataTypeSize(type)), Offset(0), Normalized(normalized)
	{
	}

	uint32_t GetComponentCount() const
	{
		switch (Type)
		{
		case ShaderDataType::Float:   return 1;
		case ShaderDataType::Float2:  return 2;
		case ShaderDataType::Float3:  return 3;
		case ShaderDataType::Float4:  return 4;
		case ShaderDataType::Mat3:    return 3; // 3* float3
		case ShaderDataType::Mat4:    return 4; // 4* float4
		case ShaderDataType::Int:     return 1;
		case ShaderDataType::Int2:    return 2;
		case ShaderDataType::Int3:    return 3;
		case ShaderDataType::Int4:    return 4;
		case ShaderDataType::Bool:    return 1;
		}

		gablog_log(LOG_ASSERT, __FILE__, __LINE__, "Unknown ShaderDataType!");
		gabdebug_break();
		return 0;
	}
};

struct BufferLayout
{
	BufferLayout() {}

	BufferLayout(std::initializer_list<BufferElement> elements)
		: m_Elements(elements)
	{
		CalculateOffsetsAndStride();
	}

	inline uint32_t GetStride() const { return m_Stride; }
	inline const std::vector<BufferElement>& GetElements() const { return m_Elements; }

	inline std::vector<BufferElement>::iterator begin() { return m_Elements.begin(); }
	inline std::vector<BufferElement>::iterator end() { return m_Elements.end(); }
	inline std::vector<BufferElement>::const_iterator begin() const { return m_Elements.begin(); }
	inline std::vector<BufferElement>::const_iterator end() const { return m_Elements.end(); }
private:
	void CalculateOffsetsAndStride()
	{
		size_t offset = 0;
		m_Stride = 0;
		for (auto& element : m_Elements)
		{
			element.Offset = offset;
			offset += element.Size;
			m_Stride += element.Size;
		}
	}
private:
	std::vector<BufferElement> m_Elements;
	uint32_t m_Stride = 0;
};

struct VertexBuffer
{
	VertexBuffer(uint32_t size);
	VertexBuffer(float* vertices, uint32_t size);
	virtual ~VertexBuffer();

	void Bind() const;
	void Unbind() const;

	void SetData(const void* data, uint32_t size);

	inline const BufferLayout& GetLayout() const { return m_Layout; }
	inline const GLuint GetID() const { return m_RendererID; }
	inline void SetLayout(const BufferLayout& layout) { m_Layout = layout; }
	inline static std::shared_ptr<VertexBuffer> Create(uint32_t size) { return std::make_shared<VertexBuffer>(size); }
	inline static std::shared_ptr<VertexBuffer> Create(float* vertices, uint32_t size) { return std::make_shared<VertexBuffer>(vertices, size); }
private:
	uint32_t m_RendererID;
	BufferLayout m_Layout;
};

struct IndexBuffer
{
	IndexBuffer(uint32_t* indices, uint32_t count);
	virtual ~IndexBuffer();

	void Bind() const;
	void Unbind() const;

	inline uint32_t GetCount() const { return m_Count; }
	inline GLuint GetID() const { return m_RendererID; }
	inline static std::shared_ptr<IndexBuffer> Create(uint32_t* indices, uint32_t count) { return std::make_shared<IndexBuffer>(indices,count); }
private:
	uint32_t m_RendererID;
	uint32_t m_Count;
};

struct VertexArray
{
	VertexArray();
	virtual ~VertexArray();

	void Bind() const;
	void Unbind() const;

	void AddVertexBuffer(const std::shared_ptr<VertexBuffer>& vertexBuffer);
	void SetIndexBuffer(const std::shared_ptr<IndexBuffer>& indexBuffer);

	inline const std::vector<std::shared_ptr<VertexBuffer>>& GetVertexBuffers() const { return m_VertexBuffers; }
	inline const std::shared_ptr<IndexBuffer>& GetIndexBuffer() const { return m_IndexBuffer; }
	inline static std::shared_ptr<VertexArray> Create() { return std::make_shared<VertexArray>(); }
private:
	uint32_t m_RendererID;
	uint32_t m_VertexBufferIndex = 0;
	std::vector<std::shared_ptr<VertexBuffer>> m_VertexBuffers;
  std::shared_ptr<IndexBuffer> m_IndexBuffer;
};

struct UniformBuffer 
{
	UniformBuffer(uint32_t size, uint32_t binding);
	virtual ~UniformBuffer();

	void SetData(const void* data, uint32_t size, uint32_t offset = 0);
	inline static std::shared_ptr<UniformBuffer> Create(uint32_t size, uint32_t binding) { return std::make_shared<UniformBuffer>(size, binding); }
private:
	uint32_t m_RendererID = 0;
};

struct StorageBuffer
{
  StorageBuffer(uint32_t size, uint32_t binding);
  virtual ~StorageBuffer();
  void Allocate(size_t size);
  void SetData(size_t size, const void* data);
  void SetSubData(GLintptr offset, GLsizeiptr size, const void* data);
  void Bind() const;
  void CleanUp();
  void* MapBuffer();
  void UnmapBuffer();
  GLuint GetID() const { return m_RendererID; }
  inline static std::shared_ptr<StorageBuffer> Create(uint32_t size, uint32_t binding) { return std::make_shared<StorageBuffer>(size, binding); }
private:
  uint32_t m_RendererID = 0;
  uint32_t m_Binding = 0;
  size_t bufferSize = 0;
};

struct PixelBuffer
{
  explicit PixelBuffer(size_t size);
  ~PixelBuffer();

  void* Map();
  void Unmap();
  void Bind() const;
  void Unbind() const;
  void WaitForCompletion();

  GLuint GetID() const { return m_ID; }
  size_t GetSize() const { return m_Size; }

private:
  PixelBuffer(const PixelBuffer&) = delete;
  PixelBuffer& operator=(const PixelBuffer&) = delete;

  GLuint m_ID = 0;
  size_t m_Size = 0;
  GLsync m_Sync = nullptr;
};

enum class FramebufferTextureFormat
{
  None = 0,

  // Color
  RGBA8,
  RGBA16F,          
  R11F_G11F_B10F,   
  RED_INTEGER,

  // Depth/stencil
  DEPTH24STENCIL8,
  DEPTH,
};

struct FramebufferTextureSpecification
{
	FramebufferTextureSpecification() = default;
	FramebufferTextureSpecification(FramebufferTextureFormat format)
		: TextureFormat(format) {}

	FramebufferTextureFormat TextureFormat = FramebufferTextureFormat::None;
	// TODO: filtering/wrap
};

struct FramebufferAttachmentSpecification
{
	FramebufferAttachmentSpecification() = default;
	FramebufferAttachmentSpecification(std::initializer_list<FramebufferTextureSpecification> attachments)
		: Attachments(attachments) {}

	std::vector<FramebufferTextureSpecification> Attachments;
};

struct FramebufferSpecification
{
	uint32_t Width = 0, Height = 0;
	FramebufferAttachmentSpecification Attachments;
	uint32_t Samples = 1;
	bool NearestFiltering = false;

	bool SwapChainTarget = false;
};

struct FrameBuffer
{
	FrameBuffer(const FramebufferSpecification& spec);
	~FrameBuffer();

	void Invalidate();

	void Bind();
	void UnBind();

	void Resize(uint32_t width, uint32_t height);
	int ReadPixel(uint32_t attachmentIndex, int x, int y);

	void ClearAttachment(uint32_t attachmentIndex, int value);
  void SetDrawBuffer(uint32_t attachmentIndex) const;
  void SetDrawBuffers() const;
  void AttachExternalColorTexture(GLuint textureID, uint32_t slot = 0);
  void AttachExternalDepthTexture(GLuint textureID);

	inline uint32_t GetColorAttachmentRendererID(uint32_t index = 0) const
	{
		if (index >= m_ColorAttachments.size())
		{
			gablog_log(LOG_ASSERT, __FILE__, __LINE__, "Invalid color attachment index");
			gabdebug_break();
		}
		return m_ColorAttachments[index];
	}
	inline const FramebufferSpecification& GetSpecification() const { return m_Specification; };
  inline uint32_t GetID() const { return m_RendererID; }
  void BlitColor(const std::shared_ptr<FrameBuffer>& dst);

	static std::shared_ptr<FrameBuffer> Create(const FramebufferSpecification& spec);

private:
	uint32_t m_RendererID = 0;
	FramebufferSpecification m_Specification;

	std::vector<FramebufferTextureSpecification> m_ColorAttachmentSpecifications;
	FramebufferTextureSpecification m_DepthAttachmentSpecification = FramebufferTextureFormat::None;

	std::vector<uint32_t> m_ColorAttachments;
	uint32_t m_DepthAttachment = 0;
};

struct GeometryBuffer
{
  GeometryBuffer(uint32_t width, uint32_t height);
  ~GeometryBuffer();

  void Bind() const;
  void UnBind() const;
  void Resize(uint32_t width, uint32_t height);

  void BindPositionTextureForReading(GLenum textureUnit);
  void BindDepthTextureForReading(GLenum textureUnit);
  void BindNormalTextureForReading(GLenum textureUnit);
  void BindAlbedoTextureForReading(GLenum textureUnit);
  void BlitDepthTo(const std::shared_ptr<FrameBuffer>& dst);
  GLuint GetPositionAttachmentRendererID() const { return m_PositionAttachment; }
  GLuint GetNormalAttachmentRendererID() const { return m_NormalAttachment; }
  GLuint GetAlbedoSpecAttachmentRendererID() const { return m_AlbedoSpecAttachment; }
  GLuint GetDepthAttachmentRendererID() const { return m_DepthAttachment; }

  static std::shared_ptr<GeometryBuffer> Create(uint32_t width, uint32_t height);

private:
  void Invalidate();  

  uint32_t m_Width = 0, m_Height = 0;
  GLuint m_FBO = 0;
  GLuint m_PositionAttachment = 0;
  GLuint m_NormalAttachment = 0;
  GLuint m_AlbedoSpecAttachment = 0;
  GLuint m_DepthAttachment = 0;
};

struct BloomBuffer
{
	BloomBuffer(const std::shared_ptr<Shader>& downsampleShader, const std::shared_ptr<Shader>& upsampleShader, const std::shared_ptr<Shader>& finalShader); 
	~BloomBuffer();

	void RenderBloomTexture(float filterRadius, uint32_t mipCount);
  void Bind() const;
  void UnBind() const;
  void Resize(int32_t newWidth, int32_t newHeight);
  void CompositeTo(const std::shared_ptr<FrameBuffer>& dst, bool bloomEnabled,
    float exposure, float strength, float gamma);

	static std::shared_ptr<BloomBuffer> Create(const std::shared_ptr<Shader>& downsampleShader, const std::shared_ptr<Shader>& upsampleShader, const std::shared_ptr<Shader>& finalShader);

private:

  std::shared_ptr<FrameBuffer> m_hdrFB;
  std::shared_ptr<FrameBuffer> m_mipFB;

  const std::shared_ptr<Shader>& downsampleShader; 
  const std::shared_ptr<Shader>& upsampleShader;
  const std::shared_ptr<Shader>& finalShader;
  glm::ivec2 mSrcViewportSize;
  glm::vec2 mSrcViewportSizeFloat;

  struct bloomMip
  {
    glm::vec2 size;
    glm::ivec2 intSize;
    uint32_t texture;
  };

  std::vector<bloomMip> mMipChain;
	const uint32_t mipChainLength = 6; // TODO: Play around with this value
  
	bool mKarisAverageOnDownsample = true;
};

struct DirectShadowBuffer
{
  DirectShadowBuffer(float shadowWidth, float shadowHeight, float offsetSize, float filterSize, float randomRadius);
  ~DirectShadowBuffer();

  void Bind() const;
  void UnBind() const;

  void BindShadowTextureForReading(GLenum textureUnit) const;
  void BindOffsetTextureForReading(GLenum textureUnit) const;

  void UpdateShadowView(const glm::vec3& rotation, const glm::vec3& focusPoint);

  inline glm::mat4 GetShadowViewProj() const { return m_shadowProj * m_shadowVIew; }

  static std::shared_ptr<DirectShadowBuffer> Create(float shadowWidth, float shadowHeight, float offsetSize, float filterSize, float randomRadius);

private:

  float Jitter();

  uint32_t m_shadowWidth = 0, m_shadowHeight = 0;

  GLuint m_FBO = 0;
  GLuint m_depthMap = 0;
  GLuint m_offsetTexture;

  std::shared_ptr<UniformBuffer> buffer;

  glm::mat4 m_shadowProj;
  glm::mat4 m_shadowVIew = glm::mat4(1.0f);
  float m_OrthoSize = 90.0f;
};

struct CubemapDirection
{
  GLenum CubemapFace;
  glm::vec3 Target;
  glm::vec3 Up;
};

struct OmniDirectShadowBuffer
{
  OmniDirectShadowBuffer(uint32_t shadowWidth, uint32_t shadowHeight, uint32_t shadowLightCapacity);
  ~OmniDirectShadowBuffer();

  void Bind() const;
  void UnBind() const;
  void BindCubemapFaceForWriting(uint32_t cubemapIndex, uint32_t faceIndex);
  void BindShadowTextureForReading(GLenum TextureUnit);

  inline glm::mat4 GetShadowProj() const { return m_shadowProj; }

  inline std::span<const CubemapDirection, 6> GetFaceDirections() const { return m_Directions; }

	static auto Create(uint32_t shadowWidth,
	                   uint32_t shadowHeight, uint32_t shadowLightCapacity) -> std::shared_ptr<OmniDirectShadowBuffer>;

private:

  std::array<CubemapDirection, 6> m_Directions;
  std::shared_ptr<FrameBuffer> m_testFB;

  glm::mat4 m_shadowProj;
  uint32_t m_depthCubemapArray;
  uint32_t m_ShadowLightCapacity = 0;
};

struct DrawIndirectBuffer
{
  DrawIndirectBuffer() = default;
  ~DrawIndirectBuffer() = default;

  void AddData(const std::string& modelName, uint32_t verticesSize, uint32_t indicesSize);
  void UploadToGPU();

private:
  struct DrawElementsIndirectCommand
  {
    GLuint count;         // Number of indices
    GLuint instanceCount; // This is for instancing
    GLuint firstIndex;    // Offset into the index buffer
    GLint baseVertex;    // Base vertex for this draw
    GLuint baseInstance;  // Use this to index per-object data
  };

  std::vector<DrawElementsIndirectCommand> m_DrawCommands;
  std::unordered_map<std::string, std::vector<size_t>> m_ModelDrawCommandIndices;
  uint32_t m_DrawIndexOffset = 0;
  uint32_t m_DrawVertexOffset = 0;
  uint32_t m_cmdBufer;
};
