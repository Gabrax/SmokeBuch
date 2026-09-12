#pragma once
#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <memory>
#include <filesystem>
#include <chrono>
#include <ctime>
#include <string_view>
#include <unordered_map>
#include <vector>

struct Shader
{
  struct Bytecode
  {
    std::vector<uint8_t> Bytes;

    [[nodiscard]] const void* GetBufferPointer() const { return Bytes.data(); }
    [[nodiscard]] size_t GetBufferSize() const { return Bytes.size(); }
  };

  explicit Shader(const char* fullshader);
  Shader(const char* vertexPath, const char* fragmentPath, const char* geometryPath = nullptr);
  ~Shader();

  void Load(const char* fullshader);
  void Load(const char* vertexPath, const char* fragmentPath, const char* geometryPath = nullptr);
  void Bind() const;
  void UnBind() const;
  uint32_t GetID() const;

  static bool CheckIfModified(std::shared_ptr<Shader>& shader, const char* fullshader);

  void SetBool(const std::string& name, bool value) const;
  void SetInt(const std::string& name, int value) const;
  void SetFloat(const std::string& name, float value) const;
  void SetVec2(const std::string& name, const glm::vec2& value) const;
  void SetVec2(const std::string& name, float x, float y) const;
  void SetVec3(const std::string& name, const glm::vec3& value) const;
  void SetVec3(const std::string& name, float x, float y, float z) const;
  void SetVec4(const std::string& name, const glm::vec4& value) const;
  void SetVec4(const std::string& name, float x, float y, float z, float w) const;
  void SetMat2(const std::string& name, const glm::mat2& mat) const;
  void SetMat3(const std::string& name, const glm::mat3& mat) const;
  void SetMat4(const std::string& name, const glm::mat4& mat) const;
  
  static void Create(std::shared_ptr<Shader>& shader, const char* fullshader);
  static void Create(std::shared_ptr<Shader>& shader, const char* vertexPath, const char* fragmentPath, const char* geometryPath = nullptr);
  static Bytecode CompileSlang(const std::filesystem::path& path, std::string_view entryPoint,
                               std::string_view target);
  static Bytecode CompileSlangSPIRV(const std::filesystem::path& path,
                                    std::string_view entryPoint,
                                    std::string_view stage);

private:
  int GetUniformLocation(const std::string& name) const;

  uint32_t m_ID = 0;
  std::time_t m_lastTimeModified = 0;
  bool m_firstTimeCompile = true;
  mutable std::unordered_map<std::string, int> m_UniformLocations;
};
