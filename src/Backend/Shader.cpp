#include "Shader.h"
#include "Timer.hpp"

#include <gabdebug.h>
#include <glad/glad.h>

#include <slang-com-ptr.h>
#include <slang.h>

#include <array>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <iostream>
#include <stdexcept>

namespace
{
using Slang::ComPtr;

void LogSlangDiagnostics(slang::IBlob* diagnostics)
{
  if (diagnostics && diagnostics->getBufferSize() > 0)
    gablog_log(LOG_WARN, __FILE__, __LINE__, "%s", static_cast<const char*>(diagnostics->getBufferPointer()));
}

slang::IGlobalSession* GetSlangGlobalSession()
{
  static ComPtr<slang::IGlobalSession> session = []
  {
    ComPtr<slang::IGlobalSession> result;
    SlangGlobalSessionDesc description{};
    description.enableGLSL = true;
    if (SLANG_FAILED(slang::createGlobalSession(&description, result.writeRef())))
      throw std::runtime_error("Failed to initialize the Slang compiler");
    return result;
  }();
  return session.get();
}

Shader::Bytecode CompileSlangSource(const std::string& source,
                                    const std::filesystem::path& path,
                                    std::string_view entryPoint,
                                    SlangStage stage,
                                    SlangCompileTarget format,
                                    const char* profile,
                                    bool allowGLSLSyntax)
{
  auto* globalSession = GetSlangGlobalSession();

  slang::TargetDesc target{};
  target.format = format;
  target.profile = globalSession->findProfile(profile);

  slang::CompilerOptionEntry options[3]{};
  options[0].name = slang::CompilerOptionName::NoMangle;
  options[0].value.intValue0 = 1;
  options[1].name = slang::CompilerOptionName::PreserveParameters;
  options[1].value.intValue0 = 1;
  // OpenGL uses separate binding namespaces for textures, UBOs and SSBOs,
  // while Slang's cross-API validator conservatively treats them as shared.
  options[2].name = slang::CompilerOptionName::DisableWarnings;
  options[2].value.kind = slang::CompilerOptionValueKind::String;
  options[2].value.stringValue0 = "39001";

  const std::string searchPath = path.parent_path().string();
  const char* searchPaths[] = {searchPath.c_str()};
  slang::SessionDesc sessionDescription{};
  // Slang's GLSL compatibility front-end reverses this setting when emitting
  // the explicit GLSL buffer qualifier. OpenGL therefore needs row-major here
  // to retain column-major UBO/SSBO storage, while DXBC must keep the original
  // HLSL column-major constant-buffer ABI used by GLM uploads.
  sessionDescription.defaultMatrixLayoutMode = format == SLANG_GLSL
    ? SLANG_MATRIX_LAYOUT_ROW_MAJOR
    : SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;
  sessionDescription.targets = &target;
  sessionDescription.targetCount = 1;
  sessionDescription.searchPaths = searchPaths;
  sessionDescription.searchPathCount = 1;
  sessionDescription.allowGLSLSyntax = allowGLSLSyntax;
  sessionDescription.compilerOptionEntries = options;
  sessionDescription.compilerOptionEntryCount = static_cast<uint32_t>(std::size(options));

  ComPtr<slang::ISession> session;
  if (SLANG_FAILED(globalSession->createSession(sessionDescription, session.writeRef())))
    throw std::runtime_error("Failed to create a Slang compilation session");

  ComPtr<slang::IBlob> sourceBlob;
  sourceBlob.attach(slang_createBlob(source.data(), source.size()));
  ComPtr<slang::IBlob> diagnostics;
  const std::string moduleName = path.stem().string() + "_" + std::string(entryPoint);
  const std::string sourcePath = path.string();
  slang::IModule* module = session->loadModuleFromSource(
    moduleName.c_str(), sourcePath.c_str(), sourceBlob.get(), diagnostics.writeRef());
  LogSlangDiagnostics(diagnostics.get());
  if (!module)
    throw std::runtime_error("Slang failed to load shader " + sourcePath);

  ComPtr<slang::IEntryPoint> entry;
  diagnostics.setNull();
  const std::string entryName(entryPoint);
  const SlangResult entryResult = module->findAndCheckEntryPoint(
    entryName.c_str(), stage, entry.writeRef(), diagnostics.writeRef());
  LogSlangDiagnostics(diagnostics.get());
  if (SLANG_FAILED(entryResult))
    throw std::runtime_error("Slang entry point '" + entryName + "' failed in " + sourcePath);

  slang::IComponentType* components[] = {module, entry.get()};
  ComPtr<slang::IComponentType> program;
  diagnostics.setNull();
  const SlangResult composeResult = session->createCompositeComponentType(
    components, std::size(components), program.writeRef(), diagnostics.writeRef());
  LogSlangDiagnostics(diagnostics.get());
  if (SLANG_FAILED(composeResult))
    throw std::runtime_error("Slang failed to compose shader " + sourcePath);

  ComPtr<slang::IComponentType> linkedProgram;
  diagnostics.setNull();
  const SlangResult linkResult = program->link(linkedProgram.writeRef(), diagnostics.writeRef());
  LogSlangDiagnostics(diagnostics.get());
  if (SLANG_FAILED(linkResult))
    throw std::runtime_error("Slang failed to link shader " + sourcePath);

  ComPtr<slang::IBlob> code;
  diagnostics.setNull();
  const SlangResult codeResult = linkedProgram->getEntryPointCode(
    0, 0, code.writeRef(), diagnostics.writeRef());
  LogSlangDiagnostics(diagnostics.get());
  if (SLANG_FAILED(codeResult))
    throw std::runtime_error("Slang code generation failed for " + sourcePath);

  Shader::Bytecode bytecode;
  const auto* begin = static_cast<const uint8_t*>(code->getBufferPointer());
  bytecode.Bytes.assign(begin, begin + code->getBufferSize());
  return bytecode;
}

std::string UnpackSlangGlobalUniforms(std::string source)
{
  constexpr std::string_view blockMarker = "uniform block_GlobalParams_";
  size_t marker = source.find(blockMarker);
  while (marker != std::string::npos)
  {
    const size_t layoutStart = source.rfind("layout(binding", marker);
    const size_t bodyStart = source.find('{', marker);
    const size_t bodyEnd = source.find('}', bodyStart);
    const size_t declarationEnd = source.find(';', bodyEnd);
    if (layoutStart == std::string::npos || bodyStart == std::string::npos ||
        bodyEnd == std::string::npos || declarationEnd == std::string::npos)
      break;

    const size_t instanceStart = source.find_first_not_of(" \t\r\n", bodyEnd + 1);
    const std::string instanceName = source.substr(instanceStart, declarationEnd - instanceStart);
    std::istringstream members(source.substr(bodyStart + 1, bodyEnd - bodyStart - 1));
    std::string uniforms;
    std::string line;
    while (std::getline(members, line))
    {
      const size_t first = line.find_first_not_of(" \t\r");
      if (first == std::string::npos)
        continue;
      uniforms += "uniform " + line.substr(first) + "\n";
    }

    source.replace(layoutStart, declarationEnd - layoutStart + 1, uniforms);
    const std::string prefix = instanceName + ".";
    size_t use = source.find(prefix);
    while (use != std::string::npos)
    {
      source.erase(use, prefix.size());
      use = source.find(prefix, use);
    }
    marker = source.find(blockMarker);
  }
  return source;
}

void ReplaceIdentifier(std::string& source, std::string_view from, std::string_view to)
{
  size_t position = source.find(from);
  while (position != std::string::npos)
  {
    const auto isIdentifier = [](const char character)
    {
      return std::isalnum(static_cast<unsigned char>(character)) || character == '_';
    };
    const bool startsIdentifier = position > 0 && isIdentifier(source[position - 1]);
    const size_t after = position + from.size();
    const bool endsIdentifier = after < source.size() && isIdentifier(source[after]);
    if (!startsIdentifier && !endsIdentifier)
    {
      source.replace(position, from.size(), to);
      position = source.find(from, position + to.size());
    }
    else
      position = source.find(from, after);
  }
}

void CheckCompileErrors(GLuint object, std::string_view type, std::string_view generatedSource = {})
{
  GLint success = GL_FALSE;
  if (type != "PROGRAM")
  {
    glGetShaderiv(object, GL_COMPILE_STATUS, &success);
    if (success == GL_TRUE) return;
    GLint length = 0;
    glGetShaderiv(object, GL_INFO_LOG_LENGTH, &length);
    std::string log(static_cast<size_t>(std::max(length, 1)), '\0');
    glGetShaderInfoLog(object, length, nullptr, log.data());
    gablog_log(LOG_ERROR, __FILE__, __LINE__, "OpenGL shader compilation failed for %.*s: %s",
      static_cast<int>(type.size()), type.data(), log.c_str());
    if (!generatedSource.empty())
      gablog_log(LOG_ERROR, __FILE__, __LINE__, "Generated GLSL source:\n%.*s",
        static_cast<int>(generatedSource.size()), generatedSource.data());
    throw std::runtime_error("OpenGL shader compilation failed");
  }

  glGetProgramiv(object, GL_LINK_STATUS, &success);
  if (success == GL_TRUE) return;
  GLint length = 0;
  glGetProgramiv(object, GL_INFO_LOG_LENGTH, &length);
  std::string log(static_cast<size_t>(std::max(length, 1)), '\0');
  glGetProgramInfoLog(object, length, nullptr, log.data());
  gablog_log(LOG_ERROR, __FILE__, __LINE__, "OpenGL shader linking failed: %s", log.c_str());
  throw std::runtime_error("OpenGL shader linking failed");
}

GLuint CompileOpenGLStage(const std::string& source,
                          const std::filesystem::path& path,
                          GLenum shaderType,
                          SlangStage stage)
{
  auto generated = CompileSlangSource(
    source, path, "main", stage, SLANG_GLSL, "glsl_460", true);
  std::string glsl(reinterpret_cast<const char*>(generated.Bytes.data()), generated.Bytes.size());
  glsl = UnpackSlangGlobalUniforms(std::move(glsl));
  if (source.find("GL_ARB_bindless_texture") != std::string::npos)
  {
    const size_t versionEnd = glsl.find('\n');
    glsl.insert(versionEnd + 1, "#extension GL_ARB_bindless_texture : require\n");
  }
  // Slang's GLSL target uses Vulkan names for these two built-ins. OpenGL
  // exposes their equivalent core names instead.
  ReplaceIdentifier(glsl, "gl_InstanceIndex", "gl_InstanceID");
  ReplaceIdentifier(glsl, "gl_VertexIndex", "gl_VertexID");
  const char* code = glsl.c_str();
  const GLuint shader = glCreateShader(shaderType);
  glShaderSource(shader, 1, &code, nullptr);
  glCompileShader(shader);
  CheckCompileErrors(shader, path.string(), glsl);
  return shader;
}

std::string ReadTextFile(const std::filesystem::path& path)
{
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("Cannot open shader " + path.string());
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

std::string SelectApiSection(const std::string& source,
                             std::string_view api,
                             const std::filesystem::path& path)
{
  constexpr std::string_view markerPrefix = "#api ";
  if (source.find(markerPrefix) == std::string::npos)
    return source;

  const std::string marker = std::string(markerPrefix) + std::string(api);
  const size_t markerPosition = source.find(marker);
  if (markerPosition == std::string::npos)
    throw std::runtime_error("Shader " + path.string() + " has no " + std::string(api) + " section");

  const size_t sectionStart = source.find('\n', markerPosition);
  if (sectionStart == std::string::npos)
    return {};

  const size_t nextMarker = source.find("\n#api ", sectionStart + 1);
  return source.substr(sectionStart + 1,
                       nextMarker == std::string::npos ? std::string::npos
                                                       : nextMarker - sectionStart - 1);
}
} // namespace

Shader::Shader(const char* fullshader)
{
    Timer timer;
    Load(fullshader);
    gablog_log(LOG_WARN, __FILE__, __LINE__, "Shader creation took %.3f ms", timer.ElapsedMillis());
}

Shader::Shader(const char* vertexPath, const char* fragmentPath, const char* geometryPath)
{
    Timer timer;
    Load(vertexPath, fragmentPath, geometryPath);
    gablog_log(LOG_WARN, __FILE__, __LINE__, "Shader creation took %.3f ms", timer.ElapsedMillis());
}

Shader::~Shader()
{
  if (m_ID != 0)
    glDeleteProgram(m_ID);
}

void Shader::Load(const char* fullshader)
{
  m_UniformLocations.clear();
  const std::filesystem::path path(fullshader);
  const std::string fileContent = SelectApiSection(ReadTextFile(path), "OPENGL", path);

  // Parse the shader file into sections based on #type
  std::unordered_map<std::string, std::string> shaderSources;
  const std::string typeToken = "#type";
  size_t pos = 0;
  while ((pos = fileContent.find(typeToken, pos)) != std::string::npos) {
      size_t endOfLine = fileContent.find('\n', pos);
      std::string type = fileContent.substr(pos + typeToken.length(), endOfLine - pos - typeToken.length());
      type = type.substr(type.find_first_not_of(" \t\r\n")); // Trim leading whitespace
      type = type.substr(0, type.find_last_not_of(" \t\r\n") + 1); // Trim trailing whitespace

      size_t nextTypePos = fileContent.find(typeToken, endOfLine + 1);
      if (nextTypePos == std::string::npos)
          nextTypePos = fileContent.size();

      std::string source = fileContent.substr(endOfLine + 1, nextTypePos - endOfLine - 1);
      shaderSources[type] = source;

      pos = nextTypePos;
  }

  struct StageDescription
  {
    const char* Name;
    GLenum OpenGLStage;
    SlangStage SlangStageValue;
  };
  constexpr StageDescription stages[] = {
    {"VERTEX", GL_VERTEX_SHADER, SLANG_STAGE_VERTEX},
    {"FRAGMENT", GL_FRAGMENT_SHADER, SLANG_STAGE_FRAGMENT},
    {"GEOMETRY", GL_GEOMETRY_SHADER, SLANG_STAGE_GEOMETRY},
    {"TESS_CONTROL", GL_TESS_CONTROL_SHADER, SLANG_STAGE_HULL},
    {"TESS_EVALUATION", GL_TESS_EVALUATION_SHADER, SLANG_STAGE_DOMAIN},
    {"COMPUTE", GL_COMPUTE_SHADER, SLANG_STAGE_COMPUTE},
  };

  std::vector<GLuint> compiledStages;
  for (const auto& stage : stages)
  {
    const auto source = shaderSources.find(stage.Name);
    if (source != shaderSources.end())
      compiledStages.push_back(CompileOpenGLStage(
        source->second, path, stage.OpenGLStage, stage.SlangStageValue));
  }
  if (compiledStages.empty())
    throw std::runtime_error("Shader has no #type sections: " + path.string());

  // Link shaders into a program
  this->m_ID = glCreateProgram();
  for (const GLuint stage : compiledStages)
    glAttachShader(this->m_ID, stage);
  glLinkProgram(this->m_ID);
  CheckCompileErrors(this->m_ID, "PROGRAM");

  // Validate the program
  glValidateProgram(this->m_ID);
  GLint isValid;
  glGetProgramiv(this->m_ID, GL_VALIDATE_STATUS, &isValid);
  if (!isValid) {
      char infoLog[1024];
      glGetProgramInfoLog(this->m_ID, 1024, NULL, infoLog);
      gablog_log(LOG_ERROR, __FILE__, __LINE__, "ERROR::PROGRAM_VALIDATION_ERROR: %s", infoLog);
  }

  for (const GLuint stage : compiledStages)
    glDeleteShader(stage);
}

void Shader::Load(const char* vertexPath, const char* fragmentPath, const char* geometryPath)
{
  m_UniformLocations.clear();
  const GLuint vertex = CompileOpenGLStage(
    SelectApiSection(ReadTextFile(vertexPath), "OPENGL", vertexPath),
    vertexPath, GL_VERTEX_SHADER, SLANG_STAGE_VERTEX);
  const GLuint fragment = CompileOpenGLStage(
    SelectApiSection(ReadTextFile(fragmentPath), "OPENGL", fragmentPath),
    fragmentPath, GL_FRAGMENT_SHADER, SLANG_STAGE_FRAGMENT);
  GLuint geometry = 0;
  if (geometryPath)
    geometry = CompileOpenGLStage(
      SelectApiSection(ReadTextFile(geometryPath), "OPENGL", geometryPath),
      geometryPath, GL_GEOMETRY_SHADER, SLANG_STAGE_GEOMETRY);

  this->m_ID = glCreateProgram();
  glAttachShader(this->m_ID, vertex);
  glAttachShader(this->m_ID, fragment);
  if (geometry != 0) glAttachShader(this->m_ID, geometry);
  glLinkProgram(this->m_ID);
  CheckCompileErrors(this->m_ID, "PROGRAM");

  glDeleteShader(vertex);
  glDeleteShader(fragment);
  if (geometry != 0) glDeleteShader(geometry);
}

void Shader::Bind() const
{
  glUseProgram(this->m_ID);
}
void Shader::UnBind() const
{
  glUseProgram(0);
}
uint32_t Shader::GetID() const
{
  return this->m_ID;
}
bool Shader::CheckIfModified(std::shared_ptr<Shader>& shader, const char* fullshader)
{  
  auto ftime = std::filesystem::last_write_time(fullshader);

  auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
      ftime - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now()
  );
  std::time_t cftime = std::chrono::system_clock::to_time_t(sctp);

  return (cftime != shader->m_lastTimeModified) ? true : false;
}

int Shader::GetUniformLocation(const std::string& name) const
{
  if (const auto found = m_UniformLocations.find(name);
      found != m_UniformLocations.end())
    return found->second;

  const GLint location = glGetUniformLocation(m_ID, name.c_str());
  m_UniformLocations.emplace(name, location);
  return location;
}

void Shader::SetBool(const std::string& name, bool value) const
{
  glUniform1i(GetUniformLocation(name), static_cast<int>(value));
}
void Shader::SetInt(const std::string& name, int value) const
{
  glUniform1i(GetUniformLocation(name), value);
}
void Shader::SetFloat(const std::string& name, float value) const
{
  glUniform1f(GetUniformLocation(name), value);
}
void Shader::SetVec2(const std::string& name, const glm::vec2& value) const
{
  glUniform2fv(GetUniformLocation(name), 1, &value[0]);
}
void Shader::SetVec2(const std::string& name, float x, float y) const
{
  glUniform2f(GetUniformLocation(name), x, y);
}
void Shader::SetVec3(const std::string& name, const glm::vec3& value) const
{
  glUniform3fv(GetUniformLocation(name), 1, &value[0]);
}
void Shader::SetVec3(const std::string& name, float x, float y, float z) const
{
  glUniform3f(GetUniformLocation(name), x, y, z);
}
void Shader::SetVec4(const std::string& name, const glm::vec4& value) const
{
  glUniform4fv(GetUniformLocation(name), 1, &value[0]);
}
void Shader::SetVec4(const std::string& name, float x, float y, float z, float w) const
{
  glUniform4f(GetUniformLocation(name), x, y, z, w);
}
void Shader::SetMat2(const std::string& name, const glm::mat2& mat) const
{
  glUniformMatrix2fv(GetUniformLocation(name), 1, GL_FALSE, &mat[0][0]);
}
void Shader::SetMat3(const std::string& name, const glm::mat3& mat) const
{
  glUniformMatrix3fv(GetUniformLocation(name), 1, GL_FALSE, &mat[0][0]);
}
void Shader::SetMat4(const std::string& name, const glm::mat4& mat) const
{
  glUniformMatrix4fv(GetUniformLocation(name), 1, GL_FALSE, &mat[0][0]);
}

void Shader::Create(std::shared_ptr<Shader>& shader, const char* fullshader)
{ 
  auto ftime = std::filesystem::last_write_time(fullshader);

  auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
      ftime - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now()
  );
  std::time_t cftime = std::chrono::system_clock::to_time_t(sctp);

  if(shader && !shader->m_firstTimeCompile && cftime == shader->m_lastTimeModified) return;
   
  shader = std::make_shared<Shader>(fullshader);
  shader->m_lastTimeModified = cftime;
  shader->m_firstTimeCompile = false;
}

void Shader::Create(std::shared_ptr<Shader>& shader, const char* vertexPath, const char* fragmentPath, const char* geometryPath)
{
  auto vtime = std::filesystem::last_write_time(vertexPath);
  auto ftime = std::filesystem::last_write_time(fragmentPath);

  std::filesystem::file_time_type gtime{};
  if (geometryPath && std::filesystem::exists(geometryPath))
  {
    gtime = std::filesystem::last_write_time(geometryPath);
  } 
  else
  {
    gtime = std::filesystem::file_time_type::min();
  }

  auto latest = std::max({ vtime, ftime, gtime });

  auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
      latest - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now()
  );

  std::time_t cftime = std::chrono::system_clock::to_time_t(sctp);

  if(shader && !shader->m_firstTimeCompile && cftime == shader->m_lastTimeModified) return;

  shader = std::make_shared<Shader>(vertexPath,fragmentPath,geometryPath);
  shader->m_lastTimeModified = cftime;
  shader->m_firstTimeCompile = false;
}

Shader::Bytecode Shader::CompileSlang(const std::filesystem::path& path, std::string_view entryPoint,
                                      std::string_view target)
{
#if defined(GABGL_ENABLE_DX12) && defined(_WIN32)
  SlangStage stage = SLANG_STAGE_NONE;
  if (target.starts_with("vs_")) stage = SLANG_STAGE_VERTEX;
  else if (target.starts_with("ps_")) stage = SLANG_STAGE_FRAGMENT;
  else if (target.starts_with("gs_")) stage = SLANG_STAGE_GEOMETRY;
  else if (target.starts_with("hs_")) stage = SLANG_STAGE_HULL;
  else if (target.starts_with("ds_")) stage = SLANG_STAGE_DOMAIN;
  else if (target.starts_with("cs_")) stage = SLANG_STAGE_COMPUTE;
  if (stage == SLANG_STAGE_NONE)
    throw std::runtime_error("Unsupported Slang shader profile: " + std::string(target));

  return CompileSlangSource(
    SelectApiSection(ReadTextFile(path), "DX12", path),
    path, entryPoint, stage, SLANG_DXBC, std::string(target).c_str(), false);
#else
  gablog_log(LOG_ERROR, __FILE__, __LINE__, "Cannot compile Slang shader '%s': DirectX 12 support is not enabled",
    path.string().c_str());
  return {};
#endif
}

Shader::Bytecode Shader::CompileSlangSPIRV(const std::filesystem::path& path,
                                           std::string_view entryPoint,
                                           std::string_view stageName)
{
#if defined(GABGL_ENABLE_VULKAN)
  SlangStage stage = SLANG_STAGE_NONE;
  if (stageName == "vertex") stage = SLANG_STAGE_VERTEX;
  else if (stageName == "fragment") stage = SLANG_STAGE_FRAGMENT;
  else if (stageName == "compute") stage = SLANG_STAGE_COMPUTE;
  if (stage == SLANG_STAGE_NONE)
    throw std::runtime_error("Unsupported Vulkan shader stage: " + std::string(stageName));

  return CompileSlangSource(
    SelectApiSection(ReadTextFile(path), "VULKAN", path),
    path, entryPoint, stage, SLANG_SPIRV, "spirv_1_5", false);
#else
  gablog_log(LOG_ERROR, __FILE__, __LINE__, "Cannot compile Slang shader '%s': Vulkan support is not enabled",
    path.string().c_str());
  return {};
#endif
}

