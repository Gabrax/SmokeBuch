#include "window.h"
#if defined(GABGL_ENABLE_DX12) && defined(_WIN32)
#ifdef APIENTRY
#undef APIENTRY
#endif
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif
#include "../input/EngineEvent.h"
#include "../input/KeyEvent.h"
#include <gabdebug.h>
#include <stb_image.h>
#include "Camera.h"
#include "Settings.h"

#include <algorithm>

GLFWwindow* m_Window;
GLFWmonitor* m_Monitor;
const GLFWvidmode* m_Mode;

int32_t currWidth, currHeight;
int32_t currX = 100, currY = 100;
bool m_WindowClosed = false;

struct WindowSpecificData
{
  std::string title;
  uint32_t Width, Height;
  bool VSync;
  GraphicsAPI API;

  Window::EventCallbackFn EventCallback;
} m_Data;

bool m_isMinimized = false;
bool m_isRunning = true;
bool m_closed = false;

static void GLFWErrorCallback(int error, const char* description)
{
	gablog_log(LOG_ERROR, __FILE__, __LINE__, "GLFW Error (%d): %s", error, description);
}

void Window::Init(const std::string& windowTitle, uint32_t windowWidth, uint32_t windowHeight,
                  GraphicsAPI graphicsAPI)
{
  m_Data.title = windowTitle;
  m_Data.Width = windowWidth;
  m_Data.Height = windowHeight;
  m_Data.API = graphicsAPI;

  gablog_log(LOG_INFO, __FILE__, __LINE__, "Creating window %s (%u,%u)", m_Data.title.c_str(), m_Data.Width, m_Data.Height);

  int GLFWstatus = glfwInit();
  if (!GLFWstatus)
  {
    gablog_log(LOG_ASSERT, __FILE__, __LINE__, "Failed to init GLFW");
    gabdebug_break();
  }
  glfwSetErrorCallback(GLFWErrorCallback);

  glfwDefaultWindowHints();
  if (m_Data.API == GraphicsAPI::OpenGL)
  {
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifndef NDEBUG
	  glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif
  }
  else
  {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  }

  m_Window = glfwCreateWindow((int)m_Data.Width, (int)m_Data.Height, m_Data.title.c_str(), nullptr, nullptr);
  if (!m_Window)
  {
    gablog_log(LOG_ASSERT, __FILE__, __LINE__, "Failed to create GLFW window");
    gabdebug_break();
  }
  m_Monitor = glfwGetPrimaryMonitor();
  m_Mode = glfwGetVideoMode(m_Monitor);
  if (m_Data.API == GraphicsAPI::OpenGL)
  {
    glfwMakeContextCurrent(m_Window);

	  int status = gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
	  if (!status)
	  {
	    gablog_log(LOG_ASSERT, __FILE__, __LINE__, "Failed to initialize Glad!");
	    gabdebug_break();
	  }

	  gablog_log(LOG_INFO, __FILE__, __LINE__, "OpenGL Info:");
	  gablog_log(LOG_INFO, __FILE__, __LINE__, "  Vendor: %s", reinterpret_cast<const char*>(glGetString(GL_VENDOR)));
	  gablog_log(LOG_INFO, __FILE__, __LINE__, "  Renderer: %s", reinterpret_cast<const char*>(glGetString(GL_RENDERER)));
	  gablog_log(LOG_INFO, __FILE__, __LINE__, "  Version: %s", reinterpret_cast<const char*>(glGetString(GL_VERSION)));

	  if (GLVersion.major < 4 || (GLVersion.major == 4 && GLVersion.minor < 5))
	  {
	    gablog_log(LOG_ASSERT, __FILE__, __LINE__, "GABGL requires at least OpenGL version 4.5!");
	    gabdebug_break();
	  }
  }

  glfwSetWindowUserPointer(m_Window, &m_Data);

  SetWindowIcon("res/textures/gabglicon.png", m_Window);
  SetResizable(false);
  CenterWindowPos();
  SetVSync(false);
  Maximize(false);

  // Set GLFW callbacks
  glfwSetWindowSizeCallback(m_Window, [](GLFWwindow* window, int width, int height)
	  {
		  WindowSpecificData& data = *static_cast<WindowSpecificData*>(glfwGetWindowUserPointer(window));
		  data.Width = width;
		  data.Height = height;

		  WindowResizeEvent event(width, height);
		  Window::OnEvent(event);
	  });

  glfwSetWindowCloseCallback(m_Window, [](GLFWwindow* window)
	  {
		  WindowCloseEvent event;
		  Window::OnEvent(event);
	  });

  glfwSetKeyCallback(m_Window, [](GLFWwindow* window, int key, int scancode, int action, int mods)
	  {
		  switch (action)
		  {
			  case GLFW_PRESS:
			  {
				  KeyPressedEvent event(key, 0);
				  Window::OnEvent(event);
				  break;
			  }
			  case GLFW_RELEASE:
			  {
				  KeyReleasedEvent event(key);
				  Window::OnEvent(event);
				  break;
			  }
			  case GLFW_REPEAT:
			  {
				  KeyPressedEvent event(key, true);
				  Window::OnEvent(event);
				  break;
			  }
		  }
	  });

  glfwSetCharCallback(m_Window, [](GLFWwindow* window, unsigned int keycode)
	  {
		  KeyTypedEvent event(keycode);
		  Window::OnEvent(event);
	  });

  glfwSetMouseButtonCallback(m_Window, [](GLFWwindow* window, int button, int action, int mods)
	  {
		  switch (action)
		  {
			  case GLFW_PRESS:
			  {
				  MouseButtonPressedEvent event(button);
				  Window::OnEvent(event);
				  break;
			  }
			  case GLFW_RELEASE:
			  {
				  MouseButtonReleasedEvent event(button);
				  Window::OnEvent(event);
				  break;
			  }
		  }
	  });

  glfwSetScrollCallback(m_Window, [](GLFWwindow* window, double xOffset, double yOffset)
	  {
		  MouseScrolledEvent event(static_cast<float>(xOffset), static_cast<float>(yOffset));
		  Window::OnEvent(event);
	  });

  glfwSetCursorPosCallback(m_Window, [](GLFWwindow* window, double xPos, double yPos)
	  {
		  MouseMovedEvent event(static_cast<float>(xPos), static_cast<float>(yPos));
		  Window::OnEvent(event);
	  });

  if (m_Data.API == GraphicsAPI::OpenGL)
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

  SetEventCallback({});
}

void Window::Terminate()
{
  if (m_Window)
  {
	  glfwDestroyWindow(m_Window);
    m_Window = nullptr;
  }
  glfwTerminate();
  m_isRunning = false;
}

void Window::Update()
{
	PollEvents();
	Present();
}

void Window::PollEvents()
{
	glfwPollEvents();
	if (glfwWindowShouldClose(m_Window))
		m_isRunning = false;
}

void Window::Present()
{
	if (m_Data.API == GraphicsAPI::OpenGL)
	  glfwSwapBuffers(m_Window);
}

void Window::SetWindowIcon(const char* iconpath, GLFWwindow* window)
{
  stbi_set_flip_vertically_on_load(0);
  GLFWimage images[1];
  images[0].pixels = stbi_load(iconpath, &images[0].width, &images[0].height, 0, 4);
  if (images[0].pixels) {
    glfwSetWindowIcon(window, 1, images);
    stbi_image_free(images[0].pixels);
  }
}

void Window::SetVSync(bool enabled)
{
  if (m_Data.API == GraphicsAPI::OpenGL)
  {
    if(enabled) glfwSwapInterval(1);
    else glfwSwapInterval(0);
  }

  m_Data.VSync = enabled;
}

bool Window::IsVSync() 
{
  return m_Data.VSync;
}

void Window::SetResolution(uint32_t width, uint32_t height) 
{ 
  glfwSetWindowSize(m_Window, static_cast<int>(width), static_cast<int>(height));
  if (m_Data.API == GraphicsAPI::OpenGL)
    glViewport(0, 0, width, height);
  m_Data.Width = width;
  m_Data.Height = height;
}

void Window::CenterWindowPos()
{
  glfwGetWindowSize(m_Window, &currWidth, &currHeight);
  int monitorX = 0, monitorY = 0, workWidth = m_Mode->width, workHeight = m_Mode->height;
  glfwGetMonitorWorkarea(m_Monitor, &monitorX, &monitorY, &workWidth, &workHeight);
  glfwSetWindowPos(m_Window,
    monitorX + (workWidth - currWidth) / 2,
    monitorY + (workHeight - currHeight) / 2);
}

void Window::SetFullscreen(bool full)
{
  SetWindowMode(full ? WindowMode::Fullscreen : WindowMode::Windowed, currWidth, currHeight);
}

void Window::SetWindowMode(WindowMode mode, uint32_t width, uint32_t height)
{
  if (glfwGetWindowMonitor(m_Window) == nullptr && glfwGetWindowAttrib(m_Window, GLFW_DECORATED))
  {
    glfwGetWindowPos(m_Window, &currX, &currY);
    glfwGetWindowSize(m_Window, &currWidth, &currHeight);
  }

  width = std::max(width, 1u);
  height = std::max(height, 1u);

  int monitorX = 0, monitorY = 0;
  int workWidth = m_Mode->width, workHeight = m_Mode->height;
  glfwGetMonitorWorkarea(m_Monitor, &monitorX, &monitorY, &workWidth, &workHeight);

  switch (mode)
  {
    case WindowMode::Fullscreen:
      glfwSetWindowAttrib(m_Window, GLFW_DECORATED, GLFW_TRUE);
      glfwSetWindowMonitor(m_Window, m_Monitor, 0, 0, static_cast<int>(width), static_cast<int>(height), m_Mode->refreshRate);
      break;
    case WindowMode::Borderless:
    {
      glfwGetMonitorPos(m_Monitor, &monitorX, &monitorY);
      glfwSetWindowAttrib(m_Window, GLFW_DECORATED, GLFW_FALSE);
      glfwSetWindowMonitor(m_Window, nullptr, monitorX, monitorY,
                           m_Mode->width, m_Mode->height, GLFW_DONT_CARE);
      break;
    }
    case WindowMode::Windowed:
    default:
    {
      // GLFW sizes are client-area sizes in logical screen coordinates. Keep
      // the requested window inside the usable monitor area and restore the
      // decoration before selecting its final client size.
      width = std::min(width, static_cast<uint32_t>(std::max(workWidth, 1)));
      height = std::min(height, static_cast<uint32_t>(std::max(workHeight, 1)));
      glfwSetWindowAttrib(m_Window, GLFW_DECORATED, GLFW_TRUE);
      glfwSetWindowMonitor(m_Window, nullptr, currX, currY,
                           static_cast<int>(width), static_cast<int>(height), GLFW_DONT_CARE);
      glfwSetWindowPos(m_Window,
        monitorX + (workWidth - static_cast<int>(width)) / 2,
        monitorY + (workHeight - static_cast<int>(height)) / 2);
      break;
    }
  }

  int actualWidth = 0, actualHeight = 0;
  glfwGetWindowSize(m_Window, &actualWidth, &actualHeight);
  m_Data.Width = static_cast<uint32_t>(std::max(actualWidth, 1));
  m_Data.Height = static_cast<uint32_t>(std::max(actualHeight, 1));
  Camera::SetViewportSize(m_Data.Width, m_Data.Height);
  if (m_Data.API == GraphicsAPI::OpenGL)
    glViewport(0, 0, m_Data.Width, m_Data.Height);
  SetVSync(Settings::GetVSync());
}

void Window::RequestClose()
{
  m_isRunning = false;
  glfwSetWindowShouldClose(m_Window, GLFW_TRUE);
}

void Window::Maximize(bool maximize)  
{
  if(maximize) glfwMaximizeWindow(m_Window);
}

void Window::SetResizable(bool enable)
{
  glfwSetWindowAttrib(m_Window, GLFW_RESIZABLE, enable);
}

void Window::SetCursorVisible(bool enable)
{
  if (enable)
  {
    glfwSetInputMode(m_Window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    int32_t width, height;
    glfwGetWindowSize(m_Window, &width, &height);
    glfwSetCursorPos(m_Window, width / 2.0, height / 2.0);

  }
  else
  {
    glfwSetInputMode(m_Window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  }
}

void Window::OnEvent(Event& e)
{
	EventDispatcher dispatcher(e);
	dispatcher.Dispatch<WindowCloseEvent>(
		[](WindowCloseEvent& ev)
		{
			return Window::OnWindowClose(ev);
		});

	dispatcher.Dispatch<WindowResizeEvent>(
		[](WindowResizeEvent& ev)
		{
			return Window::OnWindowResize(ev);
		});

  if (!e.Handled)
  {
    if (m_Data.EventCallback) m_Data.EventCallback(e);
  }
}

bool Window::OnWindowClose(WindowCloseEvent& e)
{
	m_isRunning = false;
	return true;
}

bool Window::OnWindowResize(WindowResizeEvent& e)
{

	if (e.GetWidth() == 0 || e.GetHeight() == 0)
	{
		m_isMinimized = true;
		return false;
	}

	m_isMinimized = false;
  Camera::SetViewportSize(e.GetWidth(), e.GetHeight());
  if (m_Data.API == GraphicsAPI::OpenGL)
    glViewport(0, 0, e.GetWidth(), e.GetHeight());

	return false;
}

uint32_t Window::GetWidth(){ return m_Data.Width; }
uint32_t Window::GetHeight(){ return m_Data.Height; }
GLFWwindow* Window::GetWindowPtr(){ return m_Window; }
void* Window::GetNativeHandle()
{
#if defined(GABGL_ENABLE_DX12) && defined(_WIN32)
  return m_Window ? static_cast<void*>(glfwGetWin32Window(m_Window)) : nullptr;
#else
  return nullptr;
#endif
}
void Window::SetEventCallback(const Window::EventCallbackFn& callback) { m_Data.EventCallback = callback; }
bool Window::isClosed() { return m_WindowClosed; }
bool Window::IsRunning() { return m_isRunning; }
bool Window::IsMinimized() { return m_isMinimized; }


