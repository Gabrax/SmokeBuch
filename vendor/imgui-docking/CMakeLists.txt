cmake_minimum_required(VERSION 3.5)
project(imgui)

cmake_policy(SET CMP0069 NEW)
add_library(imgui STATIC)
target_sources(imgui PRIVATE 
"${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui_demo.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui_draw.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui_tables.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui_widgets.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends/imgui_impl_glfw.cpp"
"${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends/imgui_impl_opengl3.cpp"
)

if(GABGL_ENABLE_DX12 AND WIN32)
target_sources(imgui PRIVATE
"${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends/imgui_impl_dx12.cpp"
)
target_link_libraries(imgui PUBLIC d3d12)
endif()

if(GABGL_ENABLE_VULKAN)
target_sources(imgui PRIVATE
"${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends/imgui_impl_vulkan.cpp"
)
target_link_libraries(imgui PUBLIC Vulkan::Vulkan)
endif()

target_include_directories(imgui PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/imgui")


target_link_libraries(imgui PUBLIC glfw glad)
