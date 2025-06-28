# 设置CMake最低版本和项目名称
cmake_minimum_required(VERSION 3.14)
project(myQATModel CXX C)

# --- 查找并设置必需的包 ---
find_package(Threads REQUIRED)

# --- 设置C++标准 ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置可执行文件的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- 定义所有头文件搜索路径 ---
# 这是解决 "找不到文件" 错误的关键。
# 这些路径将对本项目中的所有目标（可执行文件、库）生效。
include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp/common"
        "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp/ggml/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp/ggml/src"
)

# --- BitNet/llama.cpp 选项 ---
# 关闭我们不需要的组件，以加快编译速度
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Build llama.cpp server" FORCE)
set(LLAMA_MPI OFF CACHE BOOL "" FORCE)
set(LLAMA_METAL OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENVINO OFF CACHE BOOL "" FORCE)

# 启用针对x86 CPU的高性能内核
add_compile_definitions(GGML_BITNET_X86_TL2)

# --- 添加子目录 ---
# 让CMake进入3rdparty/llama.cpp目录，并根据其内部的CMakeLists.txt
# 构建它的目标 (主要是 'llama' 和 'ggml' 库)
add_subdirectory(3rdparty/llama.cpp)

# --- 编译我们自己的内核 ---
add_library(bitnet_kernels STATIC
        src/ggml-bitnet-lut.cpp
        src/ggml-bitnet-mad.cpp
)

# --- 构建最终的主程序 ---
add_executable(myQATModel main.cpp)

# --- 链接库 ---
# 将我们的主程序链接到所有必需的库上
target_link_libraries(myQATModel PRIVATE
        llama           # llama.cpp 提供的核心库
        bitnet_kernels  # 我们自己编译的内核库
        Threads::Threads  # 线程库
)

# 解决MSVC编译器关于'getenv'等函数的警告
if(MSVC)
    target_compile_definitions(myQATModel PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_definitions(bitnet_kernels PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

message(STATUS "Project configured successfully. This is the final version.")