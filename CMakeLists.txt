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

# 启用针对x86 CPU的高性能内核
add_compile_definitions(GGML_BITNET_X86_TL2)
add_compile_definitions(GGML_BITNET_X86_AVX2)

# --- 编译我们自己的内核 ---
add_library(bitnet_kernels STATIC
        src/ggml-bitnet-lut.cpp
        src/ggml-bitnet-mad.cpp
)

# --- 构建最终的主程序 ---
add_executable(myQATModel
        src/main.cpp
        src/types.cpp
        src/utils.cpp
        src/kernels.cpp
)

target_include_directories(myQATModel PUBLIC include)
# --- 链接库 ---
# 将我们的主程序链接到所有必需的库上
target_link_libraries(myQATModel PRIVATE
        Threads::Threads  # 线程库
)

# 解决MSVC编译器关于'getenv'等函数的警告
if(MSVC)
    target_compile_definitions(myQATModel PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
# 添加MSVC编译器优化
if(MSVC)
    target_compile_options(myQATModel PRIVATE /arch:AVX2)
    target_compile_options(bitnet_kernels PRIVATE /arch:AVX2)
endif()

message(STATUS "Project configured successfully. This is the final version.")