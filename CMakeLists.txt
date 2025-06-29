# 设置CMake最低版本和项目名称
cmake_minimum_required(VERSION 3.14)
project(myQATModel CXX C)

# --- 查找并设置必需的包 ---
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED) # Add this line
# --- 设置C++标准 ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置可执行文件的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 启用针对x86 CPU的高性能内核
add_compile_definitions(GGML_BITNET_X86_TL2)
add_compile_definitions(GGML_BITNET_X86_AVX2)
#add_compile_definitions(GGML_BITNET_X86_AVX512)

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
        OpenMP::OpenMP_CXX # Add this line
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(myQATModel PRIVATE -mavx2)
elseif(MSVC)
#    target_compile_options(myQATModel PRIVATE /arch:AVX512)
    target_compile_options(myQATModel PRIVATE /arch:AVX2)
endif()

message(STATUS "Project configured successfully. This is the final version.")