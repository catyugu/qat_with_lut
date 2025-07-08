# 设置CMake最低版本和项目名称
cmake_minimum_required(VERSION 3.14)
project(myQATModel CXX C)

# --- 查找并设置必要的包 ---
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# --- 设置C++标准 ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置可执行文件的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#-----------------------------------------------------------------------------
# 1. 为所有目标共享的通用代码创建一个静态库 ⚙️
#-----------------------------------------------------------------------------
add_library(qat_core STATIC
    src/types.cpp
    src/utils.cpp
    src/kernels.cpp
    src/profiler.cpp
)

# 为核心库设置PUBLIC属性，这样链接到它的目标会自动继承这些设置
target_include_directories(qat_core PUBLIC 
    include
)

target_compile_definitions(qat_core PUBLIC
    GGML_BITNET_X86_TL2
    GGML_BITNET_X86_AVX2
    #GGML_BITNET_X86_AVX512
)

target_link_libraries(qat_core PUBLIC
    Threads::Threads
    OpenMP::OpenMP_CXX
)

# 将编译器特定优化选项也应用到核心库
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(qat_core PUBLIC -mavx2 -Ofast)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(qat_core PUBLIC /arch:AVX2)
endif()


#-----------------------------------------------------------------------------
# 2. 构建可执行文件，并链接到核心库 🔗
#-----------------------------------------------------------------------------

# --- 构建主应用程序 ---
add_executable(myQATModel
    src/main.cpp
)
target_link_libraries(myQATModel PRIVATE qat_core)

# --- 构建速度测试程序 ---
add_executable(speed_test
    src/speed_test.cpp
)
target_link_libraries(speed_test PRIVATE qat_core)

# --- 构建图像生成程序 ---
add_executable(generate_image 
    src/generate_image.cpp
    src/qat_unet_model.cpp  # 这个源文件是此目标特有的
)
target_link_libraries(generate_image PRIVATE qat_core)

add_executable(test_attention 
    src/test_attention.cpp
)
target_link_libraries(test_attention PRIVATE qat_core)
