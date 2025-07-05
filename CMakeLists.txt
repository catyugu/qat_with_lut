# Set the minimum required version of CMake and project name
cmake_minimum_required(VERSION 3.14)
project(myQATModel CXX C)

# --- Find and set required packages ---
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED) # Add this line
# --- Set C++ standard ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable high-performance kernels for x86 CPUs
add_compile_definitions(GGML_BITNET_X86_TL2)
add_compile_definitions(GGML_BITNET_X86_AVX2)
#add_compile_definitions(GGML_BITNET_X86_AVX512)

# --- Build the main application ---
add_executable(myQATModel
        src/main.cpp
        src/types.cpp
        src/utils.cpp
        src/kernels.cpp
        src/profiler.cpp
)

target_include_directories(myQATModel PUBLIC include)
# --- Link libraries ---
# Link our main application against all required libraries
target_link_libraries(myQATModel PRIVATE
        Threads::Threads  # Threads library
        OpenMP::OpenMP_CXX # Add this line
)

# --- Build the new speed test executable ---
add_executable(speed_test
        src/speed_test.cpp
        src/types.cpp
        src/utils.cpp
        src/kernels.cpp
        src/profiler.cpp
)
add_executable(generate_image 
        src/generate_image.cpp
        src/qat_unet_model.cpp
        src/kernels.cpp 
        src/types.cpp
        src/utils.cpp
        src/profiler.cpp
        )
target_include_directories(speed_test PUBLIC include)
target_include_directories(generate_image PUBLIC include)

target_link_libraries(speed_test PRIVATE
        Threads::Threads
        OpenMP::OpenMP_CXX
)
target_link_libraries(generate_image PRIVATE
        Threads::Threads
        OpenMP::OpenMP_CXX
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(myQATModel PRIVATE -mavx2 -Ofast )
    target_compile_options(speed_test PRIVATE -mavx2 -Ofast)
    target_compile_options(generate_image PRIVATE -mavx2 -Ofast )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(myQATModel PRIVATE /arch:AVX2)
    target_compile_options(speed_test PRIVATE /arch:AVX2)
    target_compile_options(generate_image PRIVATE /arch:AVX2)
endif()