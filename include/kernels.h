#ifndef KERNELS_H
#define KERNELS_H

#include <vector>
#include <cstdint> // For uint8_t, int16_t, int32_t
#include <omp.h>   // Add this line
#include "qat_unet_model.h"
#include "types.h"
#include "utils.h" // Assuming Tensor is defined here or in types.h
// Forward declarations for structs defined in types.h
struct LutLayer;
struct WeightsOnlyQuantLayer;
struct FloatLayer;

#ifdef __AVX512F__
#include <immintrin.h> // For AVX512 intrinsics

// Horizontal sum for AVX512 (16 int32_t elements)
int hsum_i32_16(const __m512i a);

// Bit-Slice GEMM kernel using AVX512 intrinsics
int32_t avx512_bit_slice_gemm_kernel(
    const uint8_t* input_packed_ptr,
    const uint8_t* weights_packed_ptr,
    const int32_t* precomputed_lut_ptr, // Changed to int32_t
    int k_dim
);

#else // Fallback to AVX2 if AVX512F is not defined
#ifdef __AVX2__
#include <immintrin.h> // For AVX2 intrinsics

// Horizontal sum for AVX2 (8 int32_t elements)
int hsum_i32_8(const __m256i a);

// Bit-Slice GEMM kernel using AVX2 intrinsics
int32_t avx2_bit_slice_gemm_kernel(
    const uint8_t* input_packed_ptr,
    const uint8_t* weights_packed_ptr,
    const int32_t* precomputed_lut_ptr, // Changed to int32_t
    int k_dim
);

#endif // __AVX2__
#endif // __AVX512F__


// --- 高层线性层前向传播函数声明 ---

// LUT 量化 MLP 前向传播
void lut_linear_forward(
    const LutLayer& layer,
    const std::vector<uint8_t>& input_packed_batched,
    std::vector<float>& output_f32_batched,
    int input_dim,
    int output_dim,
    int batch_size,
    const int32_t* precomputed_lut_ptr // Changed to int32_t
);

// 仅权重量化 MLP 前向传播
void weights_only_linear_forward(
    const WeightsOnlyQuantLayer& layer,
    const std::vector<int8_t>& input_i8_batched,
    std::vector<float>& output_f32_batched,
    int input_dim,
    int output_dim,
    int batch_size
);

// 标准浮点 MLP 前向传播
void standard_linear_forward(
    const FloatLayer& layer,
    const std::vector<float>& input_f32_batched,
    std::vector<float>& output_f32_batched,
    int input_dim,
    int output_dim,
    int batch_size
);


void load_lut(const std::string& path);
const int8_t* get_lut();

Tensor conv2d(const Tensor& input, const QATConv2dLayer& layer);

// Performs a standard linear (fully connected) layer operation
Tensor linear(const Tensor& input, const LinearLayer& layer);

// Applies Group Normalization
Tensor group_norm(const Tensor& input, const GroupNormLayer& layer);


// --- Composite Block Kernels ---

// Performs the forward pass for an Attention block
Tensor attention_block(const Tensor& input, const AttentionBlock& block);

// Performs the forward pass for a complete QATResidualBlock
Tensor residual_block(const Tensor& input, const Tensor& time_emb, const QATResidualBlock& block);
Tensor residual_block(
    const Tensor& input,
    const QATResidualBlock& block,
    const Tensor& time_emb,
    const Tensor& y // <-- 新增：类别标签张量
);
// --- Main Model Forward Pass ---

// Orchestrates the full forward pass of the QATUNet model
Tensor forward_qat_unet(
    const QATUNetModel& model,
    const Tensor& x,
    const Tensor& time
);
Tensor forward_qat_unet(
    const QATUNetModel& model,
    const Tensor& x,
    const Tensor& time,
    const Tensor& y // <-- 新增：类别标签张量
);
#endif // KERNELS_H
