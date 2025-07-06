#include "kernels.h" // Include the corresponding header

#include <immintrin.h>
#include "types.h"   // Include types.h for struct definitions
#include "utils.h"   // Include utils.h for functions like convert_int8_to_ternary_activation, pack_ternary_activations_5x3bit etc.
#include "profiler.h" // <-- 包含头文件
#include <iostream> // For debugging, remove in final build
#include <cmath>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <limits> // Required for std::numeric_limits


// --- SIMD 核心内核实现 ---

#ifdef __AVX512F__
// Horizontal sum of 16 int32_t elements in a __m512i vector
int hsum_i32_16(const __m512i a) {
    __m256i sum_low_256 = _mm512_extracti32x8_epi32(a, 0); // Extract lower 8 int32_t from 512-bit
    __m256i sum_high_256 = _mm512_extracti32x8_epi32(a, 1); // Extract higher 8 int32_t from 512-bit

    __m256i sum_avx2 = _mm256_add_epi32(sum_low_256, sum_high_256);

    // Now perform AVX2 horizontal sum on the single __m256i result
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum_avx2), _mm256_extractf128_si256(sum_avx2, 1));
    __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    __m128i sum64 = _mm_add_epi32(hi64, sum128);
    __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

// Bit-Slice GEMM kernel using AVX512 intrinsics (LUT-based)
int32_t avx512_bit_slice_gemm_kernel(
    const uint8_t* input_packed_ptr,
    const uint8_t* weights_packed_ptr,
    const int32_t* precomputed_lut_ptr, // <-- CORRECTED: Was int16_t
    int k_dim
) {
    __m512i accumulated_sum_vec_32 = _mm512_setzero_si512();

    const int num_packed_bytes = k_dim / 5;
    if (k_dim % 320 != 0) { // AVX512 loop processes 64 packed bytes = 320 elements
        std::cerr << "Error: k_dim must be a multiple of 320 for AVX512 bit-slice kernel (" << k_dim << ")." << std::endl;
        return 0;
    }

    for (int byte_idx_block = 0; byte_idx_block < num_packed_bytes; byte_idx_block += 64) {
        // Prefetch data for the *next* block of inputs and weights
        _mm_prefetch(reinterpret_cast<const char*>(input_packed_ptr + byte_idx_block + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(weights_packed_ptr + byte_idx_block + 64), _MM_HINT_T0);

        __m512i packed_acts_bytes = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(input_packed_ptr + byte_idx_block));
        __m512i packed_weights_bytes = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(weights_packed_ptr + byte_idx_block));

        alignas(64) uint8_t temp_packed_acts_raw[64];
        alignas(64) uint8_t temp_packed_weights_raw[64];
        alignas(64) int16_t block_sum_of_products[64];

        _mm512_storeu_si512(reinterpret_cast<__m512i*>(temp_packed_acts_raw), packed_acts_bytes);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(temp_packed_weights_raw), packed_weights_bytes);

        // Manually unroll the loop by a factor of 8 (64 / 8 = 8 iterations)
        // Original loop was for (int i = 0; i < 64; ++i)
        for (int i = 0; i < 64; i += 8) {
            uint8_t packed_act_byte_0 = temp_packed_acts_raw[i + 0];
            uint8_t packed_weight_byte_0 = temp_packed_weights_raw[i + 0];
            block_sum_of_products[i + 0] = precomputed_lut_ptr[(static_cast<uint32_t>(packed_act_byte_0) << 8) | packed_weight_byte_0];

            uint8_t packed_act_byte_1 = temp_packed_acts_raw[i + 1];
            uint8_t packed_weight_byte_1 = temp_packed_weights_raw[i + 1];
            block_sum_of_products[i + 1] = precomputed_lut_ptr[(static_cast<uint32_t>(packed_act_byte_1) << 8) | packed_weight_byte_1];

            uint8_t packed_act_byte_2 = temp_packed_acts_raw[i + 2];
            uint8_t packed_weight_byte_2 = temp_packed_weights_raw[i + 2];
            block_sum_of_products[i + 2] = precomputed_lut_ptr[(static_cast<uint32_t>(packed_act_byte_2) << 8) | packed_weight_byte_2];

            uint8_t packed_act_byte_3 = temp_packed_acts_raw[i + 3];
            uint8_t packed_weight_byte_3 = temp_packed_weights_raw[i + 3];
            block_sum_of_products[i + 3] = precomputed_lut_ptr[(static_cast<uint32_t>(packed_act_byte_3) << 8) | packed_weight_byte_3];

            uint8_t packed_act_byte_4 = temp_packed_acts_raw[i + 4];
            uint8_t packed_weight_byte_4 = temp_packed_weights_raw[i + 4];
            block_sum_of_products[i + 4] = precomputed_lut_ptr[(static_cast<uint32_t>(packed_act_byte_4) << 8) | packed_weight_byte_4];

            uint8_t packed_act_byte_5 = temp_packed_acts_raw[i + 5];
            uint8_t packed_weight_byte_5 = temp_packed_weights_raw[i + 5];
            block_sum_of_products[i + 5] = precomputed_lut_ptr[(static_cast<uint32_t>(packed_act_byte_5) << 8) | packed_weight_byte_5];

            uint8_t packed_act_byte_6 = temp_packed_acts_raw[i + 6];
            uint8_t packed_weight_byte_6 = temp_packed_weights_raw[i + 6];
            block_sum_of_products[i + 6] = precomputed_lut_ptr[(static_cast<uint32_t>(packed_act_byte_6) << 8) | packed_weight_byte_6];

            uint8_t packed_act_byte_7 = temp_packed_acts_raw[i + 7];
            uint8_t packed_weight_byte_7 = temp_packed_weights_raw[i + 7];
            block_sum_of_products[i + 7] = precomputed_lut_ptr[(static_cast<uint32_t>(packed_act_byte_7) << 8) | packed_weight_byte_7];
        }

        // Accumulate the 64 int16_t results using AVX512 intrinsics
        __m512i sum_of_products_vec0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(block_sum_of_products));
        __m512i sum_of_products_vec1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(block_sum_of_products + 32));

        // Convert 16-bit sums to 32-bit sums for accumulation.
        __m512i sum_0_lo_32 = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(sum_of_products_vec0, 0)); // Lower 16 int16_t -> 16 int32_t
        __m512i sum_0_hi_32 = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(sum_of_products_vec0, 1)); // Higher 16 int16_t -> 16 int32_t
        accumulated_sum_vec_32 = _mm512_add_epi32(accumulated_sum_vec_32, sum_0_lo_32);
        accumulated_sum_vec_32 = _mm512_add_epi32(accumulated_sum_vec_32, sum_0_hi_32);

        __m512i sum_1_lo_32 = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(sum_of_products_vec1, 0));
        __m512i sum_1_hi_32 = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(sum_of_products_vec1, 1));
        accumulated_sum_vec_32 = _mm512_add_epi32(accumulated_sum_vec_32, sum_1_lo_32);
        accumulated_sum_vec_32 = _mm512_add_epi32(accumulated_sum_vec_32, sum_1_hi_32);
    }
    return hsum_i32_16(accumulated_sum_vec_32);
}

#else // Fallback to AVX2 if __AVX512F__ is not defined
// Horizontal sum of 8 int32_t elements in a __m256i vector (adapted from ggml-bitnet-mad.cpp)
int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}


int32_t avx2_bit_slice_gemm_kernel(
    const uint8_t* input_packed_ptr,
    const uint8_t* weights_packed_ptr,
    const int32_t* precomputed_lut_ptr, // Changed to int32_t
    int k_dim
) {
    __m256i accumulated_sum_vec = _mm256_setzero_si256();
    const int num_packed_bytes = k_dim / 5;

    for (int j = 0; j < num_packed_bytes; j += 32) {
        __m256i packed_acts_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_packed_ptr + j));
        __m256i packed_weights_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights_packed_ptr + j));

        // Unpack bytes to 32-bit indices for gather
        __m128i p_acts_lo = _mm256_castsi256_si128(packed_acts_vec);
        __m128i p_acts_hi = _mm256_extracti128_si256(packed_acts_vec, 1);
        __m256i acts_idx1 = _mm256_cvtepu8_epi32(p_acts_lo);
        __m256i acts_idx2 = _mm256_cvtepu8_epi32(_mm_srli_si128(p_acts_lo, 8));
        __m256i acts_idx3 = _mm256_cvtepu8_epi32(p_acts_hi);
        __m256i acts_idx4 = _mm256_cvtepu8_epi32(_mm_srli_si128(p_acts_hi, 8));

        __m128i p_weights_lo = _mm256_castsi256_si128(packed_weights_vec);
        __m128i p_weights_hi = _mm256_extracti128_si256(packed_weights_vec, 1);
        __m256i weights_idx1 = _mm256_cvtepu8_epi32(p_weights_lo);
        __m256i weights_idx2 = _mm256_cvtepu8_epi32(_mm_srli_si128(p_weights_lo, 8));
        __m256i weights_idx3 = _mm256_cvtepu8_epi32(p_weights_hi);
        __m256i weights_idx4 = _mm256_cvtepu8_epi32(_mm_srli_si128(p_weights_hi, 8));

        // Combine indices: final_idx = (act_idx << 8) | weight_idx
        const __m256i shift_8 = _mm256_set1_epi32(256);
        __m256i final_idx1 = _mm256_or_si256(_mm256_mullo_epi32(acts_idx1, shift_8), weights_idx1);
        __m256i final_idx2 = _mm256_or_si256(_mm256_mullo_epi32(acts_idx2, shift_8), weights_idx2);
        __m256i final_idx3 = _mm256_or_si256(_mm256_mullo_epi32(acts_idx3, shift_8), weights_idx3);
        __m256i final_idx4 = _mm256_or_si256(_mm256_mullo_epi32(acts_idx4, shift_8), weights_idx4);

        // Gather 32-bit values from the LUT using a scale of 4 bytes
        __m256i gathered1 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(precomputed_lut_ptr), final_idx1, 4);
        __m256i gathered2 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(precomputed_lut_ptr), final_idx2, 4);
        __m256i gathered3 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(precomputed_lut_ptr), final_idx3, 4);
        __m256i gathered4 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(precomputed_lut_ptr), final_idx4, 4);

        // Accumulate results
        __m256i sum12 = _mm256_add_epi32(gathered1, gathered2);
        __m256i sum34 = _mm256_add_epi32(gathered3, gathered4);
        accumulated_sum_vec = _mm256_add_epi32(accumulated_sum_vec, _mm256_add_epi32(sum12, sum34));
    }
    return hsum_i32_8(accumulated_sum_vec);
}

#endif // __AVX512F__


// --- 高层线性层前向传播函数实现 ---

void lut_linear_forward(
    const LutLayer& layer,
    const std::vector<uint8_t>& input_packed_batched,
    std::vector<float>& output_f32_batched,
    int input_dim,
    int output_dim,
    int batch_size,
    const int32_t* precomputed_lut_ptr // Changed to int32_t
) {
    output_f32_batched.resize(batch_size * output_dim);
    const uint8_t* weights_packed_ptr = layer.packed_weights.data();
    const int packed_input_bytes_per_sample = (input_dim + 4) / 5;
    for (int b = 0; b < batch_size; ++b) {
        const uint8_t* current_batch_input_packed_ptr = input_packed_batched.data() + b * packed_input_bytes_per_sample;
        for (int i = 0; i < output_dim; ++i) {
            const uint8_t* current_neuron_weights_packed_ptr = weights_packed_ptr + i * packed_input_bytes_per_sample;

            int32_t sum = avx2_bit_slice_gemm_kernel(
                current_batch_input_packed_ptr,
                current_neuron_weights_packed_ptr,
                precomputed_lut_ptr,
                input_dim
            );
            output_f32_batched[b * output_dim + i] = (static_cast<float>(sum) / layer.activation_scale) + layer.bias[i];
        }
    }
}

void weights_only_linear_forward(
    const WeightsOnlyQuantLayer& layer,
    const std::vector<int8_t>& input_i8_batched,
    std::vector<float>& output_f32_batched,
    int input_dim,
    int output_dim,
    int batch_size
) {
    output_f32_batched.resize(batch_size * output_dim);
    const int8_t* weights_ptr = layer.weights.data();
    for (int b = 0; b < batch_size; ++b) {
        const int8_t* current_batch_input_ptr = input_i8_batched.data() + b * input_dim;
        for (int i = 0; i < output_dim; ++i) {
            const int8_t* current_neuron_weights_ptr = weights_ptr + i * input_dim;

            __m256i accumulated_sum_vec_32 = _mm256_setzero_si256();

            // **CORRECTED KERNEL LOGIC**
            for (int j = 0; j < input_dim; j += 32) {
                // Load 16 bytes (128 bits) at a time
                __m128i input_i8_1 = _mm_loadu_si128((__m128i const*)(current_batch_input_ptr + j));
                __m128i weights_i8_1 = _mm_loadu_si128((__m128i const*)(current_neuron_weights_ptr + j));
                __m128i input_i8_2 = _mm_loadu_si128((__m128i const*)(current_batch_input_ptr + j + 16));
                __m128i weights_i8_2 = _mm_loadu_si128((__m128i const*)(current_neuron_weights_ptr + j + 16));

                // Convert int8 to int16
                __m256i input_i16_1 = _mm256_cvtepi8_epi16(input_i8_1);
                __m256i weights_i16_1 = _mm256_cvtepi8_epi16(weights_i8_1);
                __m256i input_i16_2 = _mm256_cvtepi8_epi16(input_i8_2);
                __m256i weights_i16_2 = _mm256_cvtepi8_epi16(weights_i8_2);

                // Multiply and horizontally add adjacent pairs, results in int32
                __m256i mad1 = _mm256_madd_epi16(input_i16_1, weights_i16_1);
                __m256i mad2 = _mm256_madd_epi16(input_i16_2, weights_i16_2);

                // Accumulate the int32 results
                accumulated_sum_vec_32 = _mm256_add_epi32(accumulated_sum_vec_32, _mm256_add_epi32(mad1, mad2));
            }
            int32_t sum = hsum_i32_8(accumulated_sum_vec_32);
            output_f32_batched[b * output_dim + i] = (static_cast<float>(sum) / layer.activation_scale) + layer.bias[i];
        }
    }
}

void standard_linear_forward(
    const FloatLayer& layer,
    const std::vector<float>& input_f32_batched,
    std::vector<float>& output_f32_batched,
    int input_dim,
    int output_dim,
    int batch_size
) {
    output_f32_batched.resize(batch_size * output_dim);
    for (int b = 0; b < batch_size; ++b) {
        const float* current_batch_input_ptr = input_f32_batched.data() + b * input_dim;
        for (int i = 0; i < output_dim; ++i) {
            float sum = 0.0f;
            const float* current_neuron_weights_ptr = layer.weights.data() + i * input_dim;
            for (int j = 0; j < input_dim; ++j) {
                sum += current_neuron_weights_ptr[j] * current_batch_input_ptr[j];
            }
            output_f32_batched[b * output_dim + i] = sum + layer.bias[i];
        }
    }
}

void im2col(float* data_col, const float* data_im,
            int B, int C_in, int H_in, int W_in,
            int KH, int KW, int pad, int stride) {

    // 1. 根据卷积参数计算输出特征图的尺寸
    const int H_out = (H_in + 2 * pad - KH) / stride + 1;
    const int W_out = (W_in + 2 * pad - KW) / stride + 1;

    // 2. 预计算输出矩阵的维度，用于索引计算
    // 输出矩阵的行数 K = C_in * KH * KW (一个 patch 拉平后的长度)
    const int K = C_in * KH * KW;
    // 输出矩阵的列数 N = B * H_out * W_out (所有 patch 的总数)
    const int N = B * H_out * W_out;

    // 3. 核心循环：填充输出矩阵 data_col
    // 外层循环遍历输出矩阵的每一行 (由 c, kh, kw 唯一确定)
    for (int k = 0; k < K; ++k) {
        // 从一维索引 k 中分解出通道、卷积核高和宽的索引
        const int kw = k % KW;
        const int kh = (k / KW) % KH;
        const int c = k / (KW * KH);

        // 内层循环遍历输出矩阵的每一列 (由 b, h_out, w_out 唯一确定)
        for (int n = 0; n < N; ++n) {
            // 从一维索引 n 中分解出批次、输出高和宽的索引
            const int w_out = n % W_out;
            const int h_out = (n / W_out) % H_out;
            const int b = n / (W_out * H_out);

            // 4. 计算当前 patch 元素在原始输入图像中的坐标
            const int h_in = h_out * stride - pad + kh;
            const int w_in = w_out * stride - pad + kw;

            // 5. 计算在输入和输出一维数组中的索引
            // 使用 long long 防止大尺寸时整数溢出
            const long long col_idx = (long long)k * N + n;
            const long long im_idx = (long long)b * C_in * H_in * W_in +
                                     (long long)c * H_in * W_in +
                                     (long long)h_in * W_in + w_in;

            // 6. 边界检查与数据填充
            // 检查计算出的输入坐标是否在原始图像的有效范围内 (不包括填充)
            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                // 如果是有效位置，从输入图像复制数据
                data_col[col_idx] = data_im[im_idx];
            } else {
                // 如果是在填充区域 (padding)，则用 0 填充
                data_col[col_idx] = 0.0f;
            }
        }
    }
}
// --- Activation Functions ---
Tensor silu(const Tensor& input) {
    Tensor output = input; // Copy shape and data
    for (size_t i = 0; i < output.data.size(); ++i) {
        output.data[i] = input.data[i] / (1.0f + std::exp(-input.data[i]));
    }
    return output;
}

Tensor softmax(const Tensor& input) {
    Tensor output = input;
    size_t last_dim = input.shape.back();
    size_t outer_dims = input.data.size() / last_dim;

    for (size_t i = 0; i < outer_dims; ++i) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < last_dim; ++j) {
            if (input.data[i * last_dim + j] > max_val) {
                max_val = input.data[i * last_dim + j];
            }
        }

        float sum_exp = 0.0f;
        for (size_t j = 0; j < last_dim; ++j) {
            float val = std::exp(input.data[i * last_dim + j] - max_val);
            output.data[i * last_dim + j] = val;
            sum_exp += val;
        }

        for (size_t j = 0; j < last_dim; ++j) {
            output.data[i * last_dim + j] /= sum_exp;
        }
    }
    return output;
}


// --- Layer Kernels ---

std::vector<float> unpack_ternary_weights(const std::vector<uint8_t>& packed_weights, size_t num_weights) {
    std::vector<float> unpacked(num_weights);
    size_t unpacked_idx = 0;

    for (uint8_t byte : packed_weights) {
        for (int i = 0; i < 4; ++i) {
            if (unpacked_idx >= num_weights) break;

            // 从低位到高位依次提取 2-bit
            uint8_t two_bits = (byte >> (i * 2)) & 0x03;

            switch (two_bits) {
                case 0b00: unpacked[unpacked_idx++] = 0.0f; break;
                case 0b01: unpacked[unpacked_idx++] = 1.0f; break;
                case 0b10: unpacked[unpacked_idx++] = -1.0f; break;
                // case 0b11: // 11 是保留位，可以报错或设为默认值
                default:   unpacked[unpacked_idx++] = 0.0f; break; // 默认为0
            }
        }
    }
    return unpacked;
}


// 保留之前实现的 AVX2 优化的三值 GEMM 函数
void ternary_gemm_avx2(int M, int N, int K, const float* A, const float* B, float* C) {
    // ... (此函数无需修改，与上一版完全相同) ...
    constexpr int AVX_FLOAT_COUNT = 8;
    for (int i = 0; i < M; ++i) {
        int j = 0;
        for (; j + AVX_FLOAT_COUNT <= N; j += AVX_FLOAT_COUNT) {
            __m256 acc = _mm256_setzero_ps();
            for (int k = 0; k < K; ++k) {
                const float weight_val = A[i * K + k];
                if (weight_val == 0.0f) continue;
                __m256 b_vals = _mm256_loadu_ps(&B[k * N + j]);
                if (weight_val == 1.0f) {
                    acc = _mm256_add_ps(acc, b_vals);
                } else {
                    acc = _mm256_sub_ps(acc, b_vals);
                }
            }
            _mm256_storeu_ps(&C[i * N + j], acc);
        }
        for (; j < N; ++j) {
            float acc_scalar = 0.0f;
            for (int k = 0; k < K; ++k) {
                const float weight_val = A[i * K + k];
                if (weight_val == 0.0f) continue;
                const float b_val = B[k * N + j];
                if (weight_val == 1.0f) acc_scalar += b_val;
                else acc_scalar -= b_val;
            }
            C[i * N + j] = acc_scalar;
        }
    }
}


/**
 * @brief 最终优化的卷积函数，接口与您的结构体完全匹配
 */
Tensor conv2d(const Tensor& input, const QATConv2dLayer& layer) {
    // 基本参数检查
    if (layer.groups != 1) {
        throw std::runtime_error("Optimized conv2d currently only supports groups=1");
    }

    // 使用 const auto& 是一种好习惯
    const auto& input_shape = input.shape;
    const int B = input_shape[0];
    const int C_in = input_shape[1];
    const int H_in = input_shape[2];
    const int W_in = input_shape[3];

    // 从 layer 结构体中获取参数
    const int C_out = layer.out_channels;
    const int KH = layer.kernel_size_h;
    const int KW = layer.kernel_size_w;
    const int stride = layer.stride_h;
    const int padding = layer.pad_h;

    // 1. 解包权重
    const size_t num_weights = C_out * C_in * KH * KW;
    std::vector<float> unpacked_weights = unpack_ternary_weights(layer.packed_weights, num_weights);

    // 2. 计算输出尺寸并创建输出 Tensor
    const int H_out = (H_in + 2 * padding - KH) / stride + 1;
    const int W_out = (W_in + 2 * padding - KW) / stride + 1;

    // --- FIX START ---
    // 修复 narrowing conversion 警告和类型不匹配问题
    // 将 int 显式转换为 size_t (或 long unsigned int) 来匹配 Tensor.shape 的类型
    Tensor output({(size_t)B, (size_t)C_out, (size_t)H_out, (size_t)W_out});
    // --- FIX END ---


    // 3. im2col 转换
    Tensor col_buffer;
    // --- FIX START ---
    // 修复 operator= 错误
    // 将 col_shape 的类型从 std::vector<int> 改为和 Tensor.shape 匹配的类型
    const std::vector<size_t> col_shape = {(size_t)(C_in * KH * KW), (size_t)(B * H_out * W_out)};
    // --- FIX END ---
    col_buffer.shape = col_shape;
    col_buffer.data.resize(col_shape[0] * col_shape[1]);
    im2col(col_buffer.data.data(), input.data.data(), B, C_in, H_in, W_in, KH, KW, padding, stride);

    // 4. 调用 AVX2 GEMM 进行核心计算
    const int M = C_out;
    const int N = B * H_out * W_out;
    const int K = C_in * KH * KW;
    ternary_gemm_avx2(M, N, K, unpacked_weights.data(), col_buffer.data.data(), output.data.data());

    // 5. 添加偏置
    if (!layer.bias.empty()) {
        for (int b = 0; b < B; ++b) {
            for (int c = 0; c < C_out; ++c) {
                float* out_ptr_base = &output.data[(b * C_out + c) * (H_out * W_out)];
                __m256 bias_vec = _mm256_set1_ps(layer.bias[c]);
                int i = 0;
                for (; i + 8 <= H_out * W_out; i += 8) {
                    float* out_ptr = out_ptr_base + i;
                    _mm256_storeu_ps(out_ptr, _mm256_add_ps(_mm256_loadu_ps(out_ptr), bias_vec));
                }
                for (; i < H_out * W_out; ++i) {
                    out_ptr_base[i] += layer.bias[c];
                }
            }
        }
    }

    return output;
}
Tensor linear(const Tensor& input, const LinearLayer& layer) {
    PROFILE_SCOPE("linear"); // <-- 添加宏
    const auto& in_shape = input.shape;
    size_t B = (in_shape.size() > 1) ? in_shape[0] : 1;
    size_t in_features = in_shape.back();
    size_t out_features = layer.out_features;
    
    Tensor output({B, out_features});

    for (size_t b = 0; b < B; ++b) {
        for (size_t out_f = 0; out_f < out_features; ++out_f) {
            float acc = 0.0f;
            for (size_t in_f = 0; in_f < in_features; ++in_f) {
                float in_val = input.data[b * in_features + in_f];
                float w_val = layer.weights[out_f * in_features + in_f];
                acc += in_val * w_val;
            }
            output.data[b * out_features + out_f] = acc + layer.bias[out_f];
        }
    }
    return output;
}

Tensor group_norm(const Tensor& input, const GroupNormLayer& layer) {
    PROFILE_SCOPE("group_norm");
    const auto& shape = input.shape;
    size_t B = shape[0], C = shape[1], H = shape[2], W = shape[3];
    size_t G = layer.num_groups, C_per_group = C / G;

    Tensor output = input;

    for (size_t b = 0; b < B; ++b) {
        for (size_t g = 0; g < G; ++g) {
            float sum = 0.0f;
            for (size_t c = 0; c < C_per_group; ++c) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t w = 0; w < W; ++w) {
                        sum += input.at({b, g * C_per_group + c, h, w});
                    }
                }
            }
            float mean = sum / (C_per_group * H * W);

            float sum_sq_diff = 0.0f;
            for (size_t c = 0; c < C_per_group; ++c) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t w = 0; w < W; ++w) {
                        float diff = input.at({b, g * C_per_group + c, h, w}) - mean;
                        sum_sq_diff += diff * diff;
                    }
                }
            }
            float variance = sum_sq_diff / (C_per_group * H * W);
            float std_dev = std::sqrt(variance + layer.eps);

            for (size_t c = 0; c < C_per_group; ++c) {
                size_t channel_idx = g * C_per_group + c;
                for (size_t h = 0; h < H; ++h) {
                    for (size_t w = 0; w < W; ++w) {
                        float normalized = (input.at({b, channel_idx, h, w}) - mean) / std_dev;
                        output.at({b, channel_idx, h, w}) = normalized * layer.weight[channel_idx] + layer.bias[channel_idx];
                    }
                }
            }
        }
    }
    return output;
}

Tensor attention_block(const Tensor& input, const AttentionBlock& block) {
    PROFILE_SCOPE("attention_block");
    const auto& shape = input.shape;
    size_t B = shape[0], C = shape[1], H = shape[2], W = shape[3];

    Tensor h = group_norm(input, *block.norm);
    Tensor qkv = conv2d(h, *block.to_qkv);

    Tensor q = qkv.slice(1, 0, C);
    Tensor k = qkv.slice(1, C, C);
    Tensor v = qkv.slice(1, C * 2, C);

    q.reshape({B, C, H * W});
    k.reshape({B, C, H * W});
    v.reshape({B, C, H * W});

    q = q.permute({0, 2, 1}); 
    k = k.permute({0, 2, 1}); 
    v = v.permute({0, 2, 1}); 
    
    Tensor scores = q.matmul(k.transpose(1, 2)); 
    scores = scores.mul_scalar(1.0f / std::sqrt(static_cast<float>(C)));
    scores = softmax(scores);

    Tensor out = scores.matmul(v);
    out.reshape({B, H, W, C});
    out = out.permute({0, 3, 1, 2});

    out = conv2d(out, *block.to_out);
    return input.add(out);
}

Tensor residual_block(const Tensor& input, const Tensor& time_emb, const QATResidualBlock& block) {
    PROFILE_SCOPE("residual_block");
    Tensor h = group_norm(input, *block.norm_1);
    h = silu(h);
    h = conv2d(h, *block.conv_1);

    if (block.time_bias) {
        Tensor time_bias = linear(silu(time_emb), *block.time_bias);
        time_bias.reshape({1, (size_t)block.conv_1->out_channels, 1, 1});
        h = h.add(time_bias);
    }

    Tensor h2 = group_norm(h, *block.norm_2);
    h2 = silu(h2);
    h2 = conv2d(h2, *block.conv_2);

    Tensor residual_conn = input;
    if (block.residual_connection) {
        residual_conn = conv2d(input, *block.residual_connection);
    }

    Tensor out = h2.add(residual_conn);

    if (block.attention) {
        out = attention_block(out, *block.attention);
    }
    
    return out;
}

Tensor positional_embedding(const Tensor& time, const PositionalEmbedding& layer) {
    size_t B = time.shape[0];
    size_t dim = layer.dim;
    size_t half_dim = dim / 2;
    
    Tensor output({B, dim});

    float log_10000 = std::log(10000.0f);

    for(size_t b = 0; b < B; ++b) {
        for(size_t i = 0; i < half_dim; ++i) {
            float emb = std::exp(static_cast<float>(i) * -log_10000 / (half_dim - 1.0f));
            float arg = time.data[b] * layer.time_emb_scale * emb;
            output.at({b, i}) = std::sin(arg);
            output.at({b, i + half_dim}) = std::cos(arg);
        }
    }
    return output;
}

// --- Main Model Forward Pass ---

Tensor forward_qat_unet(const QATUNetModel& model, const Tensor& x, const Tensor& time) {
    PROFILE_SCOPE("forward_qat_unet"); 
    Tensor time_emb = positional_embedding(time, *model.time_mlp_pos_emb);
    time_emb = linear(time_emb, *model.time_mlp_linear1);
    time_emb = silu(time_emb);
    time_emb = linear(time_emb, *model.time_mlp_linear2);

    Tensor h = conv2d(x, *model.init_conv);
    
    std::vector<Tensor> skips;
    skips.push_back(h);

    for (const auto& layer_ptr : model.downs) {
        if (auto* res_block = dynamic_cast<QATResidualBlock*>(layer_ptr.get())) {
            h = residual_block(h, time_emb, *res_block);
        } else if (auto* downsample_block = dynamic_cast<DownsampleLayer*>(layer_ptr.get())) {
            h = conv2d(h, *downsample_block->conv);
        }
        skips.push_back(h);
    }

    h = residual_block(h, time_emb, *model.middle_block1);
    h = residual_block(h, time_emb, *model.middle_block2);

    for (const auto& layer_ptr : model.ups) {
        if (auto* res_block = dynamic_cast<QATResidualBlock*>(layer_ptr.get())) {
            Tensor skip = skips.back();
            skips.pop_back();
            h = h.cat(skip, 1); 
            h = residual_block(h, time_emb, *res_block);
        } else if (auto* upsample_block = dynamic_cast<UpsampleLayer*>(layer_ptr.get())) {
            h = h.upsample(2);
            h = conv2d(h, *upsample_block->conv);
        }
    }
    
    h = group_norm(h, *model.final_norm);
    h = silu(h);
    h = conv2d(h, *model.final_conv);

    return h;
}
// --- Main UNet Forward Pass ---
// --- Residual Block ---
Tensor residual_block(const Tensor& input, const Tensor& time_emb, const QATResidualBlock& block, const Tensor& y) {
    Tensor h = input;

    // First normalization and convolution
    h = group_norm(h, *block.norm_1);
    h = silu(h);
    h = conv2d(h, *block.conv_1);

    // Add time and class embeddings
    if (block.time_bias) {
        Tensor time_bias = linear(time_emb, *block.time_bias);
        h = h.add(time_bias.view({h.shape[0], h.shape[1], 1, 1})); // 修正：使用view
    }

    if (block.class_bias) {
        int class_idx = static_cast<int>(y.data[0]);
        int embedding_dim = block.class_bias->embedding_dim;
        Tensor class_bias_tensor({1, (size_t)embedding_dim}); 
        
        const auto& weights = block.class_bias->weight;
        std::copy(weights.begin() + class_idx * embedding_dim,
                  weights.begin() + (class_idx + 1) * embedding_dim,
                  class_bias_tensor.data.begin());

        h = h.add(class_bias_tensor.view({h.shape[0], h.shape[1], 1, 1})); // 修正：使用view
    }

    // Second normalization and convolution
    h = group_norm(h, *block.norm_2);
    h = silu(h);
    h = conv2d(h, *block.conv_2);

    // Add residual connection
    if (block.residual_connection) {
        return h.add(conv2d(input, *block.residual_connection));
    }
    return h.add(input);
}


Tensor forward_qat_unet(const QATUNetModel& model, const Tensor& x, const Tensor& time, const Tensor& y) {
    PROFILE_SCOPE("forward_qat_unet"); 
    Tensor time_emb = positional_embedding(time, *model.time_mlp_pos_emb);
    time_emb = linear(time_emb, *model.time_mlp_linear1);
    time_emb = silu(time_emb);
    time_emb = linear(time_emb, *model.time_mlp_linear2);

    Tensor h = conv2d(x, *model.init_conv);
    
    std::vector<Tensor> skips;
    skips.push_back(h);

    for (const auto& layer_ptr : model.downs) {
        if (auto* res_block = dynamic_cast<QATResidualBlock*>(layer_ptr.get())) {
            h = residual_block(h, time_emb, *res_block,y);
        } else if (auto* downsample_block = dynamic_cast<DownsampleLayer*>(layer_ptr.get())) {
            h = conv2d(h, *downsample_block->conv);
        }
        skips.push_back(h);
    }

    h = residual_block(h, time_emb, *model.middle_block1, y);
    h = residual_block(h, time_emb, *model.middle_block2, y);

    for (const auto& layer_ptr : model.ups) {
        if (auto* res_block = dynamic_cast<QATResidualBlock*>(layer_ptr.get())) {
            Tensor skip = skips.back();
            skips.pop_back();
            h = h.cat(skip, 1); 
            h = residual_block(h, time_emb, *res_block,y);
        } else if (auto* upsample_block = dynamic_cast<UpsampleLayer*>(layer_ptr.get())) {
            h = h.upsample(2);
            h = conv2d(h, *upsample_block->conv);
        }
    }
    
    h = group_norm(h, *model.final_norm);
    h = silu(h);
    h = conv2d(h, *model.final_conv);

    return h;
}

