#include "kernels.h" // Include the corresponding header

#include <immintrin.h>

#include "types.h"   // Include types.h for struct definitions
#include "utils.h"   // Include utils.h for functions like convert_int8_to_ternary_activation, pack_ternary_activations_5x3bit etc.

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

void im2col(const std::vector<float>& input, int channels, int height, int width,
            int kernel_h, int kernel_w, int stride_h, int stride_w,
            int pad_h, int pad_w, std::vector<float>& col) {

    int output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int col_rows = channels * kernel_h * kernel_w;
    int col_cols = output_h * output_w;

    col.assign(col_rows * col_cols, 0.0f);

    for (int c = 0; c < channels; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int input_row = -pad_h + kh;
                for (int oh = 0; oh < output_h; ++oh) {
                    int input_col = -pad_w + kw;
                    for (int ow = 0; ow < output_w; ++ow) {
                        int col_index = (c * kernel_h * kernel_w + kh * kernel_w + kw) * (output_h * output_w) + oh * output_w + ow;
                        int input_h_idx = input_row + oh * stride_h;
                        int input_w_idx = input_col + ow * stride_w;

                        if (input_h_idx >= 0 && input_h_idx < height && input_w_idx >= 0 && input_w_idx < width) {
                            col[col_index] = input[c * width * height + input_h_idx * width + input_w_idx];
                        } else {
                            col[col_index] = 0; // Padding
                        }
                    }
                }
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

Tensor conv2d(const Tensor& input, const QATConv2dLayer& layer) {
    const auto& in_shape = input.shape;
    size_t B = in_shape[0], C_in = in_shape[1], H_in = in_shape[2], W_in = in_shape[3];
    size_t C_out = layer.out_channels, KH = layer.kernel_size_h, KW = layer.kernel_size_w;
    size_t SH = layer.stride_h, SW = layer.stride_w, PH = layer.pad_h, PW = layer.pad_w;
    size_t G = layer.groups;

    size_t H_out = (H_in + 2 * PH - KH) / SH + 1;
    size_t W_out = (W_in + 2 * PW - KW) / SW + 1;

    Tensor output({B, C_out, H_out, W_out});

    size_t C_in_per_group = C_in / G;

    for (size_t b = 0; b < B; ++b) {
        for (size_t g = 0; g < G; ++g) {
            for (size_t c_out_g = 0; c_out_g < C_out / G; ++c_out_g) {
                size_t c_out = g * (C_out / G) + c_out_g;
                for (size_t h_out = 0; h_out < H_out; ++h_out) {
                    for (size_t w_out = 0; w_out < W_out; ++w_out) {
                        float acc = 0.0f;
                        for (size_t c_in_g = 0; c_in_g < C_in_per_group; ++c_in_g) {
                            size_t c_in = g * C_in_per_group + c_in_g;
                            for (size_t kh = 0; kh < KH; ++kh) {
                                for (size_t kw = 0; kw < KW; ++kw) {
                                    int h_in_idx = h_out * SH - PH + kh;
                                    int w_in_idx = w_out * SW - PW + kw;
                                    if (h_in_idx >= 0 && h_in_idx < H_in && w_in_idx >= 0 && w_in_idx < W_in) {
                                        int8_t packed_w = static_cast<int8_t>(layer.packed_weights.at(
                                            c_out * (C_in_per_group * KH * KW) +
                                            c_in_g * (KH * KW) +
                                            kh * KW + kw
                                        ));
                                        if (packed_w == 1) {
                                            acc += input.at({b, c_in, (size_t)h_in_idx, (size_t)w_in_idx});
                                        } else if (packed_w == -1) {
                                            acc -= input.at({b, c_in, (size_t)h_in_idx, (size_t)w_in_idx});
                                        }
                                    }
                                }
                            }
                        }
                        output.at({b, c_out, h_out, w_out}) = acc * layer.weight_scale + layer.bias[c_out];
                    }
                }
            }
        }
    }
    return output;
}


Tensor linear(const Tensor& input, const LinearLayer& layer) {
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