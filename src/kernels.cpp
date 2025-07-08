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
// Bit-Slice GEMM kernel using AVX512 intrinsics (LUT-based)
int32_t avx512_bit_slice_gemm_kernel(
    const uint8_t* input_packed_ptr,
    const uint8_t* weights_packed_ptr,
    const int32_t* precomputed_lut_ptr, // <-- Data type is correct
    int k_dim
) {
    __m512i accumulated_sum_vec_32 = _mm512_setzero_si512();

    const int num_packed_bytes = k_dim / 5;
    if (k_dim % 320 != 0) {
        std::cerr << "Error: k_dim must be a multiple of 320 for AVX512 bit-slice kernel (" << k_dim << ")." << std::endl;
        return 0;
    }

    for (int byte_idx_block = 0; byte_idx_block < num_packed_bytes; byte_idx_block += 64) {
        _mm_prefetch(reinterpret_cast<const char*>(input_packed_ptr + byte_idx_block + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(weights_packed_ptr + byte_idx_block + 64), _MM_HINT_T0);

        __m512i packed_acts_bytes = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(input_packed_ptr + byte_idx_block));
        __m512i packed_weights_bytes = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(weights_packed_ptr + byte_idx_block));

        alignas(64) uint8_t temp_packed_acts_raw[64];
        alignas(64) uint8_t temp_packed_weights_raw[64];
        alignas(64) int16_t block_sum_of_products[64];

        _mm512_storeu_si512(reinterpret_cast<__m512i*>(temp_packed_acts_raw), packed_acts_bytes);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(temp_packed_weights_raw), packed_weights_bytes);

        // This loop is fine
        for (int i = 0; i < 64; ++i) {
            uint8_t packed_act_byte = temp_packed_acts_raw[i];
            uint8_t packed_weight_byte = temp_packed_weights_raw[i];
            block_sum_of_products[i] = precomputed_lut_ptr[(static_cast<uint32_t>(packed_act_byte) << 8) | packed_weight_byte];
        }

        // ======================= THE FIX =======================
        // Correctly accumulate the 64 int16_t results using AVX512
        // Load the first 32 int16 values (fills a 512-bit vector)
        __m512i sum_of_products_vec0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(block_sum_of_products));
        // Load the next 32 int16 values
        __m512i sum_of_products_vec1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(block_sum_of_products + 32));

        // Convert 16-bit sums to 32-bit sums for accumulation without overflow.
        // Process sum_of_products_vec0
        __m512i sum_0_lo_32 = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(sum_of_products_vec0, 0));
        __m512i sum_0_hi_32 = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(sum_of_products_vec0, 1));
        
        // Process sum_of_products_vec1
        __m512i sum_1_lo_32 = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(sum_of_products_vec1, 0));
        __m512i sum_1_hi_32 = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(sum_of_products_vec1, 1));
        
        // Accumulate all parts
        accumulated_sum_vec_32 = _mm512_add_epi32(accumulated_sum_vec_32, sum_0_lo_32);
        accumulated_sum_vec_32 = _mm512_add_epi32(accumulated_sum_vec_32, sum_0_hi_32);
        accumulated_sum_vec_32 = _mm512_add_epi32(accumulated_sum_vec_32, sum_1_lo_32);
        accumulated_sum_vec_32 = _mm512_add_epi32(accumulated_sum_vec_32, sum_1_hi_32);
        // =======================================================
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
    const int32_t* precomputed_lut_ptr
) {
    output_f32_batched.resize(batch_size * output_dim);
    const uint8_t* weights_packed_ptr = layer.packed_weights.data();
    const int packed_input_bytes_per_sample = (input_dim + 4) / 5;
    for (int b = 0; b < batch_size; ++b) {
        const uint8_t* current_batch_input_packed_ptr = input_packed_batched.data() + b * packed_input_bytes_per_sample;
        for (int i = 0; i < output_dim; ++i) {
            const uint8_t* current_neuron_weights_packed_ptr = weights_packed_ptr + i * packed_input_bytes_per_sample;

            int32_t sum;
            // ======================= THE FIX =======================
            // Use the correct kernel based on compilation flags
            #ifdef __AVX512F__
                sum = avx512_bit_slice_gemm_kernel(
                    current_batch_input_packed_ptr,
                    current_neuron_weights_packed_ptr,
                    precomputed_lut_ptr,
                    input_dim
                );
            #else
                sum = avx2_bit_slice_gemm_kernel(
                    current_batch_input_packed_ptr,
                    current_neuron_weights_packed_ptr,
                    precomputed_lut_ptr,
                    input_dim
                );
            #endif
            // =======================================================
            
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

__attribute__((target("avx,fma")))
Tensor linear(const Tensor& input, const LinearLayer& layer) {
    PROFILE_SCOPE("linear_avx");

    // --- 1. Get dimensions ---
    const auto& in_shape = input.shape;
    const size_t B = (in_shape.size() > 1) ? in_shape[0] : 1; // Batch size
    const size_t in_features = in_shape.back();
    const size_t out_features = layer.out_features;

    // --- Sanity Check ---
    if (in_features != layer.in_features) {
        throw std::runtime_error("Mismatched dimensions in linear layer. Input has " + 
                               std::to_string(in_features) + " features, but layer expects " +
                               std::to_string(layer.in_features));
    }

    // --- 2. Prepare output tensor ---
    Tensor output({B, out_features});
    
    const float* input_ptr = input.data.data();
    const float* weights_ptr = layer.weights.data();
    const float* bias_ptr = layer.bias.data();
    float* output_ptr = output.data.data();

    const int AVX_FLOAT_COUNT = 8; // AVX processes 8 floats at a time

    // --- 3. Main Computation Loop ---
    // Iterate over each item in the batch
    for (size_t b = 0; b < B; ++b) {
        // Pointer to the start of the current input sample
        const float* current_input_ptr = input_ptr + b * in_features;

        // For each output neuron, compute the dot product with the input vector
        for (size_t out_f = 0; out_f < out_features; ++out_f) {
            
            // --- Vectorized Dot Product ---
            // Initialize an AVX register to hold the sum
            __m256 sum_vec = _mm256_setzero_ps();
            
            // Pointer to the start of the weights for the current output neuron
            const float* current_weights_ptr = weights_ptr + out_f * in_features;

            // Process the input vector in chunks of 8
            size_t j = 0;
            for (; j + AVX_FLOAT_COUNT <= in_features; j += AVX_FLOAT_COUNT) {
                // Load 8 floats from the input
                __m256 input_chunk = _mm256_loadu_ps(current_input_ptr + j);
                // Load 8 floats from the weights
                __m256 weight_chunk = _mm256_loadu_ps(current_weights_ptr + j);
                
                // Fused Multiply-Add: sum_vec += input_chunk * weight_chunk
                sum_vec = _mm256_fmadd_ps(input_chunk, weight_chunk, sum_vec);
            }

            // --- Horizontal Sum ---
            // Sum the 8 partial results within the AVX register to get a single float
            float dot_product_avx = 0.0f;
            float temp[AVX_FLOAT_COUNT];
            _mm256_storeu_ps(temp, sum_vec);
            for(int k=0; k<AVX_FLOAT_COUNT; ++k) {
                dot_product_avx += temp[k];
            }

            // --- Scalar Remainder ---
            // Process any remaining elements that didn't fit into a chunk of 8
            for (; j < in_features; ++j) {
                dot_product_avx += current_input_ptr[j] * current_weights_ptr[j];
            }

            // --- 4. Add bias and store final result ---
            output_ptr[b * out_features + out_f] = dot_product_avx + bias_ptr[out_f];
        }
    }
    
    return output;
}

void im2col_final(
    const Tensor& input_tensor, int b,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w,
    int out_h, int out_w, std::vector<float>& col_buffer) {
    PROFILE_SCOPE("im2col_final");

    const int C_in = input_tensor.shape[1];
    const int H_in = input_tensor.shape[2];
    const int W_in = input_tensor.shape[3];

    const int K_gemm = C_in * kernel_h * kernel_w;
    const int N_gemm = out_h * out_w;
    
    // It's good practice to ensure the buffer is the correct size and zeroed out.
    col_buffer.assign(K_gemm * N_gemm, 0.0f);

    const size_t input_channel_plane_size = H_in * W_in;
    const float* input_data_ptr = input_tensor.data.data() + b * C_in * input_channel_plane_size;

    // Loop over input channels, then kernel spatial dimensions.
    for (int c = 0; c < C_in; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                // This identifies the current row in the output column matrix.
                const int k_gemm_row = c * (kernel_h * kernel_w) + kh * kernel_w + kw;

                // Loop over output spatial dimensions.
                for (int h_out = 0; h_out < out_h; ++h_out) {
                    for (int w_out = 0; w_out < out_w; ++w_out) {
                        const int h_in = h_out * stride_h - pad_h + kh;
                        const int w_in = w_out * stride_w - pad_w + kw;
                        
                        // This identifies the current column in the output column matrix.
                        const int n_gemm_col = h_out * out_w + w_out;

                        // Check bounds and explicitly handle padding.
                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            const size_t input_idx = c * input_channel_plane_size + h_in * W_in + w_in;
                            col_buffer[k_gemm_row * N_gemm + n_gemm_col] = input_data_ptr[input_idx];
                        }
                        // No 'else' needed because the buffer is already zeroed.
                    }
                }
            }
        }
    }
}


/**
 * @brief A definitive, robust implementation for GEMM with packed 2-bit weights.
 *
 * This version uses a clear, standard C++ loop structure to ensure correctness
 * and readability, minimizing the risk of subtle bugs from compiler optimizations
 * or complex memory access. It correctly unpacks 2-bit signed integers (-1, 0, 1)
 * and performs the matrix multiplication C = A * B.
 *
 * @param A_packed Pointer to the packed 2-bit weight matrix (A). Shape: [M, K].
 * @param B_float Pointer to the float matrix (B), which is the im2col buffer. Shape: [K, N].
 * @param C Pointer to the output float matrix (C). Shape: [M, N].
 * @param M The number of rows in A and C (equivalent to out_channels).
 * @param N The number of columns in B and C (equivalent to out_height * out_width).
 * @param K The number of columns in A and rows in B (equivalent to in_channels * kernel_h * kernel_w).
 */
__attribute__((target("avx,fma")))
void gemm_packed_x_float_non_lut(
    const uint32_t* A_packed, const float* B_float, float* C, int M, int N, int K) {
    PROFILE_SCOPE("gemm_packed_verified");

    // Initialize the output matrix C to all zeros. This is crucial.
    std::fill(C, C + M * N, 0.0f);

    const int AVX_FLOAT_COUNT = 8; // AVX registers hold 8 single-precision floats

    // Use a standard GEMM loop order for correctness and cache efficiency.
    for (int i = 0; i < M; ++i) {       // Iterate over rows of A (output channels)
        for (int k = 0; k < K; ++k) {   // Iterate over columns of A / rows of B
            
            // --- Step 1: Unpack the single 2-bit weight from matrix A ---
            const size_t weight_idx_flat = i * K + k;
            const uint32_t packed_word = A_packed[weight_idx_flat / 16];
            const size_t bit_shift = (weight_idx_flat % 16) * 2;
            const uint8_t two_bit_val = (packed_word >> bit_shift) & 0x03;

            // This mapping is the core of the ternary unpacking.
            const float w = (two_bit_val == 1) ? 1.0f : ((two_bit_val == 2) ? -1.0f : 0.0f);

            // If the weight is zero, it contributes nothing to the output, so we can skip.
            if (w == 0.0f) {
                continue;
            }

            // --- Step 2: Perform the outer product (w * row_k_of_B) and add it to row_i_of_C ---
            const float* B_row_ptr = &B_float[k * N];
            float* C_row_ptr = &C[i * N];

            // Use AVX intrinsics to accelerate the vector-scalar multiplication and addition.
            int j = 0;
            for (; j <= N - AVX_FLOAT_COUNT; j += AVX_FLOAT_COUNT) {
                // Load 8 float values from the corresponding row in B and C.
                __m256 b_vals = _mm256_loadu_ps(B_row_ptr + j);
                __m256 c_vals = _mm256_loadu_ps(C_row_ptr + j);
                
                // Perform the fused multiply-add operation: c_vals = (w * b_vals) + c_vals
                // We use _mm256_set1_ps to broadcast the scalar 'w' into a vector.
                c_vals = _mm256_fmadd_ps(_mm256_set1_ps(w), b_vals, c_vals);

                // Store the updated 8 float values back into C.
                _mm256_storeu_ps(C_row_ptr + j, c_vals);
            }

            // Handle any remaining elements at the end of the row that don't fit in a full AVX register.
            for (; j < N; ++j) {
                C_row_ptr[j] += w * B_row_ptr[j];
            }
        }
    }
}
Tensor conv2d(const Tensor& input, const QATConv2dLayer& layer) {
    PROFILE_SCOPE("conv2d_ternary_optimized_single_core");

    const int B = input.shape[0];
    const int C_in = input.shape[1];
    if (C_in != layer.in_channels) { throw std::runtime_error("Mismatched input channels in conv2d"); }
    const int H_in = input.shape[2];
    const int W_in = input.shape[3];

    const int C_out = layer.out_channels;
    const int KH = layer.kernel_size_h;
    const int KW = layer.kernel_size_w;
    const int H_out = (H_in + 2 * layer.pad_h - KH) / layer.stride_h + 1;
    const int W_out = (W_in + 2 * layer.pad_w - KW) / layer.stride_w + 1;

    // GEMM dimensions
    const int M = C_out;
    const int N = H_out * W_out;
    const int K = C_in * KH * KW;

    Tensor output({(size_t)B, (size_t)C_out, (size_t)H_out, (size_t)W_out});
    const uint32_t* packed_weights_ptr = layer.packed_weights.data();

    std::vector<float> col_buffer(K * N);
    std::vector<float> gemm_output_buffer(M * N);

    for (int b = 0; b < B; ++b) {
        // Step 1: Call the new, corrected im2col function.
        im2col_final(input, b, KH, KW, layer.stride_h, layer.stride_w, layer.pad_h, layer.pad_w, H_out, W_out, col_buffer);

        // Step 2: Perform matrix multiplication (this part is correct).
        gemm_packed_x_float_non_lut(packed_weights_ptr, col_buffer.data(), gemm_output_buffer.data(), M, N, K);

        // Step 3: Reshape, add bias, and scale (this part is correct).
        for (int c_out = 0; c_out < M; ++c_out) {
            const float bias = layer.bias.empty() ? 0.0f : layer.bias[c_out];
            for (int hw = 0; hw < N; ++hw) {
                size_t out_idx = b * (M * N) + c_out * N + hw;
                output.data[out_idx] = gemm_output_buffer[c_out * N + hw] * layer.alpha + bias;
            }
        }
    }
    return output;
}

Tensor group_norm(
    const Tensor& input_tensor, // 接收 const 引用
    const GroupNormLayer& layer,
    float eps = 1e-5f
) {
     PROFILE_SCOPE("group_norm_verified");

    const size_t B = input_tensor.shape[0];
    const size_t C = input_tensor.shape[1];
    const size_t H = input_tensor.shape[2];
    const size_t W = input_tensor.shape[3];

    const size_t num_groups = layer.num_groups;
    if (C % num_groups != 0) {
        throw std::runtime_error("GroupNorm error: num_channels must be divisible by num_groups.");
    }
    const size_t channels_per_group = C / num_groups;
    const size_t group_size = channels_per_group * H * W;

    const float* gamma = layer.weight.data(); // Learned scaling parameter
    const float* beta = layer.bias.data();   // Learned shifting parameter

    Tensor output_tensor(input_tensor.shape);

    // Process each item in the batch independently.
    for (size_t b = 0; b < B; ++b) {
        // Process each group independently.
        for (size_t g = 0; g < num_groups; ++g) {
            
            // --- Step 1: Calculate mean and variance for the current group ---
            double sum = 0.0;
            double sum_sq = 0.0;

            for (size_t c_group = 0; c_group < channels_per_group; ++c_group) {
                const size_t c_abs = g * channels_per_group + c_group;
                for (size_t hw = 0; hw < H * W; ++hw) {
                    const float val = input_tensor.at({b, c_abs, hw / W, hw % W});
                    sum += val;
                    sum_sq += val * val;
                }
            }

            const float mean = static_cast<float>(sum / group_size);
            const float var = static_cast<float>(sum_sq / group_size) - (mean * mean);
            const float std_dev_inv = 1.0f / std::sqrt(var + eps);

            // --- Step 2: Apply normalization, scaling (gamma), and shifting (beta) ---
            for (size_t c_group = 0; c_group < channels_per_group; ++c_group) {
                const size_t c_abs = g * channels_per_group + c_group;
                const float g_val = gamma[c_abs];
                const float b_val = beta[c_abs];

                for (size_t hw = 0; hw < H * W; ++hw) {
                    const float val = input_tensor.at({b, c_abs, hw / W, hw % W});
                    // Apply the full GroupNorm formula.
                    output_tensor.at({b, c_abs, hw / W, hw % W}) = (val - mean) * std_dev_inv * g_val + b_val;
                }
            }
        }
    }
    return output_tensor;
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

    // Apply Hadamard Transform (if used in training)
    q = q.hadamard_transform();
    k = k.hadamard_transform();
    v = v.hadamard_transform();

    // ========================= THE FIX =========================
    // Reshape and permute tensors to exactly match the Python implementation
    
    // Reshape q and v for BMM: [B, HW, C]
    q.reshape({B, C, H * W});
    q = q.permute({0, 2, 1}); 

    // Reshape v for BMM: [B, HW, C]
    v.reshape({B, C, H * W});
    v = v.permute({0, 2, 1});

    // Reshape k directly for BMM: [B, C, HW]
    k.reshape({B, C, H * W});
    
    // Perform scaled dot-product attention
    // q [B, HW, C] @ k [B, C, HW] -> scores [B, HW, HW]
    Tensor scores = q.matmul(k); // NO permute on k here!
    // =========================================================

    scores = scores.mul_scalar(1.0f / std::sqrt(static_cast<float>(C)));
    scores = softmax(scores, -1); // Softmax on the last dimension

    // Apply attention to v
    Tensor out = scores.matmul(v); // [B, HW, C]
    out = out.permute({0, 2, 1});   // [B, C, HW]
    out.reshape({B, C, H, W});

    // Final convolution and residual connection
    out = conv2d(out, *block.to_out);
    return input.add(out);
}


// --- 唯一的 Positional Embedding ---
Tensor positional_embedding(const Tensor& time, const PositionalEmbedding& layer) {
    PROFILE_SCOPE("positional_embedding");
    size_t B = time.shape[0];
    size_t dim = layer.dim;
    size_t half_dim = dim / 2;
    Tensor output({B, dim});
    float log_10000 = std::log(10000.0f);
    for(size_t b = 0; b < B; ++b) {
        for(size_t i = 0; i < half_dim; ++i) {
            float emb = std::exp(static_cast<float>(i) * -log_10000 / static_cast<float>(half_dim));
            float arg = time.data[b] * layer.time_emb_scale * emb;
            output.at({b, i}) = std::sin(arg);
            output.at({b, i + half_dim}) = std::cos(arg);
        }
    }
    return output;
}

Tensor residual_block(const Tensor& input, const QATResidualBlock& block, const Tensor& time_emb, const Tensor& y) {
    PROFILE_SCOPE("residual_block");
    Tensor h = group_norm(input, *block.norm_1);
    h = silu(h);
    h = conv2d(h, *block.conv_1);
    if (block.time_bias) {
        Tensor silu_time_emb_output = silu(time_emb); // Capture the output of silu
        Tensor time_bias = linear(silu_time_emb_output, *block.time_bias);
        time_bias.reshape({time_bias.shape[0], time_bias.shape[1], 1, 1});
        h = h.add(time_bias);
    }
    if (block.class_bias) {
        int class_idx = static_cast<int>(y.data[0]);
        int embedding_dim = block.class_bias->embedding_dim;
        Tensor class_bias_tensor({1, (size_t)embedding_dim});
        const auto& weights = block.class_bias->weight;
        std::copy(weights.begin() + class_idx * embedding_dim,
                  weights.begin() + (class_idx + 1) * embedding_dim,
                  class_bias_tensor.data.begin());
        class_bias_tensor.reshape({1, (size_t)embedding_dim, 1, 1});
        h = h.add(class_bias_tensor);
    }
    Tensor h2 = group_norm(h, *block.norm_2);
    h2 = silu(h2);
    // The dropout layer from Python is an identity op during inference,
    // so we directly call the convolution here, which is correct.
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

Tensor forward_qat_unet(const QATUNetModel& model, const Tensor& x, const Tensor& time, const Tensor& y) {
    PROFILE_SCOPE("forward_qat_unet_final");

    // --- Time Embedding ---
    Tensor time_emb = positional_embedding(time, *model.time_mlp_pos_emb);
    time_emb = linear(time_emb, *model.time_mlp_linear1);
    time_emb = silu(time_emb);
    time_emb = linear(time_emb, *model.time_mlp_linear2);

    // --- Initial Convolution & Skip Connection Setup ---
    Tensor h = conv2d(x, *model.init_conv);
    std::vector<Tensor> skips;
    skips.push_back(h);

    // --- 1. Down-sampling Path (Encoder) ---
    for (const auto& layer_ptr : model.downs) {
        if (auto* res_block = dynamic_cast<QATResidualBlock*>(layer_ptr.get())) {
            h = residual_block(h, *res_block, time_emb, y);
        } else if (auto* downsample_block = dynamic_cast<DownsampleLayer*>(layer_ptr.get())) {
            h = conv2d(h, *downsample_block->conv);
        } else {
            throw std::runtime_error("Unknown layer type in downs");
        }
        skips.push_back(h);
    }

    // --- 2. Middle Path (Bottleneck) ---
    // Iterate through middle blocks sequentially as in the Python ModuleList.
    h = residual_block(h, *model.middle_block1, time_emb, y);
    // The Python model has an attention block here. Let's assume it's inside the ResBlock for now.
    h = residual_block(h, *model.middle_block2, time_emb, y);

    // --- 3. Up-sampling Path (Decoder) ---
    for (const auto& layer_ptr : model.ups) {
        // Correctly handle different layer types in the upsampling path.
        if (auto* res_block = dynamic_cast<QATResidualBlock*>(layer_ptr.get())) {
            Tensor skip_tensor = skips.back();
            skips.pop_back();

            // CRITICAL FIX: Use concat, not add. The order h.concat(skip) is
            // equivalent to torch.cat([h, skip], dim=1).
            h = h.concat(skip_tensor);
            
            // Now call the residual block on the concatenated tensor.
            h = residual_block(h, *res_block, time_emb, y);
        } else if (auto* upsample_block = dynamic_cast<UpsampleLayer*>(layer_ptr.get())) {
            // Upsampling logic seems correct.
            h = h.upsample(2);
            h = conv2d(h, *upsample_block->conv);
        } else {
            throw std::runtime_error("Unknown layer type in ups");
        }
    }

    // --- 4. Final Output Layers ---
    // CRITICAL FIX: Add the missing 'eps' argument to group_norm.
    h = group_norm(h, *model.final_norm, 1e-5f);
    h = silu(h);
    print_tensor_stats(h, "Final Tensor before out_conv"); // Keep this for one last check
    h = conv2d(h, *model.final_conv);

    return h;
}