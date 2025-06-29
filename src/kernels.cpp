#include "kernels.h" // Include the corresponding header

#include <immintrin.h>

#include "types.h"   // Include types.h for struct definitions
#include "utils.h"   // Include utils.h for functions like convert_int8_to_ternary_activation, pack_ternary_activations_5x3bit etc.

#include <iostream> // For debugging, remove in final build

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
    const int16_t* precomputed_lut_ptr,
    int k_dim
) {
    __m512i accumulated_sum_vec_32 = _mm512_setzero_si512();

    const int num_packed_bytes = k_dim / 5;
    if (k_dim % 320 != 0) { // AVX512 loop processes 64 packed bytes = 320 elements
        std::cerr << "Error: k_dim must be a multiple of 320 for AVX512 bit-slice kernel (" << k_dim << ")." << std::endl;
        return 0;
    }

    for (int byte_idx_block = 0; byte_idx_block < num_packed_bytes; byte_idx_block += 64) {
        __m512i packed_acts_bytes = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(input_packed_ptr + byte_idx_block));
        __m512i packed_weights_bytes = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(weights_packed_ptr + byte_idx_block));

        alignas(64) uint8_t temp_packed_acts_raw[64];
        alignas(64) uint8_t temp_packed_weights_raw[64];
        alignas(64) int16_t block_sum_of_products[64];

        _mm512_storeu_si512(reinterpret_cast<__m512i*>(temp_packed_acts_raw), packed_acts_bytes);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(temp_packed_weights_raw), packed_weights_bytes);

        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            uint8_t packed_act_byte = temp_packed_acts_raw[i];
            uint8_t packed_weight_byte = temp_packed_weights_raw[i];
            block_sum_of_products[i] = precomputed_lut_ptr[(static_cast<uint32_t>(packed_act_byte) << 8) | packed_weight_byte];
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

// Bit-Slice GEMM kernel using AVX2 intrinsics (LUT-based)
int32_t avx2_bit_slice_gemm_kernel(
    const uint8_t* input_packed_ptr,
    const uint8_t* weights_packed_ptr,
    const int16_t* precomputed_lut_ptr,
    int k_dim
) {
    __m256i accumulated_sum_vec_32 = _mm256_setzero_si256();

    const int num_packed_bytes = k_dim / 5;

    for (int byte_idx_block = 0; byte_idx_block < num_packed_bytes; byte_idx_block += 32) {
        alignas(32) uint8_t temp_packed_acts_raw[32];
        alignas(32) uint8_t temp_packed_weights_raw[32];
        alignas(32) int16_t block_sum_of_products[32];

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(temp_packed_acts_raw), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_packed_ptr + byte_idx_block)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(temp_packed_weights_raw), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights_packed_ptr + byte_idx_block)));

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            uint8_t packed_act_byte = temp_packed_acts_raw[i];
            uint8_t packed_weight_byte = temp_packed_weights_raw[i];
            block_sum_of_products[i] = precomputed_lut_ptr[(static_cast<uint32_t>(packed_act_byte) << 8) | packed_weight_byte];
        }

        __m256i sum_of_products_vec0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block_sum_of_products));
        __m256i sum_of_products_vec1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block_sum_of_products + 16));

        __m256i sum_lo_32_0 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_of_products_vec0, 0));
        __m256i sum_hi_32_0 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_of_products_vec0, 1));
        accumulated_sum_vec_32 = _mm256_add_epi32(accumulated_sum_vec_32, sum_lo_32_0);
        accumulated_sum_vec_32 = _mm256_add_epi32(accumulated_sum_vec_32, sum_hi_32_0);

        __m256i sum_lo_32_1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_of_products_vec1, 0));
        __m256i sum_hi_32_1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_of_products_vec1, 1));
        accumulated_sum_vec_32 = _mm256_add_epi32(accumulated_sum_vec_32, sum_lo_32_1);
        accumulated_sum_vec_32 = _mm256_add_epi32(accumulated_sum_vec_32, sum_hi_32_1);
    }
    return hsum_i32_8(accumulated_sum_vec_32);
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
    const int16_t* precomputed_lut_ptr
) {
    output_f32_batched.resize(batch_size * output_dim);
    const uint8_t* weights_packed_ptr = layer.packed_weights.data();
    const int packed_input_bytes_per_sample = (input_dim + 4) / 5;
    #pragma omp parallel for collapse(2) // Add this pragma
    for (int b = 0; b < batch_size; ++b) {
        const uint8_t* current_batch_input_packed_ptr = input_packed_batched.data() + b * packed_input_bytes_per_sample;
        for (int i = 0; i < output_dim; ++i) { // Fixed loop condition
            const uint8_t* current_neuron_weights_packed_ptr = weights_packed_ptr + i * packed_input_bytes_per_sample;
            int32_t sum = 0;

#ifdef __AVX512F__
            sum = avx512_bit_slice_gemm_kernel(
                current_batch_input_packed_ptr,
                current_neuron_weights_packed_ptr,
                precomputed_lut_ptr,
                input_dim
            );
#else // Fallback to AVX2
            sum = avx2_bit_slice_gemm_kernel(
                current_batch_input_packed_ptr,
                current_neuron_weights_packed_ptr,
                precomputed_lut_ptr,
                input_dim
            );
#endif
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
    #pragma omp parallel for collapse(2) // Add this pragma
    for (int b = 0; b < batch_size; ++b) {
        const int8_t* current_batch_input_ptr = input_i8_batched.data() + b * input_dim;
        for (int i = 0; i < output_dim; ++i) {
            int32_t sum = 0;
            const int8_t* current_neuron_weights_ptr = weights_ptr + i * input_dim;

#ifdef __AVX512F__
            __m512i accumulated_sum_vec_32 = _mm512_setzero_si512();

            for (int j = 0; j < input_dim; j += 64) {
                __m512i input_vals_512 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(current_batch_input_ptr + j));
                __m512i weights_block_512 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(current_neuron_weights_ptr + j));

                __m512i input_vals_lo_16 = _mm512_cvtepi8_epi16(_mm512_extracti32x8_epi32(input_vals_512, 0)); // Lower 32 int8_t -> 32 int16_t
                __m512i weights_block_lo_16 = _mm512_cvtepi8_epi16(_mm512_extracti32x8_epi32(weights_block_512, 0)); // Lower 32 int8_t -> 32 int16_t
                __m512i partial_sums_lo_16 = _mm512_mullo_epi16(input_vals_lo_16, weights_block_lo_16);

                __m512i input_vals_hi_16 = _mm512_cvtepi8_epi16(_mm512_extracti32x8_epi32(input_vals_512, 1)); // Higher 32 int8_t -> 32 int16_t
                __m512i weights_block_hi_16 = _mm512_cvtepi8_epi16(_mm512_extracti32x8_epi32(weights_block_512, 1)); // Higher 32 int8_t -> 32 int16_t
                __m512i partial_sums_hi_16 = _mm512_mullo_epi16(input_vals_hi_16, weights_block_hi_16);


                // __m512i partial_sums_lo_16 = _mm512_cvtepi8_epi16(_mm512_extracti32x8_epi32(partial_products_i8_512, 0));
                // __m512i partial_sums_hi_16 = _mm512_cvtepi8_epi16(_mm512_extracti32x8_epi32(partial_products_i8_512, 1));

                accumulated_sum_vec_32 = _mm512_add_epi32(accumulated_sum_vec_32, _mm512_madd_epi16(partial_sums_lo_16, _mm512_set1_epi16(1)));
                accumulated_sum_vec_32 = _mm512_add_epi32(accumulated_sum_vec_32, _mm512_madd_epi16(partial_sums_hi_16, _mm512_set1_epi16(1)));
            }
            sum = hsum_i32_16(accumulated_sum_vec_32);

#else // Fallback to AVX2 if AVX512F is not defined
            __m256i accumulated_sum_vec_32 = _mm256_setzero_si256();

            for (int j = 0; j < input_dim; j += 32) {
                __m256i input_vals = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(current_batch_input_ptr + j));
                __m256i weights_block = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(current_neuron_weights_ptr + j));

                __m256i partial_products_i8 = _mm256_sign_epi8(input_vals, weights_block);

                __m256i partial_sums_lo_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(partial_products_i8, 0));
                __m256i partial_sums_hi_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(partial_products_i8, 1));

                accumulated_sum_vec_32 = _mm256_add_epi32(accumulated_sum_vec_32, _mm256_madd_epi16(partial_sums_lo_16, _mm256_set1_epi16(1)));
                accumulated_sum_vec_32 = _mm256_add_epi32(accumulated_sum_vec_32, _mm256_madd_epi16(partial_sums_hi_16, _mm256_set1_epi16(1)));
            }
            sum = hsum_i32_8(accumulated_sum_vec_32);
#endif
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
    #pragma omp parallel for collapse(2) // Add this pragma
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
