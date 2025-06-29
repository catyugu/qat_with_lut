#include "bit_slice_kernels.h"

#include <immintrin.h>
#include <iostream> // For debugging, remove in final build
#include "bit_slice_kernels.h"

// Helper for horizontal sum (from main.cpp)
static inline int hsum_i32_8_bsk(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

// Function to perform Bit-Slice GEMM using AVX2 intrinsics.
// This function aims to directly perform LUT lookups,
// processing one output neuron's contribution from K_dim inputs.
// It processes 32 packed bytes (160 activations/weights pairs) per block.
int32_t avx2_bit_slice_gemm_kernel(
    const uint8_t* input_packed_ptr,
    const uint8_t* weights_packed_ptr,
    const int16_t* precomputed_lut_ptr,
    int k_dim // The K dimension (e.g., padded_input_dim or hidden_dim)
) {
    __m256i accumulated_sum_vec_32 = _mm256_setzero_si256();

    // Loop processes 32 packed bytes at a time (which is 32 * 5 = 160 elements)
    // k_dim must be a multiple of 160 for this simplified kernel.
    const int num_packed_bytes = k_dim / 5; // Total number of packed bytes for this dimension

    // Each iteration of the loop processes 32 packed bytes
    for (int byte_idx_block = 0; byte_idx_block < num_packed_bytes; byte_idx_block += 32) {
        // Load 32 packed activation bytes and 32 packed weight bytes
        __m256i packed_acts_bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_packed_ptr + byte_idx_block));
        __m256i packed_weights_bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights_packed_ptr + byte_idx_block));

        // Store to raw arrays to access bytes.
        // This is a workaround for the difficulty of general-purpose 16-bit indexed gather/shuffle in AVX2,
        // allowing scalar LUT lookups on the loaded data.
        alignas(32) uint8_t temp_packed_acts_raw[32];
        alignas(32) uint8_t temp_packed_weights_raw[32];
        alignas(32) int16_t block_sum_of_products[32];

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(temp_packed_acts_raw), packed_acts_bytes);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(temp_packed_weights_raw), packed_weights_bytes);

        // Perform the LUT lookup for each of the 32 byte pairs
        // Loop unrolled for potential minor performance gain
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            uint8_t packed_act_byte = temp_packed_acts_raw[i];
            uint8_t packed_weight_byte = temp_packed_weights_raw[i];

            block_sum_of_products[i] = precomputed_lut_ptr[(static_cast<uint32_t>(packed_act_byte) << 8) | packed_weight_byte];
        }

        // Accumulate the 32 int16_t results using AVX2 intrinsics
        // Two _mm256i vectors hold 16 int16_t values each (total 32 int16_t)
        __m256i sum_of_products_vec0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block_sum_of_products));
        __m256i sum_of_products_vec1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block_sum_of_products + 16));

        // Convert 16-bit sums to 32-bit sums for accumulation
        __m256i sum_lo_32_0 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_of_products_vec0, 0)); // First 8 int16_t -> 8 int32_t
        __m256i sum_hi_32_0 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_of_products_vec0, 1)); // Next 8 int16_t -> 8 int32_t

        __m256i sum_lo_32_1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_of_products_vec1, 0));
        __m256i sum_hi_32_1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum_of_products_vec1, 1));

        // Add to the accumulated sum vector
        accumulated_sum_vec_32 = _mm256_add_epi32(accumulated_sum_vec_32, sum_lo_32_0);
        accumulated_sum_vec_32 = _mm256_add_epi32(accumulated_sum_vec_32, sum_hi_32_0);
        accumulated_sum_vec_32 = _mm256_add_epi32(accumulated_sum_vec_32, sum_lo_32_1);
        accumulated_sum_vec_32 = _mm256_add_epi32(accumulated_sum_vec_32, sum_hi_32_1);
    }

    return hsum_i32_8_bsk(accumulated_sum_vec_32);
}
