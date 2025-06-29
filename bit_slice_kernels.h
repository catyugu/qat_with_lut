#ifndef BIT_SLICE_KERNELS_H
#define BIT_SLICE_KERNELS_H

#include <cstdint> // For uint8_t, int16_t, int32_t
#include <vector>  // For std::vector (though raw pointers will be used in kernel)

#ifdef __AVX2__
#include <immintrin.h> // For AVX2 intrinsics

int32_t avx2_bit_slice_gemm_kernel(
    const uint8_t* input_packed_ptr,    // Pointer to packed activations (NxK_packed)
    const uint8_t* weights_packed_ptr,  // Pointer to packed weights (MxK_packed)
    const int16_t* precomputed_lut_ptr, // Pointer to the 256x256 precomputed LUT
    int k_dim                          // The K dimension (input_dim for layer1, hidden_dim for layer2)
);

#endif // __AVX2__

#endif // BIT_SLICE_KERNELS_H
