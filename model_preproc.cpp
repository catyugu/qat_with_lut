#include "model_preproc.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>

void quantize_float_to_int8_with_scale(const float* float_ptr, int8_t* int_ptr, size_t size, float fixed_scale) {
    for (size_t i = 0; i < size; ++i) {
        auto val = static_cast<int32_t>(roundf(float_ptr[i] * fixed_scale));
        int_ptr[i] = static_cast<int8_t>(std::max(-128, std::min(127, val)));
    }
}

// NEW: Function to build the 256x256-entry Bit-Slice LUT
void build_bit_slice_lut_5x3(std::vector<int16_t>& precomputed_lut) {
    const int LUT_DIM = 256; // 2^8 possibilities for packed byte
    const int LUT_SIZE = LUT_DIM * LUT_DIM; // 256 * 256 = 65536 entries
    precomputed_lut.resize(LUT_SIZE);

    // Temp arrays for unpacking 5 *ENCODED* ternary values (0,1,2) from a byte
    uint8_t encoded_acts_5[5];    // These are 0, 1, or 2
    uint8_t encoded_weights_5[5]; // These are 0, 1, or 2

    // Iterate through all possible 256 packed activation bytes
    for (int packed_act_byte_int = 0; packed_act_byte_int < LUT_DIM; ++packed_act_byte_int) {
        uint8_t packed_act_byte = static_cast<uint8_t>(packed_act_byte_int);
        // Step 1: Unpack the packed_act_byte into 5 encoded ternary values (0,1,2)
        unpack_five_ternary(packed_act_byte, reinterpret_cast<int8_t*>(encoded_acts_5)); // unpack_five_ternary takes int8_t array

        for (int packed_weight_byte_int = 0; packed_weight_byte_int < LUT_DIM; ++packed_weight_byte_int) {
            uint8_t packed_weight_byte = static_cast<uint8_t>(packed_weight_byte_int);
            // Step 2: Unpack the packed_weight_byte into 5 encoded ternary values (0,1,2)
            unpack_five_ternary(packed_weight_byte, reinterpret_cast<int8_t*>(encoded_weights_5)); // unpack_five_ternary takes int8_t array

            int16_t current_sum_of_products = 0;
            // Step 3: Iterate over the 5 pairs of encoded values, decode them, multiply, and sum
            for (int i = 0; i < 5; ++i) {
                // Decode each 3-bit encoded value (0,1,2) back to actual ternary (-1, 0, 1)
                int8_t decoded_act_val = decode_3bit_val_to_ternary(encoded_acts_5[i]);
                int8_t decoded_weight_val = decode_3bit_val_to_ternary(encoded_weights_5[i]);
                
                current_sum_of_products += (int16_t)decoded_act_val * decoded_weight_val;
            }
            
            // Store the sum of products in the LUT
            precomputed_lut[(static_cast<uint32_t>(packed_act_byte) << 8) | packed_weight_byte] = current_sum_of_products;
        }
    }
}

// Function to pack int8_t weights into 5x3bit packed uint8_t format
std::vector<uint8_t> pack_weights_5x3bit(const std::vector<int8_t>& unpacked_weights, int original_size) {
    std::vector<uint8_t> packed_weights_vec;
    packed_weights_vec.reserve((original_size + 4) / 5);

    // Temp array to hold 5 ternary values (0,1,2) for current byte
    int8_t five_ternary_vals[5];

    // Initialize with padding value '1' (for ternary 0, encoded)
    for(int k=0; k<5; ++k) five_ternary_vals[k] = encode_ternary_to_3bit_val(0);

    for (int i = 0; i < original_size; ++i) {
        int pack_idx_in_five = i % 5;
        five_ternary_vals[pack_idx_in_five] = encode_ternary_to_3bit_val(unpacked_weights[i]);

        if (pack_idx_in_five == 4) { // If 5 values collected (a full byte)
            packed_weights_vec.push_back(pack_five_ternary(five_ternary_vals));
            // Initialize for the next group with padding
             for(int k=0; k<5; ++k) five_ternary_vals[k] = encode_ternary_to_3bit_val(0);
        }
    }

    // Handle the last, possibly incomplete byte
    if (original_size % 5 != 0) {
        packed_weights_vec.push_back(pack_five_ternary(five_ternary_vals));
    }

    return packed_weights_vec;
}

// NEW: Function to pack int8_t ternary activations into 5x3bit packed uint8_t format
std::vector<uint8_t> pack_ternary_activations_5x3bit(const std::vector<int8_t>& unpacked_activations, int original_size) {
    std::vector<uint8_t> packed_activations_vec;
    packed_activations_vec.reserve((original_size + 4) / 5);

    int8_t five_ternary_vals[5];
    // Initialize with padding value '1' (for ternary 0, encoded)
    for(int k=0; k<5; ++k) five_ternary_vals[k] = encode_ternary_to_3bit_val(0);

    for (int i = 0; i < original_size; ++i) {
        int pack_idx_in_five = i % 5;
        // Ensure input activations are truly ternary or map them here
        five_ternary_vals[pack_idx_in_five] = encode_ternary_to_3bit_val(unpacked_activations[i]);

        if (pack_idx_in_five == 4) { // If 5 values collected (a full byte)
            packed_activations_vec.push_back(pack_five_ternary(five_ternary_vals));
            for(int k=0; k<5; ++k) five_ternary_vals[k] = encode_ternary_to_3bit_val(0); // Initialize for the next group with padding
        }
    }

    if (original_size % 5 != 0) {
        packed_activations_vec.push_back(pack_five_ternary(five_ternary_vals));
    }
    return packed_activations_vec;
}
