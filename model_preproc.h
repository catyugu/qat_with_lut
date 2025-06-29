#ifndef MODEL_PREPROC_H
#define MODEL_PREPROC_H

#include <vector>
#include <cstdint> // For uint8_t, int8_t, int16_t
#include <algorithm> // For std::max, std::min
#include <cmath>     // For roundf

// --- NEW: 5x3 bit Encoding/Decoding Functions ---
// Encoding: packed_byte = t0 + t1*3 + t2*9 + t3*27 + t4*81
// where t_i are 0, 1, or 2 (representing -1, 0, 1)

// Maps ternary (-1, 0, 1) to encoded (0, 1, 2)
inline uint8_t encode_ternary_to_3bit_val(int8_t val) {
    if (val == -1) return 0;
    if (val == 0)  return 1;
    if (val == 1)  return 2;
    return 1; // Default to 0 (encoded as 1)
}

// Decodes 3bit value (0,1,2) back to ternary (-1, 0, 1)
inline int8_t decode_3bit_val_to_ternary(uint8_t encoded_val) {
    if (encoded_val == 0) return -1;
    if (encoded_val == 1) return 0;
    if (encoded_val == 2) return 1;
    return 0; // Default for invalid codes
}

// NEW: Pack 5 ternary values (each 0,1,2) into a single uint8_t byte
inline uint8_t pack_five_ternary(const int8_t t_values[5]) {
    uint8_t packed_byte = 0;
    packed_byte += t_values[0] * 1;
    packed_byte += t_values[1] * 3;
    packed_byte += t_values[2] * 9;
    packed_byte += t_values[3] * 27;
    packed_byte += t_values[4] * 81;
    return packed_byte;
}

// NEW: Decode a single uint8_t byte back into 5 ternary values (0,1,2)
inline void unpack_five_ternary(uint8_t packed_byte, int8_t t_values[5]) {
    int current_val = packed_byte;
    for (int i = 0; i < 5; ++i) {
        t_values[i] = current_val % 3;
        current_val /= 3;
    }
}

// Quantizes float to int8_t
void quantize_float_to_int8_with_scale(const float* float_ptr, int8_t* int_ptr, size_t size, float fixed_scale);

// NEW: Function to build the 256x256-entry Bit-Slice LUT for 5x3bit activations and weights
// LUT will be indexed by (packed_activation_byte << 8) | packed_weight_byte
// Each entry stores the sum of 5 (decoded_act * decoded_weight) products (int16_t)
void build_bit_slice_lut_5x3(std::vector<int16_t>& precomputed_lut);

// NEW: Function to pack int8_t weights into 5x3bit packed uint8_t format
std::vector<uint8_t> pack_weights_5x3bit(const std::vector<int8_t>& unpacked_weights, int original_size);

// NEW: Function to pack int8_t ternary activations into 5x3bit packed uint8_t format
std::vector<uint8_t> pack_ternary_activations_5x3bit(const std::vector<int8_t>& unpacked_activations, int original_size);

#endif // MODEL_PREPROC_H
