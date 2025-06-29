#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <cstdint> // For uint8_t, int8_t, int16_t
#include <string>  // For std::string

// Forward declarations for structs defined in types.h
struct FloatLayer;
struct LutLayer;
struct WeightsOnlyQuantLayer;

// --- 5x3 bit 编码/解码函数 ---
// These functions encode/decode a single ternary value to/from its 3-bit representation (0, 1, 2).
uint8_t encode_ternary_to_3bit_val(int8_t val);
int8_t decode_3bit_val_to_ternary(uint8_t encoded_val);

// Packs 5 encoded ternary values (0, 1, 2) into a single uint8_t using base-3 representation.
// This version is now primarily used for building the lookup table.
uint8_t pack_five_ternary(const uint8_t t_encoded_values[5]);
// Unpacks a single uint8_t into 5 encoded ternary values (0, 1, 2).
void unpack_five_ternary(uint8_t packed_byte, uint8_t t_encoded_values[5]);

// --- Quantization/Packing/Conversion Functions ---
// Quantizes a block of float values to int8_t values with a fixed scale.
void quantize_float_to_int8_with_scale(const float* float_ptr, int8_t* int_ptr, size_t size, float fixed_scale);
// Builds the 5x3-bit look-up table for bit-slice GEMM.
void build_bit_slice_lut_5x3(std::vector<int16_t>& precomputed_lut);

// Packs unpacked int8_t weights (which are effectively ternary {-1,0,1}) into 5x3-bit packed uint8_t format.
// This version returns a new vector and is mainly for initial model packing.
std::vector<uint8_t> pack_weights_5x3bit(const std::vector<int8_t>& unpacked_weights, int original_size);

// Packs unpacked int8_t activations (which are quantized floats, needs ternarization) into 5x3-bit packed uint8_t format.
// This version writes directly to a provided output pointer, avoiding dynamic allocations per call.
void pack_ternary_activations_5x3bit_to_ptr(const int8_t* unpacked_activations_ptr, int original_size, uint8_t* output_packed_ptr);


// C++ 端用于将 int8_t 值转换为三值 (-1, 0, 1) 的阈值。
extern const int8_t C_TERNARY_ACTIVATION_THRESHOLD;

// Converts a single int8_t value to a ternary activation (-1, 0, or 1) based on a threshold.
// This is the scalar logic for activations (first quantize float to int8, then ternarize).
inline int8_t convert_int8_to_ternary_activation(int8_t val) {
    if (val > C_TERNARY_ACTIVATION_THRESHOLD) return 1;
    if (val < -C_TERNARY_ACTIVATION_THRESHOLD) return -1;
    return 0; // Values within [-THRESHOLD, THRESHOLD] map to 0
}

// NEW: Vectorized function to encode a block of already-ternary int8_t values (-1, 0, 1)
// to 3-bit encoded uint8_t (0, 1, 2). Used for weights.
void encode_int8_to_3bit_simd(const int8_t* input, uint8_t* output, size_t size);

// NEW: Vectorized function to ternarize a block of int8_t activations (quantized floats)
// and then encode them to 3-bit encoded uint8_t (0, 1, 2). Used for activations.
void ternarize_int8_to_3bit_simd(const int8_t* input, uint8_t* output, size_t size);


// --- Packing Look-Up Table (LUT) ---
// Global precomputed LUT for packing 5 3-bit ternary encoded values into a single byte.
// Size: 3^5 = 243 entries.
extern std::vector<uint8_t> g_packing_lut;

// Builds the packing lookup table once at startup.
void build_packing_lut();


// --- 激活函数 ---
void relu(float* vec_ptr, size_t size);
void log_softmax(float* vec_ptr, size_t size);

// --- 数据加载函数 ---
bool load_images_from_file(const std::string& path, std::vector<float>& images, int num_images, int image_size);
bool load_labels_from_file(const std::string& path, std::vector<int>& labels, int num_labels);
bool load_full_precision_mlp(const std::string& path, FloatLayer& layer1, FloatLayer& layer2, int input_dim_padded, int hidden_dim, int output_dim);

// --- 其他通用工具函数 ---
int argmax(const std::vector<float>& vec, int offset, int size);

#endif // UTILS_H
