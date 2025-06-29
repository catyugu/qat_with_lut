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
uint8_t encode_ternary_to_3bit_val(int8_t val);
int8_t decode_3bit_val_to_ternary(uint8_t encoded_val);
uint8_t pack_five_ternary(const int8_t t_values[5]);
void unpack_five_ternary(uint8_t packed_byte, int8_t t_values[5]);

// --- 量化/打包/转换函数 ---
void quantize_float_to_int8_with_scale(const float* float_ptr, int8_t* int_ptr, size_t size, float fixed_scale);
void build_bit_slice_lut_5x3(std::vector<int16_t>& precomputed_lut);
std::vector<uint8_t> pack_weights_5x3bit(const std::vector<int8_t>& unpacked_weights, int original_size);
std::vector<uint8_t> pack_ternary_activations_5x3bit(const std::vector<int8_t>& unpacked_activations, int original_size);

// C++ 端用于将 int8_t 值转换为三值 (-1, 0, 1) 的阈值。
extern const int8_t C_TERNARY_ACTIVATION_THRESHOLD;
inline int8_t convert_int8_to_ternary_activation(int8_t val) {
    if (val > C_TERNARY_ACTIVATION_THRESHOLD) return 1;
    if (val < -C_TERNARY_ACTIVATION_THRESHOLD) return -1;
    return 0; // Values within [-THRESHOLD, THRESHOLD] map to 0
}

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
