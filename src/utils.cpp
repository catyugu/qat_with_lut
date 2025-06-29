#include "utils.h" // Include the corresponding header
#include "types.h" // Include types.h for struct definitions

#include <algorithm> // For std::max, std::min
#include <cmath>     // For roundf, std::exp, std::log
#include <iostream>  // For std::cerr, std::cout (for debugging/errors in loading)
#include <fstream>   // For std::ifstream

// C++ 端用于将 int8_t 值转换为三值 (-1, 0, 1) 的阈值。
// 对应 Python 中 TERNARY_THRESHOLD (0.01) * 127.0 = 1.27, 四舍五入为 1
const int8_t C_TERNARY_ACTIVATION_THRESHOLD = 1;

// --- 5x3 bit 编码/解码函数实现 ---
uint8_t encode_ternary_to_3bit_val(int8_t val) {
    if (val == -1) return 0;
    if (val == 0)  return 1;
    if (val == 1)  return 2;
    return 1; // Default to 0 (encoded as 1)
}

int8_t decode_3bit_val_to_ternary(uint8_t encoded_val) {
    if (encoded_val == 0) return -1;
    if (encoded_val == 1) return 0;
    if (encoded_val == 2) return 1;
    return 0; // Default for invalid codes
}

uint8_t pack_five_ternary(const int8_t t_values[5]) {
    uint8_t packed_byte = 0;
    packed_byte += t_values[0] * 1;
    packed_byte += t_values[1] * 3;
    packed_byte += t_values[2] * 9;
    packed_byte += t_values[3] * 27;
    packed_byte += t_values[4] * 81;
    return packed_byte;
}

void unpack_five_ternary(uint8_t packed_byte, int8_t t_values[5]) {
    int current_val = packed_byte;
    for (int i = 0; i < 5; ++i) {
        t_values[i] = current_val % 3;
        current_val /= 3;
    }
}

// --- 量化/打包/转换函数实现 ---
void quantize_float_to_int8_with_scale(const float* float_ptr, int8_t* int_ptr, size_t size, float fixed_scale) {
    for (size_t i = 0; i < size; ++i) {
        auto val = static_cast<int32_t>(roundf(float_ptr[i] * fixed_scale));
        int_ptr[i] = static_cast<int8_t>(std::max(-128, std::min(127, val)));
    }
}

void build_bit_slice_lut_5x3(std::vector<int16_t>& precomputed_lut) {
    const int LUT_DIM = 256; // 2^8 possibilities for packed byte
    const int LUT_SIZE = LUT_DIM * LUT_DIM; // 256 * 256 = 65536 entries
    precomputed_lut.resize(LUT_SIZE);

    uint8_t encoded_acts_5[5];
    uint8_t encoded_weights_5[5];

    for (int packed_act_byte_int = 0; packed_act_byte_int < LUT_DIM; ++packed_act_byte_int) {
        uint8_t packed_act_byte = static_cast<uint8_t>(packed_act_byte_int);
        unpack_five_ternary(packed_act_byte, reinterpret_cast<int8_t*>(encoded_acts_5));

        for (int packed_weight_byte_int = 0; packed_weight_byte_int < LUT_DIM; ++packed_weight_byte_int) {
            uint8_t packed_weight_byte = static_cast<uint8_t>(packed_weight_byte_int);
            unpack_five_ternary(packed_weight_byte, reinterpret_cast<int8_t*>(encoded_weights_5));

            int16_t current_sum_of_products = 0;
            for (int i = 0; i < 5; ++i) {
                int8_t decoded_act_val = decode_3bit_val_to_ternary(encoded_acts_5[i]);
                int8_t decoded_weight_val = decode_3bit_val_to_ternary(encoded_weights_5[i]);
                current_sum_of_products += (int16_t)decoded_act_val * decoded_weight_val;
            }
            precomputed_lut[(static_cast<uint32_t>(packed_act_byte) << 8) | packed_weight_byte] = current_sum_of_products;
        }
    }
}

std::vector<uint8_t> pack_weights_5x3bit(const std::vector<int8_t>& unpacked_weights, int original_size) {
    std::vector<uint8_t> packed_weights_vec;
    packed_weights_vec.reserve((original_size + 4) / 5);

    int8_t five_ternary_vals[5];
    for(int k=0; k<5; ++k) five_ternary_vals[k] = encode_ternary_to_3bit_val(0);

    for (int i = 0; i < original_size; ++i) {
        int pack_idx_in_five = i % 5;
        five_ternary_vals[pack_idx_in_five] = encode_ternary_to_3bit_val(unpacked_weights[i]);

        if (pack_idx_in_five == 4) {
            packed_weights_vec.push_back(pack_five_ternary(five_ternary_vals));
             for(int k=0; k<5; ++k) five_ternary_vals[k] = encode_ternary_to_3bit_val(0);
        }
    }
    if (original_size % 5 != 0) {
        packed_weights_vec.push_back(pack_five_ternary(five_ternary_vals));
    }
    return packed_weights_vec;
}

std::vector<uint8_t> pack_ternary_activations_5x3bit(const std::vector<int8_t>& unpacked_activations, int original_size) {
    std::vector<uint8_t> packed_activations_vec;
    packed_activations_vec.reserve((original_size + 4) / 5);

    int8_t five_ternary_vals[5];
    for(int k=0; k<5; ++k) five_ternary_vals[k] = encode_ternary_to_3bit_val(0);

    for (int i = 0; i < original_size; ++i) {
        int pack_idx_in_five = i % 5;
        five_ternary_vals[pack_idx_in_five] = encode_ternary_to_3bit_val(unpacked_activations[i]);

        if (pack_idx_in_five == 4) {
            packed_activations_vec.push_back(pack_five_ternary(five_ternary_vals));
            for(int k=0; k<5; ++k) five_ternary_vals[k] = encode_ternary_to_3bit_val(0);
        }
    }
    if (original_size % 5 != 0) {
        packed_activations_vec.push_back(pack_five_ternary(five_ternary_vals));
    }
    return packed_activations_vec;
}

// --- 激活函数实现 ---
void relu(float* vec_ptr, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        vec_ptr[i] = std::max(0.0f, vec_ptr[i]);
    }
}
void log_softmax(float* vec_ptr, size_t size) {
    if (size == 0) return;
    float max_val = vec_ptr[0];
    for (size_t i = 0; i < size; ++i) if (vec_ptr[i] > max_val) max_val = vec_ptr[i];
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) sum += std::exp(vec_ptr[i] - max_val);
    float log_sum = std::log(sum);
    for (size_t i = 0; i < size; ++i) vec_ptr[i] = (vec_ptr[i] - max_val) - log_sum;
}

// --- 数据加载函数实现 ---
bool load_images_from_file(const std::string& path, std::vector<float>& images, int num_images, int image_size) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open image file: " << path << std::endl;
        return false;
    }
    images.resize(num_images * image_size);
    file.read(reinterpret_cast<char*>(images.data()), images.size() * sizeof(float));
    return file.good();
}

bool load_labels_from_file(const std::string& path, std::vector<int>& labels, int num_labels) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open label file: " << path << std::endl;
        return false;
    }
    labels.resize(num_labels);
    file.read(reinterpret_cast<char*>(labels.data()), labels.size() * sizeof(int));
    return file.good();
}

bool load_full_precision_mlp(const std::string& path, FloatLayer& layer1, FloatLayer& layer2, int input_dim_padded, int hidden_dim, int output_dim) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open full precision model file: " << path << std::endl;
        return false;
    }
    char magic[4]; file.read(magic, 4);
    if (std::string(magic, 4) != "MLPF") { // Check for 'MLPF' magic
        std::cerr << "Error: Invalid full precision model file format. Expected 'MLPF'." << std::endl;
        return false;
    }
    int file_input_dim, file_hidden_dim, file_output_dim;
    file.read(reinterpret_cast<char*>(&file_input_dim), sizeof(int));
    file.read(reinterpret_cast<char*>(&file_hidden_dim), sizeof(int));
    file.read(reinterpret_cast<char*>(&file_output_dim), sizeof(int));

    if (file_input_dim != 784 || file_hidden_dim != hidden_dim || file_output_dim != output_dim) {
        std::cerr << "Error: Full precision model dimensions mismatch. Expected " << 784 << "x" << hidden_dim << "x" << output_dim
                  << ", got " << file_input_dim << "x" << file_hidden_dim << "x" << file_output_dim << std::endl;
        return false;
    }

    layer1.weights.resize(file_input_dim * hidden_dim);
    file.read(reinterpret_cast<char*>(layer1.weights.data()), layer1.weights.size() * sizeof(float));
    layer1.bias.resize(hidden_dim);
    file.read(reinterpret_cast<char*>(layer1.bias.data()), layer1.bias.size() * sizeof(float));

    float dummy_scale; // Skip activation scales (2 floats)
    file.read(reinterpret_cast<char*>(&dummy_scale), sizeof(float));
    file.read(reinterpret_cast<char*>(&dummy_scale), sizeof(float));

    layer2.weights.resize(hidden_dim * output_dim);
    file.read(reinterpret_cast<char*>(layer2.weights.data()), layer2.weights.size() * sizeof(float));
    layer2.bias.resize(output_dim);
    file.read(reinterpret_cast<char*>(layer2.bias.data()), layer2.bias.size() * sizeof(float));

    std::vector<float> temp_w1_float = layer1.weights; // Copy original 784x320 weights
    layer1.weights.assign(hidden_dim * input_dim_padded, 0.0f); // Resize to padded, init to 0
    for(int i = 0; i < hidden_dim; ++i) {
        std::copy(temp_w1_float.begin() + i * file_input_dim,
                  temp_w1_float.begin() + (i + 1) * file_input_dim,
                  layer1.weights.begin() + i * input_dim_padded);
    }
    return file.good();
}

// --- 其他通用工具函数实现 ---
int argmax(const std::vector<float>& vec, int offset, int size) {
    if (size == 0) return -1;
    return static_cast<int>(std::distance(vec.begin() + offset, std::max_element(vec.begin() + offset, vec.begin() + offset + size)));
}
