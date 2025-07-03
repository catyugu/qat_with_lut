#include "utils.h" // Include the corresponding header
#include "types.h" // Include types.h for struct definitions

#include <algorithm> // For std::max, std::min
#include <cmath>     // For roundf, std::exp, std::log
#include <iostream>  // For std::cerr, std::cout (for debugging/errors in loading)
#include <fstream>   // For std::ifstream

// Include for image writing (define STB_IMAGE_WRITE_IMPLEMENTATION once in one .cpp file)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifdef __AVX512F__
#include <immintrin.h> // For AVX512 intrinsics
#endif
#if defined(__AVX2__) || defined(GGML_BITNET_X86_AVX2)
#include <immintrin.h> // For AVX2 intrinsics
#endif


const int8_t C_TERNARY_ACTIVATION_THRESHOLD = 0;
std::vector<uint8_t> g_packing_lut;


uint8_t encode_ternary_to_3bit_val(int8_t val) {
    if (val == -1) return 0;
    if (val == 0)  return 1;
    if (val == 1)  return 2;
    return 1; // Default to 0 for unexpected values
}

int8_t decode_3bit_val_to_ternary(uint8_t encoded_val) {
    if (encoded_val == 0) return -1;
    if (encoded_val == 1) return 0;
    if (encoded_val == 2) return 1;
    return 0; // Default to 0 for unexpected values
}

uint8_t pack_five_ternary(const uint8_t t_encoded_values[5]) {
    uint8_t packed_byte = 0;
    packed_byte += t_encoded_values[0] * 1;
    packed_byte += t_encoded_values[1] * 3;
    packed_byte += t_encoded_values[2] * 9;
    packed_byte += t_encoded_values[3] * 27;
    packed_byte += t_encoded_values[4] * 81;
    return packed_byte;
}

void unpack_five_ternary(uint8_t packed_byte, uint8_t t_encoded_values[5]) {
    int current_val = packed_byte;
    for (int i = 0; i < 5; ++i) {
        t_encoded_values[i] = current_val % 3;
        current_val /= 3;
    }
}

// ... (SIMD functions remain the same)
#ifdef __AVX512F__
void encode_int8_to_3bit_simd(const int8_t* input, uint8_t* output, size_t size) {
    const __m512i neg_one_val_simd = _mm512_set1_epi8(-1); // For ternary -1 -> encoded 0
    const __m512i zero_val_simd = _mm512_set1_epi8(0);    // For ternary 0 -> encoded 1
    const __m512i one_val_simd = _mm512_set1_epi8(1);     // For ternary 1 -> encoded 2

    // Target encoded values
    const __m512i encoded_zero = _mm512_set1_epi8(0);   // Target for -1
    const __m512i encoded_one = _mm512_set1_epi8(1);    // Target for 0
    const __m512i encoded_two = _mm512_set1_epi8(2);    // Target for 1

    size_t i = 0;
    for (; i + 63 < size; i += 64) {
        __m512i data = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(input + i));

        // Start with default encoded_one (ternary 0)
        __m512i result = encoded_one;

        // If data == -1 (ternary -1), set result to encoded_zero
        __mmask64 mask_neg_one = _mm512_cmpeq_epi8_mask(data, neg_one_val_simd);
        result = _mm512_mask_blend_epi8(mask_neg_one, result, encoded_zero);

        // If data == 1 (ternary 1), set result to encoded_two
        __mmask64 mask_one = _mm512_cmpeq_epi8_mask(data, one_val_simd);
        result = _mm512_mask_blend_epi8(mask_one, result, encoded_two);

        _mm512_storeu_si512(reinterpret_cast<__m512i*>(output + i), result);
    }
    // Handle remaining elements with scalar loop
    for (; i < size; ++i) {
        output[i] = encode_ternary_to_3bit_val(input[i]);
    }
}
#elif defined(__AVX2__) || defined(GGML_BITNET_X86_AVX2)
void encode_int8_to_3bit_simd(const int8_t* input, uint8_t* output, size_t size) {
    const __m256i neg_one_val_simd = _mm256_set1_epi8(-1); // For ternary -1 -> encoded 0
    const __m256i zero_val_simd = _mm256_set1_epi8(0);    // For ternary 0 -> encoded 1
    const __m256i one_val_simd = _mm256_set1_epi8(1);     // For ternary 1 -> encoded 2

    // Target encoded values
    const __m256i encoded_zero = _mm256_set1_epi8(0);   // Target for -1
    const __m256i encoded_one = _mm256_set1_epi8(1);    // Target for 0
    const __m256i encoded_two = _mm256_set1_epi8(2);     // Target for 1

    size_t i = 0;
    for (; i + 31 < size; i += 32) {
        __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + i));

        // Start with default encoded_one (ternary 0)
        __m256i result = encoded_one;

        // If data == -1 (ternary -1), set result to encoded_zero
        __m256i mask_neg_one = _mm256_cmpeq_epi8(data, neg_one_val_simd);
        result = _mm256_blendv_epi8(result, encoded_zero, mask_neg_one);

        // If data == 1 (ternary 1), set result to encoded_two
        __m256i mask_one = _mm256_cmpeq_epi8(data, one_val_simd);
        result = _mm256_blendv_epi8(result, encoded_two, mask_one);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i), result);
    }
    // Handle remaining elements with scalar loop
    for (; i < size; ++i) {
        output[i] = encode_ternary_to_3bit_val(input[i]);
    }
}
#else // Fallback scalar implementation if no SIMD intrinsics are available
void encode_int8_to_3bit_simd(const int8_t* input, uint8_t* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = encode_ternary_to_3bit_val(input[i]);
    }
}
#endif


// NEW: Vectorized function to ternarize a block of int8_t activations (quantized floats)
// and then encode them to 3-bit encoded uint8_t (0, 1, 2). Used for activations.
#ifdef __AVX512F__
void ternarize_int8_to_3bit_simd(const int8_t* input, uint8_t* output, size_t size) {
    const __m512i threshold_pos_vec = _mm512_set1_epi8(C_TERNARY_ACTIVATION_THRESHOLD);
    const __m512i threshold_neg_vec = _mm512_set1_epi8(-C_TERNARY_ACTIVATION_THRESHOLD);

    // Encoded output values for ternary -1, 0, 1
    const __m512i encoded_neg_one = _mm512_set1_epi8(0); // Ternary -1 -> encoded 0
    const __m512i encoded_zero = _mm512_set1_epi8(1);    // Ternary 0 -> encoded 1
    const __m512i encoded_one = _mm512_set1_epi8(2);     // Ternary 1 -> encoded 2

    size_t i = 0;
    for (; i + 63 < size; i += 64) {
        __m512i data = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(input + i));

        // Default result to encoded_zero (ternary 0)
        __m512i result = encoded_zero;

        // Mask for values > C_TERNARY_ACTIVATION_THRESHOLD (ternary 1, encoded as 2)
        __mmask64 mask_gt_pos = _mm512_cmpgt_epi8_mask(data, threshold_pos_vec);
        result = _mm512_mask_blend_epi8(mask_gt_pos, result, encoded_one);

        // Mask for values < -C_TERNARY_ACTIVATION_THRESHOLD (ternary -1, encoded as 0)
        __mmask64 mask_lt_neg = _mm512_cmpgt_epi8_mask(threshold_neg_vec, data); // A < B means _mm512_cmpgt_epi8(B, A)
        result = _mm512_mask_blend_epi8(mask_lt_neg, result, encoded_neg_one);

        _mm512_storeu_si512(reinterpret_cast<__m512i*>(output + i), result);
    }
    // Handle remaining elements with scalar loop
    for (; i < size; ++i) {
        int8_t ternary_val = convert_int8_to_ternary_activation(input[i]);
        output[i] = encode_ternary_to_3bit_val(ternary_val);
    }
}
#elif defined(__AVX2__) || defined(GGML_BITNET_X86_AVX2)
void ternarize_int8_to_3bit_simd(const int8_t* input, uint8_t* output, size_t size) {
    const __m256i threshold_pos_vec = _mm256_set1_epi8(C_TERNARY_ACTIVATION_THRESHOLD);
    const __m256i threshold_neg_vec = _mm256_set1_epi8(-C_TERNARY_ACTIVATION_THRESHOLD);

    // Encoded output values for ternary -1, 0, 1
    const __m256i encoded_neg_one = _mm256_set1_epi8(0); // Ternary -1 -> encoded 0
    const __m256i encoded_zero = _mm256_set1_epi8(1);    // Ternary 0 -> encoded 1
    const __m256i encoded_one = _mm256_set1_epi8(2);     // Ternary 1 -> encoded 2

    size_t i = 0;
    for (; i + 31 < size; i += 32) {
        __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + i));

        // Default result to encoded_zero (ternary 0)
        __m256i result = encoded_zero;

        // Mask for values > C_TERNARY_ACTIVATION_THRESHOLD (ternary 1, encoded as 2)
        __m256i mask_gt_pos = _mm256_cmpgt_epi8(data, threshold_pos_vec);
        result = _mm256_blendv_epi8(result, encoded_one, mask_gt_pos);

        // Mask for values < -C_TERNARY_ACTIVATION_THRESHOLD (ternary -1, encoded as 0)
        __m256i mask_lt_neg = _mm256_cmpgt_epi8(threshold_neg_vec, data); // A < B means _mm256_cmpgt_epi8(B, A)
        result = _mm256_blendv_epi8(result, encoded_neg_one, mask_lt_neg);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i), result);
    }
    // Handle remaining elements with scalar loop
    for (; i < size; ++i) {
        int8_t ternary_val = convert_int8_to_ternary_activation(input[i]);
        output[i] = encode_ternary_to_3bit_val(ternary_val);
    }
}
#else // Fallback scalar implementation if no SIMD intrinsics are available
void ternarize_int8_to_3bit_simd(const int8_t* input, uint8_t* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        int8_t ternary_val = convert_int8_to_ternary_activation(input[i]);
        output[i] = encode_ternary_to_3bit_val(ternary_val);
    }
}
#endif

void quantize_float_to_int8_with_scale(const float* float_ptr, int8_t* int_ptr, size_t size, float fixed_scale) {
    for (size_t i = 0; i < size; ++i) {
        auto val = static_cast<int32_t>(roundf(float_ptr[i] * fixed_scale));
        int_ptr[i] = static_cast<int8_t>(std::max(-128, std::min(127, val)));
    }
}

// **CORRECTED LUT BUILDER**
void build_bit_slice_lut_5x3(std::vector<int32_t>& precomputed_lut) { // Changed to int32_t
    const int LUT_DIM = 256;
    const int LUT_SIZE = LUT_DIM * LUT_DIM;
    precomputed_lut.resize(LUT_SIZE);

    uint8_t encoded_acts_5[5];
    uint8_t encoded_weights_5[5];

    for (int packed_act_byte_int = 0; packed_act_byte_int < LUT_DIM; ++packed_act_byte_int) {
        uint8_t packed_act_byte = static_cast<uint8_t>(packed_act_byte_int);
        unpack_five_ternary(packed_act_byte, encoded_acts_5);

        for (int packed_weight_byte_int = 0; packed_weight_byte_int < LUT_DIM; ++packed_weight_byte_int) {
            uint8_t packed_weight_byte = static_cast<uint8_t>(packed_weight_byte_int);
            unpack_five_ternary(packed_weight_byte, encoded_weights_5);

            int32_t current_sum_of_products = 0; // Changed to int32_t
            for (int i = 0; i < 5; ++i) {
                int8_t decoded_act_val = decode_3bit_val_to_ternary(encoded_acts_5[i]);
                int8_t decoded_weight_val = decode_3bit_val_to_ternary(encoded_weights_5[i]);
                current_sum_of_products += (int32_t)decoded_act_val * decoded_weight_val; // Cast to int32_t
            }
            precomputed_lut[(static_cast<uint32_t>(packed_act_byte) << 8) | packed_weight_byte] = current_sum_of_products;
        }
    }
}

std::vector<uint8_t> pack_weights_5x3bit(const std::vector<int8_t>& unpacked_weights, int original_size) {
    std::vector<uint8_t> packed_weights_vec;
    packed_weights_vec.reserve((original_size + 4) / 5);

    std::vector<uint8_t> encoded_unpacked_weights(original_size);
    encode_int8_to_3bit_simd(unpacked_weights.data(), encoded_unpacked_weights.data(), original_size);

    uint8_t five_ternary_encoded_vals[5];
    for(int k=0; k<5; ++k) five_ternary_encoded_vals[k] = 1;

    for (int i = 0; i < original_size; ++i) {
        int pack_idx_in_five = i % 5;
        five_ternary_encoded_vals[pack_idx_in_five] = encoded_unpacked_weights[i];

        if (pack_idx_in_five == 4) {
            packed_weights_vec.push_back(pack_five_ternary(five_ternary_encoded_vals));
        }
    }
    if (original_size % 5 != 0) {
        for(int k = original_size % 5; k < 5; ++k) {
            five_ternary_encoded_vals[k] = 1;
        }
        packed_weights_vec.push_back(pack_five_ternary(five_ternary_encoded_vals));
    }
    return packed_weights_vec;
}

void pack_ternary_activations_5x3bit_to_ptr(const int8_t* unpacked_activations_ptr, int original_size, uint8_t* output_packed_ptr) {
    std::vector<uint8_t> encoded_unpacked_activations(original_size);
    ternarize_int8_to_3bit_simd(unpacked_activations_ptr, encoded_unpacked_activations.data(), original_size);

    uint8_t five_ternary_encoded_vals[5];
    int packed_idx = 0;

    for (int i = 0; i < original_size; ++i) {
        int pack_idx_in_five = i % 5;
        five_ternary_encoded_vals[pack_idx_in_five] = encoded_unpacked_activations[i];
        if (pack_idx_in_five == 4) {
            output_packed_ptr[packed_idx++] = pack_five_ternary(five_ternary_encoded_vals);
        }
    }

    if (original_size % 5 != 0) {
        for(int k = original_size % 5; k < 5; ++k) {
            five_ternary_encoded_vals[k] = 1;
        }
        output_packed_ptr[packed_idx++] = pack_five_ternary(five_ternary_encoded_vals);
    }
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
// SiLU is now defined here.
void silu(float* vec_ptr, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        vec_ptr[i] = vec_ptr[i] * (1.0f / (1.0f + std::exp(-vec_ptr[i])));
    }
}


// --- Data Transformation Functions Implementation ---
void im2col(const float* data_im, int channels, int height, int width,
            int kernel_h, int kernel_w, int stride_h, int stride_w,
            int pad_h, int pad_w, std::vector<float>& col) {

    int output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int col_rows = channels * kernel_h * kernel_w;
    int col_cols = output_h * output_w;

    col.assign(col_rows * col_cols, 0.0f); // Resize and initialize with zeros

    for (int c = 0; c < channels; ++c) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int input_row_start = -pad_h + kh;
                for (int oh = 0; oh < output_h; ++oh) {
                    int input_col_start = -pad_w + kw;
                    for (int ow = 0; ow < output_w; ++ow) {
                        int col_index = (c * kernel_h * kernel_w + kh * kernel_w + kw) * (output_h * output_w) + oh * output_w + ow;
                        int input_h_idx = input_row_start + oh * stride_h;
                        int input_w_idx = input_col_start + ow * stride_w;

                        if (input_h_idx >= 0 && input_h_idx < height && input_w_idx >= 0 && input_w_idx < width) {
                            // Convert NCHW to NHWC for input indexing if needed, but here it's flat NCHW
                            col[col_index] = data_im[c * height * width + input_h_idx * width + input_w_idx];
                        } else {
                            col[col_index] = 0; // Padding (already initialized to 0, but explicit for clarity)
                        }
                    }
                }
            }
        }
    }
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

// --- Image Saving Function Implementation ---
// Saves a float array (NCHW format, values in [-1, 1]) as a PNG image.
bool save_image_from_float_array(const std::string& filename, const std::vector<float>& image_data,
                                 int channels, int height, int width) {
    if (image_data.empty() || channels <= 0 || height <= 0 || width <= 0) {
        std::cerr << "Error: Invalid image data for saving." << std::endl;
        return false;
    }

    // Convert float image data [-1, 1] to uint8_t [0, 255]
    // The input image_data is assumed to be in NCHW format (C, H, W for a single image)
    // stb_image_write expects HWC (Height, Width, Channels) format.
    std::vector<unsigned char> reordered_pixel_data(height * width * channels);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                // Calculate original NCHW index
                size_t original_idx = c * (height * width) + y * width + x;
                // Calculate new HWC index
                size_t new_idx = y * (width * channels) + x * channels + c;

                // Scale from [-1, 1] to [0, 1] then to [0, 255]
                float val_0_1 = (image_data[original_idx] + 1.0f) / 2.0f;
                reordered_pixel_data[new_idx] = static_cast<unsigned char>(std::round(val_0_1 * 255.0f));
            }
        }
    }
    
    // For 3-channel images (RGB), stb_image_write_png works directly.
    // For 1-channel images (grayscale), stb_image_write_png also works.
    int stride_bytes = width * channels; // bytes per row

    if (stbi_write_png(filename.c_str(), width, height, channels, reordered_pixel_data.data(), stride_bytes)) {
        return true;
    } else {
        std::cerr << "Error: Failed to write PNG image to " << filename << std::endl;
        return false;
    }
}


// --- 其他通用工具函数实现 ---
int argmax(const std::vector<float>& vec, int offset, int size) {
    if (size == 0) return -1;
    return static_cast<int>(std::distance(vec.begin() + offset, std::max_element(vec.begin() + offset, vec.begin() + offset + size)));
}

