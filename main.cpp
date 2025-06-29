#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <chrono>   // For timing
#include <iomanip>  // For formatting the output table
#include <numeric>  // For std::iota and std::accumulate

// Revert to AVX2 definition
#define __AVX2__
#ifdef __AVX2__
#include <immintrin.h> // For AVX2 intrinsics
#endif

// Include the new bit-slice kernel and preprocessor headers
#include "bit_slice_kernels.h" // This header now declares avx2_bit_slice_gemm_kernel
#include "model_preproc.h"

// --- Data Structures for our Models ---
struct FloatLayer {
    std::vector<float> weights;
    std::vector<float> bias;
};

// Our LUT-based layer will use packed quantized weights and an activation scale
struct LutLayer {
    std::vector<uint8_t> packed_weights; // Changed to uint8_t for packed 5x3bit weights
    std::vector<float> bias;
    float activation_scale = 1.0f; // Scale for the *input* activations to this layer
};

// Struct for Weights-Only Quantization (unpacked int8_t ternary weights)
struct WeightsOnlyQuantLayer {
    std::vector<int8_t> weights; // Unpacked ternary weights {-1, 0, 1}
    std::vector<float> bias;
    float activation_scale = 1.0f; // Scale for input activations
};


// C++ 端用于将 int8_t 值转换为三值 (-1, 0, 1) 的阈值。
// 对应 Python 中 TERNARY_THRESHOLD (0.01) * 127.0 = 1.27, 四舍五入为 1
const int8_t C_TERNARY_ACTIVATION_THRESHOLD = 1;

inline int8_t convert_int8_to_ternary_activation(int8_t val) {
    if (val > C_TERNARY_ACTIVATION_THRESHOLD) return 1;
    if (val < -C_TERNARY_ACTIVATION_THRESHOLD) return -1;
    return 0; // Values within [-THRESHOLD, THRESHOLD] map to 0
}


#ifdef __AVX2__
// Horizontal sum of 8 int32_t elements in a __m256i vector (adapted from ggml-bitnet-mad.cpp)
static inline int hsum_i32_8(const __m256i a) {
    // Extract lower 128 bits and higher 128 bits, then add them
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    // Unpack high 64 bits to lower 64 bits and add to original
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    // Add sum64 and hi32, then extract the lowest 32-bit integer (which now contains the total sum)
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}
#endif

// The high-performance LUT-based forward pass (pure integer math)
// Optimized for ternary weights using AVX2 intrinsics.
// This version handles a batch of inputs and uses the global LUT.
void lut_linear_forward(
    const LutLayer& layer,
    const std::vector<uint8_t>& input_packed_batched, // Changed to uint8_t for packed activations
    std::vector<float>& output_f32_batched,
    int input_dim,  // This is the dimension in original elements, needed for looping
    int output_dim,
    int batch_size,
    const int16_t* precomputed_lut_ptr // Pass the global LUT pointer
) {
    output_f32_batched.resize(batch_size * output_dim);

    // Pointers to raw data for efficient access
    const uint8_t* weights_packed_ptr = layer.packed_weights.data(); // Access packed weights

    // Number of packed bytes per input sample
    const int packed_input_bytes_per_sample = (input_dim + 4) / 5;

    // Iterate over each item in the batch
    for (int b = 0; b < batch_size; ++b) {
        const uint8_t* current_batch_input_packed_ptr = input_packed_batched.data() + b * packed_input_bytes_per_sample;

        // Loop over each output neuron (rows of the weight matrix)
        for (int i = 0; i < output_dim; ++i) {
            // Get pointer to the packed weights for the current output neuron
            const uint8_t* current_neuron_weights_packed_ptr = weights_packed_ptr + i * packed_input_bytes_per_sample;

            // Call the AVX2 specific kernel
#ifdef __AVX2__
            int32_t sum = avx2_bit_slice_gemm_kernel( // Call AVX2 specific kernel
                current_batch_input_packed_ptr,
                current_neuron_weights_packed_ptr,
                precomputed_lut_ptr,
                input_dim // Pass the original (padded) dimension. Kernel uses this to know total elements.
            );
#else
            // Fallback to scalar (non-AVX) if AVX2 is not enabled
            int32_t sum = 0;
            // A simple scalar fallback for testing (not optimized)
            for (int k_idx = 0; k_idx < input_dim; k_idx += 5) {
                uint8_t packed_act = current_batch_input_packed_ptr[k_idx / 5];
                uint8_t packed_weight = current_neuron_weights_packed_ptr[k_idx / 5];
                sum += precomputed_lut_ptr[(static_cast<uint32_t>(packed_act) << 8) | packed_weight];
            }
#endif

            // Dequantize the final integer sum back to float and add the bias
            output_f32_batched[b * output_dim + i] = (static_cast<float>(sum) / layer.activation_scale) + layer.bias[i];
        }
    }
}

// Forward pass for Weights-Only Quantization (int8_t activations, int8_t ternary weights)
// This uses unpacked int8_t weights and does not use the 5x3 bit packing or the LUT.
void weights_only_linear_forward(const WeightsOnlyQuantLayer& layer, const std::vector<int8_t>& input_i8_batched, std::vector<float>& output_f32_batched, int input_dim, int output_dim, int batch_size) {
    output_f32_batched.resize(batch_size * output_dim);
    const int8_t* weights_ptr = layer.weights.data();

    for (int b = 0; b < batch_size; ++b) {
        const int8_t* current_batch_input_ptr = input_i8_batched.data() + b * input_dim;
        for (int i = 0; i < output_dim; ++i) {
            int32_t sum = 0;
            const int8_t* current_neuron_weights_ptr = weights_ptr + i * input_dim;

#ifdef __AVX2__
            __m256i accumulated_sum_vec_32 = _mm256_setzero_si256();

            // Process input and weights in chunks of 32 int8_t elements (one __m256i register)
            // Assumes input_dim is a multiple of 32 for optimal vectorization.
            for (int j = 0; j < input_dim; j += 32) {
                __m256i input_vals = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(current_batch_input_ptr + j));
                __m256i weights_block = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(current_neuron_weights_ptr + j));

                // Perform ternary multiplication: _mm256_sign_epi8(a, b) computes a * sign(b)
                // Since weights are -1, 0, 1, this directly gives the product.
                __m256i partial_products_i8 = _mm256_sign_epi8(input_vals, weights_block);

                // Convert 8-bit partial products to 16-bit for accumulation
                // Extract lower and upper 128-bit lanes and sign-extend to 16-bit.
                __m256i partial_sums_lo_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(partial_products_i8, 0));
                __m256i partial_sums_hi_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(partial_products_i8, 1));

                // Accumulate 16-bit sums into 32-bit sum vector
                // _mm256_madd_epi16 performs pairwise products and horizontal sum of pairs.
                // We want to sum all 16-bit values. A common way is to multiply by 1 and sum.
                accumulated_sum_vec_32 = _mm256_add_epi32(accumulated_sum_vec_32, _mm256_madd_epi16(partial_sums_lo_16, _mm256_set1_epi16(1)));
                accumulated_sum_vec_32 = _mm256_add_epi32(accumulated_sum_vec_32, _mm256_madd_epi16(partial_sums_hi_16, _mm256_set1_epi16(1)));
            }
            sum = hsum_i32_8(accumulated_sum_vec_32); // Use AVX2 horizontal sum
#else // Fallback to scalar (non-AVX) if AVX2 is not enabled
            for (int j = 0; j < input_dim; ++j) {
                if (current_neuron_weights_ptr[j] == 1) {
                    sum += current_batch_input_ptr[j];
                } else if (current_neuron_weights_ptr[j] == -1) {
                    sum -= current_batch_input_ptr[j];
                }
            }
#endif
            output_f32_batched[b * output_dim + i] = (static_cast<float>(sum) / layer.activation_scale) + layer.bias[i];
        }
    }
}


// The standard float-based forward pass for comparison (now also batched)
void standard_linear_forward(const FloatLayer& layer, const std::vector<float>& input_f32_batched, std::vector<float>& output_f32_batched, int input_dim, int output_dim, int batch_size) {
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

// Activation functions
// Now take pointer and size to avoid vector slicing overhead
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

// Helper to find the index of the maximum value in a vector (for a single output)
int argmax(const std::vector<float>& vec, int offset, int size) {
    if (size == 0) return -1;
    return static_cast<int>(std::distance(vec.begin() + offset, std::max_element(vec.begin() + offset, vec.begin() + offset + size)));
}

// Function to load image data from a binary file
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

// Function to load labels from a binary file
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

// Function to load full precision MLP from binary file
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

    // Load Layer 1 weights and bias
    layer1.weights.resize(file_input_dim * hidden_dim);
    file.read(reinterpret_cast<char*>(layer1.weights.data()), layer1.weights.size() * sizeof(float));
    layer1.bias.resize(hidden_dim);
    file.read(reinterpret_cast<char*>(layer1.bias.data()), layer1.bias.size() * sizeof(float));

    // Skip activation scales (2 floats)
    float dummy_scale;
    file.read(reinterpret_cast<char*>(&dummy_scale), sizeof(float));
    file.read(reinterpret_cast<char*>(&dummy_scale), sizeof(float));

    // Load Layer 2 weights and bias
    layer2.weights.resize(hidden_dim * output_dim);
    file.read(reinterpret_cast<char*>(layer2.weights.data()), layer2.weights.size() * sizeof(float));
    layer2.bias.resize(output_dim);
    file.read(reinterpret_cast<char*>(layer2.bias.data()), layer2.bias.size() * sizeof(float));

    // Handle padding for input_dim_padded if necessary for the float model.
    // The loaded weights from mlp_model_float.bin are for original 784 dim.
    // We need to expand them to 960 with zeros.
    std::vector<float> temp_w1_float = layer1.weights; // Copy original 784x320 weights
    layer1.weights.assign(hidden_dim * input_dim_padded, 0.0f); // Resize to padded, init to 0
    for(int i = 0; i < hidden_dim; ++i) {
        std::copy(temp_w1_float.begin() + i * file_input_dim,
                  temp_w1_float.begin() + (i + 1) * file_input_dim,
                  layer1.weights.begin() + i * input_dim_padded);
    }

    return file.good();
}


int main() {
    try {
        // --- Dimensions from the QAT model file ---
        // These dimensions will be used consistently across all models for comparison
        int input_dim_original = 784; // Fixed original input dimension
        int hidden_dim = 320;
        int output_dim = 10;
        const int input_dim_padded = 960; // Fixed padded input dimension

        if (input_dim_original > input_dim_padded) {
             throw std::runtime_error("Original input dimension is larger than padded dimension.");
        }
        if (input_dim_padded % 160 != 0) {
            throw std::runtime_error("Padded input dimension must be a multiple of 160 for AVX2 kernel.");
        }
        if (hidden_dim % 160 != 0) {
            throw std::runtime_error("Hidden dimension must be a multiple of 160 for AVX2 kernel.");
        }


        // --- 1. Load Model Weights for LUT and Weights-Only Quantized MLPs ---
        // These are loaded from mlp_model_aq.bin (QAT trained model)
        std::ifstream qat_model_file("mlp_model_aq.bin", std::ios::binary);
        if (!qat_model_file) { throw std::runtime_error("Failed to open mlp_model_aq.bin. Please run convert.py first."); }
        char magic_qat[4]; qat_model_file.read(magic_qat, 4);
        if (std::string(magic_qat, 4) != "MLP3") { throw std::runtime_error("Invalid QAT model file format."); }
        int qat_input_dim, qat_hidden_dim, qat_output_dim;
        qat_model_file.read(reinterpret_cast<char*>(&qat_input_dim), sizeof(int));
        qat_model_file.read(reinterpret_cast<char*>(&qat_hidden_dim), sizeof(int));
        qat_model_file.read(reinterpret_cast<char*>(&qat_output_dim), sizeof(int));

        // Read Layer 1 (fc1) weights (int8_t from file)
        std::vector<int8_t> w1_int8_unpacked(qat_input_dim * qat_hidden_dim);   qat_model_file.read(reinterpret_cast<char*>(w1_int8_unpacked.data()), w1_int8_unpacked.size() * sizeof(int8_t));
        std::vector<float> b1(qat_hidden_dim);                    qat_model_file.read(reinterpret_cast<char*>(b1.data()), b1.size() * sizeof(float));

        // Read Activation Scales
        float input_to_fc1_scale;                             qat_model_file.read(reinterpret_cast<char*>(&input_to_fc1_scale), sizeof(float)); // Scale for input to first layer
        float input_to_fc2_scale;                             qat_model_file.read(reinterpret_cast<char*>(&input_to_fc2_scale), sizeof(float)); // Scale for input to second layer

        // Read Layer 2 (fc2) weights (int8_t from file)
        std::vector<int8_t> w2_int8_unpacked(qat_hidden_dim * qat_output_dim); qat_model_file.read(reinterpret_cast<char*>(w2_int8_unpacked.data()), w2_int8_unpacked.size() * sizeof(int8_t));
        std::vector<float> b2(qat_output_dim);                    qat_model_file.read(reinterpret_cast<char*>(b2.data()), b2.size() * sizeof(float));

        std::cout << "QAT Model (mlp_model_aq.bin) loaded successfully." << std::endl;
        std::cout << "QAT Model Dims: Input " << qat_input_dim << ", Hidden " << qat_hidden_dim << ", Output " << qat_output_dim << std::endl;


        // --- 2. Load Full Precision Float MLP --- (NEW)
        FloatLayer fp_layer1, fp_layer2; // Using fp_ prefix for Full Precision
        if (!load_full_precision_mlp("mlp_model_float.bin", fp_layer1, fp_layer2, input_dim_padded, hidden_dim, output_dim)) {
            throw std::runtime_error("Failed to load full precision float model.");
        }
        std::cout << "Full Precision Float Model (mlp_model_float.bin) loaded successfully." << std::endl;


        // --- 3. Prepare All Models for Inference ---
        // 3a. LUT Quantized MLP Model
        LutLayer q_layer1, q_layer2;

        q_layer1.packed_weights.reserve(hidden_dim * ((input_dim_padded + 4) / 5));
        for (int i = 0; i < hidden_dim; ++i) {
            std::vector<int8_t> current_row_unpacked(input_dim_padded, 0);
            std::copy(w1_int8_unpacked.begin() + i * qat_input_dim,
                      w1_int8_unpacked.begin() + (i + 1) * qat_input_dim,
                      current_row_unpacked.begin());
            std::vector<uint8_t> packed_row = pack_weights_5x3bit(current_row_unpacked, input_dim_padded);
            q_layer1.packed_weights.insert(q_layer1.packed_weights.end(), packed_row.begin(), packed_row.end());
        }
        q_layer1.bias = b1;
        q_layer1.activation_scale = input_to_fc1_scale;

        q_layer2.packed_weights.reserve(output_dim * ((hidden_dim + 4) / 5));
        for (int i = 0; i < output_dim; ++i) {
            std::vector<int8_t> current_row_unpacked(hidden_dim);
            std::copy(w2_int8_unpacked.begin() + i * hidden_dim,
                      w2_int8_unpacked.begin() + (i + 1) * hidden_dim,
                      current_row_unpacked.begin());
            std::vector<uint8_t> packed_row = pack_weights_5x3bit(current_row_unpacked, hidden_dim);
            q_layer2.packed_weights.insert(q_layer2.packed_weights.end(), packed_row.begin(), packed_row.end());
        }
        q_layer2.bias = b2;
        q_layer2.activation_scale = input_to_fc2_scale;

        // --- NEW: Global Precomputed LUT ---
        std::vector<int16_t> precomputed_bit_slice_lut;
        build_bit_slice_lut_5x3(precomputed_bit_slice_lut); // Build the LUT once at startup
        std::cout << "Bit-Slice LUT built. Size: " << precomputed_bit_slice_lut.size() * sizeof(int16_t) / 1024.0 << " KB" << std::endl;


        // 3b. Weights-Only Quant MLP Model (unpacked ternary weights, int8_t activations)
        WeightsOnlyQuantLayer wo_q_layer1, wo_q_layer2;

        wo_q_layer1.weights.resize(hidden_dim * input_dim_padded, 0);
        for(int i = 0; i < hidden_dim; ++i) {
            std::copy(w1_int8_unpacked.begin() + i * qat_input_dim,
                      w1_int8_unpacked.begin() + (i + 1) * qat_input_dim,
                      wo_q_layer1.weights.begin() + i * input_dim_padded);
        }
        wo_q_layer1.bias = b1;
        wo_q_layer1.activation_scale = input_to_fc1_scale;

        wo_q_layer2.weights = w2_int8_unpacked;
        wo_q_layer2.bias = b2;
        wo_q_layer2.activation_scale = input_to_fc2_scale;


        // (Original) Standard Float MLP - THIS IS NOW THE TRUE FULL PRECISION MLP
        // The `f_layer1`, `f_layer2` will now hold the weights from `mlp_model_float.bin`
        // They were already correctly loaded into `fp_layer1`, `fp_layer2` above.
        // So simply assign them:
        FloatLayer& f_layer1 = fp_layer1;
        FloatLayer& f_layer2 = fp_layer2;


        // --- 4. Load FashionMNIST Test Data ---
        const int NUM_TEST_IMAGES = 10000; // FashionMNIST test set size
        std::vector<float> test_images_f32;
        std::vector<int> test_labels;

        std::cout << "\nLoading FashionMNIST test data..." << std::endl;
        if (!load_images_from_file("test_images.bin", test_images_f32, NUM_TEST_IMAGES, input_dim_original)) {
            throw std::runtime_error("Could not load test_images.bin. Make sure it's in the same directory.");
        }
        if (!load_labels_from_file("test_labels.bin", test_labels, NUM_TEST_IMAGES)) {
            throw std::runtime_error("Could not load test_labels.bin. Make sure it's in the same directory.");
        }
        std::cout << "Successfully loaded " << NUM_TEST_IMAGES << " test images and labels." << std::endl;

        // --- Batching Configuration ---
        const int BATCH_SIZE = 64; // Example batch size.
        int num_batches = (NUM_TEST_IMAGES + BATCH_SIZE - 1) / BATCH_SIZE;

        // --- 5. Performance and Accuracy Evaluation ---
        int correct_quant_lut = 0;
        int correct_quant_wo = 0;
        int correct_float_fp = 0; // Renamed for Full Precision Float

        std::chrono::duration<double, std::milli> total_quant_lut_duration(0.0);
        std::chrono::duration<double, std::milli> total_quant_wo_duration(0.0);
        std::chrono::duration<double, std::milli> total_float_fp_duration(0.0); // Renamed

        std::cout << "\nStarting batched inference for " << NUM_TEST_IMAGES << " test images (Batch Size: " << BATCH_SIZE << ")..." << std::endl;

        // Vectors for batched inputs and outputs (declared once with max capacity)
        std::vector<float> batch_input_f32(BATCH_SIZE * input_dim_padded);

        // Buffers for LUT Quantized MLP path
        std::vector<int8_t> batch_input_i8_temp_lut(BATCH_SIZE * input_dim_padded);
        std::vector<int8_t> batch_input_ternary_L1_lut(BATCH_SIZE * input_dim_padded);
        std::vector<uint8_t> batch_input_packed_activations_L1_lut(BATCH_SIZE * ((input_dim_padded + 4) / 5));
        std::vector<float> batch_hidden_q_f32_lut(BATCH_SIZE * hidden_dim);
        std::vector<int8_t> batch_hidden_i8_temp_lut(BATCH_SIZE * hidden_dim);
        std::vector<uint8_t> batch_hidden_packed_activations_L2_lut(BATCH_SIZE * ((hidden_dim + 4) / 5));
        std::vector<float> batch_final_q_lut(BATCH_SIZE * output_dim);

        // Buffers for Weights-Only Quant MLP path
        std::vector<int8_t> batch_input_i8_wo(BATCH_SIZE * input_dim_padded);
        std::vector<float> batch_hidden_q_f32_wo(BATCH_SIZE * hidden_dim);
        std::vector<int8_t> batch_hidden_i8_wo(BATCH_SIZE * hidden_dim);
        std::vector<float> batch_final_q_wo(BATCH_SIZE * output_dim);

        // Buffers for Full Precision Float MLP path
        std::vector<float> batch_hidden_f_fp(BATCH_SIZE * hidden_dim);
        std::vector<float> batch_final_f_fp(BATCH_SIZE * output_dim);


        for (int b_idx = 0; b_idx < num_batches; ++b_idx) {
            int current_batch_actual_size = std::min(BATCH_SIZE, NUM_TEST_IMAGES - b_idx * BATCH_SIZE);

            // Copy current batch of images from test_images_f32 into batch_input_f32
            // And pad to input_dim_padded. This buffer is shared across all models.
            for (int k = 0; k < current_batch_actual_size; ++k) {
                std::copy(test_images_f32.begin() + (b_idx * BATCH_SIZE + k) * input_dim_original,
                          test_images_f32.begin() + (b_idx * BATCH_SIZE + k + 1) * input_dim_original,
                          batch_input_f32.begin() + k * input_dim_padded);
                std::fill(batch_input_f32.begin() + k * input_dim_padded + input_dim_original,
                          batch_input_f32.begin() + (k + 1) * input_dim_padded,
                          0.0f);
            }

            // --- LUT Quantized Model Inference for Batch ---
            auto start_quant_lut = std::chrono::high_resolution_clock::now();

            // L1 input: float -> int8_t -> ternary -> pack
            for (int k = 0; k < current_batch_actual_size; ++k) {
                quantize_float_to_int8_with_scale(batch_input_f32.data() + k * input_dim_padded, batch_input_i8_temp_lut.data() + k * input_dim_padded, input_dim_padded, q_layer1.activation_scale);
                for (int l = 0; l < input_dim_padded; ++l) {
                    batch_input_ternary_L1_lut[k * input_dim_padded + l] = convert_int8_to_ternary_activation(batch_input_i8_temp_lut[k * input_dim_padded + l]);
                }
            }
            for (int k = 0; k < current_batch_actual_size; ++k) {
                std::vector<uint8_t> packed_act_sample = pack_ternary_activations_5x3bit(std::vector<int8_t>(batch_input_ternary_L1_lut.begin() + k * input_dim_padded, batch_input_ternary_L1_lut.begin() + (k + 1) * input_dim_padded), input_dim_padded);
                std::copy(packed_act_sample.begin(), packed_act_sample.end(), batch_input_packed_activations_L1_lut.begin() + k * ((input_dim_padded + 4) / 5));
            }
            lut_linear_forward(q_layer1, batch_input_packed_activations_L1_lut, batch_hidden_q_f32_lut, input_dim_padded, hidden_dim, current_batch_actual_size, precomputed_bit_slice_lut.data());

            // L2 input: float -> int8_t (no second ternary conversion needed) -> pack
            for (int k = 0; k < current_batch_actual_size; ++k) {
                quantize_float_to_int8_with_scale(batch_hidden_q_f32_lut.data() + k * hidden_dim, batch_hidden_i8_temp_lut.data() + k * hidden_dim, hidden_dim, q_layer2.activation_scale);
            }
            for (int k = 0; k < current_batch_actual_size; ++k) {
                 std::vector<uint8_t> packed_act_sample = pack_ternary_activations_5x3bit(std::vector<int8_t>(batch_hidden_i8_temp_lut.begin() + k * hidden_dim, batch_hidden_i8_temp_lut.begin() + (k + 1) * hidden_dim), hidden_dim);
                std::copy(packed_act_sample.begin(), packed_act_sample.end(), batch_hidden_packed_activations_L2_lut.begin() + k * ((hidden_dim + 4) / 5));
            }
            lut_linear_forward(q_layer2, batch_hidden_packed_activations_L2_lut, batch_final_q_lut, hidden_dim, output_dim, current_batch_actual_size, precomputed_bit_slice_lut.data());
            log_softmax(batch_final_q_lut.data() + 0, output_dim * current_batch_actual_size); // Apply log_softmax to entire batch output

            auto end_quant_lut = std::chrono::high_resolution_clock::now();
            total_quant_lut_duration += (end_quant_lut - start_quant_lut);

            for (int k = 0; k < current_batch_actual_size; ++k) {
                int true_label = test_labels[b_idx * BATCH_SIZE + k];
                if (argmax(batch_final_q_lut, k * output_dim, output_dim) == true_label) {
                    correct_quant_lut++;
                }
            }


            // --- Weights-Only Quantized Model Inference for Batch ---
            auto start_quant_wo = std::chrono::high_resolution_clock::now();

            // L1 input: float -> int8_t (no ternary or packing for WO)
            for (int k = 0; k < current_batch_actual_size; ++k) {
                quantize_float_to_int8_with_scale(batch_input_f32.data() + k * input_dim_padded, batch_input_i8_wo.data() + k * input_dim_padded, input_dim_padded, wo_q_layer1.activation_scale);
            }
            weights_only_linear_forward(wo_q_layer1, batch_input_i8_wo, batch_hidden_q_f32_wo, input_dim_padded, hidden_dim, current_batch_actual_size);

            // ReLU on float output
            for (int k = 0; k < current_batch_actual_size; ++k) {
                relu(batch_hidden_q_f32_wo.data() + k * hidden_dim, hidden_dim);
            }

            // L2 input: float -> int8_t (no ternary or packing for WO)
            for (int k = 0; k < current_batch_actual_size; ++k) {
                quantize_float_to_int8_with_scale(batch_hidden_q_f32_wo.data() + k * hidden_dim, batch_hidden_i8_wo.data() + k * hidden_dim, hidden_dim, wo_q_layer2.activation_scale);
            }
            weights_only_linear_forward(wo_q_layer2, batch_hidden_i8_wo, batch_final_q_wo, hidden_dim, output_dim, current_batch_actual_size);
            log_softmax(batch_final_q_wo.data() + 0, output_dim * current_batch_actual_size);

            auto end_quant_wo = std::chrono::high_resolution_clock::now();
            total_quant_wo_duration += (end_quant_wo - start_quant_wo);

            for (int k = 0; k < current_batch_actual_size; ++k) {
                int true_label = test_labels[b_idx * BATCH_SIZE + k];
                if (argmax(batch_final_q_wo, k * output_dim, output_dim) == true_label) {
                    correct_quant_wo++;
                }
            }


            // --- Full Precision Float MLP Inference for Batch ---
            auto start_float_fp = std::chrono::high_resolution_clock::now();
            standard_linear_forward(f_layer1, batch_input_f32, batch_hidden_f_fp, input_dim_padded, hidden_dim, current_batch_actual_size);
            relu(batch_hidden_f_fp.data() + 0, hidden_dim * current_batch_actual_size);
            standard_linear_forward(f_layer2, batch_hidden_f_fp, batch_final_f_fp, hidden_dim, output_dim, current_batch_actual_size);
            log_softmax(batch_final_f_fp.data() + 0, output_dim * current_batch_actual_size);

            auto end_float_fp = std::chrono::high_resolution_clock::now();
            total_float_fp_duration += (end_float_fp - start_float_fp);

            for (int k = 0; k < current_batch_actual_size; ++k) {
                int true_label = test_labels[b_idx * BATCH_SIZE + k];
                if (argmax(batch_final_f_fp, k * output_dim, output_dim) == true_label) {
                    correct_float_fp++;
                }
            }

            if ((b_idx + 1) % 10 == 0) { // Print progress every 10 batches
                std::cout << "Processed " << std::min((b_idx + 1) * BATCH_SIZE, NUM_TEST_IMAGES) << "/" << NUM_TEST_IMAGES << " images." << std::endl;
            }
        }

        double quant_lut_accuracy = static_cast<double>(correct_quant_lut) / NUM_TEST_IMAGES * 100.0;
        double quant_wo_accuracy = static_cast<double>(correct_quant_wo) / NUM_TEST_IMAGES * 100.0;
        double float_fp_accuracy = static_cast<double>(correct_float_fp) / NUM_TEST_IMAGES * 100.0;

        // --- 6. Calculate Memory and Print Comparison Table ---
        // Model Parameter Memory (Weights + Bias + LUT for LUT model)
        size_t quant_lut_model_mem = (q_layer1.packed_weights.size() + q_layer2.packed_weights.size()) * sizeof(uint8_t) + precomputed_bit_slice_lut.size() * sizeof(int16_t) + (b1.size() + b2.size()) * sizeof(float);
        size_t quant_wo_model_mem = (wo_q_layer1.weights.size() + wo_q_layer2.weights.size()) * sizeof(int8_t) + (b1.size() + b2.size()) * sizeof(float);
        size_t float_fp_model_mem = (fp_layer1.weights.size() + fp_layer2.weights.size()) * sizeof(float) + (fp_layer1.bias.size() + fp_layer2.bias.size()) * sizeof(float);

        // Runtime Buffer Memory (Activations + Temps for 1 Batch)
        // Calculated based on maximum capacities of declared vectors.
        // This represents the memory allocated for these buffers once.
        size_t runtime_buffers_mem_lut =
            batch_input_f32.capacity() * sizeof(float) + // Shared input buffer
            batch_input_i8_temp_lut.capacity() * sizeof(int8_t) +
            batch_input_ternary_L1_lut.capacity() * sizeof(int8_t) +
            batch_input_packed_activations_L1_lut.capacity() * sizeof(uint8_t) +
            batch_hidden_q_f32_lut.capacity() * sizeof(float) +
            batch_hidden_i8_temp_lut.capacity() * sizeof(int8_t) +
            batch_hidden_packed_activations_L2_lut.capacity() * sizeof(uint8_t) +
            batch_final_q_lut.capacity() * sizeof(float);

        size_t runtime_buffers_mem_wo =
            batch_input_f32.capacity() * sizeof(float) + // Shared input buffer
            batch_input_i8_wo.capacity() * sizeof(int8_t) +
            batch_hidden_q_f32_wo.capacity() * sizeof(float) +
            batch_hidden_i8_wo.capacity() * sizeof(int8_t) +
            batch_final_q_wo.capacity() * sizeof(float);

        size_t runtime_buffers_mem_fp =
            batch_input_f32.capacity() * sizeof(float) + // Shared input buffer
            batch_hidden_f_fp.capacity() * sizeof(float) +
            batch_final_f_fp.capacity() * sizeof(float);

        // Total Memory (Model Parameters + Runtime Buffers)
        // This sum gives the total allocated memory for each model's full inference path,
        // including its parameters and the temporary buffers it needs for a batch.
        size_t total_mem_lut = quant_lut_model_mem + runtime_buffers_mem_lut;
        size_t total_mem_wo = quant_wo_model_mem + runtime_buffers_mem_wo;
        size_t total_mem_fp = float_fp_model_mem + runtime_buffers_mem_fp;


        std::cout << "\n\n--- Performance and Accuracy Comparison (" << NUM_TEST_IMAGES << " images, Batch Size: " << BATCH_SIZE << ") ---" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Metric              | All Quantized MLP (LUT) | Wt-Only Quant MLP  | Full Prec. Float MLP " << std::endl;
        std::cout << "--------------------|--------------------|--------------------|----------------------" << std::endl;
        std::cout << "Total Time (ms)     | " << std::setw(18) << total_quant_lut_duration.count() << " | " << std::setw(18) << total_quant_wo_duration.count() << " | " << std::setw(20) << total_float_fp_duration.count() << std::endl;
        std::cout << "Avg. Time / iter(ms)| " << std::setw(18) << total_quant_lut_duration.count() / NUM_TEST_IMAGES << " | " << std::setw(18) << total_quant_wo_duration.count() / NUM_TEST_IMAGES << " | " << std::setw(20) << total_float_fp_duration.count() / NUM_TEST_IMAGES << std::endl;
        std::cout << "Accuracy (%)        | " << std::setw(18) << quant_lut_accuracy << " | " << std::setw(18) << quant_wo_accuracy << " | " << std::setw(20) << float_fp_accuracy << std::endl;
        // Updated memory rows
        std::cout << "Weight Memory (kB)  | " << std::setw(18) << quant_lut_model_mem / 1024.0 << " | " << std::setw(18) << quant_wo_model_mem / 1024.0 << " | " << std::setw(20) << float_fp_model_mem / 1024.0 << std::endl;
        std::cout << "Runtime Buffers (kB)| " << std::setw(18) << runtime_buffers_mem_lut / 1024.0 << " | " << std::setw(18) << runtime_buffers_mem_wo / 1024.0 << " | " << std::setw(20) << runtime_buffers_mem_fp / 1024.0 << std::endl;
        std::cout << "Total Memory (kB)   | " << std::setw(18) << total_mem_lut / 1024.0 << " | " << std::setw(18) << total_mem_wo / 1024.0 << " | " << std::setw(20) << total_mem_fp / 1024.0 << std::endl;
        std::cout << "-------------------------------------------------------------" << std::endl;
        if (total_quant_lut_duration.count() > 0.0001) {
            std::cout << "Speedup All vs Full Prec. Float: " << total_float_fp_duration.count() / total_quant_lut_duration.count() << "x" << std::endl;
        }
        if (total_quant_wo_duration.count() > 0.0001) {
            std::cout << "Speedup Wt-Only vs Full Prec. Float: " << total_float_fp_duration.count() / total_quant_wo_duration.count() << "x" << std::endl;
        }
        std::cout << "Memory Reduction All vs Full Prec. Float (Model Params): " << (1.0 - (double)quant_lut_model_mem / float_fp_model_mem) * 100.0 << "%" << std::endl;
        std::cout << "Memory Reduction Wt-Only vs Full Prec. Float (Model Params): " << (1.0 - (double)quant_wo_model_mem / float_fp_model_mem) * 100.0 << "%" << std::endl;
        std::cout << "Total Memory Reduction All vs Full Prec. Float: " << (1.0 - (double)total_mem_lut / total_mem_fp) * 100.0 << "%" << std::endl;
        std::cout << "Total Memory Reduction Wt-Only vs Full Prec. Float: " << (1.0 - (double)total_mem_wo / total_mem_fp) * 100.0 << "%" << std::endl;


    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
