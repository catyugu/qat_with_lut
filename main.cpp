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
#define __AVX2__
#ifdef __AVX2__
#include <immintrin.h> // For AVX2 intrinsics
#endif

// --- Data Structures for our Models ---
struct FloatLayer {
    std::vector<float> weights;
    std::vector<float> bias;
};

// Our LUT-based layer will use quantized weights and an activation scale
struct LutLayer {
    std::vector<int8_t> weights; // Ternary {-1, 0, 1}
    std::vector<float> bias;
    float activation_scale = 1.0f; // Scale for the *input* activations to this layer
};

// --- Helper Functions ---

// Quantizes float weights to {-1, 0, 1} (Not used for loading, but kept for consistency with training concept)
void quantize_weights_to_ternary(const std::vector<float>& float_weights, std::vector<int8_t>& ternary_weights) {
    ternary_weights.resize(float_weights.size());
    for (size_t i = 0; i < float_weights.size(); ++i) {
        if (float_weights[i] > 0.1f)      ternary_weights[i] = 1;
        else if (float_weights[i] < -0.1f) ternary_weights[i] = -1;
        else                              ternary_weights[i] = 0;
    }
}

// Quantizes float_ptr to int_ptr using a *provided* fixed_scale.
// Takes raw pointers and size to avoid vector slicing/copying overhead.
void quantize_float_to_int8_with_scale(const float* float_ptr, int8_t* int_ptr, size_t size, float fixed_scale) {
    for (size_t i = 0; i < size; ++i) {
        auto val = static_cast<int32_t>(roundf(float_ptr[i] * fixed_scale));
        int_ptr[i] = static_cast<int8_t>(std::max(-128, std::min(127, val)));
    }
}


#ifdef __AVX2__
// Horizontal sum of 8 int32_t elements in a __m256i vector (adapted from ggml-bitnet-mad.cpp)
static inline int hsum_i32_8(const __m256i a) {
    // Extract lower 128 bits and higher 128 bits, then add them
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    // Unpack high 64 bits to lower 64 bits and add to original
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    // Shuffle sum64 to get (v3, v2, v1, v0) where v0 is at lowest 32 bits
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    // Add sum64 and hi32, then extract the lowest 32-bit integer (which now contains the total sum)
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}
#endif

// The high-performance LUT-based forward pass (pure integer math)
// Optimized for ternary weights using AVX2 intrinsics (if enabled).
// This version handles a batch of inputs.
void lut_linear_forward(const LutLayer& layer, const std::vector<int8_t>& input_i8_batched, std::vector<float>& output_f32_batched, int input_dim, int output_dim, int batch_size) {
    output_f32_batched.resize(batch_size * output_dim);
    // The dequant_scale for this layer is pre-set in layer.activation_scale
    // which comes from the model file (or fixed for input layer).

    // Pointers to raw data for efficient access
    const int8_t* weights_ptr = layer.weights.data();

    // Iterate over each item in the batch
    for (int b = 0; b < batch_size; ++b) {
        const int8_t* current_batch_input_ptr = input_i8_batched.data() + b * input_dim;

        // Loop over each output neuron (rows of the weight matrix)
        for (int i = 0; i < output_dim; ++i) {
            int32_t sum = 0;
            const int8_t* current_neuron_weights_ptr = weights_ptr + i * input_dim;

#ifdef __AVX2__
            // Use AVX2 for vectorized processing
            __m256i accumulated_sum_vec_32 = _mm256_setzero_si256(); // Accumulate sums in 32-bit integers

            // Process input and weights in chunks of 64 int8_t elements
            // (two __m256i registers, each holding 32 int8_t elements)
            // This assumes input_dim is a multiple of 64 for optimal performance in this loop.
            // For real-world robustness, handle the remainder if input_dim is not perfectly divisible by 64.
            for (int j = 0; j < input_dim; j += 64) {
                // Load first 32 input activations
                __m256i input_vals0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(current_batch_input_ptr + j));
                // Load second 32 input activations
                __m256i input_vals1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(current_batch_input_ptr + j + 32));

                // Load first 32 weights
                __m256i weights_block0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(current_neuron_weights_ptr + j));
                // Load second 32 weights
                __m256i weights_block1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(current_neuron_weights_ptr + j + 32));

                // Perform ternary multiplication for first 32 elements
                __m256i partial_products0_i8 = _mm256_sign_epi8(input_vals0, weights_block0);
                // Perform ternary multiplication for second 32 elements
                __m256i partial_products1_i8 = _mm256_sign_epi8(input_vals1, weights_block1);

                // Convert 8-bit partial products to 16-bit for accumulation (from partial_products0_i8)
                __m256i partial_sums0_lo_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(partial_products0_i8, 0)); // First 16 int8_t -> 16 int16_t
                __m256i partial_sums0_hi_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(partial_products0_i8, 1)); // Next 16 int8_t -> 16 int16_t

                // Convert 8-bit partial products to 16-bit for accumulation (from partial_products1_i8)
                __m256i partial_sums1_lo_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(partial_products1_i8, 0));
                __m256i partial_sums1_hi_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(partial_products1_i8, 1));

                // Accumulate all 16-bit sums into the 32-bit sum vector
                // Each _mm256_madd_epi16 converts 16 int16_t values into 8 int32_t sums.
                accumulated_sum_vec_32 = _mm256_add_epi32(accumulated_sum_vec_32, _mm256_madd_epi16(partial_sums0_lo_16, _mm256_set1_epi16(1)));
                accumulated_sum_vec_32 = _mm256_add_epi32(accumulated_sum_vec_32, _mm256_madd_epi16(partial_sums0_hi_16, _mm256_set1_epi16(1)));
                accumulated_sum_vec_32 = _mm256_add_epi32(accumulated_sum_vec_32, _mm256_madd_epi16(partial_sums1_lo_16, _mm256_set1_epi16(1)));
                accumulated_sum_vec_32 = _mm256_add_epi32(accumulated_sum_vec_32, _mm256_madd_epi16(partial_sums1_hi_16, _mm256_set1_epi16(1)));
            }

            // Horizontal sum of the 8 int32_t values in accumulated_sum_vec_32
            // to get the final total sum for the current output neuron.
            sum = hsum_i32_8(accumulated_sum_vec_32);

#else // Fallback to scalar (non-AVX2) implementation if AVX2 is not enabled
            for (int j = 0; j < input_dim; ++j) {
                // This is the original conditional addition/subtraction logic
                if (current_neuron_weights_ptr[j] == 1) {
                    sum += current_batch_input_ptr[j];
                } else if (current_neuron_weights_ptr[j] == -1) {
                    sum -= current_batch_input_ptr[j];
                }
                // If weight is 0, sum remains unchanged.
            }
#endif
            // Dequantize the final integer sum back to float and add the bias
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
    double sum = 0.0;
    for (size_t i = 0; i < size; ++i) sum += std::exp(vec_ptr[i] - max_val);
    double log_sum = std::log(sum);
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

int main() {
    try {
        // --- 1. Load Model Weights from File ---
        std::ifstream file("mlp_model_aq.bin", std::ios::binary);
        if (!file) { throw std::runtime_error("Failed to open mlp_model_aq.bin. Please run the updated convert.py first."); }
        char magic[4]; file.read(magic, 4);
        if (std::string(magic, 4) != "MLP3") { throw std::runtime_error("Invalid model file format."); }
        int input_dim, hidden_dim, output_dim;
        file.read(reinterpret_cast<char*>(&input_dim), sizeof(int));
        file.read(reinterpret_cast<char*>(&hidden_dim), sizeof(int));
        file.read(reinterpret_cast<char*>(&output_dim), sizeof(int));

        // Read Layer 1 (fc1) weights (now int8_t)
        std::vector<int8_t> w1_int8(input_dim * hidden_dim);   file.read(reinterpret_cast<char*>(w1_int8.data()), w1_int8.size() * sizeof(int8_t));
        std::vector<float> b1(hidden_dim);                    file.read(reinterpret_cast<char*>(b1.data()), b1.size() * sizeof(float));

        // Read Activation Scales (two of them now)
        float input_to_fc1_scale;                             file.read(reinterpret_cast<char*>(&input_to_fc1_scale), sizeof(float)); // Scale for input to first layer
        float input_to_fc2_scale;                             file.read(reinterpret_cast<char*>(&input_to_fc2_scale), sizeof(float)); // Scale for input to second layer

        // Read Layer 2 (fc2) weights (now int8_t)
        std::vector<int8_t> w2_int8(hidden_dim * output_dim); file.read(reinterpret_cast<char*>(w2_int8.data()), w2_int8.size() * sizeof(int8_t));
        std::vector<float> b2(output_dim);                    file.read(reinterpret_cast<char*>(b2.data()), b2.size() * sizeof(float));

        std::cout << "Model loaded successfully." << std::endl;
        std::cout << "Input Dim: " << input_dim << ", Hidden Dim: " << hidden_dim << ", Output Dim: " << output_dim << std::endl;
        std::cout << "Activation scale for input to first layer: " << input_to_fc1_scale << std::endl;
        std::cout << "Activation scale for input to second layer: " << input_to_fc2_scale << std::endl;


        // --- 2. Prepare Both Models ---
        LutLayer q_layer1, q_layer2;
        q_layer1.weights = w1_int8; // Directly assign int8 weights
        q_layer1.bias = b1;
        q_layer1.activation_scale = input_to_fc1_scale; // Use the first scale for q_layer1 (input to fc1)

        q_layer2.weights = w2_int8; // Directly assign int8 weights
        q_layer2.bias = b2;
        q_layer2.activation_scale = input_to_fc2_scale; // Use the second scale for q_layer2 (input to fc2)

        // For float model, we need float weights by casting from the int8_t version for comparison
        // This is important for accurate comparison with the QAT float model's behavior.
        FloatLayer f_layer1, f_layer2;
        f_layer1.weights.resize(w1_int8.size());
        for(size_t i = 0; i < w1_int8.size(); ++i) f_layer1.weights[i] = static_cast<float>(w1_int8[i]);
        f_layer1.bias = b1;

        f_layer2.weights.resize(w2_int8.size());
        for(size_t i = 0; i < w2_int8.size(); ++i) f_layer2.weights[i] = static_cast<float>(w2_int8[i]);
        f_layer2.bias = b2;


        // --- 3. Load FashionMNIST Test Data ---
        const int NUM_TEST_IMAGES = 10000; // FashionMNIST test set size
        std::vector<float> test_images_f32;
        std::vector<int> test_labels;

        std::cout << "\nLoading FashionMNIST test data..." << std::endl;
        // Images are originally [0, 255] then normalized by Python to [-1, 1]
        // Our test_images.bin should reflect this [-1, 1] range.
        if (!load_images_from_file("test_images.bin", test_images_f32, NUM_TEST_IMAGES, input_dim)) {
            throw std::runtime_error("Could not load test_images.bin. Make sure it's in the same directory.");
        }
        if (!load_labels_from_file("test_labels.bin", test_labels, NUM_TEST_IMAGES)) {
            throw std::runtime_error("Could not load test_labels.bin. Make sure it's in the same directory.");
        }
        std::cout << "Successfully loaded " << NUM_TEST_IMAGES << " test images and labels." << std::endl;

        // --- Batching Configuration ---
        const int BATCH_SIZE = 64; // Example batch size. Tune this for your CPU.
        int num_batches = (NUM_TEST_IMAGES + BATCH_SIZE - 1) / BATCH_SIZE;

        // --- 4. Performance and Accuracy Evaluation ---
        int correct_quant = 0;
        int correct_float = 0;

        // Accumulate durations for all inferences
        std::chrono::duration<double, std::milli> total_quant_duration(0.0);
        std::chrono::duration<double, std::milli> total_float_duration(0.0);

        std::cout << "\nStarting batched inference for " << NUM_TEST_IMAGES << " test images (Batch Size: " << BATCH_SIZE << ")..." << std::endl;

        // Vectors for batched inputs and outputs
        std::vector<float> batch_input_f32(BATCH_SIZE * input_dim);
        std::vector<int8_t> batch_input_i8(BATCH_SIZE * input_dim);
        std::vector<float> batch_hidden_q_f32(BATCH_SIZE * hidden_dim);
        std::vector<int8_t> batch_hidden_i8(BATCH_SIZE * hidden_dim);
        std::vector<float> batch_final_q(BATCH_SIZE * output_dim);
        std::vector<float> batch_hidden_f(BATCH_SIZE * hidden_dim);
        std::vector<float> batch_final_f(BATCH_SIZE * output_dim);

        for (int b_idx = 0; b_idx < num_batches; ++b_idx) {
            int current_batch_actual_size = std::min(BATCH_SIZE, NUM_TEST_IMAGES - b_idx * BATCH_SIZE);

            // Copy current batch of images from test_images_f32 into batch_input_f32
            // Only copy the relevant portion for the current batch
            for (int k = 0; k < current_batch_actual_size; ++k) {
                std::copy(test_images_f32.begin() + (b_idx * BATCH_SIZE + k) * input_dim,
                          test_images_f32.begin() + (b_idx * BATCH_SIZE + k + 1) * input_dim,
                          batch_input_f32.begin() + k * input_dim);
            }
            // Note: `batch_input_i8`, `batch_hidden_q_f32`, etc., are pre-sized to max batch size.
            // Operations on their sub-sections will use pointers.

            // --- Quantized Model Inference for Batch ---
            auto start_quant = std::chrono::high_resolution_clock::now();

            // Quantize current batch of inputs using the fixed input_to_fc1_scale
            for (int k = 0; k < current_batch_actual_size; ++k) {
                quantize_float_to_int8_with_scale(
                    batch_input_f32.data() + k * input_dim, // Source float data
                    batch_input_i8.data() + k * input_dim,   // Destination int8 data
                    input_dim,                              // Size of slice
                    q_layer1.activation_scale               // Fixed scale for input to fc1
                );
            }

            lut_linear_forward(q_layer1, batch_input_i8, batch_hidden_q_f32, input_dim, hidden_dim, current_batch_actual_size);
            // ReLU applied to float output using pointers for in-place modification
            for (int k = 0; k < current_batch_actual_size; ++k) {
                relu(batch_hidden_q_f32.data() + k * hidden_dim, hidden_dim);
            }


            // Quantize hidden layer outputs for the next layer using input_to_fc2_scale
            for (int k = 0; k < current_batch_actual_size; ++k) {
                quantize_float_to_int8_with_scale(
                    batch_hidden_q_f32.data() + k * hidden_dim, // Source float data (after ReLU)
                    batch_hidden_i8.data() + k * hidden_dim,    // Destination int8 data
                    hidden_dim,                               // Size of slice
                    q_layer2.activation_scale                 // Fixed scale for input to fc2
                );
            }

            lut_linear_forward(q_layer2, batch_hidden_i8, batch_final_q, hidden_dim, output_dim, current_batch_actual_size);
            // LogSoftmax applied to float output using pointers for in-place modification
            for (int k = 0; k < current_batch_actual_size; ++k) {
                log_softmax(batch_final_q.data() + k * output_dim, output_dim);
            }

            auto end_quant = std::chrono::high_resolution_clock::now();
            total_quant_duration += (end_quant - start_quant);

            // Calculate accuracy for the quantized batch
            for (int k = 0; k < current_batch_actual_size; ++k) {
                int true_label = test_labels[b_idx * BATCH_SIZE + k];
                if (argmax(batch_final_q, k * output_dim, output_dim) == true_label) {
                    correct_quant++;
                }
            }

            // --- Standard Float Model Inference for Batch ---
            auto start_float = std::chrono::high_resolution_clock::now();
            standard_linear_forward(f_layer1, batch_input_f32, batch_hidden_f, input_dim, hidden_dim, current_batch_actual_size);
            // ReLU applied to float output using pointers for in-place modification
            for (int k = 0; k < current_batch_actual_size; ++k) {
                relu(batch_hidden_f.data() + k * hidden_dim, hidden_dim);
            }

            standard_linear_forward(f_layer2, batch_hidden_f, batch_final_f, hidden_dim, output_dim, current_batch_actual_size);
            // LogSoftmax applied to float output using pointers for in-place modification
            for (int k = 0; k < current_batch_actual_size; ++k) {
                log_softmax(batch_final_f.data() + k * output_dim, output_dim);
            }

            auto end_float = std::chrono::high_resolution_clock::now();
            total_float_duration += (end_float - start_float);

            // Calculate accuracy for the float batch
            for (int k = 0; k < current_batch_actual_size; ++k) {
                int true_label = test_labels[b_idx * BATCH_SIZE + k];
                if (argmax(batch_final_f, k * output_dim, output_dim) == true_label) {
                    correct_float++;
                }
            }

            if ((b_idx + 1) % 10 == 0) { // Print progress every 10 batches
                std::cout << "Processed " << std::min((b_idx + 1) * BATCH_SIZE, NUM_TEST_IMAGES) << "/" << NUM_TEST_IMAGES << " images." << std::endl;
            }
        }

        double quant_accuracy = static_cast<double>(correct_quant) / NUM_TEST_IMAGES * 100.0;
        double float_accuracy = static_cast<double>(correct_float) / NUM_TEST_IMAGES * 100.0;

        // --- 5. Calculate Memory and Print Comparison Table ---
        size_t quant_mem = (q_layer1.weights.size() + q_layer2.weights.size()) * sizeof(int8_t);
        size_t float_mem = (f_layer1.weights.size() + f_layer2.weights.size()) * sizeof(float);

        std::cout << "\n\n--- Performance and Accuracy Comparison (" << NUM_TEST_IMAGES << " images, Batch Size: " << BATCH_SIZE << ") ---" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Metric              | LUT Quantized MLP  | Standard Float MLP " << std::endl;
        std::cout << "--------------------|--------------------|--------------------" << std::endl;
        std::cout << "Total Time (ms)     | " << std::setw(18) << total_quant_duration.count() << " | " << std::setw(18) << total_float_duration.count() << std::endl;
        std::cout << "Avg. Time / iter(ms)| " << std::setw(18) << total_quant_duration.count() / NUM_TEST_IMAGES << " | " << std::setw(18) << total_float_duration.count() / NUM_TEST_IMAGES << std::endl;
        std::cout << "Accuracy (%)        | " << std::setw(18) << quant_accuracy << " | " << std::setw(18) << float_accuracy << std::endl;
        std::cout << "Weight Memory (kB)  | " << std::setw(18) << quant_mem / 1024.0 << " | " << std::setw(18) << float_mem / 1024.0 << std::endl;
        std::cout << "-------------------------------------------------------------" << std::endl;
        if (total_quant_duration.count() > 0.0001) {
            std::cout << "Speedup vs Float: " << total_float_duration.count() / total_quant_duration.count() << "x" << std::endl;
        }
        std::cout << "Memory Reduction: " << (1.0 - (double)quant_mem / float_mem) * 100.0 << "%" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}