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
// #define __AVX2__
#ifdef __AVX2__
#include <immintrin.h> // For AVX2 intrinsics
#endif

// Include the new bit-slice kernel and preprocessor headers
#include <kernels.h>
#include <types.h>
#include <utils.h>

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

        std::cout << "Packing LUT built. Size: " << g_packing_lut.size() * sizeof(uint8_t) / 1024.0 << " KB" << std::endl;


        // --- 1. Load Model Weights for LUT and Weights-Only Quantized MLPs ---
        // These are loaded from mlp_model_aq.bin (QAT trained model)
        std::ifstream qat_model_file("mlp_model_aq.bin", std::ios::binary);
        if (!qat_model_file) { throw std::runtime_error("Failed to open mlp_model_aq.bin. Please run the new convert_qat.py first."); }
        char magic_qat[4]; qat_model_file.read(magic_qat, 4);
        if (std::string(magic_qat, 4) != "MLP4") { throw std::runtime_error("Invalid QAT model file format. Expected 'MLP4'. Please run the new convert_qat.py."); }
        int qat_input_dim, qat_hidden_dim, qat_output_dim;
        qat_model_file.read(reinterpret_cast<char*>(&qat_input_dim), sizeof(int));
        qat_model_file.read(reinterpret_cast<char*>(&qat_hidden_dim), sizeof(int));
        qat_model_file.read(reinterpret_cast<char*>(&qat_output_dim), sizeof(int));

        // Read Layer 1 (fc1) weights (int8_t from file)
        std::vector<int8_t> w1_int8_unpacked(qat_input_dim * qat_hidden_dim);   qat_model_file.read(reinterpret_cast<char*>(w1_int8_unpacked.data()), w1_int8_unpacked.size() * sizeof(int8_t));
        std::vector<float> b1(qat_hidden_dim);                    qat_model_file.read(reinterpret_cast<char*>(b1.data()), b1.size() * sizeof(float));

        // Read the single learned activation scale for the hidden layer
        float hidden_activation_scale;
        qat_model_file.read(reinterpret_cast<char*>(&hidden_activation_scale), sizeof(float));

        // Read Layer 2 (fc2) weights (int8_t from file)
        std::vector<int8_t> w2_int8_unpacked(qat_hidden_dim * qat_output_dim); qat_model_file.read(reinterpret_cast<char*>(w2_int8_unpacked.data()), w2_int8_unpacked.size() * sizeof(int8_t));
        std::vector<float> b2(qat_output_dim);                    qat_model_file.read(reinterpret_cast<char*>(b2.data()), b2.size() * sizeof(float));

        std::cout << "QAT Model (mlp_model_aq.bin) loaded successfully." << std::endl;
        std::cout << "QAT Model Dims: Input " << qat_input_dim << ", Hidden " << qat_hidden_dim << ", Output " << qat_output_dim << std::endl;
        std::cout << "Loaded learned hidden activation scale: " << hidden_activation_scale << std::endl;

        // The scale for quantizing the input data is fixed because the input is always normalized to [-1, 1]
        const float input_quantization_scale = 127.0f;


        // --- 2. Load Full Precision Float MLP ---
        FloatLayer fp_layer1, fp_layer2;
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
        q_layer1.activation_scale = input_quantization_scale; // Use fixed scale for input layer

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
        q_layer2.activation_scale = hidden_activation_scale; // Use learned scale for hidden layer

        std::vector<int32_t> precomputed_bit_slice_lut;
        build_bit_slice_lut_5x3(precomputed_bit_slice_lut);
        std::cout << "Bit-Slice LUT built. Size: " << precomputed_bit_slice_lut.size() * sizeof(int32_t) / 1024.0 << " KB" << std::endl;


        // 3b. Weights-Only Quant MLP Model
        WeightsOnlyQuantLayer wo_q_layer1, wo_q_layer2;

        wo_q_layer1.weights.resize(hidden_dim * input_dim_padded, 0);
        for(int i = 0; i < hidden_dim; ++i) {
            std::copy(w1_int8_unpacked.begin() + i * qat_input_dim,
                      w1_int8_unpacked.begin() + (i + 1) * qat_input_dim,
                      wo_q_layer1.weights.begin() + i * input_dim_padded);
        }
        wo_q_layer1.bias = b1;
        wo_q_layer1.activation_scale = input_quantization_scale; // Use fixed scale for input layer

        wo_q_layer2.weights = w2_int8_unpacked;
        wo_q_layer2.bias = b2;
        wo_q_layer2.activation_scale = hidden_activation_scale; // Use learned scale for hidden layer


        // 3c. Full Precision Float MLP
        FloatLayer& f_layer1 = fp_layer1;
        FloatLayer& f_layer2 = fp_layer2;


        // --- 4. Load FashionMNIST Test Data ---
        const int NUM_TEST_IMAGES = 10000;
        std::vector<float> test_images_f32;
        std::vector<int> test_labels;

        std::cout << "\nLoading FashionMNIST test data..." << std::endl;
        if (!load_images_from_file("test_images_padded_f32.bin", test_images_f32, NUM_TEST_IMAGES, input_dim_padded)) {
            throw std::runtime_error("Could not load test_images_padded_f32.bin. Please run save_all_images.py first.");
        }
        if (!load_labels_from_file("test_labels.bin", test_labels, NUM_TEST_IMAGES)) {
            throw std::runtime_error("Could not load test_labels.bin. Make sure it's in the same directory.");
        }
        std::cout << "Successfully loaded " << NUM_TEST_IMAGES << " pre-padded test images and labels." << std::endl;

        // --- Pre-process inputs for quantized models ---
        std::vector<int8_t> preprocessed_input_i8_lut(NUM_TEST_IMAGES * input_dim_padded);
        std::vector<uint8_t> preprocessed_input_packed_lut(NUM_TEST_IMAGES * ((input_dim_padded + 4) / 5));
        std::vector<int8_t> preprocessed_input_i8_wo(NUM_TEST_IMAGES * input_dim_padded);

        std::cout << "\nPerforming one-time pre-processing of input images for quantized models..." << std::endl;
        for (int k = 0; k < NUM_TEST_IMAGES; ++k) {
            // Quantize input for both LUT and Weights-Only models using the fixed input scale
            quantize_float_to_int8_with_scale(
                test_images_f32.data() + k * input_dim_padded,
                preprocessed_input_i8_wo.data() + k * input_dim_padded,
                input_dim_padded,
                input_quantization_scale
            );

            // Pack the quantized input specifically for the LUT model
            pack_ternary_activations_5x3bit_to_ptr(
                preprocessed_input_i8_wo.data() + k * input_dim_padded,
                input_dim_padded,
                preprocessed_input_packed_lut.data() + k * ((input_dim_padded + 4) / 5)
            );
        }
        std::cout << "Input pre-processing complete." << std::endl;


        // --- Batching Configuration & Evaluation ---
        const int BATCH_SIZE = 64;
        int num_batches = (NUM_TEST_IMAGES + BATCH_SIZE - 1) / BATCH_SIZE;
        int correct_quant_lut = 0, correct_quant_wo = 0, correct_float_fp = 0;
        std::chrono::duration<double, std::milli> total_quant_lut_duration(0.0);
        std::chrono::duration<double, std::milli> total_quant_wo_duration(0.0);
        std::chrono::duration<double, std::milli> total_float_fp_duration(0.0);

        std::cout << "\nStarting batched inference..." << std::endl;

        // --- Buffers for batch processing ---
        std::vector<float> batch_input_f32(BATCH_SIZE * input_dim_padded);
        std::vector<uint8_t> batch_input_packed_activations_L1_lut(BATCH_SIZE * ((input_dim_padded + 4) / 5));
        std::vector<int8_t> batch_input_i8_wo(BATCH_SIZE * input_dim_padded);
        std::vector<float> batch_hidden_q_f32_lut(BATCH_SIZE * hidden_dim), batch_hidden_q_f32_wo(BATCH_SIZE * hidden_dim);
        std::vector<int8_t> batch_hidden_i8_temp_lut(BATCH_SIZE * hidden_dim), batch_hidden_i8_wo(BATCH_SIZE * hidden_dim);
        std::vector<uint8_t> batch_hidden_packed_activations_L2_lut(BATCH_SIZE * ((hidden_dim + 4) / 5));
        std::vector<float> batch_final_q_lut(BATCH_SIZE * output_dim), batch_final_q_wo(BATCH_SIZE * output_dim);
        std::vector<float> batch_hidden_f_fp(BATCH_SIZE * hidden_dim), batch_final_f_fp(BATCH_SIZE * output_dim);


        for (int b_idx = 0; b_idx < num_batches; ++b_idx) {
            int current_batch_actual_size = std::min(BATCH_SIZE, NUM_TEST_IMAGES - b_idx * BATCH_SIZE);
            int start_idx = b_idx * BATCH_SIZE;

            // --- Prepare batch inputs ---
            std::copy(test_images_f32.begin() + start_idx * input_dim_padded, test_images_f32.begin() + (start_idx + current_batch_actual_size) * input_dim_padded, batch_input_f32.begin());
            std::copy(preprocessed_input_packed_lut.begin() + start_idx * ((input_dim_padded + 4) / 5), preprocessed_input_packed_lut.begin() + (start_idx + current_batch_actual_size) * ((input_dim_padded + 4) / 5), batch_input_packed_activations_L1_lut.begin());
            std::copy(preprocessed_input_i8_wo.begin() + start_idx * input_dim_padded, preprocessed_input_i8_wo.begin() + (start_idx + current_batch_actual_size) * input_dim_padded, batch_input_i8_wo.begin());

            // --- LUT Quantized Model Inference ---
            auto start_quant_lut = std::chrono::high_resolution_clock::now();
            lut_linear_forward(q_layer1, batch_input_packed_activations_L1_lut, batch_hidden_q_f32_lut, input_dim_padded, hidden_dim, current_batch_actual_size, precomputed_bit_slice_lut.data());
            for(int k = 0; k < current_batch_actual_size; ++k) {
                quantize_float_to_int8_with_scale(batch_hidden_q_f32_lut.data() + k * hidden_dim, batch_hidden_i8_temp_lut.data() + k * hidden_dim, hidden_dim, q_layer1.activation_scale);
                pack_ternary_activations_5x3bit_to_ptr(batch_hidden_i8_temp_lut.data() + k * hidden_dim, hidden_dim, batch_hidden_packed_activations_L2_lut.data() + k * ((hidden_dim + 4) / 5));
            }
            lut_linear_forward(q_layer2, batch_hidden_packed_activations_L2_lut, batch_final_q_lut, hidden_dim, output_dim, current_batch_actual_size, precomputed_bit_slice_lut.data());
            log_softmax(batch_final_q_lut.data(), output_dim * current_batch_actual_size);
            auto end_quant_lut = std::chrono::high_resolution_clock::now();
            total_quant_lut_duration += (end_quant_lut - start_quant_lut);
            for (int k = 0; k < current_batch_actual_size; ++k) if (argmax(batch_final_q_lut, k * output_dim, output_dim) == test_labels[start_idx + k]) correct_quant_lut++;

            // --- Weights-Only Quantized Model Inference ---
            auto start_quant_wo = std::chrono::high_resolution_clock::now();
            weights_only_linear_forward(wo_q_layer1, batch_input_i8_wo, batch_hidden_q_f32_wo, input_dim_padded, hidden_dim, current_batch_actual_size);
            for(int k = 0; k < current_batch_actual_size; ++k) {
                quantize_float_to_int8_with_scale(batch_hidden_q_f32_wo.data() + k * hidden_dim, batch_hidden_i8_wo.data() + k * hidden_dim, hidden_dim, wo_q_layer2.activation_scale);
            }
            weights_only_linear_forward(wo_q_layer2, batch_hidden_i8_wo, batch_final_q_wo, hidden_dim, output_dim, current_batch_actual_size);
            log_softmax(batch_final_q_wo.data(), output_dim * current_batch_actual_size);
            auto end_quant_wo = std::chrono::high_resolution_clock::now();
            total_quant_wo_duration += (end_quant_wo - start_quant_wo);
            for (int k = 0; k < current_batch_actual_size; ++k) if (argmax(batch_final_q_wo, k * output_dim, output_dim) == test_labels[start_idx + k]) correct_quant_wo++;

            // --- Full Precision Float MLP Inference ---
            auto start_float_fp = std::chrono::high_resolution_clock::now();
            standard_linear_forward(f_layer1, batch_input_f32, batch_hidden_f_fp, input_dim_padded, hidden_dim, current_batch_actual_size);
            relu(batch_hidden_f_fp.data(), hidden_dim * current_batch_actual_size);
            standard_linear_forward(f_layer2, batch_hidden_f_fp, batch_final_f_fp, hidden_dim, output_dim, current_batch_actual_size);
            log_softmax(batch_final_f_fp.data(), output_dim * current_batch_actual_size);
            auto end_float_fp = std::chrono::high_resolution_clock::now();
            total_float_fp_duration += (end_float_fp - start_float_fp);
            for (int k = 0; k < current_batch_actual_size; ++k) if (argmax(batch_final_f_fp, k * output_dim, output_dim) == test_labels[start_idx + k]) correct_float_fp++;

            if ((b_idx + 1) % 10 == 0) { std::cout << "Processed " << std::min((b_idx + 1) * BATCH_SIZE, NUM_TEST_IMAGES) << "/" << NUM_TEST_IMAGES << " images." << std::endl; }
        }

        double quant_lut_accuracy = static_cast<double>(correct_quant_lut) / NUM_TEST_IMAGES * 100.0;
        double quant_wo_accuracy = static_cast<double>(correct_quant_wo) / NUM_TEST_IMAGES * 100.0;
        double float_fp_accuracy = static_cast<double>(correct_float_fp) / NUM_TEST_IMAGES * 100.0;

        // --- Print Final Results ---
        std::cout << "\n\n--- Performance and Accuracy Comparison (" << NUM_TEST_IMAGES << " images, Batch Size: " << BATCH_SIZE << ") ---" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Metric              | All Quantized MLP (LUT) | Wt-Only Quant MLP  | Full Prec. Float MLP " << std::endl;
        std::cout << "--------------------|--------------------|--------------------|----------------------" << std::endl;
        std::cout << "Total Time (ms)     | " << std::setw(18) << total_quant_lut_duration.count() << " | " << std::setw(18) << total_quant_wo_duration.count() << " | " << std::setw(20) << total_float_fp_duration.count() << std::endl;
        std::cout << "Avg. Time / iter(ms)| " << std::setw(18) << total_quant_lut_duration.count() / NUM_TEST_IMAGES << " | " << std::setw(18) << total_quant_wo_duration.count() / NUM_TEST_IMAGES << " | " << std::setw(20) << total_float_fp_duration.count() / NUM_TEST_IMAGES << std::endl;
        std::cout << "Accuracy (%)        | " << std::setw(18) << quant_lut_accuracy << " | " << std::setw(18) << quant_wo_accuracy << " | " << std::setw(20) << float_fp_accuracy << std::endl;
        std::cout << "-------------------------------------------------------------" << std::endl;

        // --- NEW: Memory Cost Comparison ---
        double float_mem = (double)(f_layer1.weights.size() * sizeof(float) + f_layer1.bias.size() * sizeof(float) +
                                    f_layer2.weights.size() * sizeof(float) + f_layer2.bias.size() * sizeof(float)) / 1024.0;

        double wo_mem = (double)(wo_q_layer1.weights.size() * sizeof(int8_t) + wo_q_layer1.bias.size() * sizeof(float) +
                                 wo_q_layer2.weights.size() * sizeof(int8_t) + wo_q_layer2.bias.size() * sizeof(float)) / 1024.0;

        double lut_mem = (double)(q_layer1.packed_weights.size() * sizeof(uint8_t) + q_layer1.bias.size() * sizeof(float) +
                                  q_layer2.packed_weights.size() * sizeof(uint8_t) + q_layer2.bias.size() * sizeof(float)) / 1024.0;


        std::cout << "\n\n--- Memory Cost Comparison ---" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Model Type          | Memory Cost (KB)   " << std::endl;
        std::cout << "--------------------|--------------------" << std::endl;
        std::cout << "Full Prec. Float MLP| " << std::setw(18) << float_mem << std::endl;
        std::cout << "Wt-Only Quant MLP   | " << std::setw(18) << wo_mem << std::endl;
        std::cout << "All Quantized (LUT) | " << std::setw(18) << lut_mem << std::endl;
        std::cout << "------------------------------------------" << std::endl;
        std::cout << "NOTE: As the size of matrix going larger, the "
                     "memory cost of LUT table should be negligible."<<std::endl;


    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}