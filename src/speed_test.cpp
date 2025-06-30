#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <numeric>
#include "kernels.h"
#include "types.h"
#include "utils.h"

// Function to generate random float data
void generate_random_data(std::vector<float>& data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (auto& d : data) {
        d = dis(gen);
    }
}

// Function to generate random int8_t data
void generate_random_data(std::vector<int8_t>& data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-1, 1);
    for (auto& d : data) {
        d = dis(gen);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_dim> <hidden_dim> <output_dim>" << std::endl;
        return 1;
    }

    const int input_dim = std::stoi(argv[1]);
    const int hidden_dim = std::stoi(argv[2]);
    const int output_dim = std::stoi(argv[3]);
    const int batch_size = 1; // Evaluate for a single input for simplicity

    std::cout << "Evaluating MLP speed with dimensions: "
              << "Input=" << input_dim
              << ", Hidden=" << hidden_dim
              << ", Output=" << output_dim << std::endl;

    // --- Randomly initialize layers ---

    // Standard Float MLP
    FloatLayer float_layer1, float_layer2;
    float_layer1.weights.resize(hidden_dim * input_dim);
    float_layer1.bias.resize(hidden_dim);
    float_layer2.weights.resize(output_dim * hidden_dim);
    float_layer2.bias.resize(output_dim);
    generate_random_data(float_layer1.weights);
    generate_random_data(float_layer1.bias);
    generate_random_data(float_layer2.weights);
    generate_random_data(float_layer2.bias);

    // Weights-Only Quantized MLP
    WeightsOnlyQuantLayer wo_layer1, wo_layer2;
    wo_layer1.weights.resize(hidden_dim * input_dim);
    wo_layer1.bias.resize(hidden_dim);
    wo_layer2.weights.resize(output_dim * hidden_dim);
    wo_layer2.bias.resize(output_dim);
    generate_random_data(wo_layer1.weights);
    generate_random_data(wo_layer1.bias);
    generate_random_data(wo_layer2.weights);
    generate_random_data(wo_layer2.bias);
    wo_layer1.activation_scale = 127.0f;
    wo_layer2.activation_scale = 127.0f;


    // LUT-based Quantized MLP
    LutLayer lut_layer1, lut_layer2;

    // Correctly pack weights for layer 1 row-by-row
    std::vector<int8_t> unpacked_weights1(hidden_dim * input_dim);
    generate_random_data(unpacked_weights1);
    lut_layer1.packed_weights.reserve(hidden_dim * ((input_dim + 4) / 5));
    for (int i = 0; i < hidden_dim; ++i) {
        std::vector<int8_t> row(unpacked_weights1.begin() + i * input_dim, unpacked_weights1.begin() + (i + 1) * input_dim);
        std::vector<uint8_t> packed_row = pack_weights_5x3bit(row, input_dim);
        lut_layer1.packed_weights.insert(lut_layer1.packed_weights.end(), packed_row.begin(), packed_row.end());
    }
    lut_layer1.bias.resize(hidden_dim);
    generate_random_data(lut_layer1.bias);
    lut_layer1.activation_scale = 127.0f;

    // Correctly pack weights for layer 2 row-by-row
    std::vector<int8_t> unpacked_weights2(output_dim * hidden_dim);
    generate_random_data(unpacked_weights2);
    lut_layer2.packed_weights.reserve(output_dim * ((hidden_dim + 4) / 5));
    for (int i = 0; i < output_dim; ++i) {
        std::vector<int8_t> row(unpacked_weights2.begin() + i * hidden_dim, unpacked_weights2.begin() + (i + 1) * hidden_dim);
        std::vector<uint8_t> packed_row = pack_weights_5x3bit(row, hidden_dim);
        lut_layer2.packed_weights.insert(lut_layer2.packed_weights.end(), packed_row.begin(), packed_row.end());
    }
    lut_layer2.bias.resize(output_dim);
    generate_random_data(lut_layer2.bias);
    lut_layer2.activation_scale = 1.0f;


    // --- Generate random inputs ---
    std::vector<float> input_f32(batch_size * input_dim);
    generate_random_data(input_f32);

    std::vector<int8_t> input_i8(batch_size * input_dim);
    quantize_float_to_int8_with_scale(input_f32.data(), input_i8.data(), input_f32.size(), 127.0f);

    const int packed_input_size = (input_dim + 4) / 5;
    std::vector<uint8_t> input_packed(batch_size * packed_input_size);
    pack_ternary_activations_5x3bit_to_ptr(input_i8.data(), input_dim, input_packed.data());

    // --- Build LUT ---
    std::vector<int16_t> precomputed_lut;
    build_bit_slice_lut_5x3(precomputed_lut);
    build_packing_lut();


    // --- Outputs ---
    std::vector<float> output_float(batch_size * output_dim);
    std::vector<float> output_wo(batch_size * output_dim);
    std::vector<float> output_lut(batch_size * output_dim);

    std::vector<float> hidden_float(batch_size * hidden_dim);
    std::vector<float> hidden_wo_f32(batch_size * hidden_dim);
    std::vector<int8_t> hidden_wo_i8(batch_size * hidden_dim);
    std::vector<float> hidden_lut_f32(batch_size * hidden_dim);
    std::vector<uint8_t> hidden_lut_packed(batch_size * ((hidden_dim + 4) / 5));


    // --- Timing ---
    const int num_iterations = 1000;

    // Standard Float
    auto start_float = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        standard_linear_forward(float_layer1, input_f32, hidden_float, input_dim, hidden_dim, batch_size);
        relu(hidden_float.data(), hidden_float.size());
        standard_linear_forward(float_layer2, hidden_float, output_float, hidden_dim, output_dim, batch_size);
    }
    auto end_float = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> float_duration = end_float - start_float;

    // Weights-Only
    auto start_wo = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        weights_only_linear_forward(wo_layer1, input_i8, hidden_wo_f32, input_dim, hidden_dim, batch_size);
        relu(hidden_wo_f32.data(), hidden_wo_f32.size());
        quantize_float_to_int8_with_scale(hidden_wo_f32.data(), hidden_wo_i8.data(), hidden_wo_f32.size(), wo_layer2.activation_scale);
        weights_only_linear_forward(wo_layer2, hidden_wo_i8, output_wo, hidden_dim, output_dim, batch_size);
    }
    auto end_wo = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> wo_duration = end_wo - start_wo;


    // LUT-based
    auto start_lut = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        lut_linear_forward(lut_layer1, input_packed, hidden_lut_f32, input_dim, hidden_dim, batch_size, precomputed_lut.data());
        std::vector<int8_t> hidden_lut_i8(batch_size*hidden_dim);
        quantize_float_to_int8_with_scale(hidden_lut_f32.data(), hidden_lut_i8.data(), hidden_lut_f32.size(), lut_layer2.activation_scale);
        pack_ternary_activations_5x3bit_to_ptr(hidden_lut_i8.data(), hidden_dim, hidden_lut_packed.data());
        lut_linear_forward(lut_layer2, hidden_lut_packed, output_lut, hidden_dim, output_dim, batch_size, precomputed_lut.data());
    }
    auto end_lut = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> lut_duration = end_lut - start_lut;


    std::cout << "\n--- Speed Evaluation Results ---" << std::endl;
    std::cout << "Number of iterations: " << num_iterations << std::endl;
    std::cout << "Standard Float MLP: " << float_duration.count() << " ms" << std::endl;
    std::cout << "Weights-Only Quantized MLP: " << wo_duration.count() << " ms" << std::endl;
    std::cout << "LUT-based Quantized MLP: " << lut_duration.count() << " ms" << std::endl;

    return 0;
}