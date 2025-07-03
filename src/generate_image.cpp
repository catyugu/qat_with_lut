#include "qat_unet_model.h" // Include your UNet model header
#include "utils.h"          // For any utility functions like saving images
#include <iostream>
#include <vector>
#include <chrono> // For timing

// Function to generate a random noisy image (placeholder for actual noise)
std::vector<float> generate_random_noise(int channels, int height, int width) {
    std::vector<float> noise(channels * height * width);
    // Fill with random values between -1 and 1
    for (size_t i = 0; i < noise.size(); ++i) {
        noise[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Random float between -1 and 1
    }
    return noise;
}

int main() {
    std::cout << "Starting QAT UNet Deployment Test..." << std::endl;

    // 1. Initialize QATUNet model
    QATUNet unet_model;
    std::vector<int32_t> precomputed_lut; // The LUT for bit-slice GEMM

    // Define the path to your exported binary model file
    const std::string model_file_path = "./qat_unet_model.bin";

    // 2. Load the model parameters
    auto start_load = std::chrono::high_resolution_clock::now();
    if (!unet_model.load_model(model_file_path, precomputed_lut)) {
        std::cerr << "Failed to load QAT UNet model. Exiting." << std::endl;
        return 1;
    }
    auto end_load = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> load_duration = end_load - start_load;
    std::cout << "Model loaded in " << load_duration.count() << " seconds." << std::endl;

    // 3. Prepare dummy input for inference (e.g., a noisy image)
    int batch_size = 1;
    int img_channels = unet_model.in_channels; // Get from loaded model
    int img_size = unet_model.image_size;     // Get from loaded model

    std::vector<float> dummy_noisy_image = generate_random_noise(img_channels, img_size, img_size);
    std::vector<long> dummy_timesteps = {500}; // Example timestep
    std::vector<int> dummy_labels = {0};    // Example class label (e.g., for CIFAR-10, class 0)

    std::cout << "Running dummy inference..." << std::endl;
    auto start_inference = std::chrono::high_resolution_clock::now();

    // 4. Run the forward pass of the UNet
    std::vector<float> output_denoised_image = unet_model.forward(
        dummy_noisy_image, batch_size, dummy_timesteps, dummy_labels, precomputed_lut
    );

    auto end_inference = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> inference_duration = end_inference - start_inference;
    std::cout << "Inference completed in " << inference_duration.count() << " seconds." << std::endl;

    // 5. Process the output (e.g., save as an image, print stats)
    if (!output_denoised_image.empty()) {
        std::cout << "Output image size: " << output_denoised_image.size() << " elements." << std::endl;
        // You'll need a function to save this float vector as an image (e.g., PNG, BMP)
        // For example, you might adapt `save_image_from_float_array` if you have one.
        // save_image_from_float_array("output_image.png", output_denoised_image, img_channels, img_size, img_size);
        std::cout << "Dummy inference successful. (Image saving not implemented in this test)." << std::endl;
    } else {
        std::cerr << "Inference returned empty output." << std::endl;
    }

    std::cout << "QAT UNet Deployment Test Finished." << std::endl;

    return 0;
}
