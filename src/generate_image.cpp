// src/generate_image.cpp
// --- FULLY CORRECTED VERSION ---

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <stdexcept>
#include "qat_unet_model.h"
#include "utils.h"
#include "kernels.h"
#include "stb_image_write.h"
#include "profiler.h"

// Defines all necessary diffusion coefficients
struct DiffusionConstants {
    int num_timesteps;
    std::vector<float> betas;
    std::vector<float> alphas;
    std::vector<float> alphas_cumprod;
    std::vector<float> reciprocal_sqrt_alphas;
    std::vector<float> remove_noise_coeff;
    std::vector<float> sigma;
};

// Sets up diffusion constants to EXACTLY match Python's generate_linear_schedule and GaussianDiffusion
DiffusionConstants setup_diffusion_constants(int timesteps, float beta_start = 0.0001f, float beta_end = 0.02f) {
    DiffusionConstants dc;
    dc.num_timesteps = timesteps;

    // Linear schedule for betas
    dc.betas.resize(timesteps);
    for (int i = 0; i < timesteps; ++i) {
        dc.betas[i] = beta_start + (static_cast<float>(i) / static_cast<float>(timesteps - 1)) * (beta_end - beta_start);
    }

    // Pre-calculate all other coefficients
    dc.alphas.resize(timesteps);
    dc.alphas_cumprod.resize(timesteps);
    dc.reciprocal_sqrt_alphas.resize(timesteps);
    dc.remove_noise_coeff.resize(timesteps);
    dc.sigma.resize(timesteps);

    float current_alpha_cumprod = 1.0f;
    for (int i = 0; i < timesteps; ++i) {
        dc.alphas[i] = 1.0f - dc.betas[i];
        current_alpha_cumprod *= dc.alphas[i];

        dc.alphas_cumprod[i] = current_alpha_cumprod;
        dc.reciprocal_sqrt_alphas[i] = 1.0f / std::sqrt(dc.alphas[i]);
        dc.remove_noise_coeff[i] = dc.betas[i] / std::sqrt(1.0f - dc.alphas_cumprod[i]);
        
        // Use posterior variance for sigma, which is sqrt(beta_t)
        dc.sigma[i] = std::sqrt(dc.betas[i]);
    }
    return dc;
}


int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <path_to_ema_model.bin> <class_label> <num_timesteps> <output_image.png>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    int class_label = std::stoi(argv[2]);
    int num_timesteps = std::stoi(argv[3]);
    std::string output_path = argv[4];

    try {
        // 1. Load the UNet model (ensure this is from the EMA weights)
        QATUNetModel model;
        model.load_model(model_path);
        std::cout << "Successfully loaded EMA model and LUT." << std::endl;

        // 2. Setup diffusion constants EXACTLY as in the Python script
        DiffusionConstants dc = setup_diffusion_constants(num_timesteps, 1e-4f, 0.02f);
        std::cout << "Diffusion constants set for " << num_timesteps << " timesteps." << std::endl;

        // 3. Create initial random noise tensor (the starting point)
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<float> dist(0.0f, 1.0f);

        Tensor image_tensor({1, (size_t)model.in_channels, (size_t)model.image_size, (size_t)model.image_size});
        for(size_t i = 0; i < image_tensor.data.size(); ++i) {
            image_tensor.data[i] = dist(gen);
        }

        // 4. Denoising loop (from t=999 down to 0)
        std::cout << "Starting sampling for class " << class_label << "..." << std::endl;
        Profiler::getInstance().reset(); 
        for (int t = num_timesteps - 1; t >= 0; --t) {
            std::cout << "Processing timestep " << t << "..." << std::endl;
            // Prepare time and class label tensors for the model
            Tensor time_tensor({1});
            time_tensor.data[0] = static_cast<float>(t);

            Tensor class_tensor({1});
            class_tensor.data[0] = static_cast<float>(class_label);

            // Predict noise using the UNet model
            Tensor predicted_noise = forward_qat_unet(model, image_tensor, time_tensor, class_tensor);

            // --- Core Denoising Logic (Mirrors python `remove_noise`) ---
            // 1. Get coefficients for the current timestep t
            float remove_coeff = dc.remove_noise_coeff[t];
            float recip_sqrt_alpha = dc.reciprocal_sqrt_alphas[t];

            // 2. Calculate: x - coeff * predicted_noise
            Tensor denoised_term = image_tensor.sub(predicted_noise.mul_scalar(remove_coeff));

            // 3. Update image: (x - coeff * predicted_noise) / sqrt(alpha_t)
            image_tensor = denoised_term.mul_scalar(recip_sqrt_alpha);
            // --- End of Core Denoising Logic ---

            // Add new noise if not the final step (t > 0)
            if (t > 0) {
                float sigma_t = dc.sigma[t];
                Tensor noise_tensor = Tensor::randn_like(image_tensor.shape); // Helper for random noise
                image_tensor = image_tensor.add(noise_tensor.mul_scalar(sigma_t));
            }
        }
        std::cout << "Sampling complete." << std::endl;
        Profiler::getInstance().report();
        if (!save_image_from_float_array(output_path, image_tensor.data, model.in_channels, model.image_size, model.image_size)) {
            std::cerr << "Error: Failed to save the final image." << std::endl;
        } else {
            std::cout << "Image saved successfully to " << output_path << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}