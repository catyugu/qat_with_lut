#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <stdexcept>
#include "qat_unet_model.h"
#include "utils.h" // For Tensor operations
#include "kernels.h" // For the actual forward pass logic

// stb_image_write is a single-header library, so we just include it.
// The implementation is created in one C++ file by defining STB_IMAGE_WRITE_IMPLEMENTATION.
// Let's assume it's defined in utils.cpp or another suitable place.
// #define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


// --- Diffusion Constants Setup ---

// Struct to hold all the constants needed for the diffusion process
struct DiffusionConstants {
    int num_timesteps;
    std::vector<float> betas;
    std::vector<float> alphas;
    std::vector<float> alphas_cumprod;
    std::vector<float> alphas_cumprod_prev;
    std::vector<float> sqrt_alphas_cumprod;
    std::vector<float> sqrt_one_minus_alphas_cumprod;
    std::vector<float> posterior_variance;
};

// Calculates the diffusion schedule (betas, alphas, etc.)
DiffusionConstants setup_diffusion_constants(int timesteps) {
    DiffusionConstants dc;
    dc.num_timesteps = timesteps;

    // Linear beta schedule
    float start = 0.0001f;
    float end = 0.02f;
    dc.betas.resize(timesteps);
    for (int i = 0; i < timesteps; ++i) {
        dc.betas[i] = start + (float(i) / (timesteps - 1)) * (end - start);
    }

    // Pre-calculate alphas and their cumulative products
    dc.alphas.resize(timesteps);
    dc.alphas_cumprod.resize(timesteps);
    dc.alphas_cumprod_prev.resize(timesteps);
    dc.sqrt_alphas_cumprod.resize(timesteps);
    dc.sqrt_one_minus_alphas_cumprod.resize(timesteps);
    dc.posterior_variance.resize(timesteps);

    float current_alpha_cumprod = 1.0f;
    for (int i = 0; i < timesteps; ++i) {
        dc.alphas[i] = 1.0f - dc.betas[i];
        dc.alphas_cumprod_prev[i] = (i == 0) ? 1.0f : dc.alphas_cumprod[i-1];
        dc.alphas_cumprod[i] = current_alpha_cumprod * dc.alphas[i];
        current_alpha_cumprod = dc.alphas_cumprod[i];

        dc.sqrt_alphas_cumprod[i] = std::sqrt(dc.alphas_cumprod[i]);
        dc.sqrt_one_minus_alphas_cumprod[i] = std::sqrt(1.0f - dc.alphas_cumprod[i]);
        
        // For DDPM sampling: q(x_{t-1} | x_t, x_0)
        dc.posterior_variance[i] = dc.betas[i] * (1.0f - dc.alphas_cumprod_prev[i]) / (1.0f - dc.alphas_cumprod[i]);
    }
    return dc;
}


// --- Main Application Logic ---

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model.bin> <diffusion_steps> <output_image.png>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    int diffusion_steps = std::stoi(argv[2]);
    std::string output_path = argv[3];

    try {
        // 1. Load the UNet model
        QATUNetModel model;
        model.load_model(model_path);

        // 2. Setup diffusion constants
        DiffusionConstants dc = setup_diffusion_constants(diffusion_steps);
        
        // 3. Create initial random noise tensor
        std::cout << "Generating initial noise..." << std::endl;
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        Tensor image_tensor({1, (size_t)model.in_channels, (size_t)model.image_size, (size_t)model.image_size});
        for(size_t i = 0; i < image_tensor.data.size(); ++i) {
            image_tensor.data[i] = dist(gen);
        }

        // 4. The Denoising Loop (p_sample_loop)
        std::cout << "Starting denoising loop for " << diffusion_steps << " steps..." << std::endl;
        for (int t = diffusion_steps - 1; t >= 0; --t) {
            std::cout << "Step " << t << "..." << std::endl;
            
            // Prepare time tensor
            Tensor time_tensor({1});
            time_tensor.data[0] = (float)t;

            // --- This is where the forward pass of the model would be called ---
            // The `forward_qat_unet` function should be implemented in kernels.cpp
            // It would take the model, the current image_tensor, and the time_tensor
            // and return the predicted noise.
            //
            // Tensor predicted_noise = forward_qat_unet(model, image_tensor, time_tensor);
            //
            // For now, we'll use a placeholder tensor for predicted noise.
            // In a real scenario, this would be the model's output.
            Tensor predicted_noise = image_tensor; // Placeholder! Replace with actual model call.

            // 5. Denoise the image one step (p_sample)
            float alpha_t = dc.alphas[t];
            float alpha_t_cumprod = dc.alphas_cumprod[t];
            float sqrt_one_minus_alpha_t_cumprod = dc.sqrt_one_minus_alphas_cumprod[t];
            float one_over_sqrt_alpha_t = 1.0f / std::sqrt(alpha_t);

            // image = (1/sqrt(alpha_t)) * (image - ((1-alpha_t)/sqrt(1-alpha_cumprod_t)) * predicted_noise)
            Tensor term2 = predicted_noise.mul_scalar((1.0f - alpha_t) / sqrt_one_minus_alpha_t_cumprod);
            image_tensor = image_tensor.sub(term2);
            image_tensor = image_tensor.mul_scalar(one_over_sqrt_alpha_t);

            // Add noise back in if not the last step
            if (t > 0) {
                float posterior_variance = dc.posterior_variance[t];
                float noise_scale = std::sqrt(posterior_variance);
                
                Tensor noise_tensor({1, (size_t)model.in_channels, (size_t)model.image_size, (size_t)model.image_size});
                for(size_t i = 0; i < noise_tensor.data.size(); ++i) {
                    noise_tensor.data[i] = dist(gen);
                }
                image_tensor = image_tensor.add(noise_tensor.mul_scalar(noise_scale));
            }
        }
        std::cout << "Denoising complete." << std::endl;

        // 6. Convert final tensor to image and save
        std::cout << "Saving image to " << output_path << std::endl;
        int width = model.image_size;
        int height = model.image_size;
        int channels = model.in_channels;
        std::vector<uint8_t> image_data(width * height * channels);

        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                for (int c = 0; c < channels; ++c) {
                    // Get float value from tensor, assuming CHW layout
                    float val = image_tensor.data[c * (width * height) + h * width + w];
                    // Clamp and scale from [-1, 1] to [0, 255]
                    val = (val + 1.0f) / 2.0f; // Scale to [0, 1]
                    val = std::max(0.0f, std::min(1.0f, val)); // Clamp
                    image_data[(h * width + w) * channels + c] = static_cast<uint8_t>(val * 255.0f);
                }
            }
        }

        if (!stbi_write_png(output_path.c_str(), width, height, channels, image_data.data(), width * channels)) {
            throw std::runtime_error("Failed to write output image.");
        }
        std::cout << "Image successfully saved." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
