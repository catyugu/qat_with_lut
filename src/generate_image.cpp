#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <stdexcept>
#include "qat_unet_model.h"
#include "utils.h"
#include "kernels.h" // Include the kernel declarations

// #define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


// --- Diffusion Constants Setup ---
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

DiffusionConstants setup_diffusion_constants(int timesteps) {
    DiffusionConstants dc;
    dc.num_timesteps = timesteps;
    float start = 0.0001f;
    float end = 0.02f;
    dc.betas.resize(timesteps);
    for (int i = 0; i < timesteps; ++i) {
        dc.betas[i] = start + (float(i) / (timesteps - 1)) * (end - start);
    }
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

        // 4. The Denoising Loop
        std::cout << "Starting denoising loop for " << diffusion_steps << " steps..." << std::endl;
        for (int t = diffusion_steps - 1; t >= 0; --t) {
            std::cout << "Step " << t << "..." << std::endl;
            
            Tensor time_tensor({1});
            time_tensor.data[0] = (float)t;

            // --- FIXED: Call the actual forward pass of the model ---
            Tensor predicted_noise = forward_qat_unet(model, image_tensor, time_tensor);

            // 5. Denoise the image one step
            float alpha_t = dc.alphas[t];
            float sqrt_one_minus_alpha_t_cumprod = dc.sqrt_one_minus_alphas_cumprod[t];
            float one_over_sqrt_alpha_t = 1.0f / std::sqrt(alpha_t);

            Tensor term2 = predicted_noise.mul_scalar((1.0f - alpha_t) / sqrt_one_minus_alpha_t_cumprod);
            image_tensor = image_tensor.sub(term2);
            image_tensor = image_tensor.mul_scalar(one_over_sqrt_alpha_t);

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
                    float val = image_tensor.at({0, (size_t)c, (size_t)h, (size_t)w});
                    val = (val + 1.0f) / 2.0f;
                    val = std::max(0.0f, std::min(1.0f, val));
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
