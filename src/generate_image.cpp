#include "qat_unet.h"
#include "utils.h" // For saving images, etc.
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

// --- Diffusion Hyperparameters (should match training) ---
const int TIMESTEPS = 1000;
const float BETA_START = 0.0001f;
const float BETA_END = 0.02f;

// --- Function to precompute diffusion schedule values ---
void get_diffusion_schedule(
    std::vector<float>& betas,
    std::vector<float>& alphas,
    std::vector<float>& alphas_cumprod
) {
    betas.resize(TIMESTEPS);
    alphas.resize(TIMESTEPS);
    alphas_cumprod.resize(TIMESTEPS);

    for (int t = 0; t < TIMESTEPS; ++t) {
        betas[t] = BETA_START + (float(t) / (TIMESTEPS - 1)) * (BETA_END - BETA_START);
        alphas[t] = 1.0f - betas[t];
        alphas_cumprod[t] = (t > 0 ? alphas_cumprod[t - 1] : 1.0f) * alphas[t];
    }
}

// --- Time Embedding ---
void get_time_embedding(float t, int dim, std::vector<float>& emb) {
    emb.resize(dim);
    float half_dim = dim / 2.0f;
    float freq = -std::log(10000.0f) / (half_dim - 1.0f);
    for (int i = 0; i < dim / 2; ++i) {
        float val = std::exp(i * freq) * t;
        emb[i] = std::cos(val);
        emb[i + dim / 2] = std::sin(val);
    }
}


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model.bin> <output_image.png>" << std::endl;
        return 1;
    }
    std::string model_path = argv[1];
    std::string output_path = argv[2];

    // 1. Load the accelerated QAT U-Net model
    QAT_UNet_CPP unet;
    if (!load_qat_unet(model_path, unet)) {
        return 1;
    }

    // 2. Get diffusion schedule
    std::vector<float> betas, alphas, alphas_cumprod;
    get_diffusion_schedule(betas, alphas, alphas_cumprod);

    // 3. Initialize random noise (the starting point for generation)
    // Assuming CIFAR-10: 3 channels, 32x32 pixels
    const int C = 3, H = 32, W = 32;
    std::vector<float> x(C * H * W);
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 1.0);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = distribution(generator);
    }

    // 4. The Denoising Loop (from T-1 down to 0)
    std::cout << "Starting denoising process..." << std::endl;
    for (int t = TIMESTEPS - 1; t >= 0; --t) {
        if (t % 100 == 0) std::cout << "Timestep " << t << "/" << TIMESTEPS << std::endl;

        // a. Get time embedding
        std::vector<float> time_emb;
        get_time_embedding(static_cast<float>(t), unet.model_channels * 4, time_emb);

        // b. Denoise using the U-Net to predict noise
        std::vector<float> predicted_noise;
        unet.forward(x, time_emb, predicted_noise);

        // c. Update the image (remove predicted noise)
        // This is the core DDPM sampling step
        float alpha_t = alphas[t];
        float alpha_cumprod_t = alphas_cumprod[t];
        float one_minus_alpha_t = 1.0f - alpha_t;
        float one_minus_alpha_cumprod_t = 1.0f - alpha_cumprod_t;

        for (size_t i = 0; i < x.size(); ++i) {
            float term1 = (1.0f / std::sqrt(alpha_t)) * (x[i] - (one_minus_alpha_t / std::sqrt(one_minus_alpha_cumprod_t)) * predicted_noise[i]);
            x[i] = term1;

            if (t > 0) {
                float sigma_t = std::sqrt(betas[t]); // Simplified sigma
                x[i] += sigma_t * distribution(generator);
            }
        }
    }
    std::cout << "Denoising complete." << std::endl;


    // 5. Post-process and save the final image
    // The 'x' vector now holds the generated image.
    // You'll need to convert it from [-1, 1] to [0, 255] and save it.
    // (This requires an image library like stb_image_write or similar)
    save_image_from_float_vec(x, H, W, C, output_path);
    std::cout << "Image saved to " << output_path << std::endl;

    return 0;
}
