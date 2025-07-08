#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <stdexcept>
#include <filesystem>
#include <iomanip>
#include "qat_unet_model.h"
#include "utils.h"
#include "kernels.h"
#include "stb_image_write.h"
#include "profiler.h"

// ===================================================================
// 1. 重写的 DiffusionConstants 结构体和设置函数
// ===================================================================

Tensor randn_like(const std::vector<size_t>& shape, std::mt19937& generator) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    Tensor t(shape);
    for (size_t i = 0; i < t.data.size(); ++i) {
        t.data[i] = dist(generator);
    }
    return t;
}

DiffusionConstants setup_diffusion_constants(int timesteps, float beta_start = 0.0001f, float beta_end = 0.02f) {
    DiffusionConstants dc;
    dc.num_timesteps = timesteps;

    // 创建 betas
    dc.betas.resize(timesteps);
    for (int i = 0; i < timesteps; ++i) {
        dc.betas[i] = beta_start + (static_cast<float>(i) / static_cast<float>(timesteps - 1)) * (beta_end - beta_start);
    }

    // 预计算所有需要的系数
    std::vector<float> alphas(timesteps);
    for(int i = 0; i < timesteps; ++i) {
        alphas[i] = 1.0f - dc.betas[i];
    }

    dc.alphas_cumprod.resize(timesteps);
    dc.alphas_cumprod.front() = alphas.front();
    for(int i = 1; i < timesteps; ++i) {
        dc.alphas_cumprod[i] = dc.alphas_cumprod[i-1] * alphas[i];
    }

    dc.alphas_cumprod_prev.resize(timesteps);
    dc.alphas_cumprod_prev[0] = 1.0f; // 特殊情况 t=0
    std::copy(dc.alphas_cumprod.begin(), dc.alphas_cumprod.end() - 1, dc.alphas_cumprod_prev.begin() + 1);

    dc.sqrt_alphas_cumprod.resize(timesteps);
    dc.sqrt_one_minus_alphas_cumprod.resize(timesteps);
    dc.posterior_variance.resize(timesteps);
    dc.posterior_mean_coef1.resize(timesteps);
    dc.posterior_mean_coef2.resize(timesteps);

    for (int i = 0; i < timesteps; ++i) {
        dc.sqrt_alphas_cumprod[i] = std::sqrt(dc.alphas_cumprod[i]);
        dc.sqrt_one_minus_alphas_cumprod[i] = std::sqrt(1.0f - dc.alphas_cumprod[i]);
        
        // 后验方差: (1 - ᾱ_{t-1}) / (1 - ᾱ_t) * β_t
        dc.posterior_variance[i] = (1.0f - dc.alphas_cumprod_prev[i]) / (1.0f - dc.alphas_cumprod[i]) * dc.betas[i];

        // 后验均值系数1 (用于 pred_x0): sqrt(ᾱ_{t-1}) * β_t / (1 - ᾱ_t)
        dc.posterior_mean_coef1[i] = std::sqrt(dc.alphas_cumprod_prev[i]) * dc.betas[i] / (1.0f - dc.alphas_cumprod[i]);

        // 后验均值系数2 (用于 x_t): sqrt(α_t) * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        dc.posterior_mean_coef2[i] = std::sqrt(alphas[i]) * (1.0f - dc.alphas_cumprod_prev[i]) / (1.0f - dc.alphas_cumprod[i]);
    }
    return dc;
}

// (无需改动 verify_final_conv_weights 和 print_channel_means 函数)
void print_channel_means(const Tensor& tensor, int timestep) {
    if (tensor.shape.size() != 4 || tensor.shape[1] != 3) {
        // Silently ignore if not a 3-channel image tensor
        return;
    }

    const size_t num_pixels_per_channel = tensor.shape[2] * tensor.shape[3];
    if (num_pixels_per_channel == 0) return;

    std::vector<double> sums = {0.0, 0.0, 0.0}; // Use double for precision in sum

    // The data is in NCHW format
    for (size_t c = 0; c < 3; ++c) {
        for (size_t i = 0; i < num_pixels_per_channel; ++i) {
            sums[c] += tensor.data[c * num_pixels_per_channel + i];
        }
    }
    
    // Set up nice formatting for cout
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  [t =" << std::setw(4) << timestep << "] Channel Means -> R: " << std::setw(10) << sums[0] / num_pixels_per_channel
              << " | G: " << std::setw(10) << sums[1] / num_pixels_per_channel
              << " | B: " << std::setw(10) << sums[2] / num_pixels_per_channel << std::endl;
    // Reset cout formatting to default
    std::cout.unsetf(std::ios_base::floatfield);
    std::cout << std::defaultfloat;
}



// ===================================================================
// 2. 重写的 main 函数
// ===================================================================

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <path_to_ema_model.bin> <class_label> <num_timesteps> <output_image.png>" << std::endl;
        return 1;
    }
    std::string model_path = argv[1];
    int class_label = std::stoi(argv[2]);
    int num_timesteps = std::stoi(argv[3]);
    std::string final_output_image_path = argv[4];
    
    // (创建文件夹的逻辑无需改动)
    std::string output_dir = "output";
    size_t last_slash_pos = final_output_image_path.find_last_of("/\\");
    if (last_slash_pos != std::string::npos) {
        output_dir = final_output_image_path.substr(0, last_slash_pos);
    }
    output_dir += "/diffusion_steps";

    try {
        if (!std::filesystem::exists(output_dir)) {
            std::filesystem::create_directories(output_dir);
        }

        QATUNetModel model;
        model.load_model(model_path);
        std::cout << "Successfully loaded EMA model." << std::endl;

        DiffusionConstants dc = setup_diffusion_constants(num_timesteps, 1e-4f, 0.02f);
        std::cout << "Diffusion constants set for " << num_timesteps << " timesteps." << std::endl;

        std::mt19937 gen(std::random_device{}());
        Tensor image_tensor = randn_like({1, (size_t)model.in_channels, (size_t)model.image_size, (size_t)model.image_size}, gen);

        std::cout << "Starting sampling for class " << class_label << "..." << std::endl;
        
        // --- 核心采样循环 (完全重写) ---
        for (int t = num_timesteps - 1; t >= 0; --t) {
            std::cout << "Processing timestep " << t << "..." << std::endl;
            
            Tensor time_tensor({1});
            time_tensor.data[0] = static_cast<float>(t);
            Tensor class_tensor({1});
            class_tensor.data[0] = static_cast<float>(class_label);

            Tensor predicted_noise = forward_qat_unet(model, image_tensor, time_tensor, class_tensor);
            
            // 步骤 1: 预测去噪后的图像 pred_x0
            // pred_x0 = (x_t - sqrt(1-ᾱ_t) * ε) / sqrt(ᾱ_t)
            Tensor pred_x0_term1 = predicted_noise.mul_scalar(dc.sqrt_one_minus_alphas_cumprod[t]);
            Tensor pred_x0_term2 = image_tensor.sub(pred_x0_term1);
            Tensor pred_x0 = pred_x0_term2.mul_scalar(1.0f / dc.sqrt_alphas_cumprod[t]);
            // (可选) 在这里对 pred_x0 进行 clamp 操作以增加稳定性
            // pred_x0.clamp(-1.0f, 1.0f);

            // 步骤 2: 使用 pred_x0 计算 x_{t-1} 的均值
            // posterior_mean = coef1 * pred_x0 + coef2 * x_t
            Tensor posterior_mean_term1 = pred_x0.mul_scalar(dc.posterior_mean_coef1[t]);
            Tensor posterior_mean_term2 = image_tensor.mul_scalar(dc.posterior_mean_coef2[t]);
            Tensor posterior_mean = posterior_mean_term1.add(posterior_mean_term2);

            // 步骤 3: 从后验分布中采样 x_{t-1}
            if (t > 0) {
                Tensor noise = randn_like(image_tensor.shape, gen);;
                float std_dev = std::sqrt(dc.posterior_variance[t]);
                image_tensor = posterior_mean.add(noise.mul_scalar(std_dev));
            } else {
                image_tensor = posterior_mean; // 最后一步是确定性的
            }
            
            print_channel_means(image_tensor, t);
            // Save image every 10 diffusion steps, and at the last step (t=0)
            if ((num_timesteps - 1 - t) % 10 == 0 || t == 0) {
                std::string step_image_filename = output_dir + "/step_" + std::to_string(num_timesteps - 1 - t) + ".png";
                if (!save_image_from_float_array(step_image_filename, image_tensor.data, model.in_channels, model.image_size, model.image_size)) {
                    std::cerr << "Error: Failed to save intermediate image for step " << (num_timesteps - 1 - t) << std::endl;
                } else {
                    std::cout << "Intermediate image saved to " << step_image_filename << std::endl;
                }
            }
        }
        std::cout << "Sampling complete." << std::endl;
        
        Profiler::getInstance().report();
        if (!save_image_from_float_array(final_output_image_path, image_tensor.data, model.in_channels, model.image_size, model.image_size)) {
            std::cerr << "Error: Failed to save the final image." << std::endl;
        } else {
            std::cout << "Final image saved successfully to " << final_output_image_path << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}