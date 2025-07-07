// src/generate_image.cpp

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <stdexcept>
#include "qat_unet_model.h"
#include "utils.h"
#include "kernels.h"
#include "profiler.h"
#include "stb_image_write.h"

// --- Diffusion Constants Setup ---
// 结构体现在包含与 Python 版本完全匹配的系数
struct DiffusionConstants {
    int num_timesteps;
    std::vector<float> betas;
    std::vector<float> alphas;
    std::vector<float> alphas_cumprod;
    std::vector<float> sqrt_alphas_cumprod;
    std::vector<float> sqrt_one_minus_alphas_cumprod;
    std::vector<float> reciprocal_sqrt_alphas;
    std::vector<float> remove_noise_coeff;
    std::vector<float> sigma;
};

// 这个函数的实现现在与 diffusion.py 完全对齐
DiffusionConstants setup_diffusion_constants(int timesteps) {
    DiffusionConstants dc;
    dc.num_timesteps = timesteps;
    
    // 使用与 Python `generate_linear_schedule` 一致的线性插值
    float start = 0.0001f; // low
    float end = 0.02f;   // high
    dc.betas.resize(timesteps);
    for (int i = 0; i < timesteps; ++i) {
        dc.betas[i] = start + (float(i) / (timesteps - 1)) * (end - start);
    }
    
    dc.alphas.resize(timesteps);
    dc.alphas_cumprod.resize(timesteps);
    dc.sqrt_alphas_cumprod.resize(timesteps);
    dc.sqrt_one_minus_alphas_cumprod.resize(timesteps);
    dc.reciprocal_sqrt_alphas.resize(timesteps);
    dc.remove_noise_coeff.resize(timesteps);
    dc.sigma.resize(timesteps);

    float current_alpha_cumprod = 1.0f;
    for (int i = 0; i < timesteps; ++i) {
        dc.alphas[i] = 1.0f - dc.betas[i];
        current_alpha_cumprod *= dc.alphas[i];
        
        dc.alphas_cumprod[i] = current_alpha_cumprod;
        dc.sqrt_alphas_cumprod[i] = std::sqrt(dc.alphas_cumprod[i]);
        dc.sqrt_one_minus_alphas_cumprod[i] = std::sqrt(1.0f - dc.alphas_cumprod[i]);
        
        // 核心修正：确保这些系数的计算与 `diffusion.py` 完全一致
        dc.reciprocal_sqrt_alphas[i] = 1.0f / std::sqrt(dc.alphas[i]);
        dc.remove_noise_coeff[i] = dc.betas[i] / std::sqrt(1.0f - dc.alphas_cumprod[i]);
        dc.sigma[i] = std::sqrt(dc.betas[i]);
    }
    return dc;
}


// --- Main Application Logic ---

int main(int argc, char* argv[]) {
    if (argc != 5) { // 新增一个参数用于指定类别
        std::cerr << "Usage: " << argv[0] << " <path_to_model.bin> <diffusion_steps> <class_label> <output_image.png>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    int diffusion_steps = std::stoi(argv[2]);
    int class_label = std::stoi(argv[3]); // 获取类别标签
    std::string output_path = argv[4];

    try {
        // 1. 加载 UNet 模型
        QATUNetModel model;
        model.load_model(model_path);
        load_lut("ternary_lut.bin");
        
        // 2. 设置扩散常量
        DiffusionConstants dc = setup_diffusion_constants(diffusion_steps);
        
        // 3. 创建初始随机噪声张量
        std::cout << "Generating initial noise..." << std::endl;
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        Tensor image_tensor({1, (size_t)model.in_channels, (size_t)model.image_size, (size_t)model.image_size});
        for(size_t i = 0; i < image_tensor.data.size(); ++i) {
            image_tensor.data[i] = dist(gen);
        }

        // 4. 去噪循环
        std::cout << "Starting denoising loop for " << diffusion_steps << " steps..." << std::endl;
        Profiler::getInstance().reset(); 
        for (int t = diffusion_steps - 1; t >= 0; --t) {
 
            std::cout << "Step " << t << "..." << std::endl;
            
            Tensor time_tensor({1});
            time_tensor.data[0] = (float)t;

            // 使用命令行传入的类别标签
            Tensor class_tensor({1});
            class_tensor.data[0] = (float)class_label; 
            
            // 模型预测噪声
            Tensor predicted_noise = forward_qat_unet(model, image_tensor, time_tensor, class_tensor);

            // **核心修正**: 完全匹配 Python 的 `remove_noise` 逻辑
            // 1. 获取当前时间步 t 的系数
            float remove_coeff = dc.remove_noise_coeff[t];
            float recip_sqrt_alpha = dc.reciprocal_sqrt_alphas[t];

            // 2. 计算 (x - coeff * predicted_noise)
            Tensor denoised_term = image_tensor.sub(predicted_noise.mul_scalar(remove_coeff));
            
            // 3. 乘以 1 / sqrt(alpha_t)
            image_tensor = denoised_term.mul_scalar(recip_sqrt_alpha);

            // 4. 如果 t > 0，添加噪声 (匹配 Python 的 sample 逻辑)
            if (t > 0) {
                float sigma_t = dc.sigma[t];
                
                Tensor noise_tensor({1, (size_t)model.in_channels, (size_t)model.image_size, (size_t)model.image_size});
                for(size_t i = 0; i < noise_tensor.data.size(); ++i) {
                    noise_tensor.data[i] = dist(gen);
                }
                // 添加噪声: image_tensor += sigma_t * noise
                image_tensor = image_tensor.add(noise_tensor.mul_scalar(sigma_t));
            }
        }
        Profiler::getInstance().report();
        std::cout << "Denoising complete." << std::endl;

        // 5. 保存图像
        std::cout << "Saving final image to " << output_path << std::endl;
        if (!save_image_from_float_array(output_path, image_tensor.data, model.in_channels, model.image_size, model.image_size)) {
            std::cerr << "Failed to save the image." << std::endl;
        } else {
            std::cout << "Image saved successfully." << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}