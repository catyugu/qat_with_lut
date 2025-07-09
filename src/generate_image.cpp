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
// (无需改动 verify_final_conv_weights 和 print_channel_means 函数)
void print_channel_means(const Tensor& tensor, int timestep) {
    if (tensor.shape.size() != 4 || tensor.shape[1] != 3) {
        // Silently ignore if not a 3-channel image tensor
        return;
    }
    const size_t num_pixels_per_channel = tensor.shape[2] * tensor.shape[3];
    if (num_pixels_per_channel == 0) return;

    std::vector<float> sums = {0.0, 0.0, 0.0}; 

    // The data is in NCHW format
    for (size_t c = 0; c < 3; ++c) {
        for (size_t i = 0; i < tensor.shape[2]; ++i) {
            for (size_t j = 0; j < tensor.shape[3]; ++ j){
                sums[c] += tensor.at({0,c,i,j});
            }
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

        DiffusionConstants dc = calculateDiffusionConstants(num_timesteps, 1e-4, 0.02);

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
            
            Tensor pred_x0_term1 = predicted_noise.mul_scalar(1.0f/dc.sqrt_one_minus_alphas_cumprod[t]).mul_scalar(dc.betas[t]);
            image_tensor  = (image_tensor.sub(pred_x0_term1)).mul_scalar(std::sqrt(1.0f / dc.alphas[t]));

            if (t > 0) {
                Tensor noise = randn_like(image_tensor.shape, gen);;
                float std_dev = std::sqrt(dc.betas[t]);
                image_tensor = image_tensor.add(noise.mul_scalar(std_dev));
            } 
            print_channel_means(image_tensor, t);

            // Save image every 10 diffusion steps, and at the last step (t=0)
            if ((num_timesteps - 1 - t) % 10 == 0) {
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