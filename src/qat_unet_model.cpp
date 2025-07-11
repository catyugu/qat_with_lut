// src/qat_unet_model.cpp (修正版)
#include "qat_unet_model.h"
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <algorithm>


// --- Helper functions to read binary data with debugging ---
static void log_pos(std::ifstream& file, const std::string& msg) {
    // std::cout << "[DEBUG] " << msg << " at file position: " << file.tellg() << std::endl;
}

template<typename T>
static void read_from_file(std::ifstream& file, T& value, const std::string& name) {
    log_pos(file, "Attempting to read '" + name + "' (" + std::to_string(sizeof(T)) + " bytes)");
    file.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!file) {
        throw std::runtime_error("Error reading value for '" + name + "'. File might be truncated or corrupt.");
    }
}

template<typename T>
static void read_vector(std::ifstream& file, std::vector<T>& vec, const std::string& name) {
    if (vec.empty()) return;
    log_pos(file, "Attempting to read vector '" + name + "' (" + std::to_string(vec.size() * sizeof(T)) + " bytes)");
    file.read(reinterpret_cast<char*>(vec.data()), vec.size() * sizeof(T));
    if (!file) {
        throw std::runtime_error("Error reading vector data for '" + name + "'. File might be truncated or corrupt.");
    }
}

// --- Layer Reading Implementations ---
void QATUNetModel::read_conv_layer(std::ifstream& file, QATConv2dLayer& layer) {
    // Read standard convolution parameters
    read_from_file(file, layer.in_channels, "conv.in_channels");
    read_from_file(file, layer.out_channels, "conv.out_channels");
    read_from_file(file, layer.kernel_size_h, "conv.kernel_h");
    read_from_file(file, layer.kernel_size_w, "conv.kernel_w");
    read_from_file(file, layer.stride_h, "conv.stride_h");
    read_from_file(file, layer.stride_w, "conv.stride_w");
    read_from_file(file, layer.pad_h, "conv.pad_h");
    read_from_file(file, layer.pad_w, "conv.pad_w");
    read_from_file(file, layer.groups, "conv.groups");

    // Read the quantization scaling factor (alpha)
    read_from_file(file, layer.alpha, "conv.alpha");

    // === NEW: Read the original weight tensor's shape ===
    int shape_dims;
    read_from_file(file, shape_dims, "conv.weight_shape_dims");
    layer.shape.resize(shape_dims);
    read_vector(file, layer.shape, "conv.weight_shape");
    // ===================================================

    // Read the packed uint32_t weights
    int packed_weights_size;
    read_from_file(file, packed_weights_size, "conv.packed_weights_size");
    layer.packed_weights.resize(packed_weights_size);
    read_vector(file, layer.packed_weights, "conv.packed_weights");

    // Read the float bias vector
    int bias_size;
    read_from_file(file, bias_size, "conv.bias_size");
    layer.bias.resize(bias_size);
    read_vector(file, layer.bias, "conv.bias");
}

void QATUNetModel::read_linear_layer(std::ifstream& file, LinearLayer& layer) {
    read_from_file(file, layer.in_features, "linear.in_features");
    read_from_file(file, layer.out_features, "linear.out_features");

    int weights_size;
    read_from_file(file, weights_size, "linear.weights_size");
    layer.weights.resize(weights_size);
    read_vector(file, layer.weights, "linear.weights");

    int bias_size;
    read_from_file(file, bias_size, "linear.bias_size");
    layer.bias.resize(bias_size);
    read_vector(file, layer.bias, "linear.bias");
}

void QATUNetModel::read_norm_layer(std::ifstream& file, GroupNormLayer& layer) {
    read_from_file(file, layer.num_groups, "norm.num_groups");
    read_from_file(file, layer.num_channels, "norm.num_channels");
    read_from_file(file, layer.eps, "norm.eps");
    int weight_size, bias_size;
    read_from_file(file, weight_size, "norm.weight_size");
    layer.weight.resize(weight_size);
    read_vector(file, layer.weight, "norm.weight");
    read_from_file(file, bias_size, "norm.bias_size");
    layer.bias.resize(bias_size);
    read_vector(file, layer.bias, "norm.bias");
}

void QATUNetModel::read_attention_block(std::ifstream& file, AttentionBlock& block) {
    std::cout << "  Reading Nested AttentionBlock..." << std::endl;
    read_norm_layer(file, *block.norm);
    read_conv_layer(file, *block.to_qkv);
    read_conv_layer(file, *block.to_out);
}

void QATUNetModel::read_res_block(std::ifstream& file, QATResidualBlock& block, int in_ch, int out_ch, bool use_attention) {
    std::cout << " Reading QATResidualBlock (in=" << in_ch << ", out=" << out_ch << ")..." << std::endl;
    
    // norm_1
    block.norm_1 = std::make_unique<GroupNormLayer>();
    read_norm_layer(file, *block.norm_1);

    // conv_1
    block.conv_1 = std::make_unique<QATConv2dLayer>();
    read_conv_layer(file, *block.conv_1);

    // time_bias
    bool has_time_bias;
    read_from_file(file, has_time_bias, "res_block.has_time_bias");
    if (has_time_bias) {
        block.time_bias = std::make_unique<LinearLayer>();
        read_linear_layer(file, *block.time_bias);
    }

    // class_bias
    bool has_class_bias;
    read_from_file(file, has_class_bias, "res_block.has_class_bias");
    if (has_class_bias) {
        block.class_bias = std::make_unique<Embedding>();
        read_from_file(file, block.class_bias->num_embeddings, "class_bias.num_embeddings");
        read_from_file(file, block.class_bias->embedding_dim, "class_bias.embedding_dim");
        int weight_size;
        read_from_file(file, weight_size, "class_bias.weight_size");
        block.class_bias->weight.resize(weight_size);
        read_vector(file, block.class_bias->weight, "class_bias.weight");
    }

    // norm_2
    block.norm_2 = std::make_unique<GroupNormLayer>();
    read_norm_layer(file, *block.norm_2);

    // conv_2 (part of a sequential block in python)
    block.conv_2 = std::make_unique<QATConv2dLayer>();
    read_conv_layer(file, *block.conv_2);

    // residual_connection
    bool has_residual_connection;
    read_from_file(file, has_residual_connection, "res_block.has_residual_connection");
    if (has_residual_connection) {
        block.residual_connection = std::make_unique<QATConv2dLayer>();
        read_conv_layer(file, *block.residual_connection);
    }

    // attention
    bool has_attention;
    read_from_file(file, has_attention, "res_block.has_attention");
    if (has_attention) {
        block.attention = std::make_unique<AttentionBlock>();
        read_attention_block(file, *block.attention);
    }
}


// --- Main Model Loading Logic ---

void QATUNetModel::load_model(const std::string& model_path) {
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open model file: " + model_path);
    }

    std::cout << "--- Starting Model Load ---" << std::endl;

    // 1. Read Hyperparameters
    std::cout << "Reading Hyperparameters..." << std::endl;
    read_from_file(file, image_size, "image_size");
    std::cout<<"Image size: " << image_size << std::endl;
    read_from_file(file, in_channels, "in_channels");
    std::cout<<"In channels: " << in_channels << std::endl;
    read_from_file(file, base_channels, "base_channels");
    int channel_mults_size;
    read_from_file(file, channel_mults_size, "channel_mults_size");
    channel_mults.resize(channel_mults_size);
    read_vector(file, channel_mults, "channel_mults");
    read_from_file(file, num_res_blocks, "num_res_blocks");
    read_from_file(file, time_emb_dim, "time_emb_dim");
    read_from_file(file, time_emb_scale, "time_emb_scale");
    read_from_file(file, num_classes, "num_classes");
    read_from_file(file, dropout, "dropout");
    int attention_resolutions_size;
    read_from_file(file, attention_resolutions_size, "attention_resolutions_size");
    attention_resolutions.resize(attention_resolutions_size);
    read_vector(file, attention_resolutions, "attention_resolutions");
    read_from_file(file, num_groups, "num_groups");
    read_from_file(file, initial_pad, "initial_pad");
    read_from_file(file, use_scale_shift_norm, "use_scale_shift_norm");
    std::cout << "Hyperparameters loaded." << std::endl;

    // 2. Read Time MLP
    std::cout << "\nReading Time MLP..." << std::endl;
    time_mlp_pos_emb = std::make_unique<PositionalEmbedding>(base_channels, time_emb_scale);
    time_mlp_linear1 = std::make_unique<LinearLayer>();
    read_linear_layer(file, *time_mlp_linear1);
    time_mlp_linear2 = std::make_unique<LinearLayer>();
    read_linear_layer(file, *time_mlp_linear2);
    std::cout << "Time MLP loaded." << std::endl;

    // 3. Read Initial Convolution
    std::cout << "\nReading Initial Convolution..." << std::endl;
    init_conv = std::make_unique<QATConv2dLayer>();
    read_conv_layer(file, *init_conv);
    std::cout << "Initial Convolution loaded." << std::endl;

    // 4. Read Downsampling Path and collect skip channels
    std::cout << "\nReading Downsampling Path..." << std::endl;
    std::vector<int> skip_channels_stack;
    int now_channels = base_channels;
    int current_res = image_size;

    skip_channels_stack.push_back(now_channels);

    for (size_t i = 0; i < channel_mults.size(); ++i) {
        int out_channels = base_channels * channel_mults[i];
        for (int j = 0; j < num_res_blocks; ++j) {
            bool use_attention = (std::find(attention_resolutions.begin(), attention_resolutions.end(), current_res) != attention_resolutions.end());
            auto res_block = std::make_unique<QATResidualBlock>(now_channels, out_channels, time_emb_dim, use_attention);
            read_res_block(file, *res_block, now_channels, out_channels, use_attention);
            downs.push_back(std::move(res_block));
            now_channels = out_channels;
            skip_channels_stack.push_back(now_channels);
        }
        if (i != channel_mults.size() - 1) {
            auto downsample = std::make_unique<DownsampleLayer>();
            read_conv_layer(file, *downsample->conv);
            downs.push_back(std::move(downsample));
            current_res /= 2;
            skip_channels_stack.push_back(now_channels);
        }
    }
    std::cout << "Downsampling Path loaded." << std::endl;

    // 5. Read Middle Path
    std::cout << "\nReading Middle Path..." << std::endl;
    middle_block1 = std::make_unique<QATResidualBlock>(now_channels, now_channels, time_emb_dim, true);
    read_res_block(file, *middle_block1, now_channels, now_channels, true);
    middle_block2 = std::make_unique<QATResidualBlock>(now_channels, now_channels, time_emb_dim, false);
    read_res_block(file, *middle_block2, now_channels, now_channels, false);
    std::cout << "Middle Path loaded." << std::endl;

    // 6. Read Upsampling Path
    std::cout << "\nReading Upsampling Path..." << std::endl;
    for (size_t i = 0; i < channel_mults.size(); ++i) {
        int level = channel_mults.size() - 1 - i;
        int out_channels = base_channels * channel_mults[level];
        for (int j = 0; j < num_res_blocks + 1; ++j) {
            int skip_ch = skip_channels_stack.back();
            skip_channels_stack.pop_back();
            int in_channels_up = now_channels + skip_ch;

            bool use_attention = (std::find(attention_resolutions.begin(), attention_resolutions.end(), current_res) != attention_resolutions.end());
            auto res_block = std::make_unique<QATResidualBlock>(in_channels_up, out_channels, time_emb_dim, use_attention);
            read_res_block(file, *res_block, in_channels_up, out_channels, use_attention);
            ups.push_back(std::move(res_block));
            now_channels = out_channels;
        }
        if (level != 0) {
            auto upsample = std::make_unique<UpsampleLayer>();
            read_conv_layer(file, *upsample->conv);
            ups.push_back(std::move(upsample));
            current_res *= 2;
        }
    }
    std::cout << "Upsampling Path loaded." << std::endl;

    // 7. Read Final Layers
    std::cout << "\nReading Final Layers..." << std::endl;
    final_norm = std::make_unique<GroupNormLayer>();
    read_norm_layer(file, *final_norm);
    final_conv = std::make_unique<QATConv2dLayer>();
    read_conv_layer(file, *final_conv);
    std::cout << "Final Layers loaded." << std::endl;

    std::cout << "\n--- Model Loading Complete ---" << std::endl;

    read_from_file(file, this->diffusion_constants.num_timesteps, "num_timesteps");
    this->diffusion_constants.betas.resize(this->diffusion_constants.num_timesteps);
    read_vector(file, this->diffusion_constants.betas, "betas");
    this->diffusion_constants.alphas.resize(this->diffusion_constants.num_timesteps);
    read_vector(file, this->diffusion_constants.alphas, "alphas");
    this->diffusion_constants.alphas_cumprod.resize(this->diffusion_constants.num_timesteps);
    read_vector(file, this->diffusion_constants.alphas_cumprod, "alphas_cumprod");
    this->diffusion_constants.sqrt_alphas_cumprod.resize(this->diffusion_constants.num_timesteps);
    read_vector(file, this->diffusion_constants.sqrt_alphas_cumprod, "sqrt_alphas_cumprod");
    this->diffusion_constants.sqrt_one_minus_alphas_cumprod.resize(this->diffusion_constants.num_timesteps);
    read_vector(file, this->diffusion_constants.sqrt_one_minus_alphas_cumprod, "sqrt_one_minus_alphas_cumprod");
    this->diffusion_constants.posterior_variance.resize(this->diffusion_constants.num_timesteps);
    read_vector(file, this->diffusion_constants.posterior_variance, "posterior_variance");
    this->diffusion_constants.posterior_mean_coef1.resize(this->diffusion_constants.num_timesteps);
    read_vector(file, this->diffusion_constants.posterior_mean_coef1, "posterior_mean_coef1");
    this->diffusion_constants.posterior_mean_coef2.resize(this->diffusion_constants.num_timesteps);
    read_vector(file, this->diffusion_constants.posterior_mean_coef2, "posterior_mean_coef2");

    // 作为一个临时的占位符（在你修改Python脚本前），你可以继续使用旧的计算方式
    // 但最终的目标是替换掉它！
    // 【临时代码，最终应删除】
    // 为了让代码能编译，我们暂时保留计算逻辑，但请务必尽快替换为加载逻辑
    // DiffusionConstants temp_dc = setup_diffusion_constants_placeholder(1000); // 假设1000个时间步
    // this->diffusion_constants = temp_dc;
    
    file.peek();
    if (!file.eof()) {
        std::cerr << "Warning: Model file was not fully read. There might be a structural mismatch." << std::endl;
    }
    std::cout << "Diffusion constants loaded/initialized." << std::endl;
}
DiffusionConstants calculateDiffusionConstants(int num_timesteps, float beta_low, float beta_high) {
    DiffusionConstants constants;
    constants.num_timesteps = num_timesteps;

    // 预分配向量大小
    constants.betas.reserve(num_timesteps);
    constants.alphas.reserve(num_timesteps);
    constants.alphas_cumprod.reserve(num_timesteps);
    constants.sqrt_alphas_cumprod.reserve(num_timesteps);
    constants.sqrt_one_minus_alphas_cumprod.reserve(num_timesteps);
    constants.posterior_variance.reserve(num_timesteps);
    constants.posterior_mean_coef1.reserve(num_timesteps);
    constants.posterior_mean_coef2.reserve(num_timesteps);

    // 1. 计算 betas (线性插值)
    for (int i = 0; i < num_timesteps; ++i) {
        float beta;
        if (num_timesteps > 1) {
            beta = beta_low + (beta_high - beta_low) * i / (static_cast<float>(num_timesteps) - 1.0f);
        } else {
            beta = beta_low; // num_timesteps == 1 时，beta_low 和 beta_high 应该相同
        }
        constants.betas.push_back(beta);
    }

    // alphas_cumprod_prev 用于后续计算，其第一个元素为 1.0
    std::vector<float> alphas_cumprod_prev;
    alphas_cumprod_prev.reserve(num_timesteps + 1);
    alphas_cumprod_prev.push_back(1.0f); // T=0 时的 alpha_cumprod 为 1.0

    float current_alphas_cumprod = 1.0f; // 累计乘积的初始值

    for (int t = 0; t < num_timesteps; ++t) {
        // 2. alphas = 1.0 - betas
        float alpha = 1.0f - constants.betas[t];
        constants.alphas.push_back(alpha);

        // 3. alphas_cumprod (累计乘积)
        current_alphas_cumprod *= alpha;
        constants.alphas_cumprod.push_back(current_alphas_cumprod);
        
        // 记录当前 alphas_cumprod 为下一个 t 的 prev 值
        if (t < num_timesteps -1) { // alphas_cumprod_prev 会比 alphas_cumprod 多一个元素
            alphas_cumprod_prev.push_back(current_alphas_cumprod);
        }
    }

    // 4. sqrt_alphas_cumprod
    // 5. sqrt_one_minus_alphas_cumprod
    for (int t = 0; t < num_timesteps; ++t) {
        constants.sqrt_alphas_cumprod.push_back(std::sqrt(constants.alphas_cumprod[t]));
        constants.sqrt_one_minus_alphas_cumprod.push_back(std::sqrt(1.0f - constants.alphas_cumprod[t]));

        // 6. posterior_variance
        // 公式: betas[t] * (1.0 - alphas_cumprod_prev[t]) / (1.0 - alphas_cumprod[t])
        float denom = 1.0f - constants.alphas_cumprod[t];
        float posterior_var = 0.0f;
        if (denom > 1e-12f) { // 避免除以零或极小值
            posterior_var = constants.betas[t] * (1.0f - alphas_cumprod_prev[t]) / denom;
        }
        constants.posterior_variance.push_back(posterior_var);

        // 7. posterior_mean_coef1
        // 公式: betas[t] * sqrt(alphas_cumprod_prev[t]) / (1.0 - alphas_cumprod[t])
        float coef1 = 0.0f;
        if (denom > 1e-12f) {
            coef1 = constants.betas[t] * std::sqrt(alphas_cumprod_prev[t]) / denom;
        }
        constants.posterior_mean_coef1.push_back(coef1);

        // 8. posterior_mean_coef2
        // 公式: (1.0 - alphas_cumprod_prev[t]) * sqrt(alphas[t]) / (1.0 - alphas_cumprod[t])
        float coef2 = 0.0f;
        if (denom > 1e-6f) {
            coef2 = (1.0f - alphas_cumprod_prev[t]) * std::sqrt(constants.alphas[t]) / denom;
        }
        constants.posterior_mean_coef2.push_back(coef2);
    }

    return constants;
}