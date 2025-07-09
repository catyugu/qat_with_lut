#ifndef QAT_UNET_MODEL_H
#define QAT_UNET_MODEL_H

#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include "types.h" // Include this to get the Module base class and layer definitions

struct DiffusionConstants {
    int num_timesteps;
    std::vector<float> betas;
    std::vector<float> alphas;
    std::vector<float> alphas_cumprod;
    std::vector<float> sqrt_alphas_cumprod;
    std::vector<float> sqrt_one_minus_alphas_cumprod;
    std::vector<float> posterior_variance;
    std::vector<float> posterior_mean_coef1;
    std::vector<float> posterior_mean_coef2;
};
class QATUNetModel {
public:
    // --- Hyperparameters ---
    int image_size;
    int in_channels;
    int base_channels;
    std::vector<int> channel_mults;
    int num_res_blocks;
    int time_emb_dim;
    float time_emb_scale;
    int num_classes;
    float dropout;
    std::vector<int> attention_resolutions;
    int num_groups;
    int initial_pad;
    bool use_scale_shift_norm;

    DiffusionConstants diffusion_constants;
    
    // --- Model Layers ---
    std::unique_ptr<PositionalEmbedding> time_mlp_pos_emb;
    std::unique_ptr<LinearLayer> time_mlp_linear1;
    std::unique_ptr<LinearLayer> time_mlp_linear2;

    std::unique_ptr<QATConv2dLayer> init_conv;

    // Use polymorphic containers for mixed layer types
    std::vector<std::unique_ptr<Module>> downs;

    std::unique_ptr<QATResidualBlock> middle_block1;
    std::unique_ptr<QATResidualBlock> middle_block2;

    std::vector<std::unique_ptr<Module>> ups;

    std::unique_ptr<GroupNormLayer> final_norm;
    std::unique_ptr<QATConv2dLayer> final_conv;

    // --- Methods ---
    void load_model(const std::string& model_path);

private:
    // Helper functions to read specific layer types from the file
    void read_conv_layer(std::ifstream& file, QATConv2dLayer& layer);
    void read_linear_layer(std::ifstream& file, LinearLayer& layer);
    void read_norm_layer(std::ifstream& file, GroupNormLayer& layer);
    void read_attention_block(std::ifstream& file, AttentionBlock& block);
    void read_res_block(std::ifstream& file, QATResidualBlock& block, int in_ch, int out_ch, bool use_attention);
};


#endif // QAT_UNET_MODEL_H
