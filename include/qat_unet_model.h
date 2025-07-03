#ifndef QAT_UNET_MODEL_H
#define QAT_UNET_MODEL_H

#include <vector>
#include <cstdint> // For uint8_t, int8_t
#include <string>
#include <memory>  // For std::unique_ptr
#include <map>     // For std::map to store parameters

// Forward declarations for helper functions (might be in utils.h or kernels.h)
// These are just reminders of what you'll need.
// void im2col(const float* data_im, int channels, int height, int width,
//             int ksize_h, int ksize_w, int stride_h, int stride_w, int pad_h, int pad_w,
//             std::vector<float>& data_col);
// void quantize_float_to_int8_with_scale(const float* input, int8_t* output, size_t size, float scale);
// void pack_ternary_activations_5x3bit_to_ptr(const int8_t* input_int8, int k_dim, uint8_t* output_packed);
// int32_t avx2_bit_slice_gemm_kernel(const uint8_t* A_packed, const uint8_t* B_packed, const int32_t* lut, int k_dim);
// void silu(float* vec_ptr, size_t size);
// std::vector<float> positional_embedding(const std::vector<long>& timesteps, int dim, float scale);
// std::vector<float> upsample_bilinear(const std::vector<float>& input, int channels, int in_h, int in_w, int scale_factor);
// std::vector<float> downsample_avgpool(const std::vector<float>& input, int channels, int in_h, int in_w, int kernel_size, int stride);


// --- Core Layer Structures ---

// Represents a quantized convolutional layer (similar to QATConv2d)
struct QATConv2dLayer {
    std::vector<uint8_t> packed_weights;
    std::vector<float> bias;
    float weight_scale; // Alpha from ScaledWeightTernary

    int in_channels;
    int out_channels;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
    int groups;

    QATConv2dLayer() : in_channels(0), out_channels(0), kernel_h(0), kernel_w(0),
                       stride_h(1), stride_w(1), pad_h(0), pad_w(0), groups(1), weight_scale(1.0f) {}

    // input_batch: flattened NCHW tensor
    // input_h, input_w: actual height and width of one image in the batch
    std::vector<float> forward(const std::vector<float>& input_batch, int batch_size,
                               int input_h, int input_w,
                               const std::vector<int32_t>& precomputed_lut);
};

// Represents a Group Normalization layer
struct GroupNormLayer {
    int num_groups;
    int num_channels;
    float eps;
    std::vector<float> gamma; // weight
    std::vector<float> beta;  // bias

    GroupNormLayer() : num_groups(0), num_channels(0), eps(1e-5f) {}

    // input: flattened NCHW tensor
    std::vector<float> forward(const std::vector<float>& input, int batch_size, int channels, int height, int width);
};

// Represents a Linear layer (for time and class embeddings)
struct LinearLayer {
    std::vector<float> weights;
    std::vector<float> bias;
    int in_features;
    int out_features;

    LinearLayer() : in_features(0), out_features(0) {}

    // input: flattened N_batch * in_features tensor
    std::vector<float> forward(const std::vector<float>& input, int batch_size);
};

// --- UNet Specific Blocks ---

// Attention Block (simplified for now, actual implementation is complex)
struct AttentionBlock {
    // These would typically be QATConv2dLayer or LinearLayer for Q, K, V projections
    // and an output projection. For now, we'll keep it simple.
    // QATConv2dLayer q_proj, k_proj, v_proj, out_proj;
    // GroupNormLayer norm;
    // int num_heads;
    // int head_channels;

    AttentionBlock() {} // Constructor

    // input: flattened NCHW tensor
    std::vector<float> forward(const std::vector<float>& input, int batch_size,
                               int channels, int height, int width);
};

// ResBlock in QAT_UNet.py
struct QATResBlock {
    QATConv2dLayer conv1;
    GroupNormLayer norm1;
    LinearLayer time_emb_proj1; // For time embedding projection
    QATConv2dLayer conv2;
    GroupNormLayer norm2;
    LinearLayer time_emb_proj2; // For time embedding projection

    // Optional skip connection convolution if input and output channels differ
    // `skip_connection_conv` is only initialized if `in_channels != out_channels`
    // It's a pointer to allow conditional allocation.
    std::unique_ptr<QATConv2dLayer> skip_connection_conv; 

    // Attention block (if resolution matches attention_resolutions)
    bool use_attention;
    std::unique_ptr<AttentionBlock> attention_block;

    QATResBlock() : use_attention(false) {}

    // input: flattened NCHW tensor
    // time_emb: flattened N * time_emb_dim tensor
    // y_labels: flattened N tensor (if class_cond)
    std::vector<float> forward(const std::vector<float>& input, int batch_size,
                               int input_h, int input_w,
                               const std::vector<float>& time_emb,
                               const std::vector<int>& y_labels, // Added y_labels
                               const std::vector<int32_t>& precomputed_lut);
};


// Main QAT UNet model structure
class QATUNet {
public:
    // Model Hyperparameters (loaded from binary)
    int image_size;
    int in_channels; // e.g., 3 for RGB
    int base_channels; // num_channels in PyTorch
    std::vector<int> channel_mults;
    int num_res_blocks;
    int time_emb_dim;
    float time_emb_scale;
    int num_classes; // 0 if not class_cond
    float dropout; // Not directly used in inference, but good to store
    std::vector<int> attention_resolutions;
    int num_groups; // For GroupNorm
    int initial_pad;
    bool use_scale_shift_norm;

    // Initial convolution layer
    QATConv2dLayer init_conv;
    LinearLayer time_embedding_mlp_0; // For time embedding first linear layer
    LinearLayer time_embedding_mlp_1; // For time embedding second linear layer
    std::unique_ptr<LinearLayer> label_embedding_mlp; // If class_cond

    // Downsampling path: vector of vectors for ResBlocks at each resolution
    // Each inner vector contains `num_res_blocks` ResBlocks + 1 (for skip connection)
    std::vector<std::vector<QATResBlock>> downs_res_blocks; 
    std::vector<QATConv2dLayer> downs_downsample_convs; // Downsample layers (Conv2d with stride 2)

    // Mid block
    QATResBlock mid_res_block1;
    AttentionBlock mid_attention_block;
    QATResBlock mid_res_block2;

    // Upsampling path
    std::vector<std::vector<QATResBlock>> ups_res_blocks; 
    std::vector<QATConv2dLayer> ups_upsample_convs; // Upsample layers (ConvTranspose2d or Conv2d + Upsample)

    // Output layers
    GroupNormLayer out_norm;
    QATConv2dLayer out_conv; // Final convolution

    // Constructor
    QATUNet();

    // Main forward pass for the UNet
    // x_input: noisy image (flat NCHW)
    // time_steps: current diffusion timestep for each image in batch
    // y_labels: class labels for each image in batch (if class_cond), flat N tensor
    std::vector<float> forward(const std::vector<float>& x_input, int batch_size,
                               const std::vector<long>& time_steps,
                               const std::vector<int>& y_labels,
                               const std::vector<int32_t>& precomputed_lut);

    // Method to load the model parameters from a binary file
    bool load_model(const std::string& model_path, std::vector<int32_t>& precomputed_lut);

private:
    // Helper to read a QATConv2dLayer from file
    void read_conv_layer(std::ifstream& file, QATConv2dLayer& layer);
    // Helper to read a GroupNormLayer from file
    void read_groupnorm_layer(std::ifstream& file, GroupNormLayer& layer);
    // Helper to read a LinearLayer from file
    void read_linear_layer(std::ifstream& file, LinearLayer& layer);
    // Helper to read an AttentionBlock from file (if it has parameters)
    void read_attention_block(std::ifstream& file, AttentionBlock& block);
    // Helper to read a QATResBlock from file
    void read_res_block(std::ifstream& file, QATResBlock& block, int in_channels, int out_channels,
                        int time_emb_dim, int num_classes, const std::vector<int>& attention_resolutions,
                        int current_resolution, int num_groups, bool use_scale_shift_norm);
};

#endif // QAT_UNET_MODEL_H
