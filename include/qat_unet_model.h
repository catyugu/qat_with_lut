#ifndef QAT_UNET_MODEL_H
#define QAT_UNET_MODEL_H

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <fstream>
#include <iostream>
#include <iomanip> // For std::hex, std::setw, std::setfill

// Forward declarations
struct QATConv2dLayer;
struct FloatConv2dLayer; // Added for unquantized convolutions
struct GroupNormLayer;
struct LinearLayer;
struct EmbeddingLayer;   // Added for nn.Embedding
struct AttentionBlock;    // Renamed from QATAttentionBlock
struct HadamardTransform; // Added for Hadamard layer
struct PositionalEmbedding; // Added declaration for PositionalEmbedding
struct QATResBlock;
struct Downsample;
struct Upsample;

// Structure for a Quantization-Aware Training (QAT) Convolutional Layer
struct QATConv2dLayer {
    // Layer dimensions and parameters - MUST match Python export order
    unsigned int in_channels;
    unsigned int out_channels;
    unsigned int kernel_h;
    unsigned int kernel_w;
    unsigned int stride_h;
    unsigned int stride_w;
    unsigned int pad_h;
    unsigned int pad_w;
    unsigned int groups;
    float weight_scale; // Only weight_scale is exported by Python

    // Quantized and packed weights, and biases
    std::vector<unsigned char> packed_weights; // Packed 3-bit ternary weights
    std::vector<float> bias;                   // Biases (float)

    // Forward method for QATConv2dLayer
    std::vector<float> forward(
        const std::vector<float>& input,
        int batch_size,
        int input_h,
        int input_w,
        const std::vector<int32_t>& precomputed_lut // For bit-slice GEMM
    );
};

// Structure for a standard (float) Convolutional Layer
struct FloatConv2dLayer {
    // Layer dimensions and parameters - MUST match Python export order
    unsigned int in_channels;
    unsigned int out_channels;
    unsigned int kernel_h;
    unsigned int kernel_w;
    unsigned int stride_h;
    unsigned int stride_w;
    unsigned int pad_h;
    unsigned int pad_w;
    unsigned int groups;

    // Full-precision weights and biases
    std::vector<float> weights; // Full-precision float weights
    std::vector<float> bias;    // Biases (float)

    // Forward method for FloatConv2dLayer
    std::vector<float> forward(
        const std::vector<float>& input,
        int batch_size,
        int input_h,
        int input_w
    );
};


// Structure for a Group Normalization Layer
struct GroupNormLayer {
    // Parameters - MUST match Python export order
    unsigned int num_groups;
    unsigned int num_channels;
    float eps; // Epsilon for numerical stability

    std::vector<float> weight; // Learnable gamma (scale)
    std::vector<float> bias;   // Learnable beta (shift)

    // Forward method for GroupNormLayer
    std::vector<float> forward(
        const std::vector<float>& input,
        int batch_size,
        int input_h,
        int input_w
    );
};

// Structure for a Linear (Fully Connected) Layer
struct LinearLayer {
    // Parameters - MUST match Python export order
    unsigned int out_features; // Python exports out_features first
    unsigned int in_features;
    float weight_scale;        // Weight scale for ternary weights

    // Quantized and packed weights, and biases
    std::vector<unsigned char> packed_weights; // Packed 3-bit ternary weights
    std::vector<float> bias;                   // Biases (float)

    // Forward method for LinearLayer
    std::vector<float> forward(
        const std::vector<float>& input,
        int batch_size,
        const std::vector<int32_t>& precomputed_lut // For bit-slice GEMM
    );
};

// Structure for an Embedding Layer (nn.Embedding)
struct EmbeddingLayer {
    // Parameters - MUST match Python export order
    unsigned int num_embeddings; // Number of classes/embeddings
    unsigned int embedding_dim;  // Dimension of each embedding

    std::vector<float> weight; // Embedding table (full-precision float)

    // Forward method for EmbeddingLayer
    std::vector<float> forward(
        const std::vector<int>& input_indices, // Input is typically indices (labels)
        int batch_size
    );
};

// Structure for Hadamard Transform Layer
struct HadamardTransform {
    unsigned int dim;        // Original dimension
    unsigned int target_dim; // Dimension padded to next power of 2
    std::vector<float> hadamard_matrix; // Precomputed Hadamard matrix

    // Constructor to initialize Hadamard matrix
    HadamardTransform(unsigned int d = 0); // Default constructor for unique_ptr

    // Forward method for HadamardTransform
    std::vector<float> forward(
        const std::vector<float>& input,
        int batch_size,
        int input_h, // Height of the feature map (for reshaping)
        int input_w  // Width of the feature map (for reshaping)
    );

private:
    // Helper to generate Hadamard matrix
    std::vector<std::vector<float>> _get_hadamard_matrix_2d(unsigned int n);
};

// Structure for Positional Embedding Layer (no parameters to load)
struct PositionalEmbedding {
    unsigned int dim;   // Dimension of the embedding
    float scale;        // Scale factor

    // Constructor to initialize dim and scale
    PositionalEmbedding(unsigned int d = 0, float s = 1.0f); // Default constructor for unique_ptr

    // Forward method for PositionalEmbedding
    std::vector<float> forward(
        long x, // Time step
        float scale
    );
};


// Structure for an Attention Block (now unquantized linear layers)
struct AttentionBlock {
    std::unique_ptr<GroupNormLayer> norm;
    std::unique_ptr<FloatConv2dLayer> to_qkv; // Changed to FloatConv2dLayer
    std::unique_ptr<FloatConv2dLayer> to_out; // Changed to FloatConv2dLayer
    std::unique_ptr<HadamardTransform> hadamard_q; // Hadamard for Query
    std::unique_ptr<HadamardTransform> hadamard_k; // Hadamard for Key
    std::unique_ptr<HadamardTransform> hadamard_v; // Hadamard for Value


    // Forward method for AttentionBlock
    std::vector<float> forward(
        const std::vector<float>& input,
        int batch_size,
        int input_h,
        int input_w
    );
};

// Structure for a QAT Residual Block
struct QATResBlock {
    std::unique_ptr<GroupNormLayer> norm1;
    std::unique_ptr<QATConv2dLayer> conv1;
    std::unique_ptr<LinearLayer> time_emb_proj; // Time embedding projection
    std::unique_ptr<GroupNormLayer> norm2;
    std::unique_ptr<QATConv2dLayer> conv2;
    bool use_conv_shortcut; // Flag to indicate if shortcut convolution is used
    std::unique_ptr<QATConv2dLayer> conv_shortcut; // Optional shortcut convolution
    std::unique_ptr<EmbeddingLayer> class_bias; // Changed to EmbeddingLayer

    // Forward method for QATResBlock
    std::vector<float> forward(
        const std::vector<float>& input,
        const std::vector<float>& time_emb, // Time embedding input
        const std::vector<int>& class_labels, // Class labels for embedding lookup
        int batch_size,
        int input_h,
        int input_w,
        const std::vector<int32_t>& precomputed_lut
    );
};

// Structure for a Downsample Block
struct Downsample {
    std::unique_ptr<QATConv2dLayer> conv;

    // Forward method for Downsample
    std::vector<float> forward(
        const std::vector<float>& input,
        int batch_size,
        int input_h,
        int input_w,
        const std::vector<int32_t>& precomputed_lut
    );
};

// Structure for an Upsample Block
struct Upsample {
    std::unique_ptr<QATConv2dLayer> conv;

    // Forward method for Upsample
    std::vector<float> forward(
        const std::vector<float>& input,
        int batch_size,
        int input_h,
        int input_w,
        const std::vector<int32_t>& precomputed_lut
    );
};

// Main QAT UNet Model Class
class QATUNet {
public:
    // Model hyperparameters - MUST match Python export order
    unsigned int image_size;
    unsigned int in_channels;
    unsigned int base_channels;
    std::vector<unsigned int> channel_mults;
    unsigned int num_res_blocks;
    unsigned int time_emb_dim;
    float time_emb_scale;
    unsigned int num_classes;
    float dropout;
    std::vector<unsigned int> attention_resolutions;
    unsigned int num_groups;
    unsigned int initial_pad;
    bool use_scale_shift_norm; // Boolean (1 byte)
    unsigned int num_down_levels; // Added as member variable

    // Model layers
    std::unique_ptr<QATConv2dLayer> init_conv;

    // Time MLP (Sequential of PositionalEmbedding, Linear, SiLU, Linear)
    // We will represent this as individual layers
    std::unique_ptr<PositionalEmbedding> time_mlp_pos_emb; // Assuming PositionalEmbedding is also exported
    std::unique_ptr<LinearLayer> time_mlp_linear1;
    // SiLU is an activation, no parameters to export
    std::unique_ptr<LinearLayer> time_mlp_linear2;


    // Downsampling path
    std::vector<std::vector<std::unique_ptr<QATResBlock>>> down_blocks_res;
    std::vector<std::vector<std::unique_ptr<AttentionBlock>>> down_blocks_attn; // Changed to AttentionBlock
    std::vector<std::unique_ptr<Downsample>> downsamples;

    // Middle block
    std::unique_ptr<QATResBlock> mid_block1;
    std::unique_ptr<AttentionBlock> mid_attn; // Changed to AttentionBlock
    std::unique_ptr<QATResBlock> mid_block2;

    // Upsampling path
    std::vector<std::vector<std::unique_ptr<QATResBlock>>> up_blocks_res;
    std::vector<std::vector<std::unique_ptr<AttentionBlock>>> up_blocks_attn; // Changed to AttentionBlock
    std::vector<std::unique_ptr<Upsample>> upsamples;

    std::unique_ptr<GroupNormLayer> final_norm;
    std::unique_ptr<FloatConv2dLayer> final_conv; // Changed to FloatConv2dLayer

    // Constructor and methods
    QATUNet();
    ~QATUNet();

    // Method to load the model from a binary file
    void load_model(const std::string& filepath);

    // Method for the forward pass of the UNet model
    std::vector<float> forward(
        const std::vector<float>& x,
        int batch_size,
        const std::vector<long>& timesteps,
        const std::vector<int>& labels,
        const std::vector<int32_t>& precomputed_lut
    );

private:
    // Helper methods to read individual layer types from the file
    void read_conv_layer(std::ifstream& file, QATConv2dLayer& layer);
    void read_float_conv_layer(std::ifstream& file, FloatConv2dLayer& layer); // Added
    void read_groupnorm_layer(std::ifstream& file, GroupNormLayer& layer);
    void read_linear_layer(std::ifstream& file, LinearLayer& layer);
    void read_embedding_layer(std::ifstream& file, EmbeddingLayer& layer); // Added
    void read_hadamard_transform(std::ifstream& file, HadamardTransform& layer); // Added
    void read_attention_block(std::ifstream& file, AttentionBlock& block); // Changed to AttentionBlock
    void read_res_block(std::ifstream& file, QATResBlock& block);
    void read_downsample_block(std::ifstream& file, Downsample& block);
    void read_upsample_block(std::ifstream& file, Upsample& block);

    // Helper for time embedding
    std::vector<float> timestep_embedding(long t, unsigned int dim, float max_period = 10000.0f);
    // Helper for class embedding (now potentially uses EmbeddingLayer)
    std::vector<float> class_embedding(int label, unsigned int dim, unsigned int num_classes); // Keep for now
    // Helper for applying attention
    std::vector<float> apply_attention(
        const std::vector<float>& input,
        const AttentionBlock& attn_block, // Changed to AttentionBlock
        int batch_size,
        int input_h,
        int input_w
    );
};

#endif // QAT_UNET_MODEL_H