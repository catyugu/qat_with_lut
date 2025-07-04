#include "qat_unet_model.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm> // For std::reverse
#include <iomanip>   // For std::hex, std::setw, std::setfill
#include <cmath>     // For sin, cos, log
#include <numeric>   // For std::accumulate
#include <limits>    // For std::numeric_limits

// Helper function to read a 32-bit unsigned integer from file in little-endian format
unsigned int read_uint32_little_endian(std::ifstream& file) {
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    // Reconstruct unsigned int from little-endian bytes
    return static_cast<unsigned int>(bytes[0]) |
           (static_cast<unsigned int>(bytes[1]) << 8) |
           (static_cast<unsigned int>(bytes[2]) << 16) |
           (static_cast<unsigned int>(bytes[3]) << 24);
}

// Constructor
QATUNet::QATUNet() :
    image_size(0), in_channels(0), base_channels(0), num_res_blocks(0),
    time_emb_dim(0), time_emb_scale(0.0f), num_classes(0), dropout(0.0f),
    num_groups(0), initial_pad(0), use_scale_shift_norm(false), num_down_levels(0) { // Initialize num_down_levels
    // Initialize unique_ptrs to nullptr or create empty structures
    init_conv = nullptr;
    time_mlp_pos_emb = nullptr;
    time_mlp_linear1 = nullptr;
    time_mlp_linear2 = nullptr;
    mid_block1 = nullptr;
    mid_attn = nullptr;
    mid_block2 = nullptr;
    final_norm = nullptr;
    final_conv = nullptr;
}

// Destructor
QATUNet::~QATUNet() {
    // unique_ptrs handle memory deallocation automatically
}

// Helper function to read a QATConv2dLayer from the file
void QATUNet::read_conv_layer(std::ifstream& file, QATConv2dLayer& layer) {
    // Read layer dimensions and parameters (order MUST match Python script)
    layer.in_channels = read_uint32_little_endian(file);
    layer.out_channels = read_uint32_little_endian(file);
    layer.kernel_h = read_uint32_little_endian(file);
    layer.kernel_w = read_uint32_little_endian(file);
    layer.stride_h = read_uint32_little_endian(file);
    layer.stride_w = read_uint32_little_endian(file);
    layer.pad_h = read_uint32_little_endian(file);
    layer.pad_w = read_uint32_little_endian(file);
    layer.groups = read_uint32_little_endian(file);
    file.read(reinterpret_cast<char*>(&layer.weight_scale), sizeof(float));

    // Read packed weights size and data
    unsigned int packed_weights_size = read_uint32_little_endian(file);
    layer.packed_weights.resize(packed_weights_size);
    file.read(reinterpret_cast<char*>(layer.packed_weights.data()), packed_weights_size);

    // Read bias size and data
    unsigned int bias_size = read_uint32_little_endian(file);
    layer.bias.resize(bias_size);
    file.read(reinterpret_cast<char*>(layer.bias.data()), bias_size * sizeof(float));

    // Debug prints for QATConv2dLayer
    std::cout << "      Conv Layer - In: " << layer.in_channels
              << ", Out: " << layer.out_channels
              << ", Kernel: " << layer.kernel_h << "x" << layer.kernel_w
              << ", Stride: " << layer.stride_h << "x" << layer.stride_w
              << ", Pad: " << layer.pad_h << "x" << layer.pad_w
              << ", Groups: " << layer.groups
              << ", Weight Scale: " << layer.weight_scale
              << ", Packed Weights Size: " << packed_weights_size
              << ", Bias Size: " << bias_size << std::endl;
}

// Helper function to read a FloatConv2dLayer from the file
void QATUNet::read_float_conv_layer(std::ifstream& file, FloatConv2dLayer& layer) {
    // Read layer dimensions and parameters (order MUST match Python script)
    layer.in_channels = read_uint32_little_endian(file);
    layer.out_channels = read_uint32_little_endian(file);
    layer.kernel_h = read_uint32_little_endian(file);
    layer.kernel_w = read_uint32_little_endian(file);
    layer.stride_h = read_uint32_little_endian(file);
    layer.stride_w = read_uint32_little_endian(file);
    layer.pad_h = read_uint32_little_endian(file);
    layer.pad_w = read_uint32_little_endian(file);
    layer.groups = read_uint32_little_endian(file);

    // Read float weights size and data
    unsigned int weights_size = read_uint32_little_endian(file);
    layer.weights.resize(weights_size);
    file.read(reinterpret_cast<char*>(layer.weights.data()), weights_size * sizeof(float));

    // Read bias size and data
    unsigned int bias_size = read_uint32_little_endian(file);
    layer.bias.resize(bias_size);
    file.read(reinterpret_cast<char*>(layer.bias.data()), bias_size * sizeof(float));

    // Debug prints for FloatConv2dLayer
    std::cout << "      Float Conv Layer - In: " << layer.in_channels
              << ", Out: " << layer.out_channels
              << ", Kernel: " << layer.kernel_h << "x" << layer.kernel_w
              << ", Stride: " << layer.stride_h << "x" << layer.stride_w
              << ", Pad: " << layer.pad_h << "x" << layer.pad_w
              << ", Groups: " << layer.groups
              << ", Weights Size (floats): " << weights_size
              << ", Bias Size: " << bias_size << std::endl;
}


// Helper function to read a GroupNormLayer from the file
void QATUNet::read_groupnorm_layer(std::ifstream& file, GroupNormLayer& layer) {
    // Read layer dimensions and parameters (order MUST match Python script)
    layer.num_groups = read_uint32_little_endian(file);
    layer.num_channels = read_uint32_little_endian(file);
    file.read(reinterpret_cast<char*>(&layer.eps), sizeof(float)); // Read epsilon

    // Read weight (gamma) size and data
    unsigned int weight_size = read_uint32_little_endian(file);
    layer.weight.resize(weight_size);
    file.read(reinterpret_cast<char*>(layer.weight.data()), weight_size * sizeof(float));

    // Read bias (beta) size and data
    unsigned int bias_size = read_uint32_little_endian(file);
    layer.bias.resize(bias_size);
    file.read(reinterpret_cast<char*>(layer.bias.data()), bias_size * sizeof(float));

    // Debug prints for GroupNormLayer
    std::cout << "      GroupNorm Layer - Groups: " << layer.num_groups
              << ", Channels: " << layer.num_channels
              << ", Epsilon: " << layer.eps
              << ", Weight Size: " << weight_size
              << ", Bias Size: " << bias_size << std::endl;
}

// Helper function to read a LinearLayer from the file
void QATUNet::read_linear_layer(std::ifstream& file, LinearLayer& layer) {
    // Read layer dimensions and parameters (order MUST match Python script: out_features then in_features)
    layer.out_features = read_uint32_little_endian(file);
    layer.in_features = read_uint32_little_endian(file);
    file.read(reinterpret_cast<char*>(&layer.weight_scale), sizeof(float)); // Read weight_scale

    // Read packed weights size and data
    unsigned int packed_weights_size = read_uint32_little_endian(file);
    layer.packed_weights.resize(packed_weights_size);
    file.read(reinterpret_cast<char*>(layer.packed_weights.data()), packed_weights_size);

    // Read bias size and data
    unsigned int bias_size = read_uint32_little_endian(file);
    layer.bias.resize(bias_size);
    file.read(reinterpret_cast<char*>(layer.bias.data()), bias_size * sizeof(float));

    // Debug prints for LinearLayer
    std::cout << "      Linear Layer - In Features: " << layer.in_features
              << ", Out Features: " << layer.out_features
              << ", Weight Scale: " << layer.weight_scale
              << ", Packed Weights Size: " << packed_weights_size
              << ", Bias Size: " << bias_size << std::endl;
}

// Helper function to read an EmbeddingLayer from the file
void QATUNet::read_embedding_layer(std::ifstream& file, EmbeddingLayer& layer) {
    // Read layer dimensions (order MUST match Python script)
    layer.num_embeddings = read_uint32_little_endian(file);
    layer.embedding_dim = read_uint32_little_endian(file);

    // Read weight (embedding table) size and data
    unsigned int weight_size = read_uint32_little_endian(file);
    layer.weight.resize(weight_size);
    file.read(reinterpret_cast<char*>(layer.weight.data()), weight_size * sizeof(float));

    // Debug prints for EmbeddingLayer
    std::cout << "      Embedding Layer - Num Embeddings: " << layer.num_embeddings
              << ", Embedding Dim: " << layer.embedding_dim
              << ", Weight Size: " << weight_size << std::endl;
}

// Helper function to read HadamardTransform parameters
void QATUNet::read_hadamard_transform(std::ifstream& file, HadamardTransform& layer) {
    // HadamardTransform does not have trainable parameters exported.
    // Its 'dim' and 'target_dim' are derived from the architecture.
    // For now, we don't need to read anything specific for it from the file.
    // If you decide to export 'dim' and 'target_dim' later, add read calls here.
    std::cout << "      HadamardTransform - (No parameters to read from file)" << std::endl;
}


// Helper function to read an AttentionBlock from the file
void QATUNet::read_attention_block(std::ifstream& file, AttentionBlock& block) {
    std::cout << "    Reading Attention Block Norm..." << std::endl;
    block.norm = std::make_unique<GroupNormLayer>();
    read_groupnorm_layer(file, *block.norm);

    std::cout << "    Reading Attention Block QKV Conv (Float)..." << std::endl;
    block.to_qkv = std::make_unique<FloatConv2dLayer>(); // Changed to FloatConv2dLayer
    read_float_conv_layer(file, *block.to_qkv);           // Use read_float_conv_layer

    std::cout << "    Reading Attention Block Proj Out Conv (Float)..." << std::endl;
    block.to_out = std::make_unique<FloatConv2dLayer>(); // Changed to FloatConv2dLayer
    read_float_conv_layer(file, *block.to_out);          // Use read_float_conv_layer

    // Hadamard Transforms are not exported as they have no trainable parameters.
    // They are reconstructed based on 'in_channels' of the AttentionBlock.
    // Initialize them here based on the in_channels (which is 'out_channels' of the parent block)
    // For now, these are just initialized, no file reading needed for them.
    // The 'dim' for HadamardTransform should be 'in_channels' of the AttentionBlock.
    // This value is not directly available here, so you might need to pass it or derive it.
    // For now, the HadamardTransform constructor can handle it with a default.
    std::cout << "    Initializing Hadamard Transforms (no parameters to read)..." << std::endl;
    block.hadamard_q = std::make_unique<HadamardTransform>(block.to_qkv->in_channels); // Pass relevant dim
    block.hadamard_k = std::make_unique<HadamardTransform>(block.to_qkv->in_channels); // Pass relevant dim
    block.hadamard_v = std::make_unique<HadamardTransform>(block.to_qkv->in_channels); // Pass relevant dim
}

// Helper function to read a QATResBlock from the file
void QATUNet::read_res_block(std::ifstream& file, QATResBlock& block) {
    std::cout << "    Reading Res Block Norm1..." << std::endl;
    block.norm1 = std::make_unique<GroupNormLayer>();
    read_groupnorm_layer(file, *block.norm1);

    std::cout << "    Reading Res Block Conv1..." << std::endl;
    block.conv1 = std::make_unique<QATConv2dLayer>();
    read_conv_layer(file, *block.conv1);

    std::cout << "    Reading Res Block Time Emb Proj..." << std::endl;
    block.time_emb_proj = std::make_unique<LinearLayer>();
    read_linear_layer(file, *block.time_emb_proj);

    std::cout << "    Reading Res Block Norm2..." << std::endl;
    block.norm2 = std::make_unique<GroupNormLayer>();
    read_groupnorm_layer(file, *block.norm2);

    std::cout << "    Reading Res Block Conv2..." << std::endl;
    block.conv2 = std::make_unique<QATConv2dLayer>();
    read_conv_layer(file, *block.conv2);

    // Read bool for use_conv_shortcut explicitly as a char (1 byte)
    char use_conv_shortcut_byte;
    file.read(&use_conv_shortcut_byte, sizeof(char));
    block.use_conv_shortcut = static_cast<bool>(use_conv_shortcut_byte);

    std::cout << "    Res Block Use Conv Shortcut: " << (block.use_conv_shortcut ? "True" : "False") << std::endl;
    if (block.use_conv_shortcut) {
        std::cout << "    Reading Res Block Conv Shortcut..." << std::endl;
        block.conv_shortcut = std::make_unique<QATConv2dLayer>();
        read_conv_layer(file, *block.conv_shortcut);
    } else {
        block.conv_shortcut = nullptr;
    }

    // Read class_bias (EmbeddingLayer) if num_classes > 0
    // This needs to be conditional based on whether the model is class_cond
    // and if the layer exists. For now, assuming it's always there if num_classes > 0
    // and it's exported.
    // The Python script exports it if num_classes is not None.
    // In C++, we check num_classes from hyperparameters.
    if (num_classes > 0) { // Assuming num_classes is a member of QATUNet and accessible
        std::cout << "    Reading Res Block Class Bias (Embedding Layer)..." << std::endl;
        block.class_bias = std::make_unique<EmbeddingLayer>();
        read_embedding_layer(file, *block.class_bias);
    } else {
        block.class_bias = nullptr;
    }
}

// Helper function to read a Downsample block from the file
void QATUNet::read_downsample_block(std::ifstream& file, Downsample& block) {
    std::cout << "    Reading Downsample Conv..." << std::endl;
    block.conv = std::make_unique<QATConv2dLayer>();
    read_conv_layer(file, *block.conv);
}

// Helper function to read an Upsample block from the file
void QATUNet::read_upsample_block(std::ifstream& file, Upsample& block) {
    std::cout << "    Reading Upsample Conv..." << std::endl;
    block.conv = std::make_unique<QATConv2dLayer>();
    read_conv_layer(file, *block.conv);
}


// Method to load the model from a binary file
void QATUNet::load_model(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open model file: " + filepath);
    }

    // Print sizeof(bool) for debugging
    std::cout << "DEBUG: sizeof(bool) on this system is: " << sizeof(bool) << " bytes." << std::endl;

    // Read hyperparameters - MUST match Python export order
    image_size = read_uint32_little_endian(file);
    in_channels = read_uint32_little_endian(file);
    base_channels = read_uint32_little_endian(file);

    unsigned int channel_mults_size = read_uint32_little_endian(file);
    channel_mults.resize(channel_mults_size);
    for (unsigned int i = 0; i < channel_mults_size; ++i) {
        channel_mults[i] = read_uint32_little_endian(file);
    }

    num_res_blocks = read_uint32_little_endian(file);
    time_emb_dim = read_uint32_little_endian(file);
    file.read(reinterpret_cast<char*>(&time_emb_scale), sizeof(float));
    num_classes = read_uint32_little_endian(file);
    file.read(reinterpret_cast<char*>(&dropout), sizeof(float));

    unsigned int attention_resolutions_size = read_uint32_little_endian(file);
    attention_resolutions.resize(attention_resolutions_size);
    for (unsigned int i = 0; i < attention_resolutions_size; ++i) {
        attention_resolutions[i] = read_uint32_little_endian(file);
    }

    num_groups = read_uint32_little_endian(file);
    initial_pad = read_uint32_little_endian(file);
    
    // Read use_scale_shift_norm explicitly as a char (1 byte)
    char use_scale_shift_norm_byte;
    file.read(&use_scale_shift_norm_byte, sizeof(char));
    use_scale_shift_norm = static_cast<bool>(use_scale_shift_norm_byte);

    // Set num_down_levels from channel_mults size
    num_down_levels = channel_mults.size();


    std::cout << "Model Hyperparameters:" << std::endl;
    std::cout << "  Image Size: " << image_size << std::endl;
    std::cout << "  In Channels: " << in_channels << std::endl;
    std::cout << "  Base Channels: " << base_channels << std::endl;
    std::cout << "  Channel Mults:";
    for (unsigned int mult : channel_mults) {
        std::cout << " " << mult;
    }
    std::cout << std::endl;
    std::cout << "  Num Res Blocks: " << num_res_blocks << std::endl;
    std::cout << "  Time Emb Dim: " << time_emb_dim << std::endl;
    std::cout << "  Time Emb Scale: " << std::fixed << std::setprecision(2) << time_emb_scale << std::endl;
    std::cout << "  Num Classes: " << num_classes << std::endl;
    std::cout << "  Dropout: " << std::fixed << std::setprecision(2) << dropout << std::endl;
    std::cout << "  Attention Resolutions:";
    for (unsigned int res : attention_resolutions) {
        std::cout << " " << res;
    }
    std::cout << std::endl;
    std::cout << "  Num Groups: " << num_groups << std::endl;
    std::cout << "  Initial Pad: " << initial_pad << std::endl;
    std::cout << "  Use Scale Shift Norm: " << (use_scale_shift_norm ? "True" : "False") << std::endl;


    // Read initial convolution
    std::cout << "Reading init_conv..." << std::endl;
    init_conv = std::make_unique<QATConv2dLayer>();
    read_conv_layer(file, *init_conv);

    // Read Time MLP layers
    if (time_emb_dim > 0) { // Only if time_mlp is present
        std::cout << "Reading time_mlp PositionalEmbedding (no params to read)..." << std::endl;
        // PositionalEmbedding has no parameters to read, just need to initialize
        // The constructor takes dim and scale, which are hyperparameters
        time_mlp_pos_emb = std::make_unique<PositionalEmbedding>(base_channels, time_emb_scale);

        std::cout << "Reading time_mlp Linear1..." << std::endl;
        time_mlp_linear1 = std::make_unique<LinearLayer>();
        read_linear_layer(file, *time_mlp_linear1);

        std::cout << "Skipping time_mlp SiLU (no params)..." << std::endl;

        std::cout << "Reading time_mlp Linear2..." << std::endl;
        time_mlp_linear2 = std::make_unique<LinearLayer>();
        read_linear_layer(file, *time_mlp_linear2);
    }


    // Read downsampling path
    down_blocks_res.resize(num_down_levels);
    down_blocks_attn.resize(num_down_levels);
    downsamples.resize(num_down_levels - 1); // One less downsample than levels

    for (unsigned int i = 0; i < num_down_levels; ++i) {
        std::cout << "Reading down block level " << i << "..." << std::endl;
        for (unsigned int j = 0; j < num_res_blocks; ++j) {
            std::cout << "  Reading res block " << j << "..." << std::endl;
            down_blocks_res[i].push_back(std::make_unique<QATResBlock>());
            read_res_block(file, *down_blocks_res[i].back());
        }

        // Check if attention is applied at this resolution
        bool apply_attention_at_res = false;
        for (unsigned int res : attention_resolutions) {
            if (res == image_size / (1 << i)) { // Check current resolution
                apply_attention_at_res = true;
                break;
            }
        }

        if (apply_attention_at_res) {
            std::cout << "  Reading attention block..." << std::endl;
            down_blocks_attn[i].push_back(std::make_unique<AttentionBlock>()); // Changed to AttentionBlock
            read_attention_block(file, *down_blocks_attn[i].back());
        }

        if (i < num_down_levels - 1) {
            std::cout << "  Reading downsample block..." << std::endl;
            downsamples[i] = std::make_unique<Downsample>();
            read_downsample_block(file, *downsamples[i]);
        }
    }

    // Read middle block
    std::cout << "Reading mid_block1..." << std::endl;
    mid_block1 = std::make_unique<QATResBlock>();
    read_res_block(file, *mid_block1);

    std::cout << "Reading mid_attn..." << std::endl;
    mid_attn = std::make_unique<AttentionBlock>(); // Changed to AttentionBlock
    read_attention_block(file, *mid_attn);

    std::cout << "Reading mid_block2..." << std::endl;
    mid_block2 = std::make_unique<QATResBlock>();
    read_res_block(file, *mid_block2);

    // Read upsampling path
    up_blocks_res.resize(num_down_levels);
    up_blocks_attn.resize(num_down_levels);
    upsamples.resize(num_down_levels - 1); // One less upsample than levels

    for (int i = num_down_levels - 1; i >= 0; --i) {
        std::cout << "Reading up block level " << i << "..." << std::endl;
        for (unsigned int j = 0; j < num_res_blocks + 1; ++j) { // +1 for skip connection
            std::cout << "  Reading res block " << j << "..." << std::endl;
            up_blocks_res[i].push_back(std::make_unique<QATResBlock>());
            read_res_block(file, *up_blocks_res[i].back());
        }

        // Check if attention is applied at this resolution
        bool apply_attention_at_res = false;
        for (unsigned int res : attention_resolutions) {
            if (res == image_size / (1 << i)) { // Check current resolution
                apply_attention_at_res = true;
                break;
            }
        }

        if (apply_attention_at_res) {
            std::cout << "  Reading attention block..." << std::endl;
            up_blocks_attn[i].push_back(std::make_unique<AttentionBlock>()); // Changed to AttentionBlock
            read_attention_block(file, *up_blocks_attn[i].back());
        }

        if (i > 0) {
            std::cout << "  Reading upsample block..." << std::endl;
            upsamples[i - 1] = std::make_unique<Upsample>();
            read_upsample_block(file, *upsamples[i - 1]);
        }
    }

    // Read final layers
    std::cout << "Reading final_norm..." << std::endl;
    final_norm = std::make_unique<GroupNormLayer>();
    read_groupnorm_layer(file, *final_norm);

    std::cout << "Reading final_conv..." << std::endl;
    final_conv = std::make_unique<FloatConv2dLayer>(); // Changed to FloatConv2dLayer
    read_float_conv_layer(file, *final_conv);          // Use read_float_conv_layer

    file.close();
    std::cout << "Model loaded successfully." << std::endl;
}

// Helper for time embedding (as per DDPM paper)
std::vector<float> QATUNet::timestep_embedding(long t, unsigned int dim, float max_period) {
    std::vector<float> emb(dim);
    if (dim % 2 != 0) {
        std::cerr << "Warning: Time embedding dimension is odd. It should be even for sine/cosine pairs." << std::endl;
    }

    for (unsigned int i = 0; i < dim / 2; ++i) {
        float exponent = -std::log(max_period) * (static_cast<float>(i) / (dim / 2));
        float arg = t * std::exp(exponent);
        emb[2 * i] = std::sin(arg);
        emb[2 * i + 1] = std::cos(arg);
    }
    return emb;
}

// Helper for class embedding (now uses EmbeddingLayer)
std::vector<float> QATUNet::class_embedding(int label, unsigned int dim, unsigned int num_classes) {
    // This function might become redundant if EmbeddingLayer::forward is used directly.
    // For now, keep it as a placeholder or if it's used for other purposes.
    std::cerr << "Warning: QATUNet::class_embedding is a placeholder. If using EmbeddingLayer, its forward method should be called." << std::endl;
    std::vector<float> emb(dim, 0.0f);
    if (label >= 0 && label < num_classes) {
        // This logic would typically be inside EmbeddingLayer::forward
        // For now, returning a dummy one-hot if dim matches num_classes
        if (dim == num_classes) {
            emb[label] = 1.0f;
        }
    }
    return emb;
}

// Helper for applying attention (placeholder)
std::vector<float> QATUNet::apply_attention(
    const std::vector<float>& input,
    const AttentionBlock& attn_block, // Changed to AttentionBlock
    int batch_size,
    int input_h,
    int input_w
) {
    // TODO: Implement attention mechanism by calling attn_block.forward()
    std::cerr << "Warning: QATUNet::apply_attention is a placeholder." << std::endl;
    return input;
}


// Implementation for QATConv2dLayer forward pass
std::vector<float> QATConv2dLayer::forward(
    const std::vector<float>& input,
    int batch_size,
    int input_h,
    int input_w,
    const std::vector<int32_t>& precomputed_lut
) {
    // Calculate output dimensions
    int output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
    std::vector<float> output(batch_size * out_channels * output_h * output_w, 0.0f);

    // --- Placeholder for actual quantized convolution logic ---
    std::cerr << "Warning: QATConv2dLayer::forward is using a simplified placeholder. Replace with actual quantized convolution." << std::endl;
    // This is a dummy loop. Replace with your optimized quantized convolution kernel.
    for (int b = 0; b < batch_size; ++b) {
        for (unsigned int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < output_h; ++oh) {
                for (int ow = 0; ow < output_w; ++ow) {
                    float sum = 0.0f;
                    if (!bias.empty()) {
                        sum += bias[oc];
                    }

                    for (unsigned int ic = 0; ic < in_channels; ++ic) {
                        for (unsigned int kh = 0; kh < kernel_h; ++kh) {
                            for (unsigned int kw = 0; kw < kernel_w; ++kw) {
                                int in_h = oh * stride_h + kh - pad_h;
                                int in_w = ow * stride_w + kw - pad_w;

                                if (in_h >= 0 && in_h < input_h && in_w >= 0 && in_w < input_w) {
                                    int input_idx = b * in_channels * input_h * input_w +
                                                    ic * input_h * input_w +
                                                    in_h * input_w + in_w;
                                    float input_val = input[input_idx];
                                    sum += input_val * 0.1f; // Dummy multiplication
                                }
                            }
                        }
                    }
                    int output_idx = b * out_channels * output_h * output_w +
                                     oc * output_h * output_w +
                                     oh * output_w + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
    return output;
}

// Implementation for FloatConv2dLayer forward pass
std::vector<float> FloatConv2dLayer::forward(
    const std::vector<float>& input,
    int batch_size,
    int input_h,
    int input_w
) {
    // Calculate output dimensions
    int output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
    std::vector<float> output(batch_size * out_channels * output_h * output_w, 0.0f);

    std::cerr << "Warning: FloatConv2dLayer::forward is a simplified placeholder. Replace with actual float convolution." << std::endl;
    // This is a dummy loop. Replace with your optimized float convolution kernel.
    for (int b = 0; b < batch_size; ++b) {
        for (unsigned int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < output_h; ++oh) {
                for (int ow = 0; ow < output_w; ++ow) {
                    float sum = 0.0f;
                    if (!bias.empty()) {
                        sum += bias[oc];
                    }

                    for (unsigned int ic = 0; ic < in_channels; ++ic) {
                        for (unsigned int kh = 0; kh < kernel_h; ++kh) {
                            for (unsigned int kw = 0; kw < kernel_w; ++kw) {
                                int in_h = oh * stride_h + kh - pad_h;
                                int in_w = ow * stride_w + kw - pad_w;

                                if (in_h >= 0 && in_h < input_h && in_w >= 0 && in_w < input_w) {
                                    int input_idx = b * in_channels * input_h * input_w +
                                                    ic * input_h * input_w +
                                                    in_h * input_w + in_w;
                                    float input_val = input[input_idx];
                                    
                                    // Assuming weights are stored as [out_channels, in_channels, kernel_h, kernel_w]
                                    int weight_idx = oc * in_channels * kernel_h * kernel_w +
                                                     ic * kernel_h * kernel_w +
                                                     kh * kernel_w + kw;
                                    float weight_val = weights[weight_idx];
                                    sum += input_val * weight_val;
                                }
                            }
                        }
                    }
                    int output_idx = b * out_channels * output_h * output_w +
                                     oc * output_h * output_w +
                                     oh * output_w + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
    return output;
}


// Implementation for GroupNormLayer forward pass
std::vector<float> GroupNormLayer::forward(
    const std::vector<float>& input,
    int batch_size,
    int input_h,
    int input_w
) {
    std::vector<float> output(input.size());
    if (num_channels == 0 || num_groups == 0 || num_channels % num_groups != 0) {
        std::cerr << "Error: Invalid GroupNormLayer parameters." << std::endl;
        return input; // Return original input on error
    }

    unsigned int channels_per_group = num_channels / num_groups;

    for (int b = 0; b < batch_size; ++b) {
        for (unsigned int g = 0; g < num_groups; ++g) {
            float mean = 0.0f;
            float variance = 0.0f;
            unsigned int group_elements_count = channels_per_group * input_h * input_w;

            for (unsigned int c_in_group = 0; c_in_group < channels_per_group; ++c_in_group) {
                unsigned int current_channel = g * channels_per_group + c_in_group;
                for (int h = 0; h < input_h; ++h) {
                    for (int w = 0; w < input_w; ++w) {
                        int idx = b * num_channels * input_h * input_w +
                                  current_channel * input_h * input_w +
                                  h * input_w + w;
                        mean += input[idx];
                    }
                }
            }
            mean /= group_elements_count;

            for (unsigned int c_in_group = 0; c_in_group < channels_per_group; ++c_in_group) {
                unsigned int current_channel = g * channels_per_group + c_in_group;
                for (int h = 0; h < input_h; ++h) {
                    for (int w = 0; w < input_w; ++w) {
                        int idx = b * num_channels * input_h * input_w +
                                  current_channel * input_h * input_w +
                                  h * input_w + w;
                        variance += (input[idx] - mean) * (input[idx] - mean);
                    }
                }
            }
            variance /= group_elements_count;

            float std_dev = std::sqrt(variance + eps); // Use layer.eps
            if (std_dev == 0) std_dev = 1.0f; // Avoid division by zero

            for (unsigned int c_in_group = 0; c_in_group < channels_per_group; ++c_in_group) {
                unsigned int current_channel = g * channels_per_group + c_in_group;
                for (int h = 0; h < input_h; ++h) {
                    for (int w = 0; w < input_w; ++w) {
                        int idx = b * num_channels * input_h * input_w +
                                  current_channel * input_h * input_w +
                                  h * input_w + w;
                        float normalized_val = (input[idx] - mean) / std_dev;
                        output[idx] = weight[current_channel] * normalized_val + bias[current_channel];
                    }
                }
            }
        }
    }
    return output;
}

// Placeholder for LinearLayer forward pass
std::vector<float> LinearLayer::forward(
    const std::vector<float>& input,
    int batch_size,
    const std::vector<int32_t>& precomputed_lut
) {
    // TODO: Implement the actual quantized linear layer logic here.
    std::cerr << "Warning: LinearLayer::forward is a placeholder. Replace with actual quantized linear operation." << std::endl;
    std::vector<float> output(batch_size * out_features, 0.0f);
    return output;
}

// Implementation for EmbeddingLayer forward pass
std::vector<float> EmbeddingLayer::forward(
    const std::vector<int>& input_indices,
    int batch_size
) {
    std::vector<float> output(batch_size * embedding_dim, 0.0f);
    std::cerr << "Warning: EmbeddingLayer::forward is a placeholder. Implement actual embedding lookup." << std::endl;

    for (int b = 0; b < batch_size; ++b) {
        int index = input_indices[b];
        if (index >= 0 && index < num_embeddings) {
            // Copy the embedding vector for the given index
            for (unsigned int d = 0; d < embedding_dim; ++d) {
                output[b * embedding_dim + d] = weight[index * embedding_dim + d];
            }
        } else {
            std::cerr << "Error: Embedding index " << index << " out of bounds [0, " << num_embeddings << ")." << std::endl;
            // Return zero vector or handle error
        }
    }
    return output;
}

// Implementation for HadamardTransform forward pass
// HadamardTransform constructor
HadamardTransform::HadamardTransform(unsigned int d) : dim(d) {
    target_dim = (d > 0) ? (1U << static_cast<unsigned int>(std::ceil(std::log2(d)))) : 0;

    if (target_dim > 0) {
        // Generate the Hadamard matrix
        std::vector<std::vector<float>> h_matrix_2d = _get_hadamard_matrix_2d(target_dim);
        hadamard_matrix.reserve(target_dim * target_dim);
        for (unsigned int i = 0; i < target_dim; ++i) {
            for (unsigned int j = 0; j < target_dim; ++j) {
                hadamard_matrix.push_back(h_matrix_2d[i][j] / std::sqrt(static_cast<float>(target_dim)));
            }
        }
    }
}

// Helper to generate Hadamard matrix (2D vector for recursive generation)
std::vector<std::vector<float>> HadamardTransform::_get_hadamard_matrix_2d(unsigned int n) {
    if (n == 1) {
        return {{1.0f}};
    } else {
        std::vector<std::vector<float>> h_n_div_2 = _get_hadamard_matrix_2d(n / 2);
        unsigned int half_n = n / 2;
        std::vector<std::vector<float>> h_n(n, std::vector<float>(n));

        for (unsigned int i = 0; i < half_n; ++i) {
            for (unsigned int j = 0; j < half_n; ++j) {
                h_n[i][j] = h_n_div_2[i][j];
                h_n[i][j + half_n] = h_n_div_2[i][j];
                h_n[i + half_n][j] = h_n_div_2[i][j];
                h_n[i + half_n][j + half_n] = -h_n_div_2[i][j];
            }
        }
        return h_n;
    }
}

std::vector<float> HadamardTransform::forward(
    const std::vector<float>& input,
    int batch_size,
    int input_h,
    int input_w
) {
    if (hadamard_matrix.empty() || dim == 0) {
        return input; // Return original if matrix not initialized
    }

    // Input shape: (B, C, H, W) -> C is 'dim'
    // Reshape to (B*H*W, C) for matrix multiplication
    unsigned int num_spatial_elements = input_h * input_w;
    unsigned int reshaped_rows = batch_size * num_spatial_elements;
    std::vector<float> x_reshaped(reshaped_rows * dim);

    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < input_h; ++h) {
            for (int w = 0; w < input_w; ++w) {
                for (unsigned int c_idx = 0; c_idx < dim; ++c_idx) {
                    int input_idx = b * dim * input_h * input_w + c_idx * input_h * input_w + h * input_w + w;
                    int output_idx = (b * num_spatial_elements + h * input_w + w) * dim + c_idx;
                    x_reshaped[output_idx] = input[input_idx];
                }
            }
        }
    }

    std::vector<float> x_padded = x_reshaped;
    if (dim < target_dim) {
        unsigned int padding_needed = target_dim - dim;
        x_padded.resize(reshaped_rows * target_dim, 0.0f); // Pad with zeros
        // Copy original data to padded buffer
        for (unsigned int r = 0; r < reshaped_rows; ++r) {
            for (unsigned int c = 0; c < dim; ++c) {
                x_padded[r * target_dim + c] = x_reshaped[r * dim + c];
            }
        }
    }

    std::vector<float> transformed_x_padded(reshaped_rows * target_dim, 0.0f);

    // Apply Hadamard transform (matrix multiplication)
    // transformed_x_padded = x_padded @ hadamard_matrix
    for (unsigned int i = 0; i < reshaped_rows; ++i) {
        for (unsigned int j = 0; j < target_dim; ++j) {
            float sum = 0.0f;
            for (unsigned int k = 0; k < target_dim; ++k) {
                sum += x_padded[i * target_dim + k] * hadamard_matrix[k * target_dim + j];
            }
            transformed_x_padded[i * target_dim + j] = sum;
        }
    }

    std::vector<float> transformed_x = transformed_x_padded;
    if (dim < target_dim) {
        transformed_x.resize(reshaped_rows * dim); // Unpad
        for (unsigned int r = 0; r < reshaped_rows; ++r) {
            for (unsigned int c = 0; c < dim; ++c) {
                transformed_x[r * dim + c] = transformed_x_padded[r * target_dim + c];
            }
        }
    }

    // Reshape back to (B, C, H, W)
    std::vector<float> output(batch_size * dim * input_h * input_w);
    for (int b = 0; b < batch_size; ++b) {
        for (unsigned int c_idx = 0; c_idx < dim; ++c_idx) {
            for (int h = 0; h < input_h; ++h) {
                for (int w = 0; w < input_w; ++w) {
                    int input_idx = (b * num_spatial_elements + h * input_w + w) * dim + c_idx;
                    int output_idx = b * dim * input_h * input_w + c_idx * input_h * input_w + h * input_w + w;
                    output[output_idx] = transformed_x[input_idx];
                }
            }
        }
    }
    return output;
}


// Implementation for PositionalEmbedding forward pass
PositionalEmbedding::PositionalEmbedding(unsigned int d, float s) : dim(d), scale(s) {
    // Constructor initializes members. No parameters to load from file.
}

std::vector<float> PositionalEmbedding::forward(long x_val, float scale_val) {
    std::vector<float> emb(dim);
    if (dim % 2 != 0) {
        std::cerr << "Warning: PositionalEmbedding dimension is odd. It should be even for sine/cosine pairs." << std::endl;
    }

    unsigned int half_dim = dim / 2;
    float emb_log_term = std::log(10000.0f) / half_dim;

    for (unsigned int i = 0; i < half_dim; ++i) {
        float exponent = -emb_log_term * (static_cast<float>(i));
        float arg = x_val * scale_val * std::exp(exponent);
        emb[2 * i] = std::sin(arg);
        emb[2 * i + 1] = std::cos(arg);
    }
    return emb;
}


// Placeholder for AttentionBlock forward pass
std::vector<float> AttentionBlock::forward(
    const std::vector<float>& input,
    int batch_size,
    int input_h,
    int input_w
) {
    std::cerr << "Warning: AttentionBlock::forward is a placeholder." << std::endl;
    return input; // Return input as placeholder
}

// Placeholder for QATResBlock forward pass
std::vector<float> QATResBlock::forward(
    const std::vector<float>& input,
    const std::vector<float>& time_emb,
    const std::vector<int>& class_labels, // Added class_labels
    int batch_size,
    int input_h,
    int input_w,
    const std::vector<int32_t>& precomputed_lut
) {
    std::cerr << "Warning: QATResBlock::forward is a placeholder. Implement actual residual block logic." << std::endl;
    return input; // Return input as placeholder
}

// Placeholder for Downsample forward pass
std::vector<float> Downsample::forward(
    const std::vector<float>& input,
    int batch_size,
    int input_h,
    int input_w,
    const std::vector<int32_t>& precomputed_lut
) {
    std::cerr << "Warning: Downsample::forward is a placeholder. Implement actual downsampling." << std::endl;
    return input; // Return input as placeholder
}

// Placeholder for Upsample forward pass
std::vector<float> Upsample::forward(
    const std::vector<float>& input,
    int batch_size,
    int input_h,
    int input_w,
    const std::vector<int32_t>& precomputed_lut
) {
    std::cerr << "Warning: Upsample::forward is a placeholder. Implement actual upsampling." << std::endl;
    return input; // Return input as placeholder
}


// Main forward pass for the QAT UNet model
std::vector<float> QATUNet::forward(
    const std::vector<float>& x,
    int batch_size,
    const std::vector<long>& timesteps,
    const std::vector<int>& labels,
    const std::vector<int32_t>& precomputed_lut
) {
    std::cerr << "Warning: QATUNet::forward is a placeholder and does not perform actual inference." << std::endl;
    return std::vector<float>(); // Return empty vector as placeholder
}
