#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <cstdint>
#include <string>
#include <map>

// Represents a standard floating-point layer (for comparison)
struct FloatLayer {
    std::vector<float> weight;
    std::vector<float> bias;
};

// Represents a LUT-based quantized linear layer
struct LutLayer {
    std::vector<uint8_t> packed_weights; // Packed 3-bit weights
    std::vector<float> bias;
    float activation_scale = 1.0f;
};

// Represents a weights-only quantized layer
struct WeightsOnlyQuantLayer {
    std::vector<uint8_t> packed_weights;
    std::vector<float> bias;
    std::vector<float> weight_scales;
};

// NEW: Struct for a LUT-quantized Convolutional Layer
struct LutConvLayer {
    std::vector<uint8_t> packed_weights; // Packed 5x3bit weights
    std::vector<float> bias;
    float activation_scale = 1.0f;

    // Convolution parameters
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;

    // Name for loading from file
    std::string name;
};

// NEW: Struct for a LUT-quantized Linear Layer (can reuse LutLayer but explicit for clarity)
using LutLinearLayer = LutLayer;


// NEW: Represents the C++ version of the Time Embedding MLP
struct TimeMLP_CPP {
    LutLinearLayer linear1;
    LutLinearLayer linear2;
};

// NEW: Represents the C++ version of a Residual Block in the U-Net
struct ResBlock_CPP {
    LutConvLayer conv1;
    LutConvLayer conv2;
    LutConvLayer shortcut; // For the skip connection projection

    TimeMLP_CPP time_mlp;
};

// NEW: Represents the C++ version of an Attention Block in the U-Net
struct AttentionBlock_CPP {
    // Note: For simplicity, we're assuming LUT-based layers.
    // A full implementation would need to handle the multi-head attention logic.
    LutLinearLayer to_q;
    LutLinearLayer to_k;
    LutLinearLayer to_v;
    LutLinearLayer to_out;
};


#endif // TYPES_H
