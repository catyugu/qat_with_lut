#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <cstdint> // For uint8_t, int8_t
#include <string>
#include <memory>
#include <initializer_list>
#include <stdexcept>
// --- 模型数据结构 ---
struct FloatLayer {
    std::vector<float> weights;
    std::vector<float> bias;
};

// LUT 量化层的结构
struct LutLayer {
    std::vector<uint8_t> packed_weights; // Packed 5x3bit weights
    std::vector<float> bias;
    float activation_scale = 1.0f; // Scale for the *input* activations to this layer
};

// 仅权重量化层的结构 (未打包的三值权重)
struct WeightsOnlyQuantLayer {
    std::vector<int8_t> weights; // Unpacked ternary weights {-1, 0, 1}
    std::vector<float> bias;
    float activation_scale = 1.0f; // Scale for input activations
};
struct Module {
    virtual ~Module() = default; // Virtual destructor for polymorphism
};

struct QATConv2dLayer : public Module {
    int in_channels, out_channels;
    int kernel_size_h, kernel_size_w;
    int stride_h, stride_w;
    int pad_h, pad_w;
    int groups;
    float weight_scale;
    // Use uint8_t for custom ternary packed weights
    std::vector<uint8_t> packed_weights; 
    std::vector<float> bias;
};

struct LinearLayer : public Module {
    int in_features, out_features;
    float weight_scale;
    // Use float for standard, non-quantized linear layers
    std::vector<float> weights;
    std::vector<float> bias;
};

struct GroupNormLayer : public Module {
    int num_groups;
    int num_channels;
    float eps;
    std::vector<float> weight;
    std::vector<float> bias;
};

struct AttentionBlock : public Module {
    std::unique_ptr<GroupNormLayer> norm;
    std::unique_ptr<QATConv2dLayer> to_qkv;
    std::unique_ptr<QATConv2dLayer> to_out;

    AttentionBlock() {
        norm = std::make_unique<GroupNormLayer>();
        to_qkv = std::make_unique<QATConv2dLayer>();
        to_out = std::make_unique<QATConv2dLayer>();
    }
};

struct QATResidualBlock : public Module {
    std::unique_ptr<GroupNormLayer> norm_1;
    std::unique_ptr<QATConv2dLayer> conv_1;
    std::unique_ptr<LinearLayer> time_bias;
    std::unique_ptr<GroupNormLayer> norm_2;
    std::unique_ptr<QATConv2dLayer> conv_2;
    std::unique_ptr<QATConv2dLayer> residual_connection;
    std::unique_ptr<AttentionBlock> attention;

    QATResidualBlock(int in_ch, int out_ch, int time_emb_dim, bool use_attention) {
        norm_1 = std::make_unique<GroupNormLayer>();
        conv_1 = std::make_unique<QATConv2dLayer>();
        if (time_emb_dim > 0) {
            time_bias = std::make_unique<LinearLayer>();
        }
        norm_2 = std::make_unique<GroupNormLayer>();
        conv_2 = std::make_unique<QATConv2dLayer>();
        if (in_ch != out_ch) {
            residual_connection = std::make_unique<QATConv2dLayer>();
        }
        if (use_attention) {
            attention = std::make_unique<AttentionBlock>();
        }
    }
};

struct DownsampleLayer : public Module {
    std::unique_ptr<QATConv2dLayer> conv;
    DownsampleLayer() { conv = std::make_unique<QATConv2dLayer>(); }
};

struct UpsampleLayer : public Module {
    std::unique_ptr<QATConv2dLayer> conv;
    UpsampleLayer() { conv = std::make_unique<QATConv2dLayer>(); }
};

struct PositionalEmbedding : public Module {
    int dim;
    float time_emb_scale;
    PositionalEmbedding(int d, float scale) : dim(d), time_emb_scale(scale) {}
};

struct Tensor {
    std::vector<size_t> shape;
    std::vector<float> data;

    Tensor() = default;

    Tensor(std::initializer_list<size_t> s) : shape(s) {
        size_t total_size = 1;
        for (size_t dim : shape) {
            total_size *= dim;
        }
        data.resize(total_size);
    }

    Tensor mul_scalar(float val) const {
        Tensor result = *this;
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] *= val;
        }
        return result;
    }
    
    Tensor sub(const Tensor& other) const {
        if (data.size() != other.data.size()) {
            throw std::runtime_error("Tensor shapes must match for subtraction.");
        }
        Tensor result = *this;
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] -= other.data[i];
        }
        return result;
    }

    Tensor add(const Tensor& other) const {
        if (data.size() != other.data.size()) {
            throw std::runtime_error("Tensor shapes must match for addition.");
        }
        Tensor result = *this;
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] += other.data[i];
        }
        return result;
    }
};

#endif // TYPES_H
