#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <memory>
#include <cstdint>
#include <string>
#include <initializer_list>
#include <stdexcept>
#include <numeric>
#include <algorithm>
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

    // Helper to calculate flat index from multi-dimensional indices
    size_t get_index(std::initializer_list<size_t> indices) const {
        if (indices.size() != shape.size()) {
            throw std::out_of_range("Index dimension mismatch");
        }
        size_t index = 0;
        size_t stride = 1;
        auto it = indices.end();
        auto shape_it = shape.end();
        while (it != indices.begin()) {
            --it;
            --shape_it;
            index += (*it) * stride;
            stride *= (*shape_it);
        }
        return index;
    }

    // Multi-dimensional accessors
    float& at(std::initializer_list<size_t> indices) {
        return data[get_index(indices)];
    }
    const float& at(std::initializer_list<size_t> indices) const {
        return data[get_index(indices)];
    }

    // Tensor manipulation methods
    Tensor mul_scalar(float val) const {
        Tensor result = *this;
        for (size_t i = 0; i < data.size(); ++i) { result.data[i] *= val; }
        return result;
    }
    
    Tensor sub(const Tensor& other) const {
        if (data.size() != other.data.size()) throw std::runtime_error("Tensor shapes must match for subtraction.");
        Tensor result = *this;
        for (size_t i = 0; i < data.size(); ++i) { result.data[i] -= other.data[i]; }
        return result;
    }

    Tensor add(const Tensor& other) const {
        // Supports broadcasting for shapes like {B,C,H,W} + {1,C,1,1}
        if (data.size() == other.data.size()) {
            Tensor result = *this;
            for (size_t i = 0; i < data.size(); ++i) { result.data[i] += other.data[i]; }
            return result;
        } else {
             Tensor result = *this;
             for(size_t b = 0; b < shape[0]; ++b) {
                for(size_t c = 0; c < shape[1]; ++c) {
                    float other_val = other.at({0, c, 0, 0});
                    for(size_t h = 0; h < shape[2]; ++h) {
                        for(size_t w = 0; w < shape[3]; ++w) {
                            result.at({b, c, h, w}) += other_val;
                        }
                    }
                }
             }
             return result;
        }
    }

    void reshape(std::initializer_list<size_t> new_shape) {
        size_t new_size = 1;
        for (size_t dim : new_shape) { new_size *= dim; }
        if (new_size != data.size()) { throw std::runtime_error("Cannot reshape tensor to different number of elements."); }
        shape = new_shape;
    }

    Tensor permute(std::initializer_list<size_t> dims) const {
        // Simplified permute for specific cases
        if (shape.size() == 3 && std::equal(dims.begin(), dims.end(), std::initializer_list<size_t>{0, 2, 1}.begin())) {
            Tensor result({shape[0], shape[2], shape[1]});
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    for (size_t k = 0; k < shape[2]; ++k) {
                        result.at({i, k, j}) = this->at({i, j, k});
                    }
                }
            }
            return result;
        }
        if (shape.size() == 4 && std::equal(dims.begin(), dims.end(), std::initializer_list<size_t>{0, 3, 1, 2}.begin())) {
             Tensor result({shape[0], shape[3], shape[1], shape[2]});
             for(size_t i=0; i<shape[0]; ++i)
                for(size_t j=0; j<shape[1]; ++j)
                    for(size_t k=0; k<shape[2]; ++k)
                        for(size_t l=0; l<shape[3]; ++l)
                            result.at({i, l, j, k}) = this->at({i, j, k, l});
             return result;
        }
        // Add other permutations as needed
        throw std::runtime_error("Unsupported permute dimensions");
    }
    
    Tensor transpose(size_t dim1, size_t dim2) const {
        // Simplified transpose for matmul
        if (shape.size() == 3 && dim1 == 1 && dim2 == 2) {
            return this->permute({0, 2, 1});
        }
        throw std::runtime_error("Unsupported transpose dimensions");
    }

    Tensor matmul(const Tensor& other) const {
        // Simplified matmul for (B, M, K) x (B, K, N) -> (B, M, N)
        if(this->shape.size() != 3 || other.shape.size() != 3) throw std::runtime_error("Matmul only supports 3D tensors");
        if(this->shape[0] != other.shape[0] || this->shape[2] != other.shape[1]) throw std::runtime_error("Matmul shape mismatch");
        
        size_t B = shape[0];
        size_t M = shape[1];
        size_t K = shape[2];
        size_t N = other.shape[2];

        Tensor result({B, M, N});
        for(size_t b=0; b<B; ++b)
            for(size_t m=0; m<M; ++m)
                for(size_t n=0; n<N; ++n) {
                    float sum = 0.0f;
                    for(size_t k=0; k<K; ++k) {
                        sum += this->at({b, m, k}) * other.at({b, k, n});
                    }
                    result.at({b, m, n}) = sum;
                }
        return result;
    }
    
    Tensor cat(const Tensor& other, size_t dim) const {
        if (dim != 1) throw std::runtime_error("Cat only implemented for channel dimension (1).");
        if (shape.size() != 4 || other.shape.size() != 4) throw std::runtime_error("Cat only implemented for 4D tensors.");
        if (shape[0] != other.shape[0] || shape[2] != other.shape[2] || shape[3] != other.shape[3]) {
            throw std::runtime_error("Dimensions must match for cat, except on the cat dimension.");
        }

        Tensor result({shape[0], shape[1] + other.shape[1], shape[2], shape[3]});
        
        for (size_t b = 0; b < shape[0]; ++b) {
            for (size_t h = 0; h < shape[2]; ++h) {
                for (size_t w = 0; w < shape[3]; ++w) {
                    // Copy data from this tensor
                    for (size_t c = 0; c < shape[1]; ++c) {
                        result.at({b, c, h, w}) = this->at({b, c, h, w});
                    }
                    // Copy data from the other tensor
                    for (size_t c = 0; c < other.shape[1]; ++c) {
                        result.at({b, shape[1] + c, h, w}) = other.at({b, c, h, w});
                    }
                }
            }
        }
        return result;
    }

    Tensor upsample(int scale_factor) const {
        if (shape.size() != 4) throw std::runtime_error("Upsample only for 4D tensors.");
        Tensor result({shape[0], shape[1], shape[2] * scale_factor, shape[3] * scale_factor});
        
        for (size_t b = 0; b < result.shape[0]; ++b) {
            for (size_t c = 0; c < result.shape[1]; ++c) {
                for (size_t y = 0; y < result.shape[2]; ++y) {
                    for (size_t x = 0; x < result.shape[3]; ++x) {
                        // Nearest-neighbor sampling
                        result.at({b, c, y, x}) = this->at({b, c, y / scale_factor, x / scale_factor});
                    }
                }
            }
        }
        return result;
    }
    
    Tensor slice(size_t dim, size_t start, size_t count) const {
        if (dim != 1) throw std::runtime_error("Slice only implemented for channel dimension (1).");
        if (start + count > shape[1]) throw std::out_of_range("Slice bounds are out of range.");

        Tensor result({shape[0], count, shape[2], shape[3]});
        
        for (size_t b = 0; b < shape[0]; ++b) {
            for (size_t c = 0; c < count; ++c) {
                for (size_t h = 0; h < shape[2]; ++h) {
                    for (size_t w = 0; w < shape[3]; ++w) {
                        result.at({b, c, h, w}) = this->at({b, start + c, h, w});
                    }
                }
            }
        }
        return result;
    }
};
#endif // TYPES_H
