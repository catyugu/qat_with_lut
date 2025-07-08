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
#include <functional>
#include <random>
#include <iostream>
#include <immintrin.h>
#include <cstring>
// --- 模型数据结构 ---
inline std::string shape_to_string(const std::vector<size_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        s += std::to_string(shape[i]) + (i == shape.size() - 1 ? "" : ", ");
    }
    s += "]";
    return s;
}

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
    float alpha;                          // 权重缩放因子
    std::vector<int> shape;               // 原始权重的形状 [out_ch, in_ch, kH, kW]
    std::vector<uint32_t> packed_weights; // 2-bit 权重被打包到 uint32_t 向量中
    std::vector<float> bias;
};

struct LinearLayer : public Module {
    int in_features, out_features;
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
struct Embedding {
    int num_embeddings;
    int embedding_dim;
    std::vector<float> weight;
};
struct QATResidualBlock : public Module {
    std::unique_ptr<GroupNormLayer> norm_1;
    std::unique_ptr<QATConv2dLayer> conv_1;
    std::unique_ptr<LinearLayer> time_bias;
    std::unique_ptr<Embedding> class_bias; 
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
    Tensor(const std::vector<size_t>& s) : shape(s) {
        size_t total_size = numel();
        if (total_size > 0) {
            data.resize(total_size);
        }
    }

    // --- 新增的成员函数 ---
    // 计算并返回张量中的元素总数
    size_t numel() const {
        if (shape.empty()) {
            return 0;
        }
        size_t total = 1;
        for (size_t dim : shape) {
            total *= dim;
        }
        return total;
    }
    // Helper to calculate flat index from multi-dimensional indices
    size_t get_index(const std::vector<size_t>& indices) const {
        if (indices.size() != shape.size()) {
            throw std::out_of_range("Index dimension mismatch.");
        }
        size_t index = 0;
        size_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("Index out of bounds.");
            }
            index += indices[i] * stride;
            stride *= shape[i];
        }
        return index;
    }

    // Multi-dimensional accessors
    float& at(const std::vector<size_t>& indices) {
        return data[get_index(indices)];
    }
    const float& at(const std::vector<size_t>& indices) const {
        return data[get_index(indices)];
    }


    // Tensor manipulation methods
    Tensor mul_scalar(float s) const {
        Tensor result(this->shape);
        for (size_t i = 0; i < this->data.size(); ++i) {
            result.data[i] = this->data[i] * s;
        }
        return result;
    }

    
    Tensor sub(const Tensor& other) const {
        if (this->shape != other.shape) {
            throw std::runtime_error("Tensor::sub requires identical shapes.");
        }
        Tensor result(this->shape);
        for (size_t i = 0; i < this->data.size(); ++i) {
            result.data[i] = this->data[i] - other.data[i];
        }
        return result;
    }

    Tensor add(const Tensor& other) const {
        // Case 1: Element-wise Addition (Shapes are identical)
        if (this->shape == other.shape) {
            Tensor result(this->shape);
            for (size_t i = 0; i < this->data.size(); ++i) {
                result.data[i] = this->data[i] + other.data[i];
            }
            return result;
        }

        // Case 2: 4D + 1D Broadcast (Conv2D bias: [B,C,H,W] + [C])
        if (this->shape.size() == 4 && other.shape.size() == 1 && this->shape[1] == other.shape[0]) {
            const auto B = this->shape[0], C = this->shape[1], H = this->shape[2], W = this->shape[3];
            Tensor result(this->shape);
            for (size_t b = 0; b < B; ++b) {
                for (size_t c = 0; c < C; ++c) {
                    const float bias_value = other.data[c];
                    for (size_t hw = 0; hw < H * W; ++hw) {
                        result.data[b*C*H*W + c*H*W + hw] = this->data[b*C*H*W + c*H*W + hw] + bias_value;
                    }
                }
            }
            return result;
        }

        // Case 3: 4D + 2D Broadcast (Time/Class embedding: [B,C,H,W] + [B,C])
        if (this->shape.size() == 4 && other.shape.size() == 2 && this->shape[0] == other.shape[0] && this->shape[1] == other.shape[1]) {
            const auto B = this->shape[0], C = this->shape[1], H = this->shape[2], W = this->shape[3];
            Tensor result(this->shape);
            for (size_t b = 0; b < B; ++b) {
                for (size_t c = 0; c < C; ++c) {
                    const float embedding_value = other.at({b, c});
                    for (size_t hw = 0; hw < H * W; ++hw) {
                        result.data[b*C*H*W + c*H*W + hw] = this->data[b*C*H*W + c*H*W + hw] + embedding_value;
                    }
                }
            }
            return result;
        }
        
        // ▼▼▼ NEWLY ADDED CASE ▼▼▼
        // Case 4: 4D + 4D Broadcast (Residual connection: [B,C,H,W] + [B,C,1,1])
        if (this->shape.size() == 4 && other.shape.size() == 4 &&
            this->shape[0] == other.shape[0] && this->shape[1] == other.shape[1] &&
            other.shape[2] == 1 && other.shape[3] == 1) {
            const auto B = this->shape[0], C = this->shape[1], H = this->shape[2], W = this->shape[3];
            Tensor result(this->shape);
            for (size_t b = 0; b < B; ++b) {
                for (size_t c = 0; c < C; ++c) {
                    // The value from the [B, C, 1, 1] tensor is constant across the HxW plane.
                    const float residual_value = other.at({b, c, 0, 0});
                    for (size_t hw = 0; hw < H * W; ++hw) {
                        result.data[b*C*H*W + c*H*W + hw] = this->data[b*C*H*W + c*H*W + hw] + residual_value;
                    }
                }
            }
            return result;
        }

        // --- Diagnostic Block ---
        std::cerr << "\n\nCRITICAL ERROR: Tensor::add encountered an unhandled broadcasting scenario.\n";
        std::cerr << "  - Shape of 'this' tensor: " << shape_to_string(this->shape) << "\n";
        std::cerr << "  - Shape of 'other' tensor: " << shape_to_string(other.shape) << "\n\n";
        throw std::runtime_error("Tensor::add unsupported shapes for broadcasting.");
    }
    Tensor concat(const Tensor& other) const {
        if (this->shape.size() != 4 || other.shape.size() != 4) {
            throw std::runtime_error("Tensor::concat only supports 4D tensors.");
        }
        // Batch, Height, and Width must be the same.
        if (this->shape[0] != other.shape[0] || this->shape[2] != other.shape[2] || this->shape[3] != other.shape[3]) {
            throw std::runtime_error("Tensor::concat dimensions mismatch (B, H, or W are not equal).");
        }

        const size_t B = this->shape[0];
        const size_t C1 = this->shape[1];
        const size_t C2 = other.shape[1];
        const size_t H = this->shape[2];
        const size_t W = this->shape[3];

        // The new tensor will have the combined number of channels.
        Tensor result({B, C1 + C2, H, W});

        const size_t plane_size = H * W;

        for (size_t b = 0; b < B; ++b) {
            // Pointer to the start of the current batch item in the source tensors.
            const float* this_ptr = this->data.data() + b * C1 * plane_size;
            const float* other_ptr = other.data.data() + b * C2 * plane_size;
            // Pointer to the start of the current batch item in the destination tensor.
            float* result_ptr = result.data.data() + b * (C1 + C2) * plane_size;

            // Copy data from the first tensor ('this').
            memcpy(result_ptr, this_ptr, C1 * plane_size * sizeof(float));

            // Copy data from the second tensor ('other'), placing it right after the first tensor's data.
            memcpy(result_ptr + C1 * plane_size, other_ptr, C2 * plane_size * sizeof(float));
        }

        return result;
    }
    void reshape(const std::vector<size_t>& new_shape) {
        size_t new_size = 1;
        for (size_t dim : new_shape) new_size *= dim;
        if (new_size != data.size()) throw std::runtime_error("Reshape size mismatch.");
        shape = new_shape;
    }

    Tensor permute(const std::vector<int>& dims) const {
        if (dims.size() != shape.size()) throw std::runtime_error("Permute dimensions mismatch.");
        std::vector<size_t> new_shape(shape.size());
        for (size_t i = 0; i < dims.size(); ++i) new_shape[i] = shape[dims[i]];
        
        Tensor result(new_shape);
        std::vector<size_t> old_indices(shape.size());
        
        std::function<void(size_t, size_t)> recurse = 
            [&](size_t dim_idx, size_t current_offset) {
            if (dim_idx == shape.size()) {
                std::vector<size_t> new_indices(dims.size());
                for(size_t i = 0; i < dims.size(); ++i) new_indices[i] = old_indices[dims[i]];
                result.at(new_indices) = data[current_offset];
            } else {
                size_t stride = 1;
                for(size_t i = dim_idx + 1; i < shape.size(); ++i) stride *= shape[i];
                for(size_t i = 0; i < shape[dim_idx]; ++i) {
                    old_indices[dim_idx] = i;
                    recurse(dim_idx + 1, current_offset + i * stride);
                }
            }
        };
        recurse(0, 0);
        return result;
    }
    Tensor transpose(size_t dim1, size_t dim2) const {
        // Simplified transpose for matmul
        if (shape.size() == 3 && dim1 == 1 && dim2 == 2) {
            return this->permute({0, 2, 1});
        }
        throw std::runtime_error("Unsupported transpose dimensions");
    }

    Tensor matmul(const Tensor& other) const {

        // --- Case 1: Batched Matrix Multiplication (3D x 3D) ---
        // This is the case needed for the attention block: (B, M, K) @ (B, K, N) -> (B, M, N)
        if (this->shape.size() == 3 && other.shape.size() == 3) {
            if (this->shape[0] != other.shape[0]) {
                throw std::runtime_error("Matmul error (3D): Batch dimensions must be equal.");
            }
            if (this->shape[2] != other.shape[1]) {
                throw std::runtime_error("Matmul error (3D): Inner dimensions are not compatible.");
            }

            size_t B = this->shape[0];
            size_t M = this->shape[1];
            size_t K = this->shape[2];
            size_t N = other.shape[2];

            Tensor result({B, M, N});
            result.zero(); // Initialize with zeros

            for (size_t b = 0; b < B; ++b) {
                for (size_t m = 0; m < M; ++m) {
                    for (size_t n = 0; n < N; ++n) {
                        for (size_t k = 0; k < K; ++k) {
                            // result[b, m, n] += this[b, m, k] * other[b, k, n]
                            result.at({b, m, n}) += this->at({b, m, k}) * other.at({b, k, n});
                        }
                    }
                }
            }
            return result;
        }
        // --- Case 2: Standard Matrix Multiplication (2D x 2D) ---
        // (M, K) @ (K, N) -> (M, N)
        else if (this->shape.size() == 2 && other.shape.size() == 2) {
            if (this->shape[1] != other.shape[0]) {
                throw std::runtime_error("Matmul error (2D): Inner dimensions are not compatible.");
            }
            size_t M = this->shape[0];
            size_t K = this->shape[1];
            size_t N = other.shape[1];
            
            Tensor result({M, N});
            result.zero();

            for (size_t m = 0; m < M; ++m) {
                for (size_t n = 0; n < N; ++n) {
                    for (size_t k = 0; k < K; ++k) {
                        result.at({m, n}) += this->at({m, k}) * other.at({k, n});
                    }
                }
            }
            return result;
        }
        
        // --- Error Case: Unsupported shapes ---
        throw std::runtime_error("Matmul only supports 3D x 3D or 2D x 2D tensor multiplication.");
    }
    __attribute__((target("avx,fma")))
    Tensor matmul(const float* other_data, size_t other_rows, size_t other_cols) const {
        // This is a specialized matrix multiplication: Tensor * raw_float_array
        if (shape.size() != 2 || shape[1] != other_rows) {
            throw std::runtime_error("Matrix multiplication dimension mismatch for matmul with raw data.");
        }

        const size_t M = shape[0];         // Rows of this tensor
        const size_t K = shape[1];         // Columns of this tensor
        const size_t N = other_cols;     // Columns of the other matrix

        Tensor output({M, N});
        output.zero(); // Ensure the output is initialized to zero

        const float* A_ptr = this->data.data(); // Pointer to this tensor's data
        const float* B_ptr = other_data;        // Pointer to the raw float array
        float* C_ptr = output.data.data();    // Pointer to the output data
        
        const int AVX_FLOAT_COUNT = 8;

        // Perform matrix multiplication C = A * B
        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                // Broadcast the single value from A
                __m256 a_broadcast = _mm256_set1_ps(A_ptr[i * K + k]);
                
                // Multiply it by the corresponding row in B and add to the output row C
                for (size_t j = 0; j < N; j += AVX_FLOAT_COUNT) {
                    if (j + AVX_FLOAT_COUNT > N) { // Handle scalar remainder
                        for(size_t j_offset = 0; j_offset < N - j; ++j_offset) {
                            C_ptr[i * N + j + j_offset] += A_ptr[i * K + k] * B_ptr[k * N + j + j_offset];
                        }
                    } else { // Fast AVX path
                        __m256 b_vals = _mm256_loadu_ps(&B_ptr[k * N + j]);
                        __m256 c_vals = _mm256_loadu_ps(&C_ptr[i * N + j]);
                        // Fused-Multiply-Add: C_vals += a_broadcast * b_vals
                        c_vals = _mm256_fmadd_ps(a_broadcast, b_vals, c_vals);
                        _mm256_storeu_ps(&C_ptr[i * N + j], c_vals);
                    }
                }
            }
        }
        return output;
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
        static Tensor randn_like(const std::vector<size_t>& shape) {
        Tensor result(shape);
        // Use a random device for a better seed
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<float> dist(0.0f, 1.0f);

        for (size_t i = 0; i < result.data.size(); ++i) {
            result.data[i] = dist(gen);
        }
        return result;
    }
    void zero() {
        std::fill(data.begin(), data.end(), 0.0f);
    }

    // 从batch中提取单个样本 (不复制数据，返回一个指向数据的视图)
    // 注意：这是一个简化的实现，为了让 conv2d 工作
    Tensor slice(int item_index) const {
        if (shape[0] <= 1) return *this;
        size_t batch_size = shape[1] * shape[2] * shape[3];
        Tensor sliced_tensor({1, shape[1], shape[2], shape[3]});

        const float* start = data.data() + item_index * batch_size;
        std::copy(start, start + batch_size, sliced_tensor.data.begin());
        return sliced_tensor;
    }
    Tensor view(const std::vector<size_t>& new_shape) const {
        size_t new_numel = 1;
        for (size_t dim : new_shape) {
            new_numel *= dim;
        }
        if (new_numel != this->numel()) {
            throw std::runtime_error("View error: number of elements must match.");
        }
        Tensor new_tensor(new_shape);
        new_tensor.data = this->data; // 共享数据指针
        return new_tensor;
    }
    Tensor interpolate(float scale_factor) const {
        if (shape.size() != 4) {
            throw std::runtime_error("Interpolate only supports 4D tensors (B, C, H, W).");
        }
        size_t b = shape[0], c = shape[1], h = shape[2], w = shape[3];
        size_t new_h = static_cast<size_t>(h * scale_factor);
        size_t new_w = static_cast<size_t>(w * scale_factor);

        Tensor output({b, c, new_h, new_w});

        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t ci = 0; ci < c; ++ci) {
                for (size_t hi = 0; hi < new_h; ++hi) {
                    for (size_t wi = 0; wi < new_w; ++wi) {
                        size_t old_h = static_cast<size_t>(hi / scale_factor);
                        size_t old_w = static_cast<size_t>(wi / scale_factor);
                        
                        size_t new_idx = bi * (c * new_h * new_w) + ci * (new_h * new_w) + hi * new_w + wi;
                        size_t old_idx = bi * (c * h * w) + ci * (h * w) + old_h * w + old_w;

                        output.data[new_idx] = this->data[old_idx];
                    }
                }
            }
        }
        return output;
    }
    Tensor hadamard_transform() const {
    if (shape.size() != 4) {
        throw std::runtime_error("Hadamard transform only supports 4D tensors (B, C, H, W).");
    }

    const size_t B = shape[0];
    const size_t C = shape[1];
    const size_t H = shape[2];
    const size_t W = shape[3];

    if (C == 0) return *this;

    size_t target_dim = 1;
    while (target_dim < C) {
        target_dim *= 2;
    }

    std::vector<float> h_matrix(target_dim * target_dim);
    if (target_dim > 0) {
        h_matrix[0] = 1.0f;
        for (size_t n = 1; n < target_dim; n *= 2) {
            // 创建一个当前矩阵的副本以从中读取
            std::vector<float> h_prev_chunk(h_matrix.begin(), h_matrix.begin() + n * n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    float val = h_prev_chunk[i * n + j];
                    // Top-left quadrant
                    h_matrix[i * (n * 2) + j] = val;
                    // Top-right quadrant
                    h_matrix[i * (n * 2) + j + n] = val;
                    // Bottom-left quadrant
                    h_matrix[(i + n) * (n * 2) + j] = val;
                    // Bottom-right quadrant
                    h_matrix[(i + n) * (n * 2) + j + n] = -val;
                }
            }
        }
    }
    float norm_factor = 1.0f / std::sqrt(static_cast<float>(target_dim));
    for (float& val : h_matrix) {
        val *= norm_factor;
    }

    // ================== FIX FOR VOID CONVERSION ERROR ==================
    // Break the chained calls. Assume permute() returns a new Tensor.
    Tensor input_reshaped = this->permute({0, 2, 3, 1});
    // Assume reshape() modifies the tensor in-place.
    input_reshaped.reshape({B * H * W, C});
    // =================================================================

    // --- Padding Logic ---
    Tensor input_padded = input_reshaped;
    if (C < target_dim) {
        input_padded = Tensor({B * H * W, target_dim});
        input_padded.zero();
        for (size_t i = 0; i < B * H * W; ++i) {
            for (size_t j = 0; j < C; ++j) {
                input_padded.at({i, j}) = input_reshaped.at({i, j});
            }
        }
    }

    // ===================== FIX FOR MATMUL_RAW ======================
    // Perform the matrix multiplication. Note the corrected function call.
    // The h_matrix does NOT need to be permuted if matmul is implemented correctly.
    Tensor transformed = input_padded.matmul(h_matrix.data(), target_dim, target_dim);
    // =============================================================

    // --- Slice and Reshape Back ---
    if (C < target_dim) {
        transformed = transformed.slice(1, 0, C); 
    }
    transformed.reshape({B, H, W, C});
    
    // Assume permute() returns a new Tensor.
    return transformed.permute({0, 3, 1, 2});
    }
    
};
#endif // TYPES_H
