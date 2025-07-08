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
    std::vector<size_t> strides; // 确认这个成员存在，permute 操作必须依赖它

    Tensor() = default;

    explicit Tensor(std::initializer_list<size_t> s) : shape(s) {
        // 使用 std::accumulate 更安全地计算元素总数
        size_t total_size = std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
        data.resize(total_size);

        // --- 以下是补充的 strides 初始化逻辑 ---
        strides.resize(shape.size());
        if (!shape.empty()) {
            // 最内层（最后一个维度）的步长总是 1
            strides.back() = 1;
            
            // 从倒数第二个维度开始，向前计算每个维度的步长
            // 每个维度的步长 = 后一个维度的步长 * 后一个维度的尺寸
            for (int i = shape.size() - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
    }
    
    explicit Tensor(const std::vector<size_t>& new_shape) {
            shape = new_shape;
            size_t total_size = 1;
            for (size_t dim : shape) {
                total_size *= dim;
            }
            data.resize(total_size);

            // 计算连续布局的 strides
            strides.resize(shape.size());
            if (!shape.empty()) {
                strides.back() = 1;
                for (int i = shape.size() - 2; i >= 0; --i) {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
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

    float at(const std::vector<size_t>& indices) const {
        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) { offset += indices[i] * strides[i]; }
        return data.at(offset); // Use .at() for bounds checking
    }

    float& at(const std::vector<size_t>& indices) {
        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) { offset += indices[i] * strides[i]; }
        return data.at(offset); // Use .at() for bounds checking
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
        // This is the tensor we are adding the 'other' tensor to.
        const Tensor& self = *this;

        // --- Input Validation ---
        if (self.shape.size() != 4 || (other.shape.size() != 4 && other.shape.size() != 1)) {
            throw std::runtime_error("Add operation expects 4D tensors or a scalar.");
        }

        // Prepare the output tensor, starting as a copy of the base tensor.
        Tensor result = self;

        // --- Broadcasting Logic ---
        // Check for the specific case: [B, C, H, W] + [B, C, 1, 1]
        bool is_broadcast_add = (other.shape.size() == 4 &&
                                self.shape[0] == other.shape[0] && // Batch matches
                                self.shape[1] == other.shape[1] && // Channels match
                                other.shape[2] == 1 &&              // H is 1
                                other.shape[3] == 1);              // W is 1

        const size_t B = self.shape[0];
        const size_t C = self.shape[1];
        const size_t H = self.shape[2];
        const size_t W = self.shape[3];

        if (is_broadcast_add) {
            // Loop through each item in the batch
            for (size_t b = 0; b < B; ++b) {
                // Loop through each channel
                for (size_t c = 0; c < C; ++c) {
                    // Get the single bias value for this batch item and channel.
                    float bias_value = other.at({b, c, 0, 0});

                    // Add this single value to all pixels in the corresponding channel.
                    for (size_t hw = 0; hw < H * W; ++hw) {
                        size_t h = hw / W;
                        size_t w = hw % W;
                        result.at({b, c, h, w}) += bias_value;
                    }
                }
            }
        } else if (self.shape == other.shape) {
            // --- Standard Element-wise Addition ---
            for (size_t i = 0; i < self.data.size(); ++i) {
                result.data[i] += other.data[i];
            }
        } else {
            // You may want to handle other broadcasting cases or throw an error.
            throw std::runtime_error("Incompatible shapes for add operation.");
        }

        return result;
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
        if (numel() != std::accumulate(new_shape.begin(), new_shape.end(), (size_t)1, std::multiplies<size_t>())) {
            throw std::runtime_error("Reshape error: total elements must be the same.");
        }
        if (!is_contiguous()) {
            *this = this->contiguous();
        }
        shape = new_shape;
        strides.resize(shape.size());
        if (!shape.empty()) {
            strides.back() = 1;
            for (int i = shape.size() - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
    }

    Tensor permute(const std::vector<size_t>& dims) const {
        Tensor result;
        result.shape.resize(dims.size());
        result.strides.resize(dims.size());
        result.data = this->data; // Views share data
        for (size_t i = 0; i < dims.size(); ++i) {
            result.shape[i] = this->shape[dims[i]];
            result.strides[i] = this->strides[dims[i]];
        }
        return result;
    }
    Tensor transpose(size_t dim1, size_t dim2) const {
        // Simplified transpose for matmul
        if (shape.size() == 3 && dim1 == 1 && dim2 == 2) {
            return this->permute({0, 2, 1});
        }
        throw std::runtime_error("Unsupported transpose dimensions");
    }
    bool is_contiguous() const {
        if (shape.empty()) return true;
        size_t current_stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            if (shape[i] != 1) { // Strides for dimensions of size 1 are not strictly defined
                if (strides[i] != current_stride) return false;
            }
            current_stride *= shape[i];
        }
        return true;
    }
    Tensor contiguous() const {
        if (is_contiguous()) return *this;
        Tensor new_tensor(this->shape);
        std::vector<size_t> idx(shape.size(), 0);
        for (size_t i = 0; i < new_tensor.numel(); ++i) {
            new_tensor.data[i] = this->at(idx); // Safe access
            // Increment multi-dimensional index
            for (int d = shape.size() - 1; d >= 0; --d) {
                if (++idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }
        return new_tensor;
    }

    Tensor matmul(const Tensor& other) const {

        // --- Case 1: Batched Matrix Multiplication (3D x 3D) ---
        // This is the case needed for the attention block: (B, M, K) @ (B, K, N) -> (B, M, N)
       if (this->shape.size() == 3 && other.shape.size() == 3) {
            if (this->shape[0] != other.shape[0] || this->shape[2] != other.shape[1]) {
                 throw std::runtime_error("Matmul (3D) shape mismatch.");
            }
            size_t B = this->shape[0], M = this->shape[1], K = this->shape[2], N = other.shape[2];
            Tensor result({B, M, N});
            result.zero();
            for (size_t b = 0; b < B; ++b) {
                for (size_t m = 0; m < M; ++m) {
                    for (size_t n = 0; n < N; ++n) {
                        for (size_t k = 0; k < K; ++k) {
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
        if (dim >= shape.size() || start + count > shape[dim]) {
            throw std::out_of_range("Slice arguments out of range.");
        }
        Tensor result; // Create a view, do not copy data
        result.shape = this->shape;
        result.shape[dim] = count;
        result.strides = this->strides;
        // This is subtle: to create the view, we don't copy data,
        // we just offset the starting point by adjusting the data pointer.
        // But since we can't do that with std::vector easily, we create a
        // new tensor and copy into it safely.
        
        Tensor final_result(result.shape);
        std::vector<size_t> idx(final_result.shape.size());
        for(size_t i = 0; i < final_result.numel(); ++i) {
             // Calculate index in the new sliced tensor
            std::vector<size_t> current_idx(final_result.shape.size());
            size_t temp_i = i;
            for(int d = final_result.shape.size() - 1; d >= 0; --d) {
                current_idx[d] = temp_i % final_result.shape[d];
                temp_i /= final_result.shape[d];
            }
            
            // Map to index in the original tensor
            std::vector<size_t> source_idx = current_idx;
            source_idx[dim] += start;

            final_result.data[i] = this->at(source_idx);
        }
        return final_result;
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
        Tensor result = *this;
        result.reshape(new_shape);
        return result;
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
    Tensor bmm(const Tensor& other) const {
        if (shape.size() != 3 || other.shape.size() != 3 || shape[0] != other.shape[0] || shape[2] != other.shape[1]) {
            throw std::runtime_error("Invalid shapes for bmm.");
        }
        const size_t B = shape[0], M = shape[1], K = shape[2], N = other.shape[2];
        Tensor result({B, M, N});
        result.zero();
        for (size_t b = 0; b < B; ++b) {
            for (size_t m = 0; m < M; ++m) {
                for (size_t n = 0; n < N; ++n) {
                    for (size_t k = 0; k < K; ++k) {
                        result.at({b, m, n}) += this->at({b, m, k}) * other.at({b, k, n});
                    }
                }
            }
        }
        return result;
    }

    Tensor softmax(int dim) const {
        if (dim < 0) dim = shape.size() + dim;
        Tensor result(this->shape);
        size_t outer_size = 1;
        for(int i=0; i<dim; ++i) outer_size *= shape[i];
        size_t dim_size = shape[dim];
        size_t inner_size = 1;
        for(size_t i=dim+1; i<shape.size(); ++i) inner_size *= shape[i];

        for(size_t i = 0; i < outer_size; ++i) {
            for(size_t j = 0; j < inner_size; ++j) {
                float max_val = -INFINITY;
                for(size_t k=0; k<dim_size; ++k) {
                    max_val = std::max(max_val, data[i*dim_size*inner_size + k*inner_size + j]);
                }
                float sum = 0.0f;
                for(size_t k=0; k<dim_size; ++k) {
                    float val = std::exp(data[i*dim_size*inner_size + k*inner_size + j] - max_val);
                    result.data[i*dim_size*inner_size + k*inner_size + j] = val;
                    sum += val;
                }
                for(size_t k=0; k<dim_size; ++k) {
                    result.data[i*dim_size*inner_size + k*inner_size + j] /= sum;
                }
            }
        }
        return result;
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
