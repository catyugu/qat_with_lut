#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <cstdint> // For uint8_t, int8_t
#include <string>
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

#endif // TYPES_H
