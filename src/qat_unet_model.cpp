#include "qat_unet_model.h"
#include "kernels.h" // For avx2_bit_slice_gemm_kernel, hsum_i32_8 etc.
#include "utils.h"   // For im2col, quantize_float_to_int8_with_scale, pack_ternary_activations_5x3bit_to_ptr, relu, silu etc.
#include "types.h"   // For C_TERNARY_ACTIVATION_THRESHOLD

#include <iostream>
#include <fstream>
#include <algorithm> // For std::min, std::max
#include <cmath>     // For std::roundf, std::exp
#include <numeric>   // For std::accumulate
#include <iomanip>   // For std::fixed, std::setprecision

// --- Helper Functions (from utils.cpp or new) ---

// Function to perform Group Normalization
std::vector<float> GroupNormLayer::forward(const std::vector<float>& input, int batch_size, int channels, int height, int width) {
    if (input.empty() || channels == 0 || height == 0 || width == 0 || num_groups == 0 || channels % num_groups != 0) {
        std::cerr << "Error: Invalid GroupNormLayer input or parameters." << std::endl;
        return {};
    }

    std::vector<float> output(input.size());
    int single_image_size = channels * height * width;
    int channels_per_group = channels / num_groups;
    int group_elements_per_image = channels_per_group * height * width;

    for (int b = 0; b < batch_size; ++b) {
        for (int g = 0; g < num_groups; ++g) {
            // Calculate mean and variance for the current group
            float sum = 0.0f;
            float sum_sq = 0.0f;

            for (int c_offset = 0; c_offset < channels_per_group; ++c_offset) {
                int current_channel_idx = g * channels_per_group + c_offset;
                for (int hw_idx = 0; hw_idx < height * width; ++hw_idx) {
                    float val = input[b * single_image_size + current_channel_idx * height * width + hw_idx];
                    sum += val;
                    sum_sq += val * val;
                }
            }

            float mean = sum / group_elements_per_image;
            float variance = (sum_sq / group_elements_per_image) - (mean * mean);
            float std_dev = std::sqrt(variance + eps);

            // Normalize and apply affine transformation
            for (int c_offset = 0; c_offset < channels_per_group; ++c_offset) {
                int current_channel_idx = g * channels_per_group + c_offset;
                float gamma_val = (gamma.empty() ? 1.0f : gamma[current_channel_idx]);
                float beta_val = (beta.empty() ? 0.0f : beta[current_channel_idx]);

                for (int hw_idx = 0; hw_idx < height * width; ++hw_idx) {
                    float val = input[b * single_image_size + current_channel_idx * height * width + hw_idx];
                    output[b * single_image_size + current_channel_idx * height * width + hw_idx] =
                        ((val - mean) / std_dev) * gamma_val + beta_val;
                }
            }
        }
    }
    return output;
}

// Linear Layer forward pass
std::vector<float> LinearLayer::forward(const std::vector<float>& input, int batch_size) {
    if (input.empty() || in_features == 0 || out_features == 0) {
        std::cerr << "Error: Invalid LinearLayer input or parameters." << std::endl;
        return {};
    }

    std::vector<float> output(batch_size * out_features, 0.0f);

    for (int b = 0; b < batch_size; ++b) {
        for (int o = 0; o < out_features; ++o) {
            float sum = 0.0f;
            for (int i = 0; i < in_features; ++i) {
                sum += input[b * in_features + i] * weights[o * in_features + i];
            }
            output[b * out_features + o] = sum + (bias.empty() ? 0.0f : bias[o]);
        }
    }
    return output;
}


// Positional embedding for time (from diffusion.py's _get_timestep_embedding)
std::vector<float> positional_embedding(const std::vector<long>& timesteps, int dim, float scale) {
    std::vector<float> embedding(timesteps.size() * dim);
    // This calculation for inv_freq needs to be careful to match PyTorch
    // PyTorch's `_get_timestep_embedding` uses `inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))`
    // This means `inv_freq` is a vector, not a single scalar.
    // We will precompute the `inv_freq` values.
    std::vector<float> inv_freq_vec(dim / 2);
    for (int i = 0; i < dim / 2; ++i) {
        inv_freq_vec[i] = 1.0f / std::pow(10000.0f, (float)(2 * i) / dim);
    }

    for (size_t b = 0; b < timesteps.size(); ++b) {
        long t = timesteps[b];
        for (int i = 0; i < dim / 2; ++i) {
            float arg = (float)t * inv_freq_vec[i];
            embedding[b * dim + 2 * i] = std::sin(arg) * scale;
            embedding[b * dim + 2 * i + 1] = std::cos(arg) * scale;
        }
    }
    return embedding;
}

// QATConv2dLayer forward pass implementation
std::vector<float> QATConv2dLayer::forward(const std::vector<float>& input_batch, int batch_size,
                                          int input_h, int input_w, // Actual H, W of input image
                                          const std::vector<int32_t>& precomputed_lut) {
    if (input_batch.empty() || in_channels == 0 || out_channels == 0 || kernel_h == 0 || kernel_w == 0) {
        std::cerr << "Error: Invalid QATConv2dLayer parameters or empty input." << std::endl;
        return {};
    }

    // Calculate output dimensions
    int output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;

    if (output_h <= 0 || output_w <= 0) {
        std::cerr << "Error: Invalid output dimensions calculated for QATConv2dLayer. Input H/W: " << input_h << "/" << input_w << ", Kernel H/W: " << kernel_h << "/" << kernel_w << ", Stride H/W: " << stride_h << "/" << stride_w << ", Pad H/W: " << pad_h << "/" << pad_w << std::endl;
        return {};
    }

    int output_plane_size = out_channels * output_h * output_w;
    std::vector<float> output_batch(batch_size * output_plane_size);

    // Determine the effective input_dim for the linear operation after im2col
    int linear_input_dim = in_channels * kernel_h * kernel_w; // Each "column" is a flattened kernel window

    // A fixed activation scale for now, as per the discussion.
    // This value should be carefully chosen or learned during QAT.
    const float activation_quantization_scale = 127.0f; // Example scale for [-1, 1] range

    for (int b = 0; b < batch_size; ++b) {
        // Extract current image's input data
        const float* current_input_ptr = input_batch.data() + b * (in_channels * input_h * input_w);

        // 1. Apply im2col to convert current input image to a "column" matrix
        // The im2col function will handle padding internally if needed.
        std::vector<float> im2col_output;
        im2col(current_input_ptr, in_channels, input_h, input_w,
               kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, im2col_output);

        // im2col_output will have shape (linear_input_dim * num_output_pixels) in a flat vector
        int num_output_pixels = output_h * output_w;

        // Prepare buffer for packed activations for the entire image
        // Each patch (column) needs to be packed separately.
        // Size: num_output_pixels * (ceil(linear_input_dim * 3 / 8))
        size_t packed_patch_size_bytes = (linear_input_dim * 3 + 7) / 8;
        std::vector<uint8_t> packed_activations_for_image(num_output_pixels * packed_patch_size_bytes);

        // 2. Quantize and pack activations from im2col output
        for (int op_idx = 0; op_idx < num_output_pixels; ++op_idx) {
            // Get the current patch (column) from im2col_output
            const float* current_patch_float_ptr = im2col_output.data() + op_idx * linear_input_dim;

            // Quantize the float patch to int8
            std::vector<int8_t> quantized_patch(linear_input_dim);
            quantize_float_to_int8_with_scale(current_patch_float_ptr, quantized_patch.data(), linear_input_dim, activation_quantization_scale);

            // Pack the quantized int8 patch into 5x3bit format
            uint8_t* current_packed_patch_ptr = packed_activations_for_image.data() + op_idx * packed_patch_size_bytes;
            pack_ternary_activations_5x3bit_to_ptr(quantized_patch.data(), linear_input_dim, current_packed_patch_ptr);
        }

        // 3. Perform GEMM for each output channel
        for (int oc = 0; oc < out_channels; ++oc) {
            const uint8_t* current_kernel_weights_packed_ptr = packed_weights.data() + oc * packed_patch_size_bytes; // Weights are (out_channels, linear_input_dim)

            for (int op_idx = 0; op_idx < num_output_pixels; ++op_idx) {
                const uint8_t* current_packed_patch_ptr = packed_activations_for_image.data() + op_idx * packed_patch_size_bytes;

                int32_t sum = avx2_bit_slice_gemm_kernel(
                    current_packed_patch_ptr,       // Input packed patch
                    current_kernel_weights_packed_ptr, // Kernel weights for this output channel
                    precomputed_lut.data(),
                    linear_input_dim                // k_dim for GEMM is the size of the flattened kernel
                );

                // Dequantize the result and add bias
                // The result `sum` is `sum(ternary_act * ternary_weight)`.
                // To get back to float, we need to divide by (activation_scale * weight_scale).
                output_batch[b * output_plane_size + oc * output_h * output_w + op_idx] =
                    (static_cast<float>(sum) / (activation_quantization_scale * weight_scale)) + bias[oc];
            }
        }
    }
    return output_batch;
}

// Function to perform bilinear upsampling (NCHW format)
std::vector<float> upsample_bilinear(const std::vector<float>& input, int channels, int in_h, int in_w, int scale_factor) {
    int out_h = in_h * scale_factor;
    int out_w = in_w * scale_factor;
    std::vector<float> output(input.size() / (in_h * in_w) * out_h * out_w); // Assuming NCHW

    for (size_t n = 0; n < input.size() / (channels * in_h * in_w); ++n) { // Iterate over batch
        for (int c = 0; c < channels; ++c) {
            for (int y_out = 0; y_out < out_h; ++y_out) {
                for (int x_out = 0; x_out < out_w; ++x_out) {
                    float x_in = (float)x_out / scale_factor;
                    float y_in = (float)y_out / scale_factor;

                    int x1 = static_cast<int>(std::floor(x_in));
                    int y1 = static_cast<int>(std::floor(y_in));
                    int x2 = std::min(x1 + 1, in_w - 1);
                    int y2 = std::min(y1 + 1, in_h - 1);

                    float dx = x_in - x1;
                    float dy = y_in - y1;

                    float val11 = input[n * (channels * in_h * in_w) + c * (in_h * in_w) + y1 * in_w + x1];
                    float val12 = input[n * (channels * in_h * in_w) + c * (in_h * in_w) + y1 * in_w + x2];
                    float val21 = input[n * (channels * in_h * in_w) + c * (in_h * in_w) + y2 * in_w + x1];
                    float val22 = input[n * (channels * in_h * in_w) + c * (in_h * in_w) + y2 * in_w + x2];

                    float interpolated_val = (val11 * (1 - dx) * (1 - dy)) +
                                             (val12 * dx * (1 - dy)) +
                                             (val21 * (1 - dx) * dy) +
                                             (val22 * dx * dy);

                    output[n * (channels * out_h * out_w) + c * (out_h * out_w) + y_out * out_w + x_out] = interpolated_val;
                }
            }
        }
    }
    return output;
}


// AttentionBlock forward pass (more representative placeholder)
std::vector<float> AttentionBlock::forward(const std::vector<float>& input, int batch_size,
                                           int channels, int height, int width) {
    // This is a more representative placeholder for self-attention.
    // A full implementation would involve:
    // 1. Reshaping input (N, C, H, W) to (N, H*W, C) or (N, C, H*W) for matrix multiplication.
    // 2. Linear projections for Query (Q), Key (K), Value (V) using QATConv2dLayer or LinearLayer.
    //    These projections would typically be `in_channels -> channels`.
    // 3. Calculating attention scores: Q @ K.T
    // 4. Applying softmax to attention scores.
    // 5. Multiplying by Value: Attention_scores @ V
    // 6. Reshaping back to (N, C, H, W).
    // 7. Final linear projection and residual connection.

    // For now, let's perform a simple element-wise multiplication with a dummy attention map
    // to show a non-identity operation, but this is NOT real attention.
    std::vector<float> output = input; // Start with input for residual connection
    std::vector<float> attention_map(input.size(), 1.0f); // Dummy attention map (all ones)

    // Simulate some "attention" effect (e.g., scaling by a dummy map)
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] *= attention_map[i]; // This is not how attention works, just for demonstration
    }

    std::cout << "Warning: AttentionBlock forward is a conceptual placeholder. Implement actual self-attention." << std::endl;
    return output; // Return the "attended" output
}


// QATResBlock forward pass implementation
std::vector<float> QATResBlock::forward(const std::vector<float>& input, int batch_size,
                                       int input_h, int input_w,
                                       const std::vector<float>& time_emb,
                                       const std::vector<int>& y_labels, // y_labels for class conditioning
                                       const std::vector<int32_t>& precomputed_lut) {
    // Store initial input for skip connection
    std::vector<float> h = input;
    int current_channels = conv1.in_channels; // This should be the input channels to the ResBlock
    int current_h = input_h;
    int current_w = input_w;

    // 1. First convolution block: conv1 -> norm1 -> SiLU
    h = conv1.forward(h, batch_size, current_h, current_w, precomputed_lut);
    current_channels = conv1.out_channels; // Channels update after conv1
    h = norm1.forward(h, batch_size, current_channels, current_h, current_w);
    silu(h.data(), h.size());

    // 2. Add time embedding projection
    // time_emb has shape (batch_size, time_emb_dim)
    // time_emb_proj1 output has shape (batch_size, current_channels)
    std::vector<float> time_proj1_output = time_emb_proj1.forward(time_emb, batch_size);
    // Broadcast time_proj1_output (N, C) to (N, C, H, W) and add to h
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < current_channels; ++c) {
            float val_to_add = time_proj1_output[b * current_channels + c];
            for (int hw_idx = 0; hw_idx < current_h * current_w; ++hw_idx) {
                h[b * (current_channels * current_h * current_w) + c * (current_h * current_w) + hw_idx] += val_to_add;
            }
        }
    }

    // 3. Second convolution block: conv2 -> norm2 -> SiLU
    h = conv2.forward(h, batch_size, current_h, current_w, precomputed_lut);
    h = norm2.forward(h, batch_size, current_channels, current_h, current_w);
    silu(h.data(), h.size());

    // 4. Add time embedding projection (second one)
    std::vector<float> time_proj2_output = time_emb_proj2.forward(time_emb, batch_size);
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < current_channels; ++c) {
            float val_to_add = time_proj2_output[b * current_channels + c];
            for (int hw_idx = 0; hw_idx < current_h * current_w; ++hw_idx) {
                h[b * (current_channels * current_h * current_w) + c * (current_h * current_w) + hw_idx] += val_to_add;
            }
        }
    }

    // 5. Optional Attention block
    if (use_attention && attention_block) {
        h = attention_block->forward(h, batch_size, current_channels, current_h, current_w);
    }

    // 6. Add skip connection
    // If input channels != output channels of the resblock, use skip_connection_conv
    // The skip connection output should have the same dimensions as `h` (current_channels, current_h, current_w)
    if (skip_connection_conv) {
        std::vector<float> skip_output = skip_connection_conv->forward(input, batch_size, input_h, input_w, precomputed_lut);
        // Element-wise addition of h and skip_output
        if (h.size() != skip_output.size()) {
            std::cerr << "Error: Mismatch in sizes for skip connection! h.size()=" << h.size() << ", skip_output.size()=" << skip_output.size() << std::endl;
            // Handle error or resize/reshape skip_output if necessary.
            // For now, we'll proceed but this indicates a potential issue in dimension tracking.
        }
        for (size_t i = 0; i < h.size(); ++i) {
            h[i] += skip_output[i];
        }
    } else {
        // Direct element-wise addition if channels match
        if (h.size() != input.size()) {
             std::cerr << "Error: Mismatch in sizes for direct skip connection! h.size()=" << h.size() << ", input.size()=" << input.size() << std::endl;
        }
        for (size_t i = 0; i < h.size(); ++i) {
            h[i] += input[i];
        }
    }

    return h;
}


// QATUNet Constructor
QATUNet::QATUNet() : image_size(0), in_channels(0), base_channels(0), num_res_blocks(0),
                     time_emb_dim(0), time_emb_scale(0.0f), num_classes(0), dropout(0.0f),
                     num_groups(0), initial_pad(0), use_scale_shift_norm(false) {
    // Initialize vectors to empty
    channel_mults.clear();
    attention_resolutions.clear();
    downs_res_blocks.clear();
    downs_downsample_convs.clear();
    ups_res_blocks.clear();
    ups_upsample_convs.clear();
}


// Helper to read a QATConv2dLayer from file
void QATUNet::read_conv_layer(std::ifstream& file, QATConv2dLayer& layer) {
    file.read((char*)&layer.in_channels, sizeof(int));
    file.read((char*)&layer.out_channels, sizeof(int));
    file.read((char*)&layer.kernel_h, sizeof(int));
    file.read((char*)&layer.kernel_w, sizeof(int));
    file.read((char*)&layer.stride_h, sizeof(int));
    file.read((char*)&layer.stride_w, sizeof(int));
    file.read((char*)&layer.pad_h, sizeof(int));
    file.read((char*)&layer.pad_w, sizeof(int));
    file.read((char*)&layer.groups, sizeof(int));
    file.read((char*)&layer.weight_scale, sizeof(float));

    int packed_weights_size;
    // Read raw bytes to inspect them
    char packed_weights_size_bytes[sizeof(int)];
    file.read(packed_weights_size_bytes, sizeof(int));
    // Correctly interpret the bytes as a little-endian integer
    packed_weights_size = 0;
    for (size_t i = 0; i < sizeof(int); ++i) {
        packed_weights_size |= (static_cast<unsigned char>(packed_weights_size_bytes[i]) << (8 * i));
    }

    // Debugging print BEFORE resize
    std::cout << "    DEBUG: read_conv_layer - packed_weights_size = " << packed_weights_size;
    std::cout << " (Raw bytes: ";
    for (size_t i = 0; i < sizeof(int); ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)(unsigned char)packed_weights_size_bytes[i] << " ";
    }
    std::cout << std::dec << ")" << std::endl;

    layer.packed_weights.resize(packed_weights_size);
    file.read((char*)layer.packed_weights.data(), packed_weights_size);

    int bias_size;
    // Read raw bytes to inspect them
    char bias_size_bytes[sizeof(int)];
    file.read(bias_size_bytes, sizeof(int));
    // Correctly interpret the bytes as a little-endian integer
    bias_size = 0;
    for (size_t i = 0; i < sizeof(int); ++i) {
        bias_size |= (static_cast<unsigned char>(bias_size_bytes[i]) << (8 * i));
    }

    // Debugging print BEFORE resize
    std::cout << "    DEBUG: read_conv_layer - bias_size = " << bias_size;
    std::cout << " (Raw bytes: ";
    for (size_t i = 0; i < sizeof(int); ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)(unsigned char)bias_size_bytes[i] << " ";
    }
    std::cout << std::dec << ")" << std::endl;

    layer.bias.resize(bias_size);
    file.read((char*)layer.bias.data(), bias_size * sizeof(float));

    // Debugging prints (these are *after* the resize calls)
    std::cout << "    Read Conv Layer: in_ch=" << layer.in_channels
              << ", out_ch=" << layer.out_channels
              << ", kernel=" << layer.kernel_h << "x" << layer.kernel_w
              << ", stride=" << layer.stride_h << "x" << layer.stride_w
              << ", pad=" << layer.pad_h << "x" << layer.pad_w
              << ", groups=" << layer.groups
              << ", weight_scale=" << layer.weight_scale
              << ", packed_weights_size=" << packed_weights_size
              << ", bias_size=" << bias_size << std::endl;
}

// Helper to read a GroupNormLayer from file
void QATUNet::read_groupnorm_layer(std::ifstream& file, GroupNormLayer& layer) {
    file.read((char*)&layer.num_groups, sizeof(int));
    file.read((char*)&layer.num_channels, sizeof(int));
    file.read((char*)&layer.eps, sizeof(float));

    int gamma_size;
    file.read((char*)&gamma_size, sizeof(int));
    layer.gamma.resize(gamma_size);
    file.read((char*)layer.gamma.data(), gamma_size * sizeof(float));

    int beta_size;
    file.read((char*)&beta_size, sizeof(int));
    layer.beta.resize(beta_size);
    file.read((char*)layer.beta.data(), beta_size * sizeof(float));

    std::cout << "    Read GroupNorm Layer: num_groups=" << layer.num_groups
              << ", num_channels=" << layer.num_channels
              << ", eps=" << layer.eps
              << ", gamma_size=" << gamma_size
              << ", beta_size=" << beta_size << std::endl;
}

// Helper to read a LinearLayer from file
void QATUNet::read_linear_layer(std::ifstream& file, LinearLayer& layer) {
    file.read((char*)&layer.out_features, sizeof(int));
    file.read((char*)&layer.in_features, sizeof(int));

    int weights_size;
    file.read((char*)&weights_size, sizeof(int));
    layer.weights.resize(weights_size);
    file.read((char*)layer.weights.data(), weights_size * sizeof(float));

    int bias_size;
    file.read((char*)&bias_size, sizeof(int));
    layer.bias.resize(bias_size);
    file.read((char*)layer.bias.data(), bias_size * sizeof(float));

    std::cout << "    Read Linear Layer: in_features=" << layer.in_features
              << ", out_features=" << layer.out_features
              << ", weights_size=" << weights_size
              << ", bias_size=" << bias_size << std::endl;
}

// Helper to read an AttentionBlock from file (if it has parameters)
void QATUNet::read_attention_block(std::ifstream& file, AttentionBlock& block) {
    // If AttentionBlock has parameters (e.g., Q, K, V projections), read them here.
    // For now, it's a placeholder as its internal structure is not yet defined in C++.
    // You would add calls to read_conv_layer or read_linear_layer here for its internal layers.
    std::cout << "Reading placeholder AttentionBlock. No specific parameters loaded yet." << std::endl;
}

// Helper to read a QATResBlock from file
void QATUNet::read_res_block(std::ifstream& file, QATResBlock& block, int in_channels, int out_channels,
                            int time_emb_dim, int num_classes_arg, const std::vector<int>& attention_resolutions_arg,
                            int current_resolution, int num_groups_arg, bool use_scale_shift_norm_arg) {
    std::cout << "  Reading ResBlock layers for in_ch: " << in_channels << ", out_ch: " << out_channels << std::endl;
    // Read conv1 and norm1
    read_conv_layer(file, block.conv1);
    read_groupnorm_layer(file, block.norm1);

    // Read time_emb_proj1
    read_linear_layer(file, block.time_emb_proj1);

    // Read conv2 and norm2
    read_conv_layer(file, block.conv2);
    read_groupnorm_layer(file, block.norm2);

    // Read time_emb_proj2
    read_linear_layer(file, block.time_emb_proj2);

    // Read skip_connection_conv if it exists
    // The Python export script needs to write a flag or size for this.
    // If the input channels to the ResBlock are different from its output channels,
    // a skip connection convolution is typically used.
    if (in_channels != out_channels) {
        std::cout << "    Reading skip_connection_conv..." << std::endl;
        block.skip_connection_conv = std::make_unique<QATConv2dLayer>();
        read_conv_layer(file, *block.skip_connection_conv);
    }

    // Read AttentionBlock if applicable
    bool resolution_matches_attention = false;
    for (int res : attention_resolutions_arg) {
        if (current_resolution == res) {
            resolution_matches_attention = true;
            break;
        }
    }
    if (resolution_matches_attention) {
        std::cout << "    Reading attention_block..." << std::endl;
        block.use_attention = true;
        block.attention_block = std::make_unique<AttentionBlock>();
        read_attention_block(file, *block.attention_block); // Read its parameters if any
    }
}


// Main model loading function
bool QATUNet::load_model(const std::string& model_path, std::vector<int32_t>& precomputed_lut) {
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open model file: " << model_path << std::endl;
        return false;
    }

    std::cout << "Loading QAT UNet model from: " << model_path << std::endl;

    // Read Model Hyperparameters
    file.read((char*)&image_size, sizeof(int));
    file.read((char*)&in_channels, sizeof(int));
    file.read((char*)&base_channels, sizeof(int));
    int num_channel_mults;
    file.read((char*)&num_channel_mults, sizeof(int));
    channel_mults.resize(num_channel_mults);
    for (int i = 0; i < num_channel_mults; ++i) {
        file.read((char*)&channel_mults[i], sizeof(int));
    }
    file.read((char*)&num_res_blocks, sizeof(int));
    file.read((char*)&time_emb_dim, sizeof(int));
    file.read((char*)&time_emb_scale, sizeof(float));
    file.read((char*)&num_classes, sizeof(int)); // Read as int, 0 if None
    file.read((char*)&dropout, sizeof(float));
    int num_attention_resolutions;
    file.read((char*)&num_attention_resolutions, sizeof(int));
    attention_resolutions.resize(num_attention_resolutions);
    for (int i = 0; i < num_attention_resolutions; ++i) {
        file.read((char*)&attention_resolutions[i], sizeof(int));
    }
    file.read((char*)&num_groups, sizeof(int));
    file.read((char*)&initial_pad, sizeof(int));
    int use_scale_shift_norm_int;
    file.read((char*)&use_scale_shift_norm_int, sizeof(int));
    use_scale_shift_norm = (use_scale_shift_norm_int == 1);

    std::cout << "Model Hyperparameters:" << std::endl;
    std::cout << "  Image Size: " << image_size << std::endl;
    std::cout << "  In Channels: " << in_channels << std::endl;
    std::cout << "  Base Channels: " << base_channels << std::endl;
    std::cout << "  Channel Mults: "; for(int m : channel_mults) std::cout << m << " "; std::cout << std::endl;
    std::cout << "  Num Res Blocks: " << num_res_blocks << std::endl;
    std::cout << "  Time Emb Dim: " << time_emb_dim << std::endl;
    std::cout << "  Time Emb Scale: " << std::fixed << std::setprecision(2) << time_emb_scale << std::endl;
    std::cout << "  Num Classes: " << num_classes << std::endl;
    std::cout << "  Dropout: " << dropout << std::endl;
    std::cout << "  Attention Resolutions: "; for(int r : attention_resolutions) std::cout << r << " "; std::cout << std::endl;
    std::cout << "  Num Groups: " << num_groups << std::endl;
    std::cout << "  Initial Pad: " << initial_pad << std::endl;
    std::cout << "  Use Scale Shift Norm: " << (use_scale_shift_norm ? "True" : "False") << std::endl;


    // Read initial conv
    std::cout << "Reading init_conv..." << std::endl;
    read_conv_layer(file, init_conv);

    // Read time embedding MLP layers
    std::cout << "Reading time_embedding_mlp_0..." << std::endl;
    read_linear_layer(file, time_embedding_mlp_0);
    std::cout << "Reading time_embedding_mlp_1..." << std::endl;
    read_linear_layer(file, time_embedding_mlp_1);

    // Read label embedding MLP if class_cond
    if (num_classes > 0) {
        std::cout << "Reading label_embedding_mlp..." << std::endl;
        label_embedding_mlp = std::make_unique<LinearLayer>();
        read_linear_layer(file, *label_embedding_mlp);
    }

    // --- Read Downsampling Path ---
    int current_channels = base_channels;
    int current_resolution = image_size;
    for (size_t i = 0; i < channel_mults.size(); ++i) {
        int out_ch = base_channels * channel_mults[i];
        downs_res_blocks.emplace_back(); // Add a new vector for this resolution level
        for (int j = 0; j < num_res_blocks; ++j) {
            std::cout << "Reading downs_res_blocks[" << i << "][" << j << "] (res: " << current_resolution << ", in_ch: " << current_channels << ", out_ch: " << out_ch << ")..." << std::endl;
            downs_res_blocks.back().emplace_back(); // Add a new ResBlock
            read_res_block(file, downs_res_blocks.back().back(), current_channels, out_ch,
                           time_emb_dim, num_classes, attention_resolutions, current_resolution, num_groups, use_scale_shift_norm);
            current_channels = out_ch; // Channels update after each res block
        }
        if (i != channel_mults.size() - 1) { // Not the last block, so there's a downsample
            std::cout << "Reading downs_downsample_convs[" << i << "] (res: " << current_resolution << ", in_ch: " << current_channels << ")..." << std::endl;
            downs_downsample_convs.emplace_back();
            read_conv_layer(file, downs_downsample_convs.back()); // Read downsample conv
            current_resolution /= 2;
        }
    }

    // --- Read Mid Block ---
    std::cout << "Reading mid_res_block1 (res: " << current_resolution << ", in_ch: " << current_channels << ")..." << std::endl;
    read_res_block(file, mid_res_block1, current_channels, current_channels,
                   time_emb_dim, num_classes, attention_resolutions, current_resolution, num_groups, use_scale_shift_norm);
    
    // Check if attention is used in mid block (usually it is at resolution 16, 8, 4 etc.)
    bool mid_has_attention = false;
    for (int res : attention_resolutions) {
        if (current_resolution == res) {
            mid_has_attention = true;
            break;
        }
    }
    if (mid_has_attention) {
        std::cout << "Reading mid_attention_block (res: " << current_resolution << ", in_ch: " << current_channels << ")..." << std::endl;
        read_attention_block(file, mid_attention_block);
    }
    
    std::cout << "Reading mid_res_block2 (res: " << current_resolution << ", in_ch: " << current_channels << ")..." << std::endl;
    read_res_block(file, mid_res_block2, current_channels, current_channels,
                   time_emb_dim, num_classes, attention_resolutions, current_resolution, num_groups, use_scale_shift_norm);


    // --- Read Upsampling Path ---
    // This is the reverse of downsampling.
    // The `channel_mults` are processed in reverse order.
    // The number of res blocks per level is `num_res_blocks + 1` due to skip connections.
    for (int i = channel_mults.size() - 1; i >= 0; --i) {
        int out_ch = base_channels * channel_mults[i];
        ups_res_blocks.emplace_back(); // Add a new vector for this resolution level
        for (int j = 0; j < num_res_blocks + 1; ++j) { // +1 for the skip connection block
            std::cout << "Reading ups_res_blocks[" << i << "][" << j << "] (res: " << current_resolution << ", in_ch: " << current_channels << ", out_ch: " << out_ch << ")..." << std::endl;
            read_res_block(file, ups_res_blocks.back().emplace_back(), current_channels, out_ch, // Use emplace_back directly
                           time_emb_dim, num_classes, attention_resolutions, current_resolution, num_groups, use_scale_shift_norm);
            current_channels = out_ch; // Channels update after each res block
        }
        if (i != 0) { // Not the first block (highest resolution), so there's an upsample
            std::cout << "Reading ups_upsample_convs[" << i << "] (res: " << current_resolution << ", in_ch: " << current_channels << ")..." << std::endl;
            ups_upsample_convs.emplace_back();
            read_conv_layer(file, ups_upsample_convs.back()); // Read upsample conv
            current_resolution *= 2;
        }
    }


    // Read output layers
    std::cout << "Reading out_norm..." << std::endl;
    read_groupnorm_layer(file, out_norm);
    std::cout << "Reading out_conv..." << std::endl;
    read_conv_layer(file, out_conv);

    file.close();

    // Build the bit-slice LUT once after loading the model
    build_bit_slice_lut_5x3(precomputed_lut);
    std::cout << "Bit-Slice LUT built. Size: " << precomputed_lut.size() * sizeof(int32_t) / 1024.0 << " KB" << std::endl;

    std::cout << "QAT UNet model loaded successfully." << std::endl;
    return true;
}

// Main QATUNet forward pass (conceptual implementation)
std::vector<float> QATUNet::forward(const std::vector<float>& x_input, int batch_size,
                                   const std::vector<long>& time_steps,
                                   const std::vector<int>& y_labels,
                                   const std::vector<int32_t>& precomputed_lut) {
    // This function needs to meticulously replicate the PyTorch QAT_UNet.forward
    // It involves:
    // 1. Time embedding generation (positional_embedding + Linear layers)
    // 2. Class embedding generation (if class_cond)
    // 3. Initial convolution
    // 4. Downsampling path (ResBlocks, Downsample layers, skip connections)
    // 5. Mid block (ResBlocks, Attention)
    // 6. Upsampling path (ResBlocks, Upsample layers, skip connections)
    // 7. Final normalization and convolution

    // 1. Time embedding
    std::vector<float> time_emb = positional_embedding(time_steps, time_emb_dim, time_emb_scale);
    time_emb = time_embedding_mlp_0.forward(time_emb, batch_size);
    silu(time_emb.data(), time_emb.size()); // Apply SiLU in-place
    time_emb = time_embedding_mlp_1.forward(time_emb, batch_size);

    // 2. Class embedding (if class_cond)
    // The PyTorch QAT_UNet uses `nn.Embedding` for class conditioning,
    // which is a lookup table. The `class_bias` error implies a direct addition
    // of a learned bias for each class.
    // Here, we'll assume `label_embedding_mlp` is a direct linear projection
    // from a one-hot encoding of `y_labels` or a similar representation.
    // For simplicity, if `num_classes > 0`, we'll generate a dummy label embedding.
    // In a real scenario, `y_labels` would be used to index into an embedding table
    // or be one-hot encoded and passed to `label_embedding_mlp`.
    std::vector<float> label_emb_output;
    if (num_classes > 0 && label_embedding_mlp) {
        // Create a dummy input for label_embedding_mlp.
        // If label_embedding_mlp.in_features == 1, then y_labels can be directly used.
        // If it expects one-hot, you'd need to create one-hot vectors.
        // For now, let's assume it expects a single float per batch item.
        std::vector<float> dummy_label_input(batch_size, 0.0f); // Placeholder
        // In a real implementation, you'd use y_labels to create this input.
        // For example, if label_embedding_mlp.in_features == num_classes,
        // you'd create one-hot vectors:
        // std::vector<float> one_hot_labels(batch_size * num_classes, 0.0f);
        // for (int b = 0; b < batch_size; ++b) {
        //     if (y_labels[b] >= 0 && y_labels[b] < num_classes) {
        //         one_hot_labels[b * num_classes + y_labels[b]] = 1.0f;
        //     }
        // }
        // label_emb_output = label_embedding_mlp->forward(one_hot_labels, batch_size);

        // For now, a simple pass-through if the linear layer has 1 input feature.
        // This is a major simplification and likely needs adjustment based on actual PyTorch model.
        label_emb_output = label_embedding_mlp->forward(std::vector<float>(y_labels.begin(), y_labels.end()), batch_size);
    }


    // 3. Initial convolution
    std::vector<float> h = init_conv.forward(x_input, batch_size, image_size, image_size, precomputed_lut);
    int current_h = image_size;
    int current_w = image_size;
    int current_channels = base_channels; // After init_conv, channels become base_channels

    // Store intermediate outputs for skip connections
    std::vector<std::vector<float>> hs;
    hs.push_back(h); // Initial input for skip connection

    // 4. Downsampling path
    for (size_t i = 0; i < channel_mults.size(); ++i) {
        int out_ch = base_channels * channel_mults[i];
        downs_res_blocks.emplace_back(); // Add a new vector for this resolution level
        for (int j = 0; j < num_res_blocks; ++j) {
            h = downs_res_blocks[i][j].forward(h, batch_size, current_h, current_w, time_emb, y_labels, precomputed_lut);
            hs.push_back(h);
        }
        if (i != channel_mults.size() - 1) { // Not the last block, so there's a downsample
            h = downs_downsample_convs[i].forward(h, batch_size, current_h, current_w, precomputed_lut);
            current_h /= 2;
            current_w /= 2;
            hs.push_back(h);
        }
        current_channels = out_ch; // Update current channels for the next block
    }

    // 5. Mid block
    h = mid_res_block1.forward(h, batch_size, current_h, current_w, time_emb, y_labels, precomputed_lut);
    
    // Check if attention is used in mid block
    bool mid_has_attention = false;
    for (int res : attention_resolutions) {
        if (current_h == res) { // Check resolution (h or w, assuming square)
            mid_has_attention = true;
            break;
        }
    }
    if (mid_has_attention) {
        h = mid_attention_block.forward(h, batch_size, current_channels, current_h, current_w);
    }
    
    h = mid_res_block2.forward(h, batch_size, current_h, current_w, time_emb, y_labels, precomputed_lut);


    // 6. Upsampling path
    // Iterate channel_mults in reverse
    for (int i = channel_mults.size() - 1; i >= 0; --i) {
        int out_ch = base_channels * channel_mults[i];
        // Loop through res blocks + 1 for skip connection block
        for (int j = 0; j < num_res_blocks + 1; ++j) { // +1 for the skip connection block
            // Pop the last element from hs for skip connection
            std::vector<float> skip_h = hs.back();
            hs.pop_back();

            // Add skip connection (element-wise addition)
            // Ensure dimensions match. This might require reshaping skip_h if it's from a different resolution.
            // For now, assuming skip_h is already at the correct size for the current h.
            if (h.size() != skip_h.size()) {
                 std::cerr << "Error: Mismatch in sizes for upsampling skip connection! h.size()=" << h.size() << ", skip_h.size()=" << skip_h.size() << std::endl;
                 // This is a critical error. You might need to resize/interpolate skip_h or debug dimension flow.
                 // For now, to prevent crash, we'll just skip adding if sizes mismatch.
            } else {
                for (size_t k = 0; k < h.size(); ++k) {
                    h[k] += skip_h[k];
                }
            }

            h = ups_res_blocks[channel_mults.size() - 1 - i][j].forward(h, batch_size, current_h, current_w, time_emb, y_labels, precomputed_lut);
        }
        if (i != 0) { // Not the first block (highest resolution), so there's an upsample
            // Upsample the feature map using bilinear interpolation
            h = upsample_bilinear(h, current_channels, current_h, current_w, 2);
            current_h *= 2;
            current_w *= 2;
            h = ups_upsample_convs[channel_mults.size() - 1 - i].forward(h, batch_size, current_h, current_w, precomputed_lut);
        }
        current_channels = out_ch;
    }


    // 7. Final normalization and convolution
    h = out_norm.forward(h, batch_size, current_channels, current_h, current_w);
    silu(h.data(), h.size());
    h = out_conv.forward(h, batch_size, current_h, current_w, precomputed_lut);

    return h;
}
