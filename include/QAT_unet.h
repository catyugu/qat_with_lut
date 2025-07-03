#ifndef QAT_UNET_H
#define QAT_UNET_H

#include "types.h"
#include <vector>
#include <map>

// Forward declaration
class QAT_UNet_CPP;

// Helper function to load the model from a file
bool load_qat_unet(const std::string& filepath, QAT_UNet_CPP& model);


class QAT_UNet_CPP {
public:
    QAT_UNet_CPP();

    // --- U-Net Layers ---
    // We will store layers in maps, keyed by their names from the Python model.
    // This makes loading flexible.
    std::map<std::string, LutConvLayer> conv_layers;
    std::map<std::string, LutLinearLayer> linear_layers;
    // You would add maps for other layer types (e.g., Attention) as needed.

    // Model parameters
    int model_channels;
    int num_res_blocks;


    // --- Forward Pass ---
    /**
     * @brief Executes the full forward pass of the U-Net.
     *
     * @param x The input noisy image tensor [C, H, W].
     * @param time_emb The time embedding vector.
     * @param output The output tensor (estimated noise).
     */
    void forward(
        const std::vector<float>& x,
        const std::vector<float>& time_emb,
        std::vector<float>& output
    );

private:
    // Precomputed LUT for GEMM
    std::vector<int32_t> precomputed_lut;

    // Helper to perform a forward pass on a specific ResBlock
    void forward_resblock(
        const std::string& block_name,
        const std::vector<float>& x,
        const std::vector<float>& emb,
        std::vector<float>& out
    );
};

#endif // QAT_UNET_H
