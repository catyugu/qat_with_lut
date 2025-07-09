# export_model.py (DEFINITIVE, ROBUST VERSION)

import torch
import torch.nn as nn
import struct
import numpy as np
from ddpm.QAT_UNet import (
    QATUNet, QATResidualBlock, AttentionBlock, 
    Downsample, Upsample, QATConv2d, ScaledWeightTernary
)
import os

# =================================================================================
# 1. ROBUST PACKING & WRITING HELPERS (VERIFIED)
# =================================================================================

def pack_ternary_weights_definitive(original_weight_tensor):
    """
    This is the definitive, correct packing function. It perfectly mirrors the
    logic of the ScaledWeightTernary.apply function, including the dead zone.
    
    It takes the ORIGINAL full-precision weight tensor as input.
    """
    tensor_flat = original_weight_tensor.cpu().view(-1)
    packed_data = []

    # --- Replicate the EXACT logic from ScaledWeightTernary ---
    alpha = torch.mean(torch.abs(original_weight_tensor)).detach()
    threshold = 0.001 * alpha

    num_chunks = (len(tensor_flat) + 15) // 16

    for i in range(num_chunks):
        packed_uint32 = 0
        chunk = tensor_flat[i*16 : (i+1)*16]
        
        for j, weight_val in enumerate(chunk):
            bits = 0
            # Use the exact same thresholding logic as the forward pass
            if weight_val.item() > threshold.item():
                bits = 1  # Corresponds to +1
            elif weight_val.item() < -threshold.item():
                bits = 2  # Corresponds to -1
            # Otherwise, it's in the dead zone and bits remains 00 for 0.

            packed_uint32 |= (bits << (j * 2))
            
        packed_data.append(packed_uint32)
        
    return packed_data
def get_diffusion_coefficients(num_timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Calculates all diffusion schedules needed for sampling, exactly matching
    the logic required by the C++ implementation.
    """
    betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), 'constant', 1.0)
    # Correct way to pad for older PyTorch versions and for clarity
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

    # Coefficients for q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    
    # Clamp variance to avoid division by zero
    posterior_variance_clipped = torch.clamp(posterior_variance, min=1e-20)

    # Coefficients for the mean of q(x_{t-1} | x_t, x_0)
    posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

    # Coefficients for the forward process (needed for pred_x0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance_clipped,
        "posterior_mean_coef1": posterior_mean_coef1,
        "posterior_mean_coef2": posterior_mean_coef2,
    }
def write_int(f, val): f.write(struct.pack('i', val))
def write_float(f, val): f.write(struct.pack('f', val))
def write_bool(f, val): f.write(struct.pack('?', val))
def write_tensor(f, tensor): f.write(tensor.detach().cpu().numpy().tobytes())

# =================================================================================
# 2. CORRECTED LAYER EXPORT FUNCTIONS
# =================================================================================

def export_conv(f, layer):
    """Exports a QATConv2d layer using the robust packing function."""
    print(f"  Exporting QAT Conv2d: in={layer.in_channels}, out={layer.out_channels}")
    
    # Step 1: Get the true quantized weights and alpha scale from the layer's function.
    quantized_weight = ScaledWeightTernary.apply(layer.weight)
    alpha_val = torch.mean(torch.abs(layer.weight)).detach().item()

    # Step 2: Write all layer parameters.
    write_int(f, layer.in_channels); write_int(f, layer.out_channels)
    write_int(f, layer.kernel_size[0]); write_int(f, layer.kernel_size[1])
    write_int(f, layer.stride[0]); write_int(f, layer.stride[1])
    write_int(f, layer.padding[0]); write_int(f, layer.padding[1])
    write_int(f, layer.groups)
    
    
    # Write quantization info
    write_float(f, alpha_val)
    write_int(f, len(layer.weight.shape))
    for dim in layer.weight.shape: write_int(f, dim)
        
    # CRITICAL: Pass the ORIGINAL weight tensor to the new packing function.
    packed_data = pack_ternary_weights_definitive(layer.weight)
    
    write_int(f, len(packed_data))
    for val in packed_data: f.write(struct.pack('I', val))
        
    write_int(f, layer.bias.numel())
    write_tensor(f, layer.bias)

def export_linear(f, layer):
    print(f"  Exporting Linear: in={layer.in_features}, out={layer.out_features}")
    write_int(f, layer.in_features); write_int(f, layer.out_features)
    write_int(f, layer.weight.numel()); write_tensor(f, layer.weight)
    write_int(f, layer.bias.numel()); write_tensor(f, layer.bias)

def export_norm(f, layer):
    print(f"  Exporting GroupNorm: groups={layer.num_groups}, channels={layer.num_channels}")
    write_int(f, layer.num_groups); write_int(f, layer.num_channels)
    write_float(f, layer.eps)
    write_int(f, layer.weight.numel()); write_tensor(f, layer.weight)
    write_int(f, layer.bias.numel()); write_tensor(f, layer.bias)

def export_attention_block(f, block):
    print("  Exporting Nested AttentionBlock...")
    export_norm(f, block.norm)
    export_conv(f, block.to_qkv)
    export_conv(f, block.to_out)

def export_res_block(f, block):
    """Exports a QATResidualBlock, accounting for all nested layers."""
    print("Exporting QATResidualBlock...")
    export_norm(f, block.norm_1)
    export_conv(f, block.conv_1)

    write_bool(f, block.time_bias is not None)
    if block.time_bias is not None: export_linear(f, block.time_bias)

    write_bool(f, block.class_bias is not None)
    if block.class_bias is not None:
        write_int(f, block.class_bias.num_embeddings)
        write_int(f, block.class_bias.embedding_dim)
        write_int(f, block.class_bias.weight.numel())
        write_tensor(f, block.class_bias.weight)

    export_norm(f, block.norm_2)
    export_conv(f, block.conv_2[1]) # Access layer inside nn.Sequential

    is_res_conv = isinstance(block.residual_connection, QATConv2d)
    write_bool(f, is_res_conv)
    if is_res_conv: export_conv(f, block.residual_connection)

    has_attention = isinstance(block.attention, AttentionBlock)
    write_bool(f, has_attention)
    if has_attention: export_attention_block(f, block.attention)

# =================================================================================
# 3. DEFINITIVE MAIN EXPORT LOGIC
# =================================================================================

def main():
    # --- Model Configuration and Loading ---
    config = { "img_channels": 3,
              "base_channels": 128, 
              "channel_mults": [1, 2, 2, 2],
              "num_res_blocks": 2, 
              "time_emb_dim": 512,
              "time_emb_scale": 1.0, 
              "num_classes": 10,
              "dropout": 0.1,
              "attention_resolutions": [1],
              "norm": "gn", 
              "num_groups": 32,
              "initial_pad": 0 }
    model_path = "qat_unet_final.pth" # Your final, non-EMA model
    output_path = "qat_unet_model_packed.bin"
    num_timesteps = 1000
    
    model = QATUNet(**config)
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}.")
        return
        
    # Directly load the state_dict, as we've confirmed this works for Python sampling
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    # --- THE CRITICAL FIX FOR STABLE WEIGHTS ---
    model.eval()
    
    print("Model loaded successfully and set to evaluation mode.")

    # --- Verification Step (Now that weights are stable) ---
    # This block will now produce consistent, correct output for comparison.
    print("\n--- Python Model: Final Convolution Layer Verification ---")
    final_conv_layer = model.out_conv
    final_bias = final_conv_layer.bias.data
    print(f"Bias (R, G, B): [{final_bias[0]:.8f}, {final_bias[1]:.8f}, {final_bias[2]:.8f}]")
    weights_flat = final_conv_layer.weight.data.view(3, -1)
    print(f"Weight for RED   (Channel 0, first 5): {weights_flat[0, :5].tolist()}")
    print(f"Weight for GREEN (Channel 1, first 5): {weights_flat[1, :5].tolist()}")
    print(f"Weight for BLUE  (Channel 2, first 5): {weights_flat[2, :5].tolist()}")
    print("---------------------------------------------------------")

    # --- Export Process ---
    with open(output_path, "wb") as f:
        print("\n--- Starting Sequential Export ---")
        
        # 1. Hyperparameters
        print("Writing hyperparameters...")
        write_int(f, 32); write_int(f, config["img_channels"]); write_int(f, config["base_channels"]);
        write_int(f, len(config["channel_mults"]));
        for mult in config["channel_mults"]: write_int(f, mult)
        write_int(f, config["num_res_blocks"]); write_int(f, config["time_emb_dim"]); write_float(f, config["time_emb_scale"]);
        write_int(f, 0 if config["num_classes"] is None else config["num_classes"]);
        write_float(f, config["dropout"]);
        write_int(f, len(config["attention_resolutions"]));
        for res in config["attention_resolutions"]: write_int(f, res)
        write_int(f, config["num_groups"]); write_int(f, config["initial_pad"]); write_bool(f, True);

        # 2. Layers in their EXACT ModuleList order
        print("\nExporting Time MLP..."); export_linear(f, model.time_mlp[1]); export_linear(f, model.time_mlp[3])
        print("\nExporting Initial Convolution..."); export_conv(f, model.init_conv)
        print("\nExporting Downsampling Path...")
        for layer in model.downs:
            if isinstance(layer, QATResidualBlock): export_res_block(f, layer)
            elif isinstance(layer, Downsample): export_conv(f, layer.downsample)
        print("\nExporting Middle Path...")
        for layer in model.mid: export_res_block(f, layer)
        print("\nExporting Upsampling Path...")
        for layer in model.ups:
            if isinstance(layer, QATResidualBlock): export_res_block(f, layer)
            elif isinstance(layer, Upsample): export_conv(f, layer.upsample[1])
        print("\nExporting Final Layers..."); export_norm(f, model.out_norm); export_conv(f, model.out_conv)

        print("\n--- Export Complete ---")
        print("\nCalculating and exporting diffusion coefficients...")
        coeffs = get_diffusion_coefficients(num_timesteps)
        
        # Write number of timesteps first, so C++ knows array sizes
        write_int(f, num_timesteps)
        
        # Write each coefficient array sequentially
        # The order here MUST MATCH the reading order in C++
        write_tensor(f, coeffs["betas"])
        write_tensor(f, coeffs["alphas"])
        write_tensor(f, coeffs["alphas_cumprod"])
        write_tensor(f, coeffs["sqrt_alphas_cumprod"])
        write_tensor(f, coeffs["sqrt_one_minus_alphas_cumprod"])
        write_tensor(f, coeffs["posterior_variance"])
        write_tensor(f, coeffs["posterior_mean_coef1"])
        write_tensor(f, coeffs["posterior_mean_coef2"])
        print("All diffusion coefficients have been written to the file.")

        print("\n--- Export Complete ---")
        print(f"Model and coefficients successfully exported to {output_path}")
        print(f"Model successfully exported to {output_path}")

if __name__ == "__main__":
    main()